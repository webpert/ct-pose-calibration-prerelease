#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import os.path as osp
import torch
from random import randint, shuffle
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

sys.path.append("./")
from pose_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from pose_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from pose_gaussian.utils.general_utils import safe_state
from pose_gaussian.utils.cfg_utils import load_config
from pose_gaussian.utils.log_utils import prepare_output_and_logger
from pose_gaussian.dataset import Scene
from pose_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss, tv_2d_loss, l2_trunc_loss
from pose_gaussian.utils.image_utils import metric_vol, metric_proj, get_foreground_threshold, get_volume_mask
from pose_gaussian.utils.plot_utils import show_two_slice, show_gaussians, show_two_volume
from pose_gaussian.utils.gaussian_utils import build_scaling_rotation


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)    

    if False:
        # early testing for sibr_viewer
        test_path = osp.join(scene.model_path, "test.ply")
        gaussians.save_ply_sibr(test_path)

    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = getattr(opt, "lambda_tv", 0.0) > 0
    use_erank = getattr(opt, "lambda_erank", 0.0) > 0
    use_mask = getattr(opt, "lambda_mask", 0.0) > 0
    pipe.use_mask = use_mask
    use_mcmc = getattr(opt, "noise_lr", 0.0) > 0
    use_gt_mask = opt.use_gt_mask
    use_gaussian_darkness = getattr(opt, "lambda_gaussian_darkness", 0.0) > 0
    use_hist_sparsity = getattr(opt, "lambda_hist_sparsity", 0.0) > 0
    use_hessian = opt.use_hessian
    use_volume_mask = opt.use_volume_mask
    use_scale_limit = getattr(opt, "lambda_scale_limit", 0.0) > 0
    use_selective_erank = getattr(opt, "lambda_selective_erank", 0.0) > 0
    use_l2_trunc_loss = getattr(opt, "l2_trunc_loss", 0.0) > 0
    use_projection_blur = getattr(opt, "projection_blur_sigma", 0.0) > 0

    if use_tv or use_hist_sparsity:
        print("Use differentiable voxelizer")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)

    # for gt masking
    if use_gt_mask or use_volume_mask:
        viewpoint_stack = scene.getTrainCameras().copy()
        image_list = []
        for i in range(len(viewpoint_stack)):
            image_list.append(viewpoint_stack[i].original_image)
        image_stack = torch.stack(image_list, dim=0)
        gt_image_threshold = get_foreground_threshold(image_stack)
    
    cam_list = scene.train_cameras
    for i in range(len(cam_list)):
        cam_list[i].training_setup(opt, 1.0)

    viewpoint_stac_index = None
    vol_pred_for_erank = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stac_index:
            # viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_stac_index = list(range(len(cam_list)))
            shuffle(viewpoint_stac_index)            
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam_index = viewpoint_stac_index.pop(0)
        viewpoint_cam = cam_list[viewpoint_cam_index]

        # Render X-ray projection
        viewpoint_cam.refresh_parameters()
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        if use_gt_mask:
            viewpoint_cam.set_mask(gt_image_threshold)
            gt_image = viewpoint_cam.original_image.cuda().clone()
            gt_image[~(viewpoint_cam.mask)] = 0.0
            image[~(viewpoint_cam.mask)] = 0.0
        elif use_projection_blur:
            gt_image = torch.from_numpy(
                gaussian_filter(
                    viewpoint_cam.original_image.cpu().numpy(),
                    sigma=getattr(opt, "projection_blur_sigma", 0.0),
                )
            ).cuda()
        else:
            gt_image = viewpoint_cam.original_image.cuda()        

        loss = {"total": 0.0}

        if use_hessian:
            if iteration > 5000:
                render_loss = tv_2d_loss(image-gt_image, reduction="mean")
                loss["render"] = render_loss
                loss["total"] += loss["render"]
            else:
                render_loss = l1_loss(image, gt_image)
                loss["render_grad"] = render_loss
                loss["total"] += loss["render_grad"]
        elif use_l2_trunc_loss:
            render_loss = l2_trunc_loss(image, gt_image, alpha=getattr(opt, "l2_trunc_loss", 0.0))
            loss["render"] = render_loss
            loss["total"] += loss["render"]
        else:
            render_loss = l1_loss(image, gt_image)
            loss["render"] = render_loss
            loss["total"] += loss["render"]
            if opt.lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(image, gt_image)
                loss["dssim"] = loss_dssim
                loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim

        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + getattr(opt, "lambda_tv", 0.0) * loss_tv
        # else:
        #     loss_tv = tv_2d_loss(image - gt_image, reduction="mean")
        #     loss["tv"] = loss_tv
        #     loss["total"] = loss["total"] + 0.05 * loss_tv

        if use_erank:
            loss_erank = gaussians.get_erank_loss(erank_value=opt.erank_value)
            loss["erank"] = loss_erank
            loss["total"] = loss["total"] + getattr(opt, "lambda_erank", 0.0) * loss_erank

        if use_selective_erank:
            if iteration == 5000:
                vol_pred_for_erank = queryfunc(gaussians)["vol"]       
            elif iteration > 5000 and (vol_pred_for_erank is not None):                
                internal_gaussian_mask = gaussians.get_internal_gaussian_mask(vol_pred_for_erank, 
                                                                              scanner_cfg["offOrigin"], 
                                                                              scanner_cfg["nVoxel"], 
                                                                              scanner_cfg["sVoxel"],)     # [N, 1]
                loss_erank = gaussians.get_selective_erank_loss(internal_gaussian_mask)
                loss["erank"] = loss_erank
                loss["total"] = loss["total"] + getattr(opt, "lambda_selective_erank", 0.0) * loss_erank
            else:
                pass    # do nothing

        if use_mask:
            loss_mask = torch.mean(gaussians.get_mask)
            loss["mask"] = loss_mask
            loss["total"] = loss["total"] + getattr(opt, "lambda_mask", 0.0) * loss["mask"]

        if use_gaussian_darkness:
            loss_gaussian_darkness = torch.mean(gaussians.get_density)
            loss["gaussian_darkness"] = loss_gaussian_darkness
            loss["total"] = loss["total"] + getattr(opt, "lambda_gaussian_darkness", 0.0) * loss["gaussian_darkness"]

        if use_hist_sparsity:
            if not use_tv:
                tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                    bbox[1] - tv_vol_sVoxel - bbox[0]
                ) * torch.rand(3)                
                vol_pred = query(
                    gaussians,
                    tv_vol_center,
                    tv_vol_nVoxel,
                    tv_vol_sVoxel,
                    pipe,
                )["vol"]
            loss_hist_sparsity = gaussians.get_hist_sparsity_loss(vol_pred)
            loss["hist_sparsity"] = loss_hist_sparsity
            loss["total"] = loss["total"] + getattr(opt, "lambda_hist_sparsity", 0.0) * loss["hist_sparsity"]
        
        if use_scale_limit:
            tmp_max, _ = gaussians.get_scaling.max(dim=1)
            loss_scale_limit = torch.mean(tmp_max)
            loss["scale_limit"] = loss_scale_limit
            loss["total"] = loss["total"] + getattr(opt, "lambda_scale_limit", 0.0) * loss["scale_limit"]

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"gaussians/lr_{param_group['name']}"] = param_group["lr"]
            
            for param_group in viewpoint_cam.optimizer.param_groups:
                metrics[f"camera[{viewpoint_cam_index}]/lr_{param_group['name']}"] = param_group["lr"]
            # viewpoint_cam.print_params(viewpoint_cam_index)

            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
                opt,
            )

            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, use_xyz_grad=opt.use_xyz_grad)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    if use_volume_mask and (iteration % opt.density_reset_interval == 0):
                        with torch.no_grad():
                            vol_pred = query(gaussians, scanner_cfg["offOrigin"], scanner_cfg["nVoxel"], scanner_cfg["sVoxel"], pipe)["vol"]
                            volume_mask = get_volume_mask(vol_pred, gt_image_threshold)      # kschoi, TODO, threshold should be  in volume space, scale
                        if False:
                            import matplotlib
                            matplotlib.use("TkAgg")
                            show_two_volume(vol_pred.cpu().numpy(), volume_mask.float().cpu().numpy(), title1='pred', title2='mask', no_diff=True)                            
                    else:
                        volume_mask = None

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, # 5e-5
                        opt.density_min_threshold,  # 1e-5
                        opt.max_screen_size,        # None
                        max_scale,                  # None
                        opt.max_num_gaussians,      # 500000
                        densify_scale_threshold,    # 0.2
                        bbox,                       # [[-1, -1, -1], [1, 1, 1]]
                        volume_mask,
                    )
                
                if (iteration % opt.density_reset_interval == 0 or iteration == opt.densify_from_iter) and opt.use_reset:
                    gaussians.reset_density(reset_density=0.01)
                
                if iteration == opt.density_reset_interval and use_hist_sparsity: # 3000
                    # set the histogram sparsity hyper parameters, TODO
                    gaussians.update_hist_sparsity_config_nms(queryfunc(gaussians)["vol"])
            else:
                if use_mask and iteration % opt.mask_prune_iter == 0:
                    gaussians.mask_prune()
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                viewpoint_cam.optimizer.step()
                viewpoint_cam.optimizer.zero_grad(set_to_none=True)
                viewpoint_cam.update_learning_rate(iteration)

                if use_mcmc:
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))

                    for param_group in gaussians.optimizer.param_groups:
                        if param_group.get('name') == 'xyz':
                            lr_xyz = param_group['lr']                    
                    
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1 - gaussians.get_density)) * getattr(opt, "noise_lr", 0.0) * lr_xyz
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()




def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
    opt,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    if opt.debug_proj_images:
                        filename = osp.join(eval_save_path, f"{config['name']}_gt_{viewpoint.image_name}.npy")
                        np.save(filename, gt_image[0].detach().cpu().numpy())
                        filename = osp.join(eval_save_path, f"{config['name']}_pred_{viewpoint.image_name}.npy")
                        np.save(filename, image[0].detach().cpu().numpy())
                    if opt.export_cam_params and config['name'] == 'render_train':
                        viewpoint.export_extrinsics(eval_save_path)
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/voxel_hitogram", vol_pred.reshape(-1), iteration
            )
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )
            tb_writer.add_histogram(
                "scene/erank_histogram", scene.gaussians.get_erank(), iteration
            )


    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 200, 300, 400, 500, 5_000, 10_000, 15_000, 20_000, 25_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")
