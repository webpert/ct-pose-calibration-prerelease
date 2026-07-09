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
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

sys.path.append("./")

from simple_knn._C import distCUDA2
from pose_gaussian.utils.general_utils import t2a
from pose_gaussian.utils.system_utils import mkdir_p
from pose_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    strip_symmetric,
    build_scaling_rotation,
)

EPS = 1e-5


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        self.density_res_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        self.mask_activation = torch.nn.Sigmoid()
        self.density_inverse_activation = inverse_softplus
        self.density_res_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, scale_bound=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self._density_res = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound        
        self.setup_functions()
        self.lac_ref = None

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self._density_res,
            self._mask,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self._density_res,
            self._mask,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.scale_bound,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.setup_functions()  # Reset activation functions

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        return self.density_activation(self._density)

    @property
    def get_density_res(self):
        return self.density_res_activation(self._density_res)
    
    @property
    def get_mask(self):
        return self.mask_activation(self._mask)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def get_erank(self):
        scales = self.get_scaling       # [N, 3]
        S = scales.abs().sum(axis=-1)   # [N, 1]
        q = scales / S.unsqueeze(-1).repeat(1,3)     # [N, 3]
        erank = torch.exp(-torch.sum(q * torch.log(q), axis=-1))                    # [N, 1]
        return erank

    def get_erank_loss(self, erank_value=2, eps = 1e-10):        
        erank = self.get_erank()
        if erank_value == 2:
            erank_loss = torch.max(-torch.log(erank - 1. + eps), torch.zeros_like(erank)) + self.get_scaling[:, -1]    # [N, 1], walnut_v25_erank_grad_fix
        elif erank_value == 3:
            erank_loss = (erank - 3.)**2
        elif erank_value == 4:
            masked_erank = erank.clone()
            masked_erank[masked_erank >= 2.] = 2.
            erank_loss = torch.max(-torch.log(masked_erank - 1. + eps), torch.zeros_like(erank))
        elif erank_value == 5:
            erank_loss = 0.001 / (erank + eps - 1.)
        elif erank_value == 6:
            erank_loss = 0.001 * (erank - 2.5)**2
            erank_loss[erank >= 2.5] = 0.0
        elif erank_value == 7:
            erank_loss = 0.001 * (erank - 2.7)**2
            erank_loss[erank >= 2.7] = 0.0
        elif erank_value == 8:
            erank_loss = 0.001 * (erank - 2.9)**2
            erank_loss[erank >= 2.9] = 0.0
        else:         
            erank_loss = torch.zeros_like(erank_loss)

        return erank_loss.sum()   

    def get_selective_erank_loss(self, selective_mask):
        erank = self.get_erank()
        erank_loss = (erank[selective_mask] - 3.)**2

        return erank_loss.mean()    # return range: 0 ~ 4

    
    def compute_3d_edges(self, volume):
        volume = volume.unsqueeze(0).unsqueeze(0).detach()
        device = volume.device

        sobel_1d = torch.tensor([-1., 0., 1.], device=device)
        smooth_1d = torch.tensor([1., 2., 1.], device=device)

        kernel_x = sobel_1d.view(3, 1, 1) * smooth_1d.view(1, 3, 1) * smooth_1d.view(1, 1, 3)
        kernel_y = smooth_1d.view(3, 1, 1) * sobel_1d.view(1, 3, 1) * smooth_1d.view(1, 1, 3)
        kernel_z = smooth_1d.view(3, 1, 1) * smooth_1d.view(1, 3, 1) * sobel_1d.view(1, 1, 3)

        kernel = torch.stack([kernel_x, kernel_y, kernel_z], dim=0).unsqueeze(1) / 8.0

        volume_padded = F.pad(volume, (1,1,1,1,1,1), mode='replicate')

        grads = F.conv3d(volume_padded, kernel)

        gradient_magnitude = torch.sqrt((grads ** 2).sum(dim=1))
        return gradient_magnitude.squeeze()


    def get_internal_gaussian_mask(self, volume, offset, nVoxel, sVoxel):
        offset = torch.from_numpy(np.array(offset)).cuda()
        nVoxel = torch.from_numpy(np.array(nVoxel)).cuda()
        sVoxel = torch.from_numpy(np.array(sVoxel)).cuda()
        dVoxel = sVoxel / nVoxel

        m = self.get_xyz.detach()
        voxel_index = torch.round((m - offset + sVoxel / 2.) / dVoxel).to(torch.long)
        voxel_index = torch.clip(voxel_index, 0, nVoxel[0] - 1)

        v = volume.detach()
        vol_edges = self.compute_3d_edges(v)
        # vol_edges = np.clip(vol_edges, 0, 1)
        vol_edges[vol_edges < 0.3] = 0.0
        vol_edges[vol_edges >= 0.3] = 1.0
        vol_internal = 1. - vol_edges

        internal_gaussian_mask = vol_internal[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]].to(torch.bool)

        return internal_gaussian_mask.cpu().numpy()


    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(
            "Initialize gaussians from {} estimated points".format(
                fused_point_cloud.shape[0]
            )
        )
        fused_density = (
            self.density_inverse_activation(torch.tensor(density)).float().cuda()
        )
        fused_density_res = (
            self.density_res_inverse_activation(torch.tensor(density*0.5)).float().cuda()   # TOOD, kschoi
        )
        
        dist = torch.sqrt(
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.001**2,
            )
        )
        if self.scale_bound is not None:
            dist = torch.clamp(
                dist, self.scale_bound[0] + EPS, self.scale_bound[1] - EPS
            )  # Avoid overflow

        scales = self.scaling_inverse_activation(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self._density_res = nn.Parameter(fused_density_res.requires_grad_(True))
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #! Generate one gaussian for debugging purpose
        if False:
            print("Initialize one gaussian")
            fused_xyz = (
                torch.tensor([[0.0, 0.0, 0.0]]).float().cuda()
            )  # position: [0,0,0]
            fused_density = self.density_inverse_activation(
                torch.tensor([[0.8]]).float().cuda()
            )  # density: 0.8
            fused_density_res = self.density_res_inverse_activation(
                torch.tensor([[0.4]]).float().cuda()
            )  # density: 0.8            
            scales = self.scaling_inverse_activation(
                torch.tensor([[0.5, 0.5, 0.5]]).float().cuda()
            )  # scale: 0.5
            rots = (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float().cuda()
            )  # quaternion: [1, 0, 0, 0]
            # rots = torch.tensor([[0.966, -0.259, 0, 0]]).float().cuda()
            self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self._density_res = nn.Parameter(fused_density_res.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            # TODO, kschoi, split learning rate?
            {
                "params": [self._density_res],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density_res",
            },            
            {
                "params": [self._mask],
                "lr": training_args.mask_lr,
                "name": "mask",
            },            
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l


    # def save_ply(self, path):
    #     # We save pickle files to store more information

    #     mkdir_p(os.path.dirname(path))

    #     xyz = t2a(self._xyz)
    #     densities = t2a(self._density)
    #     densities_res = t2a(self._density_res)
    #     scale = t2a(self._scaling)
    #     rotation = t2a(self._rotation)

    #     out = {
    #         "xyz": xyz,
    #         "density": densities,
    #         "density_res": densities_res,
    #         "scale": scale,
    #         "rotation": rotation,
    #         "scale_bound": self.scale_bound,
    #     }
    #     with open(path, "wb") as f:
    #         pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    # kschoi, for SIBR viewer
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        N = xyz.shape[0]
        normals = np.zeros_like(xyz)
        f_dc_o = torch.ones((N, 3, 1)).float()
        f_rest_o = torch.zeros((N, 3, 15)).float()
        f_dc = f_dc_o.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = f_rest_o.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_density.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(f_dc_o.shape[1]*f_dc_o.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest_o.shape[1]*f_rest_o.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        # We load pickle file.
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._xyz = nn.Parameter(
            torch.tensor(data["xyz"], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._density = nn.Parameter(
            torch.tensor(
                data["density"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        try:
            self._density_res = nn.Parameter(
                torch.tensor(
                    data["density_res"], dtype=torch.float, device="cuda"
                ).requires_grad_(True)
            )        
        except Exception as e:
            self._density_res = torch.zeros_like(self._density)

        self._scaling = nn.Parameter(
            torch.tensor(
                data["scale"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                data["rotation"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self.scale_bound = data["scale_bound"]
        self.setup_functions()  # Reset activation functions

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._density_res = optimizable_tensors["density_res"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_densities_res,
        new_mask,
        new_scaling,
        new_rotation,
        new_max_radii2D,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "density_res": new_densities_res,
            "mask": new_mask,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._density_res = optimizable_tensors["density_res"]
        self._mask = optimizable_tensors["mask"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]   # split function is invoked after clone function, so the number of GS is bigger than that of grads
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")   # [N]
        padded_grad[: grads.shape[0]] = grads.squeeze() # grads: [M] s.t. M < N
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # grad_threshold:5e-5
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )   # densify_scale_threshold: 0.2, get_scaling:[N, 3]

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_density_res = self.density_res_inverse_activation(
            self.get_density_res[selected_pts_mask].repeat(N, 1) * (1 / N)
        )        
        new_mask = self._mask[selected_pts_mask].repeat(N, 1)
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_density,
            new_density_res,
            new_mask,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]  # kschoi, new_xyz == old_xyz???
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_densities_res = self.density_res_inverse_activation(
            self.get_density_res[selected_pts_mask] * 0.5
        )        
        new_mask = self._mask[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities
        self._density_res[selected_pts_mask] = new_densities_res
        
        self.densification_postfix(
            new_xyz,
            new_densities,
            new_densities_res,
            new_mask,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
        volume_mask=None,
    ):
        # kschoi, ADC logic should be refined because of too may gaussians are generated during optimization
        grads = self.xyz_gradient_accum / self.denom    ## sum of gradient of u, v space 2D position [N, 1] / [N, 1]
        grads[grads.isnan()] = 0.0

        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, densify_scale_threshold)

        # Prune gaussians with too small density
        prune_mask = torch.logical_or((self.get_mask <= 0.01).squeeze(),(self.get_density < min_density).squeeze())
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = torch.logical_or(prune_mask, prune_mask_xyz)

        if volume_mask is not None:
            prune_mask_xyz = self.get_prune_mask_via_volume(volume_mask, bbox)
            prune_mask = torch.logical_or(prune_mask, prune_mask_xyz)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def mask_prune(self):
        prune_mask = (self.get_mask <= 0.01).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def get_prune_mask_via_volume(self, volume_mask, bbox):
        xyz = self.get_xyz.clone().detach()  # [N, 3] Gaussians
        h, w, d = volume_mask.shape          # [h, w, d] Voxels
        background_volume_mask = ~volume_mask
        
        x_idx = torch.round((xyz[:, 0] - bbox[0, 0]) / (bbox[1, 0] - bbox[0, 0]) * (w - 1)).long()
        y_idx = torch.round((xyz[:, 1] - bbox[0, 1]) / (bbox[1, 1] - bbox[0, 1]) * (h - 1)).long()
        z_idx = torch.round((xyz[:, 2] - bbox[0, 2]) / (bbox[1, 2] - bbox[0, 2]) * (d - 1)).long()

        x_idx = x_idx.clamp(0, w - 1)
        y_idx = y_idx.clamp(0, h - 1)
        z_idx = z_idx.clamp(0, d - 1)

        prune_mask = background_volume_mask[y_idx, x_idx, z_idx]

        return prune_mask       


    def add_densification_stats(self, viewspace_point_tensor, update_filter, use_xyz_grad = False):
        # viewspace_point_tensor [N, 3]:float, update_filter [N]:bool, viewspace_point_tensor.grad [N, 3]:float
        # if use_erank:            
        #     # kschoi, according to the base paper, the sum of norm is better than the norm of sum
        #     self.xyz_gradient_accum[update_filter] += viewspace_point_tensor.grad[update_filter, 2].unsqueeze(-1)
        if use_xyz_grad:
            self.xyz_gradient_accum[update_filter] += torch.norm(
                self.get_xyz.grad[update_filter, :3], dim=-1, keepdim=True
            )        
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
            )
        self.denom[update_filter] += 1  # denom [N, 1]

    def update_hist_sparsity_config(self, volume, max_components=3, threshold=0.0, debug=False):
        # Ensure the data is a 2D array (required by sklearn's GMM)
        data = volume.detach().cpu().numpy().reshape(-1, 1)

        # Fit the GMM
        gmm = GaussianMixture(n_components=max_components, tol=1e-4)
        gmm.fit(data)

        # Extract means, weights, and covariances
        means = gmm.means_.flatten()
        weights = gmm.weights_.flatten()

        # Sort the results w.r.t. means
        sorted_indices = np.argsort(means)
        means = means[sorted_indices]
        weights = weights[sorted_indices]

        # Filter means based on the threshold
        filtered_means = [mean for mean, weight in zip(means, weights) if weight > threshold]
        filtered_means[0] = 0.0   # denoise effect can be achievable

        if True:
            from pose_gaussian.utils.plot_utils import show_histogram
            show_histogram(data, filtered_means)
                
        self.lac_ref = torch.from_numpy(filtered_means).cuda()

    
    def update_hist_sparsity_config_nms(self, volume, min_distance=1, threshold=None):
        v = volume.detach().cpu().numpy().reshape(-1)
        hist, bin_edges = np.histogram(v, bins=1000, range=(0, 1), density=False)
        data = np.log10(hist + 1)
        
        # # Find all candidate peaks (local maxima)
        # peaks = np.argwhere(
        #     (np.r_[True, data[1:] > data[:-1]] &  # Rising edge
        #     np.r_[data[:-1] > data[1:], True])  # Falling edge
        # ).flatten()
        
        # # Apply thresholding if specified
        # if threshold is not None:
        #     peaks = [p for p in peaks if data[p] >= threshold]
        
        # # Apply Non-Maximum Suppression
        # suppressed_peaks = []
        # while len(peaks) > 0:
        #     # Select the peak with the highest value
        #     current_peak = peaks[np.argmax(data[peaks])]
        #     suppressed_peaks.append(current_peak)
            
        #     # Remove all peaks within the min_distance
        #     peaks = [p for p in peaks if abs(p - current_peak) > min_distance]

        # filtered_means = bin_edges[suppressed_peaks]

        # if True:
        #     from pose_gaussian.utils.plot_utils import show_histogram
        #     show_histogram(data, filtered_means)            
        filtered_means = np.array([0.105, 0.352, 0.482])

        self.lac_ref = torch.from_numpy(filtered_means).cuda()

    
    def get_hist_sparsity_loss(self, volume):                
        mus = self.lac_ref

        if mus is not None:
            loss = torch.tensor(1.0).cuda()
            m = torch.min(torch.abs(mus[1:] - mus[:-1]))
            f = volume.flatten()    # autograd works        
            for i, mu in enumerate(mus):
                loss = loss * ((f - mu)**2 / (((f - mu)**2) + (0.5 * (m**2))))
            loss = loss.abs().mean()
        else:
            loss = torch.tensor(0.0).cuda()
        
        return loss
        





