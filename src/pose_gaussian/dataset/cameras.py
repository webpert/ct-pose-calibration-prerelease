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
import sys
import torch
from torch import nn
import numpy as np

sys.path.append("./")
from pose_gaussian.utils.graphics_utils import getWorld2View2, getProjectionMatrix
from pose_gaussian.utils.gaussian_utils import get_expon_lr_func


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        scanner_cfg,
        R,
        T,
        angle,
        mode,
        FoVx,
        FoVy,
        image,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.angle = angle
        self.FoVx = torch.Tensor([FoVx]).cuda()
        self.FoVy = torch.Tensor([FoVy]).cuda()
        self.mode = mode
        self.image_name = image_name
        self.optimizer = None
        self.iteration = 0
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                fovX=self.FoVx,
                fovY=self.FoVy,
                mode=mode,  # kschoi, parallel/cone
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.mask = None

        # kschoi, self calibration parameters
        self.last_row = torch.tensor([[0., 0., 0., 1.]]).cuda()

        self.init_translation = torch.tensor(T).float().view(-1, 1).cuda()
        self.delta_translation = nn.Parameter(torch.zeros(3, 1).cuda().requires_grad_(True))
        # self.translation = self.init_translation + self.delta_translation

        self.init_quaternion = rotation_matrix_to_quaternion(torch.tensor(R).float().t().cuda())
        self.delta_quaternion = nn.Parameter(torch.zeros(4).cuda().requires_grad_(True))
        # self.quaternion = self.init_quaternion + self.delta_quaternion

        self.learnable_fovx = nn.Parameter(torch.tensor(self.FoVx).cuda().requires_grad_(True))
        self.learnable_fovy = nn.Parameter(torch.tensor(self.FoVy).cuda().requires_grad_(True)) 

        self.spatial_lr_scale = 0


    def set_mask(self, threshold):
        self.mask = (self.original_image >= threshold)


    # should be invoked before rendering
    def refresh_parameters(self):
        quaternion = self.init_quaternion + self.delta_quaternion
        rotation = quaternion_to_rotation_matrix(quaternion)

        translation = self.init_translation + self.delta_translation
        self.Rt = torch.cat((rotation, translation), dim=1)
        world_view_transform = torch.cat((self.Rt, self.last_row), dim=0).t()

        # scaling far/close to the center
        c2w = world_view_transform.inverse()
        mask = torch.ones_like(c2w)
        mask[3, :3] = torch.tensor([1.], device='cuda')

        self.world_view_transform = (c2w * mask).inverse()
        self.projection_matrix = getProjectionMatrix(fovX=self.learnable_fovx, fovY=self.learnable_fovy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.iteration = self.iteration + 1


    def training_setup(self, training_args, spatial_lr_scale=0.0):
        self.spatial_lr_scale = spatial_lr_scale
        self.iteration = 0
        l = [
            {
                "params": [self.delta_translation],
                "lr": training_args.translation_lr_init * self.spatial_lr_scale,
                "name": "delta_translation",
            },
            {
                "params": [self.delta_quaternion],
                "lr": training_args.quaternion_lr_init * self.spatial_lr_scale,
                "name": "delta_quaternion",
            },
            {
                "params": [self.learnable_fovx],
                "lr": training_args.fovx_lr_init * self.spatial_lr_scale,
                "name": "learnable_fovx",
            },            
            {
                "params": [self.learnable_fovy],
                "lr": training_args.fovy_lr_init * self.spatial_lr_scale,
                "name": "learnable_fovy",
            },            
        ]        

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.translation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.translation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.translation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.translation_lr_max_steps,
        )
        self.quaternion_scheduler_args = get_expon_lr_func(
            lr_init=training_args.quaternion_lr_init * self.spatial_lr_scale,
            lr_final=training_args.quaternion_lr_final * self.spatial_lr_scale,
            max_steps=training_args.quaternion_lr_max_steps,
        )
        self.fovx_scheduler_args = get_expon_lr_func(
            lr_init=training_args.fovx_lr_init * self.spatial_lr_scale,
            lr_final=training_args.fovx_lr_final * self.spatial_lr_scale,
            max_steps=training_args.fovx_lr_max_steps,
        )
        self.fovy_scheduler_args = get_expon_lr_func(
            lr_init=training_args.fovy_lr_init * self.spatial_lr_scale,
            lr_final=training_args.fovy_lr_final * self.spatial_lr_scale,
            max_steps=training_args.fovy_lr_max_steps,
        )


    def update_learning_rate(self, iteration=None):
        """Learning rate scheduling per step"""
        if iteration is None:
            iteration = self.iteration
            
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "delta_translation":
                lr = self.translation_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "delta_quaternion":
                lr = self.quaternion_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "learnable_fovx":
                lr = self.fovx_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "learnable_fovy":
                lr = self.fovy_scheduler_args(iteration)
                param_group["lr"] = lr

    def print_params(self, external_id=None):
        if external_id is None:
            external_id = self.uid
        print(f'***[{external_id}/{self.iteration}] t:{self.delta_translation[0,0].item():.4f},{self.delta_translation[1,0].item():.4f},{self.delta_translation[2,0].item():.4f} ', end='')
        print(f'R:{self.delta_quaternion[0].item():.4f},{self.delta_quaternion[1].item():.4f},{self.delta_quaternion[2].item():.4f},{self.delta_quaternion[3].item():.4f} ', end='')
        print(f'fov:{self.learnable_fovx.item():.4f}, {self.learnable_fovy.item():.4f}')
    
    def export_extrinsics(self, base_path):
        import os.path as osp
        quaternion = self.init_quaternion + self.delta_quaternion
        translation = self.init_translation + self.delta_translation
        np.savez(osp.join(base_path, f"cam_calib_result_{self.image_name}.npz"), quaternion=quaternion.detach().squeeze().cpu().numpy(), translation=translation.detach().squeeze().cpu().numpy())


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]



def quaternion_to_rotation_matrix(quaternion):
    # Ensure quaternion is normalized
    quaternion = quaternion / torch.norm(quaternion)

    w, x, y, z = quaternion.unbind(-1)

    # Pre-compute repeated values
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z

    # Construct rotation matrix
    R = torch.stack([
        torch.stack([1 - 2 * y2 - 2 * z2, 2 * xy - 2 * wz, 2 * xz + 2 * wy]),
        torch.stack([2 * xy + 2 * wz, 1 - 2 * x2 - 2 * z2, 2 * yz - 2 * wx]),
        torch.stack([2 * xz - 2 * wy, 2 * yz + 2 * wx, 1 - 2 * x2 - 2 * y2])
    ])

    return R

def rotation_matrix_to_quaternion(R):
    t = R.trace()
    if t > 0:
        r = torch.sqrt(1 + t)
        s = 0.5 / r
        w = 0.5 * r
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        r = torch.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.5 / r
        w = (R[2, 1] - R[1, 2]) * s
        x = 0.5 * r
        y = (R[0, 1] + R[1, 0]) * s
        z = (R[0, 2] + R[2, 0]) * s
    elif R[1, 1] > R[2, 2]:
        r = torch.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        s = 0.5 / r
        w = (R[0, 2] - R[2, 0]) * s
        x = (R[0, 1] + R[1, 0]) * s
        y = 0.5 * r
        z = (R[1, 2] + R[2, 1]) * s
    else:
        r = torch.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        s = 0.5 / r
        w = (R[1, 0] - R[0, 1]) * s
        x = (R[0, 2] + R[2, 0]) * s
        y = (R[1, 2] + R[2, 1]) * s
        z = 0.5 * r
    return torch.tensor([w, x, y, z]).cuda()