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
import numpy as np
import torch

sys.path.append("./")
from pose_gaussian.utils.loss_utils import ssim


def mse(img1, img2, mask=None):
    """MSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)

        mse = torch.stack(
            [
                (((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True)
                for i in range(img1.shape[0])
            ],
            dim=0,
        )

    else:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse


def rmse(img1, img2, mask=None):
    """RMSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    rmse = mse_out**0.5
    return rmse


@torch.no_grad()
def psnr(img1, img2, mask=None, pixel_max=1.0):
    """PSNR

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
    if mask is not None:
        if torch.isinf(psnr_out).any():
            print(mse_out.mean(), psnr_out.mean())
            psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
            psnr_out = psnr_out[~torch.isinf(psnr_out)]

    return psnr_out


@torch.no_grad()
def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT."""
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
        return psnr_out.item(), None
    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError
                if slice1.max() > 0:
                    result = ssim(slice1[None, None], slice2[None, None])
                    count += 1
                else:
                    result = 0
                results.append(result)
            results = torch.tensor(results)
            mean_results = torch.sum(results) / count
            ssims.append(mean_results.item())
        return float(np.mean(ssims)), ssims


@torch.no_grad()
def metric_proj(img1, img2, metric="psnr", axis=2, pixel_max=1.0):
    """Metrics for projection

    Args:
        img1 (_type_): [x, y, z]
        img2 (_type_): [x, y, z]
        pixel_max (float, optional): _description_. Defaults to 1.0.
    """
    assert axis in [0, 1, 2, None]
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    n_slice = img1.shape[axis]

    results = []
    count = 0
    for i in range(n_slice):
        if axis == 0:
            slice1 = img1[i, :, :]
            slice2 = img2[i, :, :]
        elif axis == 1:
            slice1 = img1[:, i, :]
            slice2 = img2[:, i, :]
        elif axis == 2:
            slice1 = img1[:, :, i]
            slice2 = img2[:, :, i]
        else:
            raise NotImplementedError
        if slice1.max() > 0:
            slice1 = slice1 / slice1.max()
            slice2 = slice2 / slice2.max()
            if metric == "psnr":
                result = psnr(
                    slice1[None, None], slice2[None, None], pixel_max=pixel_max
                )
            elif metric == "ssim":
                result = ssim(slice1[None, None], slice2[None, None])
            else:
                raise NotImplementedError
            count += 1
        else:
            result = 0
        results.append(result)
    results = torch.tensor(results)
    mean_results = torch.sum(results) / count
    return mean_results.item(), results.tolist()

def get_foreground_threshold(image_stack):
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture

    data_flat = image_stack.cpu().flatten().numpy()
    # bins = 1000
    # hist, bin_edges = np.histogram(data_flat, bins=bins, range=(0, 1), density=True)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # plt.hist(data_flat, bins=bins, density=True, alpha=0.6, color='g', label='Histogram')
    # plt.legend()
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.title('Histogram and Gaussian Mixture Fit')
    # plt.show(block=True)

    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(data_flat.reshape(-1, 1))
    means = gmm.means_.flatten()
    variances = np.sqrt(gmm.covariances_.flatten())
    if means[0] < means[1]:
        mean_background = means[0]
        sigma_background = variances[0]
    else:
        mean_background = means[1]
        sigma_background = variances[1]
    
    threshold = mean_background + 3 * sigma_background    
    print(gmm.means_)
    print(gmm.covariances_)    
    print(f"background threshold value: {threshold:.4f}")

    return threshold

def get_volume_mask(volume, threshold):
    """
    Apply masking to a 3D voxel tensor based on max projections along each axis.
    
    Parameters:
        voxel (torch.Tensor): 3D tensor representing the voxel data.
        threshold (float): Threshold value for masking.
        
    Returns:
        numpy.ndarray: Boolean mask as a numpy array.
    """
    voxel = volume.detach()
    # Compute max projections along each axis
    proj_xy = torch.max(voxel, dim=2).values  # Projection along z-axis
    proj_xz = torch.max(voxel, dim=1).values  # Projection along y-axis
    proj_yz = torch.max(voxel, dim=0).values  # Projection along x-axis
    
    # Create masks by thresholding the projections
    mask_xy = proj_xy > threshold
    mask_xz = proj_xz > threshold
    mask_yz = proj_yz > threshold
    
    # Expand dimensions to match voxel shape
    mask_xy = mask_xy.unsqueeze(2).expand_as(voxel)
    mask_xz = mask_xz.unsqueeze(1).expand_as(voxel)
    mask_yz = mask_yz.unsqueeze(0).expand_as(voxel)
    
    # Combine masks using logical AND operation
    final_mask = mask_xy & mask_xz & mask_yz
    
    return final_mask