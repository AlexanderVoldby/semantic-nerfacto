"""Script that is meant to help us load frames and calculate pointclouds from them"""
from __future__ import annotations

import dataclasses
from typing import Tuple, cast

import numpy as np
import numpy as onp
import numpy.typing as onpt
import skimage.transform
from nerfstudio.data.utils.data_utils import get_depth_image_from_path

import numpy as np
import torch


def generate_pointcloud(dataset, skip_interval=10):
    points_and_colors = []
    
    for idx in range(len(dataset)):
        image_tensor = dataset[idx]["image"]  # Assuming this retrieves an RGB image tensor
        camera = dataset.cameras[idx]
        if "depths" in dir(dataset):
            depth = dataset.depths[idx] * 1e3  # Multiply to return to original scale
        else:
            filename = dataset.metadata["depth_filenames"][idx]
            depth = get_depth_image_from_path(filename)
            depth *= 1e3
        
        depth_np = depth.cpu().numpy().astype('float32')
        image_np = image_tensor.cpu().numpy().astype(np.uint8)

        mask = np.zeros_like(depth_np, dtype=bool)
        mask[::skip_interval, ::skip_interval] = True
        print(f"Shape of depth: {depth_np.shape}, Shape of mask: {mask.shape}")
        
        K = np.array([
            [camera.fx.item(), 0, camera.cx.item()],
            [0, camera.fy.item(), camera.cy.item()],
            [0, 0, 1]
        ])
        T_world_camera = camera.camera_to_worlds.cpu().numpy()
        
        u, v = np.meshgrid(np.arange(depth_np.shape[1]), np.arange(depth_np.shape[0]))
        selected_u = u[mask]
        selected_v = v[mask]
        uv1 = np.stack([selected_u, selected_v, np.ones_like(selected_u)], axis=-1)
        
        local_dirs = np.dot(np.linalg.inv(K), uv1.T).T
        dirs = np.dot(T_world_camera[:3, :3], local_dirs.T).T
        points = T_world_camera[:3, -1] + dirs * depth_np[mask].reshape(-1, 1)

        color = image_np[selected_v, selected_u]

        points_and_colors.append((points, color))

    return points_and_colors

def generate_pointcloud_advanced(dataset, downsample_factor=1):
    """
    Generate a 3D point cloud from RGB and depth images using intrinsic and extrinsic camera parameters.
    
    Args:
        rgb (numpy.ndarray): The RGB image array of shape (H, W, 3).
        depth (numpy.ndarray): The depth image array of shape (H, W).
        K (numpy.ndarray): The 3x3 camera intrinsic matrix.
        T_world_camera (numpy.ndarray): The 4x4 camera extrinsic matrix.
        mask (numpy.ndarray): A boolean mask of shape (H, W) to select relevant pixels.
        downsample_factor (int): Factor by which to downsample the pixel grid.
    
    Returns:
        numpy.ndarray: 3D points in world coordinates.
        numpy.ndarray: Corresponding colors from the RGB image.
    """
    
    points_and_colors = []
    
    for idx in range(len(dataset)):
        image_tensor = dataset[idx]["image"]  # Assuming this retrieves an RGB image tensor
        camera = dataset.cameras[idx]
        depth = dataset.depths[idx]  # Assuming a method that handles loading and any necessary scaling
        
        depth_np = depth.cpu().numpy().astype('float32') * 1e3 # Scale to return to original NS scale
        image_np = image_tensor.cpu().numpy().astype(np.uint8) # .transpose(1, 2, 0)  # Correctly orient the image

        mask = np.zeros_like(depth_np, dtype=bool)
        mask[::10, ::10] = True
        print(f"Shape of depth: {depth_np.shape}, Shape of mask: {mask.shape}")
        
        K = np.array([
            [camera.fx.item(), 0, camera.cx.item()],
            [0, camera.fy.item(), camera.cy.item()],
            [0, 0, 1]
        ])
        T_world_camera = camera.camera_to_worlds.cpu().numpy()
    
        # Get image dimensions and adjust by the downsample factor
        img_wh = image_np.shape[:2][::-1]
        img_wh = tuple(dim // downsample_factor for dim in img_wh)
        
        # Generate a grid of (x, y) pixel coordinates
        grid = np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1]), indexing='xy'), axis=-1) + 0.5
        grid *= downsample_factor

        masked_grid = grid[mask]
        homo_grid = np.hstack([masked_grid, np.ones((masked_grid.shape[0], 1))])

        local_dirs = np.einsum('ij,nj->ni', np.linalg.inv(K), homo_grid)
        points_camera = local_dirs * depth_np[mask].reshape(-1, 1)

        points_world = np.einsum('ij,nj->ni', T_world_camera[:3, :3], points_camera) + T_world_camera[:3, -1]
        point_colors = image_np[mask]
        
        points_and_colors.append((points_world, point_colors))

    return points_and_colors