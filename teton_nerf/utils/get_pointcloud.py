"""Script that is meant to help us load frames and calculate pointclouds from them"""
from __future__ import annotations

import dataclasses
from typing import Tuple, cast

import numpy as np
import numpy as onp
import numpy.typing as onpt
import skimage.transform
from nerfstudio.data.utils.data_utils import get_depth_image_from_path

@dataclasses.dataclass
class Record3dFrame:
    """Inspiration for the method"""

    K: onpt.NDArray[onp.float32]
    rgb: onpt.NDArray[onp.uint8]
    depth: onpt.NDArray[onp.float32]
    mask: onpt.NDArray[onp.bool_]
    T_world_camera: onpt.NDArray[onp.float32]

    def get_point_cloud(self,
        downsample_factor: int = 1
    ) -> Tuple[onpt.NDArray[onp.float32], onpt.NDArray[onp.uint8]]:
        rgb = self.rgb[::downsample_factor, ::downsample_factor]
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            onpt.NDArray[onp.bool_],
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )
        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        img_wh = rgb.shape[:2][::-1]

        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), 2) + 0.5
        )
        grid = grid * downsample_factor

        homo_grid = np.pad(grid[mask], np.array([[0, 0], [0, 1]]), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], local_dirs)
        points = (T_world_camera[:, -1] + dirs * depth[mask, None]).astype(np.float32)
        point_colors = rgb[mask]

        return points, point_colors
    

import numpy as np
import torch

def generate_pointcloud(dataset, idx):
    """Function for computing the points and point colors
    in a pointcloud that can be displayed in the viewer

    Args:
        dataset (TetonNerfDataset): An instance of the TetonNerfDataset class that has
        depth images, rgb images and camera parameters
        idx (int): The index of the frame in the training dataset to access

    Returns:
        _type_: _description_
    """
    image = dataset[idx]["image"]  # Assuming this retrieves an RGB image tensor
    camera = dataset.cameras[idx]
    
    if dataset.depth_filenames is not None:
        filepath = dataset.depth_filenames[idx]
        height = int(camera.height.item())  # Camera object holds height and width
        width = int(camera.width.item())
        scale_factor = dataset.depth_unit_scale_factor * dataset.scale_factor

        depth = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )
    else:
        depth = dataset.depths[idx]

    depth = depth.cpu().numpy().astype('float32')
    # Assume RGB image is already in the correct format
    image = image.cpu().numpy().astype(np.uint8)
    
    assert depth.shape == image.shape[:2]
    
    # Get intrinsic parameters
    K = np.array([
        [camera.fx.item(), 0, camera.cx.item()],
        [0, camera.fy.item(), camera.cy.item()],
        [0, 0, 1]
    ])
    
    # Extrinsic matrix (camera to world)
    T_world_camera = camera.camera_to_worlds.cpu().numpy()
    
    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.reshape(-1)
    v = v.reshape(-1)
    ones = np.ones_like(u)
    
    # Homogeneous coordinates of the pixels
    uv1 = np.stack((u, v, ones)).T
    
    # Inverse of intrinsic matrix to get rays in camera coordinates
    rays = np.dot(np.linalg.inv(K), uv1.T).T
    
    # Scaling rays with depth to get 3D positions in camera coordinates
    points_camera = rays * depth.flatten()[:, None]
    
    # Transform points to world coordinates
    points_world = np.dot(T_world_camera[:3, :3], points_camera.T).T + T_world_camera[:3, 3]
    
    # Using RGB values from the image
    colors = image[v, u]

    return points_world, colors