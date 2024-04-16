from dataclasses import dataclass

from typing import Literal

import torch
from typing_extensions import Annotated
import json
import numpy as np

from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.scripts.render import BaseRender, _render_trajectory_video
from nerfstudio.cameras.camera_utils import normalize, rotation_matrix
from nerfstudio.data.dataparsers.base_dataparser import transform_poses_to_original_space

@dataclass
class RenderDepthBasedTransformedPath(BaseRender):
    pose_source: Literal["train", "eval"] = "eval"
    interpolation_steps: int = 20
    transform_cameras: bool = True

    def main(self) -> None:
        config, pipeline, checkpoint_path, step = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )
        

        if self.pose_source == "eval":
            cameras = pipeline.datamanager.eval_dataset.cameras
        else:
            cameras = pipeline.datamanager.train_dataset.cameras

        # Get the dataparser_transforms that is applied to take the camera frames from
        # world coordinates to camera coordinates
        f = open(checkpoint_path.parent.parent / "dataparser_transforms.json")
        dataparser_transforms = json.load(f)
        self.transform = torch.tensor(dataparser_transforms["transform"])
        self.scale = dataparser_transforms["scale"]
        f.close()

        poses_original_space = transform_poses_to_original_space(cameras.camera_to_worlds, self.transform, self.scale)
        cameras.camera_to_worlds = poses_original_space

        if self.transform_cameras:
            cameras = apply_depth_based_transformations(cameras, pipeline, self.scale)

        # Sample and interpolate cameras
        camera_path = get_interpolated_camera_path(
            cameras=cameras,
            steps=self.interpolation_steps,
            order_poses=True,  # Assuming you might want to order by proximity or another metric
        )


        # Render the trajectory video with transformed camera path
        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds= self.interpolation_steps * len(camera_path) / self.frame_rate,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )

def apply_depth_based_transformations(cameras, pipeline, scale):
	# Generate rays and sample density
	central_ray_bundles = get_central_rays(cameras)
	print(central_ray_bundles)
	# Get the depth of the central rays
	# TODO: Check if scaling by the scale from dataparser_transforms is adequate
	depths = scale * get_depths_of_central_rays(central_ray_bundles, pipeline)

	# Compute transformation for camera based on depths
	new_camera_matrices = compute_transformations(cameras, depths)
	fx = cameras.fx
	fy = cameras.fy
	cx = cameras.cx
	cy = cameras.cy
	return Cameras(new_camera_matrices, fx, fy, cx, cy)

def camera_coordinates_to_world_coordinates(transformation_matrix, scale_factor):
    """
    Computes the inverse of a scaled 3x4 transformation matrix where the scale factor
    is applied after the matrix transformation.

    Args:
    transformation_matrix (numpy.ndarray): A 3x4 matrix representing rotation and translation.
    scale_factor (float): The scale factor applied after the matrix transformation.

    Returns:
    numpy.ndarray: The inverse transformation matrix.
    """
    # Extract the rotation and translation components from the 3x4 matrix
    rotation_matrix = transformation_matrix[:, :3]
    translation_vector = transformation_matrix[:, 3]

    # Compute the inverse of the rotation matrix
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)

    # Adjust the translation vector for the scale and compute its inverse transformation
    scaled_translation_vector = translation_vector / scale_factor
    inverse_translation_vector = -inverse_rotation_matrix @ scaled_translation_vector

    # Construct the full inverse transformation matrix (3x4)
    inverse_transformation_matrix = np.hstack((inverse_rotation_matrix, inverse_translation_vector.reshape(-1, 1)))

    return inverse_transformation_matrix

def compute_transformations(cameras, depths):
    """Transform each camera based on the corresponding depth and reverse its direction."""
    new_camera_matrices = []

    for idx, depth in enumerate(depths):
        camera_matrix = cameras.camera_to_worlds[idx]
        R = camera_matrix[:3, :3]
        t = camera_matrix[:3, 3]

        # The look_at direction is the negative of the third column of R
        look_at = -R[:, 2]

        # Normalize and compute translation vector
        look_at_normalized = normalize(look_at.unsqueeze(0)).squeeze(0)
        translation_vector = look_at_normalized * depth

        # Compute new camera position
        new_position = t + translation_vector

        # Compute new look_at vector (pointing in the opposite direction)
        new_look_at = -look_at_normalized

        # Compute rotation to align the new look_at with the z-axis
        z_axis = torch.tensor([0, 0, 1], device=camera_matrix.device)
        R_new = rotation_matrix(new_look_at, z_axis)

        # Create new camera matrix
        new_camera_matrix = torch.eye(4, device=camera_matrix.device)
        new_camera_matrix[:3, :3] = R_new
        new_camera_matrix[:3, 3] = new_position

        new_camera_matrices.append(new_camera_matrix)

    return new_camera_matrices


def get_central_rays(cameras):
    """Compute the central rays for all cameras in the Cameras object."""
    # Central pixel coordinates
    central_x = cameras.cx
    central_y = cameras.cy

    # Create image coordinates for the central ray of each camera
    batch_size = central_x.shape[0]  # Number of cameras
    image_coords = torch.stack([central_y.squeeze(-1), central_x.squeeze(-1)], dim=-1)  # Ensure shape is [N, 2]

    # Generate rays for all cameras
    central_ray_bundles = []
    for idx in range(batch_size):
        ray_bundle = cameras.generate_rays(
            camera_indices=idx,
            coords=image_coords[idx].unsqueeze(0)  # Add batch dimension
        )
        central_ray_bundles.append(ray_bundle)

    return central_ray_bundles

def get_depths_of_central_rays(ray_bundle, pipeline):
    """Get the depths of the central rays for all cameras using the pipeline."""
    depths = []

    for ray in ray_bundle:
        # Retrieve depth from the model for each central ray
        outputs = pipeline.model.get_outputs(ray)
        central_ray_depth = outputs['depth']  # Assumes depth is returned here
        depths.append(central_ray_depth)

    return depths