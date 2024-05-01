"""
Pipeline for  semantic depth nerfacto. Very similar to vanilla pipeline
"""

import torch
import typing
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal, Tuple, Type
from torchtyping import TensorType
import torch
import numpy as np

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from teton_nerf.teton_datamanager import TetonNerfDatamanagerConfig
from teton_nerf.teton_nerf import TetonNerfModel, TetonNerfModelConfig
from teton_nerf.utils.random_train_pose import random_train_pose
from teton_nerf.utils.get_pointcloud import generate_point_cloud

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)

from nerfstudio.viewer.viewer_elements import ViewerControl
from nerfstudio.viewer.control_panel import ViewerCheckbox
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler

@dataclass
class TetonNerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: TetonNerfPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = TetonNerfDatamanagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = TetonNerfModelConfig()
    """specifies the model config"""
    
    # patch sampling
    num_patches: int = 10
    """Number of patches per batch for training"""
    patch_resolution: int = 32
    """Patch resolution, where DiffRF used 48x48 and RegNeRF used 8x8"""
    focal_range: Tuple[float, float] = (3.0, 3.0)
    """Range of focal length"""
    central_rotation_range: Tuple[float, float] = (-180, 180)
    """Range of central rotation"""
    vertical_rotation_range: Tuple[float, float] = (-90, 20)
    """Range of vertical rotation"""
    jitter_std: float = 0.05
    """Std of camera direction jitter, so we don't just point the cameras towards the center every time"""
    center: Tuple[float, float, float] = (0, 0, 0)
    """Center coordinate of the camera sphere"""
    aabb_scalar: float = 1.5
    
    # Losses
    use_regnerf_depth_loss: bool = True
    """Whether to use reqgularization on depth patches"""
    use_regnerf_rgb_loss: bool = True
    """Whether to use regularization on RGB patches"""
    use_regnerf_semantics_loss: bool = True
    """Whether to use regularization on semantics patches"""
    regnerf_depth_loss_mult = 1
    regnerf_rgb_loss_mult = 1
    regnerf_semantics_loss_mult = 1


class TetonNerfPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: TetonNerfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                TetonNerfModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
            
        # Stuff to visualize pointcloud in viewer
        self.show_pcd_button = ViewerCheckbox("Show Point Cloud", False, cb_hook=self.add_point_clouds)
        self.viewer_control = ViewerControl() # This will be found and _setup by viewer
        
    def add_point_clouds(self, checkbox: ViewerCheckbox):
        # TODO: add point cloud to the viewer
        # take the depth images and backproject them to 3D points
        
        pcd = generate_point_cloud(pipeline=self,
                                   remove_outliers=False,
                                   bounding_box_min=(-2, -2, -2),
                                   bounding_box_max=(2, 2, 2))

        colors = np.asarray(pcd.colors)
        points = np.asarray(pcd.points)
        poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
        poses[:, :3, 3] = points
        poses = self.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
            torch.from_numpy(poses)
        )
        points = poses[:, :3, 3].numpy()
        points *= 1 # Try to scale to fit scene better
        torch.cuda.empty_cache()
        
        if checkbox.value:
            self.viewer_control.viser_server.add_point_cloud(
                name="point_cloud",
                points=points,
                colors=colors,
                point_size=0.01,
                point_shape="rounded",
            )
        else:
            pass

    
    # TODO: Add stuff that visualizes the patches
    def apply_regnerf_loss(self, step: int, patches_density: TensorType["num_patches", "res", "res"]):
        pd = patches_density
        delta_x = pd[..., :-1, 1:] - pd[..., :-1, :-1]
        delta_y = pd[..., 1:, :-1] - pd[..., :-1, :-1]
        loss = torch.mean(delta_x**2 + delta_y**2)
        return loss
    
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)

        # --------------------- 2D losses ---------------------
        activate_patch_sampling = self.config.use_regnerf_depth_loss or self.config.use_regnerf_rgb_loss or self.config.use_regnerf_semantics_loss

        # TODO: debug why patch sampling decreases model performance
        if activate_patch_sampling:
            cameras, vertical_rotation, central_rotation = random_train_pose(
                size=self.config.num_patches,
                resolution=self.config.patch_resolution,
                device=self.device,
                radius_mean=self.config.aabb_scalar,  # no sqrt(3) here
                radius_std=0.0,
                central_rotation_range=self.config.central_rotation_range,
                vertical_rotation_range=self.config.vertical_rotation_range,
                focal_range=self.config.focal_range,
                jitter_std=self.config.jitter_std,
                center=self.config.center,
            )

            camera_indices = torch.tensor(list(range(self.config.num_patches))).unsqueeze(-1)
            ray_bundle_patches = cameras.generate_rays(
                camera_indices
            )  # (patch_resolution, patch_resolution, num_patches)
            ray_bundle_patches = ray_bundle_patches.flatten()

            model_outputs_patches = self.model(ray_bundle_patches)

        if self.config.use_regnerf_depth_loss:
            depth_patches = (
                model_outputs_patches["depth"]
                .reshape(self.config.patch_resolution, self.config.patch_resolution, self.config.num_patches, 1)
                .permute(2, 0, 1, 3)[..., 0]
            )  # (num_patches, patch_resolution, patch_resolution)
            regnerf_loss = self.apply_regnerf_loss(step, depth_patches)
            loss_dict["regnerf_depth_loss"] = self.config.regnerf_depth_loss_mult * regnerf_loss
            
        if self.config.use_regnerf_rgb_loss:
            rgb_patches = (
                model_outputs_patches["rgb"]
                .reshape(self.config.patch_resolution, self.config.patch_resolution, self.config.num_patches, 3)
                .permute(2, 0, 1, 3)
            )
            regnerf_loss = self.apply_regnerf_loss(step, rgb_patches)
            loss_dict["regnerf_rgb_loss"] = self.config.regnerf_rgb_loss_mult * regnerf_loss
            
        if self.config.use_regnerf_semantics_loss:
            semantics_patches = (
                model_outputs_patches["semantics"]
                .reshape(self.config.patch_resolution, self.config.patch_resolution, self.config.num_patches, self.model.num_classes)
                .permute(2, 0, 1, 3)[..., 0]
            )  # (num_patches, patch_resolution, patch_resolution)
            regnerf_loss = self.apply_regnerf_loss(step, semantics_patches)
            loss_dict["regnerf_semantics_loss"] = self.config.regnerf_semantics_loss_mult * regnerf_loss

        return model_outputs, loss_dict, metrics_dict
    
    