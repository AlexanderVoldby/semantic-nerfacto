"""
Pipeline for the custom model. It is in the pipeline that the sparse regularization
from regNeRF / nerfbusters is implemented
"""
import torch
import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
from torchtyping import TensorType

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from semantic_nerfacto.semantic_nerfacto_datamanager import SemanticNerfactoDataManagerConfig
from semantic_nerfacto.semantic_nerfacto import SemanticNerfactoModel, SemanticNerfactoModelConfig
from utils.random_train_pose import random_train_pose

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
class SemanticNerfactoPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SemanticNerfactoPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = SemanticNerfactoDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = SemanticNerfactoModelConfig()
    """specifies the model config"""
    use_regnerf_depth_loss: bool = False
    """Whether to use reqgularization on depth patches"""
    use_regnerf_rgb_loss: bool = False
    """Whether to use regularization on RGB patches"""
    use_regnerf_semantics_loss: bool = False
    """Whether to use regularization on RGB patches"""
    regnerf_depth_loss_mult = 1e-6
    regnerf_rgb_loss_mult = 1e-6
    regnerf_semantics_loss_mult = 1e-6
    """Loss multipliers for the regularization. Do i need to finetune these?"""
    

class SemanticNerfactoPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: SemanticNerfactoPipelineConfig,
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
                SemanticNerfactoModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
    
    # TODO: Add total variation loss as included in Nerfbusters
    # Requires getting the loss_dict from the model and adding the TV-loss
    
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
        activate_patch_sampling = self.config.use_regnerf_loss

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
            # TODO(ethan): fix indices
            camera_indices = torch.tensor(list(range(self.config.num_patches))).unsqueeze(-1)
            ray_bundle_patches = cameras.generate_rays(
                camera_indices
            )  # (patch_resolution, patch_resolution, num_patches)
            ray_bundle_patches = ray_bundle_patches.flatten()
            # TODO: deal with appearance embeddings
            model_outputs_patches = self.model.get_outputs(ray_bundle_patches)
            rgb_patches = (
                model_outputs_patches["rgb"]
                .reshape(self.config.patch_resolution, self.config.patch_resolution, self.config.num_patches, 3)
                .permute(2, 0, 1, 3)
            )  # (num_patches, patch_resolution, patch_resolution, 3)
            depth_patches = (
                model_outputs_patches["depth"]
                .reshape(self.config.patch_resolution, self.config.patch_resolution, self.config.num_patches, 1)
                .permute(2, 0, 1, 3)[..., 0]
            )  # (num_patches, patch_resolution, patch_resolution)
            semantics_patches = (
                model_outputs_patches["semantics"]
                .reshape(self.config.patch_resolution, self.config.patch_resolution, self.config.num_patches, 1)
                .permute(2, 0, 1, 3)[..., 0]
            )  # (num_patches, patch_resolution, patch_resolution)

        if self.config.use_regnerf_depth_loss:
            regnerf_loss = self.apply_regnerf_loss(step, depth_patches)
            loss_dict["regnerf_depth_loss"] = self.config.regnerf_loss_mult * regnerf_loss
            
        if self.config.use_regnerf_rgb_loss:
            regnerf_loss = self.apply_regnerf_loss(step, rgb_patches)
            loss_dict["regnerf_rgb_loss"] = self.config.regnerf_loss_mult * regnerf_loss
            
        if self.config.use_regnerf_semantics_loss:
            regnerf_loss = self.apply_regnerf_loss(step, semantics_patches)
            loss_dict["regnerf_semantics_loss"] = self.config.regnerf_loss_mult * regnerf_loss
        
        return model_outputs, loss_dict, metrics_dict