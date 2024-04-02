"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from semantic_nerfacto.semantic_depth_dataset import SemanticDepthDataset


@dataclass
class SemanticNerfactoDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: SemanticNerfactoDataManager)


class SemanticNerfactoDataManager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: SemanticNerfactoDataManagerConfig

    def __init__(
        self,
        config: SemanticNerfactoDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
    
    def create_train_dataset(self) -> SemanticDepthDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return SemanticDepthDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> SemanticDepthDataset:
        return SemanticDepthDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )