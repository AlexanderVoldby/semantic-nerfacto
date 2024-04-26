"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union
from jaxtyping import Float, UInt8
from PIL import Image
import numpy as np
import numpy.typing as npt
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from teton_nerf.teton_dataset import TetonNerfDataset


@dataclass
class TetonNerfDatamanagerConfig(VanillaDataManagerConfig):
    """Creates the semantic depth dataset for semantic depth nerfacto
    """
    
    _target: Type = field(default_factory=lambda: TetonNerfDatamanager)
    use_monocular_depth: bool = True
    """Whether to extend lidar depth with monocular depth"""


class TetonNerfDatamanager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TetonNerfDatamanagerConfig

    def __init__(
        self,
        config: TetonNerfDatamanagerConfig,
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
    
    def create_train_dataset(self) -> TetonNerfDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return TetonNerfDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            use_monocular_depth=self.config.use_monocular_depth)

    def create_eval_dataset(self) -> TetonNerfDataset:
        return TetonNerfDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            use_monocular_depth=self.config.use_monocular_depth)
    
    def get_numpy_depth(self, image_idx: int) -> npt.NDArray[np.float32]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        depth_filename = self.metadata.depth_filenames[image_idx]
        pil_image = Image.open(depth_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        depth = np.array(pil_image, dtype="float32")  # shape is (h, w) or (h, w, 3 or 4)
        assert len(depth.shape) == 2
        assert depth.dtype == np.float32
        return depth