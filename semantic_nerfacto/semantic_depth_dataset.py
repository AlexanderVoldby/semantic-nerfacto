from typing import Dict, Union
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path, get_depth_image_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE


class SemanticDepthDataset(InputDataset):
    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "semantics", "depth_image"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        self.semantics = self.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)

        # Depth image handling
        self.depth_filenames = self.metadata.get("depth_filenames")
        self.depth_unit_scale_factor = self.metadata.get("depth_unit_scale_factor", 1.0)
        if not self.depth_filenames:
            self._generate_depth_images(dataparser_outputs)

    def _generate_depth_images(self, dataparser_outputs: DataparserOutputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
        if cache.exists():
            CONSOLE.print("[bold yellow] Loading pseudodata depth from cache!")
            self.depths = torch.from_numpy(np.load(cache)).to(device)
        else:
            CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
            losses.FORCE_PSEUDODEPTH_LOSS = True
            depth_tensors = []
            repo = "isl-org/ZoeDepth"
            self.zoe = torch_compile(torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(device))
            for image_filename in track(dataparser_outputs.image_filenames, description="Generating depth images"):
                pil_image = Image.open(image_filename)
                image = np.array(pil_image, dtype="float32") / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    depth_tensor = self.zoe.infer(image).squeeze().unsqueeze(-1)
                depth_tensors.append(depth_tensor)
            self.depths = torch.stack(depth_tensors)
            np.save(cache, self.depths.cpu().numpy())
            self.depth_filenames = None

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)
        image_idx = data["image_idx"]
        if self.depth_filenames is None:
            metadata["depth_image"] = self.depths[image_idx]
        else:
            filepath = self.depth_filenames[image_idx]
            height = int(self._dataparser_outputs.cameras.height[image_idx])
            width = int(self._dataparser_outputs.cameras.width[image_idx])
            scale_factor = self.depth_unit_scale_factor * self.scale_factor
            metadata["depth_image"] = get_depth_image_from_path(
                filepath=filepath, height=height, width=width, scale_factor=scale_factor
            )
        return metadata