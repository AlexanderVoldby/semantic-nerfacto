from typing import Dict, Union
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from rich.progress import track
from scipy.stats import linregress

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
        # TODO if depth images already exist from LiDAR, extend them with pretrained model
        self.depth_filenames = self.metadata.get("depth_filenames")
        self.depth_unit_scale_factor = self.metadata.get("depth_unit_scale_factor", 1.0)
        # if not self.depth_filenames:
        # Currently always generate depth as LiDAR depth is sparse
        if len(dataparser_outputs.image_filenames) > 0 and (
            "depth_filenames" not in dataparser_outputs.metadata.keys()
            or dataparser_outputs.metadata["depth_filenames"] is None
        ):
            self._generate_depth_images(dataparser_outputs)

    def _generate_depth_images(self, dataparser_outputs: DataparserOutputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
        if cache.exists():
            CONSOLE.print("[bold yellow] Loading pseudodata depth from cache!")
            self.depths = torch.from_numpy(np.load(cache)).to(device)
        else:
            # TODO: Invert pseudo-depth image as it outputs disparity
            # Scale lidar depth so it is in meters instead of millimeters.
            # Lidar depth should probably be scaled globally but maybe we can do it in the cache
            # Fit line to each image individually and appy scale and shift.
            CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
            losses.FORCE_PSEUDODEPTH_LOSS = True
            depth_tensors = []
            repo = "LiheYoung/depth-anything-base-hf"
            image_processor = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForDepthEstimation.from_pretrained(repo)
            for image_filename, depth_filename in track(zip(dataparser_outputs.image_filenames, dataparser_outputs.depth_filenames), description="Generating depth images"):
                pil_image = Image.open(image_filename)
                depth_image = Image.open(depth_filename)
                depth_array = np.array(depth_image)
                depth_tensor = torch.from_numpy(depth_array).float()
                inputs = image_processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=pil_image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                    )
                    # Fit the predicted_depth to the LiDAR depth
                    scale, shift = self.compute_scale_shift(prediction, depth_tensor)

                depth_tensors.append(prediction)
            self.depths = torch.stack(depth_tensors)
            np.save(cache, self.depths.cpu().numpy())
            self.depth_filenames = None

    def compute_scale_shift(self, monocular_depth, lidar_depth):
        valid_mask = lidar_depth > 0  # Assuming zero where no LiDAR data
        scaled_monocular = monocular_depth[valid_mask].flatten()
        lidar_depth_flat = lidar_depth[valid_mask].flatten()

        slope, intercept, r_value, p_value, std_err = linregress(scaled_monocular.numpy(), lidar_depth_flat.numpy())
        return slope, intercept

    def get_metadata(self, data: Dict) -> Dict:
        
        # handle mask
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label, mask = get_semantics_and_mask_tensors_from_path(
            filepath=filepath, mask_indices=self.mask_indices, scale_factor=self.scale_factor
        )
        if "mask" in data.keys():
            mask = mask & data["mask"]
        
        # Handle depth stuff
        image_idx = data["image_idx"]
        if self.depth_filenames is None:
            depth_image = self.depths[image_idx]
        else:
            filepath = self.depth_filenames[image_idx]
            height = int(self._dataparser_outputs.cameras.height[image_idx])
            width = int(self._dataparser_outputs.cameras.width[image_idx])
            scale_factor = self.depth_unit_scale_factor * self.scale_factor
            depth_image = get_depth_image_from_path(
                filepath=filepath, height=height, width=width, scale_factor=scale_factor
            )
        return {"mask": mask, "semantics": semantic_label, "depth_image": depth_image}
    
    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None