from typing import Dict, Union
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import linregress
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
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
        # TODO if depth images already exist from LiDAR, extend them with pretrained model
        self.depth_filenames = self.metadata.get("depth_filenames")
        self.depth_unit_scale_factor = self.metadata.get("depth_unit_scale_factor", 1.0)
        # if not self.depth_filenames:
        # Currently always generate depth as LiDAR depth is sparse
        self._generate_depth_images(dataparser_outputs)

    def _load_lidar_depths(self):
        lidar_depths = []
        for depth_filename in self.depth_filenames:
            with Image.open(depth_filename) as img:
                depth_array = np.array(img)
                depth_tensor = torch.from_numpy(depth_array).float()
                lidar_depths.append(depth_tensor)

        return torch.stack(lidar_depths) if lidar_depths else None

    def _generate_depth_images(self, dataparser_outputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
        if cache.exists():
            print("Loading pseudodata depth from cache!")
            self.depths = torch.from_numpy(np.load(cache)).to(device)
        else:
            print("No depth data found! Generating pseudodepth...")
            depth_tensors = []
            repo = "LiheYoung/depth-anything-base-hf"
            image_processor = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForDepthEstimation.from_pretrained(repo)
            for image_filename in tqdm(dataparser_outputs.image_filenames, desc="Generating depth images"):
                pil_image = Image.open(image_filename)
                inputs = image_processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                depth_tensors.append(predicted_depth)
            depths = torch.stack(depth_tensors)
            
        # Load LiDAR depth data
        lidar_depths = self._load_lidar_depths()

        if lidar_depths is not None:
            # Compute scale and shift between monocular and LiDAR depths
            scale, shift = self.compute_scale_shift(depths, lidar_depths)
            self.adjusted_depths = scale * self.depths + shift
        else:
            print("No LiDAR depth data available for scaling and shifting.")
            
        np.save(cache, self.adjusted_depths.cpu().numpy())
        self.depth_filenames = None

    def compute_scale_shift(self, monocular_depths, lidar_depths):
        valid_mask = lidar_depths > 0  # Assuming zero where no LiDAR data
        scaled_monocular = monocular_depths[valid_mask].flatten()
        lidar_depths_flat = lidar_depths[valid_mask].flatten()

        slope, intercept, r_value, p_value, std_err = linregress(scaled_monocular.numpy(), lidar_depths_flat.numpy())
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