from typing import Dict, Union
import numpy as np
import torch
import json
import os
import gc
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import linregress
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.stats import linregress

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path, get_depth_image_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.rich_utils import CONSOLE

from semantic_nerfacto.visualizations import compare_depth_and_image, visualize_depth_before_and_after_scaling


def upsample_depth(depth_tensor, target_height, target_width):
    # Helper function to upsaple monocular depth to the same size as liDAR depth
    if depth_tensor.dim() == 2:
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    elif depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.unsqueeze(1)
    
    # Upsample to match the input image size
    depth_upsampled = F.interpolate(depth_tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)
    return depth_upsampled.squeeze()  # remove extra dimensions if added

class SemanticDepthDataset(InputDataset):
    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "semantics", "depth_image"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, use_monocular_depth= True):
        super().__init__(dataparser_outputs, scale_factor)
        # assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        self.semantics = self.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)
        self.use_monocular_depth = use_monocular_depth

        # Depth image handling
        # TODO if depth images already exist from LiDAR, extend them with pretrained model
        self.depth_filenames = self.metadata.get("depth_filenames")
        self.depth_unit_scale_factor = self.metadata.get("depth_unit_scale_factor", 1.0)
        # if not self.depth_filenames:
        # Currently always generate depth as LiDAR depth is sparse
        assert len(dataparser_outputs.image_filenames) > 0 and (
            "depth_filenames" in dataparser_outputs.metadata.keys()
            or dataparser_outputs.metadata["depth_filenames"] is not None
        ), "No depth images in dataset"
        
        if self.use_monocular_depth:
            self._generate_depth_images(dataparser_outputs)


    def _generate_depth_images(self, dataparser_outputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
        
        if cache.exists():
            print("Loading pseudodata depth from cache!")
            self.depths = torch.from_numpy(np.load(cache)).to(device)
        else:
            # TODO: Invert pseudo-depth image as it outputs disparity
            # Scale lidar depth so it is in meters instead of millimeters.
            # Lidar depth should probably be scaled globally but maybe we can do it in the cache
            # Fit line to each image individually and appy scale and shift.
            CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
            # losses.FORCE_PSEUDODEPTH_LOSS = True
            depth_tensors = []
            # Change to small to speed up otherwise use depth-anything-base
            repo = "LiheYoung/depth-anything-base-hf"
            image_processor = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForDepthEstimation.from_pretrained(repo)
            for i, (image_filename, depth_filename) in enumerate(tqdm(zip(dataparser_outputs.image_filenames, self.depth_filenames), desc="Generating depth images")):
                pil_image = Image.open(image_filename)
                depth_image = Image.open(depth_filename)
                depth_array = np.array(depth_image)
                depth_tensor = torch.from_numpy(depth_array).float() * 1e-3 # Divide by 1000 to scale to meters
                inputs = image_processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = 1 / outputs.predicted_depth
                    predicted_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=pil_image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                    )
                    prediction = prediction.squeeze()
                    # Fit the predicted_depth to the LiDAR depth
                    scale, shift = self.compute_scale_shift(prediction, depth_tensor)
                    depth  = scale * prediction + shift
                    
                    # Save every 20th image
                    if i % 20 == 0:
                        name = os.path.basename(image_filename)
                        folder = str(image_filename.parent.parent)
                        saved_name = folder + "/" + name
                        
                        try:
                            compare_depth_and_image(inputs, depth, saved_name)
                            visualize_depth_before_and_after_scaling(
                                pil_image,
                                depth_tensor,
                                prediction,
                                depth,
                                saved_name
                            )
                            print(f"Saved figures under {name}")
                        except Exception as e:
                            print(f"Error saving image: {e}")

                depth_tensors.append(depth)
                
            self.depths = torch.stack(depth_tensors)
            np.save(cache, self.depths.cpu().numpy())

            # Delete some stuff to avoid exceeding GPU memory
            del depth_tensors
            torch.cuda.empty_cache()
            gc.collect()
            self.depth_filenames = None
            
        # Save filename and index to later retrieve correspondng depth image from dataset since dataparser_outputs removes some images due to high blur score
        itd = {i: str(image_filename) for i, image_filename in enumerate(dataparser_outputs.image_filenames)}
        data_dir = str(dataparser_outputs.image_filenames[0].parent.parent)
        json_name = data_dir + "/index_to_depth.json"
        if not os.path.exists(json_name):
            with open(json_name, "w") as outfile: 
                json.dump(itd, outfile)

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