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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path, get_depth_image_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE

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

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        # assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        self.semantics = self.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)

        # Depth image handling
        # TODO if depth images already exist from LiDAR, extend them with pretrained model
        self.depth_filenames = self.metadata.get("depth_filenames")
        self.depth_unit_scale_factor = self.metadata.get("depth_unit_scale_factor", 1.0)
        if self.metadata["use_monocular_depth"]:

            CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
            losses.FORCE_PSEUDODEPTH_LOSS = True
            CONSOLE.print("[bold red] Using psueodepth: forcing depth loss to be ranking loss.")

            self._generate_depth_images(dataparser_outputs)

    def _load_lidar_depths(self):
        lidar_depths = []
        for depth_filename in self.depth_filenames:
            with Image.open(depth_filename) as img:
                depth_array = np.array(img).astype(np.float32)
                depth_tensor = torch.from_numpy(depth_array).float()
                lidar_depths.append(depth_tensor)

        return torch.stack(lidar_depths) if lidar_depths else None

    def _generate_depth_images(self, dataparser_outputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
        
        if cache.exists():
            print("Loading pseudodata depth from cache!")
            depths = torch.from_numpy(np.load(cache)).to(device)
        else:
            print("No depth data found! Generating pseudodepth...")
            depth_tensors = []
            # Change to small to speed up otherwise use depth-anything-base
            repo = "LiheYoung/depth-anything-base-hf"
            image_processor = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForDepthEstimation.from_pretrained(repo)
            
            for image_filename in tqdm(dataparser_outputs.image_filenames, desc="Generating depth images"):
                pil_image = Image.open(image_filename)
                inputs = image_processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)

                    # depth anything outputs disparity so we need to convert it to depth
                    # we also standardize the depth to be in the range [0, 1]. These will later be shifted using the LiDAR depth
                    predicted_depth = 1 / outputs.predicted_depth
                    predicted_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
                    # Hardcode the output size from Polycam
                    upsampled_depth = upsample_depth(predicted_depth, 738, 994)

                    if False:
                        # save the depth image and image
                        import mediapy as mp
                        import cv2
                        upsampled_depth = (upsampled_depth - upsampled_depth.min()) / (upsampled_depth.max() - upsampled_depth.min())
                        min_ = torch.where(upsampled_depth == upsampled_depth.min())
                        max_ = torch.where(upsampled_depth == upsampled_depth.max())
                        upsampled_depth = (upsampled_depth[..., None].repeat(1, 1, 3) * 255).cpu().numpy().astype(np.uint8)
                        upsampled_depth = cv2.circle(upsampled_depth, (min_[1].item(), min_[0].item()), 50, (0, 0, 255), -1)
                        cv2.circle(upsampled_depth, (max_[1].item(), max_[0].item()), 50, (255, 0, 0), -1)
                        mp.write_image(f"depth.png", upsampled_depth)
                        mp.write_image(f"image.png", inputs.pixel_values[0].permute(1,2,0).cpu().numpy())


                depth_tensors.append(upsampled_depth)
            depths = torch.stack(depth_tensors)
            print(f"Depths shape after generating: {depths.shape}")

            # Load LiDAR depth data
            lidar_depths = self._load_lidar_depths()

            # Convert to meters
            lidar_depths = lidar_depths / 1000.0

            if lidar_depths is not None:
                # Compute scale and shift between monocular and LiDAR depths
                self.depths = []
                for i in range(len(lidar_depths)):
                    scale, shift = self.compute_scale_shift(depths[i].cpu(), lidar_depths[i])
                    scaled_depth = scale * depths[i].cpu() + shift

                    # Use the lidar depth in places where it exists and other wise the monocular depth.
                    valid_mask = lidar_depths[i] > 0
                    scaled_depth[valid_mask] = scaled_depth[valid_mask]
                    self.depths.append(scaled_depth)
    
                    if False:
                        # visualize depth maps before and after scaling
                        import matplotlib.pyplot as plt

                        # create subfigure with three images (lidar depth, monocular depth, scaled monocular depth).
                        # include a colorbar for each image.

                        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
                        axs[0].imshow(lidar_depths[i].cpu())
                        axs[0].set_title("LiDAR depth")
                        axs[0].axis("off")
                        plt.colorbar(axs[0].imshow(lidar_depths[i].cpu()), ax=axs[0])

                        axs[1].imshow(depths[i].cpu())
                        axs[1].set_title("Monocular depth")
                        axs[1].axis("off")
                        plt.colorbar(axs[1].imshow(depths[i].cpu()), ax=axs[1])

                        axs[2].imshow(scaled_depth.cpu())
                        axs[2].set_title("Scaled monocular depth")
                        axs[2].axis("off")
                        plt.colorbar(axs[2].imshow(self.depths[i].cpu()), ax=axs[2])

                        # show valid mask
                        axs[3].imshow(valid_mask.cpu())
                        axs[3].set_title("Valid mask")
                        axs[3].axis("off")
                        plt.colorbar(axs[3].imshow(valid_mask.cpu()), ax=axs[3])

                        # show image
                        img = Image.open(dataparser_outputs.image_filenames[i])
                        axs[4].imshow(img)
                        axs[4].set_title("Image")
                        axs[4].axis("off")

                        # save the figure
                        plt.savefig("alignment.png")
                        plt.close()

                self.depths = torch.stack(self.depths)

            else:
                print("No LiDAR depth data available for scaling and shifting.")
                
            np.save(cache, self.depths)
            # Free up memory to prevent GPU from running out of memory
            del depth_tensors, depths
            torch.cuda.empty_cache()
            gc.collect()
            self.depth_filenames = None
            
        # Save filename and index to later retrieve correspondng depth image from dataset since dataparser_outputs removes some depth images
        itd = {i: str(image_filename) for i, image_filename in enumerate(dataparser_outputs.image_filenames)}
        data_dir = str(dataparser_outputs.image_filenames[0].parent.parent)
        json_name = data_dir + "/index_to_depth.json"
        if not os.path.exists(json_name):
            with open(json_name, "w") as outfile: 
                json.dump(itd, outfile)

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