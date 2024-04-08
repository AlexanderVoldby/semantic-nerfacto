from typing import Dict, Union
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from rich.progress import track
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

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
        # assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        
        # Depth image handling
        # TODO if depth images already exist from LiDAR, extend them with pretrained model
        self.semantics = self.metadata["semantics"]
        self.depth_filenames = self.metadata.get("depth_filenames")
        self.depth_unit_scale_factor = self.metadata.get("depth_unit_scale_factor", 1.0)
        # if not self.depth_filenames:
        # Currently always generate depth as LiDAR depth is sparse
        self._generate_segmentation_images(dataparser_outputs)
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
            repo = "LiheYoung/depth-anything-base-hf"
            image_processor = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForDepthEstimation.from_pretrained(repo)
            for image_filename in track(dataparser_outputs.image_filenames, description="Generating depth images"):
                pil_image = Image.open(image_filename)
                inputs = image_processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                depth_tensors.append(predicted_depth)
            self.depths = torch.stack(depth_tensors)
            np.save(cache, self.depths.cpu().numpy())
            self.depth_filenames = None

    def _initialize_segmentor(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        return predictor, metadata

    def _generate_segmentation_images(self, dataparser_outputs: DataparserOutputs):
        CONSOLE.print("[bold yellow] Generating or loading semantic segmentations...")
        segmentor, metadata = self._initialize_segmentor()
        
        cache = dataparser_outputs.image_filenames[0].parent / "semantics.npy"
        if cache.exists():
            self.segmentations = torch.from_numpy(np.load(cache))
        else:
            segmentations = []
            for image_filename in track(dataparser_outputs.image_filenames, description="Generating segmentations"):
                image = cv2.imread(str(image_filename))
                panoptic_seg, _ = segmentor(image)["panoptic_seg"]
                segmentations.append(torch.from_numpy(panoptic_seg.cpu().numpy()))
            self.segmentations = torch.stack(segmentations)
            np.save(cache, self.segmentations.numpy())
            self.semantics = None
        
        metadict = {
            "thing_classes": metadata.thing_classes,
            "stuff_classes": metadata.stuff_classes,
            "thing_colors": metadata.thing_colors,
            "stuff_colors": metadata.stuff_colors,
            "thing_dataset_id_to_contiguous_id": metadata.thing_dataset_id_to_contiguous_id,
            "stuff_dataset_id_to_contiguous_id": metadata.stuff_dataset_id_to_contiguous_id
        }
        
        data_dir = dataparser_outputs.metadata["data_dir"]
        json_file_path = f"{data_dir}/panoptic_classes.json"

        # Save the metadata to a JSON file
        with open(json_file_path, 'w') as f:
            json.dump(metadict, f, indent=4)
            

    def get_metadata(self, data: Dict) -> Dict:
        image_idx = data["image_idx"]
        # handle semantics
        if self.semantics is None:
            semantic_label = self.segmentations[image_idx]
            #TODO: We don't use masks for anything but im not sure if this is smart?
            mask = None
        else:
            filepath = self.semantics.filenames[image_idx]
            print(filepath)
            mask_indices = torch.tensor(
                [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
            ).view(1, 1, -1)
            semantic_label, mask = get_semantics_and_mask_tensors_from_path(
                filepath=filepath, mask_indices=mask_indices, scale_factor=self.scale_factor
            )
            if "mask" in data.keys():
                mask = mask & data["mask"]
        
        # Handle depth stuff
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