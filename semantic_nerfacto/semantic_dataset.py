# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semantic dataset.
"""

import json
from pathlib import Path
from typing import Dict, Union

# Import for the semantic segmentator detectron2
from semantic_nerfacto.detectron import SemanticSegmentor

import torch
import numpy as np
from PIL import Image
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.rich_utils import CONSOLE


class SemanticDataset(InputDataset):
    """Dataset that returns images and semantics and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "semantics"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        self.semantics = self.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)
    
    # if there are no semantic images than we want to generate them all with detectron2

        if len(dataparser_outputs.image_filenames) > 0 and (
            "semantics" not in dataparser_outputs.metadata.keys()
            or dataparser_outputs.metadata["semantics"] is None
        ):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            CONSOLE.print("[bold yellow] No semantics data found! Generating pseudosemantics")

            cache = dataparser_outputs.image_filenames[0].parent / "semantics.npy"
            # Note: this should probably be saved to disk as images, and then loaded with the dataparser.
            #  That will allow multi-gpu training.
            if cache.exists():
                CONSOLE.print("[bold yellow] Loading semantics data from cache!")
                # load all the depths
                self.depths = np.load(cache)
                self.depths = torch.from_numpy(self.depths).to(device)
            else:
                semantics_tensors = []
                transforms = self._find_transform(dataparser_outputs.image_filenames[0])
                data = dataparser_outputs.image_filenames[0].parent
                if transforms is not None:
                    meta = json.load(open(transforms, "r"))
                    frames = meta["frames"]
                    filenames = [data / frames[j]["file_path"].split("/")[-1] for j in range(len(frames))]
                else:
                    meta = None
                    frames = None
                    filenames = dataparser_outputs.image_filenames

                # TODO: Use detectron2 to generate semantics
                # Inference with a panoptic segmentation model
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
                predictor = DefaultPredictor(cfg)
                # Get a list of thing/stuf classes
                semantics_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
                semantics_metadata.thing_classes
                semantics_metadata.stuff_classes

                for i in track(range(len(filenames)), description="Generating semantic images"):
                    image_filename = filenames[i]
                    # pil_image = Image.open(image_filename)
                    image = cv2.imread(image_filename)
                    
                    panoptic_seg, segments_info = predictor(image)["panoptic_seg"]

                    semantics_tensors.append(panoptic_seg)

                self.semantics = torch.stack(semantics_tensors)
                np.save(cache, self.semantics.cpu().numpy())
                
            dataparser_outputs.metadata["semantics"] = None
            dataparser_outputs.metadata["semantics_unit_scale_factor"] = 1.0
            self.metadata["semantics"] = None
            self.metadata["semantics_unit_scale_factor"] = 1.0

        self.semantic_filenames = self.metadata["semantics"]
        self.semantics_unit_scale_factor = self.metadata["semantics_unit_scale_factor"]
    
    
    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label, mask = get_semantics_and_mask_tensors_from_path(
            filepath=filepath, mask_indices=self.mask_indices, scale_factor=self.scale_factor
        )
        if "mask" in data.keys():
            mask = mask & data["mask"]
        return {"mask": mask, "semantics": semantic_label}
