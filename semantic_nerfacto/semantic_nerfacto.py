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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter


from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction

from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    distortion_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig



@dataclass
class SemanticNerfactoModelConfig(NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SemanticNerfactoModel)
    use_transient_embedding: bool = False
    """Whether to use transient embedding."""
    use_appearance_embedding: bool = True
    """Whether to use appearance embeddings. Throws error if not included"""
    use_semantics: bool = True
    semantic_loss_weight: float = 1.0
    pass_semantic_gradients: bool = False


class SemanticNerfactoModel(NerfactoModel):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: SemanticNerfactoModelConfig

    def __init__(self, config: SemanticNerfactoModelConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)
        
    
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0
        
        if self.semantics is not None:
            num_classes = len(self.semantics.classes)
        else: num_classes = 0
        
        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            implementation=self.config.implementation,
            # Add semantics to field
            use_semantics=self.config.use_semantics,
            pass_semantic_gradients=self.config.pass_semantic_gradients,
            num_semantic_classes=num_classes
        )

        # Add semantic renderer
        self.renderer_semantics = SemanticRenderer()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)  # Get nerfacto outputs
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        
        # Add semantics to output
        if self.config.use_semantics: 
            semantic_weights = weights
            if not self.config.pass_semantic_gradients:
                semantic_weights = semantic_weights.detach()
            outputs["semantics"] = self.renderer_semantics(
                field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights)
        
            # semantics colormaps
            semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
            outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        if self.config.use_semantics:
            loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.cross_entropy_loss(
                outputs["semantics"], batch["semantics"][..., 0].long().to(self.device))

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        
        if self.config.use_semantics:
            semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
            images_dict["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        # valid mask
        # TODO: Include these if using masks
        # images_dict["mask"] = batch["mask"].repeat(1, 1, 3).to(self.device)
        
        return metrics_dict, images_dict
