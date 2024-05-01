"""Model that combines all the functionality from bachelorproject"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.nn import Parameter


from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction

from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    distortion_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.utils import colormaps
from nerfstudio.model_components.losses import DepthLossType, depth_loss, depth_ranking_loss
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import profiler


@dataclass
class TetonNerfModelConfig(NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: TetonNerfModel)
    use_transient_embedding: bool = False
    """Whether to use transient embedding."""
    use_appearance_embedding: bool = True
    """Whether to use appearance embeddings. Throws error if not included"""
    
    # Semantics stuff
    use_semantics: bool = True
    """Whether to train semantic labels"""
    semantic_loss_weight: float = 1e-3
    """Multiplier for the semantic loss"""
    pass_semantic_gradients: bool = False
    
    # Depth stuff
    depth_loss_mult: float = 1e-3
    is_euclidean_depth: bool = True
    depth_sigma: float = 0.01
    should_decay_sigma: bool = False
    starting_depth_sigma: float = 0.2
    sigma_decay_rate: float = 0.99985
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    use_depth: bool = True
    """Whether to use depth supervision"""


class TetonNerfModel(NerfactoModel):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: TetonNerfModelConfig

    def __init__(self, config: TetonNerfModelConfig, metadata: Dict, **kwargs) -> None:
        self.semantics = metadata["semantics"]
        super().__init__(config=config, **kwargs)
        if self.semantics is not None:
            self.colormap = self.semantics.colors.clone().detach().to(self.device)
        
        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])
    
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        if self.semantics is not None:
            self.num_classes = len(self.semantics.classes)
        else: self.num_classes = 0
        
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
            num_semantic_classes=self.num_classes
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
            
        # If depth supervision is applicable, add depth-related outputs
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

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

        # Add depth-related metrics if using depth supervision
        if self.training and self.config.use_depth:
            assert "depth_image" in batch
            depth_image = batch["depth_image"].unsqueeze(1).to(self.device)
            if self.config.depth_loss_type in (DepthLossType.DS_NERF, DepthLossType.URF):
                sigma = self._get_sigma().to(self.device)
                depth_loss_value = 0
                for i in range(len(outputs["weights_list"])):
                    depth_loss_value += depth_loss(
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                        termination_depth=depth_image,
                        predicted_depth=outputs["depth"],
                        sigma=sigma,
                        directions_norm=outputs.get("directions_norm"),
                        is_euclidean=self.config.is_euclidean_depth,
                        depth_loss_type=self.config.depth_loss_type,
                    )
                metrics_dict["depth_loss"] = depth_loss_value / len(outputs["weights_list"])
            elif self.config.depth_loss_type == DepthLossType.SPARSENERF_RANKING:
                metrics_dict["depth_ranking"] = depth_ranking_loss(
                    outputs["expected_depth"], depth_image
                )


        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        # semantic loss
        semantics_pred = outputs["semantics"]
        semantics_gt = batch["semantics"][..., 0].long().to(self.device)    
        valid_mask = semantics_gt != 0 # 0 is the null class so no supervision on these
        #TODO: Consider whether adding masks on the semantics is a good idea
        
        if self.config.use_semantics:
            loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.cross_entropy_loss(
                semantics_pred[valid_mask], semantics_gt[valid_mask])
            
        # Add depth-related losses if using depth supervision
        if self.training and self.config.use_depth:
            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * metrics_dict["depth_ranking"]
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        
        # semantics
        if self.config.use_semantics:
            semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1).int()
            gt_semantics = batch["semantics"].squeeze().to(self.device).int()
            cmap = self.colormap.to(self.device)
            semantic_colormap = cmap[gt_semantics]
            gt_semantic_colormap = cmap[semantic_labels]
            image_tensor = torch.cat([semantic_colormap, gt_semantic_colormap], dim=1)
            images_dict["semantics_colormap"] = image_tensor

        # Appends ground truth depth to the depth image
        ground_truth_depth = batch["depth_image"].to(self.device).unsqueeze(2)
        # if not self.config.is_euclidean_depth:
            # ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

        ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
        predicted_depth_colormap = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=float(torch.min(ground_truth_depth).cpu()),
            far_plane=float(torch.max(ground_truth_depth).cpu()),
        )
        images_dict["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
        depth_mask = ground_truth_depth > 0
        metrics_dict["depth_mse"] = float(
            torch.nn.functional.mse_loss(outputs["depth"][depth_mask], ground_truth_depth[depth_mask]).cpu()
        )
            
        return metrics_dict, images_dict

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma
        self.depth_sigma = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma