"Model that includes semantic support as well as depth supervision"
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from nerfstudio.model_components.losses import DepthLossType, depth_loss, depth_ranking_loss
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils import colormaps

from semantic_nerfacto.semantic_nerfacto import SemanticNerfactoModel

@dataclass
class SemanticDepthNerfactoModelConfig(NerfactoModelConfig):
    # Combine configurations of both models
    depth_loss_mult: float = 1e-3
    is_euclidean_depth: bool = False
    depth_sigma: float = 0.01
    should_decay_sigma: bool = False
    starting_depth_sigma: float = 0.2
    sigma_decay_rate: float = 0.99985
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    semantic_loss_weight: float = 1.0
    pass_semantic_gradients: bool = False

class SemanticDepthNerfactoModel(SemanticNerfactoModel):
    config: SemanticDepthNerfactoModelConfig

    def __init__(self, config: SemanticDepthNerfactoModelConfig, metadata: Dict, **kwargs) -> None:
        super().__init__(config, metadata, **kwargs)
        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)  # Get semantic outputs
        
        # If depth supervision is applicable, add depth-related outputs
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)  # Get semantic metrics
        
        # Add depth-related metrics if depth images are in the batch
        if self.training and "depth_image" in batch:
            depth_image = batch["depth_image"].to(self.device)
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

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)  # Get semantic losses
        
        # Add depth-related losses if depth images are in the batch
        if self.training and "depth_image" in batch:
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = self.config.depth_loss_mult * metrics_dict["depth_ranking"]

        return loss_dict

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma
        self.depth_sigma = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma