"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from teton_nerf.semantic_nerfacto import SemanticNerfactoModelConfig
from teton_nerf.semantic_dataparser import SemanticDataParserConfig
from teton_nerf.semantic_nerfacto_datamanager import TetonNerfDatamanagerConfig
from teton_nerf.semantic_nerfacto_pipeline import SemanticNerfactoPipelineConfig
from teton_nerf.semantic_depth_nerfacto_pipeline import SemanticDepthNerfactoPipelineConfig
from teton_nerf.semantic_depth_nerfacto import SemanticDepthNerfactoModelConfig
from teton_nerf.teton_nerf import TetonNerfModelConfig
from teton_nerf.teton_nerf_pipeline import TetonNerfPipelineConfig

semantic_dataparser = DataParserSpecification(config=SemanticDataParserConfig())

semantic_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="semantic-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=SemanticNerfactoPipelineConfig(
            datamanager=TetonNerfDatamanagerConfig(
                dataparser=SemanticDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=SemanticNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of semantic Nerf that adds semantic segmentation to the Nerfacto model.",
)

semantic_depth_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="semantic-depth-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=SemanticDepthNerfactoPipelineConfig(
            datamanager=TetonNerfDatamanagerConfig(
                dataparser=SemanticDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=SemanticDepthNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of semantic Nerf that adds semantic segmentation to the Nerfacto model and also includes depth supervision.",
)

teton_nerf = MethodSpecification(
    config=TrainerConfig(
        method_name="teton-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=TetonNerfPipelineConfig(
            datamanager=TetonNerfDatamanagerConfig(
                dataparser=SemanticDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=TetonNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of semantic Nerf that adds semantic segmentation to the Nerfacto model and also includes depth supervision.",
)