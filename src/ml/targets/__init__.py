"""Targets module for latent space training."""
from .latent_residual import (
    compute_latent_z,
    compute_structural_residual,
    prepare_training_targets,
    latent_to_raw,
    LatentResidualConfig,
    LatentTargets,
)

__all__ = [
    'compute_latent_z',
    'compute_structural_residual',
    'prepare_training_targets',
    'latent_to_raw',
    'LatentResidualConfig',
    'LatentTargets',
]
