"""
Latent Residual Target Definition.

Defines the scale-free latent residual target ΔZ_structural(x,t)
that the ST-GNN is trained to predict.

Author: Urban Drainage AI Team
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LatentResidualConfig:
    """Configuration for latent residual computation."""
    # Standardization parameters
    eps: float = 1e-8
    
    # Clipping for numerical stability
    max_residual_magnitude: float = 5.0  # Clip extreme residuals
    
    # Smoothing for target stability
    temporal_smoothing: float = 0.1  # EMA smoothing factor


# ============================================================================
# LATENT RESIDUAL COMPUTATION
# ============================================================================

def compute_latent_z(
    raw_stress: np.ndarray,
    reference_mean: Optional[float] = None,
    reference_std: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Transform raw stress to latent Z-space (scale-free).
    
    Z = (raw_stress - μ) / σ
    
    Args:
        raw_stress: Raw stress values (any scale)
        reference_mean: Optional fixed reference mean (for inference)
        reference_std: Optional fixed reference std (for inference)
    
    Returns:
        (latent_z, mean_used, std_used)
    """
    # Compute or use provided statistics
    if reference_mean is None:
        reference_mean = float(np.nanmean(raw_stress))
    if reference_std is None:
        reference_std = float(np.nanstd(raw_stress))
        reference_std = max(reference_std, 1e-6)  # Prevent division by zero
    
    # Transform to Z-space
    latent_z = (raw_stress - reference_mean) / reference_std
    
    logger.debug(
        f"Latent Z transform: mean={reference_mean:.4f}, std={reference_std:.4f}, "
        f"Z range=[{np.nanmin(latent_z):.2f}, {np.nanmax(latent_z):.2f}]"
    )
    
    return latent_z, reference_mean, reference_std


def compute_structural_residual(
    observed_z: np.ndarray,
    physics_predicted_z: np.ndarray,
    config: Optional[LatentResidualConfig] = None,
) -> np.ndarray:
    """
    Compute the structural residual ΔZ that the DL model should learn.
    
    ΔZ_structural = Z_observed - Z_physics
    
    This represents the structural patterns NOT captured by the physics model.
    
    Args:
        observed_z: Latent Z from observations (ground truth in Z-space)
        physics_predicted_z: Latent Z predicted by Bayesian physics model
        config: Optional configuration
    
    Returns:
        delta_z: Structural residual in latent space
    """
    config = config or LatentResidualConfig()
    
    # Compute residual
    delta_z = observed_z - physics_predicted_z
    
    # Clip extreme values for stability
    delta_z = np.clip(
        delta_z,
        -config.max_residual_magnitude,
        config.max_residual_magnitude
    )
    
    logger.debug(
        f"Structural residual ΔZ: "
        f"range=[{np.nanmin(delta_z):.3f}, {np.nanmax(delta_z):.3f}], "
        f"mean={np.nanmean(delta_z):.4f}, std={np.nanstd(delta_z):.4f}"
    )
    
    return delta_z


def compute_rank_residual(
    observed_ranks: np.ndarray,
    physics_predicted_ranks: np.ndarray,
) -> np.ndarray:
    """
    Compute rank-based residual for rank-preserving loss.
    
    This captures whether the physics model correctly orders stress levels.
    
    Args:
        observed_ranks: Ranks of observed stress (in [0, 1])
        physics_predicted_ranks: Ranks of physics-predicted stress
    
    Returns:
        rank_residual: Difference in ranks
    """
    return observed_ranks - physics_predicted_ranks


# ============================================================================
# TARGET PREPARATION FOR TRAINING
# ============================================================================

@dataclass
class LatentTargets:
    """Container for training targets in latent space."""
    delta_z: np.ndarray  # Primary target: structural residual
    observed_z: np.ndarray  # Latent Z of observations
    physics_z: np.ndarray  # Latent Z of physics predictions
    reference_mean: float  # For inverse transform
    reference_std: float  # For inverse transform


def prepare_training_targets(
    observed_stress: np.ndarray,
    physics_predicted_stress: np.ndarray,
    config: Optional[LatentResidualConfig] = None,
) -> LatentTargets:
    """
    Prepare all training targets from raw stress data.
    
    This is the main entry point for target preparation.
    
    Args:
        observed_stress: Ground truth stress (can be any scale)
        physics_predicted_stress: Bayesian model predictions
        config: Optional configuration
    
    Returns:
        LatentTargets container with all targets
    """
    config = config or LatentResidualConfig()
    
    # Step 1: Transform observed stress to latent Z-space
    # Using observed statistics as reference
    observed_z, ref_mean, ref_std = compute_latent_z(observed_stress)
    
    # Step 2: Transform physics predictions using SAME statistics
    # This ensures both are in the same latent space
    physics_z, _, _ = compute_latent_z(
        physics_predicted_stress,
        reference_mean=ref_mean,
        reference_std=ref_std
    )
    
    # Step 3: Compute structural residual
    delta_z = compute_structural_residual(observed_z, physics_z, config)
    
    logger.info(
        f"Prepared targets: "
        f"ΔZ mean={np.nanmean(delta_z):.4f}, "
        f"reference scale: μ={ref_mean:.2f}, σ={ref_std:.2f}"
    )
    
    return LatentTargets(
        delta_z=delta_z.astype(np.float32),
        observed_z=observed_z.astype(np.float32),
        physics_z=physics_z.astype(np.float32),
        reference_mean=ref_mean,
        reference_std=ref_std,
    )


# ============================================================================
# INVERSE TRANSFORM (FOR INTERPRETABILITY ONLY)
# ============================================================================

def latent_to_raw(
    latent_z: np.ndarray,
    reference_mean: float,
    reference_std: float,
) -> np.ndarray:
    """
    Inverse transform from latent Z to raw scale.
    
    ⚠️ WARNING: This should ONLY be used for visualization/debugging.
    The DL model should NEVER output raw scale values.
    
    Args:
        latent_z: Values in latent Z-space
        reference_mean: Reference mean from original transform
        reference_std: Reference std from original transform
    
    Returns:
        raw_scale: Values in original scale
    """
    return latent_z * reference_std + reference_mean
