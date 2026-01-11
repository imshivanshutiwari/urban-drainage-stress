"""
Uncertainty Guards for DL Component.

Ensures DL uncertainty is ADDITIVE ONLY and never reduces Bayesian uncertainty.

Author: Urban Drainage AI Team
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class UncertaintyGuardConfig:
    """Configuration for uncertainty guards."""
    # Minimum allowed uncertainty (prevents collapse)
    min_total_uncertainty: float = 0.01
    
    # Maximum reduction allowed (should be 0 for strict additive)
    max_uncertainty_reduction_fraction: float = 0.0
    
    # Abort threshold
    abort_on_violation: bool = True


# ============================================================================
# UNCERTAINTY GUARD RESULTS
# ============================================================================

@dataclass
class UncertaintyGuardResult:
    """Result of uncertainty guard validation."""
    passed: bool
    abort_required: bool
    violation_message: Optional[str]
    stats: dict


# ============================================================================
# GUARD FUNCTIONS
# ============================================================================

def validate_uncertainty_additive(
    bayesian_uncertainty: np.ndarray,
    dl_uncertainty: np.ndarray,
    total_uncertainty: np.ndarray,
    config: Optional[UncertaintyGuardConfig] = None
) -> UncertaintyGuardResult:
    """
    Validate that DL uncertainty is additive only.
    
    The total uncertainty MUST satisfy:
        total_uncertainty >= bayesian_uncertainty
    
    If DL reduces total uncertainty, the system is INVALID.
    
    Args:
        bayesian_uncertainty: Original Bayesian uncertainty
        dl_uncertainty: DL-predicted epistemic uncertainty proxy
        total_uncertainty: Final combined uncertainty
        config: Optional configuration
    
    Returns:
        UncertaintyGuardResult with pass/fail status
    """
    config = config or UncertaintyGuardConfig()
    
    # Compute statistics
    bayesian_mean = float(np.nanmean(bayesian_uncertainty))
    dl_mean = float(np.nanmean(dl_uncertainty))
    total_mean = float(np.nanmean(total_uncertainty))
    
    # Check 1: Total should not be less than Bayesian
    reduction = bayesian_uncertainty - total_uncertainty
    max_reduction = float(np.nanmax(reduction))
    reduction_fraction = max_reduction / (bayesian_mean + 1e-8)
    
    violation_message = None
    abort_required = False
    passed = True
    
    if reduction_fraction > config.max_uncertainty_reduction_fraction:
        passed = False
        violation_message = (
            f"🚨 UNCERTAINTY VIOLATION: DL reduced uncertainty!\n"
            f"  Max reduction: {max_reduction:.4f}\n"
            f"  Reduction fraction: {reduction_fraction:.2%}\n"
            f"  This violates the additive-only constraint.\n"
            f"  DL uncertainty must INCREASE total uncertainty, never decrease it!"
        )
        logger.error(violation_message)
        
        if config.abort_on_violation:
            abort_required = True
    
    # Check 2: Minimum uncertainty threshold
    if np.nanmin(total_uncertainty) < config.min_total_uncertainty:
        passed = False
        min_unc = float(np.nanmin(total_uncertainty))
        violation_message = (
            f"🚨 UNCERTAINTY COLLAPSE: Total uncertainty ({min_unc:.6f}) "
            f"below minimum ({config.min_total_uncertainty})!\n"
            f"  DL may have suppressed legitimate uncertainty."
        )
        logger.error(violation_message)
        
        if config.abort_on_violation:
            abort_required = True
    
    # Check 3: DL uncertainty should be non-negative
    if np.nanmin(dl_uncertainty) < 0:
        passed = False
        violation_message = (
            f"🚨 INVALID DL UNCERTAINTY: Negative values detected!\n"
            f"  Min DL uncertainty: {np.nanmin(dl_uncertainty):.4f}\n"
            f"  Uncertainty must be non-negative."
        )
        logger.error(violation_message)
        
        if config.abort_on_violation:
            abort_required = True
    
    if passed:
        logger.info(
            f"✅ Uncertainty guard PASSED. "
            f"Bayesian: {bayesian_mean:.4f}, DL: {dl_mean:.4f}, Total: {total_mean:.4f}"
        )
    
    return UncertaintyGuardResult(
        passed=passed,
        abort_required=abort_required,
        violation_message=violation_message,
        stats={
            'bayesian_mean': bayesian_mean,
            'dl_mean': dl_mean,
            'total_mean': total_mean,
            'max_reduction': max_reduction,
            'reduction_fraction': reduction_fraction,
        }
    )


def combine_uncertainties_safe(
    bayesian_uncertainty: np.ndarray,
    dl_uncertainty: np.ndarray,
    config: Optional[UncertaintyGuardConfig] = None
) -> np.ndarray:
    """
    Safely combine Bayesian and DL uncertainties.
    
    Uses additive combination that GUARANTEES:
        total >= bayesian
    
    Formula:
        total = sqrt(bayesian^2 + dl^2)
    
    This is the quadrature sum, which always increases total.
    """
    config = config or UncertaintyGuardConfig()
    
    # Ensure non-negative
    bayesian = np.maximum(bayesian_uncertainty, 0)
    dl = np.maximum(dl_uncertainty, 0)
    
    # Quadrature sum (always additive)
    total = np.sqrt(bayesian**2 + dl**2)
    
    # Enforce minimum
    total = np.maximum(total, config.min_total_uncertainty)
    
    return total


def assert_uncertainty_additive(
    bayesian_uncertainty: np.ndarray,
    dl_uncertainty: np.ndarray,
    total_uncertainty: np.ndarray,
    config: Optional[UncertaintyGuardConfig] = None
) -> None:
    """
    Assert that uncertainty combination is additive. Raises on violation.
    
    Use this as a runtime guard during inference.
    """
    result = validate_uncertainty_additive(
        bayesian_uncertainty, dl_uncertainty, total_uncertainty, config
    )
    
    if result.abort_required:
        raise RuntimeError(
            f"ABORT: Uncertainty guard violation!\n{result.violation_message}"
        )
