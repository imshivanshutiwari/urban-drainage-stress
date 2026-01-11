"""
Input Sanity Checks for Scale-Free DL.

This module enforces that the ST-GNN receives ONLY scale-free inputs.
Raw magnitudes (rainfall mm, absolute stress) are FORBIDDEN.

Author: Urban Drainage AI Team
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class InputSanityConfig:
    """Configuration for input sanity checks."""
    # Allowed input value ranges (scale-free)
    max_zscore_magnitude: float = 10.0  # |z| should not exceed this
    max_rank_value: float = 1.0  # Ranks should be in [0, 1]
    max_delta_magnitude: float = 5.0  # Temporal deltas bounded
    
    # Red flags for raw scale detection
    rainfall_mm_threshold: float = 50.0  # If max > this, likely raw mm
    stress_raw_threshold: float = 100.0  # If max > this, likely raw stress
    
    # Strict mode aborts on violation
    strict_mode: bool = True


# ============================================================================
# SANITY CHECK RESULTS
# ============================================================================

@dataclass
class SanityCheckResult:
    """Result of input sanity validation."""
    passed: bool
    violations: list
    warnings: list
    input_stats: Dict[str, Any]


# ============================================================================
# INPUT VALIDATORS
# ============================================================================

def check_is_zscore(data: np.ndarray, name: str, config: InputSanityConfig) -> tuple:
    """
    Validate that input appears to be z-score normalized.
    
    Z-scores should:
    - Have mean close to 0
    - Have std close to 1
    - Have magnitude bounded by config.max_zscore_magnitude
    """
    violations = []
    warnings = []
    
    mean = np.nanmean(data)
    std = np.nanstd(data)
    max_abs = np.nanmax(np.abs(data))
    
    # Check mean is close to 0
    if abs(mean) > 2.0:
        warnings.append(f"{name}: Mean ({mean:.3f}) far from 0, may not be z-score")
    
    # Check std is reasonable
    if std < 0.1 or std > 5.0:
        warnings.append(f"{name}: Std ({std:.3f}) unusual for z-scores")
    
    # Check magnitude bound
    if max_abs > config.max_zscore_magnitude:
        violations.append(
            f"{name}: Max magnitude ({max_abs:.2f}) exceeds limit "
            f"({config.max_zscore_magnitude}). Possibly raw scale!"
        )
    
    return violations, warnings


def check_is_rank(data: np.ndarray, name: str, config: InputSanityConfig) -> tuple:
    """
    Validate that input appears to be rank/percentile normalized.
    
    Ranks should:
    - Be in [0, 1] range
    - Have roughly uniform distribution
    """
    violations = []
    warnings = []
    
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
    if min_val < 0:
        violations.append(f"{name}: Contains negative values ({min_val:.3f}), not valid rank")
    
    if max_val > config.max_rank_value + 0.01:  # Small tolerance
        violations.append(
            f"{name}: Max value ({max_val:.3f}) exceeds 1.0, not valid rank. "
            f"Possibly raw counts!"
        )
    
    return violations, warnings


def check_not_raw_rainfall(data: np.ndarray, name: str, config: InputSanityConfig) -> tuple:
    """
    Detect if input looks like raw rainfall in mm.
    
    Raw rainfall typically:
    - Has large positive values (>50mm for heavy rain)
    - Is always non-negative
    - Has heavy right tail
    """
    violations = []
    warnings = []
    
    max_val = np.nanmax(data)
    min_val = np.nanmin(data)
    
    # Raw mm is always positive
    if min_val >= 0 and max_val > config.rainfall_mm_threshold:
        violations.append(
            f"🚨 {name}: Max value ({max_val:.1f}) suggests RAW RAINFALL (mm). "
            f"FORBIDDEN! Convert to z-scores or ranks first."
        )
    
    return violations, warnings


def check_not_raw_stress(data: np.ndarray, name: str, config: InputSanityConfig) -> tuple:
    """
    Detect if input looks like raw stress values.
    
    Raw stress typically:
    - Has large magnitude
    - Is city-specific
    """
    violations = []
    warnings = []
    
    max_val = np.nanmax(np.abs(data))
    
    if max_val > config.stress_raw_threshold:
        violations.append(
            f"🚨 {name}: Max magnitude ({max_val:.1f}) suggests RAW STRESS. "
            f"FORBIDDEN! Use latent Z or ranks."
        )
    
    return violations, warnings


def check_temporal_delta(data: np.ndarray, name: str, config: InputSanityConfig) -> tuple:
    """
    Validate temporal delta inputs.
    
    Deltas should:
    - Be relatively small changes
    - Center around 0
    """
    violations = []
    warnings = []
    
    max_abs = np.nanmax(np.abs(data))
    
    if max_abs > config.max_delta_magnitude:
        warnings.append(
            f"{name}: Large delta magnitude ({max_abs:.2f}) detected. "
            f"Verify normalization."
        )
    
    return violations, warnings


# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_model_inputs(
    inputs: Dict[str, np.ndarray],
    input_types: Dict[str, str],
    config: Optional[InputSanityConfig] = None
) -> SanityCheckResult:
    """
    Validate all model inputs for scale-free compliance.
    
    Args:
        inputs: Dictionary of input arrays {name: array}
        input_types: Dictionary specifying expected type for each input
                     Valid types: 'zscore', 'rank', 'delta', 'relational'
        config: Optional configuration
    
    Returns:
        SanityCheckResult with pass/fail status and details
    
    Example:
        >>> inputs = {'rainfall': rainfall_z, 'complaints': complaint_ranks}
        >>> types = {'rainfall': 'zscore', 'complaints': 'rank'}
        >>> result = validate_model_inputs(inputs, types)
        >>> if not result.passed:
        ...     raise ValueError(f"Input validation failed: {result.violations}")
    """
    config = config or InputSanityConfig()
    
    all_violations = []
    all_warnings = []
    stats = {}
    
    for name, data in inputs.items():
        expected_type = input_types.get(name, 'unknown')
        
        # Collect stats
        stats[name] = {
            'shape': data.shape,
            'mean': float(np.nanmean(data)),
            'std': float(np.nanstd(data)),
            'min': float(np.nanmin(data)),
            'max': float(np.nanmax(data)),
            'expected_type': expected_type,
        }
        
        # Always check for raw scale leakage
        violations, warnings = check_not_raw_rainfall(data, name, config)
        all_violations.extend(violations)
        all_warnings.extend(warnings)
        
        violations, warnings = check_not_raw_stress(data, name, config)
        all_violations.extend(violations)
        all_warnings.extend(warnings)
        
        # Type-specific checks
        if expected_type == 'zscore':
            violations, warnings = check_is_zscore(data, name, config)
            all_violations.extend(violations)
            all_warnings.extend(warnings)
            
        elif expected_type == 'rank':
            violations, warnings = check_is_rank(data, name, config)
            all_violations.extend(violations)
            all_warnings.extend(warnings)
            
        elif expected_type == 'delta':
            violations, warnings = check_temporal_delta(data, name, config)
            all_violations.extend(violations)
            all_warnings.extend(warnings)
            
        elif expected_type == 'relational':
            # Relational features (e.g., slope ratios) just need finite values
            if not np.all(np.isfinite(data)):
                all_warnings.append(f"{name}: Contains non-finite values")
    
    # Log results
    passed = len(all_violations) == 0
    
    if passed:
        logger.info("✅ Input sanity check PASSED. All inputs are scale-free.")
    else:
        logger.error("🚨 Input sanity check FAILED!")
        for v in all_violations:
            logger.error(f"  VIOLATION: {v}")
    
    for w in all_warnings:
        logger.warning(f"  WARNING: {w}")
    
    return SanityCheckResult(
        passed=passed,
        violations=all_violations,
        warnings=all_warnings,
        input_stats=stats,
    )


def assert_scale_free(
    inputs: Dict[str, np.ndarray],
    input_types: Dict[str, str],
    config: Optional[InputSanityConfig] = None
) -> None:
    """
    Assert that inputs are scale-free. Raises ValueError on violation.
    
    Use this as a guard before model forward pass.
    """
    result = validate_model_inputs(inputs, input_types, config)
    
    if not result.passed:
        raise ValueError(
            f"SCALE VIOLATION: Model received non-scale-free inputs!\n"
            f"Violations:\n" + "\n".join(f"  - {v}" for v in result.violations)
        )
