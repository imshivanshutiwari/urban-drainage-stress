"""
DL Role Audit Module.

Automated audit that verifies the DL component adheres to its role.

CHECKS:
1. DL never sees raw scale
2. DL outputs bounded latent corrections
3. DL does not affect thresholds
4. DL does not dominate decisions

If ANY violation occurs → system refuses to run.

Author: Urban Drainage AI Team
Date: January 2026
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# AUDIT CONFIGURATION
# ============================================================================

@dataclass
class AuditConfig:
    """Configuration for DL role audit."""
    # Bounds for latent corrections
    max_allowed_delta_z: float = 5.0
    
    # Scale detection thresholds
    raw_scale_threshold: float = 100.0
    
    # Decision influence limits
    max_decision_change_fraction: float = 0.1  # DL can flip at most 10% of decisions
    
    # Dominance check
    max_dl_contribution_fraction: float = 0.5  # DL should not dominate
    
    # Abort on violation
    abort_on_violation: bool = True


# ============================================================================
# AUDIT RESULT
# ============================================================================

@dataclass
class AuditResult:
    """Result of DL role audit."""
    passed: bool
    checks: Dict[str, bool]
    violations: List[str]
    warnings: List[str]
    details: Dict[str, Any]


# ============================================================================
# INDIVIDUAL AUDIT CHECKS
# ============================================================================

def check_no_raw_scale_input(
    model_inputs: Dict[str, np.ndarray],
    config: AuditConfig,
) -> tuple:
    """
    Check 1: DL never sees raw scale.
    
    Verify that all model inputs are scale-free.
    """
    violations = []
    warnings = []
    details = {}
    
    for name, data in model_inputs.items():
        max_val = np.nanmax(np.abs(data))
        details[f"{name}_max_abs"] = float(max_val)
        
        if max_val > config.raw_scale_threshold:
            violations.append(
                f"RAW SCALE DETECTED in '{name}': max_abs={max_val:.1f}. "
                f"DL is receiving non-scale-free inputs!"
            )
    
    return len(violations) == 0, violations, warnings, details


def check_bounded_latent_corrections(
    delta_z: np.ndarray,
    config: AuditConfig,
) -> tuple:
    """
    Check 2: DL outputs bounded latent corrections.
    
    ΔZ should be small, bounded corrections, not raw magnitudes.
    """
    violations = []
    warnings = []
    details = {}
    
    max_delta = float(np.nanmax(np.abs(delta_z)))
    mean_delta = float(np.nanmean(delta_z))
    std_delta = float(np.nanstd(delta_z))
    
    details['max_delta_z'] = max_delta
    details['mean_delta_z'] = mean_delta
    details['std_delta_z'] = std_delta
    
    if max_delta > config.max_allowed_delta_z:
        violations.append(
            f"UNBOUNDED ΔZ: max={max_delta:.2f} exceeds limit "
            f"({config.max_allowed_delta_z}). DL may be learning raw scale!"
        )
    
    if std_delta > 3.0:
        warnings.append(
            f"High ΔZ variance: std={std_delta:.2f}. "
            f"Check if DL is over-correcting."
        )
    
    return len(violations) == 0, violations, warnings, details


def check_no_threshold_influence(
    physics_decisions: np.ndarray,
    hybrid_decisions: np.ndarray,
    config: AuditConfig,
) -> tuple:
    """
    Check 3: DL does not affect decision thresholds.
    
    The DL component should not significantly change decisions.
    """
    violations = []
    warnings = []
    details = {}
    
    # Count decision changes
    n_total = physics_decisions.size
    n_changed = np.sum(physics_decisions != hybrid_decisions)
    change_fraction = n_changed / n_total
    
    details['n_decisions_changed'] = int(n_changed)
    details['change_fraction'] = float(change_fraction)
    
    if change_fraction > config.max_decision_change_fraction:
        violations.append(
            f"DECISION INFLUENCE: DL changed {change_fraction:.1%} of decisions. "
            f"Limit is {config.max_decision_change_fraction:.1%}. "
            f"DL may be influencing thresholds!"
        )
    
    return len(violations) == 0, violations, warnings, details


def check_no_dl_dominance(
    physics_output: np.ndarray,
    dl_correction: np.ndarray,
    config: AuditConfig,
) -> tuple:
    """
    Check 4: DL does not dominate.
    
    The DL contribution should not be larger than the physics contribution.
    """
    violations = []
    warnings = []
    details = {}
    
    physics_magnitude = float(np.nanmean(np.abs(physics_output)))
    dl_magnitude = float(np.nanmean(np.abs(dl_correction)))
    
    details['physics_magnitude'] = physics_magnitude
    details['dl_magnitude'] = dl_magnitude
    
    if physics_magnitude > 1e-8:
        dl_fraction = dl_magnitude / physics_magnitude
        details['dl_fraction'] = float(dl_fraction)
        
        if dl_fraction > config.max_dl_contribution_fraction:
            violations.append(
                f"DL DOMINANCE: DL contribution ({dl_magnitude:.3f}) is "
                f"{dl_fraction:.1%} of physics ({physics_magnitude:.3f}). "
                f"DL should support, not replace physics!"
            )
    
    return len(violations) == 0, violations, warnings, details


# ============================================================================
# FULL AUDIT
# ============================================================================

def run_dl_role_audit(
    model_inputs: Dict[str, np.ndarray],
    delta_z: np.ndarray,
    physics_z: np.ndarray,
    physics_decisions: Optional[np.ndarray] = None,
    hybrid_decisions: Optional[np.ndarray] = None,
    config: Optional[AuditConfig] = None,
) -> AuditResult:
    """
    Run full DL role audit.
    
    Args:
        model_inputs: Dictionary of inputs fed to the DL model
        delta_z: DL output (latent corrections)
        physics_z: Physics model output in latent space
        physics_decisions: Decisions from physics-only (for threshold check)
        hybrid_decisions: Decisions from hybrid model (for threshold check)
        config: Audit configuration
    
    Returns:
        AuditResult with pass/fail and details
    """
    config = config or AuditConfig()
    
    all_violations = []
    all_warnings = []
    all_details = {}
    checks = {}
    
    logger.info("=" * 60)
    logger.info("RUNNING DL ROLE AUDIT")
    logger.info("=" * 60)
    
    # Check 1: No raw scale inputs
    passed, violations, warnings, details = check_no_raw_scale_input(
        model_inputs, config
    )
    checks['no_raw_scale_input'] = passed
    all_violations.extend(violations)
    all_warnings.extend(warnings)
    all_details.update(details)
    logger.info(f"Check 1 (No Raw Scale Input): {'✓' if passed else '✗'}")
    
    # Check 2: Bounded latent corrections
    passed, violations, warnings, details = check_bounded_latent_corrections(
        delta_z, config
    )
    checks['bounded_latent_corrections'] = passed
    all_violations.extend(violations)
    all_warnings.extend(warnings)
    all_details.update(details)
    logger.info(f"Check 2 (Bounded ΔZ): {'✓' if passed else '✗'}")
    
    # Check 3: No threshold influence (if decisions provided)
    if physics_decisions is not None and hybrid_decisions is not None:
        passed, violations, warnings, details = check_no_threshold_influence(
            physics_decisions, hybrid_decisions, config
        )
        checks['no_threshold_influence'] = passed
        all_violations.extend(violations)
        all_warnings.extend(warnings)
        all_details.update(details)
        logger.info(f"Check 3 (No Threshold Influence): {'✓' if passed else '✗'}")
    else:
        checks['no_threshold_influence'] = None  # Not tested
        logger.info("Check 3 (No Threshold Influence): SKIPPED (no decisions)")
    
    # Check 4: No DL dominance
    passed, violations, warnings, details = check_no_dl_dominance(
        physics_z, delta_z, config
    )
    checks['no_dl_dominance'] = passed
    all_violations.extend(violations)
    all_warnings.extend(warnings)
    all_details.update(details)
    logger.info(f"Check 4 (No DL Dominance): {'✓' if passed else '✗'}")
    
    # Overall result
    overall_passed = len(all_violations) == 0
    
    logger.info("=" * 60)
    if overall_passed:
        logger.info("✅ DL ROLE AUDIT PASSED - All checks OK")
    else:
        logger.error("🚨 DL ROLE AUDIT FAILED")
        for v in all_violations:
            logger.error(f"  VIOLATION: {v}")
    logger.info("=" * 60)
    
    result = AuditResult(
        passed=overall_passed,
        checks=checks,
        violations=all_violations,
        warnings=all_warnings,
        details=all_details,
    )
    
    # Abort if configured and violations found
    if not overall_passed and config.abort_on_violation:
        raise RuntimeError(
            f"DL ROLE AUDIT FAILED - SYSTEM REFUSES TO RUN\n"
            f"Violations:\n" + "\n".join(f"  - {v}" for v in all_violations)
        )
    
    return result


def save_audit_report(result: AuditResult, path: Path) -> None:
    """Save audit report to JSON."""
    report = {
        'passed': result.passed,
        'checks': result.checks,
        'violations': result.violations,
        'warnings': result.warnings,
        'details': result.details,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Audit report saved to: {path}")
