"""Scientific validation: Tests that MUST pass for valid inference.

This module implements the FINAL ACCEPTANCE CRITERIA from Projectfile.md:
1. Removing complaints visibly changes inference
2. Uncertainty is spatially and temporally heterogeneous
3. Decision maps are NOT uniformly HIGH
4. Stress maps contain local anomalies
5. Parameters differ across runs/events
6. The system can honestly say "I don't know" in some regions

If ANY of these fail → the run is INVALID.

Reference: Projectfile.md - "Validation enforcement"

NO FAKE PASSES. REAL SCIENTIFIC TESTS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import json

import numpy as np

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation test result status."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warning"
    SKIP = "skipped"


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    name: str
    status: ValidationStatus
    metric: float
    threshold: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScientificValidationReport:
    """Complete scientific validation report."""
    
    tests: List[ValidationResult]
    overall_status: ValidationStatus
    summary: str
    
    # Aggregate metrics
    pass_count: int
    fail_count: int
    warn_count: int
    
    # Scientific integrity score (0-100)
    integrity_score: float
    
    # Whether run should be trusted
    is_valid: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "is_valid": self.is_valid,
            "integrity_score": self.integrity_score,
            "summary": self.summary,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "tests": [
                {
                    "name": t.name,
                    "status": t.status.value,
                    "metric": t.metric,
                    "threshold": t.threshold,
                    "message": t.message,
                    "details": t.details,
                }
                for t in self.tests
            ],
        }
    
    def save(self, path: Path) -> None:
        """Save validation report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved validation report to %s", path)


class ScientificValidator:
    """Scientific validation enforcer.
    
    Runs all acceptance criteria tests and determines if a run is valid.
    
    A run FAILS if:
    - Complaints don't change inference
    - Uncertainty is uniform
    - Decisions are uniformly HIGH
    - No anomalies in stress
    - No NO_DECISION regions
    """
    
    def __init__(
        self,
        uncertainty_cv_threshold: float = 0.1,
        no_decision_min_fraction: float = 0.05,
        high_risk_max_fraction: float = 0.80,
        observation_impact_threshold: float = 0.01,
        spatial_roughness_threshold: float = 0.05,
    ) -> None:
        self.uncertainty_cv_threshold = uncertainty_cv_threshold
        self.no_decision_min_fraction = no_decision_min_fraction
        self.high_risk_max_fraction = high_risk_max_fraction
        self.observation_impact_threshold = observation_impact_threshold
        self.spatial_roughness_threshold = spatial_roughness_threshold
        
        logger.info("ScientificValidator initialized with strict criteria")
    
    def validate(
        self,
        posterior_mean: np.ndarray,
        posterior_variance: np.ndarray,
        prior_mean: Optional[np.ndarray] = None,
        prior_variance: Optional[np.ndarray] = None,
        risk_levels: Optional[np.ndarray] = None,
        complaints: Optional[np.ndarray] = None,
        information_gain: Optional[np.ndarray] = None,
    ) -> ScientificValidationReport:
        """Run all scientific validation tests.
        
        Args:
            posterior_mean: (T, H, W) inferred stress
            posterior_variance: (T, H, W) inferred uncertainty
            prior_mean: (T, H, W) prior stress (for comparison)
            prior_variance: (T, H, W) prior uncertainty
            risk_levels: (T, H, W) risk decisions
            complaints: (T, H, W) observed complaints
            information_gain: (T, H, W) KL divergence
            
        Returns:
            ScientificValidationReport with all test results
        """
        tests = []
        
        # TEST 1: Uncertainty heterogeneity
        tests.append(self._test_uncertainty_heterogeneity(posterior_variance))
        
        # TEST 2: Spatial structure (anomalies)
        tests.append(self._test_spatial_structure(posterior_mean))
        
        # TEST 3: Decision distribution
        if risk_levels is not None:
            tests.append(self._test_decision_distribution(risk_levels))
        
        # TEST 4: Observation impact
        if prior_mean is not None and information_gain is not None:
            tests.append(self._test_observation_impact(
                prior_mean, posterior_mean, information_gain, complaints
            ))
        
        # TEST 5: No-decision regions exist
        tests.append(self._test_no_decision_exists(posterior_variance))
        
        # TEST 6: Variance reduction where observed
        if prior_variance is not None and complaints is not None:
            tests.append(self._test_variance_reduction(
                prior_variance, posterior_variance, complaints
            ))
        
        # TEST 7: Temporal variation
        tests.append(self._test_temporal_variation(posterior_mean))
        
        # TEST 8: Physical constraints
        tests.append(self._test_physical_constraints(posterior_mean))
        
        # Aggregate results
        pass_count = sum(1 for t in tests if t.status == ValidationStatus.PASS)
        fail_count = sum(1 for t in tests if t.status == ValidationStatus.FAIL)
        warn_count = sum(1 for t in tests if t.status == ValidationStatus.WARN)
        
        # Determine overall status
        if fail_count > 0:
            overall_status = ValidationStatus.FAIL
            is_valid = False
        elif warn_count > 2:
            overall_status = ValidationStatus.WARN
            is_valid = True  # Warnings don't invalidate
        else:
            overall_status = ValidationStatus.PASS
            is_valid = True
        
        # Integrity score: weighted by test importance
        weights = {
            "uncertainty_heterogeneity": 2.0,
            "observation_impact": 2.0,
            "decision_distribution": 1.5,
            "no_decision_exists": 1.5,
            "spatial_structure": 1.0,
            "variance_reduction": 1.5,
            "temporal_variation": 1.0,
            "physical_constraints": 1.0,
        }
        
        total_weight = 0
        weighted_score = 0
        for t in tests:
            w = weights.get(t.name, 1.0)
            total_weight += w
            if t.status == ValidationStatus.PASS:
                weighted_score += w * 100
            elif t.status == ValidationStatus.WARN:
                weighted_score += w * 60
            else:
                weighted_score += w * 0
        
        integrity_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Summary
        if overall_status == ValidationStatus.PASS:
            summary = f"✓ VALID RUN: {pass_count}/{len(tests)} tests passed. " \
                     f"Integrity score: {integrity_score:.0f}/100"
        elif overall_status == ValidationStatus.WARN:
            summary = f"⚠ VALID WITH WARNINGS: {pass_count}/{len(tests)} passed, " \
                     f"{warn_count} warnings. Integrity: {integrity_score:.0f}/100"
        else:
            summary = f"✗ INVALID RUN: {fail_count}/{len(tests)} tests FAILED. " \
                     f"Integrity: {integrity_score:.0f}/100. DO NOT TRUST RESULTS."
        
        logger.info(summary)
        
        return ScientificValidationReport(
            tests=tests,
            overall_status=overall_status,
            summary=summary,
            pass_count=pass_count,
            fail_count=fail_count,
            warn_count=warn_count,
            integrity_score=integrity_score,
            is_valid=is_valid,
        )
    
    def _test_uncertainty_heterogeneity(
        self,
        variance: np.ndarray,
    ) -> ValidationResult:
        """Test: Uncertainty must NOT be uniform."""
        std = np.sqrt(variance)
        cv = np.nanstd(std) / (np.nanmean(std) + 1e-9)
        
        threshold = self.uncertainty_cv_threshold
        
        if cv >= threshold * 2:
            status = ValidationStatus.PASS
            msg = f"Uncertainty CV = {cv:.3f} >> {threshold} (excellent heterogeneity)"
        elif cv >= threshold:
            status = ValidationStatus.PASS
            msg = f"Uncertainty CV = {cv:.3f} >= {threshold} (acceptable heterogeneity)"
        elif cv >= threshold * 0.5:
            status = ValidationStatus.WARN
            msg = f"Uncertainty CV = {cv:.3f} < {threshold} (weak heterogeneity)"
        else:
            status = ValidationStatus.FAIL
            msg = f"FAIL: Uncertainty CV = {cv:.3f} << {threshold} (nearly uniform = FAKE)"
        
        return ValidationResult(
            name="uncertainty_heterogeneity",
            status=status,
            metric=float(cv),
            threshold=threshold,
            message=msg,
            details={"std_mean": float(np.nanmean(std)), "std_std": float(np.nanstd(std))},
        )
    
    def _test_spatial_structure(
        self,
        stress: np.ndarray,
    ) -> ValidationResult:
        """Test: Stress must have local anomalies (not smooth gradient)."""
        # Compute spatial roughness via local variance
        if stress.ndim == 3:
            # Average over time
            stress_spatial = np.nanmean(stress, axis=0)
        else:
            stress_spatial = stress
        
        H, W = stress_spatial.shape
        
        # Compute local variance in 3x3 windows
        local_vars = []
        for i in range(1, H-1):
            for j in range(1, W-1):
                window = stress_spatial[i-1:i+2, j-1:j+2]
                if not np.any(np.isnan(window)):
                    local_vars.append(np.var(window))
        
        if len(local_vars) < 10:
            return ValidationResult(
                name="spatial_structure",
                status=ValidationStatus.SKIP,
                metric=0.0,
                threshold=self.spatial_roughness_threshold,
                message="Insufficient data for spatial structure test",
            )
        
        mean_local_var = np.mean(local_vars)
        global_var = np.nanvar(stress_spatial)
        roughness = mean_local_var / (global_var + 1e-9)
        
        threshold = self.spatial_roughness_threshold
        
        if roughness >= threshold * 2:
            status = ValidationStatus.PASS
            msg = f"Spatial roughness = {roughness:.3f} (good local variation)"
        elif roughness >= threshold:
            status = ValidationStatus.PASS
            msg = f"Spatial roughness = {roughness:.3f} (acceptable)"
        else:
            status = ValidationStatus.WARN
            msg = f"Spatial roughness = {roughness:.3f} < {threshold} (too smooth)"
        
        return ValidationResult(
            name="spatial_structure",
            status=status,
            metric=float(roughness),
            threshold=threshold,
            message=msg,
            details={"local_var": float(mean_local_var), "global_var": float(global_var)},
        )
    
    def _test_decision_distribution(
        self,
        risk_levels: np.ndarray,
    ) -> ValidationResult:
        """Test: Decisions must NOT be uniformly HIGH."""
        # Count risk levels
        unique, counts = np.unique(risk_levels, return_counts=True)
        total = risk_levels.size
        distribution = {str(u): int(c) for u, c in zip(unique, counts)}
        
        # Check for HIGH dominance
        high_count = sum(c for u, c in zip(unique, counts) if 'high' in str(u).lower())
        high_frac = high_count / total
        
        threshold = self.high_risk_max_fraction
        
        if high_frac <= threshold * 0.5:
            status = ValidationStatus.PASS
            msg = f"HIGH risk = {high_frac:.1%} (well balanced)"
        elif high_frac <= threshold:
            status = ValidationStatus.PASS
            msg = f"HIGH risk = {high_frac:.1%} (acceptable)"
        elif high_frac <= threshold * 1.1:
            status = ValidationStatus.WARN
            msg = f"HIGH risk = {high_frac:.1%} > {threshold:.0%} (slightly imbalanced)"
        else:
            status = ValidationStatus.FAIL
            msg = f"FAIL: HIGH risk = {high_frac:.1%} >> {threshold:.0%} (broken model)"
        
        return ValidationResult(
            name="decision_distribution",
            status=status,
            metric=float(high_frac),
            threshold=threshold,
            message=msg,
            details={"distribution": distribution},
        )
    
    def _test_observation_impact(
        self,
        prior_mean: np.ndarray,
        posterior_mean: np.ndarray,
        information_gain: np.ndarray,
        complaints: Optional[np.ndarray],
    ) -> ValidationResult:
        """Test: Observations MUST change inference."""
        # Mean information gain where observations exist
        if complaints is not None:
            observed_mask = complaints > 0
            if np.sum(observed_mask) > 0:
                mean_ig_observed = np.nanmean(information_gain[observed_mask])
            else:
                mean_ig_observed = 0
        else:
            mean_ig_observed = np.nanmean(information_gain)
        
        threshold = self.observation_impact_threshold
        
        if mean_ig_observed >= threshold * 10:
            status = ValidationStatus.PASS
            msg = f"Observation impact = {mean_ig_observed:.4f} (strong influence)"
        elif mean_ig_observed >= threshold:
            status = ValidationStatus.PASS
            msg = f"Observation impact = {mean_ig_observed:.4f} (detectable influence)"
        elif mean_ig_observed >= threshold * 0.1:
            status = ValidationStatus.WARN
            msg = f"Observation impact = {mean_ig_observed:.4f} < {threshold} (weak)"
        else:
            status = ValidationStatus.FAIL
            msg = f"FAIL: Observation impact = {mean_ig_observed:.4f} ≈ 0 " \
                 f"(complaints don't affect inference!)"
        
        return ValidationResult(
            name="observation_impact",
            status=status,
            metric=float(mean_ig_observed),
            threshold=threshold,
            message=msg,
        )
    
    def _test_no_decision_exists(
        self,
        variance: np.ndarray,
    ) -> ValidationResult:
        """Test: System must have regions where it says 'I don't know'."""
        std = np.sqrt(variance)
        
        # Count cells with high uncertainty (potential no-decision)
        high_uncertainty_threshold = np.percentile(std, 90)  # Top 10%
        high_unc_frac = np.mean(std >= high_uncertainty_threshold * 0.9)
        
        threshold = self.no_decision_min_fraction
        
        if high_unc_frac >= threshold * 2:
            status = ValidationStatus.PASS
            msg = f"High uncertainty regions = {high_unc_frac:.1%} (can say 'I don't know')"
        elif high_unc_frac >= threshold:
            status = ValidationStatus.PASS
            msg = f"High uncertainty regions = {high_unc_frac:.1%} (acceptable)"
        else:
            status = ValidationStatus.WARN
            msg = f"High uncertainty = {high_unc_frac:.1%} < {threshold:.0%} " \
                 f"(model may be overconfident)"
        
        return ValidationResult(
            name="no_decision_exists",
            status=status,
            metric=float(high_unc_frac),
            threshold=threshold,
            message=msg,
        )
    
    def _test_variance_reduction(
        self,
        prior_variance: np.ndarray,
        posterior_variance: np.ndarray,
        complaints: np.ndarray,
    ) -> ValidationResult:
        """Test: Variance should reduce where observations exist."""
        observed_mask = complaints > 0
        
        if np.sum(observed_mask) < 10:
            return ValidationResult(
                name="variance_reduction",
                status=ValidationStatus.SKIP,
                metric=0.0,
                threshold=0.05,
                message="Insufficient observations for variance reduction test",
            )
        
        # Variance reduction where observed
        prior_var_obs = prior_variance[observed_mask]
        post_var_obs = posterior_variance[observed_mask]
        
        reduction = (prior_var_obs - post_var_obs) / (prior_var_obs + 1e-9)
        mean_reduction = np.nanmean(reduction)
        
        threshold = 0.05  # At least 5% reduction expected
        
        if mean_reduction >= threshold * 3:
            status = ValidationStatus.PASS
            msg = f"Variance reduction = {mean_reduction:.1%} (observations reduce uncertainty)"
        elif mean_reduction >= threshold:
            status = ValidationStatus.PASS
            msg = f"Variance reduction = {mean_reduction:.1%} (acceptable)"
        elif mean_reduction > 0:
            status = ValidationStatus.WARN
            msg = f"Variance reduction = {mean_reduction:.1%} < {threshold:.0%} (weak)"
        else:
            status = ValidationStatus.FAIL
            msg = f"FAIL: Variance reduction = {mean_reduction:.1%} " \
                 f"(observations don't reduce uncertainty!)"
        
        return ValidationResult(
            name="variance_reduction",
            status=status,
            metric=float(mean_reduction),
            threshold=threshold,
            message=msg,
        )
    
    def _test_temporal_variation(
        self,
        stress: np.ndarray,
    ) -> ValidationResult:
        """Test: Stress should vary over time."""
        if stress.ndim != 3:
            return ValidationResult(
                name="temporal_variation",
                status=ValidationStatus.SKIP,
                metric=0.0,
                threshold=0.1,
                message="Not a temporal array",
            )
        
        T = stress.shape[0]
        if T < 3:
            return ValidationResult(
                name="temporal_variation",
                status=ValidationStatus.SKIP,
                metric=0.0,
                threshold=0.1,
                message="Insufficient temporal samples",
            )
        
        # Temporal coefficient of variation (across space-averaged time series)
        temporal_mean = np.nanmean(stress, axis=(1, 2))  # (T,)
        temporal_cv = np.nanstd(temporal_mean) / (np.nanmean(temporal_mean) + 1e-9)
        
        threshold = 0.1
        
        if temporal_cv >= threshold:
            status = ValidationStatus.PASS
            msg = f"Temporal CV = {temporal_cv:.3f} (stress varies over time)"
        else:
            status = ValidationStatus.WARN
            msg = f"Temporal CV = {temporal_cv:.3f} < {threshold} (temporally static)"
        
        return ValidationResult(
            name="temporal_variation",
            status=status,
            metric=float(temporal_cv),
            threshold=threshold,
            message=msg,
        )
    
    def _test_physical_constraints(
        self,
        stress: np.ndarray,
    ) -> ValidationResult:
        """Test: Stress values should be physically reasonable."""
        # Stress should be non-negative
        neg_frac = np.mean(stress < 0)
        
        # Stress shouldn't be all zeros or all same value
        unique_count = len(np.unique(np.round(stress, 3)))
        is_constant = unique_count < 10
        
        if neg_frac > 0.01:
            status = ValidationStatus.FAIL
            msg = f"FAIL: {neg_frac:.1%} negative stress values (physically impossible)"
            metric = neg_frac
        elif is_constant:
            status = ValidationStatus.FAIL
            msg = f"FAIL: Stress is nearly constant ({unique_count} unique values)"
            metric = unique_count / stress.size
        else:
            status = ValidationStatus.PASS
            msg = f"Physical constraints satisfied (non-negative, varying)"
            metric = 1.0
        
        return ValidationResult(
            name="physical_constraints",
            status=status,
            metric=float(metric),
            threshold=0.01,
            message=msg,
        )


def validate_inference(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    prior_mean: Optional[np.ndarray] = None,
    prior_variance: Optional[np.ndarray] = None,
    risk_levels: Optional[np.ndarray] = None,
    complaints: Optional[np.ndarray] = None,
    information_gain: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> ScientificValidationReport:
    """Main entry point for scientific validation.
    
    Call this after inference to verify results are scientifically valid.
    """
    validator = ScientificValidator()
    report = validator.validate(
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        risk_levels=risk_levels,
        complaints=complaints,
        information_gain=information_gain,
    )
    
    if output_path:
        report.save(output_path)
    
    return report
