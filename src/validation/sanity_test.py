"""Sanity Test: The Smoking Gun.

This is the MANDATORY test from Projectfile.md:

🔬 Sanity Test (MANDATORY)
1. Run system with complaints
2. Run system without complaints  
3. Plot uncertainty maps for both
4. If they look similar → observation model is NOT influencing inference

This test reveals if the model is actually data-driven or fake.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SanityTestConfig:
    """Configuration for sanity testing."""
    
    # Thresholds for pass/fail
    min_uncertainty_diff: float = 0.1      # Min 10% difference required
    min_stress_diff: float = 0.05          # Min 5% stress difference
    min_decision_diff: float = 0.1         # Min 10% decision change
    
    # Visualization settings
    generate_comparison_plots: bool = True
    save_detailed_report: bool = True


@dataclass
class SanityTestResult:
    """Result of sanity test."""
    
    passed: bool                           # Did test pass?
    
    # With complaints
    uncertainty_with: np.ndarray
    stress_with: np.ndarray
    decisions_with: np.ndarray
    
    # Without complaints
    uncertainty_without: np.ndarray
    stress_without: np.ndarray
    decisions_without: np.ndarray
    
    # Differences
    uncertainty_diff_ratio: float          # How different?
    stress_diff_ratio: float
    decision_change_pct: float
    
    # Diagnostics
    diagnosis: str                         # What's wrong?
    recommendations: list[str]             # How to fix?


class SanityTester:
    """Run the smoking gun sanity test.
    
    If uncertainty looks the same with and without complaints,
    the observation model is NOT influencing inference.
    
    That's the smoking gun for a fake model.
    """
    
    def __init__(self, config: Optional[SanityTestConfig] = None) -> None:
        self.config = config or SanityTestConfig()
        logger.info("SanityTester initialized")
    
    def run_sanity_test(
        self,
        run_inference_fn,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        complaints: np.ndarray,
        upstream: np.ndarray,
    ) -> SanityTestResult:
        """Run the sanity test.
        
        Args:
            run_inference_fn: Function that takes (rainfall_i, rainfall_a, 
                            complaints, upstream) and returns (stress, uncertainty)
            rainfall_intensity: (time, y, x) rainfall
            rainfall_accumulation: (time, y, x) accumulated rainfall
            complaints: (time, y, x) actual complaints
            upstream: (y, x) upstream contribution
            
        Returns:
            SanityTestResult with pass/fail and diagnostics
        """
        logger.info("=" * 60)
        logger.info("RUNNING SANITY TEST: With vs Without Complaints")
        logger.info("=" * 60)
        
        # 1. Run WITH complaints
        logger.info("Step 1: Running inference WITH complaints...")
        stress_with, unc_with = run_inference_fn(
            rainfall_intensity,
            rainfall_accumulation,
            complaints,
            upstream,
        )
        
        # 2. Run WITHOUT complaints (all zeros)
        logger.info("Step 2: Running inference WITHOUT complaints...")
        empty_complaints = np.zeros_like(complaints)
        stress_without, unc_without = run_inference_fn(
            rainfall_intensity,
            rainfall_accumulation,
            empty_complaints,
            upstream,
        )
        
        # 3. Compute differences
        logger.info("Step 3: Computing differences...")
        
        # Uncertainty difference
        unc_diff = np.abs(unc_with - unc_without)
        mean_unc = (np.nanmean(unc_with) + np.nanmean(unc_without)) / 2 + 1e-9
        unc_diff_ratio = np.nanmean(unc_diff) / mean_unc
        
        # Stress difference
        stress_diff = np.abs(stress_with - stress_without)
        mean_stress = (np.nanmean(stress_with) + np.nanmean(stress_without)) / 2 + 1e-9
        stress_diff_ratio = np.nanmean(stress_diff) / mean_stress
        
        # Decision changes (simplified)
        threshold = np.median(stress_with)
        decisions_with = (stress_with > threshold) & (unc_with < np.median(unc_with))
        decisions_without = (stress_without > threshold) & (unc_without < np.median(unc_without))
        decision_change_pct = np.mean(decisions_with != decisions_without) * 100
        
        # 4. Evaluate pass/fail
        cfg = self.config
        passed = (
            unc_diff_ratio >= cfg.min_uncertainty_diff and
            stress_diff_ratio >= cfg.min_stress_diff
        )
        
        # 5. Generate diagnosis
        diagnosis, recommendations = self._diagnose(
            unc_diff_ratio, stress_diff_ratio, decision_change_pct, passed
        )
        
        # 6. Log results
        logger.info("-" * 60)
        if passed:
            logger.info("✅ SANITY TEST PASSED")
        else:
            logger.error("❌ SANITY TEST FAILED")
        
        logger.info(
            "  Uncertainty difference: %.1f%% (threshold: %.1f%%)",
            unc_diff_ratio * 100, cfg.min_uncertainty_diff * 100
        )
        logger.info(
            "  Stress difference: %.1f%% (threshold: %.1f%%)",
            stress_diff_ratio * 100, cfg.min_stress_diff * 100
        )
        logger.info(
            "  Decision changes: %.1f%%",
            decision_change_pct
        )
        logger.info("  Diagnosis: %s", diagnosis)
        
        for rec in recommendations:
            logger.info("  → %s", rec)
        
        logger.info("-" * 60)
        
        return SanityTestResult(
            passed=passed,
            uncertainty_with=unc_with,
            stress_with=stress_with,
            decisions_with=decisions_with,
            uncertainty_without=unc_without,
            stress_without=stress_without,
            decisions_without=decisions_without,
            uncertainty_diff_ratio=unc_diff_ratio,
            stress_diff_ratio=stress_diff_ratio,
            decision_change_pct=decision_change_pct,
            diagnosis=diagnosis,
            recommendations=recommendations,
        )
    
    def _diagnose(
        self,
        unc_diff: float,
        stress_diff: float,
        dec_change: float,
        passed: bool,
    ) -> Tuple[str, list[str]]:
        """Generate diagnosis and recommendations."""
        cfg = self.config
        
        if passed:
            return (
                "Observation model is correctly influencing inference.",
                [
                    "Model is data-driven as expected.",
                    "Continue with production runs."
                ]
            )
        
        # Diagnose failure modes
        recommendations = []
        
        if unc_diff < cfg.min_uncertainty_diff:
            if unc_diff < 0.01:
                diagnosis = (
                    "CRITICAL: Uncertainty is IDENTICAL with/without complaints. "
                    "Observation model has NO effect!"
                )
                recommendations.extend([
                    "Check if observation_model.infer_likelihood is being used",
                    "Verify complaints are being passed to inference engine",
                    "Review uncertainty fusion in _fuse() method"
                ])
            else:
                diagnosis = (
                    "Uncertainty barely changes with complaints. "
                    "Observation influence is too weak."
                )
                recommendations.extend([
                    "Increase observation precision weight in fusion",
                    "Check spatial alignment of complaints",
                    "Verify delay kernel is appropriate"
                ])
        
        if stress_diff < cfg.min_stress_diff:
            if stress_diff < 0.01:
                diagnosis = (
                    "CRITICAL: Stress is IDENTICAL with/without complaints. "
                    "Complaints have NO effect on stress estimation!"
                )
                recommendations.extend([
                    "Check posterior mean computation",
                    "Verify observation likelihood is being computed",
                    "Review precision weighting in Bayesian update"
                ])
            else:
                diagnosis = (
                    f"Stress changes minimally ({stress_diff:.1%}). "
                    "Observation-based correction is too weak."
                )
                recommendations.extend([
                    "Increase observation weight in posterior",
                    "Review obs_precision computation in _fuse()"
                ])
        
        if dec_change < 5:
            recommendations.append(
                "Consider: if decisions don't change with new data, "
                "the system provides no decision value"
            )
        
        return diagnosis, recommendations
    
    def generate_comparison_report(
        self,
        result: SanityTestResult,
        output_dir: Path,
    ) -> Path:
        """Generate a detailed comparison report."""
        report_path = output_dir / "sanity_test_report.json"
        
        import json
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "passed": result.passed,
            "metrics": {
                "uncertainty_diff_ratio": float(result.uncertainty_diff_ratio),
                "stress_diff_ratio": float(result.stress_diff_ratio),
                "decision_change_pct": float(result.decision_change_pct),
            },
            "thresholds": {
                "min_uncertainty_diff": self.config.min_uncertainty_diff,
                "min_stress_diff": self.config.min_stress_diff,
                "min_decision_diff": self.config.min_decision_diff,
            },
            "diagnosis": result.diagnosis,
            "recommendations": result.recommendations,
            "statistics": {
                "with_complaints": {
                    "uncertainty_mean": float(np.nanmean(result.uncertainty_with)),
                    "uncertainty_std": float(np.nanstd(result.uncertainty_with)),
                    "stress_mean": float(np.nanmean(result.stress_with)),
                    "stress_std": float(np.nanstd(result.stress_with)),
                },
                "without_complaints": {
                    "uncertainty_mean": float(np.nanmean(result.uncertainty_without)),
                    "uncertainty_std": float(np.nanstd(result.uncertainty_without)),
                    "stress_mean": float(np.nanmean(result.stress_without)),
                    "stress_std": float(np.nanstd(result.stress_without)),
                },
            },
        }
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("Saved sanity test report to %s", report_path)
        return report_path


def run_sanity_test(
    inference_fn,
    rainfall_intensity: np.ndarray,
    rainfall_accumulation: np.ndarray,
    complaints: np.ndarray,
    upstream: np.ndarray,
    config: Optional[SanityTestConfig] = None,
) -> SanityTestResult:
    """Convenience function for sanity testing."""
    tester = SanityTester(config)
    return tester.run_sanity_test(
        inference_fn,
        rainfall_intensity,
        rainfall_accumulation,
        complaints,
        upstream,
    )
