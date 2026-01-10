"""Scientific Pipeline: Integrates all components into a research-grade system.

This is the MAIN ENTRY POINT that ensures:
1. Parameters are LEARNED from data (not fixed)
2. Inference is TRUE Bayesian (posterior = prior × likelihood)
3. Uncertainty is POSTERIOR variance (not injected noise)
4. Decisions use CREDIBLE INTERVALS
5. Validation ENFORCES scientific criteria
6. Reports are HONEST about limitations

Reference: Projectfile.md - "FULL SYSTEM UPGRADE TO SCIENTIFICALLY COMPLETE, 
DATA-CONSTRAINED INFERENCE"

If this pipeline passes validation → results can be trusted.
If it fails → results should NOT be used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScientificPipelineConfig:
    """Configuration for scientific pipeline."""
    
    # Parameter learning
    learning_rate: float = 0.01
    n_bootstrap: int = 100
    regularization: float = 0.1
    
    # Bayesian inference
    prior_variance_scale: float = 1.0
    observation_noise: float = 0.1
    
    # Decision thresholds
    high_threshold: float = 0.7
    medium_threshold: float = 0.4
    low_threshold: float = 0.2
    max_ci_width_for_decision: float = 0.25
    
    # Validation
    min_no_decision_fraction: float = 0.05
    max_high_risk_fraction: float = 0.80
    min_observation_impact: float = 0.01
    
    # Output
    output_dir: Path = Path("outputs/scientific_run")
    save_intermediate: bool = True


@dataclass
class ScientificPipelineResult:
    """Complete results from scientific pipeline."""
    
    # Posterior inference
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    prior_mean: np.ndarray
    prior_variance: np.ndarray
    
    # Credible intervals
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    
    # Information gain
    information_gain: np.ndarray
    
    # Risk decisions
    risk_levels: np.ndarray
    risk_stats: Dict[str, float]
    
    # Learned parameters
    learned_weights: Dict[str, float]
    weights_ci: Dict[str, Tuple[float, float]]
    memory_decay: float
    memory_decay_ci: Tuple[float, float]
    fit_metrics: Dict[str, float]
    
    # Validation
    validation_passed: bool
    validation_report: Any
    integrity_score: float
    
    # Self-awareness
    self_awareness_report: Any
    
    # Metadata
    timestamp: str
    duration_seconds: float
    
    def save(self, output_dir: Path) -> None:
        """Save all results to disk."""
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save arrays as numpy
        np.save(output_dir / "posterior_mean.npy", self.posterior_mean)
        np.save(output_dir / "posterior_variance.npy", self.posterior_variance)
        np.save(output_dir / "prior_mean.npy", self.prior_mean)
        np.save(output_dir / "prior_variance.npy", self.prior_variance)
        np.save(output_dir / "ci_lower.npy", self.ci_lower)
        np.save(output_dir / "ci_upper.npy", self.ci_upper)
        np.save(output_dir / "information_gain.npy", self.information_gain)
        np.save(output_dir / "risk_levels.npy", self.risk_levels)
        
        # Save metadata
        metadata = {
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "validation_passed": self.validation_passed,
            "integrity_score": self.integrity_score,
            "risk_stats": self.risk_stats,
            "learned_weights": self.learned_weights,
            "weights_ci": {k: list(v) for k, v in self.weights_ci.items()},
            "memory_decay": self.memory_decay,
            "memory_decay_ci": list(self.memory_decay_ci),
            "fit_metrics": self.fit_metrics,
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Saved scientific pipeline results to %s", output_dir)


class ScientificPipeline:
    """Research-grade inference pipeline.
    
    This pipeline ensures:
    1. ALL parameters are learned from data with confidence intervals
    2. Inference follows Bayes' theorem exactly
    3. Observations actively constrain posterior
    4. Uncertainty is posterior variance, not injected noise
    5. Decisions require confidence (NO_DECISION when uncertain)
    6. Results are validated against scientific criteria
    7. Limitations are honestly reported
    """
    
    def __init__(self, config: Optional[ScientificPipelineConfig] = None) -> None:
        self.config = config or ScientificPipelineConfig()
        logger.info("ScientificPipeline initialized with config: %s", self.config)
    
    def run(
        self,
        rainfall: np.ndarray,
        accumulation: np.ndarray,
        upstream_area: np.ndarray,
        complaints: np.ndarray,
        timestamps: Optional[List] = None,
    ) -> ScientificPipelineResult:
        """Run complete scientific pipeline.
        
        Args:
            rainfall: (T, H, W) daily rainfall
            accumulation: (H, W) flow accumulation
            upstream_area: (H, W) upstream area
            complaints: (T, H, W) observed complaints
            timestamps: Optional list of timestamps
            
        Returns:
            ScientificPipelineResult with all outputs
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("SCIENTIFIC PIPELINE STARTED")
        logger.info("=" * 60)
        
        # Import components
        from src.calibration.parameter_learning import calibrate_model
        from src.inference.bayesian_inference import TrueBayesianInference
        from src.decision.risk_decision_model import (
            CredibleIntervalRiskModel, 
            compute_risk_from_bayesian_posterior
        )
        from src.validation.scientific_validation import validate_inference
        from src.visualization.self_awareness import generate_self_awareness_report
        
        cfg = self.config
        
        # ============================================================
        # STEP 1: Parameter Learning
        # ============================================================
        logger.info("STEP 1: Learning parameters from data...")
        
        calibration_result = calibrate_model(
            rainfall_intensity=rainfall,
            rainfall_accumulation=accumulation,
            upstream_contribution=upstream_area,
            complaints=complaints,
        )
        
        # Extract learned parameters
        learned_weights = {
            'intensity': calibration_result.weight_intensity,
            'accumulation': calibration_result.weight_accumulation,
            'upstream': calibration_result.weight_upstream,
        }
        weights_ci = {
            'intensity': calibration_result.ci_weight_intensity,
            'accumulation': calibration_result.ci_weight_accumulation,
            'upstream': calibration_result.ci_weight_upstream,
        }
        memory_decay = calibration_result.memory_decay
        memory_decay_ci = calibration_result.ci_memory_decay
        fit_metrics = {
            'r_squared': calibration_result.r_squared,
            'aic': calibration_result.aic,
            'bic': calibration_result.bic,
            'log_likelihood': calibration_result.log_likelihood,
        }
        
        logger.info("Learned weights: %s", learned_weights)
        logger.info("Fit R²: %.3f", fit_metrics.get('r_squared', 0))
        
        # ============================================================
        # STEP 2: True Bayesian Inference
        # ============================================================
        logger.info("STEP 2: Running Bayesian inference...")
        
        inference_engine = TrueBayesianInference()
        
        # Convert learned weights dict to tuple for inference
        weights_tuple = (
            learned_weights['intensity'],
            learned_weights['accumulation'],
            learned_weights['upstream'],
        )
        
        inference_result = inference_engine.infer(
            rainfall_intensity=rainfall,
            rainfall_accumulation=accumulation,
            upstream_contribution=upstream_area,
            complaints=complaints,
            weights=weights_tuple,
        )
        
        # Extract results from BayesianPosterior
        posterior_mean = inference_result.mean
        posterior_variance = inference_result.variance
        ci_lower = inference_result.credible_lower
        ci_upper = inference_result.credible_upper
        information_gain = inference_result.information_gain
        
        # Get prior info directly from inference result
        prior_mean = inference_result.prior_mean
        prior_variance = inference_result.prior_variance
        
        logger.info("Posterior shape: %s", posterior_mean.shape)
        logger.info("Mean info gain: %.4f", np.nanmean(information_gain))
        
        # ============================================================
        # STEP 3: Risk Decision (using credible intervals)
        # ============================================================
        logger.info("STEP 3: Computing risk decisions from credible intervals...")
        
        risk_model = CredibleIntervalRiskModel(
            high_threshold=cfg.high_threshold,
            medium_threshold=cfg.medium_threshold,
            low_threshold=cfg.low_threshold,
            max_ci_width=cfg.max_ci_width_for_decision,
        )
        
        risk_result = risk_model.compute_risk_from_credible_intervals(
            posterior_mean=posterior_mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
        
        risk_levels = risk_result.risk_levels
        risk_stats = {
            'high_pct': risk_result.high_pct,
            'medium_pct': risk_result.medium_pct,
            'low_pct': risk_result.low_pct,
            'no_decision_pct': risk_result.no_decision_pct,
        }
        
        logger.info("Risk distribution: HIGH=%.1f%%, MEDIUM=%.1f%%, LOW=%.1f%%, NO_DECISION=%.1f%%",
                   risk_stats['high_pct'], risk_stats['medium_pct'], 
                   risk_stats['low_pct'], risk_stats['no_decision_pct'])
        
        # ============================================================
        # STEP 4: Scientific Validation
        # ============================================================
        logger.info("STEP 4: Validating scientific integrity...")
        
        validation_report = validate_inference(
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            risk_levels=risk_levels,
            complaints=complaints,
            information_gain=information_gain,
            output_path=cfg.output_dir / "validation_report.json" if cfg.save_intermediate else None,
        )
        
        validation_passed = validation_report.is_valid
        integrity_score = validation_report.integrity_score
        
        if validation_passed:
            logger.info("✓ VALIDATION PASSED (integrity: %.0f/100)", integrity_score)
        else:
            logger.error("✗ VALIDATION FAILED (integrity: %.0f/100)", integrity_score)
            logger.error("  %s", validation_report.summary)
        
        # ============================================================
        # STEP 5: Self-Awareness Report
        # ============================================================
        logger.info("STEP 5: Generating self-awareness report...")
        
        T, H, W = posterior_mean.shape if posterior_mean.ndim == 3 else (1,) + posterior_mean.shape
        
        self_awareness_report = generate_self_awareness_report(
            weights=learned_weights,
            weights_ci=weights_ci,
            memory_decay=memory_decay,
            memory_decay_ci=memory_decay_ci,
            fit_metrics=fit_metrics,
            posterior_variance=posterior_variance,
            risk_levels=risk_levels,
            validation_passed=validation_passed,
            integrity_score=integrity_score,
            n_timesteps=T,
            grid_shape=(H, W),
            complaints=complaints,
            output_path=cfg.output_dir / "self_awareness_report.json" if cfg.save_intermediate else None,
            print_summary=True,
        )
        
        # ============================================================
        # STEP 6: Generate Visualizations
        # ============================================================
        if cfg.save_intermediate:
            logger.info("STEP 6: Generating visualizations...")
            
            from src.visualization.bayesian_visualizations import generate_all_bayesian_visualizations
            
            vis_paths = generate_all_bayesian_visualizations(
                posterior_mean=posterior_mean,
                posterior_variance=posterior_variance,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                information_gain=information_gain,
                risk_levels=risk_levels,
                complaints=complaints,
                learned_params=learned_weights,
                param_cis=weights_ci,
                fit_metrics=fit_metrics,
                output_dir=cfg.output_dir / "visualizations",
            )
            logger.info("Generated %d visualizations", len(vis_paths))
        
        # ============================================================
        # Finalize
        # ============================================================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("SCIENTIFIC PIPELINE COMPLETED in %.1fs", duration)
        logger.info("Validation: %s", "PASSED ✓" if validation_passed else "FAILED ✗")
        logger.info("Integrity Score: %.0f/100", integrity_score)
        logger.info("=" * 60)
        
        result = ScientificPipelineResult(
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            information_gain=information_gain,
            risk_levels=risk_levels,
            risk_stats=risk_stats,
            learned_weights=learned_weights,
            weights_ci=weights_ci,
            memory_decay=memory_decay,
            memory_decay_ci=memory_decay_ci,
            fit_metrics=fit_metrics,
            validation_passed=validation_passed,
            validation_report=validation_report,
            integrity_score=integrity_score,
            self_awareness_report=self_awareness_report,
            timestamp=start_time.isoformat(),
            duration_seconds=duration,
        )
        
        if cfg.save_intermediate:
            result.save(cfg.output_dir)
        
        return result


def run_scientific_pipeline(
    rainfall: np.ndarray,
    accumulation: np.ndarray,
    upstream_area: np.ndarray,
    complaints: np.ndarray,
    output_dir: Optional[Path] = None,
    **config_kwargs,
) -> ScientificPipelineResult:
    """Convenience function to run scientific pipeline.
    
    This is the MAIN ENTRY POINT for research-grade inference.
    """
    config = ScientificPipelineConfig(**config_kwargs)
    if output_dir:
        config.output_dir = Path(output_dir)
    
    pipeline = ScientificPipeline(config)
    return pipeline.run(
        rainfall=rainfall,
        accumulation=accumulation,
        upstream_area=upstream_area,
        complaints=complaints,
    )
