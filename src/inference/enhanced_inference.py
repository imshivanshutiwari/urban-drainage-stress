"""Enhanced inference engine with scientifically valid uncertainty and dynamics.

This module integrates all the scientific corrections:
1. Data-driven heterogeneous uncertainty (NOW WITH MATH_CORE)
2. Realistic spatial structure (local anomalies)
3. Realistic temporal dynamics (lag, hysteresis)
4. Sanity-tested observation influence
5. Epistemic/aleatoric decomposition
6. Non-stationary covariance structures

Reference: Projectfile.md - Scientific validity requirements
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..decision.latent_stress_model import LatentStressModel, LatentStressResult
from ..decision.observation_model import ObservationModel, ObservationLikelihoodResult
from ..decision.uncertainty_model import (
    DataDrivenUncertaintyModel,
    UncertaintyComponents,
    UncertaintyConfig,
)
from ..decision.advanced_uncertainty import (
    AdvancedUncertaintyModel,
    AdvancedUncertaintyComponents,
    AdvancedUncertaintyConfig,
)
from ..decision.spatial_structure import (
    RealisticStressModel,
    RealisticStressResult,
    SpatialStructureConfig,
)
from ..decision.temporal_dynamics import (
    RealisticTemporalModel,
    TemporalDynamicsResult,
    TemporalDynamicsConfig,
)
from ..config.parameters import InferenceConfig, get_default_parameters

logger = logging.getLogger(__name__)


@dataclass
class EnhancedInferenceResult:
    """Enhanced inference result with scientific validity."""
    
    # Core outputs
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    reliability: np.ndarray
    
    # Enhanced uncertainty breakdown (supports both old and new format)
    uncertainty_components: Union[UncertaintyComponents, AdvancedUncertaintyComponents]
    
    # Epistemic/aleatoric decomposition (new math_core feature)
    epistemic_uncertainty: Optional[np.ndarray] = None
    aleatoric_uncertainty: Optional[np.ndarray] = None
    information_gain: Optional[np.ndarray] = None
    
    # Spatial structure diagnostics
    spatial_roughness: float = 0.0
    anomaly_locations: Optional[np.ndarray] = None
    
    # Temporal dynamics diagnostics
    temporal_asymmetry: float = 0.0
    lag_field: Optional[np.ndarray] = None
    peak_times: Optional[np.ndarray] = None
    
    # Observation influence
    observation_loglik: Optional[np.ndarray] = None
    
    # Diagnostics
    notes: list = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class EnhancedInferenceEngine:
    """Scientifically valid inference engine with MATH_CORE integration.
    
    Integrates:
    - Heterogeneous, explainable uncertainty (ADVANCED: epistemic/aleatoric)
    - Non-stationary covariance structures (MATH_CORE)
    - Realistic spatial structure with anomalies
    - Realistic temporal dynamics with lag/hysteresis
    - Data-driven observation influence
    - Information-theoretic metrics
    
    All improvements are from Projectfile.md scientific requirements + math_core.
    """
    
    def __init__(
        self,
        latent_model: Optional[LatentStressModel] = None,
        observation_model: Optional[ObservationModel] = None,
        uncertainty_model: Optional[DataDrivenUncertaintyModel] = None,
        advanced_uncertainty_model: Optional[AdvancedUncertaintyModel] = None,
        spatial_model: Optional[RealisticStressModel] = None,
        temporal_model: Optional[RealisticTemporalModel] = None,
        config: Optional[InferenceConfig] = None,
        use_advanced_uncertainty: bool = True,  # Use math_core by default
    ) -> None:
        self.latent_model = latent_model or LatentStressModel()
        self.observation_model = observation_model or ObservationModel()
        self.uncertainty_model = uncertainty_model or DataDrivenUncertaintyModel()
        self.advanced_uncertainty_model = advanced_uncertainty_model or AdvancedUncertaintyModel()
        self.use_advanced_uncertainty = use_advanced_uncertainty
        self.spatial_model = spatial_model or RealisticStressModel()
        self.temporal_model = temporal_model or RealisticTemporalModel()
        self.config = config or get_default_parameters().inference
        
        logger.info("EnhancedInferenceEngine initialized with scientific corrections")
    
    def infer(
        self,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        upstream_contribution: np.ndarray,
        complaints: np.ndarray,
        upstream_confidence: Optional[np.ndarray] = None,
        spatial_bias: Optional[np.ndarray] = None,
        temporal_modifier: Optional[np.ndarray] = None,
        enable_spatial_structure: bool = True,
        enable_temporal_dynamics: bool = True,
        enable_enhanced_uncertainty: bool = True,
    ) -> EnhancedInferenceResult:
        """Run enhanced inference with scientific corrections.
        
        Args:
            rainfall_intensity: (time, y, x) rainfall intensity
            rainfall_accumulation: (time, y, x) accumulated rainfall
            upstream_contribution: (y, x) upstream weighting
            complaints: (time, y, x) complaint counts
            upstream_confidence: (y, x) optional confidence in upstream
            spatial_bias: (y, x) optional spatial reporting bias
            temporal_modifier: (time,) optional temporal reporting modifier
            enable_spatial_structure: Add realistic spatial anomalies
            enable_temporal_dynamics: Add lag and hysteresis
            enable_enhanced_uncertainty: Use data-driven uncertainty
            
        Returns:
            EnhancedInferenceResult with scientific diagnostics
        """
        notes = []
        
        # 1. Base latent stress model
        logger.info("Step 1/5: Computing base latent stress...")
        latent: LatentStressResult = self.latent_model.infer(
            rainfall_intensity,
            rainfall_accumulation,
            upstream_contribution,
            upstream_confidence=upstream_confidence,
        )
        notes.extend(latent.notes)
        
        # 2. Observation likelihood
        logger.info("Step 2/5: Computing observation likelihood...")
        obs: ObservationLikelihoodResult = self.observation_model.infer_likelihood(
            latent.mean,
            complaints,
            spatial_bias=spatial_bias,
            temporal_modifier=temporal_modifier,
        )
        notes.extend(obs.notes)
        
        # 3. Bayesian fusion
        logger.info("Step 3/5: Bayesian fusion...")
        posterior_mean, posterior_var, base_no_decision = self._fuse(latent, obs)
        
        # 4. Add realistic spatial structure (if enabled)
        if enable_spatial_structure:
            logger.info("Step 4a/5: Adding realistic spatial structure...")
            spatial: RealisticStressResult = self.spatial_model.add_spatial_structure(
                posterior_mean,
                upstream_contribution,
                complaints,
            )
            posterior_mean = spatial.stress
            spatial_roughness = spatial.spatial_roughness
            anomaly_locations = spatial.anomaly_mask
            notes.append(f"Spatial roughness: {spatial_roughness:.3f}")
        else:
            spatial_roughness = 0.0
            anomaly_locations = np.zeros_like(posterior_mean, dtype=bool)
        
        # 5. Add realistic temporal dynamics (if enabled)
        if enable_temporal_dynamics:
            logger.info("Step 4b/5: Adding realistic temporal dynamics...")
            temporal: TemporalDynamicsResult = self.temporal_model.apply_temporal_dynamics(
                posterior_mean,
                rainfall_intensity,
                upstream_contribution,
                posterior_var,
            )
            posterior_mean = temporal.stress
            posterior_var = temporal.uncertainty ** 2  # Convert std to variance
            temporal_asymmetry = temporal.temporal_asymmetry
            lag_field = temporal.lag_field
            peak_times = temporal.peak_time
            notes.append(f"Temporal asymmetry: {temporal_asymmetry:.3f}")
        else:
            temporal_asymmetry = 0.0
            lag_field = np.zeros(upstream_contribution.shape, dtype=np.float32)
            peak_times = np.zeros(upstream_contribution.shape, dtype=np.int32)
        
        # 6. Compute enhanced uncertainty (if enabled)
        epistemic_unc = None
        aleatoric_unc = None
        info_gain = None
        
        if enable_enhanced_uncertainty:
            if self.use_advanced_uncertainty:
                # USE MATH_CORE ADVANCED UNCERTAINTY MODEL
                logger.info("Step 5/5: Computing ADVANCED uncertainty with math_core...")
                adv_uncertainty: AdvancedUncertaintyComponents = (
                    self.advanced_uncertainty_model.compute_uncertainty(
                        rainfall_intensity,
                        rainfall_accumulation,
                        complaints,
                        upstream_contribution,
                        posterior_var,
                    )
                )
                # Replace posterior variance with advanced uncertainty
                posterior_var = adv_uncertainty.total_uncertainty ** 2
                reliability = latent.reliability & obs.confidence & adv_uncertainty.coverage_mask
                
                # Extract epistemic/aleatoric decomposition
                epistemic_unc = adv_uncertainty.epistemic_uncertainty
                aleatoric_unc = adv_uncertainty.aleatoric_uncertainty
                info_gain = adv_uncertainty.information_gain
                
                cv = adv_uncertainty.cv
                notes.append(f"Advanced Uncertainty CV: {cv:.3f} (math_core)")
                notes.append(f"Epistemic fraction: {np.mean(adv_uncertainty.reducibility_fraction):.2f}")
                
                # Convert to standard format for compatibility
                uncertainty = UncertaintyComponents(
                    rainfall_uncertainty=adv_uncertainty.rainfall_uncertainty,
                    observation_uncertainty=adv_uncertainty.observation_uncertainty,
                    structural_uncertainty=adv_uncertainty.structural_uncertainty,
                    combined_uncertainty=adv_uncertainty.total_uncertainty,
                    coverage_mask=adv_uncertainty.coverage_mask,
                )
            else:
                # Use basic uncertainty model (fallback)
                logger.info("Step 5/5: Computing basic data-driven uncertainty...")
                uncertainty: UncertaintyComponents = self.uncertainty_model.compute_uncertainty(
                    rainfall_intensity,
                    rainfall_accumulation,
                    complaints,
                    upstream_contribution,
                    posterior_var,
                )
                # Replace posterior variance with data-driven uncertainty
                posterior_var = uncertainty.combined_uncertainty ** 2
                reliability = latent.reliability & obs.confidence & uncertainty.coverage_mask
                notes.append(f"Uncertainty CV: {np.std(uncertainty.combined_uncertainty) / np.mean(uncertainty.combined_uncertainty):.3f}")
        else:
            # Create minimal uncertainty components
            uncertainty = UncertaintyComponents(
                rainfall_uncertainty=np.zeros_like(posterior_var),
                observation_uncertainty=np.zeros_like(posterior_var),
                structural_uncertainty=np.zeros_like(posterior_var),
                combined_uncertainty=np.sqrt(posterior_var),
                coverage_mask=latent.reliability & obs.confidence,
            )
            reliability = latent.reliability & obs.confidence
        
        # Log final statistics
        logger.info(
            "Inference complete: mean_stress=%.3f, mean_uncertainty=%.3f, "
            "reliable_cells=%.1f%%",
            np.nanmean(posterior_mean),
            np.nanmean(np.sqrt(posterior_var)),
            reliability.mean() * 100
        )
        
        return EnhancedInferenceResult(
            posterior_mean=posterior_mean.astype(np.float32),
            posterior_variance=posterior_var.astype(np.float32),
            reliability=reliability.astype(bool),
            uncertainty_components=uncertainty,
            epistemic_uncertainty=epistemic_unc,
            aleatoric_uncertainty=aleatoric_unc,
            information_gain=info_gain,
            spatial_roughness=spatial_roughness,
            anomaly_locations=anomaly_locations,
            temporal_asymmetry=temporal_asymmetry,
            lag_field=lag_field,
            peak_times=peak_times,
            observation_loglik=obs.log_likelihood,
            notes=notes,
        )
    
    def _fuse(
        self,
        latent: LatentStressResult,
        obs: ObservationLikelihoodResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bayesian fusion via precision addition (unchanged from base)."""
        
        prior_mean = latent.mean
        prior_var = latent.variance
        prior_precision = 1.0 / np.maximum(prior_var, 1e-6)
        
        # Observation precision from likelihood
        obs_precision = np.where(
            obs.likelihood > 0,
            np.minimum(obs.likelihood, 1.0) * self.config.anomaly_zscore,
            0.0,
        )
        obs_mean_proxy = np.clip(obs.rate, 0.0, None)
        
        post_precision = prior_precision + obs_precision
        with np.errstate(divide="ignore", invalid="ignore"):
            posterior_mean = (
                prior_precision * prior_mean + obs_precision * obs_mean_proxy
            ) / np.maximum(post_precision, 1e-9)
        posterior_var = 1.0 / np.maximum(post_precision, 1e-9)
        
        no_decision = (
            post_precision < self.config.min_event_duration_minutes * 1e-3
        )
        
        return (
            posterior_mean.astype(np.float32),
            posterior_var.astype(np.float32),
            no_decision.astype(bool),
        )
