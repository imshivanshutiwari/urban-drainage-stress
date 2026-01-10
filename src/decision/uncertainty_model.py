"""Data-driven uncertainty model: heterogeneous, explainable uncertainty.

This module implements scientifically valid uncertainty quantification that:
1. Varies spatially based on data density/quality
2. Responds to missing data (rainfall gaps, complaint gaps)
3. Is NOT uniform - must show ragged edges, holes, and spikes
4. Follows: uncertainty(x,t) = rainfall_uncertainty + observation_uncertainty + model_structural_uncertainty

Reference: Projectfile.md - "Uncertainty must be heterogeneous and explainable"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyComponents:
    """Decomposed uncertainty sources for explainability."""
    
    rainfall_uncertainty: np.ndarray      # Where rainfall data is sparse/missing
    observation_uncertainty: np.ndarray   # Where complaints are sparse/missing
    structural_uncertainty: np.ndarray    # Model extrapolation uncertainty
    combined_uncertainty: np.ndarray      # Total uncertainty (sqrt of sum of variances)
    coverage_mask: np.ndarray             # True where data is adequate
    
    def get_dominant_source(self) -> np.ndarray:
        """Return array indicating dominant uncertainty source at each point."""
        sources = np.stack([
            self.rainfall_uncertainty,
            self.observation_uncertainty, 
            self.structural_uncertainty,
        ], axis=-1)
        dominant_idx = np.argmax(sources, axis=-1)
        labels = np.array(["rainfall", "observation", "structural"])
        return labels[dominant_idx]


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty model."""
    
    # Rainfall uncertainty parameters
    rainfall_missing_penalty: float = 2.0      # Variance multiplier for missing rainfall
    rainfall_sparse_threshold: int = 3         # Min nearby valid readings
    rainfall_spatial_decay: float = 0.5        # Distance decay for interpolation uncertainty
    
    # Observation uncertainty parameters
    complaint_sparse_penalty: float = 1.5      # Variance multiplier for sparse complaints
    complaint_spatial_radius: int = 2          # Radius to check complaint density
    complaint_min_count: int = 1               # Min complaints in radius for "observed"
    complaint_temporal_decay: float = 0.3      # Uncertainty growth without recent complaints
    
    # Structural uncertainty parameters
    extrapolation_penalty: float = 1.8         # Penalty when extrapolating beyond data
    boundary_uncertainty: float = 0.5          # Additional uncertainty at grid boundaries
    
    # Combination parameters
    base_variance: float = 0.1                 # Minimum variance floor
    max_variance: float = 10.0                 # Cap on total variance
    confidence_threshold: float = 0.5          # Below this = no-decision


class DataDrivenUncertaintyModel:
    """Heterogeneous, explainable uncertainty model.
    
    Key principle: Uncertainty must SPIKE when:
    - Rainfall data is missing
    - Complaints are sparse
    - Model is extrapolating
    
    If uncertainty is uniform → model is fake.
    """
    
    def __init__(self, config: Optional[UncertaintyConfig] = None) -> None:
        self.config = config or UncertaintyConfig()
        logger.info("DataDrivenUncertaintyModel initialized")
    
    def compute_uncertainty(
        self,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        complaints: np.ndarray,
        upstream_contribution: np.ndarray,
        prior_variance: np.ndarray,
    ) -> UncertaintyComponents:
        """Compute heterogeneous uncertainty from data quality.
        
        Args:
            rainfall_intensity: (time, y, x) rainfall field
            rainfall_accumulation: (time, y, x) accumulated rainfall
            complaints: (time, y, x) complaint counts (may have NaN)
            upstream_contribution: (y, x) upstream weighting
            prior_variance: (time, y, x) base variance from latent model
            
        Returns:
            UncertaintyComponents with spatially varying uncertainty
        """
        shape = rainfall_intensity.shape
        t, h, w = shape
        
        # 1. RAINFALL UNCERTAINTY - spikes where data missing/sparse
        rainfall_unc = self._compute_rainfall_uncertainty(
            rainfall_intensity, rainfall_accumulation
        )
        
        # 2. OBSERVATION UNCERTAINTY - spikes where complaints sparse
        obs_unc = self._compute_observation_uncertainty(complaints)
        
        # 3. STRUCTURAL UNCERTAINTY - spikes at boundaries, extrapolation
        struct_unc = self._compute_structural_uncertainty(
            upstream_contribution, shape
        )
        
        # 4. COMBINE - Sum variances, not standard deviations
        total_variance = (
            prior_variance
            + rainfall_unc ** 2
            + obs_unc ** 2
            + struct_unc ** 2
        )
        total_variance = np.clip(
            total_variance,
            self.config.base_variance,
            self.config.max_variance
        )
        combined_std = np.sqrt(total_variance)
        
        # 5. COVERAGE MASK - where we have adequate data
        coverage = (
            (rainfall_unc < self.config.rainfall_missing_penalty * 0.8) &
            (obs_unc < self.config.complaint_sparse_penalty * 0.8)
        )
        
        # Log heterogeneity statistics
        cv = np.std(combined_std) / (np.mean(combined_std) + 1e-9)
        logger.info(
            "Uncertainty CV=%.3f (should be >0.2 for heterogeneity), "
            "range=[%.3f, %.3f]",
            cv, combined_std.min(), combined_std.max()
        )
        
        if cv < 0.1:
            logger.warning(
                "Uncertainty is too uniform (CV=%.3f < 0.1)! "
                "This indicates synthetic/fake uncertainty.",
                cv
            )
        
        return UncertaintyComponents(
            rainfall_uncertainty=rainfall_unc,
            observation_uncertainty=obs_unc,
            structural_uncertainty=struct_unc,
            combined_uncertainty=combined_std,
            coverage_mask=coverage,
        )
    
    def _compute_rainfall_uncertainty(
        self,
        intensity: np.ndarray,
        accumulation: np.ndarray,
    ) -> np.ndarray:
        """Uncertainty from rainfall data quality.
        
        Spikes where:
        - NaN values present
        - Neighbors have very different values (interpolation)
        - Accumulation is inconsistent
        """
        shape = intensity.shape
        unc = np.zeros(shape, dtype=np.float32)
        
        # Missing data penalty
        missing_mask = np.isnan(intensity) | np.isnan(accumulation)
        unc[missing_mask] = self.config.rainfall_missing_penalty
        
        # Spatial sparsity penalty - check neighborhood variance
        for t in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    if missing_mask[t, i, j]:
                        continue
                    
                    # Get neighborhood
                    i_lo, i_hi = max(0, i-1), min(shape[1], i+2)
                    j_lo, j_hi = max(0, j-1), min(shape[2], j+2)
                    neighborhood = intensity[t, i_lo:i_hi, j_lo:j_hi]
                    
                    # Count valid neighbors
                    valid_neighbors = np.sum(~np.isnan(neighborhood))
                    
                    if valid_neighbors < self.config.rainfall_sparse_threshold:
                        # Sparse data - increase uncertainty
                        sparsity_penalty = (
                            self.config.rainfall_spatial_decay * 
                            (self.config.rainfall_sparse_threshold - valid_neighbors)
                        )
                        unc[t, i, j] += sparsity_penalty
                    
                    # Check for large gradient (interpolation zone)
                    if valid_neighbors > 1:
                        local_std = np.nanstd(neighborhood)
                        local_mean = np.nanmean(neighborhood) + 1e-9
                        if local_std / local_mean > 0.5:  # High local variation
                            unc[t, i, j] += self.config.rainfall_spatial_decay
        
        return unc.astype(np.float32)
    
    def _compute_observation_uncertainty(
        self,
        complaints: np.ndarray,
    ) -> np.ndarray:
        """Uncertainty from complaint observation quality.
        
        Spikes where:
        - No complaints in neighborhood (unobserved area)
        - NaN/missing complaint data
        - Long time since last complaint (temporal decay)
        """
        shape = complaints.shape
        t, h, w = shape
        unc = np.zeros(shape, dtype=np.float32)
        
        # Missing data
        missing = np.isnan(complaints)
        unc[missing] = self.config.complaint_sparse_penalty
        
        # Spatial sparsity - areas without nearby complaints
        complaints_safe = np.nan_to_num(complaints, nan=0.0)
        radius = self.config.complaint_spatial_radius
        
        for ti in range(t):
            for i in range(h):
                for j in range(w):
                    if missing[ti, i, j]:
                        continue
                    
                    # Count complaints in spatial window
                    i_lo, i_hi = max(0, i-radius), min(h, i+radius+1)
                    j_lo, j_hi = max(0, j-radius), min(w, j+radius+1)
                    
                    local_complaints = complaints_safe[ti, i_lo:i_hi, j_lo:j_hi]
                    total_local = np.sum(local_complaints)
                    
                    if total_local < self.config.complaint_min_count:
                        # Unobserved area - high uncertainty
                        unc[ti, i, j] += self.config.complaint_sparse_penalty
        
        # Temporal decay - uncertainty grows without recent observations
        last_complaint = np.full((h, w), -np.inf)
        for ti in range(t):
            for i in range(h):
                for j in range(w):
                    if complaints_safe[ti, i, j] > 0:
                        last_complaint[i, j] = ti
                    elif ti > 0:
                        time_since = ti - last_complaint[i, j]
                        if time_since > 0 and np.isfinite(time_since):
                            decay_penalty = (
                                self.config.complaint_temporal_decay * 
                                np.log1p(time_since)
                            )
                            unc[ti, i, j] += min(decay_penalty, 1.0)
        
        return unc.astype(np.float32)
    
    def _compute_structural_uncertainty(
        self,
        upstream: np.ndarray,
        full_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Structural model uncertainty.
        
        Spikes at:
        - Grid boundaries (edge effects)
        - Areas with extreme upstream values (extrapolation)
        - Low upstream confidence areas
        """
        t, h, w = full_shape
        unc = np.zeros(full_shape, dtype=np.float32)
        
        # Boundary uncertainty
        boundary_band = 1
        unc[:, :boundary_band, :] += self.config.boundary_uncertainty
        unc[:, -boundary_band:, :] += self.config.boundary_uncertainty
        unc[:, :, :boundary_band] += self.config.boundary_uncertainty
        unc[:, :, -boundary_band:] += self.config.boundary_uncertainty
        
        # Extrapolation penalty - extreme upstream values
        upstream_mean = np.nanmean(upstream)
        upstream_std = np.nanstd(upstream) + 1e-9
        
        # Z-score of upstream
        z_upstream = np.abs(upstream - upstream_mean) / upstream_std
        extrapolation_mask = z_upstream > 2.0  # Outside 2 sigma
        
        for ti in range(t):
            unc[ti][extrapolation_mask] += self.config.extrapolation_penalty
        
        # Missing upstream data
        if np.any(np.isnan(upstream)):
            upstream_missing = np.isnan(upstream)
            for ti in range(t):
                unc[ti][upstream_missing] += self.config.extrapolation_penalty
        
        return unc.astype(np.float32)
    
    def with_without_complaints_test(
        self,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        complaints: np.ndarray,
        upstream: np.ndarray,
        prior_variance: np.ndarray,
    ) -> Tuple[UncertaintyComponents, UncertaintyComponents, float]:
        """Sanity test: compare uncertainty WITH vs WITHOUT complaints.
        
        If they look similar → observation model is not influencing inference.
        That's the smoking gun for a fake model.
        
        Returns:
            (unc_with_complaints, unc_without_complaints, difference_ratio)
        """
        # With complaints
        unc_with = self.compute_uncertainty(
            rainfall_intensity, rainfall_accumulation,
            complaints, upstream, prior_variance
        )
        
        # Without complaints - all zeros
        empty_complaints = np.zeros_like(complaints)
        unc_without = self.compute_uncertainty(
            rainfall_intensity, rainfall_accumulation,
            empty_complaints, upstream, prior_variance
        )
        
        # Compute difference ratio
        diff = np.abs(
            unc_with.combined_uncertainty - unc_without.combined_uncertainty
        )
        mean_unc = (
            np.mean(unc_with.combined_uncertainty) + 
            np.mean(unc_without.combined_uncertainty)
        ) / 2 + 1e-9
        
        diff_ratio = np.mean(diff) / mean_unc
        
        if diff_ratio < 0.1:
            logger.error(
                "🚨 SANITY TEST FAILED: Uncertainty with/without complaints "
                "differs by only %.1f%%. Observation model is NOT influencing "
                "inference. The model is fake.",
                diff_ratio * 100
            )
        else:
            logger.info(
                "✓ Sanity test passed: Uncertainty differs by %.1f%% "
                "with/without complaints.",
                diff_ratio * 100
            )
        
        return unc_with, unc_without, diff_ratio


def compute_heterogeneous_uncertainty(
    rainfall_intensity: np.ndarray,
    rainfall_accumulation: np.ndarray,
    complaints: np.ndarray,
    upstream: np.ndarray,
    prior_variance: np.ndarray,
    config: Optional[UncertaintyConfig] = None,
) -> UncertaintyComponents:
    """Convenience function for data-driven uncertainty computation."""
    model = DataDrivenUncertaintyModel(config)
    return model.compute_uncertainty(
        rainfall_intensity, rainfall_accumulation,
        complaints, upstream, prior_variance
    )
