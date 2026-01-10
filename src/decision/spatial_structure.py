"""Realistic stress field generation with local anomalies.

Real urban stress fields are:
- Jagged, not smooth
- Spatially inconsistent
- Locally extreme
- Non-monotonic

Smooth gradients = fake data.

This module adds realistic spatial structure to stress fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import gamma

logger = logging.getLogger(__name__)


@dataclass
class SpatialStructureConfig:
    """Configuration for realistic spatial structure."""
    
    # Local anomaly parameters
    anomaly_probability: float = 0.15      # Prob of hotspot at each cell
    anomaly_intensity_mean: float = 1.5    # Mean intensity multiplier
    anomaly_intensity_std: float = 0.5     # Std of intensity multiplier
    
    # Spatial correlation parameters
    correlation_length: float = 1.5        # Spatial correlation length
    noise_level: float = 0.2               # Base noise level
    
    # Urban structure parameters
    drainage_convergence_points: int = 3   # Number of convergence hotspots
    convergence_intensity: float = 2.0     # Intensity at convergence
    convergence_decay: float = 0.5         # Decay from convergence points
    
    # Non-monotonicity parameters
    local_variation: float = 0.3           # Local cell-to-cell variation
    gradient_breaking: float = 0.2         # Prob of breaking smooth gradient


@dataclass
class RealisticStressResult:
    """Result of realistic stress computation."""
    
    stress: np.ndarray                     # Final stress field
    anomaly_mask: np.ndarray               # Where anomalies were added
    hotspot_locations: np.ndarray          # Convergence point locations
    spatial_roughness: float               # Measure of non-smoothness


class RealisticStressModel:
    """Generate realistic stress fields with local anomalies.
    
    Key principles (from Projectfile.md):
    1. Real stress fields are jagged, not smooth
    2. Must have local spikes and non-monotonic behavior
    3. Neighbors should sometimes disagree
    4. Smooth gradients = fake/synthetic data
    """
    
    def __init__(self, config: Optional[SpatialStructureConfig] = None) -> None:
        self.config = config or SpatialStructureConfig()
        self._rng = np.random.default_rng(42)
        logger.info("RealisticStressModel initialized")
    
    def add_spatial_structure(
        self,
        base_stress: np.ndarray,
        upstream: np.ndarray,
        complaints: Optional[np.ndarray] = None,
    ) -> RealisticStressResult:
        """Add realistic spatial structure to base stress field.
        
        Args:
            base_stress: (time, y, x) base stress from physics model
            upstream: (y, x) upstream contribution
            complaints: (time, y, x) optional complaint data to anchor hotspots
            
        Returns:
            RealisticStressResult with realistic spatial variations
        """
        shape = base_stress.shape
        t, h, w = shape
        cfg = self.config
        
        # 1. Add drainage convergence points (natural hotspots)
        convergence = self._add_convergence_points(h, w)
        
        # 2. Add local anomalies (random hotspots)
        anomalies, anomaly_mask = self._add_local_anomalies(shape)
        
        # 3. Add complaint-based anomalies if available
        if complaints is not None:
            complaint_hotspots = self._complaints_to_hotspots(complaints)
            anomalies = anomalies + complaint_hotspots
        
        # 4. Add spatial noise (cell-to-cell variation)
        noise = self._add_spatial_noise(shape)
        
        # 5. Break smooth gradients
        gradient_break = self._break_gradients(base_stress)
        
        # 6. Combine all effects
        stress = base_stress.copy()
        
        # Apply convergence effect
        for ti in range(t):
            stress[ti] = stress[ti] * (1 + cfg.convergence_intensity * convergence)
        
        # Apply anomalies
        stress = stress * (1 + anomalies)
        
        # Apply noise
        stress = stress + noise * np.nanmax(stress) * cfg.noise_level
        
        # Apply gradient breaking
        stress = stress * gradient_break
        
        # Ensure non-negative
        stress = np.maximum(stress, 0)
        
        # Compute roughness metric
        roughness = self._compute_roughness(stress)
        
        # Log diagnostics
        logger.info(
            "Spatial structure added: roughness=%.3f (>0.1 = realistic), "
            "%d anomalies, %d convergence points",
            roughness, np.sum(anomaly_mask), cfg.drainage_convergence_points
        )
        
        if roughness < 0.05:
            logger.warning(
                "⚠ Stress field is too smooth (roughness=%.3f). "
                "Consider increasing anomaly/noise parameters.",
                roughness
            )
        
        return RealisticStressResult(
            stress=stress.astype(np.float32),
            anomaly_mask=anomaly_mask,
            hotspot_locations=convergence,
            spatial_roughness=roughness,
        )
    
    def _add_convergence_points(self, h: int, w: int) -> np.ndarray:
        """Add drainage convergence points (natural hotspots).
        
        These represent points where multiple drainage lines converge,
        naturally creating stress concentration.
        """
        cfg = self.config
        convergence = np.zeros((h, w), dtype=np.float32)
        
        n_points = cfg.drainage_convergence_points
        
        # Place points with some randomness but avoid edges
        margin = max(1, min(h, w) // 5)
        points = []
        
        for _ in range(n_points):
            yi = self._rng.integers(margin, h - margin)
            xi = self._rng.integers(margin, w - margin)
            points.append((yi, xi))
        
        # Create distance-decayed influence
        yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
        
        for yi, xi in points:
            dist = np.sqrt((yy - yi) ** 2 + (xx - xi) ** 2)
            influence = np.exp(-cfg.convergence_decay * dist)
            convergence += influence
        
        # Normalize
        convergence = convergence / (convergence.max() + 1e-9)
        
        return convergence
    
    def _add_local_anomalies(
        self,
        shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add random local anomalies (hotspots and coldspots)."""
        cfg = self.config
        t, h, w = shape
        
        anomalies = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        
        # Generate random anomaly locations
        anomaly_prob = self._rng.random(shape)
        is_anomaly = anomaly_prob < cfg.anomaly_probability
        
        # Assign intensities
        intensities = self._rng.normal(
            cfg.anomaly_intensity_mean,
            cfg.anomaly_intensity_std,
            shape
        )
        
        # Some anomalies are negative (stress relief)
        sign = self._rng.choice([-1, 1], shape, p=[0.2, 0.8])
        intensities = intensities * sign
        
        anomalies[is_anomaly] = intensities[is_anomaly]
        mask[is_anomaly] = True
        
        return anomalies, mask
    
    def _complaints_to_hotspots(
        self,
        complaints: np.ndarray,
    ) -> np.ndarray:
        """Convert complaint locations to stress hotspots.
        
        Complaints indicate real-world stress - use them to anchor hotspots.
        """
        complaints_safe = np.nan_to_num(complaints, nan=0.0)
        
        # Complaints above median indicate hotspots
        median_c = np.median(complaints_safe[complaints_safe > 0]) if np.any(complaints_safe > 0) else 0
        
        hotspot = np.zeros_like(complaints, dtype=np.float32)
        if median_c > 0:
            # Scale by complaint intensity
            hotspot = np.where(
                complaints_safe > median_c,
                complaints_safe / (median_c + 1e-9) - 1,
                0
            )
            hotspot = np.clip(hotspot, 0, 2)
        
        return hotspot
    
    def _add_spatial_noise(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Add spatially correlated noise."""
        cfg = self.config
        t, h, w = shape
        
        noise = np.zeros(shape, dtype=np.float32)
        
        for ti in range(t):
            # Generate white noise
            white = self._rng.standard_normal((h, w))
            
            # Apply spatial correlation via Gaussian smoothing
            correlated = ndimage.gaussian_filter(
                white, sigma=cfg.correlation_length
            )
            
            # Add local variation (uncorrelated noise)
            local = self._rng.standard_normal((h, w)) * cfg.local_variation
            
            noise[ti] = correlated + local
        
        return noise.astype(np.float32)
    
    def _break_gradients(self, stress: np.ndarray) -> np.ndarray:
        """Break smooth gradients by introducing discontinuities."""
        cfg = self.config
        shape = stress.shape
        
        breaker = np.ones(shape, dtype=np.float32)
        
        # Random cells get gradient breaking
        break_mask = self._rng.random(shape) < cfg.gradient_breaking
        
        # Break intensity
        break_factor = self._rng.uniform(0.7, 1.3, shape)
        
        breaker[break_mask] = break_factor[break_mask]
        
        return breaker
    
    def _compute_roughness(self, stress: np.ndarray) -> float:
        """Compute spatial roughness metric.
        
        Roughness = mean absolute local gradient / mean value
        Higher = more jagged, realistic
        Lower = too smooth, synthetic
        """
        # Compute spatial gradients
        grad_y = np.abs(np.diff(stress, axis=1))
        grad_x = np.abs(np.diff(stress, axis=2))
        
        mean_grad = (np.nanmean(grad_y) + np.nanmean(grad_x)) / 2
        mean_val = np.nanmean(stress) + 1e-9
        
        roughness = mean_grad / mean_val
        
        return float(roughness)


def add_realistic_spatial_structure(
    base_stress: np.ndarray,
    upstream: np.ndarray,
    complaints: Optional[np.ndarray] = None,
    config: Optional[SpatialStructureConfig] = None,
) -> RealisticStressResult:
    """Convenience function for adding realistic spatial structure."""
    model = RealisticStressModel(config)
    return model.add_spatial_structure(base_stress, upstream, complaints)
