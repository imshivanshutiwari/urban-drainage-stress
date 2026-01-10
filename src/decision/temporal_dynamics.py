"""Realistic temporal dynamics: lag, hysteresis, asymmetric decay.

Real systems show:
- Lag: stress peaks AFTER rainfall peak
- Hysteresis: decay is slower than rise
- Asymmetry: some areas decay faster than others
- Uneven: min/max not parallel

Symmetric/clean curves = hand-shaped, not inferred.

This module adds realistic temporal dynamics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class TemporalDynamicsConfig:
    """Configuration for realistic temporal dynamics."""
    
    # Lag parameters
    base_lag_steps: int = 1                # Base lag between rainfall and stress
    lag_variation: float = 0.5             # Spatial variation in lag
    upstream_lag_factor: float = 0.3       # Additional lag for downstream areas
    
    # Hysteresis parameters
    rise_rate: float = 0.8                 # How fast stress rises
    decay_rate: float = 0.4                # How fast stress decays (slower!)
    decay_variation: float = 0.3           # Spatial variation in decay
    
    # Asymmetry parameters
    asymmetry_factor: float = 0.3          # How asymmetric the response is
    local_persistence: float = 0.2         # Local memory/persistence
    
    # Uncertainty during transitions
    transition_uncertainty: float = 0.5    # Extra uncertainty during rise/fall


@dataclass
class TemporalDynamicsResult:
    """Result of temporal dynamics computation."""
    
    stress: np.ndarray                     # Stress with realistic dynamics
    uncertainty: np.ndarray                # Time-varying uncertainty
    lag_field: np.ndarray                  # Spatial lag map
    is_rising: np.ndarray                  # Where stress is rising
    is_falling: np.ndarray                 # Where stress is falling
    peak_time: np.ndarray                  # Time of peak at each location
    temporal_asymmetry: float              # Measure of rise/fall asymmetry


class RealisticTemporalModel:
    """Generate realistic temporal dynamics with lag and hysteresis.
    
    Key principles (from Projectfile.md):
    1. Stress should peak AFTER rainfall peak
    2. Decay should be SLOWER than rise
    3. Min/max should NOT be parallel
    4. Uncertainty should be WIDER during transitions
    """
    
    def __init__(self, config: Optional[TemporalDynamicsConfig] = None) -> None:
        self.config = config or TemporalDynamicsConfig()
        self._rng = np.random.default_rng(42)
        logger.info("RealisticTemporalModel initialized")
    
    def apply_temporal_dynamics(
        self,
        base_stress: np.ndarray,
        rainfall: np.ndarray,
        upstream: np.ndarray,
        base_uncertainty: np.ndarray,
    ) -> TemporalDynamicsResult:
        """Apply realistic temporal dynamics to stress field.
        
        Args:
            base_stress: (time, y, x) instantaneous stress
            rainfall: (time, y, x) rainfall intensity
            upstream: (y, x) upstream contribution
            base_uncertainty: (time, y, x) base uncertainty
            
        Returns:
            TemporalDynamicsResult with realistic dynamics
        """
        shape = base_stress.shape
        t, h, w = shape
        cfg = self.config
        
        # 1. Compute spatially-varying lag field
        lag_field = self._compute_lag_field(upstream)
        
        # 2. Apply lag to stress response
        lagged_stress = self._apply_lag(base_stress, lag_field)
        
        # 3. Apply hysteresis (asymmetric rise/decay)
        hysteresis_stress, is_rising, is_falling = self._apply_hysteresis(
            lagged_stress, rainfall
        )
        
        # 4. Add local persistence (memory effects)
        final_stress = self._add_local_persistence(hysteresis_stress)
        
        # 5. Compute peak times
        peak_time = self._compute_peak_times(final_stress)
        
        # 6. Adjust uncertainty during transitions
        adjusted_uncertainty = self._adjust_transition_uncertainty(
            base_uncertainty, is_rising, is_falling
        )
        
        # 7. Compute asymmetry metric
        asymmetry = self._compute_asymmetry(final_stress)
        
        # Log diagnostics
        rain_peak_t = np.argmax(np.nanmean(rainfall, axis=(1, 2)))
        stress_peak_t = np.argmax(np.nanmean(final_stress, axis=(1, 2)))
        
        logger.info(
            "Temporal dynamics: rainfall peaks at t=%d, stress peaks at t=%d, "
            "lag=%d steps, asymmetry=%.3f",
            rain_peak_t, stress_peak_t, stress_peak_t - rain_peak_t, asymmetry
        )
        
        if stress_peak_t <= rain_peak_t:
            logger.warning(
                "⚠ Stress peaks at same time or BEFORE rainfall (lag=%d). "
                "This is physically unrealistic!",
                stress_peak_t - rain_peak_t
            )
        
        return TemporalDynamicsResult(
            stress=final_stress.astype(np.float32),
            uncertainty=adjusted_uncertainty.astype(np.float32),
            lag_field=lag_field.astype(np.float32),
            is_rising=is_rising,
            is_falling=is_falling,
            peak_time=peak_time,
            temporal_asymmetry=asymmetry,
        )
    
    def _compute_lag_field(self, upstream: np.ndarray) -> np.ndarray:
        """Compute spatially-varying lag based on upstream contribution.
        
        Areas with higher upstream contribution (downstream) should have
        MORE lag as water takes time to travel.
        """
        cfg = self.config
        h, w = upstream.shape
        
        # Normalize upstream
        up_norm = upstream / (np.nanmax(upstream) + 1e-9)
        
        # Base lag + variation based on position
        base_lag = np.full((h, w), cfg.base_lag_steps, dtype=np.float32)
        
        # Add upstream-based lag
        upstream_lag = up_norm * cfg.upstream_lag_factor * 2
        
        # Add random spatial variation
        variation = self._rng.uniform(
            -cfg.lag_variation,
            cfg.lag_variation,
            (h, w)
        )
        
        lag_field = base_lag + upstream_lag + variation
        lag_field = np.maximum(lag_field, 0)  # No negative lag
        
        return lag_field
    
    def _apply_lag(
        self,
        stress: np.ndarray,
        lag_field: np.ndarray,
    ) -> np.ndarray:
        """Apply spatially-varying lag to stress field."""
        t, h, w = stress.shape
        lagged = np.zeros_like(stress)
        
        for i in range(h):
            for j in range(w):
                lag = int(round(lag_field[i, j]))
                if lag >= t:
                    lag = t - 1
                
                # Shift stress backward in time by lag
                if lag > 0:
                    lagged[lag:, i, j] = stress[:-lag, i, j]
                    lagged[:lag, i, j] = stress[0, i, j]  # Fill start
                else:
                    lagged[:, i, j] = stress[:, i, j]
        
        return lagged
    
    def _apply_hysteresis(
        self,
        stress: np.ndarray,
        rainfall: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply hysteresis: slow decay, faster rise.
        
        Key: decay_rate < rise_rate
        """
        cfg = self.config
        t, h, w = stress.shape
        
        result = np.zeros_like(stress)
        result[0] = stress[0]
        
        is_rising = np.zeros(stress.shape, dtype=bool)
        is_falling = np.zeros(stress.shape, dtype=bool)
        
        # Spatially varying decay rates
        decay_variation = self._rng.uniform(
            1 - cfg.decay_variation,
            1 + cfg.decay_variation,
            (h, w)
        )
        local_decay = cfg.decay_rate * decay_variation
        
        for ti in range(1, t):
            # Current target
            target = stress[ti]
            prev = result[ti - 1]
            
            # Determine rising or falling
            rising = target > prev
            falling = target < prev
            
            is_rising[ti] = rising
            is_falling[ti] = falling
            
            # Apply asymmetric rates
            rate = np.where(
                rising,
                cfg.rise_rate,
                local_decay
            )
            
            # Exponential approach to target
            result[ti] = prev + rate * (target - prev)
        
        return result, is_rising, is_falling
    
    def _add_local_persistence(self, stress: np.ndarray) -> np.ndarray:
        """Add local memory/persistence effects.
        
        Some areas "remember" stress longer than others.
        """
        cfg = self.config
        t, h, w = stress.shape
        
        # Persistence map (some areas are sticky)
        persistence = self._rng.uniform(
            0,
            cfg.local_persistence * 2,
            (h, w)
        )
        
        result = stress.copy()
        
        for ti in range(1, t):
            # Mix current with previous based on persistence
            result[ti] = (
                (1 - persistence) * stress[ti] +
                persistence * result[ti - 1]
            )
        
        return result
    
    def _compute_peak_times(self, stress: np.ndarray) -> np.ndarray:
        """Compute time of peak at each spatial location."""
        t, h, w = stress.shape
        peak_time = np.zeros((h, w), dtype=np.int32)
        
        for i in range(h):
            for j in range(w):
                peak_time[i, j] = np.argmax(stress[:, i, j])
        
        return peak_time
    
    def _adjust_transition_uncertainty(
        self,
        base_uncertainty: np.ndarray,
        is_rising: np.ndarray,
        is_falling: np.ndarray,
    ) -> np.ndarray:
        """Increase uncertainty during transitions.
        
        Model is less certain during rapid changes.
        """
        cfg = self.config
        
        # Transitions get extra uncertainty
        in_transition = is_rising | is_falling
        
        transition_boost = np.where(
            in_transition,
            1 + cfg.transition_uncertainty,
            1.0
        )
        
        return base_uncertainty * transition_boost
    
    def _compute_asymmetry(self, stress: np.ndarray) -> float:
        """Compute temporal asymmetry metric.
        
        Asymmetry = |mean_rise_rate - mean_decay_rate| / mean_rate
        Higher = more realistic hysteresis
        """
        t, h, w = stress.shape
        
        # Compute temporal differences
        diff = np.diff(stress, axis=0)
        
        rising = diff > 0
        falling = diff < 0
        
        mean_rise = np.nanmean(np.abs(diff[rising])) if rising.any() else 0
        mean_fall = np.nanmean(np.abs(diff[falling])) if falling.any() else 0
        
        mean_rate = (mean_rise + mean_fall) / 2 + 1e-9
        asymmetry = np.abs(mean_rise - mean_fall) / mean_rate
        
        return float(asymmetry)


def apply_realistic_temporal_dynamics(
    base_stress: np.ndarray,
    rainfall: np.ndarray,
    upstream: np.ndarray,
    base_uncertainty: np.ndarray,
    config: Optional[TemporalDynamicsConfig] = None,
) -> TemporalDynamicsResult:
    """Convenience function for applying temporal dynamics."""
    model = RealisticTemporalModel(config)
    return model.apply_temporal_dynamics(
        base_stress, rainfall, upstream, base_uncertainty
    )
