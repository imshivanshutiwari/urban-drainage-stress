"""Causal, probabilistic latent drainage stress model (Prompt-5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config.parameters import LatentStressModelConfig, get_default_parameters

logger = logging.getLogger(__name__)


@dataclass
class LatentStressResult:
    """Outputs of the latent stress model."""

    mean: np.ndarray
    variance: np.ndarray
    reliability: np.ndarray
    notes: list[str]


class LatentStressModel:
    """Estimate latent drainage stress with uncertainty and reliability flags.

    The model is linear-in-features with optional temporal memory. It is a
    structural prior, not a predictive classifier. Uncertainty is propagated
    through missing-data penalties and variance evolution.
    """

    def __init__(
        self, config: Optional[LatentStressModelConfig] = None
    ) -> None:
        self.config = config or get_default_parameters().latent_model
        logger.info(
            "LatentStressModel initialized with weights: intensity=%.2f, "
            "accumulation=%.2f, upstream=%.2f, memory_decay=%.2f",
            self.config.weight_intensity,
            self.config.weight_accumulation,
            self.config.weight_upstream,
            self.config.memory_decay,
        )

    def _validate_shapes(
        self,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        upstream_contribution: np.ndarray,
    ) -> None:
        if rainfall_intensity.shape != rainfall_accumulation.shape:
            raise ValueError("Rainfall intensity and accumulation must align")
        if rainfall_intensity.ndim != 3:
            raise ValueError("Rainfall inputs must be (time, y, x)")
        if upstream_contribution.shape != rainfall_intensity.shape[1:]:
            raise ValueError("Upstream contribution must match spatial grid")

    def _apply_memory(self, raw: np.ndarray) -> np.ndarray:
        decay = self.config.memory_decay
        if decay <= 0:
            return raw
        out = np.empty_like(raw)
        out[0] = raw[0]
        keep = 1.0 - decay
        for t in range(1, raw.shape[0]):
            out[t] = decay * out[t - 1] + keep * raw[t]
        return out

    def _propagate_variance(self, var_raw: np.ndarray) -> np.ndarray:
        decay = self.config.memory_decay
        if decay <= 0:
            return np.maximum(var_raw, self.config.variance_floor)
        out = np.empty_like(var_raw)
        out[0] = var_raw[0]
        keep = 1.0 - decay
        for t in range(1, var_raw.shape[0]):
            out[t] = (decay ** 2) * out[t - 1] + (keep ** 2) * var_raw[t]
        return np.maximum(out, self.config.variance_floor)

    def infer(
        self,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        upstream_contribution: np.ndarray,
        upstream_confidence: Optional[np.ndarray] = None,
    ) -> LatentStressResult:
        """Infer latent stress field with uncertainty and reliability mask.

        Inputs are vectorized fields; missing values (NaN) are handled via
        reliability penalties and variance inflation.
        """

        self._validate_shapes(
            rainfall_intensity, rainfall_accumulation, upstream_contribution
        )

        ri = np.asarray(rainfall_intensity, dtype=float)
        ra = np.asarray(rainfall_accumulation, dtype=float)
        uc = np.asarray(upstream_contribution, dtype=float)
        
        # Use WINDOWED accumulation (last 3 timesteps) instead of total cumulative
        # This prevents stress from monotonically increasing
        window_size = min(3, ri.shape[0])
        ra_windowed = np.zeros_like(ra)
        for t in range(ra.shape[0]):
            t_start = max(0, t - window_size + 1)
            ra_windowed[t] = ri[t_start:t+1].sum(axis=0)

        if upstream_confidence is not None:
            uc_conf = np.asarray(upstream_confidence, dtype=float)
            if uc_conf.shape != uc.shape:
                raise ValueError("Upstream confidence must match spatial grid")
        else:
            uc_conf = np.ones_like(uc)

        # Handle missing data: mask then fill zeros for computation
        feat_stack = np.stack([ri, ra_windowed], axis=0)
        missing = np.isnan(feat_stack)
        valid_frac = 1.0 - missing.mean(axis=0)
        ri = np.nan_to_num(ri, nan=0.0)
        ra_windowed = np.nan_to_num(ra_windowed, nan=0.0)
        uc_safe = np.nan_to_num(uc, nan=0.0)

        # Stress is driven by current intensity + recent windowed rainfall + terrain
        base = (
            self.config.intercept
            + self.config.weight_intensity * ri
            + self.config.weight_accumulation * ra_windowed  # Use windowed, not cumulative
        )
        base += self.config.weight_upstream * uc_safe[np.newaxis, ...]

        var_raw = np.full_like(base, self.config.observation_noise_sd ** 2)
        var_raw *= 1.0 + self.config.missing_variance_inflation * (
            1.0 - valid_frac
        )

        mean = self._apply_memory(base)
        variance = self._propagate_variance(var_raw)

        reliability = (valid_frac >= self.config.missing_data_tolerance) & (
            uc_conf >= self.config.upstream_confidence_floor
        )
        reliability = reliability.astype(bool)

        notes: list[str] = []
        degraded = np.logical_not(reliability)
        if degraded.any():
            frac_bad = float(degraded.mean())
            notes.append(f"Reliability degraded in {frac_bad:.2%} of cells")
            logger.warning(
                "Reliability degraded in %.2f%% of grid cells", frac_bad * 100
            )
        if self.config.memory_decay > 0:
            notes.append(
                "Temporal memory applied with "
                f"decay={self.config.memory_decay:.2f}"
            )
        else:
            notes.append("No temporal memory; instantaneous response assumed")

        return LatentStressResult(
            mean=mean.astype(np.float32),
            variance=variance.astype(np.float32),
            reliability=reliability,
            notes=notes,
        )
