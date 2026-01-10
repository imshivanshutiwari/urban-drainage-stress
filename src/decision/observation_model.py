"""Probabilistic observation model linking complaints to latent stress."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config.parameters import (
    ObservationModelConfig,
    get_default_parameters,
)

logger = logging.getLogger(__name__)


@dataclass
class ObservationLikelihoodResult:
    """Outputs of the observation model."""

    rate: np.ndarray
    log_likelihood: np.ndarray
    likelihood: np.ndarray
    confidence: np.ndarray
    notes: list[str]


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _lognormal_params(mean: float, sd: float) -> tuple[float, float]:
    var = sd ** 2
    mu = math.log((mean ** 2) / math.sqrt(var + mean ** 2))
    sigma = math.sqrt(math.log(1 + var / (mean ** 2)))
    return mu, sigma


class ObservationModel:
    """Complaint observation likelihood with delay and bias (Prompt-6).

    - Complaints are noisy, delayed, biased observations of latent stress.
    - Uses a delay kernel to align stress with reporting times.
    - Reporting probability varies spatially/temporally via bias factors.
    """

    def __init__(
        self,
        config: Optional[ObservationModelConfig] = None,
        dt_minutes: int = 60,
    ) -> None:
        self.config = config or get_default_parameters().observation
        self.dt_minutes = dt_minutes
        logger.info(
            "ObservationModel init: base_prob=%.2f, "
            "delay_mean=%.2f h, sd=%.2f",
            self.config.base_reporting_prob,
            self.config.delay_mean_hours,
            self.config.delay_sd_hours,
        )

    def _delay_kernel(self, steps: int) -> np.ndarray:
        max_steps = int(
            self.config.delay_max_hours * 60 / self.dt_minutes
        )
        steps = min(steps, max_steps + 1)
        hours = np.arange(steps) * (self.dt_minutes / 60)
        mu, sigma = _lognormal_params(
            self.config.delay_mean_hours, self.config.delay_sd_hours
        )
        with np.errstate(divide="ignore"):
            pdf = (
                1
                / (hours * sigma * math.sqrt(2 * math.pi) + 1e-9)
                * np.exp(
                    -((np.log(hours + 1e-9) - mu) ** 2) / (2 * sigma ** 2)
                )
            )
        pdf[hours > self.config.delay_max_hours] = 0.0
        kernel = pdf / (pdf.sum() + 1e-9)
        return kernel.astype(np.float32)

    def _align_stress(self, stress: np.ndarray) -> np.ndarray:
        kernel = self._delay_kernel(stress.shape[0])
        aligned = np.zeros_like(stress)
        for t in range(stress.shape[0]):
            k_max = min(t + 1, kernel.size)
            weights = kernel[:k_max]
            idx = np.arange(k_max)
            aligned[t] = np.sum(
                weights[:, None, None] * stress[t - idx], axis=0
            )
        return aligned

    def _reporting_prob(
        self,
        base_shape: tuple[int, int, int],
        spatial_bias: Optional[np.ndarray],
        temporal_modifier: Optional[np.ndarray],
    ) -> np.ndarray:
        prob = np.full(base_shape, self.config.base_reporting_prob)
        if spatial_bias is not None:
            if spatial_bias.shape != base_shape[1:]:
                raise ValueError("spatial_bias must match spatial grid")
            prob *= spatial_bias[np.newaxis, ...]
        if temporal_modifier is not None:
            if temporal_modifier.shape[0] != base_shape[0]:
                raise ValueError("temporal_modifier must match time axis")
            prob *= temporal_modifier[:, np.newaxis, np.newaxis]
        prob = np.clip(
            prob,
            self.config.min_reporting_prob,
            self.config.max_reporting_prob,
        )
        return prob

    def infer_likelihood(
        self,
        stress: np.ndarray,
        complaints: np.ndarray,
        spatial_bias: Optional[np.ndarray] = None,
        temporal_modifier: Optional[np.ndarray] = None,
    ) -> ObservationLikelihoodResult:
        """Compute complaint likelihood given latent stress.

        - stress, complaints: shape (time, y, x)
        - complaints may be sparse/zero; NaNs are treated as missing.
        """

        if stress.shape != complaints.shape:
            raise ValueError("stress and complaints must have same shape")

        stress = np.asarray(stress, dtype=float)
        complaints = np.asarray(complaints, dtype=float)

        aligned_stress = self._align_stress(stress)
        report_prob = self._reporting_prob(
            stress.shape, spatial_bias, temporal_modifier
        )

        intensity = _softplus(aligned_stress) + self.config.rate_floor
        rate = report_prob * intensity

        eps = 1e-9
        log_rate = np.log(rate + eps)
        log_fact = np.vectorize(math.lgamma)(complaints + 1)
        log_likelihood = complaints * log_rate - rate - log_fact
        likelihood = np.exp(log_likelihood)

        missing = np.isnan(complaints)
        log_likelihood = np.where(missing, np.nan, log_likelihood)
        likelihood = np.where(missing, np.nan, likelihood)

        low_cover = report_prob < self.config.coverage_threshold
        confidence = (~missing) & (~low_cover)

        notes: list[str] = []
        if missing.any():
            frac = float(missing.mean())
            notes.append(f"Missing complaints in {frac:.2%} of bins")
            logger.warning("Missing complaints in %.2f%% of bins", frac * 100)
        if low_cover.any():
            frac = float(low_cover.mean())
            notes.append(f"Low reporting coverage in {frac:.2%} of bins")
            logger.warning(
                "Low reporting coverage in %.2f%% of bins", frac * 100
            )
        notes.append(
            "Delay modeled with lognormal kernel; probabilities normalized"
        )

        return ObservationLikelihoodResult(
            rate=rate.astype(np.float32),
            log_likelihood=log_likelihood.astype(np.float32),
            likelihood=likelihood.astype(np.float32),
            confidence=confidence.astype(bool),
            notes=notes,
        )
