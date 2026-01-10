"""End-to-end inference engine with uncertainty propagation (Prompt-7)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config.parameters import (
    InferenceConfig,
    get_default_parameters,
)
from ..decision.latent_stress_model import (
    LatentStressModel,
    LatentStressResult,
)
from ..decision.observation_model import (
    ObservationLikelihoodResult,
    ObservationModel,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    reliability: np.ndarray
    no_decision: np.ndarray
    observation_loglik: np.ndarray
    notes: list[str]


class StressInferenceEngine:
    """Combine latent stress prior and complaint likelihood into posterior.

    Implements deterministic, explainable updating with explicit uncertainty
    and reliability/no-decision handling.
    """

    def __init__(
        self,
        latent_model: Optional[LatentStressModel] = None,
        observation_model: Optional[ObservationModel] = None,
        config: Optional[InferenceConfig] = None,
    ) -> None:
        self.latent_model = latent_model or LatentStressModel()
        self.observation_model = observation_model or ObservationModel()
        self.config = config or get_default_parameters().inference

    def infer(
        self,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        upstream_contribution: np.ndarray,
        complaints: np.ndarray,
        upstream_confidence: Optional[np.ndarray] = None,
        spatial_bias: Optional[np.ndarray] = None,
        temporal_modifier: Optional[np.ndarray] = None,
    ) -> InferenceResult:
        """Run inference: latent prior -> likelihood -> posterior.

        Shapes:
        - rainfall_* and complaints: (time, y, x)
        - upstream_contribution: (y, x)
        """

        latent: LatentStressResult = self.latent_model.infer(
            rainfall_intensity,
            rainfall_accumulation,
            upstream_contribution,
            upstream_confidence=upstream_confidence,
        )

        obs: ObservationLikelihoodResult = (
            self.observation_model.infer_likelihood(
                latent.mean,
                complaints,
                spatial_bias=spatial_bias,
                temporal_modifier=temporal_modifier,
            )
        )

        posterior_mean, posterior_var, no_decision = self._fuse(latent, obs)

        notes = []
        if not latent.reliability.all():
            frac = float((~latent.reliability).mean())
            notes.append(f"Latent reliability low in {frac:.2%} of cells")
            logger.warning(
                "Latent reliability low in %.2f%% of cells", frac * 100
            )
        if not obs.confidence.all():
            frac = float((~obs.confidence).mean())
            notes.append(f"Observation confidence low in {frac:.2%} of bins")
            logger.warning(
                "Observation confidence low in %.2f%% of bins", frac * 100
            )

        return InferenceResult(
            posterior_mean=posterior_mean,
            posterior_variance=posterior_var,
            reliability=latent.reliability & obs.confidence,
            no_decision=no_decision,
            observation_loglik=obs.log_likelihood,
            notes=notes + latent.notes + obs.notes,
        )

    def _fuse(
        self,
        latent: LatentStressResult,
        obs: ObservationLikelihoodResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fuse prior with observation likelihood via precision addition."""

        prior_mean = latent.mean
        prior_var = latent.variance
        prior_precision = 1.0 / np.maximum(prior_var, 1e-6)

        # Derive pseudo-precision from observations: negative Hessian proxy.
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
            post_precision
            < self.config.min_event_duration_minutes * 1e-3
        )

        # Temporal consistency check: flag large jumps
        jumps = np.abs(np.diff(posterior_mean, axis=0))
        jump_flag = jumps > self.config.anomaly_zscore
        if jump_flag.any():
            frac = float(jump_flag.mean())
            logger.warning(
                "Temporal jumps flagged in %.2f%% of cells", frac * 100
            )

        return (
            posterior_mean.astype(np.float32),
            posterior_var.astype(np.float32),
            no_decision.astype(bool),
        )
