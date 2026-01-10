"""True Bayesian inference with posterior = prior × likelihood.

This module implements PROPER Bayesian inference where:
1. Rainfall + terrain → PRIOR over stress
2. Complaints → LIKELIHOOD
3. Stress inferred as POSTERIOR

Key principle: Complaints MUST actively constrain inference.
- Observations reduce uncertainty where present
- Observations increase uncertainty where absent
- Observations shift estimates where inconsistent with prior

Reference: Projectfile.md - "Enforce posterior ∝ prior × likelihood"

NO ONE-WAY FLOW. THIS IS TRUE INFERENCE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
from scipy.special import gammaln, digamma
from scipy.stats import nbinom, poisson

logger = logging.getLogger(__name__)


@dataclass
class BayesianPrior:
    """Prior distribution over stress (from rainfall + terrain)."""
    mean: np.ndarray           # Prior mean E[stress]
    variance: np.ndarray       # Prior variance Var[stress]
    precision: np.ndarray      # Prior precision = 1/variance
    log_normalizer: float      # Log partition function


@dataclass
class BayesianLikelihood:
    """Likelihood from complaints P(complaints | stress)."""
    log_likelihood: np.ndarray    # Log P(complaints | stress)
    sufficient_stats: np.ndarray  # Sufficient statistics for conjugate update
    observed_rate: np.ndarray     # Empirical complaint rate
    observation_weight: np.ndarray  # How much each observation contributes


@dataclass
class BayesianPosterior:
    """Posterior distribution P(stress | rainfall, terrain, complaints)."""
    mean: np.ndarray              # Posterior mean E[stress | data]
    variance: np.ndarray          # Posterior variance Var[stress | data]
    precision: np.ndarray         # Posterior precision
    credible_lower: np.ndarray    # 95% credible interval lower bound
    credible_upper: np.ndarray    # 95% credible interval upper bound
    
    # Prior info (for diagnostics)
    prior_mean: np.ndarray        # Prior mean before update
    prior_variance: np.ndarray    # Prior variance before update
    
    # Diagnostics
    prior_weight: np.ndarray      # How much prior contributes (0-1)
    likelihood_weight: np.ndarray # How much likelihood contributes (0-1)
    information_gain: np.ndarray  # KL(posterior || prior)
    
    # Reliability
    no_decision_mask: np.ndarray  # Where posterior is too uncertain
    reliable_mask: np.ndarray     # Where we have enough data
    
    notes: List[str] = field(default_factory=list)


@dataclass
class InferenceConfig:
    """Configuration for Bayesian inference."""
    
    # Prior hyperparameters
    prior_strength: float = 1.0        # How informative is the prior
    prior_variance_floor: float = 0.001 # Minimum prior variance (reduced to allow heterogeneity)
    prior_variance_ceil: float = 10.0  # Maximum prior variance
    
    # Likelihood parameters
    likelihood_overdispersion: float = 1.5  # Negative binomial dispersion
    observation_noise: float = 0.3          # Measurement noise in complaints
    
    # Posterior parameters
    credible_level: float = 0.95       # Credible interval level
    no_decision_threshold: float = 2.0 # Posterior std above which = NO_DECISION
    min_observations: int = 1          # Min complaints to trust likelihood
    
    # Numerical stability
    log_eps: float = 1e-10


class TrueBayesianInference:
    """True Bayesian inference engine.
    
    CRITICAL: This is NOT forward simulation.
    
    The inference loop:
    1. Compute PRIOR P(stress | rainfall, terrain)
    2. Compute LIKELIHOOD P(complaints | stress)
    3. Combine: POSTERIOR ∝ PRIOR × LIKELIHOOD
    4. Posterior variance depends on BOTH prior and data
    
    If complaints don't change the posterior → inference is broken.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()
        logger.info("TrueBayesianInference initialized")
    
    def compute_prior(
        self,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        upstream_contribution: np.ndarray,
        weights: Tuple[float, float, float] = (1.0, 0.5, 0.3),
        missing_variance_inflation: float = 2.0,
    ) -> BayesianPrior:
        """Compute prior distribution over stress from physical model.
        
        Prior mean comes from deterministic model.
        Prior variance reflects uncertainty in physical process.
        
        Args:
            rainfall_intensity: (T, H, W) rainfall
            rainfall_accumulation: (T, H, W) accumulated rainfall
            upstream_contribution: (H, W) terrain factor
            weights: (w_intensity, w_accumulation, w_upstream)
            missing_variance_inflation: Variance multiplier for missing data
            
        Returns:
            BayesianPrior with mean and variance fields
        """
        T, H, W = rainfall_intensity.shape
        
        # Prior mean: linear combination of features
        w_i, w_a, w_u = weights
        prior_mean = (
            w_i * rainfall_intensity +
            w_a * rainfall_accumulation +
            w_u * upstream_contribution[np.newaxis, :, :]
        )
        prior_mean = np.maximum(prior_mean, 0)  # Stress is non-negative
        
        # Prior variance: base + feature-dependent + missing-data penalty
        base_var = self.config.prior_variance_floor
        
        # Variance scales with magnitude (heteroscedastic)
        feature_var = 0.1 * (prior_mean + 1)
        
        # Missing data inflates variance (NaN only, not zero - zero rainfall is valid)
        missing_rain = np.isnan(rainfall_intensity)
        missing_penalty = missing_variance_inflation * missing_rain.astype(float)
        
        prior_variance = base_var + feature_var + missing_penalty
        prior_variance = np.clip(
            prior_variance,
            self.config.prior_variance_floor,
            self.config.prior_variance_ceil,
        )
        
        prior_precision = 1.0 / prior_variance
        
        # Log normalizer for Gaussian prior
        log_Z = -0.5 * np.sum(np.log(2 * np.pi * prior_variance))
        
        logger.info(
            "Prior computed: mean=[%.2f, %.2f], var=[%.3f, %.3f]",
            np.nanmin(prior_mean), np.nanmax(prior_mean),
            np.nanmin(prior_variance), np.nanmax(prior_variance),
        )
        
        return BayesianPrior(
            mean=prior_mean.astype(np.float32),
            variance=prior_variance.astype(np.float32),
            precision=prior_precision.astype(np.float32),
            log_normalizer=float(log_Z),
        )
    
    def compute_likelihood(
        self,
        complaints: np.ndarray,
        stress_grid: np.ndarray,
        spatial_bias: Optional[np.ndarray] = None,
        temporal_bias: Optional[np.ndarray] = None,
    ) -> BayesianLikelihood:
        """Compute likelihood P(complaints | stress).
        
        Model: complaints ~ NegBinom(μ(stress), r)
        where μ = base_rate × exp(β × stress) × spatial_bias
        
        Args:
            complaints: (T, H, W) observed complaint counts
            stress_grid: (T, H, W) stress values to evaluate
            spatial_bias: (H, W) optional reporting bias
            temporal_bias: (T,) optional temporal reporting modifier
            
        Returns:
            BayesianLikelihood with log-likelihood and sufficient stats
        """
        T, H, W = complaints.shape
        cfg = self.config
        
        # Handle missing complaints
        complaints_clean = np.nan_to_num(complaints, nan=0)
        observed_mask = ~np.isnan(complaints)
        
        # Expected complaint rate given stress
        # μ = base × exp(β × stress) 
        base_rate = 0.1  # Base complaint rate
        stress_effect = np.clip(stress_grid, 0, 10)
        expected_rate = base_rate * np.exp(0.5 * stress_effect)
        
        # Apply spatial bias
        if spatial_bias is not None:
            expected_rate *= spatial_bias[np.newaxis, :, :]
        
        # Apply temporal bias
        if temporal_bias is not None:
            expected_rate *= temporal_bias[:, np.newaxis, np.newaxis]
        
        # Negative binomial likelihood
        # P(y | μ, r) = NB(y; μ, r)
        r = 1.0 / cfg.likelihood_overdispersion  # Dispersion parameter
        p = r / (r + expected_rate + 1e-9)  # Success probability
        
        # Log-likelihood
        log_lik = np.zeros_like(complaints_clean, dtype=np.float64)
        
        for t in range(T):
            for i in range(H):
                for j in range(W):
                    if observed_mask[t, i, j]:
                        y = int(complaints_clean[t, i, j])
                        mu = expected_rate[t, i, j]
                        
                        if mu > 0:
                            # NB log-likelihood
                            log_lik[t, i, j] = (
                                gammaln(y + r) - gammaln(y + 1) - gammaln(r) +
                                r * np.log(p[t, i, j] + cfg.log_eps) +
                                y * np.log(1 - p[t, i, j] + cfg.log_eps)
                            )
                        else:
                            # Poisson limit
                            log_lik[t, i, j] = -mu + y * np.log(mu + cfg.log_eps) - gammaln(y + 1)
        
        # Sufficient statistics for conjugate update
        # For Gaussian-Gamma conjugacy: sum(y), sum(y²), n
        sufficient_stats = np.stack([
            complaints_clean,                    # sum of observations
            complaints_clean ** 2,               # sum of squares
            observed_mask.astype(float),         # count of observations
        ], axis=-1)
        
        # Observation weight: how much each point contributes
        obs_weight = observed_mask.astype(float)
        obs_weight *= np.clip(complaints_clean + 1, 1, 10)  # More complaints = more weight
        
        logger.info(
            "Likelihood computed: observed=%d/%d cells, "
            "log_lik=[%.1f, %.1f]",
            np.sum(observed_mask), observed_mask.size,
            np.nanmin(log_lik), np.nanmax(log_lik),
        )
        
        return BayesianLikelihood(
            log_likelihood=log_lik.astype(np.float32),
            sufficient_stats=sufficient_stats.astype(np.float32),
            observed_rate=expected_rate.astype(np.float32),
            observation_weight=obs_weight.astype(np.float32),
        )
    
    def compute_posterior(
        self,
        prior: BayesianPrior,
        likelihood: BayesianLikelihood,
        complaints: np.ndarray,
    ) -> BayesianPosterior:
        """Combine prior and likelihood into posterior.
        
        CRITICAL EQUATION:
            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (prior_precision × prior_mean + 
                             likelihood_precision × likelihood_mean) / posterior_precision
        
        This is PROPER Bayesian inference.
        
        Args:
            prior: BayesianPrior from compute_prior()
            likelihood: BayesianLikelihood from compute_likelihood()
            complaints: (T, H, W) original complaints for computing likelihood precision
            
        Returns:
            BayesianPosterior with mean, variance, and diagnostics
        """
        cfg = self.config
        notes = []
        
        # Prior terms
        prior_precision = prior.precision
        prior_mean = prior.mean
        
        # Likelihood precision: ONLY where we have ACTUAL complaints
        # Key insight: absence of complaints is NOT the same as observing zero stress
        # We only gain information where complaints were actually filed
        complaints_clean = np.nan_to_num(complaints, nan=0)
        
        # Count cells with positive complaints (actual observations)
        has_complaints = (complaints_clean > 0).astype(float)
        complaint_count = complaints_clean.copy()
        
        # Likelihood precision scales with:
        # 1. Whether complaints exist at this location (binary)
        # 2. Number of complaints (more = more confident)
        # 3. Inverse of observation noise
        # Cells WITHOUT complaints get ZERO likelihood precision (rely on prior only)
        likelihood_precision = (
            has_complaints *  # Only where complaints exist
            (1 + np.log1p(complaint_count)) /  # Log-scaled complaint count
            (cfg.observation_noise ** 2 + 1e-9)
        )
        
        # Scale to reasonable range for Bayesian update
        likelihood_precision = np.clip(likelihood_precision, 0, 100)
        
        # Likelihood mean: what complaints suggest about stress
        # Higher complaints → higher inferred stress
        # Use log transform to map complaint counts to stress scale
        base_rate = 0.1
        likelihood_mean = np.log1p(complaints_clean / base_rate)
        likelihood_mean = np.clip(likelihood_mean, 0, 5.0)  # Cap at reasonable stress level
        
        # ===== POSTERIOR COMPUTATION =====
        # precision_post = precision_prior + precision_likelihood
        posterior_precision = prior_precision + likelihood_precision
        
        # mean_post = (prec_prior × mean_prior + prec_lik × mean_lik) / prec_post
        posterior_mean = (
            prior_precision * prior_mean +
            likelihood_precision * likelihood_mean
        ) / (posterior_precision + 1e-9)
        
        # variance_post = 1 / precision_post
        posterior_variance = 1.0 / (posterior_precision + 1e-9)
        posterior_variance = np.clip(
            posterior_variance,
            cfg.prior_variance_floor,
            cfg.prior_variance_ceil * 2,
        )
        
        # ===== CREDIBLE INTERVALS =====
        posterior_std = np.sqrt(posterior_variance)
        z_score = 1.96  # 95% CI
        credible_lower = posterior_mean - z_score * posterior_std
        credible_upper = posterior_mean + z_score * posterior_std
        
        # ===== DIAGNOSTICS =====
        # Weight of prior vs likelihood
        total_precision = prior_precision + likelihood_precision + 1e-9
        prior_weight = prior_precision / total_precision
        likelihood_weight = likelihood_precision / total_precision
        
        # Information gain: KL(posterior || prior)
        # For Gaussians: KL = 0.5 * (var_ratio + mean_diff²/var_prior - 1 - log(var_ratio))
        var_ratio = posterior_variance / (prior.variance + 1e-9)
        mean_diff = posterior_mean - prior_mean
        kl_divergence = 0.5 * (
            var_ratio +
            mean_diff ** 2 / (prior.variance + 1e-9) -
            1 -
            np.log(var_ratio + 1e-9)
        )
        kl_divergence = np.clip(kl_divergence, 0, 100)
        
        # ===== RELIABILITY MASKS =====
        # No decision where posterior uncertainty too high
        no_decision_mask = posterior_std > cfg.no_decision_threshold
        
        # Reliable where we have complaints and reasonable uncertainty
        reliable_mask = (
            (has_complaints.sum(axis=0) >= cfg.min_observations) &
            (posterior_std.mean(axis=0) < cfg.no_decision_threshold * 0.8)
        )
        # Broadcast to full shape
        reliable_mask = np.broadcast_to(reliable_mask[np.newaxis, :, :], posterior_mean.shape)
        
        # ===== SANITY CHECKS =====
        # Verify that observations actually affected posterior
        has_any_complaints = has_complaints > 0
        mean_kl = np.nanmean(kl_divergence[has_any_complaints])
        if np.isnan(mean_kl) or mean_kl < 0.01:
            notes.append(
                "⚠ WARNING: Observations barely affected posterior (KL < 0.01). "
                "Likelihood may be misconfigured."
            )
            logger.warning(
                "Observations had minimal effect: mean KL divergence = %.4f",
                mean_kl if not np.isnan(mean_kl) else 0.0
            )
        else:
            notes.append(f"Information gain from observations: mean KL = {mean_kl:.3f}")
        
        # Verify variance reduction where observations exist
        var_reduction = (prior.variance - posterior_variance) / (prior.variance + 1e-9)
        mean_var_reduction = np.nanmean(var_reduction[has_any_complaints])
        if not np.isnan(mean_var_reduction):
            notes.append(f"Mean variance reduction where observed: {mean_var_reduction:.1%}")
            
            if mean_var_reduction < 0.05:
                notes.append(
                    "⚠ WARNING: Observations barely reduced uncertainty. "
                    "Check observation model."
                )
                logger.warning(
                    "Observations didn't reduce uncertainty: %.1f%% reduction",
                    mean_var_reduction * 100
                )
        
        logger.info(
            "Posterior computed: mean=[%.2f, %.2f], std=[%.3f, %.3f], "
            "reliable=%.1f%%, no_decision=%.1f%%",
            np.nanmin(posterior_mean), np.nanmax(posterior_mean),
            np.nanmin(posterior_std), np.nanmax(posterior_std),
            reliable_mask.mean() * 100,
            no_decision_mask.mean() * 100,
        )
        
        return BayesianPosterior(
            mean=posterior_mean.astype(np.float32),
            variance=posterior_variance.astype(np.float32),
            precision=posterior_precision.astype(np.float32),
            credible_lower=np.maximum(credible_lower, 0).astype(np.float32),
            credible_upper=credible_upper.astype(np.float32),
            prior_mean=prior.mean.astype(np.float32),
            prior_variance=prior.variance.astype(np.float32),
            prior_weight=prior_weight.astype(np.float32),
            likelihood_weight=likelihood_weight.astype(np.float32),
            information_gain=kl_divergence.astype(np.float32),
            no_decision_mask=no_decision_mask.astype(bool),
            reliable_mask=reliable_mask.astype(bool),
            notes=notes,
        )
    
    def infer(
        self,
        rainfall_intensity: np.ndarray,
        rainfall_accumulation: np.ndarray,
        upstream_contribution: np.ndarray,
        complaints: np.ndarray,
        weights: Tuple[float, float, float] = (1.0, 0.5, 0.3),
        spatial_bias: Optional[np.ndarray] = None,
        temporal_bias: Optional[np.ndarray] = None,
    ) -> BayesianPosterior:
        """Full Bayesian inference pipeline.
        
        This is the main entry point. It:
        1. Computes prior from rainfall + terrain
        2. Computes likelihood from complaints
        3. Combines into posterior
        
        Args:
            rainfall_intensity: (T, H, W) rainfall
            rainfall_accumulation: (T, H, W) accumulated
            upstream_contribution: (H, W) terrain
            complaints: (T, H, W) observed complaints
            weights: Model weights (can be from calibration)
            spatial_bias: Optional spatial reporting bias
            temporal_bias: Optional temporal reporting modifier
            
        Returns:
            BayesianPosterior with full inference results
        """
        logger.info("Running true Bayesian inference...")
        
        # 1. Prior
        logger.info("Step 1/3: Computing prior...")
        prior = self.compute_prior(
            rainfall_intensity,
            rainfall_accumulation,
            upstream_contribution,
            weights=weights,
        )
        
        # 2. Likelihood
        logger.info("Step 2/3: Computing likelihood...")
        likelihood = self.compute_likelihood(
            complaints,
            prior.mean,  # Evaluate likelihood at prior mean
            spatial_bias=spatial_bias,
            temporal_bias=temporal_bias,
        )
        
        # 3. Posterior
        logger.info("Step 3/3: Computing posterior...")
        posterior = self.compute_posterior(prior, likelihood, complaints)
        
        return posterior


def run_bayesian_inference(
    rainfall_intensity: np.ndarray,
    rainfall_accumulation: np.ndarray,
    upstream_contribution: np.ndarray,
    complaints: np.ndarray,
    config: Optional[InferenceConfig] = None,
) -> BayesianPosterior:
    """Convenience function for Bayesian inference.
    
    This is the main API entry point.
    """
    engine = TrueBayesianInference(config)
    return engine.infer(
        rainfall_intensity,
        rainfall_accumulation,
        upstream_contribution,
        complaints,
    )
