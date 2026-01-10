"""Data-driven risk decision model: decisions depend on BOTH stress AND confidence.

Key principle from Projectfile.md:
- IF stress high AND uncertainty low → HIGH
- IF stress high AND uncertainty high → NO_DECISION
- IF stress moderate → MEDIUM

A system that says "Everything is high risk" is worse than useless.
Risk must depend on both stress AND confidence.

CRITICAL: Now uses CREDIBLE INTERVALS from Bayesian inference.
- If 95% CI spans LOW to HIGH → NO_DECISION
- If CI lower bound is above threshold → confident decision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk classification levels."""
    NO_DECISION = "no-decision"  # Can't decide - uncertainty too high
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RiskDecisionConfig:
    """Configuration for risk decision model."""
    
    # Stress thresholds
    high_stress_threshold: float = 0.7       # Above this = potentially high risk
    medium_stress_threshold: float = 0.4     # Above this = potentially medium risk
    low_stress_threshold: float = 0.2        # Below this = low risk
    
    # Uncertainty thresholds (CRITICAL)
    uncertainty_low: float = 0.3             # Below this = confident
    uncertainty_medium: float = 0.6          # Below this = moderate confidence
    uncertainty_high: float = 0.9            # Above this = NO_DECISION
    
    # Confidence ratio threshold
    confidence_ratio_min: float = 0.3        # stress/uncertainty ratio threshold


@dataclass
class RiskDecisionResult:
    """Result of risk decision computation."""
    
    risk_levels: np.ndarray              # Risk category at each point
    confidence_scores: np.ndarray        # How confident we are (0-1)
    no_decision_mask: np.ndarray         # Where we can't decide
    decision_reasons: np.ndarray         # Why each decision was made
    
    # Statistics
    high_pct: float
    medium_pct: float
    low_pct: float
    no_decision_pct: float


class DataDrivenRiskDecisionModel:
    """Risk decisions that depend on BOTH stress AND confidence.
    
    Key rules (from Projectfile.md):
    1. High stress + Low uncertainty → HIGH
    2. High stress + High uncertainty → NO_DECISION (can't commit!)
    3. Moderate stress + Reasonable uncertainty → MEDIUM
    4. Low stress → LOW
    
    A valid system MUST have NO_DECISION regions.
    If everything is HIGH → model is broken.
    """
    
    def __init__(self, config: Optional[RiskDecisionConfig] = None) -> None:
        self.config = config or RiskDecisionConfig()
        logger.info("DataDrivenRiskDecisionModel initialized")
    
    def compute_risk(
        self,
        stress: np.ndarray,
        uncertainty: np.ndarray,
        reliability_mask: Optional[np.ndarray] = None,
    ) -> RiskDecisionResult:
        """Compute risk levels considering BOTH stress AND uncertainty.
        
        Args:
            stress: (time, y, x) or (y, x) stress values [0, 1+]
            uncertainty: Same shape, standard deviation (raw values)
            reliability_mask: Optional mask of reliable data
            
        Returns:
            RiskDecisionResult with risk levels and statistics
        """
        stress = np.asarray(stress)
        uncertainty = np.asarray(uncertainty)
        
        # Normalize stress to [0, 1] range for thresholding
        stress_norm = stress / (np.nanmax(stress) + 1e-9)
        
        # CRITICAL: Normalize uncertainty to [0, 1] range
        # The uncertainty model may produce raw values like [1.5, 2.7]
        # We need to map this to thresholds in [0, 1]
        unc_min = np.nanmin(uncertainty)
        unc_max = np.nanmax(uncertainty)
        unc_range = unc_max - unc_min + 1e-9
        uncertainty_norm = (uncertainty - unc_min) / unc_range
        
        logger.debug(
            "Uncertainty normalization: raw=[%.3f, %.3f] -> norm=[%.3f, %.3f]",
            unc_min, unc_max, 
            np.nanmin(uncertainty_norm), np.nanmax(uncertainty_norm)
        )
        
        # Initialize outputs
        shape = stress.shape
        risk_levels = np.full(shape, RiskLevel.NO_DECISION.value, dtype=object)
        confidence = np.zeros(shape, dtype=np.float32)
        reasons = np.full(shape, "", dtype=object)
        
        # Compute confidence ratio: low uncertainty = high confidence
        confidence = 1.0 - uncertainty_norm
        confidence = np.clip(confidence, 0, 1)
        
        # Apply decision rules (order matters!)
        cfg = self.config
        
        # Rule 1: NO_DECISION - uncertainty too high (top percentile)
        # Use normalized uncertainty for thresholding
        no_decision = uncertainty_norm >= cfg.uncertainty_high
        risk_levels[no_decision] = RiskLevel.NO_DECISION.value
        reasons[no_decision] = "uncertainty too high"
        
        # Rule 2: HIGH - high stress AND low uncertainty
        # Only commit to HIGH if we're confident
        high_stress = stress_norm >= cfg.high_stress_threshold
        low_uncertainty = uncertainty_norm <= cfg.uncertainty_low
        high_mask = high_stress & low_uncertainty & ~no_decision
        risk_levels[high_mask] = RiskLevel.HIGH.value
        reasons[high_mask] = "high stress + confident"
        
        # Rule 3: High stress but uncertain → NO_DECISION
        # This is the key fix - high stress alone is NOT enough
        high_but_uncertain = (
            high_stress & 
            (uncertainty > cfg.uncertainty_low) & 
            (uncertainty < cfg.uncertainty_high)
        )
        risk_levels[high_but_uncertain] = RiskLevel.NO_DECISION.value
        reasons[high_but_uncertain] = "high stress but uncertain"
        
        # Rule 4: MEDIUM - moderate stress with reasonable uncertainty
        medium_stress = (
            (stress_norm >= cfg.medium_stress_threshold) & 
            (stress_norm < cfg.high_stress_threshold)
        )
        medium_confidence = uncertainty <= cfg.uncertainty_medium
        medium_mask = medium_stress & medium_confidence & ~no_decision
        risk_levels[medium_mask] = RiskLevel.MEDIUM.value
        reasons[medium_mask] = "moderate stress + reasonable confidence"
        
        # Rule 5: LOW - low stress (any confidence)
        low_stress = stress_norm < cfg.low_stress_threshold
        low_mask = low_stress & ~no_decision
        risk_levels[low_mask] = RiskLevel.LOW.value
        reasons[low_mask] = "low stress"
        
        # Rule 6: Moderate stress but high uncertainty → NO_DECISION
        moderate_uncertain = (
            medium_stress & 
            (uncertainty > cfg.uncertainty_medium) & 
            ~no_decision
        )
        risk_levels[moderate_uncertain] = RiskLevel.NO_DECISION.value
        reasons[moderate_uncertain] = "moderate stress but too uncertain"
        
        # Apply reliability mask if provided
        if reliability_mask is not None:
            unreliable = ~reliability_mask
            risk_levels[unreliable] = RiskLevel.NO_DECISION.value
            reasons[unreliable] = "unreliable data"
        
        # Compute statistics
        total = risk_levels.size
        high_pct = np.sum(risk_levels == RiskLevel.HIGH.value) / total * 100
        medium_pct = np.sum(risk_levels == RiskLevel.MEDIUM.value) / total * 100
        low_pct = np.sum(risk_levels == RiskLevel.LOW.value) / total * 100
        no_dec_pct = np.sum(risk_levels == RiskLevel.NO_DECISION.value) / total * 100
        
        # Sanity check logging
        if high_pct > 80:
            logger.error(
                "🚨 BROKEN MODEL: %.1f%% classified as HIGH risk. "
                "A useful model should have mixed categories!",
                high_pct
            )
        elif no_dec_pct < 5:
            logger.warning(
                "⚠ Suspiciously few NO_DECISION regions (%.1f%%). "
                "Check if uncertainty model is working correctly.",
                no_dec_pct
            )
        else:
            logger.info(
                "Risk distribution: HIGH=%.1f%%, MEDIUM=%.1f%%, "
                "LOW=%.1f%%, NO_DECISION=%.1f%%",
                high_pct, medium_pct, low_pct, no_dec_pct
            )
        
        return RiskDecisionResult(
            risk_levels=risk_levels,
            confidence_scores=confidence,
            no_decision_mask=(risk_levels == RiskLevel.NO_DECISION.value),
            decision_reasons=reasons,
            high_pct=high_pct,
            medium_pct=medium_pct,
            low_pct=low_pct,
            no_decision_pct=no_dec_pct,
        )
    
    def explain_decision(
        self,
        stress_val: float,
        uncertainty_val: float,
    ) -> Dict[str, str]:
        """Explain why a specific decision was made."""
        cfg = self.config
        
        stress_norm = min(stress_val, 1.0)  # Assume normalized
        
        if uncertainty_val >= cfg.uncertainty_high:
            return {
                "decision": RiskLevel.NO_DECISION.value,
                "reason": f"Uncertainty ({uncertainty_val:.3f}) exceeds threshold "
                         f"({cfg.uncertainty_high}). Cannot commit to a decision.",
                "recommendation": "Collect more data before acting."
            }
        
        if stress_norm >= cfg.high_stress_threshold:
            if uncertainty_val <= cfg.uncertainty_low:
                return {
                    "decision": RiskLevel.HIGH.value,
                    "reason": f"High stress ({stress_norm:.3f}) with high confidence "
                             f"(uncertainty={uncertainty_val:.3f}).",
                    "recommendation": "Prioritize immediate action."
                }
            else:
                return {
                    "decision": RiskLevel.NO_DECISION.value,
                    "reason": f"High stress ({stress_norm:.3f}) but uncertainty "
                             f"({uncertainty_val:.3f}) is too high to commit.",
                    "recommendation": "Potential issue but verify before acting."
                }
        
        if stress_norm >= cfg.medium_stress_threshold:
            if uncertainty_val <= cfg.uncertainty_medium:
                return {
                    "decision": RiskLevel.MEDIUM.value,
                    "reason": f"Moderate stress ({stress_norm:.3f}) with acceptable "
                             f"uncertainty ({uncertainty_val:.3f}).",
                    "recommendation": "Monitor closely, plan intervention."
                }
            else:
                return {
                    "decision": RiskLevel.NO_DECISION.value,
                    "reason": f"Moderate stress but uncertainty too high.",
                    "recommendation": "Continue monitoring."
                }
        
        return {
            "decision": RiskLevel.LOW.value,
            "reason": f"Low stress ({stress_norm:.3f}).",
            "recommendation": "No action required."
        }


def compute_risk_decisions(
    stress: np.ndarray,
    uncertainty: np.ndarray,
    reliability_mask: Optional[np.ndarray] = None,
    config: Optional[RiskDecisionConfig] = None,
) -> RiskDecisionResult:
    """Convenience function for risk decision computation."""
    model = DataDrivenRiskDecisionModel(config)
    return model.compute_risk(stress, uncertainty, reliability_mask)


class CredibleIntervalRiskModel:
    """Risk decisions based on Bayesian credible intervals.
    
    This is the PROPER way to make decisions:
    - Look at the 95% credible interval [lower, upper]
    - If ENTIRE interval is above high threshold → HIGH
    - If ENTIRE interval is below low threshold → LOW
    - If interval SPANS multiple categories → NO_DECISION
    
    This ensures we only commit when we're confident.
    """
    
    def __init__(
        self,
        high_threshold: float = 0.7,
        medium_threshold: float = 0.4,
        low_threshold: float = 0.2,
        max_ci_width: float = 0.5,  # NO_DECISION if CI wider than this
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        self.max_ci_width = max_ci_width
        logger.info("CredibleIntervalRiskModel initialized")
    
    def compute_risk_from_credible_intervals(
        self,
        posterior_mean: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        reliability_mask: Optional[np.ndarray] = None,
    ) -> RiskDecisionResult:
        """Compute risk using credible intervals directly.
        
        Args:
            posterior_mean: (T, H, W) posterior mean stress
            ci_lower: (T, H, W) 2.5th percentile
            ci_upper: (T, H, W) 97.5th percentile
            reliability_mask: Optional mask
            
        Returns:
            RiskDecisionResult with credible-interval-based decisions
        """
        shape = posterior_mean.shape
        
        # Normalize to [0, 1]
        max_val = np.nanmax(ci_upper) + 1e-9
        mean_norm = posterior_mean / max_val
        lower_norm = ci_lower / max_val
        upper_norm = ci_upper / max_val
        
        # CI width
        ci_width = upper_norm - lower_norm
        
        # Initialize
        risk_levels = np.full(shape, RiskLevel.NO_DECISION.value, dtype=object)
        confidence = np.zeros(shape, dtype=np.float32)
        reasons = np.full(shape, "", dtype=object)
        
        # Rule 1: CI too wide → NO_DECISION
        wide_ci = ci_width > self.max_ci_width
        risk_levels[wide_ci] = RiskLevel.NO_DECISION.value
        reasons[wide_ci] = f"CI too wide (>{self.max_ci_width:.2f})"
        
        # Rule 2: HIGH - entire CI above high threshold
        confident_high = (lower_norm >= self.high_threshold) & ~wide_ci
        risk_levels[confident_high] = RiskLevel.HIGH.value
        reasons[confident_high] = "CI entirely above high threshold"
        confidence[confident_high] = 1.0 - ci_width[confident_high]
        
        # Rule 3: LOW - entire CI below low threshold
        confident_low = (upper_norm <= self.low_threshold) & ~wide_ci
        risk_levels[confident_low] = RiskLevel.LOW.value
        reasons[confident_low] = "CI entirely below low threshold"
        confidence[confident_low] = 1.0 - ci_width[confident_low]
        
        # Rule 4: MEDIUM - CI within medium band
        medium_band = (
            (lower_norm >= self.low_threshold) &
            (upper_norm <= self.high_threshold) &
            ~wide_ci &
            (risk_levels != RiskLevel.HIGH.value) &
            (risk_levels != RiskLevel.LOW.value)
        )
        risk_levels[medium_band] = RiskLevel.MEDIUM.value
        reasons[medium_band] = "CI within medium range"
        confidence[medium_band] = 1.0 - ci_width[medium_band]
        
        # Rule 5: CI spans thresholds → NO_DECISION
        # E.g., lower in LOW, upper in HIGH
        spans_categories = (
            (lower_norm < self.medium_threshold) &
            (upper_norm > self.medium_threshold) &
            ~wide_ci &
            (risk_levels == RiskLevel.NO_DECISION.value)
        )
        reasons[spans_categories] = "CI spans multiple risk categories"
        
        # Rule 6: Apply reliability mask
        if reliability_mask is not None:
            unreliable = ~reliability_mask
            risk_levels[unreliable] = RiskLevel.NO_DECISION.value
            reasons[unreliable] = "unreliable data"
        
        # Statistics
        total = risk_levels.size
        high_pct = np.sum(risk_levels == RiskLevel.HIGH.value) / total * 100
        medium_pct = np.sum(risk_levels == RiskLevel.MEDIUM.value) / total * 100
        low_pct = np.sum(risk_levels == RiskLevel.LOW.value) / total * 100
        no_dec_pct = np.sum(risk_levels == RiskLevel.NO_DECISION.value) / total * 100
        
        logger.info(
            "CI-based risk: HIGH=%.1f%%, MEDIUM=%.1f%%, LOW=%.1f%%, NO_DECISION=%.1f%%",
            high_pct, medium_pct, low_pct, no_dec_pct
        )
        
        # Validation
        if high_pct > 80:
            logger.error("🚨 >80%% HIGH risk - model likely broken!")
        if no_dec_pct < 10:
            logger.warning("⚠ <10%% NO_DECISION - may be overconfident")
        
        return RiskDecisionResult(
            risk_levels=risk_levels,
            confidence_scores=confidence,
            no_decision_mask=(risk_levels == RiskLevel.NO_DECISION.value),
            decision_reasons=reasons,
            high_pct=high_pct,
            medium_pct=medium_pct,
            low_pct=low_pct,
            no_decision_pct=no_dec_pct,
        )


def compute_risk_from_bayesian_posterior(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    confidence_level: float = 0.95,
    reliability_mask: Optional[np.ndarray] = None,
) -> RiskDecisionResult:
    """Compute risk from Bayesian posterior mean and variance.
    
    Constructs credible intervals from mean and variance (Gaussian assumption),
    then applies credible interval decision rules.
    """
    from scipy import stats
    
    # Compute credible intervals
    z = stats.norm.ppf((1 + confidence_level) / 2)  # ~1.96 for 95%
    std = np.sqrt(posterior_variance)
    ci_lower = posterior_mean - z * std
    ci_upper = posterior_mean + z * std
    
    # Use CI-based model
    model = CredibleIntervalRiskModel()
    return model.compute_risk_from_credible_intervals(
        posterior_mean, ci_lower, ci_upper, reliability_mask
    )
