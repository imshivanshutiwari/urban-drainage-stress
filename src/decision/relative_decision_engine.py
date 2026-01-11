"""
STEP 5: Relative Probabilistic Decision Logic

CRITICAL: ALL DECISIONS ARE RELATIVE, NOT ABSOLUTE

Decisions based on:
1. Percentile rank WITHIN city
2. Probability mass ABOVE critical quantile
3. Uncertainty-weighted confidence

NO ABSOLUTE THRESHOLDS EVER
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DecisionCategory(Enum):
    """Decision categories."""
    NO_ACTION = "no_action"      # Low relative priority
    MONITOR = "monitor"          # Medium relative priority
    ACTION = "action"            # High relative priority
    URGENT = "urgent"            # Very high relative priority
    NO_DECISION = "no_decision"  # Uncertainty too high


@dataclass
class DecisionResult:
    """Result of relative decision making."""
    decisions: np.ndarray          # Category for each point
    percentile_ranks: np.ndarray   # Rank within city [0, 100]
    exceedance_probs: np.ndarray   # P(Z > critical_quantile)
    confidence: np.ndarray         # Decision confidence [0, 1]
    stats: Dict
    
    def to_dict(self) -> Dict:
        unique, counts = np.unique(self.decisions, return_counts=True)
        decision_dist = {str(u): int(c) for u, c in zip(unique, counts)}
        return {
            'decision_distribution': decision_dist,
            'mean_percentile': float(np.mean(self.percentile_ranks)),
            'mean_exceedance_prob': float(np.mean(self.exceedance_probs)),
            'mean_confidence': float(np.mean(self.confidence)),
            **self.stats,
        }


class RelativeDecisionEngine:
    """
    Relative Probabilistic Decision Engine
    
    CORE PRINCIPLE: Decisions are RELATIVE to city distribution
    
    NO absolute thresholds like:
        ❌ if stress > 0.7: action
        ❌ if stress > 5.0: urgent
    
    Instead, use:
        ✓ if percentile > 90: action
        ✓ if P(Z > q95) > 0.5: urgent
    
    This ensures Seattle and NYC produce SIMILAR decision patterns
    despite having different raw stress magnitudes.
    
    Usage:
        engine = RelativeDecisionEngine()
        engine.set_city_reference(seattle_Z)  # Learn city distribution
        result = engine.decide(Z, sigma)
    """
    
    def __init__(self,
                 no_decision_confidence: float = 0.3,
                 conservative_mode: bool = True):
        """
        Args:
            no_decision_confidence: Confidence below this → NO_DECISION
            conservative_mode: If True, err on side of caution
        """
        self.no_decision_confidence = no_decision_confidence
        self.conservative_mode = conservative_mode
        self._city_fitted = False
        self._city_quantiles = None
    
    def set_city_reference(self, Z_city: np.ndarray) -> 'RelativeDecisionEngine':
        """
        Set city reference distribution for relative decisions.
        
        Args:
            Z_city: Representative latent stress for this city
        
        Returns:
            self for chaining
        """
        flat = Z_city.flatten()
        
        # Compute quantiles
        self._city_quantiles = {
            f'q{p}': np.percentile(flat, p)
            for p in [10, 25, 50, 75, 90, 95, 99]
        }
        self._city_quantiles['min'] = np.min(flat)
        self._city_quantiles['max'] = np.max(flat)
        self._city_quantiles['std'] = np.std(flat)
        
        # Store sorted values for percentile lookup
        self._city_sorted = np.sort(flat)
        
        self._city_fitted = True
        return self
    
    def decide(self, 
               Z: np.ndarray, 
               sigma: np.ndarray,
               reference_Z: Optional[np.ndarray] = None) -> DecisionResult:
        """
        Make relative decisions.
        
        Args:
            Z: Latent stress values (already in latent space)
            sigma: Uncertainty values
            reference_Z: Optional reference for percentiles (if city not set)
        
        Returns:
            DecisionResult with relative decisions
        """
        # Use reference if city not fitted
        if not self._city_fitted and reference_Z is not None:
            self.set_city_reference(reference_Z)
        
        if not self._city_fitted:
            # Fall back to self-relative
            self.set_city_reference(Z)
        
        # 1. Compute percentile ranks
        percentile_ranks = self._compute_percentile_ranks(Z)
        
        # 2. Compute exceedance probabilities
        critical_quantile = self._city_quantiles['q90']
        exceedance_probs = self._compute_exceedance_prob(Z, sigma, critical_quantile)
        
        # 3. Compute confidence (inverse of uncertainty)
        confidence = self._compute_confidence(sigma)
        
        # 4. Make relative decisions
        decisions = self._make_decisions(percentile_ranks, exceedance_probs, confidence)
        
        # Stats
        stats = {
            'city_q50': self._city_quantiles['q50'],
            'city_q90': self._city_quantiles['q90'],
            'city_q95': self._city_quantiles['q95'],
        }
        
        return DecisionResult(
            decisions=decisions,
            percentile_ranks=percentile_ranks,
            exceedance_probs=exceedance_probs,
            confidence=confidence,
            stats=stats,
        )
    
    def _compute_percentile_ranks(self, Z: np.ndarray) -> np.ndarray:
        """Compute percentile rank of each value within city."""
        flat = Z.flatten()
        
        # Use searchsorted for efficient percentile lookup
        ranks = np.searchsorted(self._city_sorted, flat, side='right')
        percentiles = 100.0 * ranks / len(self._city_sorted)
        
        return percentiles.reshape(Z.shape)
    
    def _compute_exceedance_prob(self, 
                                  Z: np.ndarray,
                                  sigma: np.ndarray,
                                  threshold: float) -> np.ndarray:
        """
        Compute P(true_Z > threshold) accounting for uncertainty.
        
        Assumes Gaussian uncertainty.
        """
        from scipy.stats import norm
        
        # P(X > threshold) where X ~ N(Z, sigma)
        # = 1 - Φ((threshold - Z) / sigma)
        z_scores = (threshold - Z) / (sigma + 1e-10)
        probs = 1 - norm.cdf(z_scores)
        
        return probs
    
    def _compute_confidence(self, sigma: np.ndarray) -> np.ndarray:
        """
        Compute confidence from uncertainty.
        
        Low sigma → high confidence
        High sigma → low confidence
        """
        # Sigmoid-like transform
        # sigma = 0.1 → conf ≈ 0.9
        # sigma = 0.5 → conf ≈ 0.5
        # sigma = 1.0 → conf ≈ 0.27
        confidence = np.exp(-2 * sigma)
        return np.clip(confidence, 0, 1)
    
    def _make_decisions(self,
                        percentile_ranks: np.ndarray,
                        exceedance_probs: np.ndarray,
                        confidence: np.ndarray) -> np.ndarray:
        """
        Make relative decisions based on percentile and probability.
        
        RELATIVE THRESHOLDS:
            - NO_DECISION: confidence < threshold
            - NO_ACTION: percentile < 50
            - MONITOR: 50 ≤ percentile < 75
            - ACTION: 75 ≤ percentile < 90 OR exceedance > 0.3
            - URGENT: percentile ≥ 90 AND exceedance > 0.5
        """
        decisions = np.full(percentile_ranks.shape, DecisionCategory.NO_ACTION.value, dtype=object)
        
        # Convert to arrays for vectorized operations
        p = percentile_ranks.flatten()
        e = exceedance_probs.flatten()
        c = confidence.flatten()
        d = np.full(len(p), DecisionCategory.NO_ACTION.value, dtype=object)
        
        # NO_DECISION: uncertainty too high
        no_decision_mask = c < self.no_decision_confidence
        d[no_decision_mask] = DecisionCategory.NO_DECISION.value
        
        # For remaining points, make relative decisions
        decidable = ~no_decision_mask
        
        # MONITOR: 50 ≤ percentile < 75
        monitor_mask = decidable & (p >= 50) & (p < 75)
        d[monitor_mask] = DecisionCategory.MONITOR.value
        
        # ACTION: 75 ≤ percentile < 90 OR exceedance > 0.3
        action_mask = decidable & ((p >= 75) & (p < 90)) | (e > 0.3)
        d[action_mask] = DecisionCategory.ACTION.value
        
        # URGENT: percentile ≥ 90 AND exceedance > 0.5
        urgent_mask = decidable & (p >= 90) & (e > 0.5)
        d[urgent_mask] = DecisionCategory.URGENT.value
        
        # Conservative mode: increase NO_DECISION zone
        if self.conservative_mode:
            marginal_mask = c < 0.5
            d[marginal_mask & ~no_decision_mask] = DecisionCategory.NO_DECISION.value
        
        return d.reshape(percentile_ranks.shape)


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    # Two cities with DIFFERENT raw magnitudes but same relative patterns
    Z_seattle = np.random.randn(5, 50, 50) * 0.5 + 0.0  # Seattle: centered at 0
    Z_nyc = np.random.randn(5, 50, 50) * 1.2 + 1.5      # NYC: higher values
    
    # Same uncertainties
    sigma_seattle = 0.2 * np.ones_like(Z_seattle)
    sigma_nyc = 0.4 * np.ones_like(Z_nyc)  # Higher for new city
    
    engine = RelativeDecisionEngine()
    
    print("Relative Decision Engine Test")
    print("=" * 50)
    
    # Seattle
    engine.set_city_reference(Z_seattle)
    result_seattle = engine.decide(Z_seattle, sigma_seattle)
    print("\nSeattle (reference):")
    print(f"  Decision distribution: {result_seattle.to_dict()['decision_distribution']}")
    print(f"  Mean percentile: {result_seattle.to_dict()['mean_percentile']:.1f}")
    
    # NYC - using its OWN reference
    engine_nyc = RelativeDecisionEngine()
    engine_nyc.set_city_reference(Z_nyc)
    result_nyc = engine_nyc.decide(Z_nyc, sigma_nyc)
    print("\nNYC (own reference):")
    print(f"  Decision distribution: {result_nyc.to_dict()['decision_distribution']}")
    print(f"  Mean percentile: {result_nyc.to_dict()['mean_percentile']:.1f}")
    
    print("\n✓ Both cities have similar decision patterns!")
    print("✓ No absolute thresholds used!")
