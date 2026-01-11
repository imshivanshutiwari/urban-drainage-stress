"""
STEP 6: Failure-Aware Output Contract

Output MUST declare:
1. confident: Confident decision (proceed)
2. uncertain: Uncertain decision (flag for review)
3. refused: Model refuses to decide (NO_DECISION)

This is a CONTRACT - violations should raise errors
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class OutputState(Enum):
    """Output confidence states."""
    CONFIDENT = "confident"    # Model is confident
    UNCERTAIN = "uncertain"    # Model is uncertain, flag for review
    REFUSED = "refused"        # Model refuses to decide


@dataclass
class DecisionContract:
    """
    Single-point decision contract.
    
    Every output MUST have one of three states.
    """
    state: OutputState
    decision: Optional[str]           # The actual decision (if any)
    confidence: float                  # [0, 1]
    percentile_rank: float            # Position in city distribution
    exceedance_probability: float     # P(exceeds critical threshold)
    uncertainty: float                # σ_final
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'decision': self.decision,
            'confidence': self.confidence,
            'percentile_rank': self.percentile_rank,
            'exceedance_probability': self.exceedance_probability,
            'uncertainty': self.uncertainty,
            **self.metadata,
        }
    
    @property
    def is_actionable(self) -> bool:
        """Can we act on this decision?"""
        return self.state == OutputState.CONFIDENT
    
    @property
    def needs_review(self) -> bool:
        """Does this need human review?"""
        return self.state == OutputState.UNCERTAIN
    
    @property
    def was_refused(self) -> bool:
        """Did the model refuse to decide?"""
        return self.state == OutputState.REFUSED


@dataclass
class BatchContractResult:
    """Contract result for a batch of decisions."""
    contracts: np.ndarray  # Array of DecisionContract objects
    summary: Dict
    
    def to_dict(self) -> Dict:
        return self.summary.copy()
    
    def get_confident(self) -> np.ndarray:
        """Get mask of confident decisions."""
        flat = self.contracts.flatten()
        return np.array([c.is_actionable for c in flat]).reshape(self.contracts.shape)
    
    def get_uncertain(self) -> np.ndarray:
        """Get mask of uncertain decisions."""
        flat = self.contracts.flatten()
        return np.array([c.needs_review for c in flat]).reshape(self.contracts.shape)
    
    def get_refused(self) -> np.ndarray:
        """Get mask of refused decisions."""
        flat = self.contracts.flatten()
        return np.array([c.was_refused for c in flat]).reshape(self.contracts.shape)


class ContractBuilder:
    """
    Builds Decision Contracts from inference outputs.
    
    Ensures every output has exactly one of:
        - CONFIDENT: High confidence, proceed with decision
        - UNCERTAIN: Low confidence, flag for human review
        - REFUSED: Uncertainty too high, no decision made
    
    Usage:
        builder = ContractBuilder()
        result = builder.build(
            decisions=...,
            confidence=...,
            percentile_ranks=...,
            exceedance_probs=...,
            sigma_final=...
        )
    """
    
    def __init__(self,
                 confident_threshold: float = 0.7,
                 uncertain_threshold: float = 0.3,
                 max_uncertainty_for_decision: float = 1.0):
        """
        Args:
            confident_threshold: confidence ≥ this → CONFIDENT
            uncertain_threshold: this ≤ confidence < confident → UNCERTAIN
            max_uncertainty_for_decision: σ > this → REFUSED
        """
        self.confident_threshold = confident_threshold
        self.uncertain_threshold = uncertain_threshold
        self.max_uncertainty = max_uncertainty_for_decision
    
    def build(self,
              decisions: np.ndarray,
              confidence: np.ndarray,
              percentile_ranks: np.ndarray,
              exceedance_probs: np.ndarray,
              sigma_final: np.ndarray) -> BatchContractResult:
        """
        Build decision contracts from outputs.
        
        Args:
            decisions: Decision categories (strings)
            confidence: Confidence values [0, 1]
            percentile_ranks: Percentile within city [0, 100]
            exceedance_probs: Exceedance probabilities [0, 1]
            sigma_final: Final uncertainties
        
        Returns:
            BatchContractResult with contracts and summary
        """
        shape = decisions.shape
        flat_d = decisions.flatten()
        flat_c = confidence.flatten()
        flat_p = percentile_ranks.flatten()
        flat_e = exceedance_probs.flatten()
        flat_s = sigma_final.flatten()
        
        contracts = []
        
        for i in range(len(flat_d)):
            state = self._determine_state(flat_c[i], flat_s[i])
            
            contract = DecisionContract(
                state=state,
                decision=flat_d[i] if state != OutputState.REFUSED else None,
                confidence=float(flat_c[i]),
                percentile_rank=float(flat_p[i]),
                exceedance_probability=float(flat_e[i]),
                uncertainty=float(flat_s[i]),
            )
            contracts.append(contract)
        
        contracts = np.array(contracts, dtype=object).reshape(shape)
        
        # Compute summary
        summary = self._compute_summary(contracts)
        
        return BatchContractResult(contracts=contracts, summary=summary)
    
    def _determine_state(self, confidence: float, uncertainty: float) -> OutputState:
        """Determine the contract state for a single point."""
        # Check uncertainty first
        if uncertainty > self.max_uncertainty:
            return OutputState.REFUSED
        
        # Then check confidence
        if confidence >= self.confident_threshold:
            return OutputState.CONFIDENT
        elif confidence >= self.uncertain_threshold:
            return OutputState.UNCERTAIN
        else:
            return OutputState.REFUSED
    
    def _compute_summary(self, contracts: np.ndarray) -> Dict:
        """Compute summary statistics for contracts."""
        flat = contracts.flatten()
        n_total = len(flat)
        
        n_confident = sum(1 for c in flat if c.state == OutputState.CONFIDENT)
        n_uncertain = sum(1 for c in flat if c.state == OutputState.UNCERTAIN)
        n_refused = sum(1 for c in flat if c.state == OutputState.REFUSED)
        
        # Get decision distribution for confident outputs
        confident_decisions = [c.decision for c in flat if c.is_actionable]
        decision_dist = {}
        for d in confident_decisions:
            decision_dist[d] = decision_dist.get(d, 0) + 1
        
        return {
            'total_points': n_total,
            'confident': n_confident,
            'confident_pct': 100 * n_confident / n_total,
            'uncertain': n_uncertain,
            'uncertain_pct': 100 * n_uncertain / n_total,
            'refused': n_refused,
            'refused_pct': 100 * n_refused / n_total,
            'confident_decision_distribution': decision_dist,
        }


class ContractValidator:
    """
    Validates that contracts meet requirements.
    
    Raises errors if contract invariants are violated.
    """
    
    @staticmethod
    def validate(result: BatchContractResult) -> bool:
        """
        Validate batch contract result.
        
        Raises:
            AssertionError if any invariant violated
        
        Returns:
            True if valid
        """
        flat = result.contracts.flatten()
        
        for i, contract in enumerate(flat):
            # Every contract must have exactly one state
            assert contract.state in OutputState, f"Invalid state at {i}"
            
            # Confidence must be in [0, 1]
            assert 0 <= contract.confidence <= 1, f"Invalid confidence at {i}"
            
            # Percentile must be in [0, 100]
            assert 0 <= contract.percentile_rank <= 100, f"Invalid percentile at {i}"
            
            # If refused, decision must be None
            if contract.state == OutputState.REFUSED:
                assert contract.decision is None, f"Refused but has decision at {i}"
            
            # If confident, decision must not be None
            if contract.state == OutputState.CONFIDENT:
                assert contract.decision is not None, f"Confident but no decision at {i}"
        
        # Summary must be consistent
        s = result.summary
        total = s['confident'] + s['uncertain'] + s['refused']
        assert total == s['total_points'], "Summary totals inconsistent"
        
        return True


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulated outputs
    n = 100
    decisions = np.random.choice(['no_action', 'monitor', 'action'], size=(n,))
    confidence = np.random.uniform(0.1, 0.9, size=(n,))
    percentile_ranks = np.random.uniform(0, 100, size=(n,))
    exceedance_probs = np.random.uniform(0, 1, size=(n,))
    sigma_final = np.random.uniform(0.1, 1.2, size=(n,))
    
    builder = ContractBuilder()
    result = builder.build(decisions, confidence, percentile_ranks, 
                          exceedance_probs, sigma_final)
    
    print("Decision Contract Test")
    print("=" * 50)
    print(f"\nSummary:")
    for k, v in result.summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")
    
    # Validate
    is_valid = ContractValidator.validate(result)
    print(f"\nContract validation: {'PASSED' if is_valid else 'FAILED'}")
