"""
Pre-check module for the Self-Aware Automated Optimization System.
Implements Step 0 of the project file.
"""

from typing import Dict, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelStatus:
    is_optimal: bool
    reason: str
    metrics: Dict[str, float]

class MaxLevelCheck:
    """
    Evaluates if the current model has reached 'Max Level' performance
    based on scientific constraints and improvement thresholds.
    """
    
    def __init__(self, 
                 improvement_threshold: float = 15.0,
                 min_uncertainty_variance: float = 0.01,
                 dl_dominance_margin: float = 0.05):
        self.improvement_threshold = improvement_threshold
        self.min_uncertainty_variance = min_uncertainty_variance
        self.dl_dominance_margin = dl_dominance_margin

    def evaluate(self, current_metrics: Dict[str, float]) -> ModelStatus:
        """
        Evaluate model status against constraints.
        
        Required metrics in current_metrics:
        - improvement: float (percentage)
        - hybrid_rmse: float
        - dl_only_rmse: float
        - base_rmse: float
        """
        hybrid_rmse = current_metrics.get('hybrid_rmse', 1.0)
        base_rmse = current_metrics.get('base_rmse', 1.0)
        dl_only_rmse = current_metrics.get('dl_only_rmse', 1.0)
        improvement = current_metrics.get('improvement', 0.0)
        
        # Check 1: Scientific Superiority (Hybrid < Base)
        if hybrid_rmse >= base_rmse:
            return ModelStatus(False, "Hybrid model worse than baseline (Constraint Violation)", current_metrics)
            
        # Check 2: DL-Only Inferiority (Hybrid < DL-Only)
        # We want hybrid to be significantly better than dl-only
        if dl_only_rmse < (hybrid_rmse * (1 + self.dl_dominance_margin)):
             return ModelStatus(False, "Hybrid model not sufficiently better than DL-only", current_metrics)

        # Check 3: Improvement Magnitude
        if improvement < self.improvement_threshold:
            return ModelStatus(False, f"Improvement {improvement:.1f}% below threshold {self.improvement_threshold}%", current_metrics)
            
        # If all pass
        return ModelStatus(True, "Model has reached max-level performance constraints.", current_metrics)
