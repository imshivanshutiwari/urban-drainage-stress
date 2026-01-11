"""
Result Analyzer: Classifies the outcome of an experiment.
Implements Step 3.
"""

from typing import Dict, Tuple

class ResultAnalyzer:
    """Analyzes experiment results to determine the outcome category."""
    
    def analyze(self, prev_metrics: Dict, current_metrics: Dict) -> str:
        """
        Returns one of: 'IMPROVEMENT', 'STAGNATION', 'REGRESSION'
        """
        prev_rmse = prev_metrics.get('hybrid_rmse', float('inf'))
        curr_rmse = current_metrics.get('hybrid_rmse', float('inf'))
        
        # Significant improvement threshold (e.g., 1% rel)
        threshold = 0.01 
        
        if curr_rmse < prev_rmse * (1 - threshold):
            return 'IMPROVEMENT'
        elif curr_rmse > prev_rmse:
            return 'REGRESSION'
        else:
            return 'STAGNATION'
