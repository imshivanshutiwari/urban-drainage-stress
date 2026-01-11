"""
Stopping Criteria.
Implements Step 5.
"""

from typing import List, Dict

class StoppingCriteria:
    def __init__(self, max_iterations: int = 10, patience: int = 3):
        self.max_iterations = max_iterations
        self.patience = patience
        
    def should_stop(self, history: List[Dict]) -> bool:
        if len(history) >= self.max_iterations:
            return True
            
        # Check patience (last N runs no best)
        # Assuming history is ordered
        if len(history) < self.patience:
            return False
            
        # If last N runs were not 'is_best'
        last_n = history[-self.patience:]
        if not any(r['is_best'] for r in last_n):
             return True
             
        return False
