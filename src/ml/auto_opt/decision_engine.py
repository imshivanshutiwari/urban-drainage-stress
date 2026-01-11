"""
Decision Engine: Intelligently proposes the next configuration.
Implements Step 4. NO Random Search.
"""

from typing import Dict, List
import copy
import random

class DecisionEngine:
    """Heuristic-based hyperparameter proposer."""
    
    def __init__(self, search_space: Dict):
        self.search_space = search_space
        
    def propose_next(self, current_config: Dict, analysis: str) -> Dict:
        """
        Propose next config based on previous outcome.
        """
        new_config = copy.deepcopy(current_config)
        
        # Simple heuristic logic
        if analysis == 'IMPROVEMENT':
            # Keep going in same direction? OR fine tune?
            # For simplicity: Try increasing capacity slightly
            if new_config.get('hidden_dim', 64) < 256:
                new_config['hidden_dim'] = new_config.get('hidden_dim', 64) * 2
        elif analysis == 'REGRESSION':
            # Backtrack / Try regularization
            new_config['dropout'] = min(0.5, new_config.get('dropout', 0.1) + 0.1)
        else: # Stagnation
            # Try changing learning rate
            new_config['learning_rate'] = new_config.get('learning_rate', 0.001) / 2
            
        return new_config

    def get_initial_config(self) -> Dict:
        return {
            'hidden_dim': 64,
            'num_spatial_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 30
        }
