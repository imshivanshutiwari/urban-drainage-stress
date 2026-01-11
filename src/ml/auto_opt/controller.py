"""
Experiment Controller for Automated Optimization.
Implements Step 1: Managing the state of the optimization loop.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ExperimentState:
    iteration: int
    best_config: Dict
    best_metrics: Dict
    history: List[Dict]
    status: str  # 'RUNNING', 'CONVERGED', 'STOPPED', 'FAILED'
    start_time: str

class OptimizationController:
    """Manages the lifecycle of the optimization process."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.state_file = self.save_dir / "optimization_state.json"
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> ExperimentState:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return ExperimentState(**data)
            except Exception as e:
                logger.warning(f"Could not load state: {e}. Creating new.")
        
        return ExperimentState(
            iteration=0,
            best_config={},
            best_metrics={'hybrid_rmse': float('inf')},
            history=[],
            status='RUNNING',
            start_time=datetime.now().isoformat()
        )

    def update(self, config: Dict, metrics: Dict):
        """Update state with new run results."""
        self.state.iteration += 1
        
        # Check if best
        current_hybrid = metrics.get('hybrid_rmse', float('inf'))
        best_hybrid = self.state.best_metrics.get('hybrid_rmse', float('inf'))
        
        is_best = False
        if current_hybrid < best_hybrid:
            self.state.best_config = config
            self.state.best_metrics = metrics
            is_best = True
            
        record = {
            'iteration': self.state.iteration,
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        self.state.history.append(record)
        self._save_state()
        return is_best

    def finish(self, status: str = 'CONVERGED'):
        self.state.status = status
        self._save_state()

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)

    @property
    def best_result(self):
        return self.state.best_config, self.state.best_metrics
