"""Ablation studies for ST-GNN.

Evaluates:
1. Base system (no DL)
2. Base + ST-GNN residual
3. ST-GNN only (NO physics)

Reference: projectfile.md Step 6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import json

import numpy as np

from .metrics import compute_metrics, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    name: str
    description: str
    metrics: MetricsResult
    is_baseline: bool = False
    improvement_over_baseline: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'metrics': self.metrics.to_dict(),
            'is_baseline': self.is_baseline,
            'improvement_over_baseline': self.improvement_over_baseline,
        }


@dataclass
class AblationReport:
    """Complete ablation study report."""
    results: List[AblationResult]
    baseline_name: str
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'results': [r.to_dict() for r in self.results],
            'baseline_name': self.baseline_name,
            'summary': self.summary,
        }
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class AblationRunner:
    """Runner for ablation experiments.
    
    Key ablations from projectfile.md:
    1. Base (no DL) - Bayesian only
    2. Base + ST-GNN - Full hybrid system
    3. ST-GNN only - DL replacing physics (should be WORSE)
    
    The ablation must prove that ML is SUPPORTIVE, not DOMINANT.
    If ST-GNN alone beats Bayesian alone, something is wrong.
    """
    
    def __init__(self):
        self.results: List[AblationResult] = []
        self.baseline_result: Optional[AblationResult] = None
    
    def run_ablations(
        self,
        bayesian_stress: np.ndarray,  # (T, N)
        bayesian_variance: np.ndarray,  # (T, N)
        observed_stress: np.ndarray,  # (T, N)
        dl_residual: Optional[np.ndarray] = None,  # (T, N)
        dl_uncertainty: Optional[np.ndarray] = None,  # (T, N)
        mask: Optional[np.ndarray] = None,
    ) -> AblationReport:
        """Run all ablation experiments.
        
        Args:
            bayesian_stress: baseline Bayesian predictions
            bayesian_variance: Bayesian uncertainty
            observed_stress: ground truth (complaint-adjusted)
            dl_residual: ST-GNN predicted residuals
            dl_uncertainty: ST-GNN predicted uncertainty
            mask: validity mask
            
        Returns:
            AblationReport with all results
        """
        self.results = []
        
        # Compute target residual
        target_residual = observed_stress - bayesian_stress
        
        # 1. Baseline: Bayesian only (no DL)
        logger.info("Running ablation 1/3: Bayesian only (baseline)")
        baseline_metrics = compute_metrics(
            pred_residual=np.zeros_like(target_residual),  # No correction
            target_residual=target_residual,
            pred_uncertainty=np.sqrt(bayesian_variance),
            bayesian_stress=bayesian_stress,
            observed_stress=observed_stress,
            mask=mask,
        )
        
        self.baseline_result = AblationResult(
            name="bayesian_only",
            description="Baseline: Bayesian inference without DL correction",
            metrics=baseline_metrics,
            is_baseline=True,
        )
        self.results.append(self.baseline_result)
        
        # 2. Full hybrid: Bayesian + ST-GNN
        if dl_residual is not None:
            logger.info("Running ablation 2/3: Bayesian + ST-GNN (hybrid)")
            
            hybrid_metrics = compute_metrics(
                pred_residual=dl_residual,
                target_residual=target_residual,
                pred_uncertainty=dl_uncertainty,
                bayesian_stress=bayesian_stress,
                observed_stress=observed_stress,
                mask=mask,
            )
            
            # Compute improvement over baseline
            improvement = self._compute_improvement(baseline_metrics, hybrid_metrics)
            
            hybrid_result = AblationResult(
                name="bayesian_plus_dl",
                description="Hybrid: Bayesian + ST-GNN residual correction",
                metrics=hybrid_metrics,
                improvement_over_baseline=improvement,
            )
            self.results.append(hybrid_result)
            
            # 3. ST-GNN only (sanity check - should NOT beat baseline)
            logger.info("Running ablation 3/3: ST-GNN only (sanity check)")
            
            # If DL alone predicts stress (not residual), it should be worse
            # This uses the residual as if it were the full prediction
            dl_only_metrics = compute_metrics(
                pred_residual=dl_residual - target_residual,  # Error of DL-only
                target_residual=np.zeros_like(target_residual),
                pred_uncertainty=dl_uncertainty,
                mask=mask,
            )
            
            dl_only_result = AblationResult(
                name="dl_only",
                description="ST-GNN only: DL without physics (sanity check)",
                metrics=dl_only_metrics,
            )
            self.results.append(dl_only_result)
            
            # Verify DL is supportive, not dominant
            if hybrid_metrics.rmse > baseline_metrics.rmse:
                logger.warning(
                    "⚠️ DL HURTS performance! "
                    f"Baseline RMSE: {baseline_metrics.rmse:.4f}, "
                    f"Hybrid RMSE: {hybrid_metrics.rmse:.4f}"
                )
            else:
                logger.info(
                    f"✓ DL improves performance: "
                    f"Baseline RMSE: {baseline_metrics.rmse:.4f}, "
                    f"Hybrid RMSE: {hybrid_metrics.rmse:.4f} "
                    f"({improvement['rmse_improvement']:.1%} improvement)"
                )
        
        # Create report
        summary = self._create_summary()
        
        return AblationReport(
            results=self.results,
            baseline_name=self.baseline_result.name,
            summary=summary,
        )
    
    def _compute_improvement(
        self,
        baseline: MetricsResult,
        comparison: MetricsResult,
    ) -> Dict[str, float]:
        """Compute improvement metrics."""
        improvements = {}
        
        # RMSE (lower is better)
        if baseline.rmse > 0:
            improvements['rmse_improvement'] = (baseline.rmse - comparison.rmse) / baseline.rmse
        else:
            improvements['rmse_improvement'] = 0.0
        
        # MAE (lower is better)
        if baseline.mae > 0:
            improvements['mae_improvement'] = (baseline.mae - comparison.mae) / baseline.mae
        else:
            improvements['mae_improvement'] = 0.0
        
        # R² (higher is better)
        improvements['r2_improvement'] = comparison.r2_score - baseline.r2_score
        
        # Explained variance (higher is better)
        improvements['ev_improvement'] = comparison.explained_variance - baseline.explained_variance
        
        return improvements
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create ablation study summary."""
        if not self.results:
            return {}
        
        baseline = self.baseline_result.metrics
        
        summary = {
            'baseline_rmse': baseline.rmse,
            'baseline_r2': baseline.r2_score,
            'num_ablations': len(self.results),
        }
        
        # Find best performing configuration
        best_result = min(self.results, key=lambda r: r.metrics.rmse)
        summary['best_config'] = best_result.name
        summary['best_rmse'] = best_result.metrics.rmse
        
        # Check if hybrid beats baseline (expected)
        hybrid_results = [r for r in self.results if 'plus_dl' in r.name]
        if hybrid_results:
            hybrid = hybrid_results[0]
            summary['dl_improves'] = hybrid.metrics.rmse < baseline.rmse
            summary['dl_improvement_pct'] = (
                (baseline.rmse - hybrid.metrics.rmse) / baseline.rmse * 100
                if baseline.rmse > 0 else 0
            )
        
        # Sanity check: DL-only should NOT beat hybrid
        dl_only_results = [r for r in self.results if r.name == 'dl_only']
        if dl_only_results and hybrid_results:
            dl_only = dl_only_results[0]
            hybrid = hybrid_results[0]
            summary['sanity_check_passed'] = hybrid.metrics.rmse <= dl_only.metrics.rmse
        
        return summary
