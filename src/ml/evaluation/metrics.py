"""Evaluation metrics for ST-GNN.

Metrics:
- RMSE on residuals
- Explained variance
- Decision stability improvement
- Uncertainty calibration

Reference: projectfile.md Step 6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Evaluation metrics result."""
    rmse: float
    mae: float
    explained_variance: float
    r2_score: float
    correlation: float
    uncertainty_calibration: float
    decision_stability: float
    coverage_95: float
    mean_absolute_residual: float
    metrics_dict: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'explained_variance': self.explained_variance,
            'r2_score': self.r2_score,
            'correlation': self.correlation,
            'uncertainty_calibration': self.uncertainty_calibration,
            'decision_stability': self.decision_stability,
            'coverage_95': self.coverage_95,
            'mean_absolute_residual': self.mean_absolute_residual,
            **self.metrics_dict,
        }


def compute_metrics(
    pred_residual: np.ndarray,
    target_residual: np.ndarray,
    pred_uncertainty: Optional[np.ndarray] = None,
    bayesian_stress: Optional[np.ndarray] = None,
    observed_stress: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> MetricsResult:
    """Compute comprehensive evaluation metrics.
    
    Args:
        pred_residual: predicted residuals
        target_residual: ground truth residuals
        pred_uncertainty: predicted uncertainty (std dev)
        bayesian_stress: baseline Bayesian stress
        observed_stress: observed stress
        mask: validity mask
        
    Returns:
        MetricsResult with all metrics
    """
    # Apply mask
    if mask is not None:
        pred_residual = pred_residual[mask]
        target_residual = target_residual[mask]
        if pred_uncertainty is not None:
            pred_uncertainty = pred_uncertainty[mask]
        if bayesian_stress is not None and observed_stress is not None:
            bayesian_stress = bayesian_stress[mask]
            observed_stress = observed_stress[mask]
    
    # Flatten
    pred = pred_residual.flatten()
    target = target_residual.flatten()
    
    # Remove NaNs
    valid = ~np.isnan(pred) & ~np.isnan(target)
    pred = pred[valid]
    target = target[valid]
    
    if len(pred) == 0:
        logger.warning("No valid predictions for metrics")
        return MetricsResult(
            rmse=np.nan, mae=np.nan, explained_variance=np.nan,
            r2_score=np.nan, correlation=np.nan, uncertainty_calibration=np.nan,
            decision_stability=np.nan, coverage_95=np.nan, mean_absolute_residual=np.nan,
        )
    
    # 1. RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    
    # 2. MAE
    mae = np.mean(np.abs(pred - target))
    
    # 3. Explained variance
    target_var = np.var(target)
    if target_var > 0:
        residual_var = np.var(target - pred)
        explained_variance = 1.0 - residual_var / target_var
    else:
        explained_variance = 0.0
    
    # 4. R² score
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2_score = 1.0 - ss_res / (ss_tot + 1e-8)
    
    # 5. Correlation
    if len(pred) > 1:
        correlation, _ = pearsonr(pred, target)
    else:
        correlation = np.nan
    
    # 6. Uncertainty calibration
    if pred_uncertainty is not None:
        uncertainty_calibration = compute_uncertainty_calibration(
            pred, target, pred_uncertainty[valid]
        )
        coverage_95 = compute_coverage(pred, target, pred_uncertainty[valid], level=0.95)
    else:
        uncertainty_calibration = np.nan
        coverage_95 = np.nan
    
    # 7. Decision stability (if stress provided)
    if bayesian_stress is not None and observed_stress is not None:
        decision_stability = compute_decision_stability(
            bayesian_stress.flatten()[valid],
            observed_stress.flatten()[valid],
            pred,
        )
    else:
        decision_stability = np.nan
    
    # 8. Mean absolute residual
    mean_absolute_residual = np.mean(np.abs(target))
    
    return MetricsResult(
        rmse=float(rmse),
        mae=float(mae),
        explained_variance=float(explained_variance),
        r2_score=float(r2_score),
        correlation=float(correlation),
        uncertainty_calibration=float(uncertainty_calibration),
        decision_stability=float(decision_stability),
        coverage_95=float(coverage_95),
        mean_absolute_residual=float(mean_absolute_residual),
    )


def compute_uncertainty_calibration(
    pred: np.ndarray,
    target: np.ndarray,
    uncertainty: np.ndarray,
) -> float:
    """Compute uncertainty calibration score.
    
    A well-calibrated model should have:
    - High uncertainty when predictions are wrong
    - Low uncertainty when predictions are accurate
    
    Returns negative Pearson correlation between |error| and uncertainty.
    Perfect calibration = 1.0 (high uncertainty = high error)
    """
    errors = np.abs(pred - target)
    uncertainty = np.maximum(uncertainty, 1e-8)
    
    if len(errors) < 2:
        return np.nan
    
    # Correlation between error and uncertainty
    corr, _ = pearsonr(errors, uncertainty)
    
    return float(corr)


def compute_coverage(
    pred: np.ndarray,
    target: np.ndarray,
    uncertainty: np.ndarray,
    level: float = 0.95,
) -> float:
    """Compute coverage of credible intervals.
    
    For a well-calibrated model with 95% CI, 95% of true values
    should fall within the predicted interval.
    
    Args:
        pred: predicted values
        target: true values
        uncertainty: predicted std dev
        level: confidence level
        
    Returns:
        coverage: fraction of targets within CI
    """
    from scipy.stats import norm
    
    z = norm.ppf((1 + level) / 2)
    lower = pred - z * uncertainty
    upper = pred + z * uncertainty
    
    within = (target >= lower) & (target <= upper)
    coverage = np.mean(within)
    
    return float(coverage)


def compute_decision_stability(
    bayesian_stress: np.ndarray,
    observed_stress: np.ndarray,
    residual: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute decision stability improvement.
    
    Measures how much the ML residual reduces decision changes
    between Bayesian prediction and observed stress.
    
    Args:
        bayesian_stress: baseline Bayesian stress
        observed_stress: ground truth stress
        residual: predicted residual correction
        threshold: decision threshold
        
    Returns:
        stability: fraction of decisions that become consistent
    """
    # Decisions from Bayesian alone
    bayesian_decision = bayesian_stress > threshold
    observed_decision = observed_stress > threshold
    
    # Decisions with ML correction
    corrected_stress = bayesian_stress + residual
    corrected_decision = corrected_stress > threshold
    
    # How many incorrect Bayesian decisions are fixed?
    bayesian_wrong = bayesian_decision != observed_decision
    corrected_correct = corrected_decision == observed_decision
    
    if bayesian_wrong.sum() == 0:
        return 1.0  # No wrong decisions to fix
    
    # Fraction of wrong decisions that are corrected
    fixed = bayesian_wrong & corrected_correct
    stability = fixed.sum() / bayesian_wrong.sum()
    
    return float(stability)


def compute_spatial_metrics(
    pred_residual: np.ndarray,  # (H, W)
    target_residual: np.ndarray,  # (H, W)
) -> Dict[str, float]:
    """Compute spatial coherence metrics."""
    from scipy.ndimage import sobel
    
    # Gradient magnitude for smoothness
    pred_grad = np.sqrt(sobel(pred_residual, 0)**2 + sobel(pred_residual, 1)**2)
    target_grad = np.sqrt(sobel(target_residual, 0)**2 + sobel(target_residual, 1)**2)
    
    pred_smoothness = np.mean(pred_grad)
    target_smoothness = np.mean(target_grad)
    
    return {
        'pred_spatial_roughness': float(pred_smoothness),
        'target_spatial_roughness': float(target_smoothness),
        'roughness_ratio': float(pred_smoothness / (target_smoothness + 1e-8)),
    }
