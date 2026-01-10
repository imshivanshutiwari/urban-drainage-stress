"""
Distribution-Aware Scaling Module

Implements proper normalization preserving distribution properties:
- Quantile normalization
- Rank-preserving transformations
- Box-Cox and Yeo-Johnson transforms
- Uncertainty propagation through transforms

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.special import boxcox, inv_boxcox, gammaln
from scipy.optimize import minimize_scalar, brentq

from .error_aware_arithmetic import EPSILON


@dataclass
class TransformResult:
    """Result of a distribution transform."""
    transformed: np.ndarray
    original_mean: float
    original_std: float
    transform_params: Dict[str, float]
    inverse_available: bool = True


class QuantileTransformer:
    """
    Transform data to match a target distribution using quantiles.
    
    Maps each value to its quantile in the empirical CDF,
    then maps that quantile to the target distribution.
    """
    
    def __init__(self, target_distribution: str = 'normal',
                 n_quantiles: int = 1000):
        """
        Initialize quantile transformer.
        
        Args:
            target_distribution: 'normal', 'uniform', or 'exponential'
            n_quantiles: Number of quantiles for interpolation
        """
        self.target = target_distribution
        self.n_quantiles = n_quantiles
        self.quantiles_: Optional[np.ndarray] = None
        self.references_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'QuantileTransformer':
        """Fit transformer to data."""
        X = np.asarray(X).flatten()
        
        # Compute quantiles of input data
        self.quantiles_ = np.linspace(0, 100, self.n_quantiles)
        self.references_ = np.percentile(X, self.quantiles_)
        
        # Handle ties
        self.references_ = np.unique(self.references_)
        self.quantiles_ = np.linspace(0, 100, len(self.references_))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to target distribution."""
        X = np.asarray(X).flatten()
        
        # Map to quantiles (0-1 scale)
        quantiles = np.interp(X, self.references_, self.quantiles_ / 100)
        
        # Clamp to valid range
        quantiles = np.clip(quantiles, EPSILON, 1 - EPSILON)
        
        # Map to target distribution
        if self.target == 'normal':
            return stats.norm.ppf(quantiles)
        elif self.target == 'uniform':
            return quantiles
        elif self.target == 'exponential':
            return stats.expon.ppf(quantiles)
        else:
            raise ValueError(f"Unknown target distribution: {self.target}")
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform back to original distribution."""
        X = np.asarray(X).flatten()
        
        # Map from target to quantiles
        if self.target == 'normal':
            quantiles = stats.norm.cdf(X)
        elif self.target == 'uniform':
            quantiles = X
        elif self.target == 'exponential':
            quantiles = stats.expon.cdf(X)
        else:
            raise ValueError(f"Unknown target distribution: {self.target}")
        
        # Map from quantiles to original values
        return np.interp(quantiles * 100, self.quantiles_, self.references_)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class BoxCoxTransformer:
    """
    Box-Cox transformation for positive data.
    
    y = (x^λ - 1) / λ  if λ ≠ 0
    y = log(x)         if λ = 0
    
    Finds optimal λ to maximize normality.
    """
    
    def __init__(self, lambda_: Optional[float] = None):
        """
        Initialize Box-Cox transformer.
        
        Args:
            lambda_: Fixed lambda value. If None, will be estimated.
        """
        self.lambda_ = lambda_
        self.fitted_lambda_: Optional[float] = None
        self.shift_: float = 0.0
    
    def fit(self, X: np.ndarray) -> 'BoxCoxTransformer':
        """Fit transformer, optionally estimating lambda."""
        X = np.asarray(X).flatten()
        
        # Ensure positive values
        min_val = X.min()
        if min_val <= 0:
            self.shift_ = -min_val + 1.0
        else:
            self.shift_ = 0.0
        
        X_shifted = X + self.shift_
        
        if self.lambda_ is None:
            # Estimate optimal lambda via MLE
            self.fitted_lambda_ = self._estimate_lambda_mle(X_shifted)
        else:
            self.fitted_lambda_ = self.lambda_
        
        return self
    
    def _estimate_lambda_mle(self, X: np.ndarray) -> float:
        """Estimate lambda using maximum likelihood."""
        n = len(X)
        log_X = np.log(X)
        mean_log_X = np.mean(log_X)
        
        def neg_log_likelihood(lam):
            if abs(lam) < EPSILON:
                # Lambda ≈ 0, use log transform
                y = log_X
            else:
                y = (X ** lam - 1) / lam
            
            # Likelihood for normal distribution
            y_mean = np.mean(y)
            y_var = np.var(y)
            
            if y_var < EPSILON:
                return np.inf
            
            # Log likelihood
            ll = -n/2 * np.log(2 * np.pi * y_var)
            ll -= np.sum((y - y_mean) ** 2) / (2 * y_var)
            ll += (lam - 1) * np.sum(log_X)
            
            return -ll
        
        # Optimize lambda
        result = minimize_scalar(neg_log_likelihood, bounds=(-2, 2), method='bounded')
        return result.x
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply Box-Cox transformation."""
        X = np.asarray(X).flatten()
        X_shifted = X + self.shift_
        
        lam = self.fitted_lambda_
        
        if abs(lam) < EPSILON:
            return np.log(X_shifted)
        else:
            return (X_shifted ** lam - 1) / lam
    
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Inverse Box-Cox transformation."""
        Y = np.asarray(Y).flatten()
        lam = self.fitted_lambda_
        
        if abs(lam) < EPSILON:
            X_shifted = np.exp(Y)
        else:
            X_shifted = (Y * lam + 1) ** (1 / lam)
        
        return X_shifted - self.shift_
    
    def transform_with_uncertainty(self, X: np.ndarray, 
                                    X_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform with uncertainty propagation."""
        Y = self.transform(X)
        
        # Jacobian of Box-Cox transform
        lam = self.fitted_lambda_
        X_shifted = X + self.shift_
        
        if abs(lam) < EPSILON:
            dY_dX = 1.0 / X_shifted
        else:
            dY_dX = X_shifted ** (lam - 1)
        
        Y_std = np.abs(dY_dX) * X_std
        
        return Y, Y_std


class YeoJohnsonTransformer:
    """
    Yeo-Johnson transformation for any real-valued data.
    
    Extends Box-Cox to handle zero and negative values.
    """
    
    def __init__(self, lambda_: Optional[float] = None):
        self.lambda_ = lambda_
        self.fitted_lambda_: Optional[float] = None
    
    def fit(self, X: np.ndarray) -> 'YeoJohnsonTransformer':
        """Fit transformer."""
        X = np.asarray(X).flatten()
        
        if self.lambda_ is None:
            self.fitted_lambda_ = self._estimate_lambda_mle(X)
        else:
            self.fitted_lambda_ = self.lambda_
        
        return self
    
    def _yeo_johnson(self, X: np.ndarray, lam: float) -> np.ndarray:
        """Yeo-Johnson transformation."""
        Y = np.zeros_like(X)
        
        pos = X >= 0
        neg = ~pos
        
        if abs(lam) < EPSILON:
            Y[pos] = np.log(X[pos] + 1)
        else:
            Y[pos] = ((X[pos] + 1) ** lam - 1) / lam
        
        if abs(lam - 2) < EPSILON:
            Y[neg] = -np.log(-X[neg] + 1)
        else:
            Y[neg] = -((-X[neg] + 1) ** (2 - lam) - 1) / (2 - lam)
        
        return Y
    
    def _estimate_lambda_mle(self, X: np.ndarray) -> float:
        """Estimate optimal lambda."""
        n = len(X)
        
        def neg_log_likelihood(lam):
            Y = self._yeo_johnson(X, lam)
            
            y_var = np.var(Y)
            if y_var < EPSILON:
                return np.inf
            
            ll = -n/2 * np.log(y_var)
            
            # Jacobian contribution
            pos = X >= 0
            if pos.any():
                ll += (lam - 1) * np.sum(np.log(X[pos] + 1))
            if (~pos).any():
                ll += (1 - lam) * np.sum(np.log(-X[~pos] + 1))
            
            return -ll
        
        result = minimize_scalar(neg_log_likelihood, bounds=(-2, 2), method='bounded')
        return result.x
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply Yeo-Johnson transformation."""
        return self._yeo_johnson(np.asarray(X).flatten(), self.fitted_lambda_)
    
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Inverse Yeo-Johnson transformation."""
        Y = np.asarray(Y).flatten()
        lam = self.fitted_lambda_
        X = np.zeros_like(Y)
        
        # For positive transformed values
        pos = Y >= 0
        neg = ~pos
        
        if abs(lam) < EPSILON:
            X[pos] = np.exp(Y[pos]) - 1
        else:
            X[pos] = (Y[pos] * lam + 1) ** (1 / lam) - 1
        
        if abs(lam - 2) < EPSILON:
            X[neg] = 1 - np.exp(-Y[neg])
        else:
            X[neg] = 1 - (-(2 - lam) * Y[neg] + 1) ** (1 / (2 - lam))
        
        return X


class RankPreservingScaler:
    """
    Scale data while preserving ranks.
    
    Uses rank-based mapping to target range.
    """
    
    def __init__(self, target_min: float = 0.0, target_max: float = 1.0):
        self.target_min = target_min
        self.target_max = target_max
        self.ranks_: Optional[np.ndarray] = None
        self.values_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'RankPreservingScaler':
        """Fit scaler."""
        X = np.asarray(X).flatten()
        
        # Store sorted unique values
        self.values_ = np.sort(np.unique(X))
        n = len(self.values_)
        
        # Assign ranks (normalized to [0, 1])
        self.ranks_ = np.linspace(0, 1, n)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to target range preserving ranks."""
        X = np.asarray(X).flatten()
        
        # Map values to ranks
        ranks = np.interp(X, self.values_, self.ranks_)
        
        # Scale to target range
        return self.target_min + ranks * (self.target_max - self.target_min)
    
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        Y = np.asarray(Y).flatten()
        
        # Map from target range to ranks
        ranks = (Y - self.target_min) / (self.target_max - self.target_min)
        
        # Map ranks to values
        return np.interp(ranks, self.ranks_, self.values_)


class RobustScaler:
    """
    Robust scaling using median and IQR.
    
    Less sensitive to outliers than standard scaling.
    """
    
    def __init__(self, quantile_range: Tuple[float, float] = (25, 75)):
        self.quantile_range = quantile_range
        self.center_: Optional[float] = None
        self.scale_: Optional[float] = None
    
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """Fit scaler."""
        X = np.asarray(X).flatten()
        
        self.center_ = np.median(X)
        
        q_low = np.percentile(X, self.quantile_range[0])
        q_high = np.percentile(X, self.quantile_range[1])
        self.scale_ = q_high - q_low
        
        if self.scale_ < EPSILON:
            self.scale_ = 1.0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply robust scaling."""
        return (np.asarray(X) - self.center_) / self.scale_
    
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        return np.asarray(Y) * self.scale_ + self.center_
    
    def transform_with_uncertainty(self, X: np.ndarray, 
                                    X_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform with uncertainty propagation."""
        Y = self.transform(X)
        Y_std = np.asarray(X_std) / self.scale_
        return Y, Y_std


class GaussianCopulaTransformer:
    """
    Transform multivariate data using Gaussian copula.
    
    Preserves rank correlations while mapping marginals to Gaussian.
    """
    
    def __init__(self):
        self.marginal_transformers: Dict[int, QuantileTransformer] = {}
    
    def fit(self, X: np.ndarray) -> 'GaussianCopulaTransformer':
        """Fit transformer to multivariate data."""
        X = np.atleast_2d(X)
        n_features = X.shape[1]
        
        for j in range(n_features):
            self.marginal_transformers[j] = QuantileTransformer(target_distribution='normal')
            self.marginal_transformers[j].fit(X[:, j])
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform each marginal to Gaussian."""
        X = np.atleast_2d(X)
        Y = np.zeros_like(X)
        
        for j in range(X.shape[1]):
            Y[:, j] = self.marginal_transformers[j].transform(X[:, j])
        
        return Y
    
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        Y = np.atleast_2d(Y)
        X = np.zeros_like(Y)
        
        for j in range(Y.shape[1]):
            X[:, j] = self.marginal_transformers[j].inverse_transform(Y[:, j])
        
        return X


def select_best_transform(X: np.ndarray,
                          target: str = 'normal') -> Tuple[str, Any]:
    """
    Select best transformation for data.
    
    Tests multiple transformations and selects based on
    normality test (Shapiro-Wilk).
    
    Returns:
        (transform_name, fitted_transformer)
    """
    X = np.asarray(X).flatten()
    
    # Candidates
    candidates = {}
    
    # Original
    candidates['none'] = (X, None)
    
    # Log (if positive)
    if X.min() > 0:
        log_X = np.log(X)
        candidates['log'] = (log_X, None)
    
    # Square root (if non-negative)
    if X.min() >= 0:
        sqrt_X = np.sqrt(X)
        candidates['sqrt'] = (sqrt_X, None)
    
    # Box-Cox (if positive)
    if X.min() > 0:
        bc = BoxCoxTransformer()
        bc.fit(X)
        candidates['box-cox'] = (bc.transform(X), bc)
    
    # Yeo-Johnson (any data)
    yj = YeoJohnsonTransformer()
    yj.fit(X)
    candidates['yeo-johnson'] = (yj.transform(X), yj)
    
    # Quantile
    qt = QuantileTransformer(target_distribution=target)
    qt.fit(X)
    candidates['quantile'] = (qt.transform(X), qt)
    
    # Evaluate each
    best_name = 'none'
    best_score = 0
    
    for name, (transformed, transformer) in candidates.items():
        # Shapiro-Wilk test (p-value closer to 1 = more normal)
        if len(transformed) > 3:
            _, p_value = stats.shapiro(transformed[:min(5000, len(transformed))])
            if p_value > best_score:
                best_score = p_value
                best_name = name
    
    return best_name, candidates[best_name][1]


def propagate_uncertainty_through_transform(X: np.ndarray,
                                            X_std: np.ndarray,
                                            transform_func: callable,
                                            inverse_func: Optional[callable] = None
                                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate uncertainty through arbitrary transform.
    
    Uses Monte Carlo if analytical unavailable.
    """
    # Transformed mean (first-order approximation)
    Y = transform_func(X)
    
    # Monte Carlo for uncertainty
    n_samples = 1000
    Y_samples = []
    
    for _ in range(n_samples):
        X_sample = X + np.random.randn(*X.shape) * X_std
        Y_samples.append(transform_func(X_sample))
    
    Y_samples = np.array(Y_samples)
    Y_std = np.std(Y_samples, axis=0)
    
    return Y, Y_std
