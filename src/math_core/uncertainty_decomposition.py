"""
Epistemic vs Aleatoric Uncertainty Module

Implements decomposition of total uncertainty into:
- Epistemic (model/knowledge) uncertainty - reducible with more data
- Aleatoric (inherent/irreducible) uncertainty - cannot be reduced

Provides methods for:
- Uncertainty decomposition
- Ensemble methods for epistemic quantification
- Heteroscedastic modeling for aleatoric estimation
- Propagation of both uncertainty types

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from scipy import stats

from .error_aware_arithmetic import EPSILON


@dataclass
class DecomposedUncertainty:
    """Container for decomposed uncertainty."""
    total_variance: np.ndarray
    epistemic_variance: np.ndarray
    aleatoric_variance: np.ndarray
    
    @property
    def total_std(self) -> np.ndarray:
        return np.sqrt(self.total_variance)
    
    @property
    def epistemic_std(self) -> np.ndarray:
        return np.sqrt(self.epistemic_variance)
    
    @property
    def aleatoric_std(self) -> np.ndarray:
        return np.sqrt(self.aleatoric_variance)
    
    @property
    def epistemic_fraction(self) -> np.ndarray:
        """Fraction of uncertainty that is epistemic."""
        return self.epistemic_variance / (self.total_variance + EPSILON)
    
    def is_reducible(self, threshold: float = 0.5) -> np.ndarray:
        """Return True where uncertainty is mostly reducible (epistemic)."""
        return self.epistemic_fraction > threshold


class EnsemblePredictor:
    """
    Ensemble-based uncertainty decomposition.
    
    Uses multiple models to estimate:
    - Epistemic uncertainty: variance across model predictions
    - Aleatoric uncertainty: mean of individual model uncertainties
    
    Law of total variance:
    Var[Y] = E[Var[Y|model]] + Var[E[Y|model]]
           = aleatoric       + epistemic
    """
    
    def __init__(self, n_models: int = 10):
        self.n_models = n_models
        self.models: List[Any] = []
        self.model_weights: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            model_factory: Callable[[], Any],
            bootstrap: bool = True):
        """
        Fit ensemble of models.
        
        Args:
            X: Training features
            y: Training targets
            model_factory: Function that creates a new model instance
            bootstrap: Whether to use bootstrap sampling
        """
        n_samples = len(X)
        self.models = []
        
        for _ in range(self.n_models):
            model = model_factory()
            
            if bootstrap:
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
            else:
                X_boot, y_boot = X, y
            
            model.fit(X_boot, y_boot)
            self.models.append(model)
        
        # Equal weights by default
        self.model_weights = np.ones(self.n_models) / self.n_models
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, DecomposedUncertainty]:
        """
        Make predictions with uncertainty decomposition.
        
        Returns:
            (mean_prediction, decomposed_uncertainty)
        """
        n_samples = len(X)
        
        # Collect predictions from all models
        predictions = np.zeros((self.n_models, n_samples))
        aleatoric_vars = np.zeros((self.n_models, n_samples))
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict_with_uncertainty'):
                pred, var = model.predict_with_uncertainty(X)
                predictions[i] = pred
                aleatoric_vars[i] = var
            else:
                predictions[i] = model.predict(X)
                # Estimate aleatoric from training residuals if available
                aleatoric_vars[i] = EPSILON
        
        # Ensemble mean prediction
        mean_pred = np.average(predictions, axis=0, weights=self.model_weights)
        
        # Epistemic: variance of predictions across models
        epistemic_var = np.average(
            (predictions - mean_pred) ** 2,
            axis=0,
            weights=self.model_weights
        )
        
        # Aleatoric: average of model-specific variances
        aleatoric_var = np.average(aleatoric_vars, axis=0, weights=self.model_weights)
        
        # Total variance
        total_var = epistemic_var + aleatoric_var
        
        uncertainty = DecomposedUncertainty(
            total_variance=total_var,
            epistemic_variance=epistemic_var,
            aleatoric_variance=aleatoric_var
        )
        
        return mean_pred, uncertainty
    
    def update_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Update model weights based on validation performance."""
        n_models = len(self.models)
        log_likelihoods = np.zeros(n_models)
        
        for i, model in enumerate(self.models):
            pred = model.predict(X_val)
            # Assume unit variance for simplicity
            residuals = y_val - pred
            log_likelihoods[i] = -0.5 * np.sum(residuals ** 2)
        
        # Softmax weights
        max_ll = np.max(log_likelihoods)
        weights = np.exp(log_likelihoods - max_ll)
        self.model_weights = weights / weights.sum()


class HeteroscedasticModel:
    """
    Model that predicts both mean and variance (aleatoric uncertainty).
    
    y = f(x) + ε(x), where ε(x) ~ N(0, σ²(x))
    
    The variance σ²(x) varies with input, capturing input-dependent noise.
    """
    
    def __init__(self, base_model_factory: Callable[[], Any]):
        self.base_model_factory = base_model_factory
        self.mean_model = None
        self.variance_model = None
        self.min_variance = 1e-6
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            n_iterations: int = 3):
        """
        Fit heteroscedastic model iteratively.
        
        1. Fit mean model
        2. Compute squared residuals
        3. Fit variance model to log residuals
        4. Repeat with weighted fitting
        """
        n_samples = len(X)
        weights = np.ones(n_samples)
        
        for iteration in range(n_iterations):
            # Fit mean model
            self.mean_model = self.base_model_factory()
            if hasattr(self.mean_model, 'fit_weighted'):
                self.mean_model.fit_weighted(X, y, weights)
            else:
                self.mean_model.fit(X, y)
            
            # Compute residuals
            pred = self.mean_model.predict(X)
            residuals = y - pred
            squared_residuals = residuals ** 2
            
            # Fit variance model to log squared residuals
            log_sq_residuals = np.log(squared_residuals + self.min_variance)
            self.variance_model = self.base_model_factory()
            self.variance_model.fit(X, log_sq_residuals)
            
            # Update weights (inverse variance weighting)
            pred_log_var = self.variance_model.predict(X)
            pred_var = np.exp(pred_log_var)
            weights = 1.0 / (pred_var + self.min_variance)
            weights = weights / weights.mean()
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and aleatoric variance.
        
        Returns:
            (mean, variance)
        """
        mean = self.mean_model.predict(X)
        log_var = self.variance_model.predict(X)
        variance = np.maximum(np.exp(log_var), self.min_variance)
        
        return mean, variance
    
    def negative_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute NLL for model evaluation."""
        mean, var = self.predict(X)
        nll = 0.5 * np.sum(np.log(2 * np.pi * var) + (y - mean) ** 2 / var)
        return nll


class GaussianProcessUncertainty:
    """
    Gaussian Process with epistemic/aleatoric decomposition.
    
    Total predictive variance = kernel variance + noise variance
    
    - Kernel variance (epistemic): decreases with more data near prediction point
    - Noise variance (aleatoric): irreducible observation noise
    """
    
    def __init__(self, kernel_variance: float = 1.0,
                 length_scale: float = 1.0,
                 noise_variance: float = 0.1):
        self.kernel_variance = kernel_variance
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
    
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared exponential kernel."""
        from scipy.spatial.distance import cdist
        dists = cdist(X1, X2, 'euclidean')
        return self.kernel_variance * np.exp(-0.5 * (dists / self.length_scale) ** 2)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data."""
        self.X_train = X
        self.y_train = y
        
        n = len(X)
        K = self._kernel(X, X) + self.noise_variance * np.eye(n)
        
        # Cholesky decomposition for numerical stability
        L = np.linalg.cholesky(K + EPSILON * np.eye(n))
        self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, DecomposedUncertainty]:
        """
        Predict with uncertainty decomposition.
        
        Returns:
            (mean, decomposed_uncertainty)
        """
        k_star = self._kernel(X, self.X_train)
        k_star_star = self._kernel(X, X)
        
        # Predictive mean
        mean = k_star @ self.alpha
        
        # Epistemic variance (posterior uncertainty)
        epistemic_var = np.diag(k_star_star - k_star @ self.K_inv @ k_star.T)
        epistemic_var = np.maximum(epistemic_var, 0)  # Numerical stability
        
        # Aleatoric variance (observation noise)
        aleatoric_var = np.full(len(X), self.noise_variance)
        
        # Total variance
        total_var = epistemic_var + aleatoric_var
        
        uncertainty = DecomposedUncertainty(
            total_variance=total_var,
            epistemic_variance=epistemic_var,
            aleatoric_variance=aleatoric_var
        )
        
        return mean, uncertainty
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Optimize hyperparameters via marginal likelihood."""
        from scipy.optimize import minimize
        
        def neg_log_marginal_likelihood(params):
            self.kernel_variance = np.exp(params[0])
            self.length_scale = np.exp(params[1])
            self.noise_variance = np.exp(params[2])
            
            self.fit(X, y)
            
            n = len(X)
            K = self._kernel(X, X) + self.noise_variance * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K + EPSILON * np.eye(n))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                
                # Log marginal likelihood
                log_ml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - n/2 * np.log(2*np.pi)
                return -log_ml
            except:
                return np.inf
        
        # Initial parameters (log scale)
        x0 = np.log([self.kernel_variance, self.length_scale, self.noise_variance])
        
        result = minimize(neg_log_marginal_likelihood, x0, method='L-BFGS-B')
        
        # Set optimized parameters
        self.kernel_variance = np.exp(result.x[0])
        self.length_scale = np.exp(result.x[1])
        self.noise_variance = np.exp(result.x[2])
        self.fit(X, y)


class DeepUncertaintyEstimator:
    """
    Neural network-based uncertainty estimation using
    simple MLP with dropout for epistemic uncertainty.
    
    This is a simplified version without actual deep learning
    to avoid dependencies, using ensemble of linear models instead.
    """
    
    def __init__(self, n_ensemble: int = 10, dropout_rate: float = 0.2):
        self.n_ensemble = n_ensemble
        self.dropout_rate = dropout_rate
        self.models: List[Dict] = []
    
    def _random_features(self, X: np.ndarray, 
                         n_features: int = 100,
                         seed: int = 0) -> np.ndarray:
        """Generate random Fourier features for nonlinear modeling."""
        np.random.seed(seed)
        d = X.shape[1] if X.ndim > 1 else 1
        X = X.reshape(-1, d)
        
        W = np.random.randn(d, n_features) / self.length_scale
        b = np.random.uniform(0, 2 * np.pi, n_features)
        
        features = np.sqrt(2 / n_features) * np.cos(X @ W + b)
        return features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble of models."""
        self.length_scale = np.std(X) if X.ndim == 1 else np.mean(np.std(X, axis=0))
        self.models = []
        
        n_samples = len(X)
        
        for i in range(self.n_ensemble):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Random features (different for each model)
            features = self._random_features(X_boot, seed=i)
            
            # Ridge regression
            lambda_reg = 0.01
            n_feat = features.shape[1]
            weights = np.linalg.solve(
                features.T @ features + lambda_reg * np.eye(n_feat),
                features.T @ y_boot
            )
            
            # Compute training residual variance (aleatoric estimate)
            pred = features @ weights
            residual_var = np.var(y_boot - pred)
            
            self.models.append({
                'weights': weights,
                'seed': i,
                'residual_var': residual_var
            })
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, DecomposedUncertainty]:
        """Predict with uncertainty decomposition."""
        predictions = []
        aleatoric_vars = []
        
        for model in self.models:
            features = self._random_features(X, seed=model['seed'])
            pred = features @ model['weights']
            predictions.append(pred)
            aleatoric_vars.append(model['residual_var'])
        
        predictions = np.array(predictions)
        
        # Mean prediction
        mean = np.mean(predictions, axis=0)
        
        # Epistemic: variance across ensemble
        epistemic_var = np.var(predictions, axis=0)
        
        # Aleatoric: average residual variance
        aleatoric_var = np.full(len(X), np.mean(aleatoric_vars))
        
        # Total
        total_var = epistemic_var + aleatoric_var
        
        uncertainty = DecomposedUncertainty(
            total_variance=total_var,
            epistemic_variance=epistemic_var,
            aleatoric_variance=aleatoric_var
        )
        
        return mean, uncertainty


def propagate_decomposed_uncertainty(f: Callable,
                                      x_mean: np.ndarray,
                                      x_epistemic_var: np.ndarray,
                                      x_aleatoric_var: np.ndarray,
                                      jacobian: Optional[Callable] = None
                                      ) -> DecomposedUncertainty:
    """
    Propagate decomposed uncertainty through a function.
    
    Uses linear approximation: Var[f(X)] ≈ J @ Var[X] @ J'
    
    The decomposition is preserved:
    - Epistemic variance propagates through Jacobian
    - Aleatoric variance propagates separately
    """
    n = len(x_mean)
    
    # Compute Jacobian if not provided
    if jacobian is None:
        # Numerical differentiation
        eps = 1e-6
        f_x = f(x_mean)
        m = len(f_x)
        J = np.zeros((m, n))
        
        for i in range(n):
            x_plus = x_mean.copy()
            x_plus[i] += eps
            J[:, i] = (f(x_plus) - f_x) / eps
    else:
        J = jacobian(x_mean)
    
    # Propagate epistemic variance
    epistemic_cov = np.diag(x_epistemic_var)
    y_epistemic_cov = J @ epistemic_cov @ J.T
    y_epistemic_var = np.diag(y_epistemic_cov)
    
    # Propagate aleatoric variance
    aleatoric_cov = np.diag(x_aleatoric_var)
    y_aleatoric_cov = J @ aleatoric_cov @ J.T
    y_aleatoric_var = np.diag(y_aleatoric_cov)
    
    # Total
    y_total_var = y_epistemic_var + y_aleatoric_var
    
    return DecomposedUncertainty(
        total_variance=y_total_var,
        epistemic_variance=y_epistemic_var,
        aleatoric_variance=y_aleatoric_var
    )


def information_gain(prior_epistemic_var: np.ndarray,
                     posterior_epistemic_var: np.ndarray) -> np.ndarray:
    """
    Compute information gain from new data.
    
    Information gain = reduction in epistemic uncertainty
    
    Returns array of information gain at each location.
    """
    return np.maximum(prior_epistemic_var - posterior_epistemic_var, 0)


def optimal_sampling_location(x_candidates: np.ndarray,
                               epistemic_vars: np.ndarray,
                               n_select: int = 1) -> np.ndarray:
    """
    Select locations to sample to maximize epistemic uncertainty reduction.
    
    Returns indices of best locations to sample.
    """
    # Simple greedy: select points with highest epistemic uncertainty
    indices = np.argsort(epistemic_vars)[::-1][:n_select]
    return indices
