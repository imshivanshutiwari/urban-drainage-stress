"""
Mixture Distributions Module

Implements flexible mixture models for non-Gaussian data:
- Gaussian Mixture Models with full covariance
- Mixture of Experts
- Finite mixture for multi-modal uncertainty
- Automatic component selection

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
from scipy.special import logsumexp, gammaln, digamma
from scipy.stats import multivariate_normal

from .error_aware_arithmetic import EPSILON


@dataclass
class MixtureComponent:
    """A single component in a mixture model."""
    weight: float
    mean: np.ndarray
    covariance: np.ndarray
    precision: Optional[np.ndarray] = None
    log_det_cov: Optional[float] = None
    
    def __post_init__(self):
        if self.precision is None:
            self.precision = np.linalg.inv(self.covariance + EPSILON * np.eye(len(self.mean)))
        if self.log_det_cov is None:
            sign, self.log_det_cov = np.linalg.slogdet(self.covariance)
    
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute log probability density at x."""
        d = len(self.mean)
        diff = x - self.mean
        
        if x.ndim == 1:
            mahal = diff @ self.precision @ diff
        else:
            mahal = np.sum(diff @ self.precision * diff, axis=1)
        
        return -0.5 * (d * np.log(2 * np.pi) + self.log_det_cov + mahal)
    
    def sample(self, n: int) -> np.ndarray:
        """Sample from this component."""
        return np.random.multivariate_normal(self.mean, self.covariance, n)


class GaussianMixtureModel:
    """
    Full Gaussian Mixture Model with EM fitting.
    
    Supports:
    - Full, diagonal, or spherical covariance
    - BIC/AIC for model selection
    - Posterior responsibilities
    """
    
    def __init__(self, n_components: int = 3,
                 covariance_type: str = 'full',
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 reg_covar: float = 1e-6):
        """
        Initialize GMM.
        
        Args:
            n_components: Number of mixture components
            covariance_type: 'full', 'diag', or 'spherical'
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            reg_covar: Regularization for covariance
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        
        self.components: List[MixtureComponent] = []
        self.converged = False
        self.n_iter = 0
        self.log_likelihood_history: List[float] = []
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureModel':
        """
        Fit GMM using EM algorithm.
        
        Args:
            X: Data array of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # Initialize with k-means++
        self._initialize_kmeans(X)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            log_resp, log_likelihood = self._e_step(X)
            
            self.log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged = True
                self.n_iter = iteration + 1
                break
            
            prev_log_likelihood = log_likelihood
            
            # M-step: update parameters
            self._m_step(X, np.exp(log_resp))
        
        return self
    
    def _initialize_kmeans(self, X: np.ndarray):
        """Initialize using k-means++ style."""
        n_samples, n_features = X.shape
        
        # Select first center randomly
        centers = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_components):
            # Compute distances to nearest center
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            # Probability proportional to distance squared
            probs = dists / dists.sum()
            # Select next center
            idx = np.random.choice(n_samples, p=probs)
            centers.append(X[idx])
        
        centers = np.array(centers)
        
        # Initialize components
        self.components = []
        overall_cov = np.cov(X.T) + self.reg_covar * np.eye(n_features)
        
        for k in range(self.n_components):
            self.components.append(MixtureComponent(
                weight=1.0 / self.n_components,
                mean=centers[k],
                covariance=overall_cov.copy()
            ))
    
    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """E-step: compute log responsibilities."""
        n_samples = len(X)
        log_prob = np.zeros((n_samples, self.n_components))
        
        for k, comp in enumerate(self.components):
            log_prob[:, k] = np.log(comp.weight + EPSILON) + comp.log_prob(X)
        
        # Normalize to get responsibilities
        log_norm = logsumexp(log_prob, axis=1, keepdims=True)
        log_resp = log_prob - log_norm
        
        # Total log likelihood
        log_likelihood = np.sum(log_norm)
        
        return log_resp, log_likelihood
    
    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        """M-step: update component parameters."""
        n_samples, n_features = X.shape
        
        for k in range(self.n_components):
            nk = resp[:, k].sum() + EPSILON
            
            # Update weight
            self.components[k].weight = nk / n_samples
            
            # Update mean
            self.components[k].mean = np.sum(resp[:, k:k+1] * X, axis=0) / nk
            
            # Update covariance
            diff = X - self.components[k].mean
            
            if self.covariance_type == 'full':
                cov = (resp[:, k:k+1] * diff).T @ diff / nk
                cov += self.reg_covar * np.eye(n_features)
            elif self.covariance_type == 'diag':
                cov = np.diag(np.sum(resp[:, k:k+1] * diff ** 2, axis=0) / nk)
                cov += self.reg_covar * np.eye(n_features)
            else:  # spherical
                var = np.sum(resp[:, k:k+1] * np.sum(diff ** 2, axis=1, keepdims=True)) / (nk * n_features)
                cov = (var + self.reg_covar) * np.eye(n_features)
            
            self.components[k].covariance = cov
            self.components[k].precision = np.linalg.inv(cov)
            sign, self.components[k].log_det_cov = np.linalg.slogdet(cov)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely component for each sample."""
        log_resp, _ = self._e_step(X)
        return np.argmax(log_resp, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior probabilities for each component."""
        log_resp, _ = self._e_step(X)
        return np.exp(log_resp)
    
    def score(self, X: np.ndarray) -> float:
        """Compute average log likelihood."""
        _, log_likelihood = self._e_step(X)
        return log_likelihood / len(X)
    
    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the mixture.
        
        Returns:
            (samples, component_labels)
        """
        weights = np.array([c.weight for c in self.components])
        components = np.random.choice(self.n_components, size=n_samples, p=weights)
        
        samples = np.zeros((n_samples, len(self.components[0].mean)))
        for k in range(self.n_components):
            mask = components == k
            n_k = mask.sum()
            if n_k > 0:
                samples[mask] = self.components[k].sample(n_k)
        
        return samples, components
    
    def bic(self, X: np.ndarray) -> float:
        """Compute Bayesian Information Criterion."""
        n_samples, n_features = X.shape
        _, log_likelihood = self._e_step(X)
        
        # Number of parameters
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        else:
            cov_params = self.n_components
        
        n_params = (self.n_components - 1) + self.n_components * n_features + cov_params
        
        return -2 * log_likelihood + n_params * np.log(n_samples)
    
    def aic(self, X: np.ndarray) -> float:
        """Compute Akaike Information Criterion."""
        n_samples, n_features = X.shape
        _, log_likelihood = self._e_step(X)
        
        # Number of parameters
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        else:
            cov_params = self.n_components
        
        n_params = (self.n_components - 1) + self.n_components * n_features + cov_params
        
        return -2 * log_likelihood + 2 * n_params


class MixtureOfExperts:
    """
    Mixture of Experts model.
    
    Each expert handles a different region of input space,
    with gating network determining which expert to use.
    """
    
    def __init__(self, n_experts: int = 3,
                 expert_type: str = 'linear'):
        """
        Initialize MoE.
        
        Args:
            n_experts: Number of expert models
            expert_type: 'linear' or 'gaussian'
        """
        self.n_experts = n_experts
        self.expert_type = expert_type
        
        # Gating network parameters
        self.gate_weights: Optional[np.ndarray] = None
        self.gate_bias: Optional[np.ndarray] = None
        
        # Expert parameters
        self.expert_weights: List[np.ndarray] = []
        self.expert_bias: List[np.ndarray] = []
        self.expert_variance: List[float] = []
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _gate(self, X: np.ndarray) -> np.ndarray:
        """Compute gating probabilities."""
        logits = X @ self.gate_weights + self.gate_bias
        return self._softmax(logits)
    
    def _expert_output(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute expert k's output (mean and variance)."""
        mean = X @ self.expert_weights[k] + self.expert_bias[k]
        var = np.full(len(X), self.expert_variance[k])
        return mean, var
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            max_iter: int = 100) -> 'MixtureOfExperts':
        """Fit MoE using EM algorithm."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.gate_weights = np.random.randn(n_features, self.n_experts) * 0.1
        self.gate_bias = np.zeros(self.n_experts)
        
        self.expert_weights = [np.random.randn(n_features) * 0.1 for _ in range(self.n_experts)]
        self.expert_bias = [0.0 for _ in range(self.n_experts)]
        self.expert_variance = [np.var(y) for _ in range(self.n_experts)]
        
        for iteration in range(max_iter):
            # E-step: compute responsibilities
            gate_probs = self._gate(X)  # (n_samples, n_experts)
            
            expert_log_probs = np.zeros((n_samples, self.n_experts))
            for k in range(self.n_experts):
                mean, var = self._expert_output(X, k)
                expert_log_probs[:, k] = -0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mean) ** 2 / var
            
            # Posterior responsibility
            log_resp = np.log(gate_probs + EPSILON) + expert_log_probs
            log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp)
            
            # M-step: update experts
            for k in range(self.n_experts):
                # Weighted least squares for expert k
                W = np.diag(resp[:, k])
                XtWX = X.T @ W @ X + EPSILON * np.eye(n_features)
                XtWy = X.T @ W @ y
                self.expert_weights[k] = np.linalg.solve(XtWX, XtWy)
                
                mean, _ = self._expert_output(X, k)
                self.expert_bias[k] = np.sum(resp[:, k] * (y - X @ self.expert_weights[k])) / (resp[:, k].sum() + EPSILON)
                
                mean, _ = self._expert_output(X, k)
                self.expert_variance[k] = np.sum(resp[:, k] * (y - mean) ** 2) / (resp[:, k].sum() + EPSILON) + EPSILON
            
            # M-step: update gating network (gradient descent)
            for _ in range(10):
                gate_probs = self._gate(X)
                grad_w = X.T @ (resp - gate_probs)
                grad_b = np.sum(resp - gate_probs, axis=0)
                
                self.gate_weights += 0.01 * grad_w
                self.gate_bias += 0.01 * grad_b
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance.
        
        Returns:
            (mean, variance) - mixture predictions
        """
        gate_probs = self._gate(X)
        
        means = np.zeros((len(X), self.n_experts))
        variances = np.zeros((len(X), self.n_experts))
        
        for k in range(self.n_experts):
            means[:, k], variances[:, k] = self._expert_output(X, k)
        
        # Mixture mean: E[y] = Σ_k p_k μ_k
        mixture_mean = np.sum(gate_probs * means, axis=1)
        
        # Mixture variance: Var[y] = E[Var[y|k]] + Var[E[y|k]]
        within_var = np.sum(gate_probs * variances, axis=1)
        between_var = np.sum(gate_probs * (means - mixture_mean[:, np.newaxis]) ** 2, axis=1)
        mixture_var = within_var + between_var
        
        return mixture_mean, mixture_var


class FiniteMixture:
    """
    General finite mixture model for uncertainty quantification.
    
    Can represent multi-modal uncertainty by fitting
    mixture to posterior samples or prediction residuals.
    """
    
    def __init__(self, max_components: int = 10):
        self.max_components = max_components
        self.gmm: Optional[GaussianMixtureModel] = None
        self.n_components: int = 0
    
    def fit(self, samples: np.ndarray, 
            selection_criterion: str = 'bic') -> 'FiniteMixture':
        """
        Fit mixture model with automatic component selection.
        
        Args:
            samples: 1D or 2D array of samples
            selection_criterion: 'bic' or 'aic'
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        
        best_score = np.inf
        best_gmm = None
        best_k = 1
        
        for k in range(1, self.max_components + 1):
            try:
                gmm = GaussianMixtureModel(n_components=k)
                gmm.fit(samples)
                
                if selection_criterion == 'bic':
                    score = gmm.bic(samples)
                else:
                    score = gmm.aic(samples)
                
                if score < best_score:
                    best_score = score
                    best_gmm = gmm
                    best_k = k
            except:
                continue
        
        self.gmm = best_gmm
        self.n_components = best_k
        return self
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate probability density function."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        log_probs = []
        for comp in self.gmm.components:
            log_probs.append(np.log(comp.weight + EPSILON) + comp.log_prob(x))
        
        return np.exp(logsumexp(log_probs, axis=0))
    
    def cdf(self, x: np.ndarray, n_samples: int = 10000) -> np.ndarray:
        """Evaluate cumulative distribution function (Monte Carlo)."""
        samples, _ = self.gmm.sample(n_samples)
        samples = samples.flatten()
        
        return np.array([np.mean(samples <= xi) for xi in x.flatten()])
    
    def quantile(self, p: np.ndarray, n_samples: int = 10000) -> np.ndarray:
        """Compute quantiles (Monte Carlo)."""
        samples, _ = self.gmm.sample(n_samples)
        return np.percentile(samples.flatten(), p * 100)
    
    def mean(self) -> float:
        """Compute mixture mean."""
        return sum(c.weight * c.mean[0] for c in self.gmm.components)
    
    def variance(self) -> float:
        """Compute mixture variance."""
        mu = self.mean()
        
        # E[X^2] = Σ w_k (μ_k^2 + σ_k^2)
        E_X2 = sum(c.weight * (c.mean[0]**2 + c.covariance[0, 0]) for c in self.gmm.components)
        
        return E_X2 - mu**2
    
    def std(self) -> float:
        """Compute mixture standard deviation."""
        return np.sqrt(self.variance())
    
    def modes(self) -> List[float]:
        """Return modes (component means)."""
        return [c.mean[0] for c in self.gmm.components]
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute credible interval."""
        alpha = (1 - level) / 2
        return self.quantile(np.array([alpha, 1 - alpha]))


def fit_mixture_to_uncertainty(predictions: np.ndarray,
                                uncertainties: np.ndarray,
                                n_components: int = 3) -> FiniteMixture:
    """
    Fit mixture model to capture prediction uncertainty distribution.
    
    Useful when uncertainty is multi-modal or non-Gaussian.
    """
    # Generate samples representing uncertainty
    n_samples = 1000
    samples = []
    
    for pred, unc in zip(predictions, uncertainties):
        # Sample from local uncertainty
        local_samples = np.random.normal(pred, unc, n_samples // len(predictions))
        samples.extend(local_samples)
    
    samples = np.array(samples)
    
    mixture = FiniteMixture(max_components=n_components)
    mixture.fit(samples)
    
    return mixture


def mixture_kalman_update(prior_mixture: FiniteMixture,
                          observation: float,
                          observation_variance: float) -> FiniteMixture:
    """
    Kalman-style update for mixture distributions.
    
    Updates each component independently, then renormalizes weights.
    """
    updated_components = []
    
    for comp in prior_mixture.gmm.components:
        # Kalman update for this component
        prior_mean = comp.mean[0]
        prior_var = comp.covariance[0, 0]
        
        # Kalman gain
        K = prior_var / (prior_var + observation_variance)
        
        # Updated mean and variance
        post_mean = prior_mean + K * (observation - prior_mean)
        post_var = (1 - K) * prior_var
        
        # Update weight based on likelihood
        likelihood = np.exp(-0.5 * (observation - prior_mean)**2 / (prior_var + observation_variance))
        new_weight = comp.weight * likelihood
        
        updated_components.append(MixtureComponent(
            weight=new_weight,
            mean=np.array([post_mean]),
            covariance=np.array([[post_var]])
        ))
    
    # Renormalize weights
    total_weight = sum(c.weight for c in updated_components)
    for c in updated_components:
        c.weight /= total_weight
    
    # Create new mixture
    result = FiniteMixture()
    result.gmm = GaussianMixtureModel(n_components=len(updated_components))
    result.gmm.components = updated_components
    result.n_components = len(updated_components)
    
    return result
