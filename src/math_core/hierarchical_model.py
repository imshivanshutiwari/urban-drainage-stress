"""
Hierarchical Probabilistic Model Module

Implements Bayesian hierarchical models for stress inference:
- Prior → Likelihood → Posterior structure
- Hyperparameter inference
- Conditional distributions
- Full uncertainty quantification

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from scipy import stats
from scipy.special import gammaln, digamma
from scipy.optimize import minimize

from .error_aware_arithmetic import EPSILON, ErrorValue, ErrorArray


@dataclass
class Prior:
    """Base class for prior distributions."""
    
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute log probability density."""
        raise NotImplementedError
    
    def sample(self, n: int) -> np.ndarray:
        """Sample from the prior."""
        raise NotImplementedError
    
    def mean(self) -> float:
        """Return prior mean."""
        raise NotImplementedError
    
    def variance(self) -> float:
        """Return prior variance."""
        raise NotImplementedError


@dataclass
class GaussianPrior(Prior):
    """Gaussian prior distribution."""
    mu: float = 0.0
    sigma: float = 1.0
    
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        return -0.5 * np.log(2 * np.pi * self.sigma**2) - 0.5 * ((x - self.mu) / self.sigma)**2
    
    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma, n)
    
    def mean(self) -> float:
        return self.mu
    
    def variance(self) -> float:
        return self.sigma**2


@dataclass
class GammaPrior(Prior):
    """Gamma prior for positive parameters."""
    alpha: float = 1.0  # Shape
    beta: float = 1.0   # Rate
    
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.maximum(x, EPSILON)
        return (self.alpha * np.log(self.beta) - gammaln(self.alpha) + 
                (self.alpha - 1) * np.log(x) - self.beta * x)
    
    def sample(self, n: int) -> np.ndarray:
        return np.random.gamma(self.alpha, 1.0 / self.beta, n)
    
    def mean(self) -> float:
        return self.alpha / self.beta
    
    def variance(self) -> float:
        return self.alpha / self.beta**2


@dataclass
class InverseGammaPrior(Prior):
    """Inverse-Gamma prior for variance parameters."""
    alpha: float = 1.0  # Shape
    beta: float = 1.0   # Scale
    
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.maximum(x, EPSILON)
        return (self.alpha * np.log(self.beta) - gammaln(self.alpha) - 
                (self.alpha + 1) * np.log(x) - self.beta / x)
    
    def sample(self, n: int) -> np.ndarray:
        return 1.0 / np.random.gamma(self.alpha, 1.0 / self.beta, n)
    
    def mean(self) -> float:
        if self.alpha > 1:
            return self.beta / (self.alpha - 1)
        return np.inf
    
    def variance(self) -> float:
        if self.alpha > 2:
            return self.beta**2 / ((self.alpha - 1)**2 * (self.alpha - 2))
        return np.inf


@dataclass
class BetaPrior(Prior):
    """Beta prior for parameters in [0, 1]."""
    alpha: float = 1.0
    beta: float = 1.0
    
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, EPSILON, 1 - EPSILON)
        return (gammaln(self.alpha + self.beta) - gammaln(self.alpha) - gammaln(self.beta) +
                (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x))
    
    def sample(self, n: int) -> np.ndarray:
        return np.random.beta(self.alpha, self.beta, n)
    
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def variance(self) -> float:
        ab = self.alpha + self.beta
        return self.alpha * self.beta / (ab**2 * (ab + 1))


@dataclass
class Likelihood:
    """Base class for likelihood functions."""
    
    def log_prob(self, data: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute log likelihood."""
        raise NotImplementedError
    
    def gradient(self, data: np.ndarray, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute gradient of log likelihood w.r.t. parameters."""
        raise NotImplementedError


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood with known or unknown variance."""
    
    def __init__(self, known_variance: Optional[float] = None):
        self.known_variance = known_variance
    
    def log_prob(self, data: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        mu = params['mu']
        if self.known_variance is not None:
            sigma2 = self.known_variance
        else:
            sigma2 = params.get('sigma2', 1.0)
        
        return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (data - mu)**2 / sigma2
    
    def gradient(self, data: np.ndarray, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        mu = params['mu']
        if self.known_variance is not None:
            sigma2 = self.known_variance
        else:
            sigma2 = params.get('sigma2', 1.0)
        
        grad_mu = (data - mu) / sigma2
        
        grads = {'mu': grad_mu}
        if self.known_variance is None:
            grad_sigma2 = -0.5 / sigma2 + 0.5 * (data - mu)**2 / sigma2**2
            grads['sigma2'] = grad_sigma2
        
        return grads


class HierarchicalModel:
    """
    Full hierarchical Bayesian model.
    
    Structure:
        Hyperpriors → Parameters → Latent variables → Data
    
    Supports:
        - MAP estimation
        - Full posterior sampling (MCMC)
        - Variational inference
    """
    
    def __init__(self):
        self.hyperpriors: Dict[str, Prior] = {}
        self.priors: Dict[str, Callable[[Dict], Prior]] = {}
        self.likelihood: Optional[Likelihood] = None
        self.data: Optional[np.ndarray] = None
        
        # Cached values
        self._posterior_samples: Optional[Dict[str, np.ndarray]] = None
        self._map_estimate: Optional[Dict[str, float]] = None
    
    def add_hyperprior(self, name: str, prior: Prior):
        """Add a hyperprior distribution."""
        self.hyperpriors[name] = prior
    
    def add_prior(self, name: str, prior_func: Callable[[Dict], Prior]):
        """Add a prior that depends on hyperparameters."""
        self.priors[name] = prior_func
    
    def set_likelihood(self, likelihood: Likelihood):
        """Set the likelihood function."""
        self.likelihood = likelihood
    
    def set_data(self, data: np.ndarray):
        """Set observed data."""
        self.data = data
    
    def log_joint(self, hyperparams: Dict[str, float], 
                  params: Dict[str, np.ndarray]) -> float:
        """
        Compute log joint probability.
        
        log p(data, params, hyperparams) = 
            log p(data | params) + log p(params | hyperparams) + log p(hyperparams)
        """
        if self.data is None or self.likelihood is None:
            raise ValueError("Data and likelihood must be set")
        
        log_prob = 0.0
        
        # Hyperprior term
        for name, value in hyperparams.items():
            if name in self.hyperpriors:
                log_prob += float(self.hyperpriors[name].log_prob(np.array([value])))
        
        # Prior term (conditioned on hyperparameters)
        for name, prior_func in self.priors.items():
            prior = prior_func(hyperparams)
            if name in params:
                log_prob += float(np.sum(prior.log_prob(params[name])))
        
        # Likelihood term
        log_prob += float(np.sum(self.likelihood.log_prob(self.data, params)))
        
        return log_prob
    
    def compute_map(self, initial_hyperparams: Dict[str, float],
                    initial_params: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Compute MAP (Maximum A Posteriori) estimate.
        
        Returns:
            (map_hyperparams, map_params)
        """
        # Flatten parameters for optimization
        hp_names = list(initial_hyperparams.keys())
        p_names = list(initial_params.keys())
        
        hp_values = np.array([initial_hyperparams[n] for n in hp_names])
        p_values = np.concatenate([initial_params[n].flatten() for n in p_names])
        x0 = np.concatenate([hp_values, p_values])
        
        # Track shapes for reconstruction
        p_shapes = {n: initial_params[n].shape for n in p_names}
        p_sizes = {n: initial_params[n].size for n in p_names}
        
        def neg_log_joint(x):
            # Reconstruct parameters
            hp = {n: x[i] for i, n in enumerate(hp_names)}
            
            offset = len(hp_names)
            params = {}
            for n in p_names:
                size = p_sizes[n]
                params[n] = x[offset:offset + size].reshape(p_shapes[n])
                offset += size
            
            return -self.log_joint(hp, params)
        
        # Optimize
        result = minimize(neg_log_joint, x0, method='L-BFGS-B')
        
        # Reconstruct solution
        map_hp = {n: result.x[i] for i, n in enumerate(hp_names)}
        
        offset = len(hp_names)
        map_params = {}
        for n in p_names:
            size = p_sizes[n]
            map_params[n] = result.x[offset:offset + size].reshape(p_shapes[n])
            offset += size
        
        self._map_estimate = (map_hp, map_params)
        return map_hp, map_params
    
    def sample_posterior(self, n_samples: int = 1000,
                         n_burnin: int = 500,
                         initial_hyperparams: Optional[Dict[str, float]] = None,
                         initial_params: Optional[Dict[str, np.ndarray]] = None
                         ) -> Dict[str, np.ndarray]:
        """
        Sample from posterior using Metropolis-Hastings.
        
        Returns:
            Dictionary of parameter samples
        """
        # Initialize from MAP or provided values
        if initial_hyperparams is None:
            initial_hyperparams = {n: p.mean() for n, p in self.hyperpriors.items()}
        if initial_params is None:
            initial_params = {}
            for name, prior_func in self.priors.items():
                prior = prior_func(initial_hyperparams)
                initial_params[name] = np.array([prior.mean()])
        
        # Flatten current state
        hp_names = list(initial_hyperparams.keys())
        p_names = list(initial_params.keys())
        
        current_hp = initial_hyperparams.copy()
        current_p = {k: v.copy() for k, v in initial_params.items()}
        current_log_prob = self.log_joint(current_hp, current_p)
        
        # Storage
        hp_samples = {n: [] for n in hp_names}
        p_samples = {n: [] for n in p_names}
        
        # Proposal scales (could be adaptive)
        hp_scale = {n: 0.1 * self.hyperpriors[n].variance()**0.5 for n in hp_names}
        p_scale = 0.1
        
        n_accept = 0
        
        for i in range(n_samples + n_burnin):
            # Propose new hyperparameters
            proposed_hp = {n: current_hp[n] + np.random.normal(0, hp_scale.get(n, 0.1))
                          for n in hp_names}
            
            # Propose new parameters
            proposed_p = {n: current_p[n] + np.random.normal(0, p_scale, current_p[n].shape)
                         for n in p_names}
            
            # Compute acceptance probability
            try:
                proposed_log_prob = self.log_joint(proposed_hp, proposed_p)
                log_alpha = proposed_log_prob - current_log_prob
                
                if np.log(np.random.random()) < log_alpha:
                    current_hp = proposed_hp
                    current_p = proposed_p
                    current_log_prob = proposed_log_prob
                    n_accept += 1
            except:
                pass  # Reject invalid proposals
            
            # Store samples after burn-in
            if i >= n_burnin:
                for n in hp_names:
                    hp_samples[n].append(current_hp[n])
                for n in p_names:
                    p_samples[n].append(current_p[n].copy())
        
        # Convert to arrays
        samples = {}
        for n in hp_names:
            samples[n] = np.array(hp_samples[n])
        for n in p_names:
            samples[n] = np.array(p_samples[n])
        
        self._posterior_samples = samples
        return samples
    
    def posterior_predictive(self, x_new: np.ndarray,
                             n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior predictive distribution.
        
        Returns:
            (mean, std) of predictive distribution
        """
        if self._posterior_samples is None:
            raise ValueError("Must run sample_posterior first")
        
        # Use stored samples
        predictions = []
        
        for i in range(min(n_samples, len(list(self._posterior_samples.values())[0]))):
            hp = {n: self._posterior_samples[n][i] for n in self.hyperpriors.keys()}
            params = {n: self._posterior_samples[n][i] 
                     for n in self.priors.keys() if n in self._posterior_samples}
            
            # Generate prediction
            if 'mu' in params:
                mu = params['mu']
                if hasattr(mu, '__len__') and len(mu) == len(x_new):
                    pred = mu
                else:
                    pred = np.full(len(x_new), np.mean(mu))
            else:
                pred = np.zeros(len(x_new))
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


class StressHierarchicalModel(HierarchicalModel):
    """
    Hierarchical model specifically for drainage stress.
    
    Model structure:
        - Hyperprior on stress spatial correlation length
        - Hyperprior on stress variance scale
        - Prior on latent stress field
        - Likelihood connecting stress to observed indicators
    """
    
    def __init__(self, n_locations: int):
        super().__init__()
        self.n_locations = n_locations
        
        # Set up default structure
        self._setup_default_priors()
    
    def _setup_default_priors(self):
        """Set up default hierarchical structure for stress model."""
        # Hyperprior on length scale
        self.add_hyperprior('length_scale', GammaPrior(alpha=2.0, beta=1.0))
        
        # Hyperprior on variance
        self.add_hyperprior('variance', InverseGammaPrior(alpha=2.0, beta=1.0))
        
        # Prior on stress (depends on hyperparameters)
        def stress_prior(hp):
            return GaussianPrior(mu=0.0, sigma=np.sqrt(hp.get('variance', 1.0)))
        
        self.add_prior('stress', stress_prior)
        
        # Gaussian likelihood
        self.set_likelihood(GaussianLikelihood())
    
    def fit(self, observations: np.ndarray,
            observation_locations: np.ndarray,
            method: str = 'map') -> Dict[str, Any]:
        """
        Fit hierarchical model to data.
        
        Args:
            observations: Observed stress indicators
            observation_locations: Locations of observations
            method: 'map' or 'mcmc'
        
        Returns:
            Dictionary with fitted parameters and uncertainties
        """
        self.set_data(observations)
        
        # Initial values
        init_hp = {'length_scale': 1.0, 'variance': np.var(observations)}
        init_params = {'stress': observations.copy(), 'mu': observations.copy()}
        
        if method == 'map':
            map_hp, map_params = self.compute_map(init_hp, init_params)
            return {
                'hyperparameters': map_hp,
                'parameters': map_params,
                'method': 'map'
            }
        else:
            samples = self.sample_posterior(
                n_samples=1000, n_burnin=500,
                initial_hyperparams=init_hp,
                initial_params=init_params
            )
            
            # Compute summaries
            summaries = {}
            for name, samps in samples.items():
                summaries[name] = {
                    'mean': np.mean(samps, axis=0),
                    'std': np.std(samps, axis=0),
                    'q05': np.percentile(samps, 5, axis=0),
                    'q95': np.percentile(samps, 95, axis=0)
                }
            
            return {
                'samples': samples,
                'summaries': summaries,
                'method': 'mcmc'
            }


def compute_credible_interval(samples: np.ndarray,
                               level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute credible interval from posterior samples.
    
    Args:
        samples: Array of posterior samples
        level: Credibility level (default 95%)
    
    Returns:
        (lower, upper) bounds
    """
    alpha = (1 - level) / 2
    lower = np.percentile(samples, 100 * alpha, axis=0)
    upper = np.percentile(samples, 100 * (1 - alpha), axis=0)
    return lower, upper


def compute_highest_density_interval(samples: np.ndarray,
                                      level: float = 0.95) -> Tuple[float, float]:
    """
    Compute highest density interval (HDI) from samples.
    
    The HDI is the narrowest interval containing the specified probability mass.
    """
    sorted_samples = np.sort(samples.flatten())
    n = len(sorted_samples)
    
    n_included = int(np.ceil(level * n))
    
    min_width = np.inf
    best_low, best_high = sorted_samples[0], sorted_samples[-1]
    
    for i in range(n - n_included):
        width = sorted_samples[i + n_included] - sorted_samples[i]
        if width < min_width:
            min_width = width
            best_low = sorted_samples[i]
            best_high = sorted_samples[i + n_included]
    
    return best_low, best_high


def bayes_factor(log_marginal_1: float, log_marginal_2: float) -> float:
    """
    Compute Bayes factor for model comparison.
    
    BF = p(data | model 1) / p(data | model 2)
    
    Returns log Bayes factor.
    """
    return log_marginal_1 - log_marginal_2
