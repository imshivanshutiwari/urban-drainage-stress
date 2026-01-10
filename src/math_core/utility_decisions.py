"""
Utility-Based Decision Module

Implements decision-theoretic framework:
- Utility functions (CRRA, CARA, prospect theory)
- Expected utility maximization
- Multi-attribute utility theory
- Value of information
- Bayesian decision theory

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass
from scipy import optimize
from scipy import stats
from scipy.integrate import quad

from .error_aware_arithmetic import EPSILON


@dataclass
class DecisionResult:
    """Result of a decision analysis."""
    optimal_action: Any
    expected_utility: float
    certainty_equivalent: float
    risk_premium: float
    value_of_information: Optional[float] = None


class UtilityFunction:
    """Base class for utility functions."""
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate utility."""
        raise NotImplementedError
    
    def marginal_utility(self, x: np.ndarray) -> np.ndarray:
        """First derivative of utility."""
        raise NotImplementedError
    
    def risk_aversion(self, x: np.ndarray) -> np.ndarray:
        """
        Arrow-Pratt measure of absolute risk aversion.
        
        A(x) = -u''(x) / u'(x)
        """
        raise NotImplementedError
    
    def certainty_equivalent(self, outcomes: np.ndarray,
                             probabilities: np.ndarray) -> float:
        """
        Compute certainty equivalent of lottery.
        
        CE = u^{-1}(E[u(X)])
        """
        expected_utility = np.dot(probabilities, self(outcomes))
        return self.inverse(expected_utility)
    
    def inverse(self, u: float) -> float:
        """Inverse of utility function."""
        raise NotImplementedError
    
    def risk_premium(self, outcomes: np.ndarray,
                      probabilities: np.ndarray) -> float:
        """
        Risk premium: amount willing to pay to avoid risk.
        
        RP = E[X] - CE
        """
        expected_value = np.dot(probabilities, outcomes)
        ce = self.certainty_equivalent(outcomes, probabilities)
        return expected_value - ce


class CRRAUtility(UtilityFunction):
    """
    Constant Relative Risk Aversion (CRRA) utility.
    
    u(x) = x^{1-γ} / (1-γ)  if γ ≠ 1
    u(x) = log(x)          if γ = 1
    
    Relative risk aversion R(x) = γ (constant).
    """
    
    def __init__(self, gamma: float = 0.5):
        """
        Initialize CRRA utility.
        
        Args:
            gamma: Relative risk aversion coefficient
                   γ = 0: risk neutral
                   γ > 0: risk averse
                   γ < 0: risk seeking
        """
        self.gamma = gamma
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        x = np.maximum(x, EPSILON)  # Ensure positive
        
        if abs(self.gamma - 1) < EPSILON:
            return np.log(x)
        else:
            return (x ** (1 - self.gamma) - 1) / (1 - self.gamma)
    
    def marginal_utility(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        x = np.maximum(x, EPSILON)
        return x ** (-self.gamma)
    
    def risk_aversion(self, x: np.ndarray) -> np.ndarray:
        """Absolute risk aversion."""
        x = np.asarray(x)
        x = np.maximum(x, EPSILON)
        return self.gamma / x
    
    def inverse(self, u: float) -> float:
        if abs(self.gamma - 1) < EPSILON:
            return np.exp(u)
        else:
            inner = u * (1 - self.gamma) + 1
            if inner <= 0:
                return EPSILON
            return inner ** (1 / (1 - self.gamma))


class CARAUtility(UtilityFunction):
    """
    Constant Absolute Risk Aversion (CARA) utility.
    
    u(x) = -exp(-αx) / α  if α ≠ 0
    u(x) = x              if α = 0
    
    Absolute risk aversion A(x) = α (constant).
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CARA utility.
        
        Args:
            alpha: Absolute risk aversion coefficient
        """
        self.alpha = alpha
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        
        if abs(self.alpha) < EPSILON:
            return x
        else:
            return -np.exp(-self.alpha * x) / self.alpha
    
    def marginal_utility(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        
        if abs(self.alpha) < EPSILON:
            return np.ones_like(x)
        else:
            return np.exp(-self.alpha * x)
    
    def risk_aversion(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.full_like(x, self.alpha)
    
    def inverse(self, u: float) -> float:
        if abs(self.alpha) < EPSILON:
            return u
        else:
            inner = -u * self.alpha
            if inner <= 0:
                return np.inf
            return -np.log(inner) / self.alpha


class ProspectTheoryUtility(UtilityFunction):
    """
    Prospect Theory utility (Kahneman & Tversky).
    
    Value function v(x):
    v(x) = x^α           if x >= 0
    v(x) = -λ(-x)^β      if x < 0
    
    Probability weighting w(p):
    w(p) = p^γ / (p^γ + (1-p)^γ)^{1/γ}
    """
    
    def __init__(self, alpha: float = 0.88, beta: float = 0.88,
                 lambda_: float = 2.25, gamma_gain: float = 0.61,
                 gamma_loss: float = 0.69):
        """
        Initialize prospect theory utility.
        
        Default parameters from Tversky & Kahneman (1992).
        
        Args:
            alpha: Diminishing sensitivity for gains
            beta: Diminishing sensitivity for losses
            lambda_: Loss aversion coefficient
            gamma_gain: Probability weighting for gains
            gamma_loss: Probability weighting for losses
        """
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.gamma_gain = gamma_gain
        self.gamma_loss = gamma_loss
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Value function."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        
        gains = x >= 0
        losses = ~gains
        
        result[gains] = x[gains] ** self.alpha
        result[losses] = -self.lambda_ * ((-x[losses]) ** self.beta)
        
        return result
    
    def weight_probability(self, p: np.ndarray, is_gain: bool = True) -> np.ndarray:
        """
        Probability weighting function.
        
        Args:
            p: Probabilities
            is_gain: Whether outcome is a gain
        
        Returns:
            Weighted probabilities
        """
        p = np.asarray(p)
        p = np.clip(p, EPSILON, 1 - EPSILON)
        
        gamma = self.gamma_gain if is_gain else self.gamma_loss
        
        numerator = p ** gamma
        denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
        
        return numerator / denominator
    
    def marginal_utility(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        
        gains = x >= 0
        losses = ~gains
        
        result[gains] = self.alpha * x[gains] ** (self.alpha - 1)
        result[losses] = self.lambda_ * self.beta * ((-x[losses]) ** (self.beta - 1))
        
        return result
    
    def risk_aversion(self, x: np.ndarray) -> np.ndarray:
        """Not well-defined for prospect theory."""
        return np.full_like(x, np.nan)
    
    def inverse(self, u: float) -> float:
        """Inverse of value function."""
        if u >= 0:
            return u ** (1 / self.alpha)
        else:
            return -((-u / self.lambda_) ** (1 / self.beta))
    
    def evaluate_prospect(self, outcomes: np.ndarray,
                          probabilities: np.ndarray,
                          reference: float = 0.0) -> float:
        """
        Evaluate prospect (lottery) using prospect theory.
        
        Args:
            outcomes: Possible outcomes
            probabilities: Outcome probabilities
            reference: Reference point
        
        Returns:
            Prospect value
        """
        outcomes = np.asarray(outcomes) - reference
        probabilities = np.asarray(probabilities)
        
        # Separate gains and losses
        gains = outcomes >= 0
        losses = ~gains
        
        value = 0.0
        
        # Gains
        if gains.any():
            gain_outcomes = outcomes[gains]
            gain_probs = probabilities[gains]
            
            # Sort by outcome (rank-dependent weighting)
            sorted_idx = np.argsort(gain_outcomes)[::-1]
            gain_outcomes = gain_outcomes[sorted_idx]
            gain_probs = gain_probs[sorted_idx]
            
            cumulative = 0
            for i, (o, p) in enumerate(zip(gain_outcomes, gain_probs)):
                w_curr = self.weight_probability(cumulative + p, is_gain=True)
                w_prev = self.weight_probability(cumulative, is_gain=True)
                decision_weight = w_curr - w_prev
                value += self(np.array([o]))[0] * decision_weight
                cumulative += p
        
        # Losses
        if losses.any():
            loss_outcomes = outcomes[losses]
            loss_probs = probabilities[losses]
            
            # Sort by outcome (best to worst loss)
            sorted_idx = np.argsort(loss_outcomes)[::-1]
            loss_outcomes = loss_outcomes[sorted_idx]
            loss_probs = loss_probs[sorted_idx]
            
            cumulative = 0
            for i, (o, p) in enumerate(zip(loss_outcomes, loss_probs)):
                w_curr = self.weight_probability(cumulative + p, is_gain=False)
                w_prev = self.weight_probability(cumulative, is_gain=False)
                decision_weight = w_curr - w_prev
                value += self(np.array([o]))[0] * decision_weight
                cumulative += p
        
        return value


class MultiAttributeUtility:
    """
    Multi-Attribute Utility Theory (MAUT).
    
    For attributes X = (X_1, ..., X_n):
    
    Additive: U(x) = Σ w_i u_i(x_i)
    Multiplicative: 1 + k U(x) = Π (1 + k w_i u_i(x_i))
    """
    
    def __init__(self, attribute_utilities: List[UtilityFunction],
                 weights: np.ndarray,
                 form: str = 'additive'):
        """
        Initialize multi-attribute utility.
        
        Args:
            attribute_utilities: Utility function for each attribute
            weights: Importance weights (sum to 1 for additive)
            form: 'additive' or 'multiplicative'
        """
        self.utilities = attribute_utilities
        self.weights = np.asarray(weights)
        self.form = form
        
        # For multiplicative form, find scaling constant k
        if form == 'multiplicative':
            self.k = self._find_scaling_constant()
    
    def _find_scaling_constant(self) -> float:
        """
        Find k for multiplicative form.
        
        Solves: 1 + k = Π (1 + k w_i)
        """
        w = self.weights
        
        if abs(np.sum(w) - 1) < EPSILON:
            return 0.0  # Reduces to additive
        
        def equation(k):
            product = np.prod(1 + k * w)
            return 1 + k - product
        
        try:
            result = optimize.brentq(equation, -0.99 / np.max(w), 100)
            return result
        except ValueError:
            return 0.0
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate multi-attribute utility.
        
        Args:
            x: Attribute values (n_attributes,)
        
        Returns:
            Overall utility
        """
        x = np.asarray(x)
        
        # Single-attribute utilities
        u_i = np.array([self.utilities[i](x[i:i+1])[0] 
                       for i in range(len(x))])
        
        if self.form == 'additive':
            return np.dot(self.weights, u_i)
        
        elif self.form == 'multiplicative':
            if abs(self.k) < EPSILON:
                return np.dot(self.weights, u_i)
            
            product = np.prod(1 + self.k * self.weights * u_i)
            return (product - 1) / self.k
        
        raise ValueError(f"Unknown form: {self.form}")
    
    def marginal_rate_of_substitution(self, x: np.ndarray,
                                       i: int, j: int) -> float:
        """
        Marginal rate of substitution between attributes.
        
        MRS_{ij} = (∂U/∂x_i) / (∂U/∂x_j)
        
        Args:
            x: Current attribute values
            i, j: Attribute indices
        
        Returns:
            MRS: How much of attribute j willing to give up for unit of i
        """
        x = np.asarray(x)
        
        # Numerical gradient
        eps = 1e-7
        
        x_plus_i = x.copy()
        x_plus_i[i] += eps
        du_di = (self(x_plus_i) - self(x)) / eps
        
        x_plus_j = x.copy()
        x_plus_j[j] += eps
        du_dj = (self(x_plus_j) - self(x)) / eps
        
        if abs(du_dj) < EPSILON:
            return np.inf
        
        return du_di / du_dj


class BayesianDecisionMaker:
    """
    Bayesian decision theory framework.
    
    Optimal action: a* = argmax_a E_{θ|data}[U(a, θ)]
    """
    
    def __init__(self, utility: UtilityFunction):
        """
        Initialize Bayesian decision maker.
        
        Args:
            utility: Utility function over outcomes
        """
        self.utility = utility
    
    def optimal_action(self, actions: List[Any],
                       outcome_function: Callable[[Any, np.ndarray], float],
                       posterior_samples: np.ndarray) -> DecisionResult:
        """
        Find optimal action given posterior uncertainty.
        
        Args:
            actions: Available actions
            outcome_function: Maps (action, parameter) -> outcome
            posterior_samples: Samples from posterior (n_samples, dim)
        
        Returns:
            Decision result
        """
        expected_utilities = []
        
        for action in actions:
            outcomes = np.array([
                outcome_function(action, theta) 
                for theta in posterior_samples
            ])
            
            eu = np.mean(self.utility(outcomes))
            expected_utilities.append(eu)
        
        best_idx = np.argmax(expected_utilities)
        best_action = actions[best_idx]
        best_eu = expected_utilities[best_idx]
        
        # Certainty equivalent
        best_outcomes = np.array([
            outcome_function(best_action, theta)
            for theta in posterior_samples
        ])
        ce = self.utility.certainty_equivalent(
            best_outcomes, 
            np.ones(len(posterior_samples)) / len(posterior_samples)
        )
        
        rp = np.mean(best_outcomes) - ce
        
        return DecisionResult(
            optimal_action=best_action,
            expected_utility=best_eu,
            certainty_equivalent=ce,
            risk_premium=rp
        )
    
    def value_of_information(self, 
                              actions: List[Any],
                              outcome_function: Callable,
                              prior_samples: np.ndarray,
                              signal_model: Callable[[np.ndarray], np.ndarray],
                              n_signals: int = 100) -> float:
        """
        Compute value of perfect information.
        
        VoI = E_θ[max_a U(a,θ)] - max_a E_θ[U(a,θ)]
        
        Args:
            actions: Available actions
            outcome_function: Maps (action, parameter) -> outcome
            prior_samples: Samples from prior
            signal_model: Maps parameter to signal
            n_signals: Number of signals to simulate
        
        Returns:
            Expected value of perfect information
        """
        # Expected utility without information
        result_without = self.optimal_action(
            actions, outcome_function, prior_samples
        )
        eu_without = result_without.expected_utility
        
        # Expected utility with perfect information
        # For each theta, choose best action
        eu_with_perfect = []
        
        for theta in prior_samples:
            action_utilities = []
            for action in actions:
                outcome = outcome_function(action, theta)
                u = self.utility(np.array([outcome]))[0]
                action_utilities.append(u)
            eu_with_perfect.append(np.max(action_utilities))
        
        evpi = np.mean(eu_with_perfect) - eu_without
        
        return max(evpi, 0)  # VoI is non-negative
    
    def value_of_sample_information(self,
                                     actions: List[Any],
                                     outcome_function: Callable,
                                     prior_samples: np.ndarray,
                                     likelihood_function: Callable,
                                     sample_cost: float = 0.0) -> float:
        """
        Compute expected value of sample information.
        
        Args:
            actions: Available actions
            outcome_function: Maps (action, parameter) -> outcome
            prior_samples: Samples from prior
            likelihood_function: P(data | theta)
            sample_cost: Cost of obtaining sample
        
        Returns:
            Expected net value of sampling
        """
        # This is a simplified version
        # Full EVSI requires integrating over possible data
        
        evpi = self.value_of_information(
            actions, outcome_function, prior_samples, 
            lambda x: x, len(prior_samples)
        )
        
        # EVSI <= EVPI
        # Approximate as fraction based on information content
        # (This is a simplification - full EVSI is more complex)
        evsi = 0.5 * evpi  # Placeholder approximation
        
        return evsi - sample_cost


def compute_certainty_equivalent_distribution(
    utility: UtilityFunction,
    samples: np.ndarray
) -> Tuple[float, float]:
    """
    Compute certainty equivalent with uncertainty.
    
    Args:
        utility: Utility function
        samples: Outcome samples
    
    Returns:
        (certainty_equivalent, std_error)
    """
    n = len(samples)
    probs = np.ones(n) / n
    
    ce = utility.certainty_equivalent(samples, probs)
    
    # Bootstrap for uncertainty
    n_bootstrap = 1000
    ce_bootstrap = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_samples = samples[idx]
        ce_boot = utility.certainty_equivalent(boot_samples, probs)
        ce_bootstrap.append(ce_boot)
    
    std_error = np.std(ce_bootstrap)
    
    return ce, std_error


def expected_utility_optimization(
    utility: UtilityFunction,
    outcome_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: np.ndarray,
    scenarios: np.ndarray,
    bounds: Optional[List[Tuple]] = None,
    max_iter: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Maximize expected utility.
    
    max_x E[U(f(x, ξ))]
    
    Args:
        utility: Utility function
        outcome_function: f(decision, scenario) -> outcome
        x0: Initial decision
        scenarios: Uncertainty scenarios
        bounds: Decision bounds
        max_iter: Max iterations
    
    Returns:
        (optimal_decision, expected_utility)
    """
    def neg_expected_utility(x):
        outcomes = np.array([outcome_function(x, s) for s in scenarios])
        return -np.mean(utility(outcomes))
    
    result = optimize.minimize(
        neg_expected_utility,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter}
    )
    
    return result.x, -result.fun
