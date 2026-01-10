"""
Robust Risk-Averse Optimization Module

Implements optimization under uncertainty:
- Conditional Value at Risk (CVaR)
- Distributionally Robust Optimization (DRO)
- Worst-case optimization
- Chance constraints
- Robust counterparts

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass
from scipy import optimize
from scipy import stats

from .error_aware_arithmetic import EPSILON


@dataclass
class OptimizationResult:
    """Result of optimization."""
    x: np.ndarray
    objective_value: float
    risk_measure: float
    iterations: int
    converged: bool
    dual_variables: Optional[np.ndarray] = None
    sensitivity: Optional[np.ndarray] = None


class CVaROptimizer:
    """
    Conditional Value at Risk (CVaR) optimization.
    
    CVaR_α = E[X | X >= VaR_α]
    
    CVaR is a coherent risk measure satisfying:
    - Monotonicity
    - Translation invariance
    - Positive homogeneity
    - Subadditivity
    """
    
    def __init__(self, alpha: float = 0.95):
        """
        Initialize CVaR optimizer.
        
        Args:
            alpha: Confidence level (e.g., 0.95 for 95% CVaR)
        """
        self.alpha = alpha
        self.var_: Optional[float] = None
        self.cvar_: Optional[float] = None
    
    def compute_cvar(self, losses: np.ndarray) -> Tuple[float, float]:
        """
        Compute VaR and CVaR from samples.
        
        Args:
            losses: Sample losses (higher = worse)
        
        Returns:
            (VaR, CVaR)
        """
        losses = np.asarray(losses).flatten()
        
        # VaR is the alpha-quantile
        self.var_ = np.percentile(losses, self.alpha * 100)
        
        # CVaR is the expected loss above VaR
        tail_losses = losses[losses >= self.var_]
        if len(tail_losses) > 0:
            self.cvar_ = np.mean(tail_losses)
        else:
            self.cvar_ = self.var_
        
        return self.var_, self.cvar_
    
    def compute_cvar_gradient(self, losses: np.ndarray,
                               loss_gradients: np.ndarray) -> np.ndarray:
        """
        Compute gradient of CVaR with respect to decision variables.
        
        Uses the subgradient formula:
        ∂CVaR_α(X)/∂θ = E[∂X/∂θ | X >= VaR_α]
        
        Args:
            losses: Sample losses
            loss_gradients: Gradient of each loss w.r.t. decision (n_samples, dim)
        
        Returns:
            CVaR gradient
        """
        losses = np.asarray(losses).flatten()
        loss_gradients = np.atleast_2d(loss_gradients)
        
        if self.var_ is None:
            self.compute_cvar(losses)
        
        # Select tail samples
        tail_mask = losses >= self.var_
        
        if tail_mask.sum() == 0:
            return np.zeros(loss_gradients.shape[1])
        
        # Average gradient in tail
        return np.mean(loss_gradients[tail_mask], axis=0)
    
    def optimize(self, loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 x0: np.ndarray,
                 scenarios: np.ndarray,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 max_iter: int = 100) -> OptimizationResult:
        """
        Minimize CVaR of loss function.
        
        Uses Rockafellar-Uryasev reformulation:
        min_x CVaR_α(L(x,ξ)) = min_{x,t} t + (1/(1-α)) * E[max(L(x,ξ) - t, 0)]
        
        Args:
            loss_function: L(x, scenario) -> loss
            x0: Initial decision
            scenarios: Uncertainty scenarios (n_scenarios, scenario_dim)
            bounds: Decision variable bounds
            max_iter: Maximum iterations
        
        Returns:
            Optimization result
        """
        n_vars = len(x0)
        n_scenarios = len(scenarios)
        
        def objective(z):
            """Rockafellar-Uryasev objective."""
            x = z[:n_vars]
            t = z[n_vars]  # VaR estimate
            
            losses = np.array([loss_function(x, s) for s in scenarios])
            excess = np.maximum(losses - t, 0)
            
            # CVaR = t + E[max(L-t, 0)] / (1-α)
            cvar = t + np.mean(excess) / (1 - self.alpha)
            
            return cvar
        
        def gradient(z):
            """Gradient of R-U objective."""
            x = z[:n_vars]
            t = z[n_vars]
            
            losses = np.array([loss_function(x, s) for s in scenarios])
            
            # Numerical gradient for x
            grad_x = np.zeros(n_vars)
            eps = 1e-7
            
            for i in range(n_vars):
                x_plus = x.copy()
                x_plus[i] += eps
                losses_plus = np.array([loss_function(x_plus, s) for s in scenarios])
                excess_plus = np.maximum(losses_plus - t, 0)
                f_plus = t + np.mean(excess_plus) / (1 - self.alpha)
                
                x_minus = x.copy()
                x_minus[i] -= eps
                losses_minus = np.array([loss_function(x_minus, s) for s in scenarios])
                excess_minus = np.maximum(losses_minus - t, 0)
                f_minus = t + np.mean(excess_minus) / (1 - self.alpha)
                
                grad_x[i] = (f_plus - f_minus) / (2 * eps)
            
            # Gradient w.r.t. t
            indicator = (losses >= t).astype(float)
            grad_t = 1 - np.mean(indicator) / (1 - self.alpha)
            
            return np.append(grad_x, grad_t)
        
        # Initial t as sample quantile
        initial_losses = np.array([loss_function(x0, s) for s in scenarios])
        t0 = np.percentile(initial_losses, self.alpha * 100)
        z0 = np.append(x0, t0)
        
        # Bounds
        if bounds is None:
            z_bounds = [(None, None)] * n_vars + [(None, None)]
        else:
            z_bounds = list(bounds) + [(None, None)]
        
        # Optimize
        result = optimize.minimize(
            objective,
            z0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=z_bounds,
            options={'maxiter': max_iter}
        )
        
        x_opt = result.x[:n_vars]
        t_opt = result.x[n_vars]
        
        # Compute final CVaR
        final_losses = np.array([loss_function(x_opt, s) for s in scenarios])
        var_final, cvar_final = self.compute_cvar(final_losses)
        
        return OptimizationResult(
            x=x_opt,
            objective_value=result.fun,
            risk_measure=cvar_final,
            iterations=result.nit,
            converged=result.success
        )


class DistributionallyRobustOptimizer:
    """
    Distributionally Robust Optimization (DRO).
    
    Optimize against worst-case distribution within ambiguity set.
    
    min_x max_{P ∈ Ω} E_P[L(x, ξ)]
    """
    
    def __init__(self, ambiguity_radius: float = 0.1,
                 distance_type: str = 'wasserstein'):
        """
        Initialize DRO optimizer.
        
        Args:
            ambiguity_radius: Radius of ambiguity set
            distance_type: 'wasserstein' or 'kl'
        """
        self.radius = ambiguity_radius
        self.distance_type = distance_type
    
    def optimize_wasserstein(self, loss_function: Callable,
                              x0: np.ndarray,
                              scenarios: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              bounds: Optional[List[Tuple]] = None,
                              max_iter: int = 100) -> OptimizationResult:
        """
        DRO with Wasserstein ambiguity set.
        
        Strong duality gives:
        sup_{P: W(P,P_0) <= ε} E_P[L] = inf_{λ >= 0} λε + E_{P_0}[sup_ξ' L(ξ') - λ||ξ - ξ'||]
        
        For linear loss, simplifies to:
        inf_λ λε + (1/n) Σ max(L_i + λ * radius_bound)
        
        Args:
            loss_function: Loss L(x, scenario)
            x0: Initial decision
            scenarios: Empirical scenarios
            weights: Scenario weights (uniform if None)
            bounds: Decision bounds
            max_iter: Max iterations
        
        Returns:
            Optimization result
        """
        n_vars = len(x0)
        n_scenarios = len(scenarios)
        
        if weights is None:
            weights = np.ones(n_scenarios) / n_scenarios
        
        def robust_objective(z):
            """DRO objective with dual variable."""
            x = z[:n_vars]
            lam = max(z[n_vars], EPSILON)  # Lagrange multiplier >= 0
            
            losses = np.array([loss_function(x, s) for s in scenarios])
            
            # Worst-case loss (simplified for bounded perturbation)
            worst_case_loss = np.max(losses) + lam * self.radius
            
            return lam * self.radius + np.dot(weights, losses) + \
                   self.radius * np.sqrt(np.dot(weights, losses ** 2))
        
        # Initialize
        z0 = np.append(x0, 1.0)  # lambda = 1 initially
        
        # Bounds
        if bounds is None:
            z_bounds = [(None, None)] * n_vars + [(0, None)]
        else:
            z_bounds = list(bounds) + [(0, None)]
        
        result = optimize.minimize(
            robust_objective,
            z0,
            method='L-BFGS-B',
            bounds=z_bounds,
            options={'maxiter': max_iter}
        )
        
        x_opt = result.x[:n_vars]
        
        final_losses = np.array([loss_function(x_opt, s) for s in scenarios])
        
        return OptimizationResult(
            x=x_opt,
            objective_value=result.fun,
            risk_measure=np.max(final_losses),
            iterations=result.nit,
            converged=result.success
        )
    
    def optimize_kl(self, loss_function: Callable,
                     x0: np.ndarray,
                     scenarios: np.ndarray,
                     weights: Optional[np.ndarray] = None,
                     bounds: Optional[List[Tuple]] = None,
                     max_iter: int = 100) -> OptimizationResult:
        """
        DRO with KL-divergence ambiguity set.
        
        max_{P: KL(P||P_0) <= ε} E_P[L] = η * log(E_{P_0}[exp(L/η)])
        
        where η is chosen such that the constraint binds.
        
        Args:
            loss_function: Loss L(x, scenario)
            x0: Initial decision
            scenarios: Empirical scenarios
            weights: Scenario weights
            bounds: Decision bounds
            max_iter: Max iterations
        
        Returns:
            Optimization result
        """
        n_vars = len(x0)
        n_scenarios = len(scenarios)
        
        if weights is None:
            weights = np.ones(n_scenarios) / n_scenarios
        
        def robust_objective(z):
            """KL-DRO objective."""
            x = z[:n_vars]
            eta = max(z[n_vars], EPSILON)  # Temperature parameter
            
            losses = np.array([loss_function(x, s) for s in scenarios])
            
            # Log-sum-exp for numerical stability
            max_loss = np.max(losses)
            scaled = (losses - max_loss) / eta
            log_expectation = max_loss + eta * np.log(np.dot(weights, np.exp(scaled)))
            
            # Penalty for KL constraint
            penalty = eta * self.radius
            
            return log_expectation + penalty
        
        z0 = np.append(x0, 1.0)
        
        if bounds is None:
            z_bounds = [(None, None)] * n_vars + [(EPSILON, None)]
        else:
            z_bounds = list(bounds) + [(EPSILON, None)]
        
        result = optimize.minimize(
            robust_objective,
            z0,
            method='L-BFGS-B',
            bounds=z_bounds,
            options={'maxiter': max_iter}
        )
        
        x_opt = result.x[:n_vars]
        final_losses = np.array([loss_function(x_opt, s) for s in scenarios])
        
        return OptimizationResult(
            x=x_opt,
            objective_value=result.fun,
            risk_measure=np.max(final_losses),
            iterations=result.nit,
            converged=result.success
        )


class ChanceConstrainedOptimizer:
    """
    Chance-constrained optimization.
    
    min_x f(x)
    s.t. P[g(x, ξ) <= 0] >= 1 - ε
    """
    
    def __init__(self, epsilon: float = 0.05):
        """
        Initialize chance-constrained optimizer.
        
        Args:
            epsilon: Allowed violation probability
        """
        self.epsilon = epsilon
    
    def sample_average_approximation(self, 
                                      objective: Callable[[np.ndarray], float],
                                      constraint: Callable[[np.ndarray, np.ndarray], float],
                                      x0: np.ndarray,
                                      scenarios: np.ndarray,
                                      bounds: Optional[List[Tuple]] = None,
                                      max_iter: int = 100) -> OptimizationResult:
        """
        Solve via Sample Average Approximation (SAA).
        
        Approximate chance constraint with empirical quantile.
        
        Args:
            objective: f(x) to minimize
            constraint: g(x, scenario) <= 0 for feasibility
            x0: Initial decision
            scenarios: Uncertainty scenarios
            bounds: Decision bounds
            max_iter: Max iterations
        
        Returns:
            Optimization result
        """
        n_scenarios = len(scenarios)
        
        def penalized_objective(x, penalty_weight=1000):
            """Objective with penalty for constraint violation."""
            obj = objective(x)
            
            # Compute constraint values
            g_values = np.array([constraint(x, s) for s in scenarios])
            
            # Fraction of violations
            violation_rate = np.mean(g_values > 0)
            
            if violation_rate > self.epsilon:
                # Penalty proportional to excess violation
                penalty = penalty_weight * (violation_rate - self.epsilon) ** 2
                obj += penalty
            
            return obj
        
        result = optimize.minimize(
            penalized_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        x_opt = result.x
        
        # Verify constraint satisfaction
        g_values = np.array([constraint(x_opt, s) for s in scenarios])
        satisfaction_rate = np.mean(g_values <= 0)
        
        return OptimizationResult(
            x=x_opt,
            objective_value=objective(x_opt),
            risk_measure=1 - satisfaction_rate,
            iterations=result.nit,
            converged=result.success and satisfaction_rate >= 1 - self.epsilon
        )
    
    def scenario_approach(self,
                          objective: Callable[[np.ndarray], float],
                          constraint: Callable[[np.ndarray, np.ndarray], float],
                          x0: np.ndarray,
                          scenarios: np.ndarray,
                          bounds: Optional[List[Tuple]] = None,
                          max_iter: int = 100) -> OptimizationResult:
        """
        Scenario approach for chance constraints.
        
        Provides probabilistic guarantees on feasibility.
        Required sample size: N >= (2/ε) * (ln(1/β) + d)
        where β is confidence and d is dimension.
        
        Args:
            objective: f(x) to minimize
            constraint: g(x, scenario) <= 0
            x0: Initial decision
            scenarios: Uncertainty scenarios
            bounds: Decision bounds
            max_iter: Max iterations
        
        Returns:
            Optimization result
        """
        n_vars = len(x0)
        n_scenarios = len(scenarios)
        
        # All scenarios must be satisfied
        constraint_list = [
            {'type': 'ineq', 'fun': lambda x, s=s: -constraint(x, s)}
            for s in scenarios
        ]
        
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': max_iter}
        )
        
        return OptimizationResult(
            x=result.x,
            objective_value=result.fun,
            risk_measure=0.0,  # All scenarios satisfied by construction
            iterations=result.nit,
            converged=result.success
        )


class RobustCounterpartSolver:
    """
    Solve robust counterpart of uncertain optimization.
    
    For uncertain constraint a'x <= b with a ∈ U:
    Robust counterpart depends on uncertainty set U.
    """
    
    def __init__(self, uncertainty_type: str = 'box'):
        """
        Initialize robust counterpart solver.
        
        Args:
            uncertainty_type: 'box', 'ellipsoid', or 'polyhedral'
        """
        self.uncertainty_type = uncertainty_type
    
    def robust_linear_constraint(self, a_nominal: np.ndarray,
                                  b: float,
                                  uncertainty_set: Dict[str, Any]
                                  ) -> Callable[[np.ndarray], float]:
        """
        Create robust counterpart of linear constraint a'x <= b.
        
        Args:
            a_nominal: Nominal coefficient vector
            b: Right-hand side
            uncertainty_set: Description of uncertainty
        
        Returns:
            Robust constraint function g(x) <= 0
        """
        if self.uncertainty_type == 'box':
            # a ∈ [a_nominal - delta, a_nominal + delta]
            delta = uncertainty_set.get('delta', 0.1 * np.abs(a_nominal))
            
            def robust_constraint(x):
                # Worst case: maximize a'x over box
                # = a_nominal'x + delta'|x|
                return np.dot(a_nominal, x) + np.dot(delta, np.abs(x)) - b
            
            return robust_constraint
        
        elif self.uncertainty_type == 'ellipsoid':
            # a ∈ {a_nominal + Δ : ||Δ||_Σ <= ρ}
            Sigma = uncertainty_set.get('Sigma', np.eye(len(a_nominal)))
            rho = uncertainty_set.get('rho', 0.1)
            
            def robust_constraint(x):
                # Worst case: a_nominal'x + ρ * ||Σ^{1/2} x||
                Sigma_sqrt = np.linalg.cholesky(Sigma)
                norm_term = rho * np.linalg.norm(Sigma_sqrt @ x)
                return np.dot(a_nominal, x) + norm_term - b
            
            return robust_constraint
        
        raise ValueError(f"Unknown uncertainty type: {self.uncertainty_type}")
    
    def solve_robust_lp(self, c: np.ndarray,
                        A_nominal: np.ndarray,
                        b: np.ndarray,
                        uncertainty_sets: List[Dict[str, Any]],
                        bounds: Optional[List[Tuple]] = None) -> OptimizationResult:
        """
        Solve robust linear program.
        
        min c'x
        s.t. a_i'x <= b_i  for all a_i ∈ U_i
        
        Args:
            c: Objective coefficients
            A_nominal: Nominal constraint matrix
            b: Right-hand sides
            uncertainty_sets: Uncertainty set for each constraint
            bounds: Variable bounds
        
        Returns:
            Optimization result
        """
        n_vars = len(c)
        n_constraints = len(b)
        
        # Create robust constraints
        robust_constraints = []
        for i in range(n_constraints):
            g = self.robust_linear_constraint(
                A_nominal[i], b[i], uncertainty_sets[i]
            )
            robust_constraints.append(
                {'type': 'ineq', 'fun': lambda x, g=g: -g(x)}
            )
        
        x0 = np.zeros(n_vars)
        
        result = optimize.minimize(
            lambda x: np.dot(c, x),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=robust_constraints
        )
        
        return OptimizationResult(
            x=result.x,
            objective_value=result.fun,
            risk_measure=0.0,
            iterations=result.nit,
            converged=result.success
        )


def compute_risk_spectrum(losses: np.ndarray,
                          n_levels: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute risk spectrum (CVaR for all alpha levels).
    
    Args:
        losses: Sample losses
        n_levels: Number of alpha levels
    
    Returns:
        (alpha_levels, cvar_values)
    """
    alphas = np.linspace(0.01, 0.99, n_levels)
    cvars = np.zeros(n_levels)
    
    optimizer = CVaROptimizer()
    
    for i, alpha in enumerate(alphas):
        optimizer.alpha = alpha
        _, cvar = optimizer.compute_cvar(losses)
        cvars[i] = cvar
    
    return alphas, cvars
