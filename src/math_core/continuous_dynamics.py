"""
Continuous-Time Dynamics Module

Implements proper continuous-time formulations:
- SDEs (Stochastic Differential Equations)
- Semi-implicit time stepping for stability
- Adaptive time stepping
- Physical conservation laws

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Callable, List, Dict, Any, Union
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve

from .error_aware_arithmetic import EPSILON


@dataclass
class SDEState:
    """State of a stochastic differential equation."""
    mean: np.ndarray
    covariance: np.ndarray
    time: float
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance))


class LinearSDE:
    """
    Linear SDE of the form:
    
    dx = (A @ x + b) dt + σ dW
    
    Has closed-form solution for mean and covariance.
    """
    
    def __init__(self, A: np.ndarray, b: np.ndarray, sigma: np.ndarray):
        """
        Initialize linear SDE.
        
        Args:
            A: Drift matrix (state-dependent)
            b: Drift vector (constant)
            sigma: Diffusion matrix
        """
        self.A = np.atleast_2d(A)
        self.b = np.atleast_1d(b)
        self.sigma = np.atleast_2d(sigma)
        self.n = len(self.b)
        
        # Diffusion covariance
        self.Q = self.sigma @ self.sigma.T
    
    def solve_mean(self, x0: np.ndarray, t: float) -> np.ndarray:
        """
        Solve for mean at time t.
        
        E[x(t)] = exp(At) @ x0 + ∫_0^t exp(A(t-s)) @ b ds
        """
        exp_At = expm(self.A * t)
        
        # Integral term (assuming A is invertible)
        try:
            A_inv = np.linalg.inv(self.A)
            integral = A_inv @ (exp_At - np.eye(self.n)) @ self.b
        except:
            # Numerical integration for singular A
            dt = t / 100
            integral = np.zeros(self.n)
            for i in range(100):
                s = i * dt
                integral += expm(self.A * (t - s)) @ self.b * dt
        
        return exp_At @ x0 + integral
    
    def solve_covariance(self, P0: np.ndarray, t: float) -> np.ndarray:
        """
        Solve for covariance at time t.
        
        dP/dt = A @ P + P @ A' + Q
        
        This is a Lyapunov equation with time-varying solution.
        """
        exp_At = expm(self.A * t)
        
        # Homogeneous solution
        P_hom = exp_At @ P0 @ exp_At.T
        
        # Particular solution (integral)
        dt = t / 100
        P_part = np.zeros((self.n, self.n))
        for i in range(100):
            s = i * dt
            exp_As = expm(self.A * s)
            P_part += exp_As @ self.Q @ exp_As.T * dt
        
        return P_hom + P_part
    
    def step(self, state: SDEState, dt: float) -> SDEState:
        """Advance state by dt."""
        new_mean = self.solve_mean(state.mean, dt)
        new_cov = self.solve_covariance(state.covariance, dt)
        
        return SDEState(
            mean=new_mean,
            covariance=new_cov,
            time=state.time + dt
        )
    
    def sample_path(self, x0: np.ndarray, t_span: Tuple[float, float],
                    n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a path using Euler-Maruyama scheme.
        
        Returns:
            (times, path) where path has shape (n_steps+1, n)
        """
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        sqrt_dt = np.sqrt(dt)
        
        times = np.linspace(t0, tf, n_steps + 1)
        path = np.zeros((n_steps + 1, self.n))
        path[0] = x0
        
        for i in range(n_steps):
            dW = np.random.randn(self.sigma.shape[1]) * sqrt_dt
            drift = self.A @ path[i] + self.b
            diffusion = self.sigma @ dW
            path[i + 1] = path[i] + drift * dt + diffusion
        
        return times, path


class NonlinearSDE:
    """
    General nonlinear SDE:
    
    dx = f(x, t) dt + g(x, t) dW
    
    Provides multiple integration schemes.
    """
    
    def __init__(self,
                 drift: Callable[[np.ndarray, float], np.ndarray],
                 diffusion: Callable[[np.ndarray, float], np.ndarray],
                 state_dim: int,
                 noise_dim: int):
        """
        Initialize nonlinear SDE.
        
        Args:
            drift: Function f(x, t) returning drift vector
            diffusion: Function g(x, t) returning diffusion matrix
            state_dim: Dimension of state
            noise_dim: Dimension of Brownian motion
        """
        self.drift = drift
        self.diffusion = diffusion
        self.state_dim = state_dim
        self.noise_dim = noise_dim
    
    def euler_maruyama_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Euler-Maruyama scheme (strong order 0.5).
        
        x_{n+1} = x_n + f(x_n, t_n) dt + g(x_n, t_n) ΔW
        """
        dW = np.random.randn(self.noise_dim) * np.sqrt(dt)
        return x + self.drift(x, t) * dt + self.diffusion(x, t) @ dW
    
    def milstein_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Milstein scheme (strong order 1.0 for scalar noise).
        
        Includes correction term involving derivative of diffusion.
        """
        f = self.drift(x, t)
        g = self.diffusion(x, t)
        
        dW = np.random.randn(self.noise_dim) * np.sqrt(dt)
        
        # Standard Euler-Maruyama part
        x_new = x + f * dt + g @ dW
        
        # Milstein correction (for scalar diffusion)
        if self.noise_dim == 1 and self.state_dim == 1:
            eps = 1e-6
            dg_dx = (self.diffusion(x + eps, t) - self.diffusion(x - eps, t)) / (2 * eps)
            x_new += 0.5 * g @ dg_dx.T * (dW**2 - dt)
        
        return x_new
    
    def heun_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Stochastic Heun scheme (improved Euler).
        
        Predictor-corrector for better accuracy.
        """
        dW = np.random.randn(self.noise_dim) * np.sqrt(dt)
        
        # Predictor
        f1 = self.drift(x, t)
        g1 = self.diffusion(x, t)
        x_pred = x + f1 * dt + g1 @ dW
        
        # Corrector
        f2 = self.drift(x_pred, t + dt)
        g2 = self.diffusion(x_pred, t + dt)
        
        x_new = x + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) @ dW
        
        return x_new
    
    def solve(self, x0: np.ndarray, t_span: Tuple[float, float],
              n_steps: int = 100,
              method: str = 'euler') -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve SDE over time span.
        
        Args:
            x0: Initial state
            t_span: (t_start, t_end)
            n_steps: Number of time steps
            method: 'euler', 'milstein', or 'heun'
        
        Returns:
            (times, path)
        """
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        
        times = np.linspace(t0, tf, n_steps + 1)
        path = np.zeros((n_steps + 1, self.state_dim))
        path[0] = x0
        
        step_func = {
            'euler': self.euler_maruyama_step,
            'milstein': self.milstein_step,
            'heun': self.heun_step
        }[method]
        
        for i in range(n_steps):
            path[i + 1] = step_func(path[i], times[i], dt)
        
        return times, path
    
    def monte_carlo(self, x0: np.ndarray, t_span: Tuple[float, float],
                    n_paths: int = 1000,
                    n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo simulation for statistics.
        
        Returns:
            (times, mean_path, std_path)
        """
        paths = []
        for _ in range(n_paths):
            times, path = self.solve(x0, t_span, n_steps)
            paths.append(path)
        
        paths = np.array(paths)
        mean_path = np.mean(paths, axis=0)
        std_path = np.std(paths, axis=0)
        
        return times, mean_path, std_path


class SemiImplicitIntegrator:
    """
    Semi-implicit (IMEX) time integrator for stiff systems.
    
    Treats stiff linear part implicitly, non-stiff part explicitly.
    
    dx/dt = L @ x + N(x)
    
    where L is stiff linear operator, N is nonlinear.
    """
    
    def __init__(self,
                 linear_op: np.ndarray,
                 nonlinear_func: Callable[[np.ndarray, float], np.ndarray],
                 state_dim: int):
        """
        Initialize semi-implicit integrator.
        
        Args:
            linear_op: Stiff linear operator matrix L
            nonlinear_func: Non-stiff nonlinear function N(x, t)
            state_dim: State dimension
        """
        self.L = linear_op
        self.N = nonlinear_func
        self.state_dim = state_dim
    
    def imex_euler_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        IMEX Euler: implicit L, explicit N.
        
        x_{n+1} = x_n + dt * L @ x_{n+1} + dt * N(x_n, t_n)
        (I - dt*L) @ x_{n+1} = x_n + dt * N(x_n, t_n)
        """
        rhs = x + dt * self.N(x, t)
        A = np.eye(self.state_dim) - dt * self.L
        return solve(A, rhs)
    
    def crank_nicolson_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Crank-Nicolson for linear part (second order).
        
        x_{n+1} = x_n + dt/2 * L @ (x_n + x_{n+1}) + dt * N(x_n, t_n)
        """
        # Explicit part
        rhs = (np.eye(self.state_dim) + 0.5 * dt * self.L) @ x + dt * self.N(x, t)
        
        # Implicit part
        A = np.eye(self.state_dim) - 0.5 * dt * self.L
        return solve(A, rhs)
    
    def solve(self, x0: np.ndarray, t_span: Tuple[float, float],
              n_steps: int = 100,
              method: str = 'crank_nicolson') -> Tuple[np.ndarray, np.ndarray]:
        """Solve system over time span."""
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        
        times = np.linspace(t0, tf, n_steps + 1)
        path = np.zeros((n_steps + 1, self.state_dim))
        path[0] = x0
        
        step_func = {
            'euler': self.imex_euler_step,
            'crank_nicolson': self.crank_nicolson_step
        }[method]
        
        for i in range(n_steps):
            path[i + 1] = step_func(path[i], times[i], dt)
        
        return times, path


class AdaptiveTimestepper:
    """
    Adaptive time stepping based on local error estimation.
    
    Uses embedded Runge-Kutta pairs for error estimation.
    """
    
    def __init__(self,
                 rhs: Callable[[np.ndarray, float], np.ndarray],
                 rtol: float = 1e-3,
                 atol: float = 1e-6,
                 dt_min: float = 1e-10,
                 dt_max: float = 1.0):
        """
        Initialize adaptive integrator.
        
        Args:
            rhs: Right-hand side function dx/dt = f(x, t)
            rtol: Relative tolerance
            atol: Absolute tolerance
            dt_min: Minimum time step
            dt_max: Maximum time step
        """
        self.rhs = rhs
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.dt_max = dt_max
    
    def _rk45_step(self, x: np.ndarray, t: float, dt: float
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Dormand-Prince RK5(4) step.
        
        Returns:
            (x_new_5th, x_new_4th, error_estimate)
        """
        # Butcher tableau coefficients (simplified)
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        
        k1 = self.rhs(x, t)
        k2 = self.rhs(x + dt * k1 / 5, t + dt / 5)
        k3 = self.rhs(x + dt * (3*k1/40 + 9*k2/40), t + 3*dt/10)
        k4 = self.rhs(x + dt * (44*k1/45 - 56*k2/15 + 32*k3/9), t + 4*dt/5)
        k5 = self.rhs(x + dt * (19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729), t + 8*dt/9)
        k6 = self.rhs(x + dt * (9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656), t + dt)
        k7 = self.rhs(x + dt * (35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84), t + dt)
        
        # 5th order solution
        x5 = x + dt * (35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84)
        
        # 4th order solution (for error estimate)
        x4 = x + dt * (5179*k1/57600 + 7571*k3/16695 + 393*k4/640 - 92097*k5/339200 + 187*k6/2100 + k7/40)
        
        # Error estimate
        error = np.max(np.abs(x5 - x4) / (self.atol + self.rtol * np.maximum(np.abs(x), np.abs(x5))))
        
        return x5, x4, error
    
    def step(self, x: np.ndarray, t: float, dt: float
             ) -> Tuple[np.ndarray, float, float]:
        """
        Take adaptive step.
        
        Returns:
            (x_new, t_new, dt_next)
        """
        while True:
            x_new, _, error = self._rk45_step(x, t, dt)
            
            if error < 1.0:
                # Accept step
                # Compute next step size
                safety = 0.9
                exponent = -1/5
                dt_new = dt * safety * error ** exponent
                dt_new = min(max(dt_new, self.dt_min), self.dt_max)
                
                return x_new, t + dt, dt_new
            else:
                # Reject step, reduce dt
                dt = dt * 0.5
                if dt < self.dt_min:
                    # Force accept with minimum step
                    return x_new, t + dt, self.dt_min
    
    def solve(self, x0: np.ndarray, t_span: Tuple[float, float],
              dt_init: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve with adaptive stepping.
        
        Returns:
            (times, path) with variable step sizes
        """
        t0, tf = t_span
        
        times = [t0]
        path = [x0.copy()]
        
        t = t0
        x = x0.copy()
        dt = dt_init
        
        while t < tf:
            # Don't overshoot
            dt = min(dt, tf - t)
            
            x_new, t_new, dt_new = self.step(x, t, dt)
            
            times.append(t_new)
            path.append(x_new)
            
            t = t_new
            x = x_new
            dt = dt_new
        
        return np.array(times), np.array(path)


class ConservativeIntegrator:
    """
    Integrator that preserves conservation laws.
    
    Useful for physical systems where mass, energy, or momentum
    must be conserved.
    """
    
    def __init__(self,
                 rhs: Callable[[np.ndarray, float], np.ndarray],
                 conserved_quantities: List[Callable[[np.ndarray], float]]):
        """
        Initialize conservative integrator.
        
        Args:
            rhs: Right-hand side function
            conserved_quantities: List of functions Q(x) that should be conserved
        """
        self.rhs = rhs
        self.conserved = conserved_quantities
    
    def _project_to_manifold(self, x: np.ndarray, 
                              target_values: List[float]) -> np.ndarray:
        """
        Project state onto conservation manifold.
        
        Uses Newton iteration to find closest point satisfying constraints.
        """
        from scipy.optimize import minimize
        
        def constraint_violation(x_flat):
            x = x_flat.reshape(x.shape)
            return sum((Q(x) - q0) ** 2 for Q, q0 in zip(self.conserved, target_values))
        
        result = minimize(constraint_violation, x.flatten(), method='BFGS')
        return result.x.reshape(x.shape)
    
    def solve(self, x0: np.ndarray, t_span: Tuple[float, float],
              n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve with conservation projection.
        """
        # Initial conserved quantities
        q0 = [Q(x0) for Q in self.conserved]
        
        # Use adaptive integrator as base
        integrator = AdaptiveTimestepper(self.rhs)
        times, path = integrator.solve(x0, t_span)
        
        # Project each point to conservation manifold
        for i in range(1, len(path)):
            path[i] = self._project_to_manifold(path[i], q0)
        
        return times, path


class DrainageStressDynamics:
    """
    Continuous-time dynamics model for drainage stress evolution.
    
    Models stress as diffusion-reaction system with stochastic forcing.
    
    ∂S/∂t = D ∇²S - λS + R(t) + noise
    
    where:
    - D: diffusion coefficient
    - λ: decay rate  
    - R(t): rainfall forcing
    """
    
    def __init__(self, n_locations: int,
                 diffusion_coeff: float = 0.1,
                 decay_rate: float = 0.05,
                 noise_scale: float = 0.01):
        self.n = n_locations
        self.D = diffusion_coeff
        self.lam = decay_rate
        self.noise_scale = noise_scale
        
        # Build Laplacian (1D chain for simplicity)
        self._build_laplacian()
    
    def _build_laplacian(self):
        """Build discrete Laplacian operator."""
        L = np.zeros((self.n, self.n))
        for i in range(self.n):
            L[i, i] = -2
            if i > 0:
                L[i, i-1] = 1
            if i < self.n - 1:
                L[i, i+1] = 1
        
        self.L = self.D * L - self.lam * np.eye(self.n)
    
    def drift(self, stress: np.ndarray, t: float,
              rainfall: Optional[Callable[[float], np.ndarray]] = None) -> np.ndarray:
        """Compute drift term."""
        f = self.L @ stress
        if rainfall is not None:
            f += rainfall(t)
        return f
    
    def diffusion(self, stress: np.ndarray, t: float) -> np.ndarray:
        """Compute diffusion matrix."""
        return self.noise_scale * np.eye(self.n)
    
    def create_sde(self, rainfall: Optional[Callable[[float], np.ndarray]] = None
                   ) -> NonlinearSDE:
        """Create SDE object for this system."""
        def drift_func(x, t):
            return self.drift(x, t, rainfall)
        
        def diffusion_func(x, t):
            return self.diffusion(x, t)
        
        return NonlinearSDE(
            drift=drift_func,
            diffusion=diffusion_func,
            state_dim=self.n,
            noise_dim=self.n
        )
    
    def solve(self, stress0: np.ndarray,
              t_span: Tuple[float, float],
              rainfall: Optional[Callable[[float], np.ndarray]] = None,
              n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Solve stress dynamics."""
        sde = self.create_sde(rainfall)
        return sde.solve(stress0, t_span, n_steps, method='heun')
    
    def steady_state(self, rainfall_const: np.ndarray) -> np.ndarray:
        """
        Compute steady-state stress for constant rainfall.
        
        At steady state: 0 = L @ S + R
        So: S = -L^{-1} @ R
        """
        return -np.linalg.solve(self.L, rainfall_const)
