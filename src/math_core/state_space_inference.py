"""
State-Space Inference Module

Implements Kalman filtering and extensions:
- Linear Kalman Filter
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Rauch-Tung-Striebel (RTS) smoothing
- State estimation with full covariance tracking

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from scipy.linalg import cholesky, solve_triangular, sqrtm

from .error_aware_arithmetic import EPSILON


@dataclass
class StateEstimate:
    """State estimate with covariance."""
    mean: np.ndarray
    covariance: np.ndarray
    
    @property
    def dim(self) -> int:
        return len(self.mean)
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance))
    
    def credible_interval(self, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute credible interval assuming Gaussian."""
        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)
        lower = self.mean - z * self.std
        upper = self.mean + z * self.std
        return lower, upper


@dataclass
class LinearStateSpaceModel:
    """
    Linear state-space model:
    
    x_t = F @ x_{t-1} + B @ u_t + w_t,  w_t ~ N(0, Q)
    y_t = H @ x_t + v_t,                 v_t ~ N(0, R)
    """
    F: np.ndarray  # State transition matrix
    H: np.ndarray  # Observation matrix
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Observation noise covariance
    B: Optional[np.ndarray] = None  # Control input matrix
    
    @property
    def state_dim(self) -> int:
        return self.F.shape[0]
    
    @property
    def obs_dim(self) -> int:
        return self.H.shape[0]


class KalmanFilter:
    """
    Standard Kalman Filter for linear Gaussian state-space models.
    
    Provides:
    - Prediction (time update)
    - Update (measurement update)
    - Full covariance tracking
    - Innovation statistics
    """
    
    def __init__(self, model: LinearStateSpaceModel):
        self.model = model
        self.current_estimate: Optional[StateEstimate] = None
        
        # Store history for smoothing
        self.predicted_states: List[StateEstimate] = []
        self.filtered_states: List[StateEstimate] = []
        self.innovations: List[np.ndarray] = []
        self.innovation_covariances: List[np.ndarray] = []
    
    def initialize(self, mean: np.ndarray, covariance: np.ndarray):
        """Initialize filter state."""
        self.current_estimate = StateEstimate(
            mean=mean.copy(),
            covariance=covariance.copy()
        )
        self.predicted_states = []
        self.filtered_states = []
        self.innovations = []
        self.innovation_covariances = []
    
    def predict(self, u: Optional[np.ndarray] = None) -> StateEstimate:
        """
        Prediction step (time update).
        
        x_pred = F @ x + B @ u
        P_pred = F @ P @ F' + Q
        """
        if self.current_estimate is None:
            raise ValueError("Filter not initialized")
        
        F = self.model.F
        Q = self.model.Q
        
        # Predicted mean
        x_pred = F @ self.current_estimate.mean
        if u is not None and self.model.B is not None:
            x_pred += self.model.B @ u
        
        # Predicted covariance
        P_pred = F @ self.current_estimate.covariance @ F.T + Q
        
        predicted = StateEstimate(mean=x_pred, covariance=P_pred)
        self.predicted_states.append(predicted)
        
        return predicted
    
    def update(self, y: np.ndarray, predicted: Optional[StateEstimate] = None) -> StateEstimate:
        """
        Update step (measurement update).
        
        K = P_pred @ H' @ (H @ P_pred @ H' + R)^{-1}
        x = x_pred + K @ (y - H @ x_pred)
        P = (I - K @ H) @ P_pred
        """
        if predicted is None:
            predicted = self.predicted_states[-1]
        
        H = self.model.H
        R = self.model.R
        
        # Innovation
        innovation = y - H @ predicted.mean
        
        # Innovation covariance
        S = H @ predicted.covariance @ H.T + R
        
        # Kalman gain (solve S @ K' = H @ P)
        try:
            L = cholesky(S, lower=True)
            K = solve_triangular(L.T, solve_triangular(L, H @ predicted.covariance.T, lower=True)).T
        except:
            # Fallback to direct inversion
            S_inv = np.linalg.inv(S + EPSILON * np.eye(len(S)))
            K = predicted.covariance @ H.T @ S_inv
        
        # Updated state
        x_upd = predicted.mean + K @ innovation
        
        # Updated covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.model.state_dim) - K @ H
        P_upd = I_KH @ predicted.covariance @ I_KH.T + K @ R @ K.T
        
        # Store
        self.current_estimate = StateEstimate(mean=x_upd, covariance=P_upd)
        self.filtered_states.append(self.current_estimate)
        self.innovations.append(innovation)
        self.innovation_covariances.append(S)
        
        return self.current_estimate
    
    def filter(self, observations: np.ndarray,
               controls: Optional[np.ndarray] = None) -> List[StateEstimate]:
        """
        Run filter on sequence of observations.
        
        Args:
            observations: Array of shape (T, obs_dim)
            controls: Optional array of shape (T, control_dim)
        
        Returns:
            List of filtered state estimates
        """
        T = len(observations)
        results = []
        
        for t in range(T):
            u = controls[t] if controls is not None else None
            predicted = self.predict(u)
            filtered = self.update(observations[t], predicted)
            results.append(filtered)
        
        return results
    
    def smooth(self) -> List[StateEstimate]:
        """
        Rauch-Tung-Striebel (RTS) smoother.
        
        Runs backward pass to improve estimates using future observations.
        """
        if len(self.filtered_states) == 0:
            raise ValueError("Must run filter first")
        
        T = len(self.filtered_states)
        F = self.model.F
        
        smoothed = [None] * T
        smoothed[-1] = self.filtered_states[-1]
        
        for t in range(T - 2, -1, -1):
            # Smoother gain
            P_filt = self.filtered_states[t].covariance
            P_pred = self.predicted_states[t + 1].covariance
            
            try:
                L = cholesky(P_pred + EPSILON * np.eye(len(P_pred)), lower=True)
                J = solve_triangular(L.T, solve_triangular(L, (F @ P_filt).T, lower=True)).T
            except:
                J = P_filt @ F.T @ np.linalg.inv(P_pred + EPSILON * np.eye(len(P_pred)))
            
            # Smoothed mean
            x_smooth = self.filtered_states[t].mean + J @ (
                smoothed[t + 1].mean - self.predicted_states[t + 1].mean
            )
            
            # Smoothed covariance
            P_smooth = self.filtered_states[t].covariance + J @ (
                smoothed[t + 1].covariance - self.predicted_states[t + 1].covariance
            ) @ J.T
            
            smoothed[t] = StateEstimate(mean=x_smooth, covariance=P_smooth)
        
        return smoothed
    
    def log_likelihood(self) -> float:
        """Compute log likelihood of observations."""
        log_lik = 0.0
        
        for innov, S in zip(self.innovations, self.innovation_covariances):
            d = len(innov)
            sign, log_det = np.linalg.slogdet(S)
            mahal = innov @ np.linalg.solve(S, innov)
            log_lik += -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
        
        return log_lik


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state-space models.
    
    x_t = f(x_{t-1}, u_t) + w_t
    y_t = h(x_t) + v_t
    
    Uses local linearization via Jacobians.
    """
    
    def __init__(self,
                 f: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
                 h: Callable[[np.ndarray], np.ndarray],
                 F_jacobian: Callable[[np.ndarray], np.ndarray],
                 H_jacobian: Callable[[np.ndarray], np.ndarray],
                 Q: np.ndarray,
                 R: np.ndarray):
        """
        Initialize EKF.
        
        Args:
            f: State transition function
            h: Observation function
            F_jacobian: Jacobian of f w.r.t. state
            H_jacobian: Jacobian of h w.r.t. state
            Q: Process noise covariance
            R: Observation noise covariance
        """
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        
        self.current_estimate: Optional[StateEstimate] = None
    
    def initialize(self, mean: np.ndarray, covariance: np.ndarray):
        """Initialize filter state."""
        self.current_estimate = StateEstimate(
            mean=mean.copy(),
            covariance=covariance.copy()
        )
    
    def predict(self, u: Optional[np.ndarray] = None) -> StateEstimate:
        """EKF prediction step."""
        x = self.current_estimate.mean
        P = self.current_estimate.covariance
        
        # Predicted mean (nonlinear)
        x_pred = self.f(x, u)
        
        # Jacobian at current state
        F = self.F_jacobian(x)
        
        # Predicted covariance (linearized)
        P_pred = F @ P @ F.T + self.Q
        
        return StateEstimate(mean=x_pred, covariance=P_pred)
    
    def update(self, y: np.ndarray, predicted: StateEstimate) -> StateEstimate:
        """EKF update step."""
        x_pred = predicted.mean
        P_pred = predicted.covariance
        
        # Predicted observation (nonlinear)
        y_pred = self.h(x_pred)
        
        # Innovation
        innovation = y - y_pred
        
        # Jacobian at predicted state
        H = self.H_jacobian(x_pred)
        
        # Innovation covariance
        S = H @ P_pred @ H.T + self.R
        
        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S + EPSILON * np.eye(len(S)))
        
        # Update
        x_upd = x_pred + K @ innovation
        P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
        
        self.current_estimate = StateEstimate(mean=x_upd, covariance=P_upd)
        return self.current_estimate


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear state-space models.
    
    Uses sigma points to propagate mean and covariance through
    nonlinear transformations without requiring Jacobians.
    """
    
    def __init__(self,
                 f: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
                 h: Callable[[np.ndarray], np.ndarray],
                 Q: np.ndarray,
                 R: np.ndarray,
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        """
        Initialize UKF.
        
        Args:
            f: State transition function
            h: Observation function
            Q: Process noise covariance
            R: Observation noise covariance
            alpha: Scaling parameter (spread of sigma points)
            beta: Prior distribution parameter (2 for Gaussian)
            kappa: Secondary scaling parameter
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.current_estimate: Optional[StateEstimate] = None
    
    def initialize(self, mean: np.ndarray, covariance: np.ndarray):
        """Initialize filter state."""
        self.current_estimate = StateEstimate(
            mean=mean.copy(),
            covariance=covariance.copy()
        )
    
    def _compute_sigma_points(self, mean: np.ndarray, 
                               covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute sigma points and weights.
        
        Returns:
            (sigma_points, weights_mean, weights_cov)
        """
        n = len(mean)
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        
        # Weights
        Wm = np.full(2 * n + 1, 0.5 / (n + lambda_))
        Wc = Wm.copy()
        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)
        
        # Sigma points
        try:
            sqrt_P = sqrtm((n + lambda_) * covariance)
            sqrt_P = np.real(sqrt_P)
        except:
            sqrt_P = cholesky((n + lambda_) * covariance + EPSILON * np.eye(n), lower=True)
        
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_P[i]
            sigma_points[n + i + 1] = mean - sqrt_P[i]
        
        return sigma_points, Wm, Wc
    
    def predict(self, u: Optional[np.ndarray] = None) -> StateEstimate:
        """UKF prediction step."""
        mean = self.current_estimate.mean
        cov = self.current_estimate.covariance
        n = len(mean)
        
        # Generate sigma points
        sigma_points, Wm, Wc = self._compute_sigma_points(mean, cov)
        
        # Propagate sigma points through transition function
        sigma_pred = np.array([self.f(sp, u) for sp in sigma_points])
        
        # Predicted mean
        x_pred = np.sum(Wm[:, np.newaxis] * sigma_pred, axis=0)
        
        # Predicted covariance
        P_pred = self.Q.copy()
        for i in range(len(sigma_pred)):
            diff = sigma_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)
        
        return StateEstimate(mean=x_pred, covariance=P_pred)
    
    def update(self, y: np.ndarray, predicted: StateEstimate) -> StateEstimate:
        """UKF update step."""
        x_pred = predicted.mean
        P_pred = predicted.covariance
        n = len(x_pred)
        
        # Generate sigma points from predicted distribution
        sigma_points, Wm, Wc = self._compute_sigma_points(x_pred, P_pred)
        
        # Propagate through observation function
        sigma_obs = np.array([self.h(sp) for sp in sigma_points])
        
        # Predicted observation mean
        y_pred = np.sum(Wm[:, np.newaxis] * sigma_obs, axis=0)
        
        # Innovation covariance
        S = self.R.copy()
        for i in range(len(sigma_obs)):
            diff = sigma_obs[i] - y_pred
            S += Wc[i] * np.outer(diff, diff)
        
        # Cross covariance
        Pxy = np.zeros((n, len(y_pred)))
        for i in range(len(sigma_points)):
            diff_x = sigma_points[i] - x_pred
            diff_y = sigma_obs[i] - y_pred
            Pxy += Wc[i] * np.outer(diff_x, diff_y)
        
        # Kalman gain
        K = Pxy @ np.linalg.inv(S + EPSILON * np.eye(len(S)))
        
        # Update
        x_upd = x_pred + K @ (y - y_pred)
        P_upd = P_pred - K @ S @ K.T
        
        self.current_estimate = StateEstimate(mean=x_upd, covariance=P_upd)
        return self.current_estimate


class DrainageStateSpaceModel:
    """
    State-space model for drainage system stress dynamics.
    
    State vector: [stress, d(stress)/dt, latent_factor_1, latent_factor_2]
    
    Observations: noisy stress measurements
    """
    
    def __init__(self, n_locations: int, dt: float = 1.0):
        self.n_locations = n_locations
        self.dt = dt
        
        # State dimension: stress + velocity + 2 latent factors per location
        self.state_dim = 4 * n_locations
        self.obs_dim = n_locations
        
        # Build state-space matrices
        self._build_model()
    
    def _build_model(self):
        """Construct state-space matrices."""
        n = self.n_locations
        dt = self.dt
        
        # Single-location block
        F_block = np.array([
            [1, dt, 0, 0],      # stress dynamics
            [0, 0.9, 0.1, 0],   # velocity (damped)
            [0, 0, 0.95, 0],    # latent factor 1 (slow)
            [0, 0, 0, 0.8]      # latent factor 2 (fast)
        ])
        
        # Full transition matrix (block diagonal)
        self.F = np.kron(np.eye(n), F_block)
        
        # Observation: we observe stress at each location
        H_block = np.array([[1, 0, 0, 0]])
        self.H = np.kron(np.eye(n), H_block)
        
        # Process noise
        q = 0.01
        Q_block = q * np.array([
            [dt**4/4, dt**3/2, 0, 0],
            [dt**3/2, dt**2, 0, 0],
            [0, 0, dt, 0],
            [0, 0, 0, dt]
        ])
        self.Q = np.kron(np.eye(n), Q_block)
        
        # Observation noise
        r = 0.1
        self.R = r * np.eye(n)
    
    def get_linear_model(self) -> LinearStateSpaceModel:
        """Return as LinearStateSpaceModel."""
        return LinearStateSpaceModel(
            F=self.F,
            H=self.H,
            Q=self.Q,
            R=self.R
        )
    
    def filter_observations(self, observations: np.ndarray) -> List[StateEstimate]:
        """
        Filter sequence of observations.
        
        Args:
            observations: Array of shape (T, n_locations)
        
        Returns:
            List of state estimates
        """
        model = self.get_linear_model()
        kf = KalmanFilter(model)
        
        # Initialize
        x0 = np.zeros(self.state_dim)
        x0[::4] = observations[0]  # Initialize stress to first observation
        P0 = np.eye(self.state_dim) * 0.1
        kf.initialize(x0, P0)
        
        # Filter
        return kf.filter(observations)
    
    def smooth_observations(self, observations: np.ndarray) -> List[StateEstimate]:
        """Run filter + RTS smoother."""
        model = self.get_linear_model()
        kf = KalmanFilter(model)
        
        # Initialize
        x0 = np.zeros(self.state_dim)
        x0[::4] = observations[0]
        P0 = np.eye(self.state_dim) * 0.1
        kf.initialize(x0, P0)
        
        # Filter
        kf.filter(observations)
        
        # Smooth
        return kf.smooth()
    
    def extract_stress(self, estimate: StateEstimate) -> Tuple[np.ndarray, np.ndarray]:
        """Extract stress values and uncertainties from state estimate."""
        stress = estimate.mean[::4]
        stress_var = np.diag(estimate.covariance)[::4]
        return stress, np.sqrt(stress_var)
