"""
STEP 2: Region-Level Calibration Layer (NO TRAINING)

Implements POST-INFERENCE calibration:
S_calibrated = f_city(Z)

Where f_city is:
- Monotonic
- Non-parametric OR Bayesian hierarchical
- Fit using ONLY distributional statistics
- NEVER trained with gradients
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import interpolate
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationResult:
    """Result of calibration."""
    S_calibrated: np.ndarray
    method: str
    params: Dict
    
    def to_dict(self) -> Dict:
        return {
            'method': self.method,
            'params': self.params,
            'calibrated_min': float(np.min(self.S_calibrated)),
            'calibrated_max': float(np.max(self.S_calibrated)),
            'calibrated_mean': float(np.mean(self.S_calibrated)),
        }


class CityCalibrator:
    """
    City-Specific Calibration Layer
    
    Transforms latent stress Z to calibrated stress S_calibrated.
    Uses non-parametric methods that require NO gradient training.
    
    Available Methods:
    1. Quantile Mapping: Maps Z quantiles to target quantiles
    2. Isotonic Regression: Monotonic fit to reference
    3. Linear Scaling: Simple affine transform
    
    Rules:
    - Calibration is city-specific
    - Calibration is transparent (all params logged)
    - NO gradient-based training
    """
    
    def __init__(self, method: str = "quantile"):
        """
        Args:
            method: Calibration method ("quantile", "isotonic", "linear")
        """
        self.method = method
        self._fitted = False
        self._params = {}
    
    def fit(self, Z: np.ndarray, 
            raw_stress_or_quantiles: Optional[Any] = None) -> 'CityCalibrator':
        """
        Fit calibration parameters.
        
        Args:
            Z: Latent stress array
            raw_stress_or_quantiles: Optional raw stress (ignored) or 
                target quantile mapping dict {0.5: 0.5, 0.9: 1.0, 0.99: 2.0}
        
        Returns:
            self for chaining
        """
        flat_Z = Z.flatten()
        
        # Handle the second argument
        target_quantiles = None
        if isinstance(raw_stress_or_quantiles, dict):
            target_quantiles = raw_stress_or_quantiles
        # If it's an array, it's raw_stress - ignore it (not needed for calibration)
        
        if self.method == "quantile":
            self._fit_quantile(flat_Z, target_quantiles)
        elif self.method == "isotonic":
            self._fit_isotonic(flat_Z, target_quantiles)
        elif self.method == "linear":
            self._fit_linear(flat_Z)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._fitted = True
        return self
    
    def _fit_quantile(self, flat_Z: np.ndarray, 
                      target_quantiles: Optional[Dict[float, float]]):
        """Fit quantile mapping calibration."""
        # Default target: map to 0-1 operational scale
        if target_quantiles is None:
            target_quantiles = {
                0.0: 0.0,
                0.25: 0.25,
                0.5: 0.5,
                0.75: 0.75,
                0.9: 1.0,
                0.95: 1.5,
                0.99: 2.0,
                1.0: 2.5,
            }
        
        # Compute Z quantiles
        probs = sorted(target_quantiles.keys())
        z_quantiles = np.percentile(flat_Z, [p * 100 for p in probs])
        s_targets = [target_quantiles[p] for p in probs]
        
        # Create interpolation function
        self._params = {
            'probs': probs,
            'z_quantiles': z_quantiles.tolist(),
            's_targets': s_targets,
        }
        
        # Interpolator for transform
        self._interpolator = interpolate.interp1d(
            z_quantiles, s_targets,
            kind='linear',
            bounds_error=False,
            fill_value=(s_targets[0], s_targets[-1])
        )
    
    def _fit_isotonic(self, flat_Z: np.ndarray,
                      target_quantiles: Optional[Dict[float, float]]):
        """Fit isotonic regression calibration."""
        # Create monotonic mapping using isotonic regression
        sorted_idx = np.argsort(flat_Z)
        sorted_Z = flat_Z[sorted_idx]
        
        # Target: normalized to 0-1 range
        target = np.linspace(0, 2, len(sorted_Z))
        
        # Fit isotonic regression
        self._iso_reg = IsotonicRegression(out_of_bounds='clip')
        self._iso_reg.fit(sorted_Z, target)
        
        self._params = {
            'z_min': float(sorted_Z[0]),
            'z_max': float(sorted_Z[-1]),
            'n_samples': len(sorted_Z),
        }
    
    def _fit_linear(self, flat_Z: np.ndarray):
        """Fit simple linear scaling."""
        z_min, z_max = np.min(flat_Z), np.max(flat_Z)
        
        # Map to 0-2 range
        self._params = {
            'z_min': float(z_min),
            'z_max': float(z_max),
            'scale': 2.0 / (z_max - z_min + 1e-6),
            'offset': -z_min,
        }
    
    def transform(self, Z: np.ndarray) -> CalibrationResult:
        """
        Apply calibration to latent stress.
        
        Args:
            Z: Latent stress array
        
        Returns:
            CalibrationResult with calibrated stress
        """
        if not self._fitted:
            raise ValueError("Must fit before transform")
        
        if self.method == "quantile":
            S_calibrated = self._interpolator(Z)
        elif self.method == "isotonic":
            shape = Z.shape
            S_calibrated = self._iso_reg.predict(Z.flatten()).reshape(shape)
        elif self.method == "linear":
            S_calibrated = (Z + self._params['offset']) * self._params['scale']
            S_calibrated = np.clip(S_calibrated, 0, 2.5)
        
        return CalibrationResult(
            S_calibrated=S_calibrated,
            method=self.method,
            params=self._params,
        )
    
    def fit_transform(self, Z: np.ndarray,
                      target_quantiles: Optional[Dict[float, float]] = None) -> CalibrationResult:
        """Fit and transform in one step."""
        return self.fit(Z, target_quantiles).transform(Z)
    
    @property
    def params(self) -> Dict:
        """Get calibration parameters (for logging)."""
        return {
            'method': self.method,
            'fitted': self._fitted,
            **self._params,
        }


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate latent stress
    Z = np.random.randn(10, 50, 50) * 2  # Z ~ N(0, 2)
    
    print("City Calibrator Test")
    print("=" * 50)
    
    for method in ["quantile", "isotonic", "linear"]:
        calibrator = CityCalibrator(method=method)
        result = calibrator.fit_transform(Z)
        
        print(f"\nMethod: {method}")
        print(f"  Z range: [{np.min(Z):.3f}, {np.max(Z):.3f}]")
        print(f"  S_calibrated range: [{np.min(result.S_calibrated):.3f}, {np.max(result.S_calibrated):.3f}]")
        print(f"  Params: {calibrator.params}")
