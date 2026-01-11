"""
STEP 1: Scale-Free Latent Representation

Transforms raw stress into LATENT SPACE:
Z(x,t) = (S_raw(x,t) - median_city) / MAD_city

Rules:
- Model NEVER outputs absolute stress directly
- All internal logic operates on Z
- Z is dimensionless and transfer-safe
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class LatentTransformResult:
    """Result of latent transformation."""
    Z: np.ndarray  # Latent stress (dimensionless)
    median: float  # City median
    mad: float  # Median Absolute Deviation
    raw_min: float
    raw_max: float
    
    def to_dict(self) -> Dict:
        return {
            'median': float(self.median),
            'mad': float(self.mad),
            'raw_min': float(self.raw_min),
            'raw_max': float(self.raw_max),
            'Z_min': float(np.min(self.Z)),
            'Z_max': float(np.max(self.Z)),
            'Z_mean': float(np.mean(self.Z)),
        }


class LatentTransform:
    """
    Scale-Free Latent Transformation
    
    Converts raw stress to dimensionless latent space Z.
    This enables transfer across regions with different stress magnitudes.
    
    Mathematical Definition:
        Z(x,t) = (S_raw(x,t) - median_city) / MAD_city
    
    Where:
        median_city = median of all S_raw values
        MAD_city = median(|S_raw - median_city|)
    
    Properties:
        - Z is dimensionless
        - Z=0 means median stress
        - Z=1 means 1 MAD above median
        - Robust to outliers (uses median, not mean)
    """
    
    def __init__(self, min_mad: float = 1e-6):
        """
        Args:
            min_mad: Minimum MAD to prevent division by zero
        """
        self.min_mad = min_mad
        self._fitted = False
        self._median = None
        self._mad = None
    
    def fit(self, S_raw: np.ndarray) -> 'LatentTransform':
        """
        Compute robust statistics from raw stress.
        
        Args:
            S_raw: Raw stress array (any shape)
        
        Returns:
            self for chaining
        """
        flat = S_raw.flatten()
        
        # Robust statistics
        self._median = np.median(flat)
        deviations = np.abs(flat - self._median)
        self._mad = np.maximum(np.median(deviations), self.min_mad)
        
        self._fitted = True
        return self
    
    def transform(self, S_raw: np.ndarray) -> LatentTransformResult:
        """
        Transform raw stress to latent space Z.
        
        Args:
            S_raw: Raw stress array
        
        Returns:
            LatentTransformResult with Z and statistics
        """
        if not self._fitted:
            self.fit(S_raw)
        
        # Z = (S - median) / MAD
        Z = (S_raw - self._median) / self._mad
        
        return LatentTransformResult(
            Z=Z,
            median=self._median,
            mad=self._mad,
            raw_min=float(np.min(S_raw)),
            raw_max=float(np.max(S_raw)),
        )
    
    def fit_transform(self, S_raw: np.ndarray) -> LatentTransformResult:
        """Fit and transform in one step."""
        return self.fit(S_raw).transform(S_raw)
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Convert latent Z back to raw stress scale.
        
        Args:
            Z: Latent stress array
        
        Returns:
            Raw stress array
        """
        if not self._fitted:
            raise ValueError("Must fit before inverse transform")
        
        return Z * self._mad + self._median
    
    @property
    def params(self) -> Dict:
        """Get transformation parameters."""
        return {
            'median': self._median,
            'mad': self._mad,
            'fitted': self._fitted,
        }


def compute_latent_stress(S_raw: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to compute latent stress.
    
    Args:
        S_raw: Raw stress array
    
    Returns:
        Tuple of (Z, params_dict)
    """
    transform = LatentTransform()
    result = transform.fit_transform(S_raw)
    return result.Z, result.to_dict()


# Test
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    S_raw = np.random.exponential(scale=10, size=(10, 50, 50))
    
    transform = LatentTransform()
    result = transform.fit_transform(S_raw)
    
    print("Latent Transform Test")
    print("=" * 40)
    print(f"Raw range: [{result.raw_min:.3f}, {result.raw_max:.3f}]")
    print(f"Median: {result.median:.3f}")
    print(f"MAD: {result.mad:.3f}")
    print(f"Z range: [{np.min(result.Z):.3f}, {np.max(result.Z):.3f}]")
    print(f"Z mean: {np.mean(result.Z):.3f}")
    print(f"Z std: {np.std(result.Z):.3f}")
