"""
STEP 4: Uncertainty Inflation Under Distribution Shift

Formula:
    σ_final² = σ_bayes² + σ_dl² + λ·σ_shift²

Where:
    σ_bayes = Bayesian posterior uncertainty
    σ_dl = Deep learning (or model) uncertainty
    σ_shift = Distribution shift penalty
    λ = Shift score from ShiftDetector
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class UncertaintyResult:
    """Result of uncertainty inflation."""
    sigma_final: np.ndarray  # Final inflated uncertainty
    sigma_bayes: np.ndarray  # Bayesian component
    sigma_dl: np.ndarray     # Model component
    sigma_shift: np.ndarray  # Shift component
    lambda_shift: float      # Shift score [0, 1]
    inflation_factor: np.ndarray  # How much we inflated
    
    def to_dict(self) -> Dict:
        return {
            'mean_sigma_final': float(np.mean(self.sigma_final)),
            'mean_sigma_bayes': float(np.mean(self.sigma_bayes)),
            'mean_sigma_dl': float(np.mean(self.sigma_dl)),
            'mean_sigma_shift': float(np.mean(self.sigma_shift)),
            'lambda_shift': self.lambda_shift,
            'mean_inflation': float(np.mean(self.inflation_factor)),
        }


class UncertaintyInflator:
    """
    Uncertainty Inflation Under Distribution Shift
    
    Restores NO_DECISION zones by inflating uncertainty when
    the model is operating in unfamiliar territory.
    
    Formula:
        σ_final² = σ_bayes² + σ_dl² + λ·σ_shift²
    
    Where:
        λ = shift_score from ShiftDetector
        σ_shift = base shift uncertainty (proportional to value)
    
    Critical: Uncertainty MUST GROW when shift_score > 0
    
    Usage:
        inflator = UncertaintyInflator()
        result = inflator.inflate(
            Z=latent_stress,
            sigma_bayes=...,
            shift_score=0.7
        )
    """
    
    def __init__(self, 
                 base_shift_std: float = 0.5,
                 min_uncertainty: float = 0.05,
                 max_inflation: float = 5.0):
        """
        Args:
            base_shift_std: Base std for shift uncertainty
            min_uncertainty: Minimum allowed uncertainty
            max_inflation: Maximum inflation factor
        """
        self.base_shift_std = base_shift_std
        self.min_uncertainty = min_uncertainty
        self.max_inflation = max_inflation
    
    def inflate(self,
                Z: np.ndarray,
                sigma_bayes: Optional[np.ndarray] = None,
                sigma_dl: Optional[np.ndarray] = None,
                shift_score: float = 0.0) -> UncertaintyResult:
        """
        Inflate uncertainty based on shift score.
        
        Args:
            Z: Latent stress values (normalized)
            sigma_bayes: Bayesian uncertainty (epistemic)
            sigma_dl: Model uncertainty (aleatoric)
            shift_score: λ ∈ [0, 1] from ShiftDetector
        
        Returns:
            UncertaintyResult with inflated uncertainties
        """
        # Ensure shift_score is valid
        shift_score = float(np.clip(shift_score, 0, 1))
        
        # Default uncertainty if not provided
        if sigma_bayes is None:
            sigma_bayes = self._estimate_bayes_uncertainty(Z)
        
        if sigma_dl is None:
            sigma_dl = self._estimate_dl_uncertainty(Z)
        
        # Compute shift uncertainty
        sigma_shift = self._compute_shift_uncertainty(Z, shift_score)
        
        # Main formula: σ_final² = σ_bayes² + σ_dl² + λ·σ_shift²
        variance_final = (
            sigma_bayes ** 2 + 
            sigma_dl ** 2 + 
            shift_score * sigma_shift ** 2
        )
        sigma_final = np.sqrt(variance_final)
        
        # Apply minimum
        sigma_final = np.maximum(sigma_final, self.min_uncertainty)
        
        # Compute inflation factor
        original_sigma = np.sqrt(sigma_bayes ** 2 + sigma_dl ** 2)
        inflation_factor = sigma_final / (original_sigma + 1e-10)
        inflation_factor = np.clip(inflation_factor, 1.0, self.max_inflation)
        
        return UncertaintyResult(
            sigma_final=sigma_final,
            sigma_bayes=sigma_bayes,
            sigma_dl=sigma_dl,
            sigma_shift=sigma_shift,
            lambda_shift=shift_score,
            inflation_factor=inflation_factor,
        )
    
    def _estimate_bayes_uncertainty(self, Z: np.ndarray) -> np.ndarray:
        """
        Estimate Bayesian uncertainty from latent values.
        
        Higher uncertainty for extreme values.
        """
        # Base uncertainty proportional to distance from mean
        z_abs = np.abs(Z)
        base_unc = 0.1 + 0.1 * z_abs
        
        # Add epistemic component based on value density
        # More uncertainty where values are rare
        return base_unc
    
    def _estimate_dl_uncertainty(self, Z: np.ndarray) -> np.ndarray:
        """
        Estimate model (aleatoric) uncertainty.
        
        Based on local variability.
        """
        # Fixed component + value-proportional
        sigma_dl = 0.1 * np.ones_like(Z)
        
        # Add variability component if spatial
        if Z.ndim >= 2:
            from scipy.ndimage import uniform_filter
            # Use uniform_filter for local mean, then compute local variance
            local_mean = uniform_filter(Z.astype(float), size=3, mode='nearest')
            local_mean_sq = uniform_filter((Z.astype(float))**2, size=3, mode='nearest')
            local_var = local_mean_sq - local_mean**2
            local_std = np.sqrt(np.maximum(local_var, 0))
            sigma_dl = sigma_dl + 0.2 * local_std
        
        return sigma_dl
    
    def _compute_shift_uncertainty(self, Z: np.ndarray, 
                                   shift_score: float) -> np.ndarray:
        """
        Compute shift-induced uncertainty.
        
        Higher shift score → larger penalty
        """
        # Base shift uncertainty
        sigma_shift_base = self.base_shift_std * np.ones_like(Z)
        
        # Proportional to value magnitude (extreme values more uncertain)
        value_component = 0.1 * np.abs(Z)
        
        # Combined shift uncertainty
        sigma_shift = sigma_shift_base + value_component
        
        return sigma_shift
    
    def compute_no_decision_mask(self, 
                                  sigma_final: np.ndarray,
                                  threshold: float = 0.5) -> np.ndarray:
        """
        Identify NO_DECISION zones (uncertainty too high).
        
        Args:
            sigma_final: Inflated uncertainty
            threshold: Uncertainty threshold for NO_DECISION
        
        Returns:
            Boolean mask (True = NO_DECISION)
        """
        return sigma_final > threshold


def compute_full_uncertainty(Z: np.ndarray,
                             shift_score: float,
                             sigma_bayes: Optional[np.ndarray] = None,
                             sigma_dl: Optional[np.ndarray] = None) -> UncertaintyResult:
    """
    Convenience function to compute full uncertainty inflation.
    
    Args:
        Z: Latent stress values
        shift_score: Distribution shift score [0, 1]
        sigma_bayes: Optional Bayesian uncertainty
        sigma_dl: Optional model uncertainty
    
    Returns:
        UncertaintyResult
    """
    inflator = UncertaintyInflator()
    return inflator.inflate(Z, sigma_bayes, sigma_dl, shift_score)


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate test latent values
    Z = np.random.randn(5, 50, 50) * 1.5
    
    inflator = UncertaintyInflator()
    
    print("Uncertainty Inflation Test")
    print("=" * 50)
    
    for shift_score in [0.0, 0.3, 0.6, 0.9]:
        result = inflator.inflate(Z, shift_score=shift_score)
        no_decision = inflator.compute_no_decision_mask(result.sigma_final)
        
        print(f"\nShift score: {shift_score}")
        print(f"  Mean σ_bayes: {np.mean(result.sigma_bayes):.4f}")
        print(f"  Mean σ_dl: {np.mean(result.sigma_dl):.4f}")
        print(f"  Mean σ_shift: {np.mean(result.sigma_shift):.4f}")
        print(f"  Mean σ_final: {np.mean(result.sigma_final):.4f}")
        print(f"  Mean inflation: {np.mean(result.inflation_factor):.2f}x")
        print(f"  NO_DECISION %: {100 * np.mean(no_decision):.1f}%")
