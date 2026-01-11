"""
STEP 3: Distribution Shift Detection

Detects distribution shift between:
- Training distribution (Seattle)
- Inference distribution (new city)

Outputs shift score ∈ [0, 1]
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist


@dataclass
class ShiftDetectionResult:
    """Result of shift detection."""
    shift_score: float  # ∈ [0, 1]
    kl_divergence: float
    wasserstein_distance: float
    tail_divergence: float
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            'shift_score': self.shift_score,
            'kl_divergence': self.kl_divergence,
            'wasserstein_distance': self.wasserstein_distance,
            'tail_divergence': self.tail_divergence,
            **self.details,
        }


class ShiftDetector:
    """
    Distribution Shift Detection
    
    Compares inference distribution to training distribution.
    Outputs a shift score ∈ [0, 1] where:
        0 = identical distributions (no shift)
        1 = maximum shift detected
    
    Metrics computed:
    1. KL Divergence (discretized)
    2. Wasserstein Distance (1D)
    3. Tail Divergence (extreme value behavior)
    
    Usage:
        detector = ShiftDetector()
        detector.fit_reference(seattle_Z)  # Training distribution
        result = detector.detect(nyc_Z)    # Test for shift
    """
    
    def __init__(self, n_bins: int = 50, tail_quantile: float = 0.95):
        """
        Args:
            n_bins: Number of bins for histogram-based metrics
            tail_quantile: Quantile threshold for tail analysis
        """
        self.n_bins = n_bins
        self.tail_quantile = tail_quantile
        self._reference_fitted = False
        self._ref_stats = {}
    
    def fit_reference(self, Z_ref: np.ndarray) -> 'ShiftDetector':
        """
        Fit reference (training) distribution.
        
        Args:
            Z_ref: Reference latent stress (e.g., Seattle)
        
        Returns:
            self for chaining
        """
        flat = Z_ref.flatten()
        
        # Store reference statistics
        self._ref_stats = {
            'mean': np.mean(flat),
            'std': np.std(flat),
            'median': np.median(flat),
            'q05': np.percentile(flat, 5),
            'q25': np.percentile(flat, 25),
            'q75': np.percentile(flat, 75),
            'q95': np.percentile(flat, 95),
            'min': np.min(flat),
            'max': np.max(flat),
        }
        
        # Store histogram
        self._ref_hist, self._bin_edges = np.histogram(
            flat, bins=self.n_bins, density=True
        )
        self._ref_hist = self._ref_hist + 1e-10  # Avoid log(0)
        
        # Store reference samples for Wasserstein
        self._ref_samples = flat
        
        self._reference_fitted = True
        return self
    
    def detect(self, Z_new: np.ndarray) -> ShiftDetectionResult:
        """
        Detect distribution shift.
        
        Args:
            Z_new: New latent stress (inference region)
        
        Returns:
            ShiftDetectionResult with shift score and details
        """
        if not self._reference_fitted:
            raise ValueError("Must fit reference before detection")
        
        flat = Z_new.flatten()
        
        # Compute individual metrics
        kl_div = self._compute_kl_divergence(flat)
        wasserstein = self._compute_wasserstein(flat)
        tail_div = self._compute_tail_divergence(flat)
        
        # Compute new stats
        new_stats = {
            'mean': np.mean(flat),
            'std': np.std(flat),
            'median': np.median(flat),
            'q05': np.percentile(flat, 5),
            'q95': np.percentile(flat, 95),
        }
        
        # Combined shift score ∈ [0, 1]
        # Weighted combination of normalized metrics
        shift_score = self._compute_combined_score(kl_div, wasserstein, tail_div)
        
        details = {
            'ref_mean': self._ref_stats['mean'],
            'ref_std': self._ref_stats['std'],
            'new_mean': new_stats['mean'],
            'new_std': new_stats['std'],
            'mean_shift': abs(new_stats['mean'] - self._ref_stats['mean']),
            'std_ratio': new_stats['std'] / (self._ref_stats['std'] + 1e-6),
        }
        
        return ShiftDetectionResult(
            shift_score=shift_score,
            kl_divergence=kl_div,
            wasserstein_distance=wasserstein,
            tail_divergence=tail_div,
            details=details,
        )
    
    def _compute_kl_divergence(self, flat: np.ndarray) -> float:
        """Compute KL divergence (discretized)."""
        # Histogram of new distribution using same bins
        new_hist, _ = np.histogram(flat, bins=self._bin_edges, density=True)
        new_hist = new_hist + 1e-10
        
        # KL(new || ref)
        kl = np.sum(new_hist * np.log(new_hist / self._ref_hist))
        return float(np.clip(kl, 0, 100))
    
    def _compute_wasserstein(self, flat: np.ndarray) -> float:
        """Compute 1D Wasserstein distance."""
        # Subsample for efficiency
        n_samples = min(5000, len(flat), len(self._ref_samples))
        ref_sub = np.random.choice(self._ref_samples, n_samples, replace=False)
        new_sub = np.random.choice(flat, n_samples, replace=False)
        
        # Sort and compute Wasserstein
        wasserstein = stats.wasserstein_distance(ref_sub, new_sub)
        return float(wasserstein)
    
    def _compute_tail_divergence(self, flat: np.ndarray) -> float:
        """Compute tail divergence (extreme values)."""
        # Compare upper tails
        ref_tail = self._ref_stats['q95']
        new_tail = np.percentile(flat, 95)
        
        # Relative difference in tail
        tail_diff = abs(new_tail - ref_tail) / (abs(ref_tail) + 1e-6)
        
        # Also check lower tail
        ref_low = self._ref_stats['q05']
        new_low = np.percentile(flat, 5)
        low_diff = abs(new_low - ref_low) / (abs(ref_low) + 1e-6)
        
        return float(max(tail_diff, low_diff))
    
    def _compute_combined_score(self, kl: float, wasserstein: float, 
                                tail: float) -> float:
        """Combine metrics into single shift score ∈ [0, 1]."""
        # Normalize each metric to [0, 1] range
        # Using sigmoid-like transforms
        
        kl_norm = 1 - np.exp(-kl / 2)  # KL > 2 → ~0.63
        wass_norm = np.tanh(wasserstein / 2)  # Wasserstein > 2 → ~0.76
        tail_norm = np.tanh(tail)  # Tail diff > 1 → ~0.76
        
        # Weighted average
        weights = [0.3, 0.4, 0.3]  # KL, Wasserstein, Tail
        combined = (weights[0] * kl_norm + 
                   weights[1] * wass_norm + 
                   weights[2] * tail_norm)
        
        return float(np.clip(combined, 0, 1))
    
    @property
    def reference_stats(self) -> Dict:
        """Get reference distribution statistics."""
        return self._ref_stats.copy()


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    # Reference distribution (Seattle-like)
    Z_seattle = np.random.randn(10, 50, 50) * 1.5 + 0.5
    
    # Similar distribution
    Z_similar = np.random.randn(10, 50, 50) * 1.5 + 0.5
    
    # Shifted distribution (NYC-like - higher values)
    Z_shifted = np.random.randn(10, 50, 50) * 2.5 + 2.0
    
    detector = ShiftDetector()
    detector.fit_reference(Z_seattle)
    
    print("Shift Detection Test")
    print("=" * 50)
    
    result_similar = detector.detect(Z_similar)
    print(f"\nSimilar distribution:")
    print(f"  Shift score: {result_similar.shift_score:.3f}")
    print(f"  KL divergence: {result_similar.kl_divergence:.3f}")
    print(f"  Wasserstein: {result_similar.wasserstein_distance:.3f}")
    
    result_shifted = detector.detect(Z_shifted)
    print(f"\nShifted distribution:")
    print(f"  Shift score: {result_shifted.shift_score:.3f}")
    print(f"  KL divergence: {result_shifted.kl_divergence:.3f}")
    print(f"  Wasserstein: {result_shifted.wasserstein_distance:.3f}")
