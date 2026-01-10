"""Residual inference integration for ST-GNN.

CRITICAL RULES (from projectfile.md Step 7):
- DL outputs ONLY residuals
- Final stress = base + residual
- Uncertainty = Bayesian + DL proxy
- DL CANNOT override NO_DECISION zones

Reference: projectfile.md Step 7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntegratedResult:
    """Result of integrating DL residual with Bayesian baseline."""
    final_stress: np.ndarray  # base + residual
    final_uncertainty: np.ndarray  # combined uncertainty
    dl_residual: np.ndarray  # raw DL output
    dl_uncertainty: np.ndarray  # DL uncertainty estimate
    bayesian_stress: np.ndarray  # original baseline
    bayesian_variance: np.ndarray  # original variance
    no_decision_mask: np.ndarray  # preserved from Bayesian
    reliability_mask: np.ndarray  # combined reliability
    
    @property
    def correction_magnitude(self) -> float:
        """Mean absolute correction applied."""
        return float(np.nanmean(np.abs(self.dl_residual)))
    
    @property
    def uncertainty_change(self) -> float:
        """Average change in uncertainty."""
        bayesian_std = np.sqrt(np.maximum(self.bayesian_variance, 1e-8))
        change = self.final_uncertainty - bayesian_std
        return float(np.nanmean(change))


def integrate_dl_residual(
    bayesian_stress: np.ndarray,
    bayesian_variance: np.ndarray,
    dl_residual: np.ndarray,
    dl_uncertainty: Optional[np.ndarray] = None,
    no_decision_mask: Optional[np.ndarray] = None,
    reliability_mask: Optional[np.ndarray] = None,
    max_residual: float = 1.0,
    uncertainty_blend: float = 0.3,
) -> IntegratedResult:
    """Integrate DL residual predictions with Bayesian baseline.
    
    CRITICAL: This function enforces the rule that DL CANNOT override
    the Bayesian system, only CORRECT it.
    
    Args:
        bayesian_stress: (T, N) or (T, H, W) Bayesian stress prediction
        bayesian_variance: same shape, Bayesian variance
        dl_residual: same shape, predicted residual correction
        dl_uncertainty: same shape, DL uncertainty estimate (optional)
        no_decision_mask: boolean mask for NO_DECISION zones (PRESERVED)
        reliability_mask: boolean mask for reliable predictions
        max_residual: maximum allowed residual correction
        uncertainty_blend: weight for DL uncertainty (0-1)
        
    Returns:
        IntegratedResult with combined predictions
    """
    # Validate shapes
    if bayesian_stress.shape != dl_residual.shape:
        raise ValueError(
            f"Shape mismatch: bayesian_stress {bayesian_stress.shape} vs "
            f"dl_residual {dl_residual.shape}"
        )
    
    # 1. Clip residuals to prevent extreme corrections
    dl_residual_clipped = np.clip(dl_residual, -max_residual, max_residual)
    
    if np.abs(dl_residual - dl_residual_clipped).max() > 0:
        logger.warning(
            f"Clipped {np.sum(np.abs(dl_residual) > max_residual)} residuals "
            f"exceeding ±{max_residual}"
        )
    
    # 2. Compute final stress: base + residual
    final_stress = bayesian_stress + dl_residual_clipped
    
    # 3. Combine uncertainty
    bayesian_std = np.sqrt(np.maximum(bayesian_variance, 1e-8))
    
    if dl_uncertainty is not None:
        # Blend Bayesian and DL uncertainty
        # Higher uncertainty = more conservative
        dl_unc = np.maximum(dl_uncertainty, 1e-8)
        final_uncertainty = (
            (1 - uncertainty_blend) * bayesian_std + 
            uncertainty_blend * dl_unc
        )
    else:
        final_uncertainty = bayesian_std
    
    # 4. CRITICAL: Preserve NO_DECISION zones
    # DL CANNOT change decisions in high-uncertainty regions
    if no_decision_mask is not None:
        # In NO_DECISION zones, revert to Bayesian prediction
        final_stress = np.where(no_decision_mask, bayesian_stress, final_stress)
        
        # Also inflate uncertainty in NO_DECISION zones
        final_uncertainty = np.where(
            no_decision_mask, 
            np.maximum(final_uncertainty, bayesian_std * 1.5),
            final_uncertainty
        )
        
        n_preserved = no_decision_mask.sum()
        logger.info(f"Preserved {n_preserved} NO_DECISION zone cells from DL override")
    
    # 5. Update reliability mask
    if reliability_mask is None:
        reliability_mask = np.ones_like(bayesian_stress, dtype=bool)
    
    # Areas with very large DL corrections are less reliable
    large_correction = np.abs(dl_residual_clipped) > max_residual * 0.8
    combined_reliability = reliability_mask & ~large_correction
    
    return IntegratedResult(
        final_stress=final_stress.astype(np.float32),
        final_uncertainty=final_uncertainty.astype(np.float32),
        dl_residual=dl_residual_clipped.astype(np.float32),
        dl_uncertainty=dl_uncertainty.astype(np.float32) if dl_uncertainty is not None else np.zeros_like(final_stress),
        bayesian_stress=bayesian_stress.astype(np.float32),
        bayesian_variance=bayesian_variance.astype(np.float32),
        no_decision_mask=no_decision_mask if no_decision_mask is not None else np.zeros_like(final_stress, dtype=bool),
        reliability_mask=combined_reliability.astype(bool),
    )


class ResidualInferenceEngine:
    """Engine for running ST-GNN residual inference.
    
    Handles:
    - Model loading and prediction
    - Integration with Bayesian baseline
    - Uncertainty combination
    - NO_DECISION zone preservation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        max_residual: float = 1.0,
        uncertainty_blend: float = 0.3,
    ):
        self.model_path = model_path
        self.device = device
        self.max_residual = max_residual
        self.uncertainty_blend = uncertainty_blend
        
        self.model = None
        self._is_loaded = False
    
    def load_model(self, model_path: Optional[str] = None):
        """Load trained ST-GNN model."""
        path = model_path or self.model_path
        if path is None:
            raise ValueError("No model path provided")
        
        try:
            import torch
            from ..models import SpatioTemporalGNN
            
            checkpoint = torch.load(path, map_location=self.device)
            
            # Get config from checkpoint or use defaults
            config = checkpoint.get('config', None)
            self.model = SpatioTemporalGNN(config=config)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device)
            self.model.eval()
            
            self._is_loaded = True
            logger.info(f"Loaded ST-GNN model from {path}")
            
        except ImportError:
            raise ImportError("PyTorch required for model loading")
    
    def infer(
        self,
        node_features: np.ndarray,  # (T, N, F)
        edge_index: np.ndarray,  # (2, E)
        edge_weights: np.ndarray,  # (E,)
        bayesian_stress: np.ndarray,  # (T, N)
        bayesian_variance: np.ndarray,  # (T, N)
        no_decision_mask: Optional[np.ndarray] = None,
        reliability_mask: Optional[np.ndarray] = None,
    ) -> IntegratedResult:
        """Run full inference pipeline.
        
        Args:
            node_features: temporal node features for ST-GNN
            edge_index: graph edges
            edge_weights: edge weights
            bayesian_stress: baseline Bayesian predictions
            bayesian_variance: Bayesian uncertainty
            no_decision_mask: NO_DECISION zones to preserve
            reliability_mask: reliability mask
            
        Returns:
            IntegratedResult with final predictions
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        edges = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        weights = torch.tensor(edge_weights, dtype=torch.float32, device=self.device)
        
        # Run model
        with torch.no_grad():
            dl_residual, dl_uncertainty = self.model(x, edges, weights)
        
        # Convert back to numpy
        dl_residual = dl_residual.squeeze(-1).cpu().numpy()
        if dl_uncertainty is not None:
            dl_uncertainty = dl_uncertainty.squeeze(-1).cpu().numpy()
        else:
            dl_uncertainty = None
        
        # Integrate with Bayesian baseline
        return integrate_dl_residual(
            bayesian_stress=bayesian_stress,
            bayesian_variance=bayesian_variance,
            dl_residual=dl_residual,
            dl_uncertainty=dl_uncertainty,
            no_decision_mask=no_decision_mask,
            reliability_mask=reliability_mask,
            max_residual=self.max_residual,
            uncertainty_blend=self.uncertainty_blend,
        )
    
    def infer_residual_only(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run model inference only, without integration.
        
        Returns raw residual and uncertainty predictions.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        import torch
        
        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        edges = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        weights = torch.tensor(edge_weights, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            residual, uncertainty = self.model(x, edges, weights)
        
        residual = residual.squeeze(-1).cpu().numpy()
        if uncertainty is not None:
            uncertainty = uncertainty.squeeze(-1).cpu().numpy()
        
        return residual, uncertainty
