"""
Latent Space Loss Functions (LOCKED).

Scale-invariant losses for training the ST-GNN in latent Z-space.

Loss components:
- α · MSE(ΔZ): Primary residual loss
- β · NLL_latent: Negative log-likelihood with uncertainty
- γ · GraphSmoothness: Spatial coherence
- η · RankPreservation: Maintain relative ordering

Author: Urban Drainage AI Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (FIXED WEIGHTS - NO PER-CITY TUNING)
# ============================================================================

@dataclass
class LatentLossConfig:
    """
    Locked loss configuration.
    
    ⚠️ WARNING: These weights are FIXED after training.
    No per-city tuning allowed.
    """
    # Loss weights (LOCKED)
    alpha_mse: float = 10.0         # MSE weight (BOOSTED for magnitude learning)
    beta_nll: float = 0.2           # NLL weight (REDUCED to avoid zero-collapse)
    gamma_smooth: float = 0.1       # Graph smoothness weight
    eta_rank: float = 0.2           # Rank preservation weight
    
    # Uncertainty calibration
    min_uncertainty: float = 0.01   # Minimum predicted uncertainty
    max_uncertainty: float = 5.0    # Maximum predicted uncertainty
    
    # Regularization
    l2_weight: float = 1e-5         # L2 regularization


# ============================================================================
# INDIVIDUAL LOSS COMPONENTS
# ============================================================================

class LatentMSELoss(nn.Module):
    """
    MSE loss in latent Z-space.
    
    L_mse = ||ΔZ_pred - ΔZ_target||^2
    
    This is scale-invariant because both are in Z-space.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        delta_z_pred: torch.Tensor,
        delta_z_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            delta_z_pred: Predicted residual [*, 1]
            delta_z_target: Target residual [*, 1]
            mask: Optional validity mask [*]
        """
        diff = (delta_z_pred - delta_z_target) ** 2
        
        if mask is not None:
            diff = diff * mask.unsqueeze(-1)
            return diff.sum() / (mask.sum() + 1e-8)
        
        return diff.mean()


class LatentNLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss with predicted uncertainty.
    
    L_nll = 0.5 * (log(σ²) + (ΔZ_pred - ΔZ_target)² / σ²)
    
    This encourages calibrated uncertainty predictions.
    """
    
    def __init__(self, config: LatentLossConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        delta_z_pred: torch.Tensor,
        delta_z_target: torch.Tensor,
        sigma_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            delta_z_pred: Predicted residual [*, 1]
            delta_z_target: Target residual [*, 1]
            sigma_pred: Predicted uncertainty [*, 1]
            mask: Optional validity mask [*]
        """
        # Clamp uncertainty for numerical stability
        sigma = torch.clamp(
            sigma_pred,
            self.config.min_uncertainty,
            self.config.max_uncertainty
        )
        
        # NLL
        var = sigma ** 2
        nll = 0.5 * (torch.log(var) + (delta_z_pred - delta_z_target) ** 2 / var)
        
        if mask is not None:
            nll = nll * mask.unsqueeze(-1)
            return nll.sum() / (mask.sum() + 1e-8)
        
        return nll.mean()


class GraphSmoothnessLoss(nn.Module):
    """
    Graph Laplacian regularization for spatial coherence.
    
    L_smooth = Σ_{(i,j) ∈ E} w_ij * (ΔZ_i - ΔZ_j)²
    
    Penalizes spatial incoherence in predictions.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        delta_z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            delta_z: Predicted residual [T, N, 1] or [N, 1]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights [E]
        """
        row, col = edge_index
        
        # Handle temporal dimension
        if delta_z.dim() == 3:
            # [T, N, 1] -> average over time
            diff = delta_z[:, row, :] - delta_z[:, col, :]  # [T, E, 1]
            diff_sq = (diff ** 2).mean(dim=0)  # [E, 1]
        else:
            diff = delta_z[row] - delta_z[col]  # [E, 1]
            diff_sq = diff ** 2
        
        if edge_weight is not None:
            diff_sq = diff_sq.squeeze(-1) * edge_weight
        
        return diff_sq.mean()


class RankPreservationLoss(nn.Module):
    """
    Rank preservation loss to maintain relative ordering.
    
    Penalizes rank inversions between predictions and targets.
    
    For pairs (i, j) where target_i > target_j,
    we want pred_i > pred_j as well.
    """
    
    def __init__(self, margin: float = 0.1, n_samples: int = 1000):
        super().__init__()
        self.margin = margin
        self.n_samples = n_samples
        
    def forward(
        self,
        delta_z_pred: torch.Tensor,
        delta_z_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            delta_z_pred: Predicted residual [N, 1] or flattened
            delta_z_target: Target residual [N, 1] or flattened
        """
        pred = delta_z_pred.flatten()
        target = delta_z_target.flatten()
        
        n = pred.size(0)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Sample pairs for efficiency
        n_pairs = min(self.n_samples, n * (n - 1) // 2)
        idx_i = torch.randint(0, n, (n_pairs,), device=pred.device)
        idx_j = torch.randint(0, n, (n_pairs,), device=pred.device)
        
        # Target ordering
        target_diff = target[idx_i] - target[idx_j]
        
        # Predicted ordering
        pred_diff = pred[idx_i] - pred[idx_j]
        
        # Hinge loss for rank preservation
        # If target_i > target_j (target_diff > 0), we want pred_diff > 0
        # Loss = max(0, margin - sign(target_diff) * pred_diff)
        sign = torch.sign(target_diff)
        violations = F.relu(self.margin - sign * pred_diff)
        
        return violations.mean()


# ============================================================================
# COMPOSITE LOSS
# ============================================================================

class LatentCompositeLoss(nn.Module):
    """
    Composite loss for latent space training.
    
    L = α·MSE(ΔZ) + β·NLL(ΔZ, σ) + γ·Smooth(ΔZ) + η·Rank(ΔZ)
    
    All weights are FIXED (no per-city tuning).
    """
    
    def __init__(self, config: Optional[LatentLossConfig] = None):
        super().__init__()
        self.config = config or LatentLossConfig()
        
        self.mse_loss = LatentMSELoss()
        self.nll_loss = LatentNLLLoss(self.config)
        self.smooth_loss = GraphSmoothnessLoss()
        self.rank_loss = RankPreservationLoss()
        
        logger.info(
            f"LatentCompositeLoss initialized with FIXED weights: "
            f"α={self.config.alpha_mse}, β={self.config.beta_nll}, "
            f"γ={self.config.gamma_smooth}, η={self.config.eta_rank}"
        )
        
    def forward(
        self,
        delta_z_pred: torch.Tensor,
        delta_z_target: torch.Tensor,
        sigma_pred: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.
        
        Returns:
            Dictionary with 'total' loss and individual components
        """
        # MSE
        l_mse = self.mse_loss(delta_z_pred, delta_z_target, mask)
        
        # NLL
        l_nll = self.nll_loss(delta_z_pred, delta_z_target, sigma_pred, mask)
        
        # Smoothness
        l_smooth = self.smooth_loss(delta_z_pred, edge_index, edge_weight)
        
        # Rank preservation
        l_rank = self.rank_loss(delta_z_pred, delta_z_target)
        
        # Weighted sum
        total = (
            self.config.alpha_mse * l_mse +
            self.config.beta_nll * l_nll +
            self.config.gamma_smooth * l_smooth +
            self.config.eta_rank * l_rank
        )
        
        return {
            'total': total,
            'mse': l_mse,
            'nll': l_nll,
            'smooth': l_smooth,
            'rank': l_rank,
        }
