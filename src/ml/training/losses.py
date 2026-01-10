"""Loss functions for ST-GNN training.

Implements composite loss:
  Loss = α * MSE(residual) 
       + β * Uncertainty calibration loss
       + γ * Smoothness regularization
       + λ * Graph consistency penalty

Reference: projectfile.md Step 4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LossConfig:
    """Configuration for composite loss."""
    # Loss weights
    alpha_residual: float = 1.0  # MSE on residual
    beta_uncertainty: float = 0.5  # Uncertainty calibration
    gamma_smoothness: float = 0.1  # Spatial smoothness
    lambda_consistency: float = 0.1  # Graph consistency
    
    # Loss parameters
    uncertainty_target_fraction: float = 0.95  # Target coverage for CI
    smoothness_kernel_size: int = 3
    huber_delta: float = 1.0  # Use Huber loss instead of MSE if > 0


if TORCH_AVAILABLE:
    
    class ResidualLoss(nn.Module):
        """MSE/Huber loss on residual predictions."""
        
        def __init__(self, huber_delta: float = 0.0):
            super().__init__()
            self.huber_delta = huber_delta
        
        def forward(
            self,
            pred_residual: torch.Tensor,
            target_residual: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Compute residual loss.
            
            Args:
                pred_residual: (...) predicted residuals
                target_residual: (...) target residuals
                mask: (...) optional validity mask
                
            Returns:
                loss: scalar loss value
            """
            if mask is not None:
                pred_residual = pred_residual[mask]
                target_residual = target_residual[mask]
            
            if len(pred_residual) == 0:
                return torch.tensor(0.0, device=pred_residual.device)
            
            if self.huber_delta > 0:
                loss = F.huber_loss(
                    pred_residual, 
                    target_residual, 
                    reduction='mean',
                    delta=self.huber_delta,
                )
            else:
                loss = F.mse_loss(pred_residual, target_residual, reduction='mean')
            
            return loss
    
    
    class UncertaintyCalibrationLoss(nn.Module):
        """Loss for uncertainty calibration.
        
        Penalizes confident wrong predictions and uncertain correct predictions.
        Uses negative log-likelihood under Gaussian assumption.
        """
        
        def __init__(self, target_fraction: float = 0.95, min_var: float = 1e-6):
            super().__init__()
            self.target_fraction = target_fraction
            self.min_var = min_var
        
        def forward(
            self,
            pred_residual: torch.Tensor,
            pred_uncertainty: torch.Tensor,
            target_residual: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Compute uncertainty calibration loss.
            
            Penalizes:
            - High uncertainty when prediction is accurate
            - Low uncertainty when prediction is inaccurate
            
            Args:
                pred_residual: predicted residuals
                pred_uncertainty: predicted uncertainty (std dev)
                target_residual: target residuals
                mask: validity mask
            """
            if mask is not None:
                pred_residual = pred_residual[mask]
                pred_uncertainty = pred_uncertainty[mask]
                target_residual = target_residual[mask]
            
            if len(pred_residual) == 0:
                return torch.tensor(0.0, device=pred_residual.device)
            
            # Compute squared error
            sq_error = (pred_residual - target_residual) ** 2
            
            # Variance (uncertainty squared)
            variance = pred_uncertainty ** 2 + self.min_var
            
            # Negative log-likelihood (Gaussian)
            nll = 0.5 * (torch.log(variance) + sq_error / variance)
            
            return nll.mean()
    
    
    class SmoothnessLoss(nn.Module):
        """Spatial smoothness regularization.
        
        Encourages smooth residual predictions by penalizing
        large differences between neighboring nodes.
        """
        
        def __init__(self):
            super().__init__()
        
        def forward(
            self,
            pred_residual: torch.Tensor,  # (N,) or (T, N)
            edge_index: torch.Tensor,  # (2, E)
            edge_weight: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Compute smoothness loss.
            
            Penalizes differences between connected nodes.
            """
            src = edge_index[0]
            dst = edge_index[1]
            
            # Handle temporal dimension
            if pred_residual.dim() > 1:
                # Average over time, compute spatial smoothness
                pred_flat = pred_residual.mean(dim=0)
            else:
                pred_flat = pred_residual.squeeze()
            
            # Difference between connected nodes
            diff = pred_flat[src] - pred_flat[dst]
            sq_diff = diff ** 2
            
            if edge_weight is not None:
                # Weight by edge strength
                sq_diff = sq_diff * edge_weight
            
            return sq_diff.mean()
    
    
    class GraphConsistencyLoss(nn.Module):
        """Graph consistency penalty.
        
        Ensures that the learned residuals are consistent with
        the graph structure (e.g., upstream stress should influence downstream).
        """
        
        def __init__(self, flow_weight: float = 0.5):
            super().__init__()
            self.flow_weight = flow_weight
        
        def forward(
            self,
            pred_residual: torch.Tensor,  # (N,) or (T, N)
            edge_index: torch.Tensor,  # (2, E)
            edge_weight: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Compute graph consistency loss.
            
            Penalizes large positive residuals at downstream nodes
            when upstream nodes have negative residuals (inconsistent flow).
            """
            src = edge_index[0]
            dst = edge_index[1]
            
            if pred_residual.dim() > 1:
                pred_flat = pred_residual.mean(dim=0)
            else:
                pred_flat = pred_residual.squeeze()
            
            src_residual = pred_flat[src]
            dst_residual = pred_flat[dst]
            
            # Penalize when source is negative but dest is highly positive
            # (stress shouldn't appear downstream without upstream source)
            inconsistency = F.relu(-src_residual) * F.relu(dst_residual)
            
            if edge_weight is not None:
                inconsistency = inconsistency * edge_weight
            
            return inconsistency.mean()
    
    
    class CompositeLoss(nn.Module):
        """Complete composite loss for ST-GNN training.
        
        Combines:
        - Residual MSE/Huber loss
        - Uncertainty calibration loss
        - Spatial smoothness
        - Graph consistency
        """
        
        def __init__(self, config: Optional[LossConfig] = None):
            super().__init__()
            self.config = config or LossConfig()
            
            self.residual_loss = ResidualLoss(huber_delta=self.config.huber_delta)
            self.uncertainty_loss = UncertaintyCalibrationLoss(
                target_fraction=self.config.uncertainty_target_fraction
            )
            self.smoothness_loss = SmoothnessLoss()
            self.consistency_loss = GraphConsistencyLoss()
            
            logger.info(
                f"CompositeLoss: α={self.config.alpha_residual}, "
                f"β={self.config.beta_uncertainty}, "
                f"γ={self.config.gamma_smoothness}, "
                f"λ={self.config.lambda_consistency}"
            )
        
        def forward(
            self,
            pred_residual: torch.Tensor,
            pred_uncertainty: Optional[torch.Tensor],
            target_residual: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """Compute composite loss.
            
            Args:
                pred_residual: predicted residuals
                pred_uncertainty: predicted uncertainty (optional)
                target_residual: target residuals
                edge_index: graph edges
                edge_weight: edge weights
                mask: validity mask
                
            Returns:
                total_loss: combined loss
                loss_dict: breakdown of individual losses
            """
            loss_dict = {}
            total_loss = torch.tensor(0.0, device=pred_residual.device)
            
            # 1. Residual loss
            l_res = self.residual_loss(pred_residual, target_residual, mask)
            loss_dict['residual'] = l_res.item()
            total_loss = total_loss + self.config.alpha_residual * l_res
            
            # 2. Uncertainty calibration loss
            if pred_uncertainty is not None and self.config.beta_uncertainty > 0:
                l_unc = self.uncertainty_loss(
                    pred_residual, pred_uncertainty, target_residual, mask
                )
                loss_dict['uncertainty'] = l_unc.item()
                total_loss = total_loss + self.config.beta_uncertainty * l_unc
            
            # 3. Smoothness loss
            if self.config.gamma_smoothness > 0:
                l_smooth = self.smoothness_loss(pred_residual, edge_index, edge_weight)
                loss_dict['smoothness'] = l_smooth.item()
                total_loss = total_loss + self.config.gamma_smoothness * l_smooth
            
            # 4. Graph consistency loss
            if self.config.lambda_consistency > 0:
                l_consist = self.consistency_loss(pred_residual, edge_index, edge_weight)
                loss_dict['consistency'] = l_consist.item()
                total_loss = total_loss + self.config.lambda_consistency * l_consist
            
            loss_dict['total'] = total_loss.item()
            
            return total_loss, loss_dict


else:
    # Stubs
    class CompositeLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
