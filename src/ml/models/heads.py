"""Output heads for ST-GNN.

Implements:
- ResidualHead: predicts stress correction
- UncertaintyHead: learns epistemic uncertainty proxy
- DualHead: combined head for both outputs

Reference: projectfile.md Step 3
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    
    class ResidualHead(nn.Module):
        """Head for predicting residual stress correction.
        
        Output is the correction to add to Bayesian baseline:
        Final_Stress = Bayesian_Stress + ResidualHead_Output
        """
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            hidden_dim: int = 32,
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            
            # Initialize output layer to small values (residuals start near zero)
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict residual.
            
            Args:
                x: (..., input_dim) encoded features
                
            Returns:
                residual: (..., output_dim) predicted residual
            """
            return self.mlp(x)
    
    
    class UncertaintyHead(nn.Module):
        """Head for predicting epistemic uncertainty proxy.
        
        Output is a learned uncertainty estimate that can be combined
        with Bayesian variance for total uncertainty.
        
        Uses softplus to ensure positive outputs.
        """
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            hidden_dim: int = 32,
            dropout: float = 0.1,
            min_uncertainty: float = 0.01,
        ):
            super().__init__()
            self.min_uncertainty = min_uncertainty
            
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict uncertainty.
            
            Args:
                x: (..., input_dim) encoded features
                
            Returns:
                uncertainty: (..., output_dim) predicted uncertainty (always positive)
            """
            raw = self.mlp(x)
            # Softplus ensures positive output
            return F.softplus(raw) + self.min_uncertainty
    
    
    class DualHead(nn.Module):
        """Combined head for residual + uncertainty prediction.
        
        This is the main output module for ST-GNN.
        """
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            uncertainty_dim: int = 1,
            hidden_dim: int = 32,
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.residual_head = ResidualHead(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            
            self.predict_uncertainty = uncertainty_dim > 0
            if self.predict_uncertainty:
                self.uncertainty_head = UncertaintyHead(
                    input_dim=input_dim,
                    output_dim=uncertainty_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            else:
                self.uncertainty_head = None
        
        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Predict residual and uncertainty.
            
            Args:
                x: (..., input_dim) encoded features
                
            Returns:
                residual: (..., output_dim) predicted residual
                uncertainty: (..., uncertainty_dim) or None
            """
            residual = self.residual_head(x)
            
            if self.predict_uncertainty:
                uncertainty = self.uncertainty_head(x)
            else:
                uncertainty = None
            
            return residual, uncertainty
    
    
    class CalibrationHead(nn.Module):
        """Head for uncertainty calibration.
        
        Learns to adjust uncertainty estimates to be well-calibrated
        (i.e., 95% CI should contain 95% of true values).
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 16,
        ):
            super().__init__()
            
            # Learn scaling factors for uncertainty
            self.calibration = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # scale and shift
            )
            
            # Initialize to identity calibration
            nn.init.zeros_(self.calibration[0].weight)
            nn.init.zeros_(self.calibration[2].weight)
            self.calibration[2].bias.data = torch.tensor([1.0, 0.0])  # scale=1, shift=0
        
        def forward(
            self, 
            uncertainty: torch.Tensor,
            features: torch.Tensor,
        ) -> torch.Tensor:
            """Calibrate uncertainty.
            
            Args:
                uncertainty: (..., 1) raw uncertainty estimates
                features: (..., input_dim) context features
                
            Returns:
                calibrated: (..., 1) calibrated uncertainty
            """
            params = self.calibration(features)
            scale = F.softplus(params[..., :1]) + 0.1  # Positive scale
            shift = params[..., 1:]
            
            calibrated = scale * uncertainty + shift
            return F.relu(calibrated) + 0.01  # Ensure positive


else:
    # Stubs when PyTorch not available
    class ResidualHead:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class UncertaintyHead:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class DualHead:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
