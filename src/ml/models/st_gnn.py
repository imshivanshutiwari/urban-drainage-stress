"""Spatio-Temporal Graph Neural Network for residual stress prediction.

Architecture:
- Spatial: GraphSAGE / GCN layers
- Temporal: Transformer-style attention OR temporal convolution
- Dual heads: residual prediction + uncertainty proxy

Reference: projectfile.md Step 3
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed - ST-GNN will not be functional")

# Check for PyTorch Geometric
try:
    from torch_geometric.nn import SAGEConv, GCNConv, GATConv
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    if TORCH_AVAILABLE:
        logger.warning("PyTorch Geometric not installed - using fallback layers")


@dataclass
class STGNNConfig:
    """Configuration for Spatio-Temporal GNN."""
    # Architecture
    input_dim: int = 9  # Number of input node features
    hidden_dim: int = 64
    output_dim: int = 1  # Residual stress (scalar)
    num_spatial_layers: int = 3
    num_temporal_layers: int = 2
    
    # Spatial convolution
    spatial_conv_type: str = "sage"  # "sage", "gcn", or "gat"
    aggregation: str = "mean"
    
    # Temporal modeling
    temporal_type: str = "transformer"  # "transformer" or "gru"
    num_attention_heads: int = 4
    temporal_window: int = 7
    
    # Regularization
    dropout: float = 0.2
    layer_norm: bool = True
    residual_connections: bool = True
    
    # Uncertainty
    predict_uncertainty: bool = True
    uncertainty_hidden_dim: int = 32


if TORCH_AVAILABLE:
    
    class SpatialEncoder(nn.Module):
        """Spatial graph encoder using GNN layers."""
        
        def __init__(self, config: STGNNConfig):
            super().__init__()
            self.config = config
            
            self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
            
            # Build spatial convolution layers
            self.conv_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            
            for i in range(config.num_spatial_layers):
                if PYG_AVAILABLE:
                    if config.spatial_conv_type == "sage":
                        conv = SAGEConv(config.hidden_dim, config.hidden_dim, aggr=config.aggregation)
                    elif config.spatial_conv_type == "gcn":
                        conv = GCNConv(config.hidden_dim, config.hidden_dim)
                    elif config.spatial_conv_type == "gat":
                        conv = GATConv(
                            config.hidden_dim, 
                            config.hidden_dim // config.num_attention_heads,
                            heads=config.num_attention_heads,
                            concat=True,
                        )
                    else:
                        raise ValueError(f"Unknown conv type: {config.spatial_conv_type}")
                else:
                    # Fallback: simple linear layer (no graph conv)
                    conv = nn.Linear(config.hidden_dim, config.hidden_dim)
                
                self.conv_layers.append(conv)
                
                if config.layer_norm:
                    self.norms.append(nn.LayerNorm(config.hidden_dim))
                else:
                    self.norms.append(nn.Identity())
            
            self.dropout = nn.Dropout(config.dropout)
        
        def forward(
            self, 
            x: torch.Tensor,  # (N, F)
            edge_index: torch.Tensor,  # (2, E)
            edge_weight: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Encode node features with spatial convolutions."""
            
            h = self.input_proj(x)
            
            for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norms)):
                h_in = h
                
                if PYG_AVAILABLE:
                    if self.config.spatial_conv_type == "gcn" and edge_weight is not None:
                        h = conv(h, edge_index, edge_weight)
                    else:
                        h = conv(h, edge_index)
                else:
                    h = conv(h)  # Fallback linear
                
                h = norm(h)
                h = F.relu(h)
                h = self.dropout(h)
                
                # Residual connection
                if self.config.residual_connections and h.shape == h_in.shape:
                    h = h + h_in
            
            return h  # (N, hidden_dim)
    
    
    class TemporalEncoder(nn.Module):
        """Temporal encoder using Transformer or GRU."""
        
        def __init__(self, config: STGNNConfig):
            super().__init__()
            self.config = config
            
            if config.temporal_type == "transformer":
                # Positional encoding
                self.pos_encoding = PositionalEncoding(
                    config.hidden_dim, 
                    max_len=config.temporal_window * 2,
                    dropout=config.dropout,
                )
                
                # Transformer encoder layer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=config.dropout,
                    activation='gelu',
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, 
                    num_layers=config.num_temporal_layers,
                )
            
            elif config.temporal_type == "gru":
                self.gru = nn.GRU(
                    config.hidden_dim,
                    config.hidden_dim,
                    num_layers=config.num_temporal_layers,
                    batch_first=True,
                    dropout=config.dropout if config.num_temporal_layers > 1 else 0,
                    bidirectional=False,
                )
            
            else:
                raise ValueError(f"Unknown temporal type: {config.temporal_type}")
            
            self.norm = nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode temporal sequence.
            
            Args:
                x: (batch, T, N, hidden_dim) or (T, N, hidden_dim)
                
            Returns:
                h: same shape as input, temporally encoded
            """
            # Handle variable input shapes
            if x.dim() == 3:
                x = x.unsqueeze(0)  # Add batch dim
            
            B, T, N, H = x.shape
            
            # Reshape for temporal modeling: (B*N, T, H)
            x = x.permute(0, 2, 1, 3).reshape(B * N, T, H)
            
            if self.config.temporal_type == "transformer":
                x = self.pos_encoding(x)
                h = self.transformer(x)
            else:
                h, _ = self.gru(x)
            
            h = self.norm(h)
            
            # Reshape back: (B, T, N, H)
            h = h.reshape(B, N, T, H).permute(0, 2, 1, 3)
            
            return h.squeeze(0) if B == 1 else h
    
    
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for transformer."""
        
        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input."""
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
    
    
    class SpatioTemporalGNN(nn.Module):
        """Complete Spatio-Temporal GNN for residual stress prediction.
        
        Architecture:
        1. Per-timestep spatial encoding (GNN)
        2. Temporal encoding across timesteps
        3. Dual heads for residual + uncertainty
        
        Key principle: Learns RESIDUALS, not absolute predictions.
        Final_Stress = Bayesian_Stress + DL_Residual
        """
        
        def __init__(self, config: Optional[STGNNConfig] = None):
            super().__init__()
            self.config = config or STGNNConfig()
            
            # Spatial encoder (per timestep)
            self.spatial_encoder = SpatialEncoder(self.config)
            
            # Temporal encoder
            self.temporal_encoder = TemporalEncoder(self.config)
            
            # Output heads
            from .heads import DualHead
            self.head = DualHead(
                input_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                uncertainty_dim=1 if self.config.predict_uncertainty else 0,
                hidden_dim=self.config.uncertainty_hidden_dim,
            )
            
            # Weight initialization
            self.apply(self._init_weights)
            
            logger.info(
                f"SpatioTemporalGNN initialized: "
                f"spatial={self.config.spatial_conv_type}, "
                f"temporal={self.config.temporal_type}, "
                f"hidden={self.config.hidden_dim}"
            )
        
        def _init_weights(self, module):
            """Initialize weights using Xavier/Kaiming initialization."""
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        def forward(
            self,
            x: torch.Tensor,  # (T, N, F) or (B, T, N, F)
            edge_index: torch.Tensor,  # (2, E)
            edge_weight: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Forward pass.
            
            Args:
                x: (T, N, F) or (B, T, N, F) node features over time
                edge_index: (2, E) static edge connectivity
                edge_weight: (E,) optional edge weights
                
            Returns:
                residual: (T, N, 1) or (B, T, N, 1) predicted residual stress
                uncertainty: (T, N, 1) or (B, T, N, 1) predicted uncertainty (if enabled)
            """
            # Handle batch dimension
            if x.dim() == 3:
                x = x.unsqueeze(0)  # (1, T, N, F)
            B, T, N, F = x.shape
            
            # 1. Spatial encoding for each timestep
            spatial_out = []
            for t in range(T):
                x_t = x[:, t, :, :]  # (B, N, F)
                
                # Process each batch item separately for GNN
                h_list = []
                for b in range(B):
                    h = self.spatial_encoder(x_t[b], edge_index, edge_weight)
                    h_list.append(h)
                h_t = torch.stack(h_list, dim=0)  # (B, N, H)
                spatial_out.append(h_t)
            
            spatial_out = torch.stack(spatial_out, dim=1)  # (B, T, N, H)
            
            # 2. Temporal encoding
            temporal_out = self.temporal_encoder(spatial_out)  # (B, T, N, H)
            
            # 3. Output heads
            residual, uncertainty = self.head(temporal_out)  # (B, T, N, 1), (B, T, N, 1)
            
            # Remove batch dim if input was unbatched
            if B == 1:
                residual = residual.squeeze(0)
                if uncertainty is not None:
                    uncertainty = uncertainty.squeeze(0)
            
            return residual, uncertainty
        
        def predict(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            return_numpy: bool = True,
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """Inference mode prediction.
            
            Args:
                x: node features
                edge_index: edges
                edge_weight: edge weights
                return_numpy: return numpy arrays instead of tensors
                
            Returns:
                (residual, uncertainty) predictions
            """
            self.eval()
            with torch.no_grad():
                residual, uncertainty = self.forward(x, edge_index, edge_weight)
            
            if return_numpy:
                residual = residual.cpu().numpy()
                if uncertainty is not None:
                    uncertainty = uncertainty.cpu().numpy()
            
            return residual, uncertainty


else:
    # Stub class when PyTorch not available
    class SpatioTemporalGNN:
        """Stub for when PyTorch is not installed."""
        def __init__(self, config=None):
            raise ImportError("PyTorch is required for SpatioTemporalGNN")
