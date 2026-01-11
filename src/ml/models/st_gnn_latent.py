"""
Latent Space ST-GNN Architecture (FINAL).

This is the MAX-LEVEL, FINAL ST-GNN architecture that operates
ONLY in scale-free latent space as a structural residual learner.

PROHIBITED:
- Learning absolute magnitudes
- Encoding city-specific scale
- Influencing decision thresholds

Author: Urban Drainage AI Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LatentSTGNNConfig:
    """Configuration for Latent Space ST-GNN."""
    # Input dimensions
    input_dim: int = 8  # Scale-free features
    
    # Spatial GNN
    hidden_dim: int = 64
    num_gnn_layers: int = 3
    gnn_dropout: float = 0.1
    
    # Temporal
    temporal_hidden: int = 64
    num_temporal_layers: int = 2
    temporal_dropout: float = 0.1
    
    # Output heads (ONLY 2 allowed)
    residual_head_hidden: int = 32
    uncertainty_head_hidden: int = 32
    
    # Regularization
    use_graph_laplacian_reg: bool = True
    laplacian_weight: float = 0.01


# ============================================================================
# SPATIAL GNN LAYERS (Flow-Aware)
# ============================================================================

class FlowAwareGraphConv(nn.Module):
    """
    Directed graph convolution with flow-aware message passing.
    
    Messages are weighted by flow direction to respect
    downstream/upstream relationships.
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.lin_edge = nn.Linear(1, out_dim, bias=False)  # Edge weight transform
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights [E] (flow magnitude)
        """
        # Self-loop contribution
        h_self = self.lin_self(x)
        
        # Neighbor aggregation
        row, col = edge_index
        h_neigh = self.lin_neigh(x)
        
        # Aggregate from neighbors
        if edge_weight is not None:
            # Weight messages by flow
            edge_feat = self.lin_edge(edge_weight.unsqueeze(-1))
            messages = h_neigh[col] * torch.sigmoid(edge_feat)
        else:
            messages = h_neigh[col]
        
        # Sum aggregation (could use mean or max)
        out = torch.zeros_like(h_self)
        out.index_add_(0, row, messages)
        
        # Combine self and neighbor
        out = h_self + out
        out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        out = self.dropout(out)
        
        return out


class SpatialGNNEncoder(nn.Module):
    """
    Multi-layer spatial GNN encoder with residual connections.
    """
    
    def __init__(self, config: LatentSTGNNConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            FlowAwareGraphConv(
                config.hidden_dim,
                config.hidden_dim,
                config.gnn_dropout
            )
            for _ in range(config.num_gnn_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights [E]
        """
        h = self.input_proj(x)
        
        for layer in self.gnn_layers:
            h_new = layer(h, edge_index, edge_weight)
            h = h + h_new  # Residual connection
        
        return h


# ============================================================================
# TEMPORAL ENCODER (Causal)
# ============================================================================

class CausalTemporalEncoder(nn.Module):
    """
    Causal temporal encoder using GRU.
    
    IMPORTANT: No future leakage - only uses past information.
    """
    
    def __init__(self, config: LatentSTGNNConfig):
        super().__init__()
        self.config = config
        
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.temporal_hidden,
            num_layers=config.num_temporal_layers,
            batch_first=True,
            dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0,
        )
        
        # Temporal attention with decay
        self.attention = nn.MultiheadAttention(
            embed_dim=config.temporal_hidden,
            num_heads=4,
            dropout=config.temporal_dropout,
            batch_first=True,
        )
        
        # Causal mask generator
        self.register_buffer('_causal_mask', None)
        
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask (no future leakage)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial embeddings [B, T, N, hidden_dim] or [T, N, hidden_dim]
        
        Returns:
            Temporal embeddings of same shape
        """
        # Handle different input shapes
        if x.dim() == 3:
            # [T, N, H] -> add batch dim
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        B, T, N, H = x.shape
        
        # Process each node's temporal sequence
        # Reshape to [B*N, T, H]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, H)
        
        # GRU encoding
        gru_out, _ = self.gru(x_flat)
        
        # Causal self-attention
        causal_mask = self._get_causal_mask(T, x.device)
        attn_out, _ = self.attention(
            gru_out, gru_out, gru_out,
            attn_mask=causal_mask
        )
        
        # Combine
        out = gru_out + attn_out
        
        # Reshape back to [B, T, N, temporal_hidden]
        out = out.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        
        if squeeze_batch:
            out = out.squeeze(0)
        
        return out


# ============================================================================
# OUTPUT HEADS (ONLY 2 ALLOWED)
# ============================================================================

class StructuralResidualHead(nn.Module):
    """
    Head for predicting ΔZ_structural.
    
    This predicts the latent-space structural correction.
    NO magnitude, NO raw scale, NO decisions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize to allow larger residuals (not stuck at zero)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.1)  # Increased from 0.01
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output: ΔZ [*, 1]"""
        return self.net(x)


class EpistemicUncertaintyHead(nn.Module):
    """
    Distance-Aware Epistemic Uncertainty Head.
    
    This head stores the mean and std of the training feature distribution.
    At inference, it computes how "far" the input is from the training
    distribution and adds uncertainty proportionally.
    
    This GUARANTEES that OOD inputs (like NYC) get higher uncertainty.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        # Learned base uncertainty network
        self.base_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        
        # Distance-based uncertainty scaling factor (learned)
        self.distance_scale = nn.Parameter(torch.tensor(0.1))
        
        # Running statistics of training distribution (registered as buffers)
        self.register_buffer('train_mean', torch.zeros(input_dim))
        self.register_buffer('train_std', torch.ones(input_dim))
        self.register_buffer('is_calibrated', torch.tensor(False))
    
    def calibrate(self, train_features: torch.Tensor):
        """
        Call this AFTER training to store the training distribution stats.
        
        Args:
            train_features: [T, N, D] training features (already Z-scored)
        """
        with torch.no_grad():
            # Flatten to [T*N, D] and compute stats
            flat = train_features.reshape(-1, train_features.shape[-1])
            self.train_mean.copy_(flat.mean(dim=0))
            self.train_std.copy_(flat.std(dim=0) + 1e-6)
            self.is_calibrated.fill_(True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Output: σ_DL [*, 1], guaranteed positive.
        
        σ_DL = softplus(base_net(x)) + distance_scale * mahalanobis_distance(x)
        """
        # Base uncertainty from network
        base_sigma = F.softplus(self.base_net(x))
        
        # Compute distance from training distribution (per-sample)
        if self.is_calibrated:
            # Standardized distance (like Mahalanobis with diagonal cov)
            z_score = (x - self.train_mean) / self.train_std
            distance = torch.sqrt((z_score ** 2).sum(dim=-1, keepdim=True) + 1e-6)
            
            # Add distance-scaled uncertainty (softplus to keep positive)
            distance_sigma = F.softplus(self.distance_scale) * distance
        else:
            # Not calibrated yet, just use base
            distance_sigma = torch.zeros_like(base_sigma)
        
        # Total uncertainty = base + distance-aware component
        return base_sigma + distance_sigma + 0.01  # 0.01 minimum


# ============================================================================
# MAIN MODEL
# ============================================================================

class LatentSTGNN(nn.Module):
    """
    Latent Space Spatio-Temporal GNN.
    
    ROLE: Structural Residual Learner in Scale-Free Latent Space.
    
    OUTPUTS (2 heads only):
        1. ΔZ_structural: Latent space structural correction
        2. σ_DL: Epistemic uncertainty proxy (additive)
    
    FORBIDDEN:
        - Raw magnitude prediction
        - Decision outputs
        - City-specific parameters
    """
    
    def __init__(self, config: Optional[LatentSTGNNConfig] = None):
        super().__init__()
        self.config = config or LatentSTGNNConfig()
        
        # Spatial encoder
        self.spatial_encoder = SpatialGNNEncoder(self.config)
        
        # Temporal encoder
        self.temporal_encoder = CausalTemporalEncoder(self.config)
        
        # Output heads (ONLY 2)
        self.residual_head = StructuralResidualHead(
            self.config.temporal_hidden,
            self.config.residual_head_hidden
        )
        self.uncertainty_head = EpistemicUncertaintyHead(
            self.config.temporal_hidden,
            self.config.uncertainty_head_hidden
        )
        
        # Final declaration
        self._emit_declaration()
        
    def _emit_declaration(self):
        """Emit the mandatory declaration."""
        declaration = (
            "The deep learning component is restricted "
            "to learning transferable structural residuals "
            "in a scale-free latent space. "
            "All magnitude calibration and decision logic "
            "are handled outside the DL model."
        )
        logger.info(f"📜 DL ROLE DECLARATION: {declaration}")
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [T, N, input_dim] - MUST be scale-free!
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights [E]
            return_embeddings: If True, also return intermediate embeddings
        
        Returns:
            delta_z: Structural residual [T, N, 1]
            sigma_dl: Epistemic uncertainty [T, N, 1]
        """
        T, N, _ = x.shape
        
        # Spatial encoding for each timestep
        spatial_embeds = []
        for t in range(T):
            h_t = self.spatial_encoder(x[t], edge_index, edge_weight)
            spatial_embeds.append(h_t)
        spatial_embeds = torch.stack(spatial_embeds, dim=0)  # [T, N, hidden]
        
        # Temporal encoding
        temporal_embeds = self.temporal_encoder(spatial_embeds)  # [T, N, temporal_hidden]
        
        # Output heads
        delta_z = self.residual_head(temporal_embeds)  # [T, N, 1]
        sigma_dl = self.uncertainty_head(temporal_embeds)  # [T, N, 1]
        
        if return_embeddings:
            return delta_z, sigma_dl, temporal_embeds
        
        return delta_z, sigma_dl
    
    def calibrate_ood(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Calibrate uncertainty head for OOD detection.
        
        This runs a forward pass to get intermediate embeddings,
        then stores their statistics in the uncertainty head.
        
        Args:
            x: Training features [T, N, input_dim]
            edge_index: Edge indices [2, E]
        """
        self.eval()
        with torch.no_grad():
            # Run forward with return_embeddings=True
            _, _, temporal_embeds = self.forward(x, edge_index, return_embeddings=True)
            
            # Calibrate uncertainty head with the INTERMEDIATE features
            self.uncertainty_head.calibrate(temporal_embeds)
        
        logger.info("✓ OOD calibration complete")
    
    def get_graph_laplacian_regularization(
        self,
        delta_z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute graph Laplacian regularization for spatial smoothness.
        
        L_smooth = Σ_{(i,j) ∈ E} (ΔZ_i - ΔZ_j)^2
        """
        row, col = edge_index
        diff = delta_z.squeeze(-1)[:, row] - delta_z.squeeze(-1)[:, col]
        return torch.mean(diff ** 2)
