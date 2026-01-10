"""Edge weight computation for ST-GNN graph.

Implements various edge weighting schemes:
- Distance-based
- Flow-direction weighted
- Terrain-aware
- Normalized

Reference: projectfile.md Step 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EdgeWeightType(Enum):
    """Types of edge weight computation."""
    UNIFORM = "uniform"
    DISTANCE = "distance"
    FLOW = "flow"
    ELEVATION = "elevation"
    COMBINED = "combined"


@dataclass
class EdgeWeightConfig:
    """Configuration for edge weight computation."""
    weight_type: EdgeWeightType = EdgeWeightType.COMBINED
    distance_decay: float = 1.0  # larger = faster decay with distance
    flow_influence: float = 0.5  # weight of flow accumulation
    elevation_influence: float = 0.5  # weight of elevation difference
    min_weight: float = 0.01
    max_weight: float = 1.0
    normalize: bool = True


def compute_edge_weights(
    edge_index: np.ndarray,
    node_coords: np.ndarray,
    elevation: Optional[np.ndarray] = None,
    flow_accumulation: Optional[np.ndarray] = None,
    config: Optional[EdgeWeightConfig] = None,
) -> np.ndarray:
    """Compute edge weights based on spatial and hydrological properties.
    
    Args:
        edge_index: (2, E) edge connectivity [source, target]
        node_coords: (N, 2) node coordinates (y, x)
        elevation: (N,) elevation at each node (optional)
        flow_accumulation: (N,) flow accumulation at each node (optional)
        config: weight configuration
        
    Returns:
        edge_weights: (E,) computed edge weights
    """
    config = config or EdgeWeightConfig()
    
    num_edges = edge_index.shape[1]
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # Get coordinates
    src_coords = node_coords[src_nodes]
    dst_coords = node_coords[dst_nodes]
    
    # Compute distances
    distances = np.sqrt(np.sum((src_coords - dst_coords) ** 2, axis=1))
    distances = np.maximum(distances, 1e-6)  # Avoid division by zero
    
    if config.weight_type == EdgeWeightType.UNIFORM:
        weights = np.ones(num_edges, dtype=np.float32)
        
    elif config.weight_type == EdgeWeightType.DISTANCE:
        # Inverse distance weighting with decay
        weights = np.exp(-config.distance_decay * distances)
        
    elif config.weight_type == EdgeWeightType.FLOW:
        if flow_accumulation is None:
            logger.warning("Flow accumulation not provided, using distance weights")
            weights = np.exp(-config.distance_decay * distances)
        else:
            src_flow = flow_accumulation[src_nodes]
            dst_flow = flow_accumulation[dst_nodes]
            
            # Weight by flow ratio (upstream → downstream has higher weight)
            flow_ratio = (src_flow + 1) / (dst_flow + 1)
            flow_weight = np.log1p(flow_ratio)
            flow_weight = flow_weight / (flow_weight.max() + 1e-8)
            
            # Combine with distance
            dist_weight = np.exp(-config.distance_decay * distances)
            weights = config.flow_influence * flow_weight + (1 - config.flow_influence) * dist_weight
            
    elif config.weight_type == EdgeWeightType.ELEVATION:
        if elevation is None:
            logger.warning("Elevation not provided, using distance weights")
            weights = np.exp(-config.distance_decay * distances)
        else:
            src_elev = elevation[src_nodes]
            dst_elev = elevation[dst_nodes]
            
            # Weight by elevation difference (downhill flow preferred)
            elev_diff = src_elev - dst_elev
            elev_weight = np.where(
                elev_diff > 0,
                np.exp(-elev_diff / 10.0),  # Downhill: decaying weight
                0.1  # Uphill: low weight
            )
            
            # Combine with distance
            dist_weight = np.exp(-config.distance_decay * distances)
            weights = config.elevation_influence * elev_weight + (1 - config.elevation_influence) * dist_weight
            
    elif config.weight_type == EdgeWeightType.COMBINED:
        # Combined: distance + flow + elevation
        dist_weight = np.exp(-config.distance_decay * distances)
        
        if flow_accumulation is not None:
            src_flow = flow_accumulation[src_nodes]
            dst_flow = flow_accumulation[dst_nodes]
            flow_ratio = (src_flow + 1) / (dst_flow + 1)
            flow_weight = np.log1p(np.abs(np.log(flow_ratio + 1e-8)))
            flow_weight = flow_weight / (flow_weight.max() + 1e-8)
        else:
            flow_weight = np.zeros(num_edges)
        
        if elevation is not None:
            src_elev = elevation[src_nodes]
            dst_elev = elevation[dst_nodes]
            elev_diff = src_elev - dst_elev
            elev_weight = 1.0 / (1.0 + np.exp(-elev_diff))  # Sigmoid
        else:
            elev_weight = np.ones(num_edges) * 0.5
        
        # Weighted combination
        w_dist = 0.4
        w_flow = 0.3 * config.flow_influence
        w_elev = 0.3 * config.elevation_influence
        w_total = w_dist + w_flow + w_elev
        
        weights = (w_dist * dist_weight + w_flow * flow_weight + w_elev * elev_weight) / w_total
    else:
        raise ValueError(f"Unknown weight type: {config.weight_type}")
    
    # Clip weights
    weights = np.clip(weights, config.min_weight, config.max_weight)
    
    # Normalize
    if config.normalize and len(weights) > 0:
        weights = weights / (weights.max() + 1e-8)
    
    return weights.astype(np.float32)


def compute_attention_weights(
    edge_index: np.ndarray,
    node_features: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute attention-style edge weights from node features.
    
    Uses dot-product attention between source and target node features.
    
    Args:
        edge_index: (2, E) edge connectivity
        node_features: (N, F) node feature matrix
        temperature: softmax temperature (higher = more uniform)
        
    Returns:
        attention_weights: (E,) attention weights (sum to ~1 per node)
    """
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    src_features = node_features[src_nodes]
    dst_features = node_features[dst_nodes]
    
    # Dot product attention
    attention_scores = np.sum(src_features * dst_features, axis=1)
    attention_scores = attention_scores / temperature
    
    # Softmax per source node
    attention_weights = np.zeros_like(attention_scores)
    unique_src = np.unique(src_nodes)
    
    for src in unique_src:
        mask = src_nodes == src
        scores = attention_scores[mask]
        exp_scores = np.exp(scores - scores.max())  # Numerical stability
        attention_weights[mask] = exp_scores / (exp_scores.sum() + 1e-8)
    
    return attention_weights.astype(np.float32)


def normalize_edge_weights_symmetric(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Apply symmetric normalization to edge weights (GCN-style).
    
    Computes: D^(-1/2) * A * D^(-1/2) style normalization.
    
    Args:
        edge_index: (2, E) edge connectivity
        edge_weights: (E,) raw edge weights
        num_nodes: total number of nodes
        
    Returns:
        normalized_weights: (E,) symmetrically normalized weights
    """
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # Compute degree for each node
    degree = np.zeros(num_nodes, dtype=np.float32)
    for e_idx in range(len(edge_weights)):
        degree[src_nodes[e_idx]] += edge_weights[e_idx]
    
    # D^(-1/2)
    degree_inv_sqrt = 1.0 / (np.sqrt(degree) + 1e-8)
    
    # Apply symmetric normalization
    normalized = np.zeros_like(edge_weights)
    for e_idx in range(len(edge_weights)):
        src = src_nodes[e_idx]
        dst = dst_nodes[e_idx]
        normalized[e_idx] = degree_inv_sqrt[src] * edge_weights[e_idx] * degree_inv_sqrt[dst]
    
    return normalized
