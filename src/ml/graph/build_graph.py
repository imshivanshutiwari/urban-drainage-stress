"""Graph construction for Spatio-Temporal GNN.

This module builds a graph representation of the urban drainage system:
- Nodes: grid cells within ROI
- Edges: hydrological connectivity + spatial adjacency
- Features: rainfall, terrain, Bayesian stress/variance

Reference: projectfile.md Step 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import sobel

logger = logging.getLogger(__name__)


@dataclass
class DrainageGraph:
    """Graph representation of urban drainage network.
    
    Attributes:
        node_features: (N, F) node feature matrix
        edge_index: (2, E) edge connectivity [source, target]
        edge_weights: (E,) edge weights
        node_coords: (N, 2) spatial coordinates (y, x)
        num_nodes: number of nodes
        num_edges: number of edges
        feature_names: list of feature names
        temporal_features: (T, N, F_t) time-varying features if available
    """
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_weights: np.ndarray
    node_coords: np.ndarray
    num_nodes: int
    num_edges: int
    feature_names: List[str]
    temporal_features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_pyg(self):
        """Convert to PyTorch Geometric Data object."""
        try:
            import torch
            from torch_geometric.data import Data
            
            data = Data(
                x=torch.tensor(self.node_features, dtype=torch.float32),
                edge_index=torch.tensor(self.edge_index, dtype=torch.long),
                edge_attr=torch.tensor(self.edge_weights, dtype=torch.float32).unsqueeze(1),
                pos=torch.tensor(self.node_coords, dtype=torch.float32),
            )
            return data
        except ImportError:
            logger.warning("PyTorch Geometric not installed, returning raw data")
            return self


@dataclass
class GraphConfig:
    """Configuration for graph construction."""
    k_neighbors: int = 8  # k-NN for spatial adjacency
    include_hydro_edges: bool = True  # hydrological connectivity
    include_spatial_edges: bool = True  # k-NN edges
    self_loops: bool = True
    bidirectional: bool = True  # make edges bidirectional
    normalize_features: bool = True
    grid_subsample: int = 1  # subsample grid (1 = no subsampling)


def build_drainage_graph(
    rainfall_intensity: np.ndarray,
    rainfall_accumulation: np.ndarray,
    elevation: np.ndarray,
    flow_accumulation: np.ndarray,
    bayesian_stress: np.ndarray,
    bayesian_variance: np.ndarray,
    epistemic_uncertainty: Optional[np.ndarray] = None,
    complaint_density: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
    config: Optional[GraphConfig] = None,
) -> DrainageGraph:
    """Build drainage graph from spatial data.
    
    Args:
        rainfall_intensity: (H, W) or (T, H, W) rainfall intensity
        rainfall_accumulation: (H, W) or (T, H, W) accumulated rainfall
        elevation: (H, W) DEM elevation
        flow_accumulation: (H, W) upstream flow accumulation
        bayesian_stress: (H, W) or (T, H, W) Bayesian stress mean
        bayesian_variance: (H, W) or (T, H, W) Bayesian stress variance
        epistemic_uncertainty: (H, W) optional epistemic component
        complaint_density: (H, W) optional complaint density
        roi_mask: (H, W) boolean mask for valid cells
        config: graph construction configuration
        
    Returns:
        DrainageGraph with nodes, edges, and features
    """
    config = config or GraphConfig()
    
    # Handle temporal data - use snapshot or aggregate
    if rainfall_intensity.ndim == 3:
        # Use peak timestep for static graph
        t_peak = int(np.nanargmax(bayesian_stress.max(axis=(1, 2))))
        ri = rainfall_intensity[t_peak]
        ra = rainfall_accumulation[t_peak]
        stress = bayesian_stress[t_peak]
        variance = bayesian_variance[t_peak]
        temporal_data = {
            'rainfall_intensity': rainfall_intensity,
            'rainfall_accumulation': rainfall_accumulation,
            'bayesian_stress': bayesian_stress,
            'bayesian_variance': bayesian_variance,
        }
    else:
        ri = rainfall_intensity
        ra = rainfall_accumulation
        stress = bayesian_stress
        variance = bayesian_variance
        temporal_data = None
    
    H, W = elevation.shape
    
    # Create ROI mask if not provided
    if roi_mask is None:
        roi_mask = np.ones((H, W), dtype=bool)
    
    # Apply subsampling
    if config.grid_subsample > 1:
        s = config.grid_subsample
        roi_mask = roi_mask[::s, ::s]
        elevation = elevation[::s, ::s]
        flow_accumulation = flow_accumulation[::s, ::s]
        ri = ri[::s, ::s]
        ra = ra[::s, ::s]
        stress = stress[::s, ::s]
        variance = variance[::s, ::s]
        if epistemic_uncertainty is not None:
            epistemic_uncertainty = epistemic_uncertainty[::s, ::s]
        if complaint_density is not None:
            complaint_density = complaint_density[::s, ::s]
        H, W = elevation.shape
    
    # Get valid node indices
    valid_mask = roi_mask & ~np.isnan(elevation)
    node_indices = np.argwhere(valid_mask)  # (N, 2) - (y, x) coordinates
    num_nodes = len(node_indices)
    
    logger.info(f"Building graph with {num_nodes} nodes from {H}x{W} grid")
    
    # Create node-to-index mapping
    node_map = -np.ones((H, W), dtype=int)
    for i, (y, x) in enumerate(node_indices):
        node_map[y, x] = i
    
    # Compute slope from elevation
    slope_y = sobel(elevation, axis=0, mode='constant')
    slope_x = sobel(elevation, axis=1, mode='constant')
    slope = np.sqrt(slope_y**2 + slope_x**2)
    
    # Build node features
    feature_names = [
        'rainfall_intensity',
        'rainfall_accumulation', 
        'elevation',
        'slope',
        'flow_accumulation',
        'bayesian_stress_mean',
        'bayesian_stress_variance',
    ]
    
    features = [
        ri,
        ra,
        elevation,
        slope,
        np.log1p(flow_accumulation),  # log-transform for better scale
        stress,
        variance,
    ]
    
    if epistemic_uncertainty is not None:
        features.append(epistemic_uncertainty)
        feature_names.append('epistemic_uncertainty')
    
    if complaint_density is not None:
        features.append(complaint_density)
        feature_names.append('complaint_density')
    
    # Extract features for valid nodes
    node_features = np.zeros((num_nodes, len(features)), dtype=np.float32)
    for f_idx, feat in enumerate(features):
        for n_idx, (y, x) in enumerate(node_indices):
            node_features[n_idx, f_idx] = feat[y, x]
    
    # Handle NaNs
    node_features = np.nan_to_num(node_features, nan=0.0)
    
    # Normalize features if requested
    if config.normalize_features:
        feat_mean = node_features.mean(axis=0, keepdims=True)
        feat_std = node_features.std(axis=0, keepdims=True) + 1e-8
        node_features = (node_features - feat_mean) / feat_std
    
    # Build edges
    edges_src = []
    edges_dst = []
    edge_weights_list = []
    
    # 1. Hydrological edges (based on flow direction)
    if config.include_hydro_edges:
        hydro_edges, hydro_weights = _build_hydro_edges(
            node_indices, node_map, elevation, flow_accumulation, H, W
        )
        edges_src.extend(hydro_edges[0])
        edges_dst.extend(hydro_edges[1])
        edge_weights_list.extend(hydro_weights)
        logger.info(f"Added {len(hydro_weights)} hydrological edges")
    
    # 2. Spatial adjacency (k-NN)
    if config.include_spatial_edges:
        spatial_edges, spatial_weights = _build_spatial_edges(
            node_indices, config.k_neighbors
        )
        edges_src.extend(spatial_edges[0])
        edges_dst.extend(spatial_edges[1])
        edge_weights_list.extend(spatial_weights)
        logger.info(f"Added {len(spatial_weights)} spatial k-NN edges")
    
    # 3. Self-loops
    if config.self_loops:
        for i in range(num_nodes):
            edges_src.append(i)
            edges_dst.append(i)
            edge_weights_list.append(1.0)
    
    # Make bidirectional
    if config.bidirectional:
        n_edges = len(edges_src)
        edges_src_rev = edges_dst[:n_edges].copy()
        edges_dst_rev = edges_src[:n_edges].copy()
        edges_src.extend(edges_src_rev)
        edges_dst.extend(edges_dst_rev)
        edge_weights_list.extend(edge_weights_list[:n_edges])
    
    # Remove duplicates
    edge_set = set()
    unique_edges_src = []
    unique_edges_dst = []
    unique_weights = []
    for s, d, w in zip(edges_src, edges_dst, edge_weights_list):
        if (s, d) not in edge_set:
            edge_set.add((s, d))
            unique_edges_src.append(s)
            unique_edges_dst.append(d)
            unique_weights.append(w)
    
    edge_index = np.array([unique_edges_src, unique_edges_dst], dtype=np.int64)
    edge_weights = np.array(unique_weights, dtype=np.float32)
    
    # Normalize edge weights
    if len(edge_weights) > 0:
        edge_weights = edge_weights / (edge_weights.max() + 1e-8)
    
    logger.info(f"Graph built: {num_nodes} nodes, {len(unique_weights)} edges")
    
    return DrainageGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weights=edge_weights,
        node_coords=node_indices.astype(np.float32),
        num_nodes=num_nodes,
        num_edges=len(unique_weights),
        feature_names=feature_names,
        temporal_features=None,  # Set separately for temporal graphs
        metadata={
            'grid_shape': (H, W),
            'subsample': config.grid_subsample,
            'config': config,
        }
    )


def _build_hydro_edges(
    node_indices: np.ndarray,
    node_map: np.ndarray,
    elevation: np.ndarray,
    flow_accumulation: np.ndarray,
    H: int, W: int,
) -> Tuple[Tuple[List[int], List[int]], List[float]]:
    """Build hydrological connectivity edges based on flow direction."""
    
    edges_src = []
    edges_dst = []
    weights = []
    
    # 8-connectivity offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for n_idx, (y, x) in enumerate(node_indices):
        elev = elevation[y, x]
        flow = flow_accumulation[y, x]
        
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                neighbor_idx = node_map[ny, nx]
                if neighbor_idx >= 0:  # Valid node
                    neighbor_elev = elevation[ny, nx]
                    neighbor_flow = flow_accumulation[ny, nx]
                    
                    # Edge from higher elevation to lower (upstream → downstream)
                    if elev > neighbor_elev:
                        # Weight based on elevation difference and flow accumulation
                        elev_diff = elev - neighbor_elev
                        flow_ratio = (flow + 1) / (neighbor_flow + 1)
                        weight = np.exp(-elev_diff / 10.0) * np.sqrt(flow_ratio)
                        
                        edges_src.append(n_idx)
                        edges_dst.append(neighbor_idx)
                        weights.append(float(weight))
    
    return (edges_src, edges_dst), weights


def _build_spatial_edges(
    node_indices: np.ndarray,
    k: int,
) -> Tuple[Tuple[List[int], List[int]], List[float]]:
    """Build k-nearest neighbor spatial edges."""
    
    if len(node_indices) <= k:
        k = len(node_indices) - 1
    
    if k <= 0:
        return ([], []), []
    
    # Build KD-tree for efficient neighbor search
    tree = cKDTree(node_indices)
    
    edges_src = []
    edges_dst = []
    weights = []
    
    # Query k+1 neighbors (includes self)
    distances, indices = tree.query(node_indices, k=k + 1)
    
    for n_idx in range(len(node_indices)):
        for j in range(1, k + 1):  # Skip self (j=0)
            neighbor_idx = indices[n_idx, j]
            dist = distances[n_idx, j]
            
            # Weight inversely proportional to distance
            weight = 1.0 / (dist + 1.0)
            
            edges_src.append(n_idx)
            edges_dst.append(neighbor_idx)
            weights.append(float(weight))
    
    return (edges_src, edges_dst), weights
