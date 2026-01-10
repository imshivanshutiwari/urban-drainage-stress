"""Temporal graph dataset for ST-GNN training.

Creates temporal sequences of graphs for training:
- Sliding window approach
- Target: residual = observed_stress - bayesian_stress
- Time-aware train/val split (NO shuffling)

Reference: projectfile.md Step 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Iterator
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemporalGraphSequence:
    """A temporal sequence of graph snapshots.
    
    Attributes:
        node_features: (T, N, F) node features over time
        edge_index: (2, E) static edge connectivity
        edge_weights: (E,) edge weights
        targets: (T, N) target residuals
        timestamps: list of timestamps
        node_coords: (N, 2) spatial coordinates
        mask: (T, N) validity mask
    """
    node_features: np.ndarray  # (T, N, F)
    edge_index: np.ndarray  # (2, E)
    edge_weights: np.ndarray  # (E,)
    targets: np.ndarray  # (T, N)
    timestamps: List[Any]
    node_coords: np.ndarray  # (N, 2)
    mask: Optional[np.ndarray] = None  # (T, N)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_timesteps(self) -> int:
        return self.node_features.shape[0]
    
    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[1]
    
    @property
    def num_features(self) -> int:
        return self.node_features.shape[2]
    
    def to_pyg_sequence(self):
        """Convert to PyTorch Geometric temporal format."""
        try:
            import torch
            from torch_geometric.data import Data
            
            data_list = []
            for t in range(self.num_timesteps):
                data = Data(
                    x=torch.tensor(self.node_features[t], dtype=torch.float32),
                    edge_index=torch.tensor(self.edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(self.edge_weights, dtype=torch.float32).unsqueeze(1),
                    y=torch.tensor(self.targets[t], dtype=torch.float32),
                    pos=torch.tensor(self.node_coords, dtype=torch.float32),
                )
                if self.mask is not None:
                    data.mask = torch.tensor(self.mask[t], dtype=torch.bool)
                data_list.append(data)
            return data_list
        except ImportError:
            logger.warning("PyTorch Geometric not installed")
            return None


class DrainageTemporalDataset:
    """Dataset for temporal graph sequences.
    
    Handles:
    - Sliding window construction
    - Residual target computation
    - Train/validation splitting (temporal, no shuffling)
    """
    
    def __init__(
        self,
        window_size: int = 7,
        stride: int = 1,
        prediction_horizon: int = 1,
        min_valid_fraction: float = 0.5,
    ):
        """Initialize dataset.
        
        Args:
            window_size: number of timesteps in input window
            stride: step between consecutive windows
            prediction_horizon: how many steps ahead to predict
            min_valid_fraction: minimum fraction of valid data per window
        """
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.min_valid_fraction = min_valid_fraction
        
        self.sequences: List[TemporalGraphSequence] = []
        self._is_built = False
    
    def build(
        self,
        node_features: np.ndarray,  # (T, N, F)
        edge_index: np.ndarray,  # (2, E)
        edge_weights: np.ndarray,  # (E,)
        observed_stress: np.ndarray,  # (T, N) or (T, H, W)
        bayesian_stress: np.ndarray,  # (T, N) or (T, H, W)
        node_coords: np.ndarray,  # (N, 2)
        timestamps: Optional[List] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> 'DrainageTemporalDataset':
        """Build dataset from raw data.
        
        Args:
            node_features: (T, N, F) temporal node features
            edge_index: static graph connectivity
            edge_weights: edge weights
            observed_stress: (T, N) or (T, H, W) observed/adjusted stress
            bayesian_stress: (T, N) or (T, H, W) baseline Bayesian stress
            node_coords: (N, 2) node coordinates
            timestamps: optional timestamp labels
            roi_mask: optional validity mask
            
        Returns:
            self for chaining
        """
        T, N = node_features.shape[:2]
        
        # Handle grid-format stress (convert to node format)
        if observed_stress.ndim == 3:
            observed_stress = self._grid_to_nodes(observed_stress, node_coords)
        if bayesian_stress.ndim == 3:
            bayesian_stress = self._grid_to_nodes(bayesian_stress, node_coords)
        
        # Compute residual targets
        targets = compute_residual_targets(observed_stress, bayesian_stress)
        
        # Create validity mask
        if roi_mask is not None:
            if roi_mask.ndim == 2:
                # Expand to temporal
                mask = np.broadcast_to(roi_mask[np.newaxis, :], targets.shape)
            else:
                mask = roi_mask
        else:
            mask = ~np.isnan(targets)
        
        # Fill NaNs in targets
        targets = np.nan_to_num(targets, nan=0.0)
        
        # Create timestamps if not provided
        if timestamps is None:
            timestamps = list(range(T))
        
        # Build sliding windows
        self.sequences = []
        num_windows = (T - self.window_size - self.prediction_horizon) // self.stride + 1
        
        logger.info(f"Building {num_windows} temporal windows of size {self.window_size}")
        
        for w in range(num_windows):
            start = w * self.stride
            end = start + self.window_size
            pred_idx = end + self.prediction_horizon - 1
            
            if pred_idx >= T:
                break
            
            # Check validity
            window_mask = mask[start:end]
            valid_fraction = window_mask.mean()
            
            if valid_fraction < self.min_valid_fraction:
                logger.debug(f"Skipping window {w}: only {valid_fraction:.2%} valid")
                continue
            
            seq = TemporalGraphSequence(
                node_features=node_features[start:end].copy(),
                edge_index=edge_index.copy(),
                edge_weights=edge_weights.copy(),
                targets=targets[start:end].copy(),
                timestamps=timestamps[start:end],
                node_coords=node_coords.copy(),
                mask=window_mask.copy(),
                metadata={
                    'start_idx': start,
                    'end_idx': end,
                    'pred_idx': pred_idx,
                    'window_id': w,
                }
            )
            self.sequences.append(seq)
        
        self._is_built = True
        logger.info(f"Built dataset with {len(self.sequences)} valid sequences")
        
        return self
    
    def _grid_to_nodes(
        self, 
        grid_data: np.ndarray,  # (T, H, W)
        node_coords: np.ndarray,  # (N, 2)
    ) -> np.ndarray:
        """Convert grid data to node format."""
        T = grid_data.shape[0]
        N = node_coords.shape[0]
        node_data = np.zeros((T, N), dtype=np.float32)
        
        for n_idx, (y, x) in enumerate(node_coords.astype(int)):
            if 0 <= y < grid_data.shape[1] and 0 <= x < grid_data.shape[2]:
                node_data[:, n_idx] = grid_data[:, y, x]
            else:
                node_data[:, n_idx] = np.nan
        
        return node_data
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> TemporalGraphSequence:
        return self.sequences[idx]
    
    def __iter__(self) -> Iterator[TemporalGraphSequence]:
        return iter(self.sequences)


def compute_residual_targets(
    observed_stress: np.ndarray,
    bayesian_stress: np.ndarray,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """Compute residual targets for DL training.
    
    The DL model learns: residual = observed - bayesian
    So: final_stress = bayesian + dl_prediction
    
    Args:
        observed_stress: (T, N) observed/ground-truth stress
        bayesian_stress: (T, N) baseline Bayesian prediction
        clip_percentile: percentile for outlier clipping
        
    Returns:
        residuals: (T, N) target residuals
    """
    residuals = observed_stress - bayesian_stress
    
    # Handle NaNs
    valid = ~np.isnan(residuals)
    
    if valid.any():
        # Clip outliers
        lower = np.nanpercentile(residuals, 100 - clip_percentile)
        upper = np.nanpercentile(residuals, clip_percentile)
        residuals = np.clip(residuals, lower, upper)
        
        # Log statistics
        mean_residual = np.nanmean(np.abs(residuals))
        std_residual = np.nanstd(residuals)
        logger.info(
            f"Residual stats: mean_abs={mean_residual:.4f}, std={std_residual:.4f}"
        )
    
    return residuals.astype(np.float32)


def create_train_val_split(
    dataset: DrainageTemporalDataset,
    val_fraction: float = 0.2,
    temporal_split: bool = True,
) -> Tuple[List[TemporalGraphSequence], List[TemporalGraphSequence]]:
    """Split dataset into train and validation sets.
    
    CRITICAL: Uses temporal split (no shuffling) to prevent data leakage.
    
    Args:
        dataset: built DrainageTemporalDataset
        val_fraction: fraction of data for validation
        temporal_split: if True, use last portion for validation (no shuffle)
        
    Returns:
        (train_sequences, val_sequences)
    """
    if not dataset._is_built:
        raise ValueError("Dataset not built yet")
    
    n_sequences = len(dataset.sequences)
    n_val = max(1, int(n_sequences * val_fraction))
    n_train = n_sequences - n_val
    
    if temporal_split:
        # Temporal split: last n_val sequences for validation
        train_sequences = dataset.sequences[:n_train]
        val_sequences = dataset.sequences[n_train:]
        logger.info(
            f"Temporal split: {n_train} train, {n_val} val sequences "
            f"(val starts at window {train_sequences[-1].metadata['window_id'] + 1 if train_sequences else 0})"
        )
    else:
        # Random split (NOT RECOMMENDED for temporal data)
        logger.warning("Random split may cause temporal leakage - use temporal_split=True")
        indices = np.random.permutation(n_sequences)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        train_sequences = [dataset.sequences[i] for i in sorted(train_indices)]
        val_sequences = [dataset.sequences[i] for i in sorted(val_indices)]
    
    return train_sequences, val_sequences
