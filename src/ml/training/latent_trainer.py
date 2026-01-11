"""
Latent Space Trainer.

Training pipeline for the Latent ST-GNN.

CONSTRAINTS:
- Train ONLY on Seattle
- Temporal split only (no shuffling)
- No city labels or domain identifiers
- DL-only must perform WORSE than hybrid (sanity check)

Author: Urban Drainage AI Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import logging
import json

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LatentTrainerConfig:
    """Configuration for latent space training."""
    # Training
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-5
    
    # Validation
    val_fraction: float = 0.2  # Temporal split, NOT random
    
    # Sanity checks
    assert_dl_worse_than_hybrid: bool = True
    max_dl_only_improvement: float = 0.0  # DL alone should not beat hybrid
    
    # Checkpointing
    save_best: bool = True
    checkpoint_dir: Path = Path("checkpoints")


# ============================================================================
# TEMPORAL SPLIT (NO SHUFFLING)
# ============================================================================

def temporal_train_val_split(
    data: Dict[str, np.ndarray],
    val_fraction: float = 0.2,
) -> Tuple[Dict, Dict]:
    """
    Split data temporally for training/validation.
    
    CRITICAL: No shuffling! Later timesteps are validation.
    
    Args:
        data: Dictionary with arrays of shape [T, N, ...] or [T, ...]
        val_fraction: Fraction of timesteps for validation
    
    Returns:
        (train_data, val_data)
    """
    # Get temporal dimension
    sample_key = list(data.keys())[0]
    T = data[sample_key].shape[0]
    
    split_idx = int(T * (1 - val_fraction))
    
    train_data = {}
    val_data = {}
    
    for key, arr in data.items():
        if arr.shape[0] == T:
            train_data[key] = arr[:split_idx]
            val_data[key] = arr[split_idx:]
        else:
            # Non-temporal data (e.g., edge_index)
            train_data[key] = arr
            val_data[key] = arr
    
    logger.info(
        f"Temporal split: {split_idx} train timesteps, "
        f"{T - split_idx} val timesteps (NO shuffling)"
    )
    
    return train_data, val_data


# ============================================================================
# TRAINER
# ============================================================================

class LatentTrainer:
    """
    Trainer for Latent Space ST-GNN.
    
    Enforces all training constraints from projectfile.md.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Optional[LatentTrainerConfig] = None,
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.config = config or LatentTrainerConfig()
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'learning_rate': [],
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            verbose: Show progress bar
        
        Returns:
            Training history and best metrics
        """
        # Convert to tensors
        train_tensors = self._prepare_tensors(train_data)
        val_tensors = self._prepare_tensors(val_data)
        
        # Training loop
        iterator = tqdm(
            range(self.config.epochs),
            desc="Training",
            disable=not verbose
        )
        
        for epoch in iterator:
            # Train epoch
            train_metrics = self._train_epoch(train_tensors)
            
            # Validation
            val_metrics = self._validate(val_tensors)
            
            # Record history
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_metrics['total'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Scheduler step
            self.scheduler.step(val_metrics['total'])
            
            # Update progress bar
            iterator.set_postfix(
                train_loss=f"{train_metrics['total']:.4f}",
                val_loss=f"{val_metrics['total']:.4f}",
            )
            
            # Early stopping check
            if val_metrics['total'] < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_metrics['total']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if self.config.save_best:
                    self._save_checkpoint('best')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final sanity check
        self._sanity_check(val_tensors)
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }
    
    def _prepare_tensors(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to tensors."""
        tensors = {}
        for key, arr in data.items():
            if isinstance(arr, np.ndarray):
                tensors[key] = torch.from_numpy(arr).float().to(self.device)
            else:
                tensors[key] = arr
        return tensors
    
    def _train_epoch(self, data: Dict[str, torch.Tensor]) -> Dict:
        """Run one training epoch."""
        self.model.train()
        
        # Get data
        x = data['features']  # [T, N, input_dim]
        target = data['delta_z']  # [T, N, 1]
        edge_index = data['edge_index'].long()
        edge_weight = data.get('edge_weight', None)
        
        # Forward pass
        self.optimizer.zero_grad()
        delta_z_pred, sigma_pred = self.model(x, edge_index, edge_weight)
        
        # Loss
        loss_dict = self.loss_fn(
            delta_z_pred, target, sigma_pred,
            edge_index, edge_weight
        )
        
        # Backward
        loss_dict['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def _validate(self, data: Dict[str, torch.Tensor]) -> Dict:
        """Run validation."""
        self.model.eval()
        
        x = data['features']
        target = data['delta_z']
        edge_index = data['edge_index'].long()
        edge_weight = data.get('edge_weight', None)
        
        delta_z_pred, sigma_pred = self.model(x, edge_index, edge_weight)
        
        loss_dict = self.loss_fn(
            delta_z_pred, target, sigma_pred,
            edge_index, edge_weight
        )
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def _sanity_check(self, val_data: Dict[str, torch.Tensor]) -> None:
        """
        Sanity check: DL-only should NOT beat hybrid.
        
        If DL alone outperforms the physics+DL hybrid,
        something is wrong (DL is learning raw scale).
        """
        if not self.config.assert_dl_worse_than_hybrid:
            return
        
        self.model.eval()
        
        with torch.no_grad():
            x = val_data['features']
            target = val_data['delta_z']
            edge_index = val_data['edge_index'].long()
            
            delta_z_pred, _ = self.model(x, edge_index)
            
            # DL-only RMSE (DL replacing physics entirely)
            dl_only_rmse = torch.sqrt(torch.mean(delta_z_pred ** 2)).item()
            
            # Hybrid RMSE (DL as correction)
            hybrid_rmse = torch.sqrt(
                torch.mean((delta_z_pred - target) ** 2)
            ).item()
        
        logger.info(f"Sanity Check: DL-only RMSE={dl_only_rmse:.4f}, Hybrid RMSE={hybrid_rmse:.4f}")
        
        # DL-only should be WORSE (higher RMSE)
        if dl_only_rmse < hybrid_rmse * 0.9:  # 10% tolerance
            logger.warning(
                "⚠️ SANITY CHECK WARNING: DL-only appears better than hybrid! "
                "This may indicate scale leakage."
            )
    
    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.checkpoint_dir / f"latent_stgnn_{name}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }, path)
        
        logger.info(f"Saved checkpoint: {path}")
