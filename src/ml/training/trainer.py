"""Training pipeline for ST-GNN.

Implements:
- Temporal train/val split
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Deterministic seeds

Reference: projectfile.md Step 5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrainerConfig:
    """Training configuration."""
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Optimization
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "plateau"  # "plateau", "cosine", "none"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Regularization
    gradient_clip: float = 1.0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging
    log_interval: int = 10
    save_best: bool = True
    checkpoint_dir: Optional[str] = None


@dataclass
class TrainingHistory:
    """Training history tracker."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_residual: List[float] = field(default_factory=list)
    val_residual: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    training_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_residual': self.train_residual,
            'val_residual': self.val_residual,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'training_time': self.training_time,
        }
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


if TORCH_AVAILABLE:
    
    class Trainer:
        """Training pipeline for ST-GNN.
        
        Handles:
        - Training loop with validation
        - Early stopping
        - Learning rate scheduling
        - Gradient clipping
        - Checkpointing
        """
        
        def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            config: Optional[TrainerConfig] = None,
            device: Optional[str] = None,
        ):
            self.model = model
            self.loss_fn = loss_fn
            self.config = config or TrainerConfig()
            
            # Device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            self.model = self.model.to(self.device)
            
            # Set seeds
            self._set_seeds()
            
            # Optimizer
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # State
            self.history = TrainingHistory()
            self.best_model_state = None
            
            logger.info(
                f"Trainer initialized: device={device}, "
                f"epochs={self.config.epochs}, lr={self.config.learning_rate}"
            )
        
        def _set_seeds(self):
            """Set random seeds for reproducibility."""
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            
            if self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        def _create_optimizer(self) -> optim.Optimizer:
            """Create optimizer."""
            params = self.model.parameters()
            
            if self.config.optimizer == "adam":
                return optim.Adam(
                    params, 
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            elif self.config.optimizer == "adamw":
                return optim.AdamW(
                    params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            elif self.config.optimizer == "sgd":
                return optim.SGD(
                    params,
                    lr=self.config.learning_rate,
                    momentum=0.9,
                    weight_decay=self.config.weight_decay,
                )
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        def _create_scheduler(self):
            """Create learning rate scheduler."""
            if self.config.scheduler == "plateau":
                return ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=self.config.scheduler_factor,
                    patience=self.config.scheduler_patience,
                    verbose=True,
                )
            elif self.config.scheduler == "cosine":
                return CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs,
                )
            else:
                return None
        
        def train(
            self,
            train_data: List,  # List of TemporalGraphSequence
            val_data: List,
        ) -> TrainingHistory:
            """Train the model.
            
            Args:
                train_data: list of training sequences
                val_data: list of validation sequences
                
            Returns:
                TrainingHistory with metrics
            """
            start_time = time.time()
            epochs_no_improve = 0
            
            logger.info(f"Starting training: {len(train_data)} train, {len(val_data)} val")
            
            for epoch in range(self.config.epochs):
                # Training
                train_metrics = self._train_epoch(train_data)
                
                # Validation
                val_metrics = self._validate(val_data)
                
                # Record history
                self.history.train_loss.append(train_metrics['total'])
                self.history.val_loss.append(val_metrics['total'])
                self.history.train_residual.append(train_metrics.get('residual', 0))
                self.history.val_residual.append(val_metrics.get('residual', 0))
                self.history.learning_rates.append(self.optimizer.param_groups[0]['lr'])
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['total'])
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                if val_metrics['total'] < self.history.best_val_loss - self.config.early_stopping_min_delta:
                    self.history.best_val_loss = val_metrics['total']
                    self.history.best_epoch = epoch
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                # Logging
                if epoch % self.config.log_interval == 0 or epoch == self.config.epochs - 1:
                    logger.info(
                        f"Epoch {epoch}/{self.config.epochs} - "
                        f"train_loss: {train_metrics['total']:.4f}, "
                        f"val_loss: {val_metrics['total']:.4f}, "
                        f"lr: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                
                # Early stopping
                if self.config.early_stopping and epochs_no_improve >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Restore best model
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                logger.info(f"Restored best model from epoch {self.history.best_epoch}")
            
            self.history.training_time = time.time() - start_time
            logger.info(
                f"Training complete: best_val_loss={self.history.best_val_loss:.4f} "
                f"at epoch {self.history.best_epoch}, time={self.history.training_time:.1f}s"
            )
            
            return self.history
        
        def _train_epoch(self, train_data: List) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            total_metrics = {'total': 0.0, 'residual': 0.0, 'uncertainty': 0.0}
            num_batches = 0
            
            for seq in train_data:
                # Move to device
                x = torch.tensor(seq.node_features, dtype=torch.float32, device=self.device)
                edge_index = torch.tensor(seq.edge_index, dtype=torch.long, device=self.device)
                edge_weight = torch.tensor(seq.edge_weights, dtype=torch.float32, device=self.device)
                targets = torch.tensor(seq.targets, dtype=torch.float32, device=self.device)
                
                if seq.mask is not None:
                    mask = torch.tensor(seq.mask, dtype=torch.bool, device=self.device)
                else:
                    mask = None
                
                # Forward pass
                self.optimizer.zero_grad()
                pred_residual, pred_uncertainty = self.model(x, edge_index, edge_weight)
                
                # Compute loss
                loss, loss_dict = self.loss_fn(
                    pred_residual.squeeze(-1),
                    pred_uncertainty.squeeze(-1) if pred_uncertainty is not None else None,
                    targets,
                    edge_index,
                    edge_weight,
                    mask,
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
                
                # Accumulate metrics
                for k, v in loss_dict.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                num_batches += 1
            
            # Average
            return {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        
        @torch.no_grad()
        def _validate(self, val_data: List) -> Dict[str, float]:
            """Validate on validation set."""
            self.model.eval()
            total_metrics = {'total': 0.0, 'residual': 0.0}
            num_batches = 0
            
            for seq in val_data:
                x = torch.tensor(seq.node_features, dtype=torch.float32, device=self.device)
                edge_index = torch.tensor(seq.edge_index, dtype=torch.long, device=self.device)
                edge_weight = torch.tensor(seq.edge_weights, dtype=torch.float32, device=self.device)
                targets = torch.tensor(seq.targets, dtype=torch.float32, device=self.device)
                
                if seq.mask is not None:
                    mask = torch.tensor(seq.mask, dtype=torch.bool, device=self.device)
                else:
                    mask = None
                
                pred_residual, pred_uncertainty = self.model(x, edge_index, edge_weight)
                
                loss, loss_dict = self.loss_fn(
                    pred_residual.squeeze(-1),
                    pred_uncertainty.squeeze(-1) if pred_uncertainty is not None else None,
                    targets,
                    edge_index,
                    edge_weight,
                    mask,
                )
                
                for k, v in loss_dict.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                num_batches += 1
            
            return {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        
        def save_checkpoint(self, path: Path):
            """Save training checkpoint."""
            checkpoint = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'history': self.history.to_dict(),
                'config': self.config,
            }
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
        
        def load_checkpoint(self, path: Path):
            """Load training checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            logger.info(f"Checkpoint loaded from {path}")


else:
    class Trainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
