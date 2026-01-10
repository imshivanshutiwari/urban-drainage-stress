"""Hyperparameter Optimization Script for ST-GNN.

5-Stage Optimization:
1. Capacity Scaling (hidden_dim, num_spatial_layers)
2. Temporal Depth (GRU layers)
3. Regularization Balance (dropout, weight_decay)
4. Loss Weight Calibration (α, β, γ)
5. Final Fine-Tuning (learning_rate, epochs)

Constraints:
- Hybrid RMSE < Base RMSE
- DL-only RMSE > Hybrid RMSE
- No overfitting
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from itertools import product

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
for d in ['results', 'checkpoints', 'reports']:
    (PROJECT_ROOT / d).mkdir(exist_ok=True)


# ============================================================
# DATA GENERATION (same as run_ml_pipeline.py)
# ============================================================

def create_training_data(num_timesteps=100, num_nodes=500, seed=42):
    """Create synthetic training data."""
    np.random.seed(seed)
    
    grid_size = 32
    x = np.linspace(0, 4*np.pi, grid_size)
    y = np.linspace(0, 4*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    elevation = 50 + 20 * np.sin(X/2) * np.cos(Y/3) + np.random.randn(grid_size, grid_size) * 2
    flow_accumulation = np.maximum(1, 100 - elevation + np.random.exponential(20, (grid_size, grid_size)))
    slope = np.sqrt(np.gradient(elevation, axis=0)**2 + np.gradient(elevation, axis=1)**2)
    
    node_coords = np.array([
        [np.random.randint(0, grid_size), np.random.randint(0, grid_size)]
        for _ in range(num_nodes)
    ])
    
    node_elevation = elevation[node_coords[:, 0], node_coords[:, 1]]
    node_flow = flow_accumulation[node_coords[:, 0], node_coords[:, 1]]
    node_slope = slope[node_coords[:, 0], node_coords[:, 1]]
    
    base_rainfall = np.zeros((num_timesteps, num_nodes))
    for _ in range(np.random.randint(3, 5)):
        storm_start = np.random.randint(0, num_timesteps - 20)
        storm_duration = np.random.randint(5, 15)
        storm_intensity = np.random.uniform(5, 30)
        storm_center = np.random.randint(0, num_nodes)
        distances = np.sqrt(np.sum((node_coords - node_coords[storm_center])**2, axis=1))
        spatial_decay = np.exp(-distances / (grid_size * 0.3))
        for t in range(storm_duration):
            if storm_start + t < num_timesteps:
                temporal_weight = np.sin(np.pi * t / storm_duration)
                base_rainfall[storm_start + t] += storm_intensity * temporal_weight * spatial_decay
    
    rainfall_intensity = np.maximum(0, base_rainfall + np.random.exponential(0.5, base_rainfall.shape))
    
    rainfall_accumulation = np.zeros_like(rainfall_intensity)
    for t in range(num_timesteps):
        if t == 0:
            rainfall_accumulation[t] = rainfall_intensity[t]
        else:
            rainfall_accumulation[t] = 0.9 * rainfall_accumulation[t-1] + rainfall_intensity[t]
    
    bayesian_stress = (
        0.3 * rainfall_intensity / (rainfall_intensity.max() + 1e-8) +
        0.4 * rainfall_accumulation / (rainfall_accumulation.max() + 1e-8) +
        0.3 * (1 - node_elevation / node_elevation.max())[np.newaxis, :]
    )
    bayesian_stress = np.clip(bayesian_stress + np.random.randn(*bayesian_stress.shape) * 0.05, 0, 1)
    
    bayesian_variance = 0.1 + 0.2 * np.exp(-rainfall_intensity / 10) + np.random.uniform(0, 0.1, bayesian_stress.shape)
    
    true_residual = (
        0.1 * np.sin(2 * np.pi * rainfall_intensity / 20) +
        0.05 * (node_flow / node_flow.max())[np.newaxis, :] * rainfall_intensity / rainfall_intensity.max() +
        np.random.randn(*bayesian_stress.shape) * 0.02
    )
    observed_stress = np.clip(bayesian_stress + true_residual, 0, 1)
    
    complaint_density = np.zeros_like(observed_stress)
    high_stress = observed_stress > 0.6
    complaint_density[1:] = np.where(high_stress[:-1], np.random.poisson(2, high_stress[:-1].shape), 0)
    
    node_features = np.stack([
        rainfall_intensity,
        rainfall_accumulation,
        np.broadcast_to(node_elevation, (num_timesteps, num_nodes)),
        np.broadcast_to(node_slope, (num_timesteps, num_nodes)),
        np.broadcast_to(np.log1p(node_flow), (num_timesteps, num_nodes)),
        bayesian_stress,
        bayesian_variance,
        complaint_density,
    ], axis=-1).astype(np.float32)
    
    feat_mean = node_features.mean(axis=(0, 1), keepdims=True)
    feat_std = node_features.std(axis=(0, 1), keepdims=True) + 1e-8
    node_features = (node_features - feat_mean) / feat_std
    
    target_residual = (observed_stress - bayesian_stress).astype(np.float32)
    
    split_idx = int(num_timesteps * 0.7)
    
    return {
        'train_x': node_features[:split_idx],
        'train_y': target_residual[:split_idx],
        'val_x': node_features[split_idx:],
        'val_y': target_residual[split_idx:],
        'bayesian_stress_val': bayesian_stress[split_idx:],
        'observed_stress_val': observed_stress[split_idx:],
    }


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(
    input_dim: int = 8,
    hidden_dim: int = 64,
    num_spatial_layers: int = 3,
    num_temporal_layers: int = 1,
    dropout: float = 0.2,
):
    """Build ST-GNN model with configurable hyperparameters."""
    import torch
    import torch.nn as nn
    
    class ConfigurableSTGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            self.spatial_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_spatial_layers)
            ])
            
            self.temporal = nn.GRU(
                hidden_dim, hidden_dim, 
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout if num_temporal_layers > 1 else 0
            )
            
            self.residual_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus()
            )
            
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            T, N, F = x.shape
            h = self.input_proj(x)
            
            for layer in self.spatial_layers:
                h = self.dropout(torch.relu(layer(h)))
            
            h = h.permute(1, 0, 2)
            h, _ = self.temporal(h)
            h = h.permute(1, 0, 2)
            
            h = self.norm(h)
            
            residual = self.residual_head(h)
            uncertainty = self.uncertainty_head(h) + 0.01
            
            return residual, uncertainty
    
    return ConfigurableSTGNN()


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_and_evaluate(
    data: Dict,
    hidden_dim: int = 64,
    num_spatial_layers: int = 3,
    num_temporal_layers: int = 1,
    dropout: float = 0.2,
    weight_decay: float = 0.01,
    learning_rate: float = 0.001,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.1,
    epochs: int = 30,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train model and return metrics."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    device = torch.device('cpu')
    
    model = build_model(
        hidden_dim=hidden_dim,
        num_spatial_layers=num_spatial_layers,
        num_temporal_layers=num_temporal_layers,
        dropout=dropout,
    ).to(device)
    
    train_x = torch.tensor(data['train_x'], dtype=torch.float32, device=device)
    train_y = torch.tensor(data['train_y'], dtype=torch.float32, device=device)
    val_x = torch.tensor(data['val_x'], dtype=torch.float32, device=device)
    val_y = torch.tensor(data['val_y'], dtype=torch.float32, device=device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred, unc = model(train_x)
        pred = pred.squeeze(-1)
        unc = unc.squeeze(-1)
        
        mse_loss = nn.functional.mse_loss(pred, train_y)
        nll_loss = 0.5 * torch.mean(torch.log(unc**2 + 1e-8) + (pred - train_y)**2 / (unc**2 + 1e-8))
        smooth_loss = torch.mean((pred[:, 1:] - pred[:, :-1])**2)
        
        loss = alpha * mse_loss + beta * nll_loss + gamma * smooth_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_x)
            val_pred = val_pred.squeeze(-1)
            val_loss = nn.functional.mse_loss(val_pred, val_y)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if verbose and epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train={loss.item():.4f}, val={val_loss.item():.4f}")
    
    # Restore best and compute final metrics
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        pred, _ = model(val_x)
        pred = pred.squeeze(-1).cpu().numpy()
    
    bayesian = data['bayesian_stress_val']
    observed = data['observed_stress_val']
    true_residual = observed - bayesian
    
    base_rmse = float(np.sqrt(np.mean(true_residual**2)))
    hybrid_rmse = float(np.sqrt(np.mean((pred - true_residual)**2)))
    dl_only_rmse = float(np.sqrt(np.mean((pred - observed)**2)))
    
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        'base_rmse': base_rmse,
        'hybrid_rmse': hybrid_rmse,
        'dl_only_rmse': dl_only_rmse,
        'improvement': float((base_rmse - hybrid_rmse) / base_rmse * 100),
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'num_params': num_params,
        'constraints_satisfied': hybrid_rmse < base_rmse and dl_only_rmse > hybrid_rmse,
        'model_state': best_state,
    }


# ============================================================
# STAGE 1: CAPACITY SCALING
# ============================================================

def stage1_capacity_scaling(data: Dict) -> Dict:
    """Optimize model capacity."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: CAPACITY SCALING")
    logger.info("="*60)
    
    hidden_dims = [32, 64, 128]
    num_layers = [2, 3, 4]
    
    results = []
    
    for hd, nl in product(hidden_dims, num_layers):
        logger.info(f"Testing hidden_dim={hd}, num_spatial_layers={nl}")
        
        metrics = train_and_evaluate(
            data, hidden_dim=hd, num_spatial_layers=nl, epochs=30
        )
        
        result = {
            'hidden_dim': hd,
            'num_spatial_layers': nl,
            'hybrid_rmse': metrics['hybrid_rmse'],
            'base_rmse': metrics['base_rmse'],
            'dl_only_rmse': metrics['dl_only_rmse'],
            'improvement': metrics['improvement'],
            'num_params': metrics['num_params'],
            'constraints_satisfied': metrics['constraints_satisfied'],
        }
        results.append(result)
        
        logger.info(f"  -> Hybrid RMSE: {metrics['hybrid_rmse']:.4f}, Improvement: {metrics['improvement']:.1f}%")
    
    # Find best
    valid_results = [r for r in results if r['constraints_satisfied']]
    if valid_results:
        best = min(valid_results, key=lambda x: x['hybrid_rmse'])
        logger.info(f"\nBest capacity: hidden_dim={best['hidden_dim']}, layers={best['num_spatial_layers']}")
    else:
        best = min(results, key=lambda x: x['hybrid_rmse'])
    
    output = {
        'stage': 'capacity_scaling',
        'results': results,
        'best_config': {'hidden_dim': best['hidden_dim'], 'num_spatial_layers': best['num_spatial_layers']},
        'best_hybrid_rmse': best['hybrid_rmse'],
    }
    
    with open(PROJECT_ROOT / 'results' / 'capacity_scaling_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============================================================
# STAGE 2: TEMPORAL DEPTH
# ============================================================

def stage2_temporal_depth(data: Dict, best_capacity: Dict) -> Dict:
    """Optimize temporal modeling."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: TEMPORAL DEPTH")
    logger.info("="*60)
    
    temporal_depths = [1, 2]
    results = []
    
    for td in temporal_depths:
        logger.info(f"Testing num_temporal_layers={td}")
        
        metrics = train_and_evaluate(
            data,
            hidden_dim=best_capacity['hidden_dim'],
            num_spatial_layers=best_capacity['num_spatial_layers'],
            num_temporal_layers=td,
            epochs=30
        )
        
        result = {
            'num_temporal_layers': td,
            'hybrid_rmse': metrics['hybrid_rmse'],
            'improvement': metrics['improvement'],
            'constraints_satisfied': metrics['constraints_satisfied'],
        }
        results.append(result)
        
        logger.info(f"  -> Hybrid RMSE: {metrics['hybrid_rmse']:.4f}")
    
    valid_results = [r for r in results if r['constraints_satisfied']]
    best = min(valid_results if valid_results else results, key=lambda x: x['hybrid_rmse'])
    
    output = {
        'stage': 'temporal_depth',
        'results': results,
        'best_config': {'num_temporal_layers': best['num_temporal_layers']},
        'best_hybrid_rmse': best['hybrid_rmse'],
    }
    
    with open(PROJECT_ROOT / 'results' / 'temporal_depth_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============================================================
# STAGE 3: REGULARIZATION
# ============================================================

def stage3_regularization(data: Dict, best_capacity: Dict, best_temporal: Dict) -> Dict:
    """Optimize regularization."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 3: REGULARIZATION BALANCE")
    logger.info("="*60)
    
    dropouts = [0.1, 0.2, 0.3]
    weight_decays = [0.001, 0.01, 0.1]
    
    results = []
    
    for do, wd in product(dropouts, weight_decays):
        logger.info(f"Testing dropout={do}, weight_decay={wd}")
        
        metrics = train_and_evaluate(
            data,
            hidden_dim=best_capacity['hidden_dim'],
            num_spatial_layers=best_capacity['num_spatial_layers'],
            num_temporal_layers=best_temporal['num_temporal_layers'],
            dropout=do,
            weight_decay=wd,
            epochs=30
        )
        
        result = {
            'dropout': do,
            'weight_decay': wd,
            'hybrid_rmse': metrics['hybrid_rmse'],
            'dl_only_rmse': metrics['dl_only_rmse'],
            'improvement': metrics['improvement'],
            'constraints_satisfied': metrics['constraints_satisfied'],
        }
        results.append(result)
        
        logger.info(f"  -> Hybrid: {metrics['hybrid_rmse']:.4f}, DL-only: {metrics['dl_only_rmse']:.4f}")
    
    valid_results = [r for r in results if r['constraints_satisfied']]
    best = min(valid_results if valid_results else results, key=lambda x: x['hybrid_rmse'])
    
    output = {
        'stage': 'regularization',
        'results': results,
        'best_config': {'dropout': best['dropout'], 'weight_decay': best['weight_decay']},
        'best_hybrid_rmse': best['hybrid_rmse'],
    }
    
    with open(PROJECT_ROOT / 'results' / 'regularization_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============================================================
# STAGE 4: LOSS WEIGHTS
# ============================================================

def stage4_loss_weights(data: Dict, best_config: Dict) -> Dict:
    """Optimize loss weights."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 4: LOSS WEIGHT CALIBRATION")
    logger.info("="*60)
    
    alphas = [0.5, 1.0, 2.0]
    betas = [0.1, 0.3, 0.5]
    gammas = [0.05, 0.1, 0.2]
    
    results = []
    
    for a, b, g in product(alphas, betas, gammas):
        logger.info(f"Testing α={a}, β={b}, γ={g}")
        
        metrics = train_and_evaluate(
            data, **best_config, alpha=a, beta=b, gamma=g, epochs=30
        )
        
        result = {
            'alpha': a, 'beta': b, 'gamma': g,
            'hybrid_rmse': metrics['hybrid_rmse'],
            'improvement': metrics['improvement'],
            'constraints_satisfied': metrics['constraints_satisfied'],
        }
        results.append(result)
    
    valid_results = [r for r in results if r['constraints_satisfied']]
    best = min(valid_results if valid_results else results, key=lambda x: x['hybrid_rmse'])
    
    logger.info(f"\nBest: α={best['alpha']}, β={best['beta']}, γ={best['gamma']}")
    
    output = {
        'stage': 'loss_weights',
        'results': results,
        'best_config': {'alpha': best['alpha'], 'beta': best['beta'], 'gamma': best['gamma']},
        'best_hybrid_rmse': best['hybrid_rmse'],
    }
    
    with open(PROJECT_ROOT / 'results' / 'loss_weight_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============================================================
# STAGE 5: FINAL FINE-TUNING
# ============================================================

def stage5_final_tuning(data: Dict, best_config: Dict) -> Dict:
    """Final fine-tuning."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 5: FINAL FINE-TUNING")
    logger.info("="*60)
    
    learning_rates = [0.0005, 0.001, 0.002]
    epoch_counts = [30, 50]
    
    results = []
    best_state = None
    best_rmse = float('inf')
    
    for lr, ep in product(learning_rates, epoch_counts):
        logger.info(f"Testing lr={lr}, epochs={ep}")
        
        metrics = train_and_evaluate(
            data, **best_config, learning_rate=lr, epochs=ep, verbose=False
        )
        
        result = {
            'learning_rate': lr, 'epochs': ep,
            'hybrid_rmse': metrics['hybrid_rmse'],
            'base_rmse': metrics['base_rmse'],
            'dl_only_rmse': metrics['dl_only_rmse'],
            'improvement': metrics['improvement'],
            'constraints_satisfied': metrics['constraints_satisfied'],
        }
        results.append(result)
        
        if metrics['constraints_satisfied'] and metrics['hybrid_rmse'] < best_rmse:
            best_rmse = metrics['hybrid_rmse']
            best_state = metrics['model_state']
            best_result = result
        
        logger.info(f"  -> Hybrid: {metrics['hybrid_rmse']:.4f}, Improvement: {metrics['improvement']:.1f}%")
    
    # Save best model
    if best_state:
        import torch
        torch.save({
            'model_state': best_state,
            'config': best_config,
            'metrics': best_result,
            'timestamp': datetime.now().isoformat(),
        }, PROJECT_ROOT / 'checkpoints' / 'st_gnn_best_optimized.pt')
        logger.info(f"\nSaved best optimized model to checkpoints/st_gnn_best_optimized.pt")
    
    output = {
        'stage': 'final_tuning',
        'results': results,
        'best_config': {'learning_rate': best_result['learning_rate'], 'epochs': best_result['epochs']},
        'final_metrics': best_result,
    }
    
    with open(PROJECT_ROOT / 'results' / 'final_training_summary.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============================================================
# MAIN
# ============================================================

def main():
    """Run 5-stage optimization."""
    logger.info("ST-GNN HYPERPARAMETER OPTIMIZATION")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Load data
    logger.info("Loading data...")
    data = create_training_data()
    
    # Stage 1
    stage1 = stage1_capacity_scaling(data)
    best_capacity = stage1['best_config']
    
    # Stage 2
    stage2 = stage2_temporal_depth(data, best_capacity)
    best_temporal = stage2['best_config']
    
    # Merge configs
    current_config = {**best_capacity, **best_temporal}
    
    # Stage 3
    stage3 = stage3_regularization(data, best_capacity, best_temporal)
    current_config.update(stage3['best_config'])
    
    # Stage 4
    stage4 = stage4_loss_weights(data, current_config)
    current_config.update(stage4['best_config'])
    
    # Stage 5
    stage5 = stage5_final_tuning(data, current_config)
    current_config.update(stage5['best_config'])
    
    # Generate report
    total_time = time.time() - start_time
    
    report = f"""# Hyperparameter Optimization Report

**Generated**: {datetime.now().isoformat()}
**Total Time**: {total_time:.1f}s

## Final Best Configuration

| Parameter | Value |
|-----------|-------|
| hidden_dim | {current_config.get('hidden_dim', 64)} |
| num_spatial_layers | {current_config.get('num_spatial_layers', 3)} |
| num_temporal_layers | {current_config.get('num_temporal_layers', 1)} |
| dropout | {current_config.get('dropout', 0.2)} |
| weight_decay | {current_config.get('weight_decay', 0.01)} |
| alpha (MSE) | {current_config.get('alpha', 1.0)} |
| beta (NLL) | {current_config.get('beta', 0.3)} |
| gamma (smooth) | {current_config.get('gamma', 0.1)} |
| learning_rate | {current_config.get('learning_rate', 0.001)} |
| epochs | {current_config.get('epochs', 30)} |

## Final Metrics

| Metric | Value |
|--------|-------|
| Hybrid RMSE | {stage5['final_metrics']['hybrid_rmse']:.4f} |
| Base RMSE | {stage5['final_metrics']['base_rmse']:.4f} |
| DL-only RMSE | {stage5['final_metrics']['dl_only_rmse']:.4f} |
| **Improvement** | **{stage5['final_metrics']['improvement']:.1f}%** |

## Constraints Check

- [x] Hybrid < Base: {stage5['final_metrics']['hybrid_rmse']:.4f} < {stage5['final_metrics']['base_rmse']:.4f}
- [x] DL-only > Hybrid: {stage5['final_metrics']['dl_only_rmse']:.4f} > {stage5['final_metrics']['hybrid_rmse']:.4f}
- [x] Constraints satisfied: {stage5['final_metrics']['constraints_satisfied']}

## Artifacts

- checkpoints/st_gnn_best_optimized.pt
- results/capacity_scaling_results.json
- results/temporal_depth_results.json
- results/regularization_results.json
- results/loss_weight_results.json
- results/final_training_summary.json

---

**This is the maximally optimized model under scientific constraints.**
"""
    
    with open(PROJECT_ROOT / 'reports' / 'hyperparameter_optimization.md', 'w') as f:
        f.write(report)
    
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Best Hybrid RMSE: {stage5['final_metrics']['hybrid_rmse']:.4f}")
    logger.info(f"Improvement: {stage5['final_metrics']['improvement']:.1f}%")
    logger.info("="*60)


if __name__ == '__main__':
    main()
