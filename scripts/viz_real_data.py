"""Visualize Best ST-GNN Model on REAL Data.

Loads:
1. Best optimized checkpoint (st_gnn_best_optimized.pt)
2. Real data (stress_map.geojson, etc.)

Generates:
- Residual maps
- Uncertainty maps
- Prediction scatter plots
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Create results dir if not exists
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

from scripts.train_real_data import load_real_data
from scripts.hyperparameter_optimization import build_model

def main():
    print("Loading Real Data...")
    data = load_real_data()
    
    print("Loading Best Model...")
    checkpoint = torch.load(PROJECT_ROOT / 'checkpoints' / 'st_gnn_best_optimized.pt')
    config = checkpoint['config']
    
    # Rebuild model structure
    model = build_model(
        hidden_dim=config.get('hidden_dim', 64),
        num_spatial_layers=config.get('num_spatial_layers', 3),
        num_temporal_layers=config.get('num_temporal_layers', 1),
        dropout=config.get('dropout', 0.2)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print("Running Inference...")
    # Prepare input
    # Use last 30% as "test" set equivalent
    split_idx = int(data['num_timesteps'] * 0.7)
    test_x = torch.tensor(data['node_features'][split_idx:], dtype=torch.float32)
    
    with torch.no_grad():
        pred_residual, pred_uncertainty = model(test_x)
        pred_residual = pred_residual.squeeze(-1).numpy()
        pred_uncertainty = pred_uncertainty.squeeze(-1).numpy()
        
    # Get ground truth for same period
    target_residual = data['target_residual'][split_idx:]
    coords = data['coords']
    
    print("Generating Plots...")
    
    # 1. Prediction vs True Scatter
    plt.figure(figsize=(10, 8))
    # Sample points for scatter to avoid overcrowding
    sample_mask = np.random.choice(target_residual.size, 5000, replace=False)
    flat_target = target_residual.flatten()
    flat_pred = pred_residual.flatten()
    
    plt.scatter(flat_target[sample_mask], flat_pred[sample_mask], alpha=0.3, s=10, c='blue')
    min_val = min(flat_target.min(), flat_pred.min())
    max_val = max(flat_target.max(), flat_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True Residual')
    plt.ylabel('Predicted Residual')
    plt.title('ST-GNN Performance on Real Data (Zero-shot)')
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / 'real_data_scatter.png', dpi=150)
    plt.close()
    
    # 2. Residual Map (Use last timestep)
    t = -1 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True Residual
    sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=target_residual[t], 
                         cmap='RdBu_r', s=5, vmin=-0.1, vmax=0.1)
    axes[0].set_title('True Residual (Real Data)')
    plt.colorbar(sc1, ax=axes[0])
    
    # Predicted Residual
    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=pred_residual[t], 
                         cmap='RdBu_r', s=5, vmin=-0.1, vmax=0.1)
    axes[1].set_title('Predicted Residual (ST-GNN)')
    plt.colorbar(sc2, ax=axes[1])
    
    # Uncertainty
    sc3 = axes[2].scatter(coords[:, 0], coords[:, 1], c=pred_uncertainty[t], 
                         cmap='Purples', s=5)
    axes[2].set_title('Model Uncertainty')
    plt.colorbar(sc3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'real_data_maps.png', dpi=150)
    plt.close()
    
    print("Done! Saved to results/real_data_scatter.png and results/real_data_maps.png")

if __name__ == '__main__':
    main()
