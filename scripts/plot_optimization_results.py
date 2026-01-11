"""Generate visualization plots for hyperparameter optimization results.

Reads the JSON results from each optimization stage and creates:
1. Capacity scaling plot (RMSE vs hidden_dim/layers)
2. Regularization heatmap (dropout vs weight_decay)
3. Loss weights visualization
4. Final comparison summary
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'

# Import matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#1976D2', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']


def load_json(filename: str) -> dict:
    """Load JSON results file."""
    with open(RESULTS_DIR / filename, 'r') as f:
        return json.load(f)


def plot_capacity_scaling():
    """Plot RMSE vs model capacity (hidden_dim and num_layers)."""
    data = load_json('capacity_scaling_results.json')
    results = data['results']
    
    # Extract data
    hidden_dims = sorted(set(r['hidden_dim'] for r in results))
    num_layers = sorted(set(r['num_spatial_layers'] for r in results))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RMSE vs Hidden Dim (grouped by layers)
    ax1 = axes[0]
    for i, nl in enumerate(num_layers):
        layer_results = [r for r in results if r['num_spatial_layers'] == nl]
        hds = [r['hidden_dim'] for r in layer_results]
        rmses = [r['hybrid_rmse'] for r in layer_results]
        ax1.plot(hds, rmses, 'o-', color=COLORS[i], linewidth=2, markersize=8, 
                label=f'{nl} layers')
    
    ax1.set_xlabel('Hidden Dimension', fontsize=12)
    ax1.set_ylabel('Hybrid RMSE', fontsize=12)
    ax1.set_title('RMSE vs Model Capacity', fontsize=14, fontweight='bold')
    ax1.legend(title='Spatial Layers')
    ax1.set_xticks(hidden_dims)
    
    # Plot 2: Improvement % heatmap
    ax2 = axes[1]
    improvement_matrix = np.zeros((len(num_layers), len(hidden_dims)))
    
    for r in results:
        i = num_layers.index(r['num_spatial_layers'])
        j = hidden_dims.index(r['hidden_dim'])
        improvement_matrix[i, j] = r['improvement']
    
    im = ax2.imshow(improvement_matrix, cmap='YlGn', aspect='auto')
    ax2.set_xticks(range(len(hidden_dims)))
    ax2.set_xticklabels(hidden_dims)
    ax2.set_yticks(range(len(num_layers)))
    ax2.set_yticklabels(num_layers)
    ax2.set_xlabel('Hidden Dimension', fontsize=12)
    ax2.set_ylabel('Num Spatial Layers', fontsize=12)
    ax2.set_title('Improvement % Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(num_layers)):
        for j in range(len(hidden_dims)):
            text = ax2.text(j, i, f'{improvement_matrix[i, j]:.1f}%',
                           ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(im, ax=ax2, label='Improvement %')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'capacity_scaling_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved capacity_scaling_plot.png")


def plot_regularization():
    """Plot regularization effects (dropout vs weight_decay)."""
    data = load_json('regularization_results.json')
    results = data['results']
    
    dropouts = sorted(set(r['dropout'] for r in results))
    weight_decays = sorted(set(r['weight_decay'] for r in results))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RMSE heatmap
    ax1 = axes[0]
    rmse_matrix = np.zeros((len(dropouts), len(weight_decays)))
    
    for r in results:
        i = dropouts.index(r['dropout'])
        j = weight_decays.index(r['weight_decay'])
        rmse_matrix[i, j] = r['hybrid_rmse']
    
    im = ax1.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(range(len(weight_decays)))
    ax1.set_xticklabels([f'{wd}' for wd in weight_decays])
    ax1.set_yticks(range(len(dropouts)))
    ax1.set_yticklabels([f'{d}' for d in dropouts])
    ax1.set_xlabel('Weight Decay', fontsize=12)
    ax1.set_ylabel('Dropout', fontsize=12)
    ax1.set_title('Hybrid RMSE vs Regularization', fontsize=14, fontweight='bold')
    
    for i in range(len(dropouts)):
        for j in range(len(weight_decays)):
            ax1.text(j, i, f'{rmse_matrix[i, j]:.4f}',
                    ha='center', va='center', color='black', fontsize=9)
    
    plt.colorbar(im, ax=ax1, label='RMSE')
    
    # Plot 2: DL-only RMSE (sanity check - should stay high)
    ax2 = axes[1]
    dl_only_matrix = np.zeros((len(dropouts), len(weight_decays)))
    
    for r in results:
        i = dropouts.index(r['dropout'])
        j = weight_decays.index(r['weight_decay'])
        dl_only_matrix[i, j] = r['dl_only_rmse']
    
    im2 = ax2.imshow(dl_only_matrix, cmap='Reds', aspect='auto')
    ax2.set_xticks(range(len(weight_decays)))
    ax2.set_xticklabels([f'{wd}' for wd in weight_decays])
    ax2.set_yticks(range(len(dropouts)))
    ax2.set_yticklabels([f'{d}' for d in dropouts])
    ax2.set_xlabel('Weight Decay', fontsize=12)
    ax2.set_ylabel('Dropout', fontsize=12)
    ax2.set_title('DL-only RMSE (Sanity Check)', fontsize=14, fontweight='bold')
    
    for i in range(len(dropouts)):
        for j in range(len(weight_decays)):
            ax2.text(j, i, f'{dl_only_matrix[i, j]:.3f}',
                    ha='center', va='center', color='white', fontsize=9)
    
    plt.colorbar(im2, ax=ax2, label='RMSE')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'regularization_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved regularization_plot.png")


def plot_loss_weights():
    """Plot loss weight effects."""
    data = load_json('loss_weight_results.json')
    results = data['results']
    
    alphas = sorted(set(r['alpha'] for r in results))
    betas = sorted(set(r['beta'] for r in results))
    gammas = sorted(set(r['gamma'] for r in results))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot for each gamma value
    for g_idx, gamma in enumerate(gammas):
        ax = axes[g_idx]
        gamma_results = [r for r in results if r['gamma'] == gamma]
        
        rmse_matrix = np.zeros((len(betas), len(alphas)))
        for r in gamma_results:
            i = betas.index(r['beta'])
            j = alphas.index(r['alpha'])
            rmse_matrix[i, j] = r['hybrid_rmse']
        
        im = ax.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto', 
                       vmin=0.020, vmax=0.030)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f'{a}' for a in alphas])
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f'{b}' for b in betas])
        ax.set_xlabel('α (MSE weight)', fontsize=11)
        ax.set_ylabel('β (NLL weight)', fontsize=11)
        ax.set_title(f'γ = {gamma} (smoothness)', fontsize=12, fontweight='bold')
        
        for i in range(len(betas)):
            for j in range(len(alphas)):
                ax.text(j, i, f'{rmse_matrix[i, j]:.4f}',
                       ha='center', va='center', color='black', fontsize=8)
    
    plt.colorbar(im, ax=axes, label='Hybrid RMSE', shrink=0.8)
    fig.suptitle('Loss Weight Optimization', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'loss_weights_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved loss_weights_plot.png")


def plot_final_comparison():
    """Plot final comparison bar chart."""
    final = load_json('final_training_summary.json')
    results = final['results']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RMSE for each config
    ax1 = axes[0]
    labels = [f"lr={r['learning_rate']}\nep={r['epochs']}" for r in results]
    hybrid_rmses = [r['hybrid_rmse'] for r in results]
    colors = ['#4CAF50' if r == min(hybrid_rmses) else '#1976D2' for r in hybrid_rmses]
    
    bars = ax1.bar(labels, hybrid_rmses, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Hybrid RMSE', fontsize=12)
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_title('Final Tuning: RMSE by Configuration', fontsize=14, fontweight='bold')
    ax1.axhline(y=min(hybrid_rmses), color='red', linestyle='--', alpha=0.7, label='Best')
    ax1.legend()
    
    for bar, val in zip(bars, hybrid_rmses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Improvement % comparison
    ax2 = axes[1]
    improvements = [r['improvement'] for r in results]
    colors2 = ['#4CAF50' if r == max(improvements) else '#FF9800' for r in improvements]
    
    bars2 = ax2.bar(labels, improvements, color=colors2, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Improvement %', fontsize=12)
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_title('Final Tuning: Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax2.axhline(y=max(improvements), color='red', linestyle='--', alpha=0.7, label='Best')
    ax2.legend()
    
    for bar, val in zip(bars2, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'final_comparison_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved final_comparison_plot.png")


def plot_optimization_summary():
    """Create a comprehensive summary visualization."""
    # Load all stage results
    capacity = load_json('capacity_scaling_results.json')
    temporal = load_json('temporal_depth_results.json')
    reg = load_json('regularization_results.json')
    loss_w = load_json('loss_weight_results.json')
    final = load_json('final_training_summary.json')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Stage names and best RMSE for each
    stages = ['Baseline', 'Stage 1\n(Capacity)', 'Stage 2\n(Temporal)', 
              'Stage 3\n(Regularization)', 'Stage 4\n(Loss Weights)', 'Stage 5\n(Fine-tune)']
    
    baseline_rmse = 0.0296
    best_rmses = [
        baseline_rmse,
        capacity['best_hybrid_rmse'],
        min(r['hybrid_rmse'] for r in temporal['results']),
        min(r['hybrid_rmse'] for r in reg['results']),
        loss_w['best_hybrid_rmse'],
        final['final_metrics']['hybrid_rmse']
    ]
    
    # Colors: gradient from red to green
    colors = ['#F44336', '#FF9800', '#FFC107', '#CDDC39', '#8BC34A', '#4CAF50']
    
    bars = ax.bar(stages, best_rmses, color=colors, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Hybrid RMSE', fontsize=14)
    ax.set_xlabel('Optimization Stage', fontsize=14)
    ax.set_title('ST-GNN Hyperparameter Optimization Progress', fontsize=16, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, best_rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
               f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    improvement = (baseline_rmse - best_rmses[-1]) / baseline_rmse * 100
    ax.annotate(f'Total Improvement: {improvement:.1f}%', 
               xy=(5, best_rmses[-1]), xytext=(3.5, 0.026),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=14, fontweight='bold', color='green')
    
    ax.set_ylim(0, max(best_rmses) * 1.15)
    ax.axhline(y=baseline_rmse, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(5.5, baseline_rmse + 0.0005, 'Baseline', color='red', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'optimization_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved optimization_summary.png")


def main():
    """Generate all plots."""
    print("Generating hyperparameter optimization visualizations...")
    print("=" * 50)
    
    plot_capacity_scaling()
    plot_regularization()
    plot_loss_weights()
    plot_final_comparison()
    plot_optimization_summary()
    
    print("=" * 50)
    print("All plots saved to results/")


if __name__ == '__main__':
    main()
