"""
Advanced Visualization Module for ST-GNN.
Generates "Max-Level", publication-quality graphs from training history.
"""

from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class AdvancedVisualizer:
    """Generates robust, complex graphs for model analysis."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Set premium style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("deep")

    def generate_training_report(self, history: Dict, run_id: str = "latest"):
        """
        Generates 4 key graphs:
        1. Learning Dynamics (Loss + RMSE dual axis)
        2. Generalization Gap (Val - Train)
        3. Convergence Velocity (First derivative of loss)
        4. Stability Analysis (Rolling Std Dev)
        """
        df = pd.DataFrame(history)
        epochs = df['epoch']
        
        # Create a 2x2 Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
        fig.suptitle(f"MAX-LEVEL TRAINING ANALYSIS: Run {run_id}", fontsize=16, fontweight='bold', y=0.95)
        
        # --- Graph 1: Learning Dynamics (Dual Axis) ---
        ax1 = axes[0, 0]
        sns.lineplot(data=df, x='epoch', y='train_loss', ax=ax1, label='Train Loss', color='#2ecc71', linewidth=2)
        sns.lineplot(data=df, x='epoch', y='val_loss', ax=ax1, label='Val Loss', color='#e74c3c', linewidth=2)
        ax1.set_title("Learning Dynamics (Loss)", fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Annotate Best Epoch
        best_epoch = df.loc[df['val_loss'].idxmin()]
        ax1.axvline(best_epoch['epoch'], color='gold', linestyle='--', alpha=0.8)
        ax1.text(best_epoch['epoch'], best_epoch['val_loss'], f" Best: {best_epoch['val_loss']:.4f}", 
                 color='black', fontweight='bold', ha='left', va='bottom')

        # --- Graph 2: Generalization Gap ---
        ax2 = axes[0, 1]
        gap = df['val_loss'] - df['train_loss']
        # Fill area
        ax2.fill_between(epochs, gap, color='#3498db', alpha=0.3)
        ax2.plot(epochs, gap, color='#2980b9', linewidth=2)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_title("Generalization Gap (Overfitting Check)", fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val Loss - Train Loss")
        ax2.grid(True, alpha=0.3)
        
        if gap.iloc[-1] > 0.01:
             ax2.text(0.5, 0.9, "⚠️ Possible Overfitting", transform=ax2.transAxes, 
                      color='red', fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))

        # --- Graph 3: Convergence Velocity (Rate of Change) ---
        ax3 = axes[1, 0]
        velocity = np.gradient(df['val_loss'])
        ax3.plot(epochs, velocity, color='#9b59b6', linewidth=2)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title("Convergence Velocity (dLoss/dEpoch)", fontweight='bold')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Rate of Change")
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(epochs, velocity, 0, where=(velocity < 0), color='green', alpha=0.1, label='Improving')
        ax3.fill_between(epochs, velocity, 0, where=(velocity > 0), color='red', alpha=0.1, label='Degrading')
        ax3.legend()

        # --- Graph 4: Stability (Rolling Std Dev) ---
        ax4 = axes[1, 1]
        window = max(3, len(df)//10)
        stability = df['val_loss'].rolling(window=window).std()
        ax4.plot(epochs, stability, color='#f39c12', linewidth=2)
        ax4.set_title(f"Stability Analysis (Rolling StdDev, w={window})", fontweight='bold')
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Volatility")
        ax4.grid(True, alpha=0.3)
        
        # Save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = self.save_dir / f"training_analysis_{run_id}.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
