"""
Baseline Methods and Comparative Analysis

This module implements TWO mandatory baseline methods for fair comparison:
- Baseline A: Rainfall Threshold Method (common municipal practice)
- Baseline B: Physics-Only Stress Model (deterministic, no uncertainty)

Both baselines use the SAME ROI, rainfall data, and DEM as the main system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import ndimage
from scipy.stats import spearmanr
import json
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.full_math_core_inference import FullMathCoreInferenceEngine, FullMathCoreConfig
from src.config.roi import PREDEFINED_ROIS

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# BASELINE A: RAINFALL THRESHOLD METHOD
# =============================================================================

@dataclass
class RainfallThresholdConfig:
    """Configuration for Rainfall Threshold baseline."""
    low_threshold: float = 5.0      # mm/hr - below this = LOW risk
    medium_threshold: float = 15.0  # mm/hr - below this = MEDIUM risk
    high_threshold: float = 30.0    # mm/hr - above this = HIGH risk


class BaselineA_RainfallThreshold:
    """
    BASELINE A — Rainfall Threshold Method
    
    Mathematical Formulation:
    -------------------------
    stress(x,y) = R(x,y)  where R = rainfall intensity
    
    risk_category(x,y) = 
        LOW     if R < 5 mm/hr
        MEDIUM  if 5 <= R < 15 mm/hr
        HIGH    if 15 <= R < 30 mm/hr
        EXTREME if R >= 30 mm/hr
    
    Information Ignored:
    -------------------
    - Terrain/DEM (slope, elevation)
    - Historical complaints
    - Upstream flow accumulation
    - Spatial autocorrelation
    - Uncertainty quantification
    - Temporal dynamics
    
    This represents common municipal practice where fixed rainfall
    thresholds trigger alerts regardless of local conditions.
    """
    
    def __init__(self, config: RainfallThresholdConfig = None):
        self.config = config or RainfallThresholdConfig()
        self.name = "Baseline A: Rainfall Threshold"
        
    def compute_stress(self, rainfall_intensity: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute stress using simple rainfall thresholds.
        
        Args:
            rainfall_intensity: (T, H, W) or (H, W) rainfall data
            
        Returns:
            Dictionary with stress, risk categories, and actions
        """
        # Handle temporal dimension
        if rainfall_intensity.ndim == 3:
            # Use maximum rainfall over time period
            rainfall = np.max(rainfall_intensity, axis=0)
        else:
            rainfall = rainfall_intensity
            
        # Stress is simply the rainfall intensity (normalized)
        stress = rainfall / self.config.high_threshold
        stress = np.clip(stress, 0, 2)  # Cap at 2x threshold
        
        # Risk categories based on fixed thresholds
        risk_category = np.zeros_like(rainfall, dtype=int)
        risk_category[rainfall >= self.config.low_threshold] = 1      # MEDIUM
        risk_category[rainfall >= self.config.medium_threshold] = 2   # HIGH
        risk_category[rainfall >= self.config.high_threshold] = 3     # EXTREME
        
        # Actions (no uncertainty, so direct mapping)
        actions = np.zeros_like(rainfall, dtype=int)
        actions[risk_category == 0] = 0  # NO_ACTION
        actions[risk_category == 1] = 1  # MONITOR
        actions[risk_category == 2] = 2  # ALERT
        actions[risk_category == 3] = 3  # DEPLOY_RESOURCES
        
        # No uncertainty in this baseline
        variance = np.zeros_like(rainfall)
        
        return {
            'stress': stress,
            'variance': variance,
            'risk_category': risk_category,
            'actions': actions,
            'no_decision_zones': np.zeros_like(rainfall, dtype=bool),  # Never uncertain
            'rainfall_used': rainfall
        }
    
    def get_formulation(self) -> str:
        """Return mathematical formulation."""
        return f"""
BASELINE A — Rainfall Threshold Method

Mathematical Formulation:
    stress(x,y) = R(x,y) / {self.config.high_threshold}
    
    risk_category(x,y) = 
        LOW (0)     if R < {self.config.low_threshold} mm/hr
        MEDIUM (1)  if {self.config.low_threshold} ≤ R < {self.config.medium_threshold} mm/hr
        HIGH (2)    if {self.config.medium_threshold} ≤ R < {self.config.high_threshold} mm/hr
        EXTREME (3) if R ≥ {self.config.high_threshold} mm/hr

Information IGNORED:
    ✗ Terrain/DEM (slope, elevation, flow direction)
    ✗ Historical drainage complaints
    ✗ Upstream flow accumulation
    ✗ Spatial autocorrelation
    ✗ Uncertainty quantification
    ✗ Temporal lag effects
"""


# =============================================================================
# BASELINE B: PHYSICS-ONLY STRESS MODEL
# =============================================================================

class BaselineB_PhysicsOnly:
    """
    BASELINE B — Physics-Only Stress Model
    
    Mathematical Formulation:
    -------------------------
    stress(x,y) = R(x,y) × S(x,y) × U(x,y)
    
    where:
        R(x,y) = rainfall intensity (mm/hr)
        S(x,y) = slope factor = 1 + tan(slope)/tan(45°)
        U(x,y) = upstream accumulation factor (normalized)
    
    Information Ignored:
    -------------------
    - Historical complaints (no learning from observations)
    - Uncertainty quantification (deterministic output)
    - Temporal dynamics (no lag/persistence modeling)
    - Spatial smoothing (no correlation structure)
    
    This represents a physics-based approach using terrain
    analysis without probabilistic/observational components.
    """
    
    def __init__(self, slope_weight: float = 0.5, upstream_weight: float = 0.3):
        self.slope_weight = slope_weight
        self.upstream_weight = upstream_weight
        self.name = "Baseline B: Physics-Only"
        
    def compute_stress(
        self, 
        rainfall_intensity: np.ndarray,
        dem: np.ndarray,
        upstream_contribution: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute stress using physics-based model.
        
        Args:
            rainfall_intensity: (T, H, W) or (H, W) rainfall data
            dem: (H, W) digital elevation model
            upstream_contribution: (H, W) upstream flow accumulation
            
        Returns:
            Dictionary with stress and derived quantities
        """
        # Handle temporal dimension
        if rainfall_intensity.ndim == 3:
            rainfall = np.max(rainfall_intensity, axis=0)
        else:
            rainfall = rainfall_intensity
            
        # Ensure DEM matches rainfall shape
        if dem.shape != rainfall.shape:
            from scipy.ndimage import zoom
            zoom_factors = (rainfall.shape[0] / dem.shape[0], 
                          rainfall.shape[1] / dem.shape[1])
            dem = zoom(dem, zoom_factors, order=1)
            
        # Compute slope from DEM
        dy, dx = np.gradient(dem)
        slope = np.sqrt(dx**2 + dy**2)
        slope_factor = 1 + slope / (np.tan(np.pi/4) + 1e-6)
        slope_factor = np.clip(slope_factor, 1, 3)
        
        # Normalize upstream contribution if provided
        if upstream_contribution is not None:
            if upstream_contribution.shape != rainfall.shape:
                upstream_contribution = zoom(upstream_contribution, zoom_factors, order=1)
            upstream_norm = upstream_contribution / (np.max(upstream_contribution) + 1e-6)
            upstream_factor = 1 + upstream_norm
        else:
            upstream_factor = np.ones_like(rainfall)
        
        # Physics-based stress computation (deterministic)
        rainfall_norm = rainfall / 30.0  # Normalize by typical heavy rain
        
        stress = rainfall_norm * (
            1.0 + 
            self.slope_weight * (slope_factor - 1) + 
            self.upstream_weight * (upstream_factor - 1)
        )
        
        # Deterministic risk categories
        risk_category = np.zeros_like(stress, dtype=int)
        risk_category[stress >= 0.25] = 1
        risk_category[stress >= 0.5] = 2
        risk_category[stress >= 1.0] = 3
        
        # Direct action mapping (no uncertainty consideration)
        actions = risk_category.copy()
        
        # No uncertainty or NO_DECISION zones (deterministic model)
        variance = np.zeros_like(stress)
        
        return {
            'stress': stress,
            'variance': variance,
            'risk_category': risk_category,
            'actions': actions,
            'no_decision_zones': np.zeros_like(stress, dtype=bool),
            'slope_factor': slope_factor,
            'upstream_factor': upstream_factor
        }
    
    def get_formulation(self) -> str:
        """Return mathematical formulation."""
        return f"""
BASELINE B — Physics-Only Stress Model

Mathematical Formulation:
    stress(x,y) = R_norm(x,y) × [1 + {self.slope_weight}×(S-1) + {self.upstream_weight}×(U-1)]
    
    where:
        R_norm = rainfall / 30 mm/hr (normalized intensity)
        S(x,y) = 1 + |∇DEM| / tan(45°)  (slope factor, range [1,3])
        U(x,y) = 1 + upstream_accum / max(upstream_accum)  (upstream factor)

Information IGNORED:
    ✗ Historical drainage complaints (no observational learning)
    ✗ Uncertainty quantification (deterministic output only)
    ✗ Temporal lag and persistence effects
    ✗ Spatial correlation/smoothing structure
    ✗ Data-driven calibration
"""


# =============================================================================
# COMPARATIVE ANALYSIS
# =============================================================================

class BaselineComparison:
    """
    Comparative analysis between baselines and main system.
    """
    
    def __init__(self, output_dir: str = "outputs/baseline_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_a = BaselineA_RainfallThreshold()
        self.baseline_b = BaselineB_PhysicsOnly()
        self.roi_config = PREDEFINED_ROIS["seattle"]
        
        # Grid setup
        self.grid_shape = (10, 50, 50)  # T, H, W
        
    def _create_test_data(self, rainfall_value: float = 20.0):
        """Create test data for comparison."""
        t, h, w = self.grid_shape
        
        # Spatially varying rainfall (same for all methods)
        x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        spatial_var = 1.0 + 0.3 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
        
        rainfall_intensity = np.zeros((t, h, w), dtype=np.float32)
        for ti in range(t):
            # Add temporal variation
            temporal_factor = 1.0 + 0.5 * np.sin(2*np.pi*ti/t)
            rainfall_intensity[ti] = rainfall_value * spatial_var * temporal_factor
        
        rainfall_accumulation = np.cumsum(rainfall_intensity, axis=0)
        
        # DEM with realistic terrain
        dem = 50 + 30 * np.sin(2*np.pi*x) * np.cos(np.pi*y) + 10 * np.random.randn(h, w) * 0.1
        
        # Upstream contribution
        upstream = (100 * np.exp(-((x-0.3)**2 + (y-0.7)**2)/0.1) +
                   80 * np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.15))
        
        # Complaints (only used by main system)
        complaints = np.zeros((t, h, w), dtype=np.float32)
        np.random.seed(42)
        for _ in range(100):
            ti = np.random.randint(0, t)
            hi = np.random.randint(0, h)
            wi = np.random.randint(0, w)
            complaints[ti, hi, wi] += 1
        for ti in range(t):
            complaints[ti] = ndimage.gaussian_filter(complaints[ti], sigma=1.0)
        
        return {
            'rainfall_intensity': rainfall_intensity.astype(np.float32),
            'rainfall_accumulation': rainfall_accumulation.astype(np.float32),
            'upstream_contribution': upstream.astype(np.float32),
            'dem': dem.astype(np.float32),
            'complaints': complaints.astype(np.float32),
        }
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run full comparison between all methods."""
        print("=" * 60)
        print("BASELINE METHODS AND COMPARATIVE ANALYSIS")
        print("=" * 60)
        
        # Create test data (SAME for all methods)
        data = self._create_test_data(rainfall_value=20.0)
        
        results = {}
        
        # =====================================================================
        # Run Baseline A
        # =====================================================================
        print("\n" + "-" * 60)
        print("BASELINE A: Rainfall Threshold Method")
        print("-" * 60)
        print(self.baseline_a.get_formulation())
        
        result_a = self.baseline_a.compute_stress(data['rainfall_intensity'])
        results['baseline_a'] = result_a
        
        # =====================================================================
        # Run Baseline B
        # =====================================================================
        print("\n" + "-" * 60)
        print("BASELINE B: Physics-Only Stress Model")
        print("-" * 60)
        print(self.baseline_b.get_formulation())
        
        result_b = self.baseline_b.compute_stress(
            data['rainfall_intensity'],
            data['dem'],
            data['upstream_contribution']
        )
        results['baseline_b'] = result_b
        
        # =====================================================================
        # Run Main System (Proposed Method)
        # =====================================================================
        print("\n" + "-" * 60)
        print("PROPOSED SYSTEM: Full Math-Core Inference")
        print("-" * 60)
        
        engine = FullMathCoreInferenceEngine(FullMathCoreConfig(target_cv=0.2))
        result_main = engine.infer(
            rainfall_intensity=data['rainfall_intensity'],
            rainfall_accumulation=data['rainfall_accumulation'],
            upstream_contribution=data['upstream_contribution'],
            complaints=data['complaints'],
            dem=data['dem']
        )
        
        # Extract comparable outputs
        results['main_system'] = {
            'stress': result_main.posterior_mean[-1],  # Final timestep
            'variance': result_main.posterior_variance[-1],
            'temporal_mean_stress': np.mean(result_main.posterior_mean, axis=0),
        }
        
        # =====================================================================
        # Generate Comparison Outputs
        # =====================================================================
        self._generate_comparison_plots(results, data)
        self._generate_comparison_report(results, data)
        
        return results
    
    def _generate_comparison_plots(self, results: Dict, data: Dict):
        """Generate comparison visualizations."""
        
        # =================================================================
        # PLOT 1: Spatial Stress Patterns Comparison
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Baseline A
        ax1 = axes[0, 0]
        im1 = ax1.imshow(results['baseline_a']['stress'], cmap='YlOrRd', 
                        origin='lower', aspect='auto', vmin=0, vmax=2)
        ax1.set_title('Baseline A: Rainfall Threshold\n(Ignores terrain, complaints, uncertainty)', 
                     fontsize=11)
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')
        plt.colorbar(im1, ax=ax1, label='Stress')
        
        # Baseline B
        ax2 = axes[0, 1]
        im2 = ax2.imshow(results['baseline_b']['stress'], cmap='YlOrRd', 
                        origin='lower', aspect='auto', vmin=0, vmax=2)
        ax2.set_title('Baseline B: Physics-Only\n(Ignores complaints, uncertainty)', fontsize=11)
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')
        plt.colorbar(im2, ax=ax2, label='Stress')
        
        # Main System
        ax3 = axes[1, 0]
        im3 = ax3.imshow(results['main_system']['stress'], cmap='YlOrRd', 
                        origin='lower', aspect='auto', vmin=0, vmax=2)
        ax3.set_title('Proposed System: Full Math-Core\n(Uses all information + uncertainty)', 
                     fontsize=11)
        ax3.set_xlabel('Grid X')
        ax3.set_ylabel('Grid Y')
        plt.colorbar(im3, ax=ax3, label='Stress')
        
        # Difference map
        ax4 = axes[1, 1]
        diff = results['main_system']['stress'] - results['baseline_a']['stress']
        im4 = ax4.imshow(diff, cmap='RdBu_r', origin='lower', aspect='auto',
                        vmin=-1, vmax=1)
        ax4.set_title('Difference: Main System - Baseline A\n(Red=higher, Blue=lower)', fontsize=11)
        ax4.set_xlabel('Grid X')
        ax4.set_ylabel('Grid Y')
        plt.colorbar(im4, ax=ax4, label='Stress Difference')
        
        plt.suptitle('COMPARISON 1: Spatial Stress Patterns', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_1_spatial_patterns.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # =================================================================
        # PLOT 2: High Risk Areas Comparison
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Define high risk threshold
        high_risk_threshold = 0.7
        
        high_risk_a = results['baseline_a']['stress'] > high_risk_threshold
        high_risk_b = results['baseline_b']['stress'] > high_risk_threshold
        high_risk_main = results['main_system']['stress'] > high_risk_threshold
        
        ax1 = axes[0, 0]
        ax1.imshow(high_risk_a.astype(float), cmap='Reds', origin='lower', aspect='auto')
        ax1.set_title(f'Baseline A: High Risk Areas\n({np.sum(high_risk_a)} cells = {100*np.mean(high_risk_a):.1f}%)', 
                     fontsize=11)
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')
        
        ax2 = axes[0, 1]
        ax2.imshow(high_risk_b.astype(float), cmap='Reds', origin='lower', aspect='auto')
        ax2.set_title(f'Baseline B: High Risk Areas\n({np.sum(high_risk_b)} cells = {100*np.mean(high_risk_b):.1f}%)', 
                     fontsize=11)
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')
        
        ax3 = axes[1, 0]
        ax3.imshow(high_risk_main.astype(float), cmap='Reds', origin='lower', aspect='auto')
        ax3.set_title(f'Proposed System: High Risk Areas\n({np.sum(high_risk_main)} cells = {100*np.mean(high_risk_main):.1f}%)', 
                     fontsize=11)
        ax3.set_xlabel('Grid X')
        ax3.set_ylabel('Grid Y')
        
        # Overlap visualization
        ax4 = axes[1, 1]
        overlap = np.zeros(high_risk_a.shape + (3,))
        overlap[high_risk_a, 0] = 1.0  # Red for Baseline A
        overlap[high_risk_b, 1] = 1.0  # Green for Baseline B  
        overlap[high_risk_main, 2] = 1.0  # Blue for Main
        ax4.imshow(overlap, origin='lower', aspect='auto')
        ax4.set_title('High Risk Overlap\n(R=BaselineA, G=BaselineB, B=Proposed)', fontsize=11)
        ax4.set_xlabel('Grid X')
        ax4.set_ylabel('Grid Y')
        
        plt.suptitle('COMPARISON 2: Areas Flagged as High Risk', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_2_high_risk_areas.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # =================================================================
        # PLOT 3: NO_DECISION Zones (Uncertainty)
        # =================================================================
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1 = axes[0]
        ax1.imshow(results['baseline_a']['no_decision_zones'].astype(float), 
                  cmap='Purples', origin='lower', aspect='auto')
        ax1.set_title('Baseline A: NO_DECISION Zones\n(None - no uncertainty model)', fontsize=11)
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')
        
        ax2 = axes[1]
        ax2.imshow(results['baseline_b']['no_decision_zones'].astype(float), 
                  cmap='Purples', origin='lower', aspect='auto')
        ax2.set_title('Baseline B: NO_DECISION Zones\n(None - deterministic model)', fontsize=11)
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')
        
        ax3 = axes[2]
        # Main system has uncertainty - compute NO_DECISION zones
        variance = results['main_system']['variance']
        stress = results['main_system']['stress']
        cv = np.sqrt(variance) / (stress + 1e-6)
        no_decision = cv > 0.5  # High uncertainty = no decision
        
        im3 = ax3.imshow(no_decision.astype(float), cmap='Purples', origin='lower', aspect='auto')
        ax3.set_title(f'Proposed System: NO_DECISION Zones\n({np.sum(no_decision)} cells = {100*np.mean(no_decision):.1f}%)', 
                     fontsize=11)
        ax3.set_xlabel('Grid X')
        ax3.set_ylabel('Grid Y')
        plt.colorbar(im3, ax=ax3, label='NO_DECISION')
        
        plt.suptitle('COMPARISON 3: Presence of NO_DECISION Zones\n'
                    '(Only proposed system can express uncertainty)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_3_no_decision_zones.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # =================================================================
        # PLOT 4: Stability Under Perturbations
        # =================================================================
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Add small perturbation to rainfall
        perturbation = np.random.randn(*data['rainfall_intensity'].shape) * 2.0
        perturbed_rainfall = data['rainfall_intensity'] + perturbation
        
        # Re-run all methods with perturbed data
        result_a_pert = self.baseline_a.compute_stress(perturbed_rainfall)
        result_b_pert = self.baseline_b.compute_stress(
            perturbed_rainfall, data['dem'], data['upstream_contribution']
        )
        
        engine = FullMathCoreInferenceEngine(FullMathCoreConfig(target_cv=0.2))
        result_main_pert = engine.infer(
            rainfall_intensity=perturbed_rainfall.astype(np.float32),
            rainfall_accumulation=np.cumsum(perturbed_rainfall, axis=0).astype(np.float32),
            upstream_contribution=data['upstream_contribution'],
            complaints=data['complaints'],
            dem=data['dem']
        )
        
        # Compute stability (change in stress)
        change_a = np.abs(result_a_pert['stress'] - results['baseline_a']['stress'])
        change_b = np.abs(result_b_pert['stress'] - results['baseline_b']['stress'])
        change_main = np.abs(result_main_pert.posterior_mean[-1] - results['main_system']['stress'])
        
        # Top row: Original
        axes[0, 0].imshow(results['baseline_a']['stress'], cmap='YlOrRd', origin='lower', aspect='auto')
        axes[0, 0].set_title('Baseline A: Original', fontsize=11)
        
        axes[0, 1].imshow(results['baseline_b']['stress'], cmap='YlOrRd', origin='lower', aspect='auto')
        axes[0, 1].set_title('Baseline B: Original', fontsize=11)
        
        axes[0, 2].imshow(results['main_system']['stress'], cmap='YlOrRd', origin='lower', aspect='auto')
        axes[0, 2].set_title('Proposed: Original', fontsize=11)
        
        # Bottom row: Change magnitude
        im1 = axes[1, 0].imshow(change_a, cmap='hot', origin='lower', aspect='auto', vmin=0, vmax=0.5)
        axes[1, 0].set_title(f'Baseline A: Change\nMean={np.mean(change_a):.4f}', fontsize=11)
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(change_b, cmap='hot', origin='lower', aspect='auto', vmin=0, vmax=0.5)
        axes[1, 1].set_title(f'Baseline B: Change\nMean={np.mean(change_b):.4f}', fontsize=11)
        plt.colorbar(im2, ax=axes[1, 1])
        
        im3 = axes[1, 2].imshow(change_main, cmap='hot', origin='lower', aspect='auto', vmin=0, vmax=0.5)
        axes[1, 2].set_title(f'Proposed: Change\nMean={np.mean(change_main):.4f}', fontsize=11)
        plt.colorbar(im3, ax=axes[1, 2])
        
        plt.suptitle('COMPARISON 4: Stability Under Small Perturbations\n'
                    '(±2 mm/hr random noise added to rainfall)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_4_stability.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # =================================================================
        # SUMMARY PLOT
        # =================================================================
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create summary table
        methods = ['Baseline A\n(Rainfall Threshold)', 'Baseline B\n(Physics-Only)', 'Proposed System\n(Full Math-Core)']
        
        metrics = {
            'Uses Terrain': [0, 1, 1],
            'Uses Complaints': [0, 0, 1],
            'Has Uncertainty': [0, 0, 1],
            'Has NO_DECISION': [0, 0, 1],
            'Temporal Modeling': [0, 0, 1],
            'Spatial Smoothing': [0, 0, 1],
        }
        
        n_methods = len(methods)
        n_metrics = len(metrics)
        
        # Create heatmap-style visualization
        data_matrix = np.array(list(metrics.values()))
        
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, fontsize=11)
        ax.set_yticks(range(n_metrics))
        ax.set_yticklabels(list(metrics.keys()), fontsize=11)
        
        # Add text annotations
        for i in range(n_metrics):
            for j in range(n_methods):
                text = '✓' if data_matrix[i, j] else '✗'
                color = 'white' if data_matrix[i, j] else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=16, 
                       fontweight='bold', color=color)
        
        ax.set_title('SUMMARY: Method Capabilities Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Saved comparison plots to: {self.output_dir}/")
        
    def _generate_comparison_report(self, results: Dict, data: Dict):
        """Generate detailed comparison report."""
        
        # Compute comparison metrics
        stress_a = results['baseline_a']['stress']
        stress_b = results['baseline_b']['stress']
        stress_main = results['main_system']['stress']
        
        # Spatial correlation between methods
        corr_a_main = np.corrcoef(stress_a.flatten(), stress_main.flatten())[0, 1]
        corr_b_main = np.corrcoef(stress_b.flatten(), stress_main.flatten())[0, 1]
        corr_a_b = np.corrcoef(stress_a.flatten(), stress_b.flatten())[0, 1]
        
        # High risk area overlap
        threshold = 0.7
        high_a = stress_a > threshold
        high_b = stress_b > threshold
        high_main = stress_main > threshold
        
        overlap_a_main = np.sum(high_a & high_main) / (np.sum(high_a | high_main) + 1e-6)
        overlap_b_main = np.sum(high_b & high_main) / (np.sum(high_b | high_main) + 1e-6)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "title": "Baseline Methods and Comparative Analysis",
            
            "baseline_a": {
                "name": "Rainfall Threshold Method",
                "formulation": "stress = R / 30, thresholds at 5, 15, 30 mm/hr",
                "ignores": ["terrain", "complaints", "uncertainty", "temporal_dynamics"],
                "mean_stress": float(np.mean(stress_a)),
                "std_stress": float(np.std(stress_a)),
                "high_risk_fraction": float(np.mean(high_a)),
                "failure_modes": [
                    "Cannot distinguish low-lying vs elevated areas",
                    "No learning from historical complaints",
                    "Overconfident (no uncertainty)",
                    "Ignores upstream accumulation effects"
                ]
            },
            
            "baseline_b": {
                "name": "Physics-Only Stress Model",
                "formulation": "stress = R_norm × [1 + 0.5×(S-1) + 0.3×(U-1)]",
                "ignores": ["complaints", "uncertainty", "temporal_dynamics"],
                "mean_stress": float(np.mean(stress_b)),
                "std_stress": float(np.std(stress_b)),
                "high_risk_fraction": float(np.mean(high_b)),
                "failure_modes": [
                    "No learning from observations",
                    "Deterministic (no uncertainty)",
                    "No temporal lag modeling",
                    "May miss historically problematic areas"
                ]
            },
            
            "proposed_system": {
                "name": "Full Math-Core Inference",
                "uses": ["rainfall", "terrain", "complaints", "upstream", "uncertainty", "temporal"],
                "mean_stress": float(np.mean(stress_main)),
                "std_stress": float(np.std(stress_main)),
                "high_risk_fraction": float(np.mean(high_main)),
                "has_no_decision_zones": True,
                "advantages": [
                    "Incorporates all available information",
                    "Quantifies uncertainty honestly",
                    "Can express NO_DECISION when uncertain",
                    "Learns from historical complaints",
                    "Models temporal dynamics and persistence"
                ]
            },
            
            "comparisons": {
                "spatial_correlation": {
                    "baseline_a_vs_main": float(corr_a_main),
                    "baseline_b_vs_main": float(corr_b_main),
                    "baseline_a_vs_b": float(corr_a_b)
                },
                "high_risk_overlap_jaccard": {
                    "baseline_a_vs_main": float(overlap_a_main),
                    "baseline_b_vs_main": float(overlap_b_main)
                }
            },
            
            "conclusion": {
                "baseline_a_limitation": "Ignores spatial context, overconfident",
                "baseline_b_limitation": "No observational learning, deterministic",
                "proposed_advantage": "Integrates all information with honest uncertainty"
            }
        }
        
        # Save report
        report_path = self.output_dir / "baseline_comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        print("\n  SPATIAL CORRELATIONS:")
        print(f"    Baseline A vs Main: {corr_a_main:.4f}")
        print(f"    Baseline B vs Main: {corr_b_main:.4f}")
        print(f"    Baseline A vs B:    {corr_a_b:.4f}")
        
        print("\n  HIGH RISK AREA OVERLAP (Jaccard):")
        print(f"    Baseline A vs Main: {overlap_a_main:.4f}")
        print(f"    Baseline B vs Main: {overlap_b_main:.4f}")
        
        print("\n  WHERE BASELINES FAIL:")
        print("    Baseline A: Cannot use terrain, no uncertainty, no learning")
        print("    Baseline B: No observational learning, overconfident")
        print("    → Proposed system addresses ALL these limitations")
        
        print(f"\n  Report saved: {report_path}")


def main():
    """Run baseline comparison analysis."""
    print("\n" + "=" * 60)
    print("Starting Baseline Comparison Analysis")
    print("=" * 60)
    
    comparison = BaselineComparison()
    results = comparison.run_comparison()
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print("\nOutputs saved to: outputs/baseline_comparison/")
    print("Files generated:")
    print("  - comparison_1_spatial_patterns.png")
    print("  - comparison_2_high_risk_areas.png")
    print("  - comparison_3_no_decision_zones.png")
    print("  - comparison_4_stability.png")
    print("  - comparison_summary.png")
    print("  - baseline_comparison_report.json")


if __name__ == "__main__":
    main()
