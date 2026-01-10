"""
Run Full Model Comparison: Baseline vs 17-Module Math Core

This script:
1. Loads data
2. Runs baseline model (0 modules)
3. Runs new model with ALL 17 math_core modules
4. Generates comprehensive comparison graphs
5. Prints detailed report
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.comparison.model_comparison import (
    BaselineModel,
    compute_metrics,
    generate_comparison_graphs,
    print_comparison_report,
)
from src.inference.full_math_core_inference import (
    FullMathCoreInferenceEngine,
    FullMathCoreConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_or_generate_data(data_dir: Path):
    """Load real data or generate synthetic test data."""
    
    # Try to load real preprocessed data
    preprocessed_dir = data_dir / "preprocessed"
    
    if preprocessed_dir.exists():
        try:
            # Look for any .npy files
            npy_files = list(preprocessed_dir.glob("*.npy"))
            if npy_files:
                logger.info("Loading preprocessed data from %s", preprocessed_dir)
                
                # Load what's available
                data = {}
                for f in npy_files:
                    key = f.stem
                    data[key] = np.load(f)
                    logger.info("  Loaded %s: shape %s", key, data[key].shape)
                
                # Return if we have the essentials
                if 'rainfall_intensity' in data:
                    return data
        except Exception as e:
            logger.warning("Failed to load preprocessed data: %s", e)
    
    # Generate synthetic data for testing
    logger.info("Generating synthetic test data...")
    
    np.random.seed(42)
    t, h, w = 13, 50, 50  # 13 timesteps, 50x50 grid
    
    # Synthetic rainfall (event pattern)
    rainfall_intensity = np.zeros((t, h, w))
    for ti in range(t):
        if 2 <= ti <= 8:  # Event period
            intensity = np.sin((ti - 2) * np.pi / 6) * 20
            rainfall_intensity[ti] = intensity + np.random.randn(h, w) * 2
    rainfall_intensity = np.maximum(rainfall_intensity, 0)
    
    # Rainfall accumulation
    rainfall_accumulation = np.cumsum(rainfall_intensity, axis=0)
    
    # Upstream contribution (drainage network pattern)
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    upstream_contribution = (
        100 * np.exp(-((x - 0.3)**2 + (y - 0.7)**2) / 0.1) +
        80 * np.exp(-((x - 0.7)**2 + (y - 0.3)**2) / 0.15) +
        60 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.2)
    )
    
    # Complaints (correlated with rain and upstream)
    complaints = np.zeros((t, h, w))
    for ti in range(t):
        if rainfall_intensity[ti].max() > 5:
            # Complaints lag rainfall by 1-2 timesteps
            lag = min(ti, 2)
            complaints[ti] = (
                0.3 * rainfall_intensity[max(0, ti-lag)] +
                0.5 * upstream_contribution / upstream_contribution.max() * rainfall_intensity[ti].mean()
            )
            complaints[ti] += np.random.randn(h, w) * 0.5
    complaints = np.maximum(complaints, 0)
    
    # DEM (topography)
    dem = 100 - 50 * x - 30 * y + 10 * np.sin(5 * x) * np.cos(5 * y)
    
    return {
        'rainfall_intensity': rainfall_intensity,
        'rainfall_accumulation': rainfall_accumulation,
        'upstream_contribution': upstream_contribution,
        'complaints': complaints,
        'dem': dem,
    }


def main():
    """Run full model comparison."""
    
    print("=" * 70)
    print("FULL MODEL COMPARISON")
    print("Baseline (0 modules) vs New Model (ALL 17 Math Core Modules)")
    print("=" * 70)
    print()
    
    # Setup paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data = load_or_generate_data(data_dir)
    
    rainfall_intensity = data['rainfall_intensity']
    rainfall_accumulation = data['rainfall_accumulation']
    upstream_contribution = data['upstream_contribution']
    complaints = data['complaints']
    dem = data.get('dem', None)
    
    t, h, w = rainfall_intensity.shape
    logger.info("Data shape: T=%d, H=%d, W=%d", t, h, w)
    
    # =========================================================================
    # RUN BASELINE MODEL (0 modules)
    # =========================================================================
    print()
    print("=" * 70)
    print("RUNNING BASELINE MODEL (0 Math Core Modules)")
    print("=" * 70)
    
    baseline_model = BaselineModel()
    baseline_outputs = baseline_model.run(
        rainfall_intensity=rainfall_intensity,
        rainfall_accumulation=rainfall_accumulation,
        upstream_contribution=upstream_contribution,
        complaints=complaints,
    )
    
    baseline_metrics = compute_metrics(
        name="Baseline",
        outputs=baseline_outputs,
        observations=complaints,
        n_modules=0,
        modules_list="None",
    )
    
    print(f"\nBaseline Results:")
    print(f"  Uncertainty CV: {baseline_metrics.uncertainty_cv:.4f} {'(FAIL)' if baseline_metrics.uncertainty_cv < 0.15 else '(PASS)'}")
    print(f"  Scientific Valid: {'NO' if not baseline_metrics.overall_valid else 'YES'}")
    
    # =========================================================================
    # RUN NEW MODEL (ALL 17 modules)
    # =========================================================================
    print()
    print("=" * 70)
    print("RUNNING NEW MODEL (ALL 17 Math Core Modules)")
    print("=" * 70)
    
    config = FullMathCoreConfig(
        target_cv=0.2,
        min_variance=0.01,
        max_variance=10.0,
    )
    
    new_engine = FullMathCoreInferenceEngine(config=config)
    result = new_engine.infer(
        rainfall_intensity=rainfall_intensity,
        rainfall_accumulation=rainfall_accumulation,
        upstream_contribution=upstream_contribution,
        complaints=complaints,
        dem=dem,
    )
    
    # Convert result to dict for metrics
    new_outputs = {
        'posterior_mean': result.posterior_mean,
        'posterior_variance': result.posterior_variance,
        'epistemic_uncertainty': result.epistemic_uncertainty,
        'aleatoric_uncertainty': result.aleatoric_uncertainty,
        'total_uncertainty': result.total_uncertainty,
        'information_gain': result.information_gain,
        'reducibility_fraction': result.reducibility_fraction,
        'cvar': result.cvar,
        'expected_utility': result.expected_utility,
    }
    
    stability_info = {
        'lipschitz_constant': result.lipschitz_constant,
        'spectral_radius': result.spectral_radius,
        'condition_number': result.condition_number,
        'is_stable': result.is_stable,
    }
    
    new_metrics = compute_metrics(
        name="New (17 modules)",
        outputs=new_outputs,
        observations=complaints,
        n_modules=len(result.modules_used),
        modules_list=", ".join(result.modules_used),
        stability_info=stability_info,
    )
    
    print(f"\nNew Model Results:")
    print(f"  Uncertainty CV: {new_metrics.uncertainty_cv:.4f} {'(PASS)' if new_metrics.uncertainty_cv >= 0.15 else '(FAIL)'}")
    print(f"  Scientific Valid: {'YES' if new_metrics.overall_valid else 'NO'}")
    print(f"  Modules Used: {new_metrics.n_modules}/17")
    
    # =========================================================================
    # GENERATE COMPARISON
    # =========================================================================
    print()
    print("=" * 70)
    print("GENERATING COMPARISON GRAPHS")
    print("=" * 70)
    
    graphs = generate_comparison_graphs(
        baseline_metrics=baseline_metrics,
        new_metrics=new_metrics,
        output_dir=output_dir,
    )
    
    print(f"\nGenerated {len(graphs)} comparison graphs:")
    for name, path in graphs.items():
        print(f"  • {name}: {path}")
    
    # =========================================================================
    # PRINT DETAILED REPORT
    # =========================================================================
    print()
    report = print_comparison_report(baseline_metrics, new_metrics)
    print(report)
    
    # Save report
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print()
    print(f"Baseline Model: {'INVALID' if not baseline_metrics.overall_valid else 'VALID'}")
    print(f"New Model (17 modules): {'VALID' if new_metrics.overall_valid else 'INVALID'}")
    print()
    
    cv_improvement = ((new_metrics.uncertainty_cv - baseline_metrics.uncertainty_cv) / 
                     max(baseline_metrics.uncertainty_cv, 0.001) * 100)
    print(f"Uncertainty CV improvement: {cv_improvement:+.1f}%")
    print(f"Modules integrated: 0 → 17 (+17 modules)")
    print()
    print(f"All graphs saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
