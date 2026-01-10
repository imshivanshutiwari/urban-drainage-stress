"""Bayesian-correct visualizations.

This module creates visualizations that HONESTLY represent Bayesian inference:
- Credible interval maps (not smooth gradients)
- Prior vs Posterior comparison
- Information gain maps
- Uncertainty that reflects POSTERIOR variance

Reference: Projectfile.md - "Visualization corrections"

NO FAKE SMOOTH GRADIENTS. SHOW THE UNCERTAINTY.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np

logger = logging.getLogger(__name__)


# Honest colormaps
STRESS_CMAP = "YlOrRd"
UNCERTAINTY_CMAP = "Purples"  # Purple = "not sure"
INFO_GAIN_CMAP = "viridis"
CI_WIDTH_CMAP = "coolwarm_r"  # Red = wide CI (uncertain)


def plot_posterior_with_credible_interval(
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    output_path: Path,
    title: str = "Posterior Stress with 95% Credible Interval",
    time_index: int = 0,
    confidence_level: float = 0.95,
) -> Path:
    """Plot posterior mean with credible interval overlay.
    
    Shows HONEST uncertainty:
    - Main map shows posterior mean
    - Hatching shows regions with wide credible intervals
    - Contours show CI boundaries
    """
    from scipy import stats
    
    if posterior_mean.ndim == 3:
        mean = posterior_mean[time_index]
        std = posterior_std[time_index]
    else:
        mean = posterior_mean
        std = posterior_std
    
    # Compute credible intervals
    z = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = mean - z * std
    ci_upper = mean + z * std
    ci_width = ci_upper - ci_lower
    
    # Normalize for display
    max_val = np.nanmax(mean) + 1e-9
    mean_norm = mean / max_val
    ci_width_norm = ci_width / max_val
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Posterior Mean
    ax1 = axes[0]
    im1 = ax1.imshow(mean_norm, cmap=STRESS_CMAP, aspect='auto', origin='upper')
    ax1.set_title("Posterior Mean (E[stress|data])", fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Normalized Stress')
    
    # Panel 2: Credible Interval Width
    ax2 = axes[1]
    im2 = ax2.imshow(ci_width_norm, cmap=CI_WIDTH_CMAP, aspect='auto', origin='upper',
                    vmin=0, vmax=np.percentile(ci_width_norm, 95))
    ax2.set_title("95% CI Width (uncertainty)", fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='CI Width')
    
    # Highlight HIGH uncertainty regions
    high_unc_mask = ci_width_norm > np.percentile(ci_width_norm, 75)
    ax2.contour(high_unc_mask, colors='black', linewidths=1, alpha=0.5)
    
    # Panel 3: Combined view with hatching
    ax3 = axes[2]
    im3 = ax3.imshow(mean_norm, cmap=STRESS_CMAP, aspect='auto', origin='upper')
    ax3.set_title("Mean + Uncertainty Overlay", fontweight='bold')
    
    # Hatch regions where CI is wide (uncertain)
    uncertain_mask = ci_width_norm > np.median(ci_width_norm)
    hatched = np.ma.masked_where(~uncertain_mask, np.ones_like(mean_norm))
    ax3.imshow(hatched, cmap=mcolors.ListedColormap(['none']), aspect='auto', 
              origin='upper', alpha=0.3)
    ax3.contour(uncertain_mask, colors='white', linewidths=0.5, alpha=0.7)
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='Stress (hatched=uncertain)')
    
    # Add legend for hatching
    legend_elements = [
        Patch(facecolor='none', edgecolor='white', label='High uncertainty'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    for ax in axes:
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved Bayesian posterior visualization: %s", output_path)
    return output_path


def plot_prior_vs_posterior(
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    output_path: Path,
    title: str = "Prior vs Posterior Comparison",
    time_index: int = 0,
) -> Path:
    """Compare prior and posterior to show how data updated beliefs.
    
    THIS IS THE KEY VALIDATION:
    - If prior and posterior are identical → observations don't matter!
    - Should see variance reduction where observations exist.
    """
    if prior_mean.ndim == 3:
        prior_m = prior_mean[time_index]
        prior_v = prior_variance[time_index]
        post_m = posterior_mean[time_index]
        post_v = posterior_variance[time_index]
    else:
        prior_m = prior_mean
        prior_v = prior_variance
        post_m = posterior_mean
        post_v = posterior_variance
    
    prior_std = np.sqrt(prior_v)
    post_std = np.sqrt(post_v)
    
    # Compute changes
    mean_shift = post_m - prior_m
    variance_reduction = (prior_v - post_v) / (prior_v + 1e-9)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Prior
    vmax_mean = max(np.nanmax(prior_m), np.nanmax(post_m))
    vmax_std = max(np.nanmax(prior_std), np.nanmax(post_std))
    
    ax = axes[0, 0]
    im = ax.imshow(prior_m, cmap=STRESS_CMAP, aspect='auto', origin='upper', vmax=vmax_mean)
    ax.set_title("Prior Mean (before observations)", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax = axes[0, 1]
    im = ax.imshow(prior_std, cmap=UNCERTAINTY_CMAP, aspect='auto', origin='upper', vmax=vmax_std)
    ax.set_title("Prior Std Dev", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Bottom row: Posterior
    ax = axes[1, 0]
    im = ax.imshow(post_m, cmap=STRESS_CMAP, aspect='auto', origin='upper', vmax=vmax_mean)
    ax.set_title("Posterior Mean (after observations)", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax = axes[1, 1]
    im = ax.imshow(post_std, cmap=UNCERTAINTY_CMAP, aspect='auto', origin='upper', vmax=vmax_std)
    ax.set_title("Posterior Std Dev", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Right column: Changes
    ax = axes[0, 2]
    vmax_shift = max(abs(np.nanmin(mean_shift)), abs(np.nanmax(mean_shift)))
    im = ax.imshow(mean_shift, cmap='coolwarm', aspect='auto', origin='upper',
                  vmin=-vmax_shift, vmax=vmax_shift)
    ax.set_title("Mean Shift (post - prior)", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax = axes[1, 2]
    im = ax.imshow(variance_reduction * 100, cmap='Greens', aspect='auto', origin='upper',
                  vmin=0, vmax=100)
    ax.set_title("Variance Reduction % (should be >0)", fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='%')
    
    # Add diagnostic text
    mean_var_red = np.nanmean(variance_reduction) * 100
    max_var_red = np.nanmax(variance_reduction) * 100
    
    if mean_var_red < 1:
        status = "⚠️ WARN: Observations barely affect posterior!"
        color = 'red'
    elif mean_var_red < 5:
        status = "Observations have modest effect"
        color = 'orange'
    else:
        status = "✓ Observations properly constrain inference"
        color = 'green'
    
    fig.text(0.5, 0.01, f"Mean variance reduction: {mean_var_red:.1f}% | Max: {max_var_red:.1f}% | {status}",
            ha='center', fontsize=11, color=color, fontweight='bold')
    
    for ax in axes.flat:
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved prior vs posterior comparison: %s", output_path)
    return output_path


def plot_information_gain_map(
    information_gain: np.ndarray,
    complaints: Optional[np.ndarray],
    output_path: Path,
    title: str = "Information Gain from Observations",
    time_index: int = 0,
) -> Path:
    """Visualize where observations provided the most information.
    
    Information gain = KL divergence = how much posterior differs from prior.
    Should be HIGH where observations exist.
    If low everywhere → observations don't matter → BROKEN.
    """
    if information_gain.ndim == 3:
        ig = information_gain[time_index]
        obs = complaints[time_index] if complaints is not None else None
    else:
        ig = information_gain
        obs = complaints
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Information gain map
    ax1 = axes[0]
    im1 = ax1.imshow(ig, cmap=INFO_GAIN_CMAP, aspect='auto', origin='upper')
    ax1.set_title("Information Gain (KL divergence)", fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='nats')
    
    # Overlay observation locations
    if obs is not None:
        obs_mask = obs > 0
        obs_y, obs_x = np.where(obs_mask)
        ax1.scatter(obs_x, obs_y, c='red', s=10, marker='x', 
                   label=f'Observations (n={obs_mask.sum()})')
        ax1.legend(loc='upper right')
    
    # Panel 2: Information gain histogram
    ax2 = axes[1]
    ig_flat = ig.flatten()
    ig_flat = ig_flat[~np.isnan(ig_flat)]
    
    ax2.hist(ig_flat, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(np.mean(ig_flat), color='red', linestyle='--', 
               label=f'Mean: {np.mean(ig_flat):.3f}')
    ax2.axvline(np.median(ig_flat), color='orange', linestyle='--',
               label=f'Median: {np.median(ig_flat):.3f}')
    ax2.set_xlabel("Information Gain (nats)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Information Gain", fontweight='bold')
    ax2.legend()
    
    # Diagnostic
    mean_ig = np.mean(ig_flat)
    if mean_ig < 0.01:
        status = "⚠️ VERY LOW - observations may not matter!"
        color = 'red'
    elif mean_ig < 0.1:
        status = "Low but detectable"
        color = 'orange'
    else:
        status = "✓ Observations provide information"
        color = 'green'
    
    fig.text(0.5, 0.01, f"Status: {status}", ha='center', fontsize=11, 
            color=color, fontweight='bold')
    
    for ax in [ax1]:
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved information gain visualization: %s", output_path)
    return output_path


def plot_risk_with_confidence(
    posterior_mean: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    risk_levels: np.ndarray,
    output_path: Path,
    title: str = "Risk Classification with Confidence",
    time_index: int = 0,
) -> Path:
    """Plot risk decisions with confidence visualization.
    
    Shows:
    - Risk categories (color)
    - CI width (opacity/hatching)
    - NO_DECISION regions highlighted
    """
    if posterior_mean.ndim == 3:
        mean = posterior_mean[time_index]
        lower = ci_lower[time_index]
        upper = ci_upper[time_index]
        risk = risk_levels[time_index]
    else:
        mean = posterior_mean
        lower = ci_lower
        upper = ci_upper
        risk = risk_levels
    
    ci_width = upper - lower
    
    # Convert risk levels to numeric
    risk_numeric = np.zeros_like(mean, dtype=int)
    risk_flat = risk.flatten()
    
    for i, r in enumerate(risk_flat):
        r_str = str(r).lower()
        if 'high' in r_str:
            risk_numeric.flat[i] = 3
        elif 'medium' in r_str:
            risk_numeric.flat[i] = 2
        elif 'low' in r_str:
            risk_numeric.flat[i] = 1
        else:  # no-decision
            risk_numeric.flat[i] = 0
    
    # Custom colormap
    colors = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#d62728']  # gray, green, orange, red
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Risk categories
    ax1 = axes[0]
    im1 = ax1.imshow(risk_numeric, cmap=cmap, norm=norm, aspect='auto', origin='upper')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, ticks=[0, 1, 2, 3])
    cbar1.ax.set_yticklabels(['No Decision', 'Low', 'Medium', 'High'])
    ax1.set_title("Risk Classification", fontweight='bold')
    
    # Panel 2: CI Width (confidence)
    ax2 = axes[1]
    ci_width_norm = ci_width / (np.nanmax(ci_width) + 1e-9)
    im2 = ax2.imshow(ci_width_norm, cmap=CI_WIDTH_CMAP, aspect='auto', origin='upper')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='CI Width (normalized)')
    ax2.set_title("Confidence (narrow=confident)", fontweight='bold')
    
    # Panel 3: NO_DECISION highlighting
    ax3 = axes[2]
    no_dec_mask = risk_numeric == 0
    
    # Show mean with NO_DECISION overlay
    mean_norm = mean / (np.nanmax(mean) + 1e-9)
    im3 = ax3.imshow(mean_norm, cmap=STRESS_CMAP, aspect='auto', origin='upper')
    
    # Overlay NO_DECISION in gray
    no_dec_overlay = np.ma.masked_where(~no_dec_mask, np.ones_like(mean_norm))
    ax3.imshow(no_dec_overlay, cmap=mcolors.ListedColormap(['gray']), 
              aspect='auto', origin='upper', alpha=0.7)
    ax3.contour(no_dec_mask, colors='white', linewidths=1)
    
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='Stress')
    ax3.set_title("NO_DECISION regions (gray)", fontweight='bold')
    
    # Statistics
    total = risk_numeric.size
    stats = {
        'high': (risk_numeric == 3).sum() / total * 100,
        'medium': (risk_numeric == 2).sum() / total * 100,
        'low': (risk_numeric == 1).sum() / total * 100,
        'no_dec': (risk_numeric == 0).sum() / total * 100,
    }
    
    stats_text = (f"HIGH: {stats['high']:.1f}% | MEDIUM: {stats['medium']:.1f}% | "
                 f"LOW: {stats['low']:.1f}% | NO_DECISION: {stats['no_dec']:.1f}%")
    
    # Validity check
    if stats['high'] > 80:
        validity = "⚠️ Too much HIGH risk - model may be broken"
        color = 'red'
    elif stats['no_dec'] < 10:
        validity = "⚠️ Too few NO_DECISION - may be overconfident"
        color = 'orange'
    else:
        validity = "✓ Reasonable distribution"
        color = 'green'
    
    fig.text(0.5, 0.01, f"{stats_text}\n{validity}", 
            ha='center', fontsize=10, color=color)
    
    for ax in axes:
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved risk with confidence visualization: %s", output_path)
    return output_path


def plot_parameter_calibration(
    learned_params: Dict[str, float],
    param_cis: Dict[str, Tuple[float, float]],
    fit_metrics: Dict[str, float],
    output_path: Path,
    title: str = "Calibrated Parameters",
) -> Path:
    """Visualize learned parameters with confidence intervals.
    
    Shows:
    - Parameter values with error bars (95% CI)
    - Fit quality metrics
    - Identifiability indicators
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Parameter values with CIs
    ax1 = axes[0]
    
    params = list(learned_params.keys())
    values = list(learned_params.values())
    
    # Get CIs or default (ensure non-negative error bars)
    lower_err = []
    upper_err = []
    for p in params:
        if p in param_cis:
            ci = param_cis[p]
            val = values[params.index(p)]
            # Ensure error bars are non-negative (absolute distance from value)
            lower_err.append(max(0, abs(val - ci[0])))
            upper_err.append(max(0, abs(ci[1] - val)))
        else:
            lower_err.append(0)
            upper_err.append(0)
    
    y_pos = np.arange(len(params))
    ax1.barh(y_pos, values, xerr=[lower_err, upper_err], 
            capsize=5, color='steelblue', edgecolor='navy')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(params)
    ax1.set_xlabel("Value")
    ax1.set_title("Learned Parameters (with 95% CI)", fontweight='bold')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Highlight non-identifiable (wide CI)
    for i, p in enumerate(params):
        if p in param_cis:
            ci_width = param_cis[p][1] - param_cis[p][0]
            rel_width = ci_width / (abs(values[i]) + 1e-9)
            if rel_width > 1.0:
                ax1.text(values[i], i, " ⚠", fontsize=12, va='center', color='red')
    
    # Panel 2: Fit metrics
    ax2 = axes[1]
    
    metrics = list(fit_metrics.keys())
    metric_vals = list(fit_metrics.values())
    
    colors = []
    for m, v in zip(metrics, metric_vals):
        if 'r_squared' in m.lower() or 'r2' in m.lower():
            if v >= 0.7:
                colors.append('green')
            elif v >= 0.4:
                colors.append('orange')
            else:
                colors.append('red')
        else:
            colors.append('steelblue')
    
    y_pos2 = np.arange(len(metrics))
    ax2.barh(y_pos2, metric_vals, color=colors, edgecolor='navy')
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(metrics)
    ax2.set_xlabel("Value")
    ax2.set_title("Fit Quality Metrics", fontweight='bold')
    
    # Add interpretation
    if 'r_squared' in fit_metrics:
        r2 = fit_metrics['r_squared']
        if r2 >= 0.7:
            quality = "Good fit"
        elif r2 >= 0.4:
            quality = "Acceptable fit"
        else:
            quality = "Poor fit - check model"
        ax2.text(0.95, 0.05, f"Overall: {quality}", transform=ax2.transAxes,
                ha='right', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved parameter calibration visualization: %s", output_path)
    return output_path


def generate_all_bayesian_visualizations(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    prior_mean: Optional[np.ndarray] = None,
    prior_variance: Optional[np.ndarray] = None,
    information_gain: Optional[np.ndarray] = None,
    risk_levels: Optional[np.ndarray] = None,
    complaints: Optional[np.ndarray] = None,
    learned_params: Optional[Dict[str, float]] = None,
    param_cis: Optional[Dict[str, Tuple[float, float]]] = None,
    fit_metrics: Optional[Dict[str, float]] = None,
    output_dir: Path = Path("outputs/visualizations"),
    time_index: int = 0,
) -> List[Path]:
    """Generate all Bayesian visualizations.
    
    Returns list of generated file paths.
    """
    from scipy import stats
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    
    posterior_std = np.sqrt(posterior_variance)
    
    # 1. Posterior with CI
    paths.append(plot_posterior_with_credible_interval(
        posterior_mean, posterior_std,
        output_dir / "posterior_with_ci.png",
        time_index=time_index,
    ))
    
    # 2. Prior vs Posterior (if prior available)
    if prior_mean is not None and prior_variance is not None:
        paths.append(plot_prior_vs_posterior(
            prior_mean, prior_variance,
            posterior_mean, posterior_variance,
            output_dir / "prior_vs_posterior.png",
            time_index=time_index,
        ))
    
    # 3. Information gain map
    if information_gain is not None:
        paths.append(plot_information_gain_map(
            information_gain, complaints,
            output_dir / "information_gain.png",
            time_index=time_index,
        ))
    
    # 4. Risk with confidence
    if risk_levels is not None:
        z = stats.norm.ppf(0.975)
        ci_lower = posterior_mean - z * posterior_std
        ci_upper = posterior_mean + z * posterior_std
        
        paths.append(plot_risk_with_confidence(
            posterior_mean, ci_lower, ci_upper, risk_levels,
            output_dir / "risk_with_confidence.png",
            time_index=time_index,
        ))
    
    # 5. Parameter calibration
    if learned_params is not None:
        paths.append(plot_parameter_calibration(
            learned_params,
            param_cis or {},
            fit_metrics or {},
            output_dir / "parameter_calibration.png",
        ))
    
    logger.info("Generated %d Bayesian visualizations in %s", len(paths), output_dir)
    return paths
