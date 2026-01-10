"""Visualization outputs - Maps, Charts, and Graphs (Prompt-5 Enhancement).

Generates actual visual outputs that users can see:
- Stress heatmaps (PNG)
- Uncertainty maps (PNG)
- Decision maps with color coding
- Time series charts
- Summary dashboard
- Risk distribution charts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

logger = logging.getLogger(__name__)

# Color schemes
STRESS_CMAP = "YlOrRd"  # Yellow-Orange-Red for stress levels
UNCERTAINTY_CMAP = "Blues"  # Blues for uncertainty
DECISION_COLORS = {
    "high": "#d62728",      # Red
    "medium": "#ff7f0e",    # Orange
    "low": "#2ca02c",       # Green
    "no-decision": "#7f7f7f"  # Gray
}


@dataclass
class VisualizationOutputs:
    """Container for all generated visualizations."""
    
    stress_map_path: Optional[Path] = None
    uncertainty_map_path: Optional[Path] = None
    decision_map_path: Optional[Path] = None
    timeseries_path: Optional[Path] = None
    dashboard_path: Optional[Path] = None
    risk_distribution_path: Optional[Path] = None
    all_paths: List[Path] = None
    
    def __post_init__(self):
        if self.all_paths is None:
            self.all_paths = []
            for attr in [
                self.stress_map_path,
                self.uncertainty_map_path,
                self.decision_map_path,
                self.timeseries_path,
                self.dashboard_path,
                self.risk_distribution_path,
            ]:
                if attr is not None:
                    self.all_paths.append(attr)


def generate_stress_heatmap(
    stress_field: np.ndarray,
    output_path: Path,
    title: str = "Drainage Stress Map",
    timestamps: Optional[List] = None,
    time_index: int = 0,
) -> Path:
    """Generate a heatmap visualization of stress levels.
    
    Args:
        stress_field: 2D or 3D numpy array (T, Y, X) or (Y, X)
        output_path: Where to save the PNG
        title: Plot title
        timestamps: List of timestamps for labeling
        time_index: Which time slice to visualize (if 3D)
    
    Returns:
        Path to saved image
    """
    # Handle 3D array (time series)
    if stress_field.ndim == 3:
        data = stress_field[time_index]
        if timestamps and time_index < len(timestamps):
            title = f"{title}\n{timestamps[time_index]}"
    else:
        data = stress_field
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        data,
        cmap=STRESS_CMAP,
        aspect='auto',
        origin='upper',
    )
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Stress Level", fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Grid X", fontsize=11)
    ax.set_ylabel("Grid Y", fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved stress heatmap: %s", output_path)
    return output_path


def generate_uncertainty_map(
    variance_field: np.ndarray,
    output_path: Path,
    title: str = "Uncertainty Map",
    time_index: int = 0,
) -> Path:
    """Generate uncertainty/variance visualization.
    
    Args:
        variance_field: 2D or 3D numpy array
        output_path: Where to save the PNG
        title: Plot title
        time_index: Which time slice (if 3D)
    
    Returns:
        Path to saved image
    """
    if variance_field.ndim == 3:
        data = np.sqrt(variance_field[time_index])  # Convert to std dev
    else:
        data = np.sqrt(variance_field)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        data,
        cmap=UNCERTAINTY_CMAP,
        aspect='auto',
        origin='upper',
    )
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Uncertainty (Std Dev)", fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Grid X", fontsize=11)
    ax.set_ylabel("Grid Y", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved uncertainty map: %s", output_path)
    return output_path


def generate_decision_map(
    stress_field: np.ndarray,
    variance_field: np.ndarray,
    output_path: Path,
    title: str = "Risk Decision Map",
    high_threshold: float = 0.7,
    medium_threshold: float = 0.4,
    low_threshold: float = 0.2,
    uncertainty_low: float = 0.3,
    uncertainty_high: float = 0.6,
    time_index: int = 0,
) -> Path:
    """Generate SCIENTIFICALLY VALID color-coded decision map.
    
    Key rules (from scientific requirements):
    - HIGH: stress >= high_threshold AND uncertainty <= low
    - NO_DECISION: stress high BUT uncertainty high (can't commit!)
    - MEDIUM: stress >= medium AND uncertainty <= high
    - LOW: stress < low_threshold
    
    A valid decision map MUST have NO_DECISION regions!
    
    Args:
        stress_field: Stress values (2D or 3D)
        variance_field: Variance values (2D or 3D)
        output_path: Where to save
        title: Plot title
        high_threshold: Threshold for high stress
        medium_threshold: Threshold for medium stress
        low_threshold: Threshold for low stress
        uncertainty_low: Below this = confident
        uncertainty_high: Above this = no-decision
        time_index: Time slice to use
    
    Returns:
        Path to saved image
    """
    if stress_field.ndim == 3:
        stress = stress_field[time_index]
        variance = variance_field[time_index]
    else:
        stress = stress_field
        variance = variance_field
    
    uncertainty = np.sqrt(variance)
    
    # Normalize stress for thresholding
    stress_max = np.nanmax(stress) if np.nanmax(stress) > 0 else 1.0
    stress_norm = stress / stress_max
    
    # Normalize uncertainty
    unc_max = np.nanmax(uncertainty) if np.nanmax(uncertainty) > 0 else 1.0
    unc_norm = uncertainty / unc_max
    
    # Create decision categories using SCIENTIFIC rules:
    # (0=no-decision, 1=low, 2=medium, 3=high)
    decisions = np.zeros_like(stress, dtype=int)
    
    # Rule 1: LOW - low stress (any confidence)
    decisions[stress_norm < low_threshold] = 1
    
    # Rule 2: MEDIUM - moderate stress + reasonable confidence
    medium_stress = (stress_norm >= medium_threshold) & (stress_norm < high_threshold)
    medium_confidence = unc_norm <= uncertainty_high
    decisions[medium_stress & medium_confidence] = 2
    
    # Rule 3: HIGH - high stress AND low uncertainty (BOTH conditions!)
    high_stress = stress_norm >= high_threshold
    low_unc = unc_norm <= uncertainty_low
    decisions[high_stress & low_unc] = 3
    
    # Rule 4: NO_DECISION - high/moderate stress BUT high uncertainty
    # This is CRITICAL - can't commit to decisions when uncertain!
    high_unc = unc_norm > uncertainty_high
    decisions[high_unc] = 0  # Override everything with no-decision
    
    # Also: high stress but not low uncertainty = no decision
    high_but_uncertain = high_stress & ~low_unc & ~high_unc
    decisions[high_but_uncertain] = 0
    
    # Custom colormap
    colors = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#d62728']  # gray, green, orange, red
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        decisions,
        cmap=cmap,
        norm=norm,
        aspect='auto',
        origin='upper',
    )
    
    # Custom colorbar with labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['No Decision', 'Low Risk', 'Medium Risk', 'High Risk'])
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Grid X", fontsize=11)
    ax.set_ylabel("Grid Y", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add summary text
    total = decisions.size
    high_pct = (decisions == 3).sum() / total * 100
    med_pct = (decisions == 2).sum() / total * 100
    low_pct = (decisions == 1).sum() / total * 100
    nodec_pct = (decisions == 0).sum() / total * 100
    
    # Warn if distribution is suspicious
    validity_note = ""
    if high_pct > 80:
        validity_note = " ⚠️ Model may be broken (too much high risk)"
    elif nodec_pct < 5:
        validity_note = " ⚠️ Check uncertainty model (too few no-decisions)"
    
    summary = f"High: {high_pct:.1f}% | Medium: {med_pct:.1f}% | Low: {low_pct:.1f}% | No-Decision: {nodec_pct:.1f}%{validity_note}"
    ax.text(
        0.5, -0.12, summary,
        transform=ax.transAxes,
        ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved decision map: %s", output_path)
    return output_path


def generate_timeseries_chart(
    stress_field: np.ndarray,
    output_path: Path,
    timestamps: Optional[List] = None,
    title: str = "Stress Evolution Over Time",
) -> Path:
    """Generate time series chart of stress evolution.
    
    Args:
        stress_field: 3D array (T, Y, X)
        output_path: Where to save
        timestamps: List of timestamps
        title: Plot title
    
    Returns:
        Path to saved image
    """
    if stress_field.ndim != 3:
        logger.warning("Time series requires 3D data, got %dD", stress_field.ndim)
        # Create dummy time series
        stress_field = stress_field.reshape(1, *stress_field.shape)
    
    T = stress_field.shape[0]
    
    # Calculate statistics per timestep
    mean_stress = [stress_field[t].mean() for t in range(T)]
    max_stress = [stress_field[t].max() for t in range(T)]
    min_stress = [stress_field[t].min() for t in range(T)]
    std_stress = [stress_field[t].std() for t in range(T)]
    
    if timestamps and len(timestamps) >= T:
        x = timestamps[:T]
        xlabel = "Time"
    else:
        x = list(range(T))
        xlabel = "Time Step"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top: Mean stress with confidence band
    ax1.plot(x, mean_stress, 'b-', linewidth=2, label='Mean Stress')
    ax1.fill_between(
        x,
        [m - s for m, s in zip(mean_stress, std_stress)],
        [m + s for m, s in zip(mean_stress, std_stress)],
        alpha=0.3, color='blue', label='±1 Std Dev'
    )
    ax1.plot(x, max_stress, 'r--', alpha=0.7, label='Max')
    ax1.plot(x, min_stress, 'g--', alpha=0.7, label='Min')
    ax1.set_ylabel("Stress Level", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Area of concern (% cells above thresholds)
    # Use percentile-based thresholds computed from the data
    all_values = stress_field.flatten()
    valid_values = all_values[~np.isnan(all_values)]
    if len(valid_values) > 0:
        high_thresh = np.percentile(valid_values, 90)  # Top 10% = high risk
        med_thresh = np.percentile(valid_values, 70)   # 70-90th percentile = medium risk
    else:
        high_thresh, med_thresh = 0.7, 0.4
    
    high_risk = [(stress_field[t] > high_thresh).mean() * 100 for t in range(T)]
    med_risk = [((stress_field[t] > med_thresh) & (stress_field[t] <= high_thresh)).mean() * 100 for t in range(T)]
    
    ax2.fill_between(x, 0, high_risk, color='#d62728', alpha=0.7, label='High Risk %')
    ax2.fill_between(x, high_risk, [h + m for h, m in zip(high_risk, med_risk)], 
                     color='#ff7f0e', alpha=0.7, label='Medium Risk %')
    ax2.set_ylabel("% Area at Risk", fontsize=11)
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved time series chart: %s", output_path)
    return output_path


def generate_risk_distribution(
    stress_field: np.ndarray,
    output_path: Path,
    title: str = "Risk Distribution",
    time_index: int = -1,  # -1 for peak
) -> Path:
    """Generate histogram of risk distribution.
    
    Args:
        stress_field: 2D or 3D stress array
        output_path: Where to save
        title: Plot title
        time_index: Which time to use (-1 for peak)
    
    Returns:
        Path to saved image
    """
    if stress_field.ndim == 3:
        if time_index == -1:
            time_index = int(np.nanargmax(stress_field.mean(axis=(1, 2))))
        data = stress_field[time_index].flatten()
    else:
        data = stress_field.flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram
    n, bins, patches = ax1.hist(
        data, bins=50, edgecolor='black', alpha=0.7
    )
    
    # Color by risk level
    for i, (patch, left, right) in enumerate(zip(patches, bins[:-1], bins[1:])):
        mid = (left + right) / 2
        if mid >= 0.7:
            patch.set_facecolor('#d62728')  # Red
        elif mid >= 0.4:
            patch.set_facecolor('#ff7f0e')  # Orange
        else:
            patch.set_facecolor('#2ca02c')  # Green
    
    ax1.axvline(0.4, color='orange', linestyle='--', linewidth=2, label='Medium threshold')
    ax1.axvline(0.7, color='red', linestyle='--', linewidth=2, label='High threshold')
    ax1.set_xlabel("Stress Level", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title(f"{title} - Histogram", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Pie chart
    low = (data < 0.4).sum()
    medium = ((data >= 0.4) & (data < 0.7)).sum()
    high = (data >= 0.7).sum()
    
    sizes = [low, medium, high]
    labels = [
        f'Low Risk\n({low/len(data)*100:.1f}%)',
        f'Medium Risk\n({medium/len(data)*100:.1f}%)',
        f'High Risk\n({high/len(data)*100:.1f}%)'
    ]
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    explode = (0, 0, 0.1)  # Explode high risk slice
    
    ax2.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='', startangle=90, shadow=True
    )
    ax2.set_title(f"{title} - Breakdown", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Saved risk distribution: %s", output_path)
    return output_path


def generate_dashboard(
    stress_field: np.ndarray,
    variance_field: np.ndarray,
    output_path: Path,
    city: str = "Unknown",
    timestamps: Optional[List] = None,
    run_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate comprehensive dashboard with all key visualizations.
    
    Args:
        stress_field: 3D stress array (T, Y, X)
        variance_field: 3D variance array
        output_path: Where to save
        city: City name
        timestamps: List of timestamps
        run_info: Additional run information
    
    Returns:
        Path to saved image
    """
    # Find peak time - use max stress (not mean) to capture hotspots from complaints
    if stress_field.ndim == 3:
        t_peak = int(np.nanargmax(stress_field.max(axis=(1, 2))))
        stress_peak = stress_field[t_peak]
        var_peak = variance_field[t_peak]
        T = stress_field.shape[0]
    else:
        t_peak = 0
        stress_peak = stress_field
        var_peak = variance_field
        T = 1
    
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle(
        f"Urban Drainage Stress Dashboard - {city.upper()}",
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Stress Heatmap (top-left, larger)
    ax1 = fig.add_subplot(gs[0, 0:2])
    im1 = ax1.imshow(stress_peak, cmap=STRESS_CMAP, aspect='auto')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Stress')
    ax1.set_title(f"Peak Stress (t={t_peak})", fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    
    # 2. Decision Map (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    uncertainty = np.sqrt(var_peak)
    decisions = np.zeros_like(stress_peak, dtype=int)
    decisions[stress_peak < 0.4] = 1
    decisions[(stress_peak >= 0.4) & (stress_peak < 0.7)] = 2
    decisions[stress_peak >= 0.7] = 3
    decisions[uncertainty > 0.5] = 0
    
    colors = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#d62728']
    cmap = mcolors.ListedColormap(colors)
    im2 = ax2.imshow(decisions, cmap=cmap, aspect='auto', vmin=0, vmax=3)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, ticks=[0, 1, 2, 3])
    cbar2.ax.set_yticklabels(['N/A', 'Low', 'Med', 'High'])
    ax2.set_title("Risk Decisions", fontweight='bold')
    
    # 3. Time series (middle row, full width)
    ax3 = fig.add_subplot(gs[1, :])
    if T > 1:
        mean_stress = [stress_field[t].mean() for t in range(T)]
        max_stress = [stress_field[t].max() for t in range(T)]
        x = list(range(T))
        ax3.plot(x, mean_stress, 'b-', linewidth=2, label='Mean')
        ax3.plot(x, max_stress, 'r--', linewidth=1.5, label='Max')
        ax3.axvline(t_peak, color='green', linestyle=':', label=f'Peak (t={t_peak})')
        ax3.fill_between(x, 0, mean_stress, alpha=0.2)
    else:
        ax3.bar([0], [stress_peak.mean()], color='blue', alpha=0.7)
        ax3.text(0, stress_peak.mean(), f'{stress_peak.mean():.2f}', ha='center', va='bottom')
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Stress Level")
    ax3.set_title("Stress Evolution", fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk Distribution (bottom-left)
    ax4 = fig.add_subplot(gs[2, 0])
    data = stress_peak.flatten()
    ax4.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(0.4, color='orange', linestyle='--', linewidth=2)
    ax4.axvline(0.7, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel("Stress Level")
    ax4.set_ylabel("Count")
    ax4.set_title("Distribution", fontweight='bold')
    
    # 5. Pie chart (bottom-middle)
    ax5 = fig.add_subplot(gs[2, 1])
    low = (data < 0.4).sum()
    medium = ((data >= 0.4) & (data < 0.7)).sum()
    high = (data >= 0.7).sum()
    sizes = [low, medium, high]
    ax5.pie(sizes, labels=['Low', 'Medium', 'High'], 
            colors=['#2ca02c', '#ff7f0e', '#d62728'],
            autopct='%1.1f%%', startangle=90)
    ax5.set_title("Risk Breakdown", fontweight='bold')
    
    # 6. Summary stats (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = f"""
    SUMMARY STATISTICS
    ══════════════════════
    
    Total Grid Cells: {data.size}
    
    Mean Stress: {data.mean():.3f}
    Max Stress: {data.max():.3f}
    Std Dev: {data.std():.3f}
    
    High Risk: {high/len(data)*100:.1f}%
    Medium Risk: {medium/len(data)*100:.1f}%
    Low Risk: {low/len(data)*100:.1f}%
    
    Peak Time: t={t_peak}
    """
    
    if run_info:
        stats_text += f"""
    Run Mode: {run_info.get('mode', 'N/A')}
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Timestamp
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    fig.text(0.99, 0.01, f"Generated: {timestamp}", ha='right', fontsize=8, alpha=0.7)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info("Saved dashboard: %s", output_path)
    return output_path


def generate_all_visualizations(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    output_dir: Path,
    city: str = "Unknown",
    timestamps: Optional[List] = None,
    run_info: Optional[Dict[str, Any]] = None,
) -> VisualizationOutputs:
    """Generate all visualization outputs.
    
    Args:
        posterior_mean: 3D array (T, Y, X) or 2D (Y, X)
        posterior_variance: Same shape as posterior_mean
        output_dir: Directory to save outputs
        city: City name for titles
        timestamps: Optional list of timestamps
        run_info: Optional run metadata
    
    Returns:
        VisualizationOutputs with paths to all generated images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating visualizations to %s", output_dir)
    
    # Find peak time - use max stress to capture complaint-driven hotspots
    if posterior_mean.ndim == 3:
        t_peak = int(np.nanargmax(posterior_mean.max(axis=(1, 2))))
    else:
        t_peak = 0
    
    outputs = VisualizationOutputs()
    
    try:
        outputs.stress_map_path = generate_stress_heatmap(
            posterior_mean,
            output_dir / "stress_map.png",
            title=f"Drainage Stress - {city}",
            timestamps=timestamps,
            time_index=t_peak,
        )
    except Exception as e:
        logger.error("Failed to generate stress map: %s", e)
    
    try:
        outputs.uncertainty_map_path = generate_uncertainty_map(
            posterior_variance,
            output_dir / "uncertainty_map.png",
            title=f"Uncertainty - {city}",
            time_index=t_peak,
        )
    except Exception as e:
        logger.error("Failed to generate uncertainty map: %s", e)
    
    try:
        outputs.decision_map_path = generate_decision_map(
            posterior_mean,
            posterior_variance,
            output_dir / "decision_map.png",
            title=f"Risk Decisions - {city}",
            time_index=t_peak,
        )
    except Exception as e:
        logger.error("Failed to generate decision map: %s", e)
    
    try:
        outputs.timeseries_path = generate_timeseries_chart(
            posterior_mean,
            output_dir / "stress_timeseries.png",
            timestamps=timestamps,
            title=f"Stress Evolution - {city}",
        )
    except Exception as e:
        logger.error("Failed to generate time series: %s", e)
    
    try:
        outputs.risk_distribution_path = generate_risk_distribution(
            posterior_mean,
            output_dir / "risk_distribution.png",
            title=f"Risk Distribution - {city}",
        )
    except Exception as e:
        logger.error("Failed to generate risk distribution: %s", e)
    
    try:
        outputs.dashboard_path = generate_dashboard(
            posterior_mean,
            posterior_variance,
            output_dir / "dashboard.png",
            city=city,
            timestamps=timestamps,
            run_info=run_info,
        )
    except Exception as e:
        logger.error("Failed to generate dashboard: %s", e)
    
    # Update all_paths
    outputs.all_paths = [
        p for p in [
            outputs.stress_map_path,
            outputs.uncertainty_map_path,
            outputs.decision_map_path,
            outputs.timeseries_path,
            outputs.risk_distribution_path,
            outputs.dashboard_path,
        ] if p is not None
    ]
    
    logger.info("Generated %d visualization(s)", len(outputs.all_paths))
    
    return outputs
