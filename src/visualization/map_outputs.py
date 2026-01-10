"""Decision outputs, maps, and textual summaries (Prompt-9 + Scientific Corrections)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from ..config.parameters import OutputsConfig, get_default_parameters
from ..decision.risk_decision_model import (
    compute_risk_decisions,
    RiskDecisionConfig,
    RiskLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class DecisionOutputs:
    decision_gdf: gpd.GeoDataFrame
    stress_gdf: gpd.GeoDataFrame
    uncertainty_gdf: gpd.GeoDataFrame
    summaries: pd.DataFrame


def _decision_categories(
    stress: np.ndarray, variance: np.ndarray, cfg: OutputsConfig
) -> np.ndarray:
    """Compute scientifically valid risk categories using data-driven model.
    
    Key: Decisions depend on BOTH stress AND confidence.
    """
    uncertainty = np.sqrt(variance)
    
    # Use the scientifically correct risk decision model
    risk_config = RiskDecisionConfig(
        high_stress_threshold=cfg.high_risk_threshold,
        medium_stress_threshold=cfg.medium_risk_threshold,
        low_stress_threshold=0.2,
        uncertainty_low=cfg.uncertainty_low,
        uncertainty_medium=(cfg.uncertainty_low + cfg.uncertainty_high) / 2,
        uncertainty_high=cfg.uncertainty_high,
    )
    
    # Compute risk decisions
    result = compute_risk_decisions(
        stress,
        uncertainty,
        reliability_mask=None,
        config=risk_config,
    )
    
    # Log statistics for validation
    logger.info(
        "Risk decisions: HIGH=%.1f%%, MEDIUM=%.1f%%, LOW=%.1f%%, NO_DECISION=%.1f%%",
        result.high_pct, result.medium_pct, result.low_pct, result.no_decision_pct
    )
    
    # Return as string array
    return result.risk_levels


def _grid_to_geodf(
    raster: np.ndarray,
    transform,
    crs: str,
    column: str,
) -> gpd.GeoDataFrame:
    h, w = raster.shape
    geoms = []
    vals = []
    for i in range(h):
        for j in range(w):
            x0, y0 = transform * (j, i)
            x1, y1 = transform * (j + 1, i + 1)
            geoms.append(box(x0, y0, x1, y1))
            vals.append(raster[i, j])
    gdf = gpd.GeoDataFrame({column: vals}, geometry=geoms, crs=crs)
    return gdf


def generate_outputs(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    transform,
    timestamps: List[pd.Timestamp],
    cfg: Optional[OutputsConfig] = None,
) -> DecisionOutputs:
    """Generate decision maps and summaries from posterior fields."""

    cfg = cfg or get_default_parameters().outputs

    # Use peak time for decision map
    t_peak = int(np.nanargmax(posterior_mean.mean(axis=(1, 2))))
    stress_peak = posterior_mean[t_peak]
    var_peak = posterior_variance[t_peak]
    categories = _decision_categories(stress_peak, var_peak, cfg)

    stress_gdf = _grid_to_geodf(stress_peak, transform, cfg.crs, "stress")
    uncert_gdf = _grid_to_geodf(var_peak, transform, cfg.crs, "variance")
    decision_gdf = _grid_to_geodf(categories, transform, cfg.crs, "decision")

    summaries = _summaries(
        decision_gdf, stress_gdf, uncert_gdf, timestamps, t_peak
    )

    return DecisionOutputs(
        decision_gdf=decision_gdf,
        stress_gdf=stress_gdf,
        uncertainty_gdf=uncert_gdf,
        summaries=summaries,
    )


def _summaries(
    decision_gdf: gpd.GeoDataFrame,
    stress_gdf: gpd.GeoDataFrame,
    uncert_gdf: gpd.GeoDataFrame,
    timestamps: List[pd.Timestamp],
    t_peak: int,
) -> pd.DataFrame:
    peak_time = timestamps[t_peak] if 0 <= t_peak < len(timestamps) else None
    records = []
    for idx, row in decision_gdf.iterrows():
        decision = row["decision"]
        stress_val = float(stress_gdf.iloc[idx]["stress"])
        uncert_val = float(uncert_gdf.iloc[idx]["variance"])
        if decision == "high":
            msg = (
                "High drainage stress with low uncertainty; "
                "prioritize action; "
                f"peak at {peak_time}"
            )
        elif decision == "medium":
            msg = (
                "Moderate stress with bounded uncertainty; monitor closely; "
                f"peak at {peak_time}"
            )
        else:
            msg = (
                "No-decision due to high uncertainty or low stress; "
                "avoid over-commitment"
            )
        records.append(
            {
                "decision": decision,
                "stress": stress_val,
                "variance": uncert_val,
                "message": msg,
            }
        )
    return pd.DataFrame(records)


def save_outputs(
    outputs: DecisionOutputs, out_dir: Path, driver: Optional[str] = None
) -> None:
    driver = (
        driver
        or outputs.decision_gdf.meta.get("driver", "GeoJSON")
        if hasattr(outputs.decision_gdf, "meta")
        else "GeoJSON"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs.decision_gdf.to_file(
        out_dir / "decision_map.geojson", driver=driver
    )
    outputs.stress_gdf.to_file(out_dir / "stress_map.geojson", driver=driver)
    outputs.uncertainty_gdf.to_file(
        out_dir / "uncertainty_map.geojson", driver=driver
    )
    outputs.summaries.to_csv(out_dir / "decision_summaries.csv", index=False)
    logger.info("Decision outputs written to %s", out_dir)
