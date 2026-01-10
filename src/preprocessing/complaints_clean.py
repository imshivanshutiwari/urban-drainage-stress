"""Complaint preprocessing: spatial normalization, delay modeling.

Aggregation retains uncertainty and bias metadata.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..config.parameters import get_default_parameters
from ..utils.validation_utils import require_columns

logger = logging.getLogger(__name__)


def complaints_to_grid(
    complaints_df: pd.DataFrame,
    transform: Tuple,
    grid_shape: Tuple[int, int],
    timestamps: list,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "timestamp",
) -> np.ndarray:
    """Convert complaints DataFrame to a spatial-temporal grid.
    
    Args:
        complaints_df: DataFrame with lat, lon, timestamp columns
        transform: Geotransform tuple (origin_x, pixel_width, 0, origin_y, 0, pixel_height)
        grid_shape: (ny, nx) grid dimensions
        timestamps: List of timestamps for the time dimension
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        time_col: Name of timestamp column
        
    Returns:
        3D array (T, H, W) with complaint counts per cell per timestep
    """
    ny, nx = grid_shape
    n_times = len(timestamps)
    grid = np.zeros((n_times, ny, nx), dtype=np.float64)
    
    if complaints_df is None or len(complaints_df) == 0:
        logger.warning("No complaints to grid - returning zeros")
        return grid
    
    # Extract geotransform - handle both Affine objects and tuples
    try:
        # Rasterio Affine object
        origin_x = transform.c  # x offset
        pixel_width = transform.a  # pixel width
        origin_y = transform.f  # y offset  
        pixel_height = transform.e  # pixel height (negative for north-up)
    except AttributeError:
        # GDAL-style tuple (origin_x, pixel_width, 0, origin_y, 0, pixel_height)
        origin_x, pixel_width, _, origin_y, _, pixel_height = transform
    
    # Compute bounds
    # pixel_height is typically negative for north-up images
    min_lon = origin_x
    max_lon = origin_x + nx * pixel_width
    min_lat = origin_y + ny * pixel_height  # pixel_height is negative
    max_lat = origin_y
    
    logger.info(
        "Gridding complaints: bounds=[%.4f,%.4f,%.4f,%.4f], grid=%dx%d",
        min_lon, max_lon, min_lat, max_lat, ny, nx
    )
    
    # Filter complaints to valid coordinates
    df = complaints_df.copy()
    
    # Check for required columns
    if lat_col not in df.columns or lon_col not in df.columns:
        logger.warning(f"Complaints missing {lat_col}/{lon_col} columns")
        return grid
    
    # Filter by bounds
    valid_mask = (
        (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
        (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon) &
        df[lat_col].notna() & df[lon_col].notna()
    )
    df_valid = df[valid_mask].copy()
    
    logger.info(
        "Complaints in grid bounds: %d / %d (%.1f%%)",
        len(df_valid), len(df), 100 * len(df_valid) / max(len(df), 1)
    )
    
    if len(df_valid) == 0:
        logger.warning(
            "No complaints within grid bounds! Check coordinate alignment. "
            f"Complaints: lat=[{df[lat_col].min():.4f}, {df[lat_col].max():.4f}], "
            f"lon=[{df[lon_col].min():.4f}, {df[lon_col].max():.4f}]"
        )
        return grid
    
    # Convert timestamps to indices
    if time_col in df_valid.columns:
        df_valid[time_col] = pd.to_datetime(df_valid[time_col], errors='coerce')
        timestamps_dt = pd.to_datetime(timestamps)
        
        # Create time index mapping (daily matching)
        time_to_idx = {ts.date(): i for i, ts in enumerate(timestamps_dt)}
    else:
        # If no timestamp, spread uniformly
        time_to_idx = None
    
    # Convert coordinates to grid indices
    count = 0
    for _, row in df_valid.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        
        # Compute grid indices
        col_idx = int((lon - origin_x) / pixel_width)
        row_idx = int((lat - origin_y) / pixel_height)  # pixel_height is negative
        
        # Bounds check
        if not (0 <= row_idx < ny and 0 <= col_idx < nx):
            continue
        
        # Time index
        if time_to_idx is not None and time_col in df_valid.columns:
            ts = row[time_col]
            if pd.notna(ts):
                t_idx = time_to_idx.get(ts.date(), 0)
            else:
                t_idx = 0
        else:
            t_idx = 0
        
        grid[t_idx, row_idx, col_idx] += 1.0
        count += 1
    
    total_complaints = grid.sum()
    cells_with_complaints = (grid > 0).sum()
    
    logger.info(
        "Gridded %d complaints -> %d grid cells (%.2f%% coverage)",
        count, cells_with_complaints,
        100 * cells_with_complaints / (n_times * ny * nx)
    )
    
    return grid


def to_geodf_with_uncertainty(
    df: pd.DataFrame,
    ward_geometries: Optional[gpd.GeoDataFrame] = None,
    ward_name_col: str = "ward",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> gpd.GeoDataFrame:
    """Convert complaints to GeoDataFrame preserving coarse spatial resolution.

    - Point locations are used directly.
    - Ward/area-level complaints become centroids with an uncertainty radius.
    - Records without any spatial ref are flagged via geometry=None.
    """

    params = get_default_parameters().complaints

    has_point = (
        df[[lat_col, lon_col]].notna().all(axis=1)
        if {lat_col, lon_col} <= set(df.columns)
        else pd.Series(False, index=df.index)
    )
    geoms = []
    radius_m = []
    precision = []

    for idx, row in df.iterrows():
        if has_point.loc[idx]:
            geoms.append(Point(row[lon_col], row[lat_col]))
            radius_m.append(params.geo_epsilon_meters)
            precision.append("point")
            continue

        ward_name = (
            str(row.get(ward_name_col, ""))
            if ward_name_col in df.columns
            else ""
        )
        ward_in = None
        if ward_geometries is not None:
            ward_in = ward_geometries[ward_name_col].astype(str).values
        if (
            ward_geometries is not None
            and ward_name
            and ward_in is not None
            and ward_name in ward_in
        ):
            match = ward_geometries[
                ward_geometries[ward_name_col].astype(str) == ward_name
            ]
            centroid = match.geometry.iloc[0].centroid
            geoms.append(centroid)
            radius_m.append(params.spatial_uncertainty_radius_m)
            precision.append("ward_centroid")
        else:
            geoms.append(None)
            radius_m.append(np.nan)
            precision.append("unknown")

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
    gdf["spatial_uncertainty_radius_m"] = radius_m
    gdf["location_precision"] = precision

    missing_geom = gdf["geometry"].isna().sum()
    if missing_geom:
        logger.warning(
            "Spatial normalization: %d records lack any resolvable geometry",
            missing_geom,
        )

    return gdf


def _lognormal_params_from_mean_sd(
    mean_hours: float, sd_hours: float
) -> Tuple[float, float]:
    var = sd_hours ** 2
    mu = np.log((mean_hours ** 2) / np.sqrt(var + mean_hours ** 2))
    sigma = np.sqrt(np.log(1 + var / (mean_hours ** 2)))
    return mu, sigma


def model_reporting_delay(
    df: pd.DataFrame,
    prior_mean_hours: Optional[float] = None,
    prior_sd_hours: Optional[float] = None,
    max_delay_hours: Optional[float] = None,
    event_time_col: str = "event_time",
    complaint_time_col: str = "complaint_time",
) -> pd.DataFrame:
    """Attach probabilistic event time windows given reporting delays.

    - If event_time is available, empirical delays refine the distribution.
    - Otherwise, a lognormal prior (configurable) defines quantile windows.
    - Outputs lower/upper bounds for plausible event times, not point labels.
    """

    params = get_default_parameters().complaints
    prior_mean_hours = prior_mean_hours or params.delay_prior_mean_hours
    prior_sd_hours = prior_sd_hours or params.delay_prior_sd_hours
    max_delay_hours = max_delay_hours or params.max_delay_hours

    require_columns(df, [complaint_time_col], context="delay modeling")
    out = df.copy()
    complaint_time = pd.to_datetime(out[complaint_time_col], errors="coerce")

    # Empirical delays if event times exist.
    empirical_delays = None
    if event_time_col in out.columns:
        event_time = pd.to_datetime(out[event_time_col], errors="coerce")
        empirical_delays = (
            (complaint_time - event_time).dt.total_seconds().div(3600)
        )
        empirical_delays = empirical_delays[
            (empirical_delays > 0) & (empirical_delays <= max_delay_hours)
        ]

    mu, sigma = _lognormal_params_from_mean_sd(
        prior_mean_hours, prior_sd_hours
    )
    if empirical_delays is not None and len(empirical_delays.dropna()) >= 10:
        mu = np.log(empirical_delays.mean()) - 0.5 * np.log(
            empirical_delays.var() / (empirical_delays.mean() ** 2) + 1
        )
        sigma = np.sqrt(
            np.log(empirical_delays.var() / (empirical_delays.mean() ** 2) + 1)
        )
        logger.info(
            "Reporting delay: using empirical lognormal fit from %d samples",
            len(empirical_delays),
        )
    else:
        logger.info(
            "Reporting delay: using prior (mean=%.2f h, sd=%.2f h)",
            prior_mean_hours,
            prior_sd_hours,
        )

    # Quantiles to define windows.
    q05, q50, q95 = [np.exp(mu + sigma * z) for z in [-1.645, 0.0, 1.645]]
    q95 = min(q95, max_delay_hours)

    out["delay_hours_p50"] = q50
    out["delay_hours_p05"] = q05
    out["delay_hours_p95"] = q95

    out["event_window_start"] = complaint_time - pd.to_timedelta(q95, unit="h")
    out["event_window_end"] = complaint_time - pd.to_timedelta(q05, unit="h")
    empirical_ok = (
        empirical_delays is not None
        and len(empirical_delays.dropna()) >= 10
    )
    out["delay_source"] = "empirical" if empirical_ok else "prior"

    return out


def aggregate_complaints(
    df: pd.DataFrame,
    spatial_key: str = "ward",
    time_window_minutes: Optional[int] = None,
    bias_lookup: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Aggregate complaints while preserving bias and confidence metadata.

    - Uses event_window_start (if present) else complaint_time for binning.
    - Reports counts, mean reporting confidence, and coverage bias indicator.
    - No record is dropped; bins with low confidence remain visible.
    """

    params = get_default_parameters().complaints
    time_window_minutes = (
        time_window_minutes or params.time_aggregation_minutes
    )

    time_col = (
        "event_window_start"
        if "event_window_start" in df.columns
        else "complaint_time"
    )
    require_columns(
        df,
        [time_col, spatial_key, "reporting_confidence"],
        context="aggregation",
    )

    times = pd.to_datetime(df[time_col], errors="coerce")
    df = df.copy()
    df["time_bin"] = times.dt.floor(f"{time_window_minutes}min")

    grouped = df.groupby([spatial_key, "time_bin"], dropna=False)

    records = []
    for (area, tbin), grp in grouped:
        count = len(grp)
        mean_conf = (
            float(grp["reporting_confidence"].mean())
            if "reporting_confidence" in grp
            else np.nan
        )
        bias = bias_lookup.get(area, 1.0) if bias_lookup else 1.0
        records.append(
            {
                spatial_key: area,
                "time_bin": tbin,
                "complaint_count": count,
                "reporting_confidence": mean_conf,
                "coverage_bias": bias,
            }
        )

    agg = pd.DataFrame(records)
    logger.info(
        "Aggregated complaints into %d bins (window=%d min)",
        len(agg),
        time_window_minutes,
    )
    return agg
