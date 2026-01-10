"""Robust rainfall ingestion for AWS gauges and radar NetCDF.

This module implements the Phase 2 ingestion requirements:
- Strict schema and unit validation for AWS gauges (CSV/Excel).
- Radar NetCDF dimension validation, missing-scan detection, and reliability proxy.
- Temporal harmonization with tolerance-aware nearest neighbor (no interpolation beyond tolerance).
- Spatial helpers that preserve AWS point geometry and radar grid structure.
- Explicit uncertainty flags for downstream probabilistic workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ..config.parameters import get_default_parameters
from ..utils.validation_utils import assert_non_empty, require_columns

logger = logging.getLogger(__name__)


TimestampCols = [
    "timestamp",
    "time",
    "datetime",
    "obs_time",
]

ValueCols = [
    "rainfall",
    "rainfall_mm",
    "rain",
    "precip",
    "precip_mm",
]


@dataclass
class AWSIngestionResult:
    data: pd.DataFrame
    flags: pd.DataFrame
    metadata: dict


class AWSIngestor:
    """Ingest IMD AWS rainfall with schema, unit, and quality validation."""

    def __init__(self, expected_unit: str = "mm"):
        self.expected_unit = expected_unit
        self.params = get_default_parameters().rainfall

    def load_files(self, files: Iterable[Path | str]) -> AWSIngestionResult:
        frames: List[pd.DataFrame] = []
        source_files: List[str] = []
        for f in files:
            df = self._read_file(Path(f))
            frames.append(df)
            source_files.append(str(f))
        if not frames:
            raise ValueError("No AWS files provided")
        combined = pd.concat(frames, ignore_index=True)
        result = self._validate_and_normalize(combined)
        result.metadata["source_files"] = source_files
        return result

    def _read_file(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"AWS file not found: {path}")
        if path.suffix.lower() in {".csv"}:
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".xls", ".xlsx"}:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported AWS format: {path.suffix}")
        assert_non_empty(df, context=f"AWS file {path}")
        return df

    def _find_timestamp_col(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if col.lower() in TimestampCols:
                return col
        # Fallback: choose the first datetime-like column with minimal NaNs after parsing
        parsed_candidates: List[Tuple[str, int]] = []
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            n_invalid = parsed.isna().sum()
            if n_invalid < len(parsed):
                parsed_candidates.append((col, n_invalid))
        if parsed_candidates:
            parsed_candidates.sort(key=lambda x: x[1])
            best = parsed_candidates[0][0]
            logger.warning("Using fallback timestamp column '%s' based on parseability", best)
            return best
        raise ValueError("No timestamp-like column found in AWS data")

    def _find_value_col(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if col.lower() in ValueCols:
                return col
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 1:
            logger.warning("Using fallback numeric column '%s' as rainfall value", numeric_cols[0])
            return numeric_cols[0]
        raise ValueError("No rainfall value column found in AWS data")

    def _normalize_units(self, series: pd.Series) -> Tuple[pd.Series, str]:
        unit_hint = self.expected_unit.lower()
        series = pd.to_numeric(series, errors="coerce")

        if unit_hint not in {"mm", "cm"}:
            unit_hint = "mm"

        if unit_hint == "cm":
            logger.info("Converting AWS rainfall from cm to mm based on expected_unit")
            return series * 10.0, "mm"

        # Heuristic check: if median is very small (<2) but max < 30, it might still be mm for short-step.
        if series.max(skipna=True) > 500:
            logger.warning(
                "AWS values exceed 500; units may be cumulative or incorrect; leaving as-is but flagged"
            )
        return series, "mm"

    def _validate_and_normalize(self, df: pd.DataFrame) -> AWSIngestionResult:
        ts_col = self._find_timestamp_col(df)
        val_col = self._find_value_col(df)

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df["rainfall_raw"] = pd.to_numeric(df[val_col], errors="coerce")

        missing_ts = df["timestamp"].isna()
        missing_val = df["rainfall_raw"].isna()

        duplicates = df.duplicated(subset=["timestamp"], keep=False)
        constant_seq = self._detect_constant_sequences(df["rainfall_raw"], window=5)

        rainfall_mm, unit = self._normalize_units(df["rainfall_raw"])

        negative = rainfall_mm < 0
        extreme = rainfall_mm > 200.0  # mm/hr level extremes

        measurement_sd = 0.2  # sensor precision heuristic in mm
        uncertainty_mm = np.where(missing_val | negative | extreme, np.nan, measurement_sd)

        df_out = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "rainfall_mm": rainfall_mm,
                "missing_flag": missing_ts | missing_val,
                "quality_flag": "ok",
                "uncertainty_mm": uncertainty_mm,
            }
        )

        df_out.loc[missing_ts, "quality_flag"] = "missing_timestamp"
        df_out.loc[missing_val, "quality_flag"] = "missing_value"
        df_out.loc[duplicates, "quality_flag"] = "duplicate_timestamp"
        df_out.loc[constant_seq, "quality_flag"] = "constant_sequence"
        df_out.loc[negative, "quality_flag"] = "negative_value"
        df_out.loc[extreme, "quality_flag"] = "extreme_value"

        flags = df_out[["timestamp", "quality_flag", "missing_flag"]].copy()

        logger.info(
            "AWS ingestion: %d records | missing=%d | duplicates=%d | constant=%d | extreme=%d",
            len(df_out),
            int(missing_ts.sum() + missing_val.sum()),
            int(duplicates.sum()),
            int(constant_seq.sum()),
            int(extreme.sum()),
        )

        return AWSIngestionResult(
            data=df_out,
            flags=flags,
            metadata={"unit": unit, "expected_unit": self.expected_unit},
        )

    @staticmethod
    def _detect_constant_sequences(values: pd.Series, window: int = 5) -> pd.Series:
        """Flag simple stuck-gauge patterns (repeated identical values)."""
        vals = values.fillna(method="ffill")
        rolling = vals.rolling(window=window, min_periods=window)
        return (rolling.max() - rolling.min()) == 0


@dataclass
class RadarIngestionResult:
    dataset: xr.Dataset
    reliability: xr.DataArray


class RadarIngestor:
    """Ingest radar NetCDF with dimension validation and uncertainty proxy."""

    def __init__(self, rain_var: str = "rain_rate"):
        self.rain_var = rain_var
        self.params = get_default_parameters().rainfall

    def load_netcdf(self, path: Path | str) -> RadarIngestionResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Radar file not found: {path}")
        ds = xr.open_dataset(path)
        self._validate_dataset(ds)
        ds = ds.copy()

        missing_scan = ds[self.rain_var].isnull().all(dim=("latitude", "longitude"))
        gaps = self._detect_time_gaps(ds)

        intensity = ds[self.rain_var]
        high_bias = intensity > (self.params.intensity_threshold_mm_hr * 4)
        reliability = xr.where(high_bias, 0.5, 0.9)
        reliability = reliability.where(~missing_scan, other=0.0)

        ds["reliability"] = reliability
        ds["missing_scan"] = missing_scan
        ds["temporal_gap"] = gaps

        logger.info(
            "Radar ingestion: %d times | missing scans=%d | gaps=%d",
            ds.dims.get("time", 0),
            int(missing_scan.sum()),
            int(gaps.sum()),
        )

        return RadarIngestionResult(dataset=ds, reliability=reliability)

    def _validate_dataset(self, ds: xr.Dataset) -> None:
        required_dims = {"time", "latitude", "longitude"}
        missing_dims = required_dims - set(ds.dims)
        if missing_dims:
            raise ValueError(f"Radar dataset missing dimensions: {missing_dims}")
        if self.rain_var not in ds:
            raise ValueError(f"Radar dataset missing rain variable '{self.rain_var}'")

    def _detect_time_gaps(self, ds: xr.Dataset) -> xr.DataArray:
        times = pd.to_datetime(ds["time"].values)
        if len(times) < 2:
            return xr.DataArray(np.zeros(len(times), dtype=bool), dims=["time"], coords={"time": ds["time"]})
        diffs = np.diff(times) / np.timedelta64(1, "m")
        expected = np.median(diffs)
        gap = np.insert(diffs > (expected * 1.5), 0, False)
        return xr.DataArray(gap, dims=["time"], coords={"time": ds["time"]})


def harmonize_time(
    aws_df: pd.DataFrame,
    radar_ds: xr.Dataset,
    tolerance_minutes: int = 10,
) -> xr.Dataset:
    """Align AWS and radar to a common time grid with tolerance-aware nearest matching.

    - If no radar time within tolerance, AWS point is marked as unaligned.
    - No interpolation is performed.
    """

    require_columns(aws_df, ["timestamp", "rainfall_mm"], context="aws_df for harmonization")

    aws_times = pd.to_datetime(aws_df["timestamp"])
    radar_times = pd.to_datetime(radar_ds["time"].values)

    if len(radar_times) == 0:
        raise ValueError("Radar dataset has no time dimension")

    tol = pd.Timedelta(minutes=tolerance_minutes)
    aligned_values: List[float] = []
    alignment_quality: List[str] = []
    aws_missing_flags = aws_df.get("missing_flag", pd.Series([False] * len(aws_times)))

    for rt in radar_times:
        diffs = np.abs(aws_times - rt)
        idx = diffs.argmin()
        if diffs[idx] <= tol:
            aligned_values.append(float(aws_df.iloc[idx]["rainfall_mm"]))
            alignment_quality.append("aligned" if not aws_missing_flags.iloc[idx] else "aws_missing")
        else:
            aligned_values.append(np.nan)
            alignment_quality.append("no_match")

    aws_da = xr.DataArray(
        aligned_values,
        dims=["time"],
        coords={"time": radar_times},
        name="aws_rainfall_mm",
    )
    quality_da = xr.DataArray(
        alignment_quality,
        dims=["time"],
        coords={"time": radar_times},
        name="alignment_quality",
    )

    unified = radar_ds.copy()
    unified["aws_rainfall_mm"] = aws_da
    unified["alignment_quality"] = quality_da

    logger.info(
        "Time harmonization complete: aligned=%d, aws_missing=%d, no_match=%d",
        alignment_quality.count("aligned"),
        alignment_quality.count("aws_missing"),
        alignment_quality.count("no_match"),
    )

    return unified


def summarize_uncertainty(aws_result: AWSIngestionResult, radar_result: RadarIngestionResult) -> dict:
    """Provide a quick uncertainty summary for downstream decision-making."""
    aws_flags = aws_result.flags["quality_flag"].value_counts().to_dict()
    radar_missing = int(radar_result.dataset.get("missing_scan", xr.DataArray()).sum()) if "missing_scan" in radar_result.dataset else 0
    radar_gaps = int(radar_result.dataset.get("temporal_gap", xr.DataArray()).sum()) if "temporal_gap" in radar_result.dataset else 0

    return {
        "aws_flags": aws_flags,
        "radar_missing_scans": radar_missing,
        "radar_temporal_gaps": radar_gaps,
    }


def aws_points_as_geodf(
    aws_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Represent AWS observations as point geometry without rasterizing.

    Raises if latitude/longitude are unavailable to avoid silent spatial assumptions.
    """

    require_columns(aws_df, [lat_col, lon_col], context="aws_df spatial conversion")
    gdf = gpd.GeoDataFrame(
        aws_df.copy(),
        geometry=gpd.points_from_xy(aws_df[lon_col], aws_df[lat_col], crs=crs),
    )
    return gdf


def radar_grid_metadata(radar_ds: xr.Dataset) -> dict:
    """Return radar grid description for downstream fusion without altering grid."""

    if not {"latitude", "longitude"}.issubset(set(radar_ds.dims)):
        raise ValueError("Radar dataset lacks latitude/longitude dimensions for grid metadata")

    return {
        "lat_min": float(radar_ds["latitude"].min()),
        "lat_max": float(radar_ds["latitude"].max()),
        "lon_min": float(radar_ds["longitude"].min()),
        "lon_max": float(radar_ds["longitude"].max()),
        "n_lat": radar_ds.dims["latitude"],
        "n_lon": radar_ds.dims["longitude"],
    }
