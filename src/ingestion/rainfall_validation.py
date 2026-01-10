"""Rainfall data validation (Data Automation Prompt-2).

Validates:
- Units (mm/hr vs mm)
- Timestamp continuity
- Impossible rainfall values
- Stuck gauge behavior

Flags but does NOT delete suspect data - preserves for downstream
probabilistic handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationFlag:
    """A flag indicating a data quality issue."""

    flag_type: str
    severity: str  # "error", "warning", "info"
    description: str
    affected_rows: List[int] = field(default_factory=list)
    affected_fraction: float = 0.0


@dataclass
class RainfallValidationResult:
    """Complete validation result for a rainfall dataset."""

    is_valid: bool = True
    file_path: Optional[Path] = None
    row_count: int = 0
    valid_row_count: int = 0
    temporal_range: Tuple[str, str] = ("", "")
    temporal_resolution_minutes: Optional[float] = None
    flags: List[ValidationFlag] = field(default_factory=list)
    station_coverage: Dict[str, float] = field(default_factory=dict)
    summary_stats: Dict = field(default_factory=dict)

    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.flags)

    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.flags)


class RainfallValidator:
    """Comprehensive rainfall data validator.

    Performs checks without deleting data - flags issues for
    downstream probabilistic handling.
    """

    def __init__(
        self,
        expected_unit: str = "mm",
        max_intensity_mm_hr: float = 300.0,
        min_valid_value: float = 0.0,
        max_gap_hours: float = 6.0,
        stuck_gauge_threshold: int = 24,
    ):
        """Initialize validator with thresholds.

        Args:
            expected_unit: Expected rainfall unit
            max_intensity_mm_hr: Maximum realistic intensity
            min_valid_value: Minimum valid value (negative = error)
            max_gap_hours: Maximum acceptable timestamp gap
            stuck_gauge_threshold: Consecutive identical readings
                to flag as stuck gauge
        """
        self.expected_unit = expected_unit
        self.max_intensity = max_intensity_mm_hr
        self.min_valid = min_valid_value
        self.max_gap_hours = max_gap_hours
        self.stuck_threshold = stuck_gauge_threshold

    def validate(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        value_col: str = "rainfall_mm",
        station_col: Optional[str] = None,
    ) -> RainfallValidationResult:
        """Validate rainfall data comprehensively.

        Args:
            data: DataFrame with rainfall data
            timestamp_col: Name of timestamp column
            value_col: Name of rainfall value column
            station_col: Optional station identifier column

        Returns:
            RainfallValidationResult with all flags and metrics
        """
        result = RainfallValidationResult(row_count=len(data))

        if len(data) == 0:
            result.is_valid = False
            result.flags.append(ValidationFlag(
                flag_type="empty_data",
                severity="error",
                description="Dataset is empty",
            ))
            return result

        # Validate columns exist
        if timestamp_col not in data.columns:
            result.is_valid = False
            result.flags.append(ValidationFlag(
                flag_type="missing_column",
                severity="error",
                description=f"Missing timestamp column: {timestamp_col}",
            ))
            return result

        if value_col not in data.columns:
            result.is_valid = False
            result.flags.append(ValidationFlag(
                flag_type="missing_column",
                severity="error",
                description=f"Missing value column: {value_col}",
            ))
            return result

        # Parse timestamps
        df = data.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        # Check timestamp parsing
        ts_invalid = df[timestamp_col].isna()
        if ts_invalid.any():
            invalid_rows = df.index[ts_invalid].tolist()
            result.flags.append(ValidationFlag(
                flag_type="invalid_timestamp",
                severity="warning",
                description="Some timestamps could not be parsed",
                affected_rows=invalid_rows[:100],  # Cap list size
                affected_fraction=ts_invalid.mean(),
            ))

        # Work with valid timestamps only
        df_valid = df[~ts_invalid].copy()
        result.valid_row_count = len(df_valid)

        if len(df_valid) == 0:
            result.is_valid = False
            result.flags.append(ValidationFlag(
                flag_type="no_valid_timestamps",
                severity="error",
                description="No valid timestamps in dataset",
            ))
            return result

        # Temporal range
        ts_min = df_valid[timestamp_col].min()
        ts_max = df_valid[timestamp_col].max()
        result.temporal_range = (
            ts_min.strftime("%Y-%m-%d %H:%M"),
            ts_max.strftime("%Y-%m-%d %H:%M"),
        )

        # Check temporal resolution
        self._check_temporal_resolution(df_valid, timestamp_col, result)

        # Check for gaps
        self._check_temporal_gaps(df_valid, timestamp_col, result)

        # Validate values
        self._check_value_range(df_valid, value_col, result)

        # Check for stuck gauges
        if station_col and station_col in df_valid.columns:
            self._check_stuck_gauges_multi(
                df_valid, timestamp_col, value_col, station_col, result
            )
        else:
            self._check_stuck_gauge(df_valid, value_col, result)

        # Check units (heuristic)
        self._check_units(df_valid, value_col, result)

        # Station coverage
        if station_col and station_col in df_valid.columns:
            self._check_station_coverage(
                df_valid, timestamp_col, station_col, result
            )

        # Summary statistics
        values = pd.to_numeric(df_valid[value_col], errors="coerce")
        valid_values = values.dropna()
        if len(valid_values) > 0:
            result.summary_stats = {
                "min": float(valid_values.min()),
                "max": float(valid_values.max()),
                "mean": float(valid_values.mean()),
                "median": float(valid_values.median()),
                "std": float(valid_values.std()),
                "zeros_fraction": float((valid_values == 0).mean()),
                "nonzero_count": int((valid_values > 0).sum()),
            }

        # Overall validity (has no errors)
        result.is_valid = not result.has_errors()

        return result

    def _check_temporal_resolution(
        self,
        df: pd.DataFrame,
        ts_col: str,
        result: RainfallValidationResult,
    ) -> None:
        """Detect temporal resolution of the data."""
        df_sorted = df.sort_values(ts_col)
        diffs = df_sorted[ts_col].diff().dropna()

        if len(diffs) > 0:
            # Get mode of time differences
            diff_minutes = diffs.dt.total_seconds() / 60
            resolution = diff_minutes.mode()
            if len(resolution) > 0:
                result.temporal_resolution_minutes = float(resolution.iloc[0])
                logger.info(
                    "Detected temporal resolution: %.1f minutes",
                    result.temporal_resolution_minutes,
                )

    def _check_temporal_gaps(
        self,
        df: pd.DataFrame,
        ts_col: str,
        result: RainfallValidationResult,
    ) -> None:
        """Check for gaps in timestamp sequence."""
        df_sorted = df.sort_values(ts_col)
        diffs = df_sorted[ts_col].diff()

        gap_threshold = timedelta(hours=self.max_gap_hours)
        gaps = diffs > gap_threshold

        if gaps.any():
            gap_rows = df_sorted.index[gaps].tolist()
            gap_sizes = diffs[gaps].dt.total_seconds() / 3600  # hours

            result.flags.append(ValidationFlag(
                flag_type="temporal_gaps",
                severity="warning",
                description=(
                    f"Found {gaps.sum()} gaps > {self.max_gap_hours} hours. "
                    f"Max gap: {gap_sizes.max():.1f} hours"
                ),
                affected_rows=gap_rows[:100],
                affected_fraction=gaps.mean(),
            ))

    def _check_value_range(
        self,
        df: pd.DataFrame,
        val_col: str,
        result: RainfallValidationResult,
    ) -> None:
        """Check for impossible or suspect values."""
        values = pd.to_numeric(df[val_col], errors="coerce")

        # Negative values (error)
        negative = values < self.min_valid
        if negative.any():
            result.flags.append(ValidationFlag(
                flag_type="negative_values",
                severity="error",
                description="Negative rainfall values found",
                affected_rows=df.index[negative].tolist()[:100],
                affected_fraction=negative.mean(),
            ))

        # Extreme values (warning)
        extreme = values > self.max_intensity
        if extreme.any():
            max_val = values.max()
            result.flags.append(ValidationFlag(
                flag_type="extreme_values",
                severity="warning",
                description=(
                    f"Values > {self.max_intensity} mm/hr found. "
                    f"Max: {max_val:.1f}"
                ),
                affected_rows=df.index[extreme].tolist()[:100],
                affected_fraction=extreme.mean(),
            ))

        # NaN values (info)
        nan_mask = values.isna()
        if nan_mask.any():
            result.flags.append(ValidationFlag(
                flag_type="missing_values",
                severity="info",
                description="Missing/non-numeric values found",
                affected_rows=df.index[nan_mask].tolist()[:100],
                affected_fraction=nan_mask.mean(),
            ))

    def _check_stuck_gauge(
        self,
        df: pd.DataFrame,
        val_col: str,
        result: RainfallValidationResult,
    ) -> None:
        """Detect stuck gauge behavior (identical consecutive readings)."""
        values = pd.to_numeric(df[val_col], errors="coerce")

        # Find runs of identical values
        run_lengths = []
        current_run = 1
        prev_val = None

        for val in values:
            if pd.isna(val):
                continue
            if val == prev_val:
                current_run += 1
            else:
                if current_run > 1:
                    run_lengths.append(current_run)
                current_run = 1
            prev_val = val

        if current_run > 1:
            run_lengths.append(current_run)

        # Flag long runs
        long_runs = [r for r in run_lengths if r >= self.stuck_threshold]
        if long_runs:
            result.flags.append(ValidationFlag(
                flag_type="stuck_gauge",
                severity="warning",
                description=(
                    f"Possible stuck gauge: {len(long_runs)} runs of "
                    f">={self.stuck_threshold} identical readings. "
                    f"Longest: {max(long_runs)}"
                ),
                affected_fraction=sum(long_runs) / len(values),
            ))

    def _check_stuck_gauges_multi(
        self,
        df: pd.DataFrame,
        ts_col: str,
        val_col: str,
        station_col: str,
        result: RainfallValidationResult,
    ) -> None:
        """Check stuck gauge per station."""
        stuck_stations = []

        for station in df[station_col].unique():
            station_df = df[df[station_col] == station]
            values = pd.to_numeric(station_df[val_col], errors="coerce")

            # Check for long identical runs
            same = (values == values.shift()).fillna(False)
            run_lengths = same.groupby(
                (~same).cumsum()
            ).cumsum()

            if run_lengths.max() >= self.stuck_threshold:
                stuck_stations.append(str(station))

        if stuck_stations:
            result.flags.append(ValidationFlag(
                flag_type="stuck_gauge_stations",
                severity="warning",
                description=(
                    f"Possible stuck gauges at {len(stuck_stations)} "
                    f"stations: {', '.join(stuck_stations[:5])}"
                ),
            ))

    def _check_units(
        self,
        df: pd.DataFrame,
        val_col: str,
        result: RainfallValidationResult,
    ) -> None:
        """Heuristic check for unit consistency."""
        values = pd.to_numeric(df[val_col], errors="coerce").dropna()

        if len(values) == 0:
            return

        # If max is very small (<1) and we expect mm, might be in meters
        if values.max() < 1 and self.expected_unit == "mm":
            result.flags.append(ValidationFlag(
                flag_type="unit_suspect",
                severity="warning",
                description=(
                    f"Max value {values.max():.4f} is very small for mm. "
                    "Check if data is in meters or inches."
                ),
            ))

        # If max is very large (>1000) might be cumulative
        if values.max() > 1000:
            result.flags.append(ValidationFlag(
                flag_type="unit_suspect",
                severity="warning",
                description=(
                    f"Max value {values.max():.1f} is very large. "
                    "Check if data is cumulative or in different units."
                ),
            ))

    def _check_station_coverage(
        self,
        df: pd.DataFrame,
        ts_col: str,
        station_col: str,
        result: RainfallValidationResult,
    ) -> None:
        """Check coverage per station."""
        # Get expected timestamp range
        ts_range = df[ts_col].max() - df[ts_col].min()
        expected_records = ts_range.total_seconds() / 3600  # hourly

        for station in df[station_col].unique():
            station_df = df[df[station_col] == station]
            coverage = len(station_df) / max(expected_records, 1)
            result.station_coverage[str(station)] = min(coverage, 1.0)

        # Flag low coverage stations
        low_coverage = [
            s for s, c in result.station_coverage.items() if c < 0.5
        ]
        if low_coverage:
            result.flags.append(ValidationFlag(
                flag_type="low_station_coverage",
                severity="warning",
                description=(
                    f"{len(low_coverage)} stations have <50% coverage: "
                    f"{', '.join(low_coverage[:5])}"
                ),
            ))


def validate_rainfall_file(
    file_path: Path,
    timestamp_col: str = "timestamp",
    value_col: str = "rainfall_mm",
) -> RainfallValidationResult:
    """Convenience function to validate a rainfall CSV file.

    Args:
        file_path: Path to CSV file
        timestamp_col: Timestamp column name
        value_col: Rainfall value column name

    Returns:
        Validation result with all checks performed
    """
    if not file_path.exists():
        result = RainfallValidationResult(is_valid=False)
        result.flags.append(ValidationFlag(
            flag_type="file_not_found",
            severity="error",
            description=f"File not found: {file_path}",
        ))
        return result

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        result = RainfallValidationResult(is_valid=False)
        result.flags.append(ValidationFlag(
            flag_type="read_error",
            severity="error",
            description=f"Failed to read file: {e}",
        ))
        return result

    # Auto-detect columns if not found
    if timestamp_col not in df.columns:
        for col in ["time", "datetime", "DATE", "date"]:
            if col in df.columns:
                timestamp_col = col
                break

    if value_col not in df.columns:
        for col in ["precipitation", "rain", "precip", "rainfall"]:
            if col in df.columns:
                value_col = col
                break

    validator = RainfallValidator()
    result = validator.validate(df, timestamp_col, value_col)
    result.file_path = file_path

    return result
