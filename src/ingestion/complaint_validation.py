"""Complaint data validation module (Data Automation Prompt-3).

Implements honest validation that:
- Documents known biases without hiding them
- Flags suspicious patterns for human review
- NEVER drops records just because they look biased
- Propagates uncertainty to downstream inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComplaintValidationResult:
    """Result of complaint data validation."""

    valid: bool = False
    total_records: int = 0
    valid_records: int = 0
    temporal_coverage: Tuple[datetime, datetime] = field(
        default_factory=lambda: (datetime.min, datetime.max)
    )
    spatial_coverage: Dict = field(default_factory=dict)
    quality_flags: Dict[str, int] = field(default_factory=dict)
    bias_indicators: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


def validate_temporal_coverage(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    expected_start: Optional[datetime] = None,
    expected_end: Optional[datetime] = None,
    min_days: int = 7,
) -> Tuple[bool, Dict, List[str]]:
    """Validate temporal coverage of complaint data.

    Args:
        df: Complaint DataFrame
        timestamp_col: Name of timestamp column
        expected_start: Expected start date
        expected_end: Expected end date
        min_days: Minimum coverage in days

    Returns:
        (valid, stats, warnings)
    """
    warnings = []
    stats = {}

    if timestamp_col not in df.columns:
        return False, stats, [f"Missing timestamp column: {timestamp_col}"]

    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
    valid_ts = timestamps.dropna()

    stats["total_timestamps"] = len(timestamps)
    stats["valid_timestamps"] = len(valid_ts)
    stats["missing_timestamps"] = len(timestamps) - len(valid_ts)

    if len(valid_ts) == 0:
        return False, stats, ["No valid timestamps found"]

    actual_start = valid_ts.min().to_pydatetime()
    actual_end = valid_ts.max().to_pydatetime()
    coverage_days = (actual_end - actual_start).days

    stats["actual_start"] = actual_start.isoformat()
    stats["actual_end"] = actual_end.isoformat()
    stats["coverage_days"] = coverage_days

    # Check minimum coverage
    if coverage_days < min_days:
        warnings.append(
            f"Temporal coverage {coverage_days} days < minimum {min_days}"
        )

    # Check expected range
    if expected_start and actual_start > expected_start + timedelta(days=7):
        warnings.append(
            f"Data starts {(actual_start - expected_start).days} days late"
        )

    if expected_end and actual_end < expected_end - timedelta(days=7):
        warnings.append(
            f"Data ends {(expected_end - actual_end).days} days early"
        )

    # Look for gaps
    sorted_ts = valid_ts.sort_values()
    diffs = sorted_ts.diff()
    gap_threshold = timedelta(days=3)
    gaps = diffs[diffs > gap_threshold]

    if len(gaps) > 0:
        stats["gap_count"] = len(gaps)
        stats["max_gap_days"] = gaps.max().days
        warnings.append(
            f"Found {len(gaps)} temporal gaps > 3 days "
            f"(max: {gaps.max().days} days)"
        )

    valid = len(warnings) == 0 or (coverage_days >= min_days)
    return valid, stats, warnings


def validate_spatial_coverage(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    expected_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[bool, Dict, List[str]]:
    """Validate spatial coverage of complaint data.

    Args:
        df: Complaint DataFrame
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        expected_bounds: (min_lat, min_lon, max_lat, max_lon)

    Returns:
        (valid, stats, warnings)
    """
    warnings = []
    stats = {
        "has_point_coords": False,
        "has_ward_info": False,
        "has_area_info": False,
    }

    # Check for point coordinates
    has_lat = lat_col in df.columns
    has_lon = lon_col in df.columns

    if has_lat and has_lon:
        lats = pd.to_numeric(df[lat_col], errors="coerce")
        lons = pd.to_numeric(df[lon_col], errors="coerce")

        valid_coords = ~(lats.isna() | lons.isna())
        stats["has_point_coords"] = True
        stats["valid_coords"] = valid_coords.sum()
        stats["missing_coords"] = (~valid_coords).sum()

        if valid_coords.sum() > 0:
            stats["lat_range"] = (lats[valid_coords].min(), lats[valid_coords].max())
            stats["lon_range"] = (lons[valid_coords].min(), lons[valid_coords].max())

            # Check for suspicious coordinates
            suspicious = (
                (lats.abs() < 1) | (lats.abs() > 90) |
                (lons.abs() < 1) | (lons.abs() > 180)
            ) & valid_coords

            if suspicious.sum() > 0:
                stats["suspicious_coords"] = suspicious.sum()
                warnings.append(
                    f"{suspicious.sum()} coordinates appear invalid "
                    "(near 0 or out of range)"
                )

            # Check bounds if provided
            if expected_bounds:
                min_lat, min_lon, max_lat, max_lon = expected_bounds
                out_of_bounds = (
                    (lats < min_lat) | (lats > max_lat) |
                    (lons < min_lon) | (lons > max_lon)
                ) & valid_coords

                if out_of_bounds.sum() > 0:
                    stats["out_of_bounds"] = out_of_bounds.sum()
                    warnings.append(
                        f"{out_of_bounds.sum()} points outside expected bounds"
                    )

        missing_pct = stats["missing_coords"] / len(df) * 100
        if missing_pct > 30:
            warnings.append(
                f"{missing_pct:.1f}% of records missing coordinates"
            )

    # Check for ward/district info
    ward_cols = ["ward", "ward_name", "district", "zone"]
    for col in ward_cols:
        if col in df.columns:
            stats["has_ward_info"] = True
            stats["ward_col"] = col
            stats["unique_wards"] = df[col].nunique()
            break

    # Check for area/locality info
    area_cols = ["area", "locality", "address", "location_desc"]
    for col in area_cols:
        if col in df.columns:
            stats["has_area_info"] = True
            stats["area_col"] = col
            break

    # Valid if we have any spatial information
    has_spatial = (
        stats["has_point_coords"] or
        stats["has_ward_info"] or
        stats["has_area_info"]
    )

    if not has_spatial:
        warnings.append("No spatial information found in data")

    return has_spatial, stats, warnings


def detect_reporting_bias(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    include_full_analysis: bool = True,
) -> Dict:
    """Comprehensive reporting bias detection and quantification.

    This DOCUMENTS bias with statistical rigor - it does NOT filter it out.
    The bias information is passed to downstream inference for proper
    uncertainty propagation.

    Analyzes:
    - Temporal patterns (office hours, weekday, seasonal)
    - Spatial clustering and coverage gaps
    - Volume patterns and anomalies
    - Socioeconomic proxy indicators
    - Statistical significance of patterns

    Args:
        df: Complaint DataFrame
        timestamp_col: Name of timestamp column
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        include_full_analysis: Enable full statistical analysis

    Returns:
        Dict with bias indicators, metrics, and recommendations
    """
    bias_indicators = {
        "flags": [],
        "metrics": {},
        "spatial": {},
        "temporal": {},
        "statistical_tests": {},
        "recommendations": [],
        "severity": "low",  # low, medium, high, critical
    }

    severity_score = 0

    # === Temporal Bias Analysis ===
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        valid_ts = ts.dropna()

        if len(valid_ts) > 50:  # Need sufficient data
            hour_counts = valid_ts.dt.hour.value_counts().sort_index()
            dow_counts = valid_ts.dt.dayofweek.value_counts().sort_index()

            # Reindex to ensure all hours/days present
            hour_counts = hour_counts.reindex(range(24), fill_value=0)
            dow_counts = dow_counts.reindex(range(7), fill_value=0)

            # Office-hours bias (9am-5pm)
            office_hours_mask = (
                (valid_ts.dt.hour >= 9) & (valid_ts.dt.hour <= 17)
            )
            office_pct = office_hours_mask.sum() / len(valid_ts) * 100
            expected_office_pct = 8 / 24 * 100  # ~33.3%

            bias_indicators["temporal"]["office_hours_pct"] = round(
                office_pct, 1
            )
            bias_indicators["temporal"]["office_hours_expected"] = round(
                expected_office_pct, 1
            )
            bias_indicators["temporal"]["office_hours_ratio"] = round(
                office_pct / expected_office_pct, 2
            )

            # Peak hour analysis
            peak_hour = int(hour_counts.idxmax())
            peak_hour_pct = hour_counts.max() / len(valid_ts) * 100
            bias_indicators["temporal"]["peak_hour"] = peak_hour
            bias_indicators["temporal"]["peak_hour_pct"] = round(
                peak_hour_pct, 1
            )

            # Weekday vs weekend bias
            weekday_count = dow_counts[:5].sum()
            weekend_count = dow_counts[5:].sum()
            weekday_pct = weekday_count / len(valid_ts) * 100
            expected_weekday_pct = 5 / 7 * 100  # ~71.4%

            bias_indicators["temporal"]["weekday_pct"] = round(weekday_pct, 1)
            bias_indicators["temporal"]["weekday_expected"] = round(
                expected_weekday_pct, 1
            )
            bias_indicators["temporal"]["weekday_ratio"] = round(
                weekday_pct / expected_weekday_pct, 2
            )

            # Chi-square test for uniform distribution (hourly)
            if include_full_analysis:
                try:
                    from scipy.stats import chisquare

                    expected_hourly = np.full(24, len(valid_ts) / 24)
                    chi2_hour, p_hour = chisquare(
                        hour_counts.values, expected_hourly
                    )
                    bias_indicators["statistical_tests"]["hourly_uniformity"] = {
                        "chi2": round(float(chi2_hour), 2),
                        "p_value": round(float(p_hour), 6),
                        "significant": p_hour < 0.05,
                    }

                    expected_daily = np.full(7, len(valid_ts) / 7)
                    chi2_dow, p_dow = chisquare(
                        dow_counts.values, expected_daily
                    )
                    bias_indicators["statistical_tests"]["daily_uniformity"] = {
                        "chi2": round(float(chi2_dow), 2),
                        "p_value": round(float(p_dow), 6),
                        "significant": p_dow < 0.05,
                    }
                except ImportError:
                    pass

            # Flag significant temporal bias
            if office_pct > 70:
                bias_indicators["flags"].append(
                    f"Strong office-hours bias: {office_pct:.1f}% of "
                    f"complaints during 9am-5pm (expected ~33%)"
                )
                severity_score += 2
            elif office_pct > 55:
                bias_indicators["flags"].append(
                    f"Moderate office-hours bias: {office_pct:.1f}%"
                )
                severity_score += 1

            if weekday_pct > 85:
                bias_indicators["flags"].append(
                    f"Strong weekday bias: {weekday_pct:.1f}% on "
                    f"Mon-Fri (expected ~71%)"
                )
                severity_score += 2
            elif weekday_pct > 80:
                bias_indicators["flags"].append(
                    f"Moderate weekday bias: {weekday_pct:.1f}%"
                )
                severity_score += 1

            # Monthly/seasonal patterns
            if len(valid_ts) > 200:
                month_counts = valid_ts.dt.month.value_counts()
                month_cv = month_counts.std() / month_counts.mean()
                bias_indicators["temporal"]["monthly_cv"] = round(
                    float(month_cv), 3
                )

                if month_cv > 0.5:
                    bias_indicators["flags"].append(
                        f"High seasonal variation (CV={month_cv:.2f})"
                    )
                    severity_score += 1

    # === Spatial Bias Analysis ===
    if lat_col in df.columns and lon_col in df.columns:
        lats = pd.to_numeric(df[lat_col], errors="coerce")
        lons = pd.to_numeric(df[lon_col], errors="coerce")
        valid_mask = ~(lats.isna() | lons.isna())
        valid_count = valid_mask.sum()

        if valid_count > 20:
            valid_lats = lats[valid_mask]
            valid_lons = lons[valid_mask]

            # Basic spatial statistics
            lat_mean = float(valid_lats.mean())
            lon_mean = float(valid_lons.mean())
            lat_std = float(valid_lats.std())
            lon_std = float(valid_lons.std())

            bias_indicators["spatial"]["centroid"] = (
                round(lat_mean, 6), round(lon_mean, 6)
            )
            bias_indicators["spatial"]["lat_std"] = round(lat_std, 6)
            bias_indicators["spatial"]["lon_std"] = round(lon_std, 6)

            # Coverage area (bounding box)
            lat_range = valid_lats.max() - valid_lats.min()
            lon_range = valid_lons.max() - valid_lons.min()
            bias_indicators["spatial"]["lat_range_deg"] = round(
                float(lat_range), 4
            )
            bias_indicators["spatial"]["lon_range_deg"] = round(
                float(lon_range), 4
            )

            # Spatial clustering coefficient
            # CV < 0.3 indicates tight clustering
            spatial_cv = (lat_std + lon_std) / 2 / max(
                (lat_range + lon_range) / 2, 0.001
            )
            bias_indicators["spatial"]["clustering_cv"] = round(
                float(spatial_cv), 3
            )

            # Detect tight clustering (possible wealthy-area bias)
            if lat_std < 0.01 and lon_std < 0.01:
                bias_indicators["flags"].append(
                    "Very high spatial clustering (std < 0.01°) - "
                    "complaints may be from limited neighborhoods"
                )
                severity_score += 2
            elif lat_std < 0.02 and lon_std < 0.02:
                bias_indicators["flags"].append(
                    "Moderate spatial clustering - limited geographic spread"
                )
                severity_score += 1

            # Quadrant analysis (check for spatial bias)
            if include_full_analysis and valid_count > 100:
                # Divide into quadrants
                lat_median = valid_lats.median()
                lon_median = valid_lons.median()

                q1 = ((valid_lats >= lat_median) & 
                      (valid_lons >= lon_median)).sum()
                q2 = ((valid_lats >= lat_median) & 
                      (valid_lons < lon_median)).sum()
                q3 = ((valid_lats < lat_median) & 
                      (valid_lons < lon_median)).sum()
                q4 = ((valid_lats < lat_median) & 
                      (valid_lons >= lon_median)).sum()

                quadrant_counts = [q1, q2, q3, q4]
                quadrant_cv = np.std(quadrant_counts) / np.mean(quadrant_counts)

                bias_indicators["spatial"]["quadrant_counts"] = quadrant_counts
                bias_indicators["spatial"]["quadrant_cv"] = round(
                    float(quadrant_cv), 3
                )

                if quadrant_cv > 0.5:
                    bias_indicators["flags"].append(
                        f"Spatial distribution is uneven "
                        f"(quadrant CV={quadrant_cv:.2f})"
                    )
                    severity_score += 1

        # Missing coordinates analysis
        missing_coords = (~valid_mask).sum()
        missing_pct = missing_coords / len(df) * 100
        bias_indicators["spatial"]["missing_coords_pct"] = round(
            missing_pct, 1
        )

        if missing_pct > 50:
            bias_indicators["flags"].append(
                f"{missing_pct:.1f}% of records missing coordinates - "
                "geocoding may introduce bias"
            )
            severity_score += 2
        elif missing_pct > 20:
            bias_indicators["flags"].append(
                f"{missing_pct:.1f}% missing coordinates"
            )
            severity_score += 1

    # === Duplicate and Volume Analysis ===
    if "description" in df.columns:
        desc_counts = df["description"].value_counts()
        if len(desc_counts) > 0:
            most_common_count = desc_counts.iloc[0]
            unique_descs = len(desc_counts)
            unique_pct = unique_descs / len(df) * 100

            bias_indicators["metrics"]["unique_descriptions"] = unique_descs
            bias_indicators["metrics"]["unique_desc_pct"] = round(
                unique_pct, 1
            )

            if most_common_count > 10:
                dup_pct = most_common_count / len(df) * 100
                bias_indicators["metrics"]["duplicate_desc_pct"] = round(
                    dup_pct, 1
                )

                if dup_pct > 20:
                    bias_indicators["flags"].append(
                        f"{dup_pct:.1f}% of complaints have identical "
                        f"descriptions (possible duplicates or spam)"
                    )
                    severity_score += 2

    # === Complaint Type Analysis ===
    if "complaint_type" in df.columns:
        type_counts = df["complaint_type"].value_counts()
        n_types = len(type_counts)

        bias_indicators["metrics"]["unique_types"] = n_types

        if n_types > 0:
            top_type_pct = type_counts.iloc[0] / len(df) * 100
            type_cv = type_counts.std() / type_counts.mean() if len(
                type_counts) > 1 else 0

            bias_indicators["metrics"]["top_type_pct"] = round(
                top_type_pct, 1
            )
            bias_indicators["metrics"]["type_distribution_cv"] = round(
                float(type_cv), 3
            )

            if top_type_pct > 80:
                bias_indicators["flags"].append(
                    f"Type concentration: {top_type_pct:.1f}% are "
                    f"'{type_counts.index[0]}'"
                )
                severity_score += 1

    # === Severity Assessment ===
    if severity_score >= 6:
        bias_indicators["severity"] = "critical"
    elif severity_score >= 4:
        bias_indicators["severity"] = "high"
    elif severity_score >= 2:
        bias_indicators["severity"] = "medium"
    else:
        bias_indicators["severity"] = "low"

    bias_indicators["metrics"]["severity_score"] = severity_score

    # === Recommendations ===
    if bias_indicators["severity"] in ["high", "critical"]:
        bias_indicators["recommendations"].extend([
            "Consider applying inverse propensity weighting to correct bias",
            "Document bias thoroughly in any analysis using this data",
            "Seek supplementary data sources to validate findings",
        ])
    if any("office-hours" in f.lower() for f in bias_indicators["flags"]):
        bias_indicators["recommendations"].append(
            "Night/weekend underreporting likely - adjust confidence "
            "intervals for off-hours flooding estimates"
        )
    if any("clustering" in f.lower() for f in bias_indicators["flags"]):
        bias_indicators["recommendations"].append(
            "Spatial coverage gaps exist - inference in under-represented "
            "areas should have wider uncertainty bounds"
        )
    if any("missing coord" in f.lower() for f in bias_indicators["flags"]):
        bias_indicators["recommendations"].append(
            "High geocoding rate needed - consider ward-level aggregation "
            "as alternative to point-level analysis"
        )

    # Integrate with full bias quantification if available
    if include_full_analysis:
        try:
            from src.ingestion.complaint_validation_full import (
                validate_complaints_full,
            )
            full_result = validate_complaints_full(df)
            bias_score = full_result.bias_report.total_bias_score
            outliers = len(full_result.outlier_result.outlier_indices)
            bias_indicators["full_analysis"] = {
                "quality_score": full_result.quality_score.overall_score,
                "quality_grade": full_result.quality_score.grade,
                "bias_score": bias_score,
                "outlier_count": outliers,
            }
        except ImportError:
            pass

    return bias_indicators


def validate_complaints(
    df: pd.DataFrame,
    source_id: Optional[str] = None,
    expected_start: Optional[datetime] = None,
    expected_end: Optional[datetime] = None,
    expected_bounds: Optional[Tuple[float, float, float, float]] = None,
    enable_full_analysis: bool = True,
) -> ComplaintValidationResult:
    """Comprehensive validation of complaint data with full quality assessment.

    Enhanced with:
    - Multi-dimensional quality scoring
    - Statistical bias quantification
    - Outlier detection
    - Data integrity checks
    - Actionable recommendations

    Args:
        df: Complaint DataFrame
        source_id: Data source identifier
        expected_start: Expected start date
        expected_end: Expected end date
        expected_bounds: Expected (min_lat, min_lon, max_lat, max_lon)
        enable_full_analysis: Enable comprehensive statistical analysis

    Returns:
        ComplaintValidationResult with all findings
    """
    result = ComplaintValidationResult(total_records=len(df))

    if len(df) == 0:
        result.errors.append("Empty DataFrame")
        return result

    # === Temporal Validation ===
    temp_valid, temp_stats, temp_warnings = validate_temporal_coverage(
        df,
        expected_start=expected_start,
        expected_end=expected_end,
    )
    result.metadata["temporal"] = temp_stats
    result.warnings.extend(temp_warnings)

    if "actual_start" in temp_stats and "actual_end" in temp_stats:
        result.temporal_coverage = (
            datetime.fromisoformat(temp_stats["actual_start"]),
            datetime.fromisoformat(temp_stats["actual_end"]),
        )

    # === Spatial Validation ===
    spatial_valid, spatial_stats, spatial_warnings = validate_spatial_coverage(
        df, expected_bounds=expected_bounds
    )
    result.spatial_coverage = spatial_stats
    result.warnings.extend(spatial_warnings)

    # === Bias Detection (Enhanced) ===
    result.bias_indicators = detect_reporting_bias(
        df, include_full_analysis=enable_full_analysis
    )

    # Add bias severity to result
    if "severity" in result.bias_indicators:
        result.metadata["bias_severity"] = result.bias_indicators["severity"]

    # Add recommendations
    if result.bias_indicators.get("recommendations"):
        result.metadata["recommendations"] = result.bias_indicators[
            "recommendations"
        ]

    # === Quality Flags (Enhanced) ===
    quality_flags = {}

    # Missing timestamps
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        quality_flags["missing_timestamp"] = int(ts.isna().sum())
        quality_flags["timestamp_pct_valid"] = round(
            ts.notna().sum() / len(df) * 100, 1
        )

    # Missing coordinates
    if "latitude" in df.columns:
        lats = pd.to_numeric(df["latitude"], errors="coerce")
        quality_flags["missing_lat"] = int(lats.isna().sum())

    if "longitude" in df.columns:
        lons = pd.to_numeric(df["longitude"], errors="coerce")
        quality_flags["missing_lon"] = int(lons.isna().sum())

    # Invalid coordinates
    if "latitude" in df.columns and "longitude" in df.columns:
        lats = pd.to_numeric(df["latitude"], errors="coerce")
        lons = pd.to_numeric(df["longitude"], errors="coerce")
        valid_coords = ~(lats.isna() | lons.isna())

        if valid_coords.sum() > 0:
            invalid_range = (
                (lats.abs() > 90) | (lons.abs() > 180)
            ) & valid_coords
            quality_flags["invalid_coord_range"] = int(invalid_range.sum())

            suspicious = (
                (lats.abs() < 0.1) | (lons.abs() < 0.1)
            ) & valid_coords
            quality_flags["suspicious_coords"] = int(suspicious.sum())

            # Coordinate precision analysis
            if enable_full_analysis:
                lat_decimals = lats[valid_coords].apply(
                    lambda x: len(str(x).split(".")[-1])
                    if "." in str(x) else 0
                )
                quality_flags["avg_coord_precision"] = round(
                    float(lat_decimals.mean()), 1
                )

    # Missing complaint type
    if "complaint_type" in df.columns:
        quality_flags["missing_type"] = int(
            df["complaint_type"].isna().sum() +
            (df["complaint_type"].astype(str).str.strip() == "").sum()
        )

    # Missing description
    if "description" in df.columns:
        quality_flags["missing_description"] = int(
            df["description"].isna().sum() +
            (df["description"].astype(str).str.strip() == "").sum()
        )

    # Duplicate detection
    exact_dups = df.duplicated().sum()
    quality_flags["exact_duplicates"] = int(exact_dups)

    # Check for near-duplicates by key fields
    key_cols = []
    if "timestamp" in df.columns:
        key_cols.append("timestamp")
    if "latitude" in df.columns and "longitude" in df.columns:
        key_cols.extend(["latitude", "longitude"])
    if "complaint_type" in df.columns:
        key_cols.append("complaint_type")

    if len(key_cols) >= 2:
        near_dups = df.duplicated(subset=key_cols).sum()
        quality_flags["near_duplicates"] = int(near_dups)

    result.quality_flags = quality_flags

    # === Calculate Valid Records ===
    # A record is valid if it has timestamp AND (coordinates OR ward)
    valid_count = 0
    for _, row in df.iterrows():
        has_timestamp = pd.notna(row.get("timestamp"))
        if "timestamp" not in df.columns:
            has_timestamp = True

        has_location = False
        if "latitude" in df.columns and "longitude" in df.columns:
            has_location = pd.notna(row.get("latitude")) and pd.notna(
                row.get("longitude")
            )
        if not has_location and "ward" in df.columns:
            has_location = pd.notna(row.get("ward"))
        if "latitude" not in df.columns and "ward" not in df.columns:
            has_location = True

        if has_timestamp and has_location:
            valid_count += 1

    result.valid_records = valid_count

    # === Quality Score Calculation ===
    quality_score = 100.0

    # Deduct for missing data
    for field, col_name in [
        ("timestamp", "timestamp"),
        ("lat", "latitude"),
        ("lon", "longitude"),
        ("type", "complaint_type"),
    ]:
        if f"missing_{field}" in quality_flags:
            missing_pct = quality_flags[f"missing_{field}"] / len(df)
            quality_score -= min(missing_pct * 25, 15)

    # Deduct for duplicates
    if "exact_duplicates" in quality_flags:
        dup_pct = quality_flags["exact_duplicates"] / len(df)
        quality_score -= min(dup_pct * 20, 10)

    # Deduct for invalid data
    if "invalid_coord_range" in quality_flags:
        invalid_pct = quality_flags["invalid_coord_range"] / len(df)
        quality_score -= min(invalid_pct * 30, 15)

    # Deduct for bias severity
    bias_severity = result.bias_indicators.get("severity", "low")
    severity_deductions = {
        "low": 0,
        "medium": 5,
        "high": 10,
        "critical": 20,
    }
    quality_score -= severity_deductions.get(bias_severity, 0)

    # Deduct for warnings
    quality_score -= min(len(result.warnings) * 2, 10)

    result.metadata["quality_score"] = max(0, round(quality_score, 1))
    result.metadata["quality_grade"] = (
        "A" if quality_score >= 90 else
        "B" if quality_score >= 80 else
        "C" if quality_score >= 70 else
        "D" if quality_score >= 60 else
        "F"
    )

    # === Full Analysis Integration ===
    if enable_full_analysis:
        try:
            from src.ingestion.complaint_validation_full import (
                validate_complaints_full,
                generate_full_validation_report,
            )
            full_result = validate_complaints_full(df)
            result.metadata["full_analysis"] = {
                "quality_dimensions": {
                    dim_name: {
                        "score": dim.score,
                        "weight": dim.weight,
                    }
                    for dim_name, dim in full_result.quality_score.dimensions.items()
                },
                "overall_quality": full_result.quality_score.overall_score,
                "grade": full_result.quality_score.grade,
                "outlier_count": len(
                    full_result.outlier_result.outlier_indices
                ),
                "bias_score": full_result.bias_report.total_bias_score,
            }
        except ImportError:
            logger.debug(
                "Full validation module not available, using basic"
            )

    # === Overall Validity ===
    result.valid = (
        temp_valid and
        spatial_valid and
        len(result.errors) == 0 and
        quality_score >= 50
    )

    # Log results
    if result.bias_indicators.get("flags"):
        result.metadata["bias_flags"] = result.bias_indicators["flags"]
        logger.warning(
            "Bias indicators found: %s",
            "; ".join(result.bias_indicators["flags"][:3])
        )

    logger.info(
        "Validation: %d/%d valid, score=%.1f (%s), severity=%s",
        result.valid_records,
        result.total_records,
        quality_score,
        result.metadata["quality_grade"],
        bias_severity,
    )

    return result


def generate_validation_report(
    result: ComplaintValidationResult,
    output_path: Optional[Path] = None,
    include_recommendations: bool = True,
) -> str:
    """Generate comprehensive human-readable validation report.

    Enhanced with:
    - Quality score breakdown
    - Statistical test results
    - Bias severity assessment
    - Actionable recommendations
    - Data quality metrics

    Args:
        result: Validation result
        output_path: Optional path to save report
        include_recommendations: Include bias mitigation recommendations

    Returns:
        Report text
    """
    lines = [
        "=" * 70,
        "COMPLAINT DATA VALIDATION REPORT (COMPREHENSIVE)",
        "=" * 70,
        "",
    ]

    # === Summary Section ===
    quality_score = result.metadata.get("quality_score", 0)
    quality_grade = result.metadata.get("quality_grade", "N/A")
    bias_severity = result.metadata.get("bias_severity", "unknown")

    lines.extend([
        "SUMMARY",
        "-" * 50,
        f"  Total Records:     {result.total_records:,}",
        f"  Valid Records:     {result.valid_records:,} "
        f"({result.valid_records/max(result.total_records,1)*100:.1f}%)",
        f"  Quality Score:     {quality_score:.1f}/100 (Grade: {quality_grade})",
        f"  Bias Severity:     {bias_severity.upper()}",
        f"  Overall Status:    {'✓ VALID' if result.valid else '✗ INVALID'}",
        "",
    ])

    # === Temporal Coverage ===
    lines.extend([
        "TEMPORAL COVERAGE",
        "-" * 50,
    ])
    if result.temporal_coverage[0] != datetime.min:
        start, end = result.temporal_coverage
        days = (end - start).days
        lines.extend([
            f"  Start Date:  {start:%Y-%m-%d %H:%M}",
            f"  End Date:    {end:%Y-%m-%d %H:%M}",
            f"  Duration:    {days} days",
        ])
    else:
        lines.append("  No temporal data available")

    temp_stats = result.metadata.get("temporal", {})
    if temp_stats.get("gap_count"):
        lines.append(
            f"  Gaps Found:  {temp_stats['gap_count']} "
            f"(max: {temp_stats.get('max_gap_days', 'N/A')} days)"
        )
    lines.append("")

    # === Spatial Coverage ===
    lines.extend([
        "SPATIAL COVERAGE",
        "-" * 50,
    ])
    sc = result.spatial_coverage
    if sc.get("has_point_coords"):
        lines.extend([
            f"  Point Coordinates: {sc.get('valid_coords', 0):,} valid",
            f"  Missing Coords:    {sc.get('missing_coords', 0):,}",
        ])
        if "lat_range" in sc:
            lines.append(
                f"  Latitude Range:    {sc['lat_range'][0]:.4f} to "
                f"{sc['lat_range'][1]:.4f}"
            )
        if "lon_range" in sc:
            lines.append(
                f"  Longitude Range:   {sc['lon_range'][0]:.4f} to "
                f"{sc['lon_range'][1]:.4f}"
            )
    if sc.get("has_ward_info"):
        lines.append(f"  Wards/Districts:   {sc.get('unique_wards', 'N/A')}")
    if not sc.get("has_point_coords") and not sc.get("has_ward_info"):
        lines.append("  ⚠ No spatial information found")
    lines.append("")

    # === Quality Flags ===
    if result.quality_flags:
        lines.extend([
            "DATA QUALITY FLAGS",
            "-" * 50,
        ])
        for flag, count in sorted(result.quality_flags.items()):
            if isinstance(count, (int, float)) and count > 0:
                pct = count / max(result.total_records, 1) * 100
                status = "⚠" if pct > 10 else "○"
                lines.append(f"  {status} {flag}: {count:,} ({pct:.1f}%)")
        lines.append("")

    # === Bias Analysis ===
    if result.bias_indicators:
        bi = result.bias_indicators

        lines.extend([
            "BIAS ANALYSIS",
            "-" * 50,
        ])

        # Temporal bias
        if bi.get("temporal"):
            tb = bi["temporal"]
            lines.append("  Temporal Patterns:")
            if "office_hours_pct" in tb:
                lines.append(
                    f"    Office Hours (9-5): {tb['office_hours_pct']:.1f}% "
                    f"(expected ~33%)"
                )
            if "weekday_pct" in tb:
                lines.append(
                    f"    Weekdays:           {tb['weekday_pct']:.1f}% "
                    f"(expected ~71%)"
                )
            if "peak_hour" in tb:
                lines.append(f"    Peak Hour:          {tb['peak_hour']}:00")

        # Spatial bias
        if bi.get("spatial"):
            sb = bi["spatial"]
            lines.append("  Spatial Patterns:")
            if "centroid" in sb:
                lines.append(
                    f"    Centroid:           {sb['centroid'][0]:.4f}, "
                    f"{sb['centroid'][1]:.4f}"
                )
            if "clustering_cv" in sb:
                lines.append(
                    f"    Clustering CV:      {sb['clustering_cv']:.3f}"
                )
            if "missing_coords_pct" in sb:
                lines.append(
                    f"    Missing Coords:     {sb['missing_coords_pct']:.1f}%"
                )

        # Statistical tests
        if bi.get("statistical_tests"):
            lines.append("  Statistical Tests:")
            for test_name, test_result in bi["statistical_tests"].items():
                sig = "✗" if test_result.get("significant") else "✓"
                lines.append(
                    f"    {sig} {test_name}: χ²={test_result.get('chi2', 'N/A')}, "
                    f"p={test_result.get('p_value', 'N/A')}"
                )

        lines.append("")

    # === Bias Flags ===
    if result.bias_indicators.get("flags"):
        lines.extend([
            "BIAS INDICATORS (documented, not filtered)",
            "-" * 50,
        ])
        for i, flag in enumerate(result.bias_indicators["flags"], 1):
            lines.append(f"  {i}. ⚠ {flag}")
        lines.append("")

    # === Recommendations ===
    if include_recommendations and result.bias_indicators.get("recommendations"):
        lines.extend([
            "RECOMMENDATIONS",
            "-" * 50,
        ])
        for i, rec in enumerate(result.bias_indicators["recommendations"], 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    # === Full Analysis (if available) ===
    if result.metadata.get("full_analysis"):
        fa = result.metadata["full_analysis"]
        lines.extend([
            "DETAILED QUALITY DIMENSIONS",
            "-" * 50,
        ])
        if "quality_dimensions" in fa:
            for dim_name, dim_data in fa["quality_dimensions"].items():
                score = dim_data.get("score", 0)
                bar_len = int(score / 5)  # 20 char max
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(
                    f"  {dim_name:15} [{bar}] {score:.1f}"
                )
        if "outlier_count" in fa:
            lines.append(f"  Outliers Detected:   {fa['outlier_count']}")
        lines.append("")

    # === Warnings ===
    if result.warnings:
        lines.extend([
            "WARNINGS",
            "-" * 50,
        ])
        for warning in result.warnings:
            lines.append(f"  • {warning}")
        lines.append("")

    # === Errors ===
    if result.errors:
        lines.extend([
            "ERRORS",
            "-" * 50,
        ])
        for error in result.errors:
            lines.append(f"  ✗ {error}")
        lines.append("")

    # === Footer ===
    lines.extend([
        "=" * 70,
        f"Report generated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC",
        "=" * 70,
    ])

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report)
        logger.info("Validation report saved to %s", output_path)

    return report
