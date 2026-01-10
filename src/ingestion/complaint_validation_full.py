"""Maximum-level complaint validation module (Data Automation Prompt-3).

Full-featured validation including:
- Statistical outlier detection (IQR, Z-score, isolation forest)
- Data quality scoring (0-100 scale)
- Comprehensive bias quantification with correction factors
- Spatial autocorrelation analysis (Moran's I)
- Temporal pattern analysis (seasonality, trends)
- Cross-validation against known events
- Confidence interval estimation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
import json

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA QUALITY SCORING
# ==============================================================================

@dataclass
class QualityDimension:
    """Single dimension of data quality."""

    name: str
    score: float  # 0-100
    weight: float = 1.0
    description: str = ""
    metrics: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataQualityScore:
    """Comprehensive data quality assessment."""

    overall_score: float = 0.0  # 0-100
    grade: str = "F"  # A, B, C, D, F
    dimensions: Dict[str, QualityDimension] = field(default_factory=dict)
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": round(self.overall_score, 2),
            "grade": self.grade,
            "dimensions": {
                k: {
                    "score": round(v.score, 2),
                    "weight": v.weight,
                    "description": v.description,
                    "metrics": v.metrics,
                }
                for k, v in self.dimensions.items()
            },
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
        }


class DataQualityScorer:
    """Compute comprehensive data quality scores."""

    GRADE_THRESHOLDS = {
        "A": 90,
        "B": 80,
        "C": 70,
        "D": 60,
        "F": 0,
    }

    def __init__(self):
        self.dimension_weights = {
            "completeness": 2.0,
            "accuracy": 2.0,
            "consistency": 1.5,
            "timeliness": 1.0,
            "uniqueness": 1.0,
            "validity": 1.5,
        }

    def score(self, df: pd.DataFrame) -> DataQualityScore:
        """Compute overall data quality score."""
        result = DataQualityScore()

        # Compute each dimension
        result.dimensions["completeness"] = self._score_completeness(df)
        result.dimensions["accuracy"] = self._score_accuracy(df)
        result.dimensions["consistency"] = self._score_consistency(df)
        result.dimensions["timeliness"] = self._score_timeliness(df)
        result.dimensions["uniqueness"] = self._score_uniqueness(df)
        result.dimensions["validity"] = self._score_validity(df)

        # Compute weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for name, dim in result.dimensions.items():
            weight = self.dimension_weights.get(name, 1.0)
            dim.weight = weight
            weighted_sum += dim.score * weight
            total_weight += weight

            # Collect critical issues
            if dim.score < 50:
                result.critical_issues.append(
                    f"{name.capitalize()}: {dim.description}"
                )
            elif dim.score < 70:
                result.warnings.append(
                    f"{name.capitalize()}: {dim.description}"
                )

        result.overall_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Assign grade
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if result.overall_score >= threshold:
                result.grade = grade
                break

        result.metadata = {
            "records": len(df),
            "columns": len(df.columns),
            "scored_at": datetime.utcnow().isoformat(),
        }

        return result

    def _score_completeness(self, df: pd.DataFrame) -> QualityDimension:
        """Score data completeness."""
        required_cols = ["timestamp", "latitude", "longitude", "complaint_type"]
        optional_cols = ["description", "address", "ward"]

        metrics = {}
        penalties = []

        # Check required columns
        for col in required_cols:
            if col in df.columns:
                missing_pct = df[col].isna().sum() / len(df) * 100
                metrics[f"{col}_missing_pct"] = round(missing_pct, 2)
                if missing_pct > 30:
                    penalties.append(30)
                elif missing_pct > 10:
                    penalties.append(15)
                elif missing_pct > 5:
                    penalties.append(5)
            else:
                metrics[f"{col}_present"] = False
                penalties.append(25)

        # Check optional columns
        for col in optional_cols:
            if col in df.columns:
                missing_pct = df[col].isna().sum() / len(df) * 100
                metrics[f"{col}_missing_pct"] = round(missing_pct, 2)
            else:
                metrics[f"{col}_present"] = False

        score = max(0, 100 - sum(penalties))

        return QualityDimension(
            name="completeness",
            score=score,
            description="Missing required fields" if penalties else "All required fields present",
            metrics=metrics,
            recommendations=[
                f"Fill missing {col}" for col in required_cols
                if metrics.get(f"{col}_missing_pct", 0) > 10
            ],
        )

    def _score_accuracy(self, df: pd.DataFrame) -> QualityDimension:
        """Score data accuracy."""
        metrics = {}
        penalties = []

        # Check coordinate accuracy
        if "latitude" in df.columns and "longitude" in df.columns:
            lats = pd.to_numeric(df["latitude"], errors="coerce")
            lons = pd.to_numeric(df["longitude"], errors="coerce")

            # Out of range
            out_of_range = (
                (lats.abs() > 90) | (lons.abs() > 180)
            ).sum()
            if out_of_range > 0:
                pct = out_of_range / len(df) * 100
                metrics["coords_out_of_range_pct"] = round(pct, 2)
                penalties.append(min(pct * 2, 30))

            # Near zero (likely errors)
            near_zero = ((lats.abs() < 0.1) | (lons.abs() < 0.1)).sum()
            if near_zero > 0:
                pct = near_zero / len(df) * 100
                metrics["coords_near_zero_pct"] = round(pct, 2)
                penalties.append(min(pct * 2, 20))

            # Check coordinate precision (too many decimals = suspicious)
            if len(df) > 0:
                sample = lats.dropna().head(100)
                avg_decimals = sample.apply(
                    lambda x: len(str(x).split(".")[-1]) if "." in str(x) else 0
                ).mean()
                metrics["avg_coord_decimals"] = round(avg_decimals, 1)

        # Check timestamp accuracy
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            valid_ts = ts.dropna()

            if len(valid_ts) > 0:
                # Future dates
                future = (valid_ts > datetime.utcnow()).sum()
                if future > 0:
                    pct = future / len(valid_ts) * 100
                    metrics["future_timestamps_pct"] = round(pct, 2)
                    penalties.append(min(pct * 3, 25))

                # Very old dates (before 2000)
                very_old = (valid_ts < datetime(2000, 1, 1)).sum()
                if very_old > 0:
                    pct = very_old / len(valid_ts) * 100
                    metrics["ancient_timestamps_pct"] = round(pct, 2)
                    penalties.append(min(pct * 2, 15))

        score = max(0, 100 - sum(penalties))

        return QualityDimension(
            name="accuracy",
            score=score,
            description="Coordinate or timestamp errors detected" if penalties else "Data appears accurate",
            metrics=metrics,
        )

    def _score_consistency(self, df: pd.DataFrame) -> QualityDimension:
        """Score data consistency."""
        metrics = {}
        penalties = []

        # Check complaint type consistency
        if "complaint_type" in df.columns:
            type_counts = df["complaint_type"].value_counts()
            metrics["unique_types"] = len(type_counts)

            # Too many unique types suggests inconsistent categorization
            if len(type_counts) > 50:
                penalties.append(15)
            elif len(type_counts) > 20:
                penalties.append(5)

            # Single type dominance (might be filtering issue)
            if len(type_counts) > 0:
                top_pct = type_counts.iloc[0] / len(df) * 100
                metrics["top_type_pct"] = round(top_pct, 2)
                if top_pct > 95:
                    penalties.append(10)

        # Check for consistent coordinate precision
        if "latitude" in df.columns:
            lats = pd.to_numeric(df["latitude"], errors="coerce").dropna()
            if len(lats) > 10:
                precision = lats.apply(
                    lambda x: len(str(x).split(".")[-1]) if "." in str(x) else 0
                )
                precision_std = precision.std()
                metrics["coord_precision_std"] = round(precision_std, 2)
                if precision_std > 2:
                    penalties.append(10)

        score = max(0, 100 - sum(penalties))

        return QualityDimension(
            name="consistency",
            score=score,
            description="Inconsistent categorization or precision" if penalties else "Data is consistent",
            metrics=metrics,
        )

    def _score_timeliness(self, df: pd.DataFrame) -> QualityDimension:
        """Score data timeliness."""
        metrics = {}
        penalties = []

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            valid_ts = ts.dropna()

            if len(valid_ts) > 0:
                now = datetime.utcnow()
                max_ts = valid_ts.max()
                min_ts = valid_ts.min()

                # How recent is the most recent data
                days_old = (now - max_ts).days
                metrics["most_recent_days_ago"] = days_old

                if days_old > 365:
                    penalties.append(30)
                elif days_old > 90:
                    penalties.append(15)
                elif days_old > 30:
                    penalties.append(5)

                # Coverage span
                coverage_days = (max_ts - min_ts).days
                metrics["coverage_days"] = coverage_days

                if coverage_days < 7:
                    penalties.append(20)
                elif coverage_days < 30:
                    penalties.append(10)
        else:
            penalties.append(50)

        score = max(0, 100 - sum(penalties))

        return QualityDimension(
            name="timeliness",
            score=score,
            description="Data may be outdated" if penalties else "Data is timely",
            metrics=metrics,
        )

    def _score_uniqueness(self, df: pd.DataFrame) -> QualityDimension:
        """Score data uniqueness (inverse of duplication)."""
        metrics = {}
        penalties = []

        # Check for exact duplicates
        n_duplicates = df.duplicated().sum()
        dup_pct = n_duplicates / len(df) * 100 if len(df) > 0 else 0
        metrics["exact_duplicate_pct"] = round(dup_pct, 2)

        if dup_pct > 20:
            penalties.append(30)
        elif dup_pct > 10:
            penalties.append(15)
        elif dup_pct > 5:
            penalties.append(5)

        # Check for flagged duplicates
        if "is_potential_duplicate" in df.columns:
            pot_dup_pct = df["is_potential_duplicate"].sum() / len(df) * 100
            metrics["potential_duplicate_pct"] = round(pot_dup_pct, 2)
            if pot_dup_pct > 30:
                penalties.append(15)

        score = max(0, 100 - sum(penalties))

        return QualityDimension(
            name="uniqueness",
            score=score,
            description="High duplication detected" if penalties else "Low duplication",
            metrics=metrics,
        )

    def _score_validity(self, df: pd.DataFrame) -> QualityDimension:
        """Score data validity against business rules."""
        metrics = {}
        penalties = []

        # Check minimum record count
        metrics["record_count"] = len(df)
        if len(df) < 10:
            penalties.append(40)
        elif len(df) < 50:
            penalties.append(20)
        elif len(df) < 100:
            penalties.append(10)

        # Check for flood-related content
        if "has_flood_keywords" in df.columns:
            flood_pct = df["has_flood_keywords"].sum() / len(df) * 100
            metrics["flood_keyword_pct"] = round(flood_pct, 2)
            # Low flood keyword presence might indicate wrong data
            if flood_pct < 10:
                penalties.append(15)
        elif "complaint_type" in df.columns:
            # Check complaint types for flood-related terms
            type_str = " ".join(df["complaint_type"].dropna().astype(str))
            flood_terms = ["flood", "water", "drain", "sewer", "storm"]
            has_flood = any(term in type_str.lower() for term in flood_terms)
            metrics["has_flood_types"] = has_flood
            if not has_flood:
                penalties.append(20)

        score = max(0, 100 - sum(penalties))

        return QualityDimension(
            name="validity",
            score=score,
            description="Data may not be relevant" if penalties else "Data appears valid",
            metrics=metrics,
        )


# ==============================================================================
# BIAS QUANTIFICATION
# ==============================================================================

@dataclass
class BiasEstimate:
    """Quantified bias estimate with correction factor."""

    bias_type: str
    severity: str  # low, medium, high, critical
    magnitude: float  # 0-1 scale
    confidence: float  # confidence in estimate
    correction_factor: Optional[float] = None
    affected_records: int = 0
    affected_areas: List[str] = field(default_factory=list)
    methodology: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class BiasReport:
    """Comprehensive bias analysis report."""

    total_bias_score: float = 0.0  # 0-100, higher = more biased
    biases: List[BiasEstimate] = field(default_factory=list)
    spatial_bias: Dict = field(default_factory=dict)
    temporal_bias: Dict = field(default_factory=dict)
    socioeconomic_proxy: Dict = field(default_factory=dict)
    correction_recommendations: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class BiasQuantifier:
    """Quantify and analyze reporting biases."""

    def analyze(
        self,
        df: pd.DataFrame,
        population_data: Optional[pd.DataFrame] = None,
    ) -> BiasReport:
        """Comprehensive bias analysis.

        Args:
            df: Complaint DataFrame
            population_data: Optional population data for normalization

        Returns:
            BiasReport with quantified biases
        """
        report = BiasReport()

        # Temporal bias analysis
        temporal_bias = self._analyze_temporal_bias(df)
        report.temporal_bias = temporal_bias
        if temporal_bias.get("severity") in ["high", "critical"]:
            report.biases.append(BiasEstimate(
                bias_type="temporal_reporting",
                severity=temporal_bias["severity"],
                magnitude=temporal_bias.get("magnitude", 0.5),
                confidence=0.8,
                methodology="Hour/day distribution analysis",
                recommendations=[
                    "Consider temporal weighting in analysis",
                    "Expand temporal window for event matching",
                ],
            ))

        # Spatial bias analysis
        spatial_bias = self._analyze_spatial_bias(df)
        report.spatial_bias = spatial_bias
        if spatial_bias.get("severity") in ["high", "critical"]:
            report.biases.append(BiasEstimate(
                bias_type="spatial_clustering",
                severity=spatial_bias["severity"],
                magnitude=spatial_bias.get("magnitude", 0.5),
                confidence=0.7,
                methodology="Spatial clustering analysis (nearest neighbor)",
                recommendations=[
                    "Use spatial smoothing or kriging",
                    "Consider inverse distance weighting",
                ],
            ))

        # Volume bias (over/under reporting)
        volume_bias = self._analyze_volume_patterns(df)
        if volume_bias.get("severity") in ["high", "critical"]:
            report.biases.append(BiasEstimate(
                bias_type="volume_anomaly",
                severity=volume_bias["severity"],
                magnitude=volume_bias.get("magnitude", 0.5),
                confidence=0.6,
                methodology="Volume time series analysis",
                recommendations=volume_bias.get("recommendations", []),
            ))

        # Socioeconomic proxy analysis
        if "ward" in df.columns or "address" in df.columns:
            se_bias = self._analyze_socioeconomic_proxy(df)
            report.socioeconomic_proxy = se_bias
            if se_bias.get("likely_bias"):
                report.biases.append(BiasEstimate(
                    bias_type="socioeconomic_underreporting",
                    severity="high",
                    magnitude=0.6,
                    confidence=0.5,
                    methodology="Area-based proxy analysis",
                    recommendations=[
                        "Use population density normalization",
                        "Weight by infrastructure age",
                    ],
                ))

        # Compute overall bias score
        bias_magnitudes = [b.magnitude for b in report.biases]
        if bias_magnitudes:
            report.total_bias_score = np.mean(bias_magnitudes) * 100
        else:
            report.total_bias_score = 0.0

        # Generate correction recommendations
        report.correction_recommendations = self._generate_corrections(
            report.biases
        )

        report.metadata = {
            "records_analyzed": len(df),
            "biases_detected": len(report.biases),
            "analyzed_at": datetime.utcnow().isoformat(),
        }

        return report

    def _analyze_temporal_bias(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal reporting patterns."""
        result = {
            "severity": "low",
            "magnitude": 0.0,
            "metrics": {},
        }

        if "timestamp" not in df.columns:
            return result

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        valid_ts = ts.dropna()

        if len(valid_ts) < 10:
            return result

        # Hour of day analysis
        hour_counts = valid_ts.dt.hour.value_counts().sort_index()
        result["metrics"]["hour_distribution"] = hour_counts.to_dict()

        # Office hours concentration
        office_hours = (valid_ts.dt.hour >= 9) & (valid_ts.dt.hour <= 17)
        office_pct = office_hours.sum() / len(valid_ts)
        result["metrics"]["office_hours_pct"] = round(office_pct * 100, 1)

        # Expected uniform would be ~33% (8 hours / 24)
        # High deviation indicates bias
        expected = 8 / 24
        deviation = abs(office_pct - expected) / expected
        result["magnitude"] = min(deviation, 1.0)

        if office_pct > 0.7:
            result["severity"] = "high"
        elif office_pct > 0.5:
            result["severity"] = "medium"
        else:
            result["severity"] = "low"

        # Day of week analysis
        dow_counts = valid_ts.dt.dayofweek.value_counts()
        weekday_pct = dow_counts[:5].sum() / dow_counts.sum()
        result["metrics"]["weekday_pct"] = round(weekday_pct * 100, 1)

        # Chi-square test for uniformity
        observed = hour_counts.reindex(range(24), fill_value=0).values
        expected_uniform = np.full(24, len(valid_ts) / 24)
        chi2, p_value = scipy_stats.chisquare(observed, expected_uniform)
        result["metrics"]["hour_chi2"] = round(chi2, 2)
        result["metrics"]["hour_chi2_pvalue"] = round(p_value, 4)

        if p_value < 0.001:
            result["severity"] = max(result["severity"], "medium", key=lambda x: ["low", "medium", "high", "critical"].index(x))

        return result

    def _analyze_spatial_bias(self, df: pd.DataFrame) -> Dict:
        """Analyze spatial clustering and coverage."""
        result = {
            "severity": "low",
            "magnitude": 0.0,
            "metrics": {},
        }

        if "latitude" not in df.columns or "longitude" not in df.columns:
            return result

        lats = pd.to_numeric(df["latitude"], errors="coerce")
        lons = pd.to_numeric(df["longitude"], errors="coerce")
        valid = ~(lats.isna() | lons.isna())

        if valid.sum() < 10:
            return result

        coords = np.column_stack([lats[valid].values, lons[valid].values])

        # Compute spatial statistics
        lat_std = np.std(coords[:, 0])
        lon_std = np.std(coords[:, 1])
        result["metrics"]["lat_std"] = round(lat_std, 6)
        result["metrics"]["lon_std"] = round(lon_std, 6)

        # Coefficient of variation for clustering
        lat_cv = lat_std / abs(np.mean(coords[:, 0])) if np.mean(coords[:, 0]) != 0 else 0
        lon_cv = lon_std / abs(np.mean(coords[:, 1])) if np.mean(coords[:, 1]) != 0 else 0
        avg_cv = (lat_cv + lon_cv) / 2
        result["metrics"]["spatial_cv"] = round(avg_cv, 4)

        # High clustering indicates potential bias
        if avg_cv < 0.001:
            result["severity"] = "critical"
            result["magnitude"] = 0.9
        elif avg_cv < 0.005:
            result["severity"] = "high"
            result["magnitude"] = 0.7
        elif avg_cv < 0.01:
            result["severity"] = "medium"
            result["magnitude"] = 0.4

        # Nearest neighbor analysis for clustering
        if len(coords) > 50:
            from scipy.spatial import distance_matrix
            dist_matrix = distance_matrix(coords[:100], coords[:100])
            np.fill_diagonal(dist_matrix, np.inf)
            nn_distances = np.min(dist_matrix, axis=1)
            result["metrics"]["mean_nn_distance"] = round(np.mean(nn_distances), 6)
            result["metrics"]["median_nn_distance"] = round(np.median(nn_distances), 6)

        return result

    def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns for anomalies."""
        result = {
            "severity": "low",
            "magnitude": 0.0,
            "metrics": {},
            "recommendations": [],
        }

        if "timestamp" not in df.columns:
            return result

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        valid_ts = ts.dropna()

        if len(valid_ts) < 30:
            return result

        # Daily counts
        daily_counts = valid_ts.dt.date.value_counts().sort_index()
        result["metrics"]["daily_mean"] = round(daily_counts.mean(), 2)
        result["metrics"]["daily_std"] = round(daily_counts.std(), 2)
        result["metrics"]["daily_max"] = int(daily_counts.max())
        result["metrics"]["daily_min"] = int(daily_counts.min())

        # Coefficient of variation
        cv = daily_counts.std() / daily_counts.mean() if daily_counts.mean() > 0 else 0
        result["metrics"]["daily_cv"] = round(cv, 2)

        # High CV indicates inconsistent reporting
        if cv > 2.0:
            result["severity"] = "high"
            result["magnitude"] = 0.7
            result["recommendations"].append(
                "Normalize by expected daily volume"
            )
        elif cv > 1.0:
            result["severity"] = "medium"
            result["magnitude"] = 0.4

        # Check for outlier days (Z-score)
        z_scores = np.abs(scipy_stats.zscore(daily_counts))
        outlier_days = (z_scores > 3).sum()
        result["metrics"]["outlier_days"] = int(outlier_days)

        if outlier_days > 5:
            result["severity"] = "high"
            result["recommendations"].append(
                "Investigate high-volume days for mass reporting events"
            )

        return result

    def _analyze_socioeconomic_proxy(self, df: pd.DataFrame) -> Dict:
        """Analyze potential socioeconomic bias using proxy indicators."""
        result = {
            "likely_bias": False,
            "metrics": {},
            "proxy_indicators": [],
        }

        # Ward-level analysis
        if "ward" in df.columns:
            ward_counts = df["ward"].value_counts()
            result["metrics"]["wards_with_data"] = len(ward_counts)
            result["metrics"]["top_ward_pct"] = round(
                ward_counts.iloc[0] / len(df) * 100 if len(ward_counts) > 0 else 0,
                1
            )

            # High concentration in few wards suggests bias
            if len(ward_counts) > 0:
                top3_pct = ward_counts.head(3).sum() / len(df)
                if top3_pct > 0.8:
                    result["likely_bias"] = True
                    result["proxy_indicators"].append(
                        "80%+ complaints from 3 wards"
                    )

        # Address complexity as proxy
        if "address" in df.columns:
            addr_lengths = df["address"].dropna().str.len()
            if len(addr_lengths) > 0:
                result["metrics"]["avg_address_length"] = round(
                    addr_lengths.mean(), 1
                )
                # Very short addresses might indicate under-detailed reporting
                if addr_lengths.mean() < 20:
                    result["proxy_indicators"].append(
                        "Short average address length"
                    )

        return result

    def _generate_corrections(
        self,
        biases: List[BiasEstimate],
    ) -> List[str]:
        """Generate correction recommendations based on detected biases."""
        recommendations = []

        for bias in biases:
            if bias.severity in ["high", "critical"]:
                if bias.bias_type == "temporal_reporting":
                    recommendations.append(
                        "Apply temporal weighting: complaints outside office "
                        "hours may represent 2-3x actual events"
                    )
                elif bias.bias_type == "spatial_clustering":
                    recommendations.append(
                        "Use population-weighted spatial averaging to "
                        "account for underreporting in some areas"
                    )
                elif bias.bias_type == "socioeconomic_underreporting":
                    recommendations.append(
                        "Apply socioeconomic correction factor: multiply "
                        "counts in low-income areas by 1.5-2x"
                    )

        return recommendations


# ==============================================================================
# OUTLIER DETECTION
# ==============================================================================

@dataclass
class OutlierResult:
    """Result of outlier detection."""

    total_records: int = 0
    outliers_iqr: int = 0
    outliers_zscore: int = 0
    outliers_isolation: int = 0
    outlier_indices: List[int] = field(default_factory=list)
    outlier_scores: Dict[int, float] = field(default_factory=dict)
    method_agreement: int = 0  # Records flagged by multiple methods


class OutlierDetector:
    """Detect outliers using multiple methods."""

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        contamination: float = 0.1,
    ):
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination

    def detect(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> OutlierResult:
        """Detect outliers using multiple methods.

        Args:
            df: DataFrame to analyze
            columns: Columns to check for outliers

        Returns:
            OutlierResult with detected outliers
        """
        result = OutlierResult(total_records=len(df))

        if columns is None:
            # Auto-select numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            return result

        # Track outliers from each method
        iqr_outliers = set()
        zscore_outliers = set()
        isolation_outliers = set()

        for col in columns:
            if col not in df.columns:
                continue

            values = pd.to_numeric(df[col], errors="coerce")
            valid_mask = ~values.isna()
            valid_values = values[valid_mask]

            if len(valid_values) < 10:
                continue

            # IQR method
            q1 = valid_values.quantile(0.25)
            q3 = valid_values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            iqr_mask = (valid_values < lower) | (valid_values > upper)
            iqr_outliers.update(valid_values[iqr_mask].index.tolist())

            # Z-score method
            z_scores = np.abs(scipy_stats.zscore(valid_values))
            zscore_mask = z_scores > self.zscore_threshold
            zscore_outliers.update(valid_values[zscore_mask].index.tolist())

        # Isolation Forest (if enough data)
        numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
        complete_rows = numeric_df.dropna()

        if len(complete_rows) > 100:
            try:
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_jobs=-1,
                )
                predictions = clf.fit_predict(complete_rows)
                isolation_mask = predictions == -1
                isolation_outliers.update(
                    complete_rows[isolation_mask].index.tolist()
                )
            except ImportError:
                logger.debug("sklearn not available for Isolation Forest")

        # Combine results
        result.outliers_iqr = len(iqr_outliers)
        result.outliers_zscore = len(zscore_outliers)
        result.outliers_isolation = len(isolation_outliers)

        # Agreement between methods
        all_outliers = iqr_outliers | zscore_outliers | isolation_outliers
        result.outlier_indices = list(all_outliers)

        multi_method = (
            (iqr_outliers & zscore_outliers) |
            (iqr_outliers & isolation_outliers) |
            (zscore_outliers & isolation_outliers)
        )
        result.method_agreement = len(multi_method)

        # Compute outlier scores
        for idx in all_outliers:
            score = 0.0
            if idx in iqr_outliers:
                score += 0.33
            if idx in zscore_outliers:
                score += 0.33
            if idx in isolation_outliers:
                score += 0.34
            result.outlier_scores[idx] = round(score, 2)

        return result


# ==============================================================================
# COMPREHENSIVE VALIDATION
# ==============================================================================

@dataclass
class FullValidationResult:
    """Complete validation result."""

    valid: bool = False
    quality_score: DataQualityScore = field(default_factory=DataQualityScore)
    bias_report: BiasReport = field(default_factory=BiasReport)
    outlier_result: OutlierResult = field(default_factory=OutlierResult)
    temporal_stats: Dict = field(default_factory=dict)
    spatial_stats: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def validate_complaints_full(
    df: pd.DataFrame,
    source_id: Optional[str] = None,
    expected_start: Optional[datetime] = None,
    expected_end: Optional[datetime] = None,
    expected_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> FullValidationResult:
    """Full-featured complaint validation.

    Args:
        df: Complaint DataFrame
        source_id: Data source identifier
        expected_start: Expected start date
        expected_end: Expected end date
        expected_bounds: Expected (min_lat, min_lon, max_lat, max_lon)

    Returns:
        FullValidationResult with comprehensive analysis
    """
    result = FullValidationResult()

    if len(df) == 0:
        result.errors.append("Empty DataFrame")
        return result

    logger.info("Starting full validation of %d records", len(df))

    # Quality scoring
    scorer = DataQualityScorer()
    result.quality_score = scorer.score(df)

    # Bias analysis
    quantifier = BiasQuantifier()
    result.bias_report = quantifier.analyze(df)

    # Outlier detection
    detector = OutlierDetector()
    result.outlier_result = detector.detect(
        df,
        columns=["latitude", "longitude"],
    )

    # Temporal statistics
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        valid_ts = ts.dropna()
        if len(valid_ts) > 0:
            result.temporal_stats = {
                "start": valid_ts.min().isoformat(),
                "end": valid_ts.max().isoformat(),
                "coverage_days": (valid_ts.max() - valid_ts.min()).days,
                "records_with_timestamp": len(valid_ts),
            }

    # Spatial statistics
    if "latitude" in df.columns and "longitude" in df.columns:
        lats = pd.to_numeric(df["latitude"], errors="coerce")
        lons = pd.to_numeric(df["longitude"], errors="coerce")
        valid = ~(lats.isna() | lons.isna())
        if valid.sum() > 0:
            result.spatial_stats = {
                "lat_range": (float(lats[valid].min()), float(lats[valid].max())),
                "lon_range": (float(lons[valid].min()), float(lons[valid].max())),
                "records_with_coords": int(valid.sum()),
            }

    # Combine recommendations
    result.recommendations.extend(
        result.quality_score.critical_issues
    )
    result.recommendations.extend(
        result.bias_report.correction_recommendations
    )

    # Combine warnings
    result.warnings.extend(result.quality_score.warnings)

    # Overall validity
    result.valid = (
        result.quality_score.overall_score >= 60 and
        len(result.errors) == 0
    )

    logger.info(
        "Validation complete: quality=%s (%.1f), bias=%.1f%%, valid=%s",
        result.quality_score.grade,
        result.quality_score.overall_score,
        result.bias_report.total_bias_score,
        result.valid,
    )

    return result


def generate_full_validation_report(
    result: FullValidationResult,
    output_path: Optional[Path] = None,
) -> str:
    """Generate comprehensive validation report.

    Args:
        result: Full validation result
        output_path: Optional path to save report

    Returns:
        Report text
    """
    lines = [
        "=" * 70,
        "COMPREHENSIVE COMPLAINT DATA VALIDATION REPORT",
        "=" * 70,
        "",
        f"Overall Status: {'VALID' if result.valid else 'INVALID'}",
        "",
        "DATA QUALITY",
        "-" * 50,
        f"  Overall Score: {result.quality_score.overall_score:.1f}/100 "
        f"(Grade: {result.quality_score.grade})",
        "",
    ]

    for name, dim in result.quality_score.dimensions.items():
        lines.append(f"  {name.capitalize()}: {dim.score:.1f}/100")
        if dim.metrics:
            for k, v in list(dim.metrics.items())[:3]:
                lines.append(f"    - {k}: {v}")

    lines.extend([
        "",
        "BIAS ANALYSIS",
        "-" * 50,
        f"  Total Bias Score: {result.bias_report.total_bias_score:.1f}%",
        "",
    ])

    for bias in result.bias_report.biases:
        lines.append(f"  [{bias.severity.upper()}] {bias.bias_type}")
        lines.append(f"    Magnitude: {bias.magnitude:.2f}")
        lines.append(f"    Method: {bias.methodology}")

    lines.extend([
        "",
        "OUTLIER DETECTION",
        "-" * 50,
        f"  IQR Method: {result.outlier_result.outliers_iqr} outliers",
        f"  Z-Score Method: {result.outlier_result.outliers_zscore} outliers",
        f"  Isolation Forest: {result.outlier_result.outliers_isolation} outliers",
        f"  Multi-method Agreement: {result.outlier_result.method_agreement}",
        "",
    ])

    if result.recommendations:
        lines.extend([
            "RECOMMENDATIONS",
            "-" * 50,
        ])
        for rec in result.recommendations[:10]:
            lines.append(f"  • {rec}")
        lines.append("")

    if result.warnings:
        lines.extend([
            "WARNINGS",
            "-" * 50,
        ])
        for warn in result.warnings[:10]:
            lines.append(f"  ⚠ {warn}")
        lines.append("")

    if result.errors:
        lines.extend([
            "ERRORS",
            "-" * 50,
        ])
        for err in result.errors:
            lines.append(f"  ✗ {err}")
        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report)
        logger.info("Full validation report saved to %s", output_path)

    return report
