"""Dataset Integrity Checks (Data Automation Prompt-4).

Implements comprehensive integrity checks for scientific validity:
- Required vs optional dataset verification
- Temporal overlap validation
- Spatial coverage validation
- Data sufficiency thresholds

Each check returns PASS / WARN / FAIL - NOT binary OK/NOT OK.
The system NEVER silently downgrades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.data_registry.registry import (
    DataRegistry,
    DatasetMetadata,
    DatasetType,
    ValidationStatus,
    SpatialCoverage,
    TemporalCoverage,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# CHECK STATUS
# ==============================================================================

class CheckStatus(str, Enum):
    """Result status for integrity checks."""
    PASS = "pass"           # Fully valid
    WARN = "warn"           # Degraded but usable
    FAIL = "fail"           # Scientifically invalid


# ==============================================================================
# CHECK RESULT
# ==============================================================================

@dataclass
class CheckResult:
    """Result of a single integrity check."""

    check_name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    impact_on_inference: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def is_pass(self) -> bool:
        return self.status == CheckStatus.PASS

    def is_warn(self) -> bool:
        return self.status == CheckStatus.WARN

    def is_fail(self) -> bool:
        return self.status == CheckStatus.FAIL

    def to_dict(self) -> Dict:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "recommendations": self.recommendations,
            "impact_on_inference": self.impact_on_inference,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IntegrityReport:
    """Complete integrity check report."""

    checks: List[CheckResult] = field(default_factory=list)
    overall_status: CheckStatus = CheckStatus.FAIL
    summary: str = ""
    can_proceed: bool = False
    degraded_components: List[str] = field(default_factory=list)
    missing_components: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def add_check(self, result: CheckResult) -> None:
        """Add a check result and update overall status."""
        self.checks.append(result)
        self._update_overall_status()

    def _update_overall_status(self) -> None:
        """Recalculate overall status from checks."""
        if not self.checks:
            self.overall_status = CheckStatus.FAIL
            self.can_proceed = False
            return

        has_fail = any(c.status == CheckStatus.FAIL for c in self.checks)
        has_warn = any(c.status == CheckStatus.WARN for c in self.checks)

        if has_fail:
            self.overall_status = CheckStatus.FAIL
            self.can_proceed = False
        elif has_warn:
            self.overall_status = CheckStatus.WARN
            self.can_proceed = True
        else:
            self.overall_status = CheckStatus.PASS
            self.can_proceed = True

    def get_failures(self) -> List[CheckResult]:
        """Get all failed checks."""
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    def get_warnings(self) -> List[CheckResult]:
        """Get all warning checks."""
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    def to_dict(self) -> Dict:
        return {
            "overall_status": self.overall_status.value,
            "can_proceed": self.can_proceed,
            "summary": self.summary,
            "degraded_components": self.degraded_components,
            "missing_components": self.missing_components,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp.isoformat(),
        }


# ==============================================================================
# INTEGRITY CHECKER
# ==============================================================================

class IntegrityChecker:
    """Performs comprehensive integrity checks on registered datasets.

    Checks are NOT binary - they return:
    - PASS: Fully valid for scientific inference
    - WARN: Degraded but usable (with documented limitations)
    - FAIL: Scientifically invalid - cannot proceed

    The system NEVER silently downgrades.
    """

    # Thresholds
    MIN_TEMPORAL_COVERAGE = 0.7   # 70%
    WARN_TEMPORAL_COVERAGE = 0.9  # 90%
    MIN_SPATIAL_OVERLAP = 0.5     # 50%
    WARN_SPATIAL_OVERLAP = 0.8    # 80%
    MIN_QUALITY_SCORE = 50.0
    WARN_QUALITY_SCORE = 70.0
    MIN_RECORDS = 10
    WARN_RECORDS = 100

    def __init__(self, registry: DataRegistry):
        """Initialize checker with registry.

        Args:
            registry: Data registry to check
        """
        self.registry = registry

    def run_all_checks(
        self,
        city: str,
        target_start: datetime,
        target_end: datetime,
        target_bounds: Optional[Tuple[float, float, float, float]] = None,
        run_mode: str = "full_run",
    ) -> IntegrityReport:
        """Run all integrity checks.

        Args:
            city: Target city
            target_start: Start of analysis period
            target_end: End of analysis period
            target_bounds: (min_lat, min_lon, max_lat, max_lon)
            run_mode: 'full_run', 'degraded_run', or 'minimal_run'

        Returns:
            IntegrityReport with all check results
        """
        report = IntegrityReport()

        logger.info(
            "Running integrity checks for %s (%s to %s)",
            city,
            target_start.strftime("%Y-%m-%d"),
            target_end.strftime("%Y-%m-%d"),
        )

        # Check required datasets
        report.add_check(self.check_required_datasets(run_mode))

        # Check each required dataset type
        required_types = self.registry.config.required_datasets.get(run_mode, [])

        for dtype in required_types:
            # Temporal coverage
            report.add_check(
                self.check_temporal_coverage(
                    dtype, target_start, target_end
                )
            )

            # Spatial coverage
            if target_bounds:
                report.add_check(
                    self.check_spatial_coverage(dtype, target_bounds)
                )

            # Quality
            report.add_check(self.check_quality_threshold(dtype))

            # Sufficiency
            report.add_check(self.check_data_sufficiency(dtype))

        # Cross-dataset consistency
        report.add_check(
            self.check_cross_dataset_consistency(
                required_types, target_start, target_end
            )
        )

        # Validation freshness
        report.add_check(self.check_validation_freshness())

        # Generate summary
        report.summary = self._generate_summary(report, city)

        logger.info(
            "Integrity check complete: %s (%d pass, %d warn, %d fail)",
            report.overall_status.value,
            len([c for c in report.checks if c.is_pass()]),
            len(report.get_warnings()),
            len(report.get_failures()),
        )

        return report

    # ==========================================================================
    # INDIVIDUAL CHECKS
    # ==========================================================================

    def check_required_datasets(
        self,
        run_mode: str = "full_run",
    ) -> CheckResult:
        """Check if all required dataset types are available.

        Args:
            run_mode: Run mode to check requirements for

        Returns:
            CheckResult
        """
        missing = self.registry.get_missing_required(run_mode)

        if not missing:
            return CheckResult(
                check_name="required_datasets",
                status=CheckStatus.PASS,
                message=f"All required datasets for {run_mode} are available",
                details={"run_mode": run_mode},
            )

        # Check if we can fall back to degraded mode
        missing_in_degraded = self.registry.get_missing_required("degraded_run")

        if not missing_in_degraded and run_mode == "full_run":
            return CheckResult(
                check_name="required_datasets",
                status=CheckStatus.WARN,
                message=(
                    f"Missing {len(missing)} dataset(s) for full run, "
                    "but degraded run is possible"
                ),
                details={
                    "missing": [m.value for m in missing],
                    "run_mode": run_mode,
                    "fallback_available": True,
                },
                recommendations=[
                    f"Acquire {m.value} data for full analysis" for m in missing
                ],
                impact_on_inference=(
                    "Analysis will proceed with reduced features"
                ),
            )

        return CheckResult(
            check_name="required_datasets",
            status=CheckStatus.FAIL,
            message=f"Missing {len(missing)} required dataset(s): "
                    f"{', '.join(m.value for m in missing)}",
            details={
                "missing": [m.value for m in missing],
                "run_mode": run_mode,
            },
            recommendations=[
                f"Acquire {m.value} data before proceeding" for m in missing
            ],
            impact_on_inference="Cannot proceed without required datasets",
        )

    def check_temporal_coverage(
        self,
        dataset_type: DatasetType,
        target_start: datetime,
        target_end: datetime,
    ) -> CheckResult:
        """Check temporal coverage for target period.

        Args:
            dataset_type: Dataset type to check
            target_start: Start of target period
            target_end: End of target period

        Returns:
            CheckResult
        """
        datasets = self.registry.get_by_type(dataset_type)
        available = [d for d in datasets if d.is_available]

        if not available:
            return CheckResult(
                check_name=f"temporal_coverage_{dataset_type.value}",
                status=CheckStatus.FAIL,
                message=f"No {dataset_type.value} data available",
                details={"dataset_type": dataset_type.value},
                impact_on_inference="Cannot analyze without this data",
            )

        # Find best coverage
        best_coverage = 0.0
        best_dataset = None

        for ds in available:
            if ds.temporal_coverage.is_static:
                # Static data (DEM) always covers
                return CheckResult(
                    check_name=f"temporal_coverage_{dataset_type.value}",
                    status=CheckStatus.PASS,
                    message=f"{dataset_type.value} is static data (always valid)",
                    details={"dataset_id": ds.dataset_id, "is_static": True},
                )

            coverage = ds.temporal_coverage.overlap_fraction(
                target_start, target_end
            )
            if coverage > best_coverage:
                best_coverage = coverage
                best_dataset = ds

        if best_coverage >= self.WARN_TEMPORAL_COVERAGE:
            return CheckResult(
                check_name=f"temporal_coverage_{dataset_type.value}",
                status=CheckStatus.PASS,
                message=(
                    f"{dataset_type.value} covers {best_coverage*100:.1f}% "
                    "of target period"
                ),
                details={
                    "coverage_fraction": best_coverage,
                    "dataset_id": best_dataset.dataset_id if best_dataset else None,
                },
            )

        if best_coverage >= self.MIN_TEMPORAL_COVERAGE:
            return CheckResult(
                check_name=f"temporal_coverage_{dataset_type.value}",
                status=CheckStatus.WARN,
                message=(
                    f"{dataset_type.value} covers only {best_coverage*100:.1f}% "
                    "of target period"
                ),
                details={
                    "coverage_fraction": best_coverage,
                    "threshold": self.WARN_TEMPORAL_COVERAGE,
                    "dataset_id": best_dataset.dataset_id if best_dataset else None,
                },
                recommendations=[
                    f"Acquire additional {dataset_type.value} data for "
                    f"{target_start:%Y-%m-%d} to {target_end:%Y-%m-%d}"
                ],
                impact_on_inference=(
                    f"Analysis limited to {best_coverage*100:.1f}% of period"
                ),
            )

        return CheckResult(
            check_name=f"temporal_coverage_{dataset_type.value}",
            status=CheckStatus.FAIL,
            message=(
                f"{dataset_type.value} covers only {best_coverage*100:.1f}% "
                f"(minimum {self.MIN_TEMPORAL_COVERAGE*100:.0f}% required)"
            ),
            details={
                "coverage_fraction": best_coverage,
                "threshold": self.MIN_TEMPORAL_COVERAGE,
            },
            impact_on_inference="Insufficient data for valid inference",
        )

    def check_spatial_coverage(
        self,
        dataset_type: DatasetType,
        target_bounds: Tuple[float, float, float, float],
    ) -> CheckResult:
        """Check spatial coverage for target area.

        Args:
            dataset_type: Dataset type to check
            target_bounds: (min_lat, min_lon, max_lat, max_lon)

        Returns:
            CheckResult
        """
        min_lat, min_lon, max_lat, max_lon = target_bounds
        target = SpatialCoverage(
            min_lat=min_lat, max_lat=max_lat,
            min_lon=min_lon, max_lon=max_lon,
        )

        datasets = self.registry.get_by_type(dataset_type)
        available = [d for d in datasets if d.is_available]

        if not available:
            return CheckResult(
                check_name=f"spatial_coverage_{dataset_type.value}",
                status=CheckStatus.FAIL,
                message=f"No {dataset_type.value} data available",
                details={"dataset_type": dataset_type.value},
            )

        # Check for overlap
        overlapping = []
        for ds in available:
            if ds.spatial_coverage.overlaps(target):
                overlapping.append(ds)

        if not overlapping:
            return CheckResult(
                check_name=f"spatial_coverage_{dataset_type.value}",
                status=CheckStatus.FAIL,
                message=(
                    f"No {dataset_type.value} data overlaps target area "
                    f"({min_lat:.4f}, {min_lon:.4f}) to "
                    f"({max_lat:.4f}, {max_lon:.4f})"
                ),
                details={
                    "target_bounds": target_bounds,
                    "available_count": len(available),
                },
                recommendations=[
                    f"Acquire {dataset_type.value} data covering target area"
                ],
                impact_on_inference="Cannot analyze area without spatial coverage",
            )

        # Calculate overlap quality (simplified - full overlap)
        best_ds = overlapping[0]
        ds_bounds = best_ds.spatial_coverage

        # Approximate overlap fraction
        overlap_lat = min(ds_bounds.max_lat, max_lat) - max(ds_bounds.min_lat, min_lat)
        overlap_lon = min(ds_bounds.max_lon, max_lon) - max(ds_bounds.min_lon, min_lon)
        target_lat = max_lat - min_lat
        target_lon = max_lon - min_lon

        if target_lat > 0 and target_lon > 0:
            overlap_frac = (
                max(0, overlap_lat) * max(0, overlap_lon)
            ) / (target_lat * target_lon)
        else:
            overlap_frac = 1.0 if overlapping else 0.0

        if overlap_frac >= self.WARN_SPATIAL_OVERLAP:
            return CheckResult(
                check_name=f"spatial_coverage_{dataset_type.value}",
                status=CheckStatus.PASS,
                message=(
                    f"{dataset_type.value} covers ~{overlap_frac*100:.0f}% "
                    "of target area"
                ),
                details={
                    "overlap_fraction": overlap_frac,
                    "dataset_id": best_ds.dataset_id,
                },
            )

        if overlap_frac >= self.MIN_SPATIAL_OVERLAP:
            return CheckResult(
                check_name=f"spatial_coverage_{dataset_type.value}",
                status=CheckStatus.WARN,
                message=(
                    f"{dataset_type.value} covers only ~{overlap_frac*100:.0f}% "
                    "of target area"
                ),
                details={
                    "overlap_fraction": overlap_frac,
                    "threshold": self.WARN_SPATIAL_OVERLAP,
                },
                recommendations=[
                    "Inference will be limited to overlapping area",
                    f"Consider acquiring more {dataset_type.value} data",
                ],
                impact_on_inference=(
                    f"Analysis limited to {overlap_frac*100:.0f}% of area"
                ),
            )

        return CheckResult(
            check_name=f"spatial_coverage_{dataset_type.value}",
            status=CheckStatus.FAIL,
            message=(
                f"{dataset_type.value} covers only ~{overlap_frac*100:.0f}% "
                f"(minimum {self.MIN_SPATIAL_OVERLAP*100:.0f}% required)"
            ),
            details={
                "overlap_fraction": overlap_frac,
                "threshold": self.MIN_SPATIAL_OVERLAP,
            },
            impact_on_inference="Insufficient spatial coverage",
        )

    def check_quality_threshold(
        self,
        dataset_type: DatasetType,
    ) -> CheckResult:
        """Check data quality meets threshold.

        Args:
            dataset_type: Dataset type to check

        Returns:
            CheckResult
        """
        best_ds = self.registry.get_best_for_type(dataset_type)

        if not best_ds:
            return CheckResult(
                check_name=f"quality_{dataset_type.value}",
                status=CheckStatus.FAIL,
                message=f"No {dataset_type.value} data available",
                details={"dataset_type": dataset_type.value},
            )

        score = best_ds.quality_metrics.overall_score

        if score >= self.WARN_QUALITY_SCORE:
            return CheckResult(
                check_name=f"quality_{dataset_type.value}",
                status=CheckStatus.PASS,
                message=(
                    f"{dataset_type.value} quality score: {score:.1f}/100 "
                    f"(grade: {best_ds.quality_metrics.grade})"
                ),
                details={
                    "quality_score": score,
                    "grade": best_ds.quality_metrics.grade,
                    "dataset_id": best_ds.dataset_id,
                },
            )

        if score >= self.MIN_QUALITY_SCORE:
            warnings = best_ds.quality_metrics.warnings
            limitations = best_ds.quality_metrics.known_limitations

            return CheckResult(
                check_name=f"quality_{dataset_type.value}",
                status=CheckStatus.WARN,
                message=(
                    f"{dataset_type.value} quality score: {score:.1f}/100 "
                    "(below recommended threshold)"
                ),
                details={
                    "quality_score": score,
                    "grade": best_ds.quality_metrics.grade,
                    "warnings": warnings,
                    "limitations": limitations,
                },
                recommendations=[
                    "Consider acquiring higher quality data",
                    "Increase uncertainty bounds in inference",
                ],
                impact_on_inference=(
                    "Results may have increased uncertainty due to data quality"
                ),
            )

        return CheckResult(
            check_name=f"quality_{dataset_type.value}",
            status=CheckStatus.FAIL,
            message=(
                f"{dataset_type.value} quality score: {score:.1f}/100 "
                f"(minimum {self.MIN_QUALITY_SCORE:.0f} required)"
            ),
            details={
                "quality_score": score,
                "threshold": self.MIN_QUALITY_SCORE,
                "validation_errors": best_ds.validation_errors,
            },
            impact_on_inference="Data quality insufficient for valid inference",
        )

    def check_data_sufficiency(
        self,
        dataset_type: DatasetType,
    ) -> CheckResult:
        """Check if dataset has sufficient records.

        Args:
            dataset_type: Dataset type to check

        Returns:
            CheckResult
        """
        best_ds = self.registry.get_best_for_type(dataset_type)

        if not best_ds:
            return CheckResult(
                check_name=f"sufficiency_{dataset_type.value}",
                status=CheckStatus.FAIL,
                message=f"No {dataset_type.value} data available",
            )

        # For raster data (DEM), check resolution instead of records
        if dataset_type == DatasetType.DEM:
            resolution = best_ds.spatial_coverage.resolution_m
            if resolution and resolution <= 30:
                return CheckResult(
                    check_name=f"sufficiency_{dataset_type.value}",
                    status=CheckStatus.PASS,
                    message=f"DEM resolution: {resolution}m (sufficient)",
                    details={"resolution_m": resolution},
                )
            elif resolution and resolution <= 90:
                return CheckResult(
                    check_name=f"sufficiency_{dataset_type.value}",
                    status=CheckStatus.WARN,
                    message=f"DEM resolution: {resolution}m (coarse)",
                    details={"resolution_m": resolution},
                    impact_on_inference="Coarse resolution may miss small features",
                )
            else:
                return CheckResult(
                    check_name=f"sufficiency_{dataset_type.value}",
                    status=CheckStatus.WARN,
                    message=f"DEM resolution unknown or very coarse",
                    details={"resolution_m": resolution},
                )

        # For tabular data, check record count
        completeness = best_ds.quality_metrics.completeness
        # Estimate record count from completeness if available
        record_estimate = completeness * 100 if completeness > 0 else self.MIN_RECORDS

        if record_estimate >= self.WARN_RECORDS:
            return CheckResult(
                check_name=f"sufficiency_{dataset_type.value}",
                status=CheckStatus.PASS,
                message=f"{dataset_type.value} has sufficient data volume",
                details={"completeness": completeness},
            )

        if record_estimate >= self.MIN_RECORDS:
            return CheckResult(
                check_name=f"sufficiency_{dataset_type.value}",
                status=CheckStatus.WARN,
                message=f"{dataset_type.value} has limited data volume",
                details={"completeness": completeness},
                impact_on_inference="Limited data may affect statistical power",
            )

        return CheckResult(
            check_name=f"sufficiency_{dataset_type.value}",
            status=CheckStatus.FAIL,
            message=f"{dataset_type.value} has insufficient data volume",
            details={"completeness": completeness},
        )

    def check_cross_dataset_consistency(
        self,
        dataset_types: List[DatasetType],
        target_start: datetime,
        target_end: datetime,
    ) -> CheckResult:
        """Check consistency across datasets.

        Verifies:
        - CRS compatibility
        - Temporal overlap
        - Spatial overlap

        Args:
            dataset_types: Types to check consistency for
            target_start: Start of target period
            target_end: End of target period

        Returns:
            CheckResult
        """
        datasets = []
        for dtype in dataset_types:
            best = self.registry.get_best_for_type(dtype)
            if best:
                datasets.append(best)

        if len(datasets) < 2:
            return CheckResult(
                check_name="cross_dataset_consistency",
                status=CheckStatus.PASS,
                message="Insufficient datasets for consistency check",
                details={"dataset_count": len(datasets)},
            )

        issues = []
        warnings = []

        # Check CRS compatibility
        crs_set = {ds.spatial_coverage.crs for ds in datasets}
        if len(crs_set) > 1:
            warnings.append(
                f"Multiple CRS found: {crs_set}. Reprojection will be needed."
            )

        # Check temporal overlap
        temporal_datasets = [
            ds for ds in datasets
            if not ds.temporal_coverage.is_static
        ]

        if len(temporal_datasets) >= 2:
            starts = [ds.temporal_coverage.start for ds in temporal_datasets
                      if ds.temporal_coverage.start]
            ends = [ds.temporal_coverage.end for ds in temporal_datasets
                    if ds.temporal_coverage.end]

            if starts and ends:
                common_start = max(starts)
                common_end = min(ends)

                if common_start >= common_end:
                    issues.append(
                        "No temporal overlap between datasets"
                    )
                else:
                    overlap_days = (common_end - common_start).days
                    if overlap_days < 7:
                        warnings.append(
                            f"Limited temporal overlap: {overlap_days} days"
                        )

        # Check spatial overlap
        if len(datasets) >= 2:
            base_coverage = datasets[0].spatial_coverage
            for ds in datasets[1:]:
                if not base_coverage.overlaps(ds.spatial_coverage):
                    issues.append(
                        f"No spatial overlap between {datasets[0].dataset_type.value} "
                        f"and {ds.dataset_type.value}"
                    )

        if issues:
            return CheckResult(
                check_name="cross_dataset_consistency",
                status=CheckStatus.FAIL,
                message=f"Consistency issues: {'; '.join(issues)}",
                details={
                    "issues": issues,
                    "warnings": warnings,
                },
                impact_on_inference="Datasets cannot be combined for analysis",
            )

        if warnings:
            return CheckResult(
                check_name="cross_dataset_consistency",
                status=CheckStatus.WARN,
                message=f"Minor consistency concerns: {'; '.join(warnings)}",
                details={"warnings": warnings},
                recommendations=[
                    "Verify CRS transformations are correct",
                    "Check temporal alignment carefully",
                ],
            )

        return CheckResult(
            check_name="cross_dataset_consistency",
            status=CheckStatus.PASS,
            message="All datasets are consistent and compatible",
            details={"dataset_count": len(datasets)},
        )

    def check_validation_freshness(self) -> CheckResult:
        """Check if validations are fresh or stale.

        Returns:
            CheckResult
        """
        ttl_hours = self.registry.config.validation_ttl_hours
        cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)

        stale_datasets = []
        for ds in self.registry.get_available():
            if ds.validation_timestamp and ds.validation_timestamp < cutoff:
                stale_datasets.append(ds.dataset_id)

        if not stale_datasets:
            return CheckResult(
                check_name="validation_freshness",
                status=CheckStatus.PASS,
                message="All validations are fresh",
                details={"ttl_hours": ttl_hours},
            )

        if len(stale_datasets) <= 2:
            return CheckResult(
                check_name="validation_freshness",
                status=CheckStatus.WARN,
                message=f"{len(stale_datasets)} dataset(s) have stale validations",
                details={
                    "stale_datasets": stale_datasets,
                    "ttl_hours": ttl_hours,
                },
                recommendations=["Re-run validation on stale datasets"],
            )

        return CheckResult(
            check_name="validation_freshness",
            status=CheckStatus.WARN,
            message=f"Many datasets ({len(stale_datasets)}) have stale validations",
            details={
                "stale_count": len(stale_datasets),
                "ttl_hours": ttl_hours,
            },
            recommendations=[
                "Consider re-validating all datasets",
                "Check if source data has been updated",
            ],
        )

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _generate_summary(
        self,
        report: IntegrityReport,
        city: str,
    ) -> str:
        """Generate human-readable summary.

        Args:
            report: Integrity report
            city: Target city

        Returns:
            Summary string
        """
        lines = []

        if report.overall_status == CheckStatus.PASS:
            lines.append(f"✓ All integrity checks passed for {city}")
        elif report.overall_status == CheckStatus.WARN:
            lines.append(f"⚠ Integrity checks passed with warnings for {city}")
            for w in report.get_warnings():
                lines.append(f"  - {w.message}")
        else:
            lines.append(f"✗ Integrity checks FAILED for {city}")
            for f in report.get_failures():
                lines.append(f"  - {f.message}")

        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def run_integrity_checks(
    registry: DataRegistry,
    city: str,
    target_start: datetime,
    target_end: datetime,
    target_bounds: Optional[Tuple[float, float, float, float]] = None,
    run_mode: str = "full_run",
) -> IntegrityReport:
    """Run all integrity checks.

    Args:
        registry: Data registry
        city: Target city
        target_start: Start of analysis period
        target_end: End of analysis period
        target_bounds: (min_lat, min_lon, max_lat, max_lon)
        run_mode: 'full_run', 'degraded_run', or 'minimal_run'

    Returns:
        IntegrityReport
    """
    checker = IntegrityChecker(registry)
    return checker.run_all_checks(
        city=city,
        target_start=target_start,
        target_end=target_end,
        target_bounds=target_bounds,
        run_mode=run_mode,
    )


def generate_integrity_report(report: IntegrityReport) -> str:
    """Generate human-readable integrity report.

    Args:
        report: Integrity report

    Returns:
        Report text
    """
    lines = [
        "=" * 60,
        "DATA INTEGRITY REPORT",
        "=" * 60,
        "",
        f"Overall Status: {report.overall_status.value.upper()}",
        f"Can Proceed: {'Yes' if report.can_proceed else 'No'}",
        "",
    ]

    # Summary
    if report.summary:
        lines.extend([report.summary, ""])

    # Checks by status
    passed = [c for c in report.checks if c.is_pass()]
    warned = report.get_warnings()
    failed = report.get_failures()

    if passed:
        lines.append("PASSED CHECKS")
        lines.append("-" * 40)
        for c in passed:
            lines.append(f"  ✓ {c.check_name}: {c.message}")
        lines.append("")

    if warned:
        lines.append("WARNINGS")
        lines.append("-" * 40)
        for c in warned:
            lines.append(f"  ⚠ {c.check_name}: {c.message}")
            if c.impact_on_inference:
                lines.append(f"    Impact: {c.impact_on_inference}")
        lines.append("")

    if failed:
        lines.append("FAILURES")
        lines.append("-" * 40)
        for c in failed:
            lines.append(f"  ✗ {c.check_name}: {c.message}")
            if c.impact_on_inference:
                lines.append(f"    Impact: {c.impact_on_inference}")
            for rec in c.recommendations:
                lines.append(f"    → {rec}")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"Generated: {report.timestamp:%Y-%m-%d %H:%M:%S} UTC")
    lines.append("=" * 60)

    return "\n".join(lines)
