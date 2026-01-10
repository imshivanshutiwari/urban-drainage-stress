"""Run-Readiness Gate (Data Automation Prompt-4).

The pipeline's decision engine that:
- Queries the registry
- Evaluates integrity checks  
- Decides: FULL_RUN / DEGRADED_RUN / ABORT

NO major pipeline stage runs before the gate passes.
The system NEVER silently downgrades.
All decisions are logged and explainable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.data_registry.registry import (
    DataRegistry,
    DatasetMetadata,
    DatasetType,
    ValidationStatus,
)
from src.data_registry.integrity_checks import (
    IntegrityChecker,
    IntegrityReport,
    CheckStatus,
    CheckResult,
    run_integrity_checks,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# RUN DECISION
# ==============================================================================

class RunDecision(str, Enum):
    """Pipeline run decision."""
    FULL_RUN = "full_run"         # All data available, proceed fully
    DEGRADED_RUN = "degraded_run" # Some data missing, proceed with limitations
    ABORT = "abort"               # Cannot proceed, critical data missing


# ==============================================================================
# GATE RESULT
# ==============================================================================

@dataclass
class GateResult:
    """Result of run-readiness gate evaluation."""

    # Decision
    decision: RunDecision = RunDecision.ABORT
    can_proceed: bool = False

    # Explanation
    explanation: str = ""
    detailed_reasons: List[str] = field(default_factory=list)

    # Component status
    datasets_used: List[str] = field(default_factory=list)
    datasets_missing: List[str] = field(default_factory=list)
    degraded_components: List[str] = field(default_factory=list)

    # Impact
    inference_impacts: List[str] = field(default_factory=list)
    uncertainty_adjustments: Dict[str, float] = field(default_factory=dict)

    # Underlying reports
    integrity_report: Optional[IntegrityReport] = None

    # Metadata
    city: str = ""
    target_start: Optional[datetime] = None
    target_end: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    gate_version: str = "1.0"

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "decision": self.decision.value,
            "can_proceed": self.can_proceed,
            "explanation": self.explanation,
            "detailed_reasons": self.detailed_reasons,
            "datasets_used": self.datasets_used,
            "datasets_missing": self.datasets_missing,
            "degraded_components": self.degraded_components,
            "inference_impacts": self.inference_impacts,
            "uncertainty_adjustments": self.uncertainty_adjustments,
            "city": self.city,
            "target_start": self.target_start.isoformat() if self.target_start else None,
            "target_end": self.target_end.isoformat() if self.target_end else None,
            "timestamp": self.timestamp.isoformat(),
            "gate_version": self.gate_version,
            "integrity_summary": (
                self.integrity_report.to_dict() if self.integrity_report else None
            ),
        }

    def save(self, path: Path) -> None:
        """Save gate result to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Gate result saved to %s", path)


# ==============================================================================
# RUN-READINESS GATE
# ==============================================================================

class RunGate:
    """Pipeline run-readiness gate.

    Evaluates whether the system can execute end-to-end:
    - FULL_RUN: All required data available with good quality
    - DEGRADED_RUN: Proceed with documented limitations
    - ABORT: Cannot proceed, critical data missing

    The gate NEVER silently downgrades.
    All decisions are logged with clear explanations.
    """

    # Uncertainty multipliers for degraded scenarios
    UNCERTAINTY_MULTIPLIERS = {
        "missing_radar": 1.3,
        "missing_complaints": 1.2,
        "low_quality_rainfall": 1.25,
        "low_quality_dem": 1.4,
        "limited_temporal_coverage": 1.15,
        "limited_spatial_coverage": 1.2,
    }

    def __init__(
        self,
        registry: DataRegistry,
        reports_dir: Optional[Path] = None,
    ):
        """Initialize the gate.

        Args:
            registry: Data registry to evaluate
            reports_dir: Directory for run reports
        """
        self.registry = registry
        self.reports_dir = reports_dir or Path("data/run_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._last_result: Optional[GateResult] = None

    def evaluate(
        self,
        city: str,
        target_start: datetime,
        target_end: datetime,
        target_bounds: Optional[Tuple[float, float, float, float]] = None,
        strict_mode: bool = False,
    ) -> GateResult:
        """Evaluate run-readiness.

        Args:
            city: Target city
            target_start: Start of analysis period
            target_end: End of analysis period
            target_bounds: (min_lat, min_lon, max_lat, max_lon)
            strict_mode: If True, WARN -> ABORT

        Returns:
            GateResult with decision and explanation
        """
        header = (
            "=" * 60 + "\n"
            f"RUN-READINESS GATE EVALUATION\n"
            f"City: {city} | Period: {target_start.strftime('%Y-%m-%d')} to {target_end.strftime('%Y-%m-%d')}\n"
            + "=" * 60
        )
        logger.info(header)

        result = GateResult(
            city=city,
            target_start=target_start,
            target_end=target_end,
        )

        # Run integrity checks
        integrity_report = run_integrity_checks(
            registry=self.registry,
            city=city,
            target_start=target_start,
            target_end=target_end,
            target_bounds=target_bounds,
            run_mode="full_run",
        )
        result.integrity_report = integrity_report

        # Analyze registry state
        self._analyze_datasets(result)

        # Make decision
        self._make_decision(result, integrity_report, strict_mode)

        # Calculate uncertainty adjustments
        self._calculate_uncertainty_adjustments(result)

        # Generate explanation
        self._generate_explanation(result)

        # Log decision
        self._log_decision(result)

        # Save report
        self._save_report(result)

        self._last_result = result
        return result

    def _analyze_datasets(self, result: GateResult) -> None:
        """Analyze dataset availability.

        Args:
            result: Gate result to populate
        """
        # Collect available datasets
        for ds in self.registry.get_available():
            result.datasets_used.append(
                f"{ds.dataset_type.value}: {ds.name} ({ds.source_name})"
            )

            # Check for degraded datasets
            if ds.validation_status == ValidationStatus.DEGRADED:
                result.degraded_components.append(
                    f"{ds.dataset_type.value}: {ds.degraded_reason}"
                )
            elif ds.validation_status == ValidationStatus.VALID_WITH_WARNINGS:
                for warning in ds.quality_metrics.warnings:
                    result.degraded_components.append(
                        f"{ds.dataset_type.value}: {warning}"
                    )

        # Collect missing datasets
        missing_types = self.registry.get_missing_required("full_run")
        for dtype in missing_types:
            result.datasets_missing.append(dtype.value)

    def _make_decision(
        self,
        result: GateResult,
        integrity_report: IntegrityReport,
        strict_mode: bool,
    ) -> None:
        """Make run decision based on integrity report.

        Args:
            result: Gate result to populate
            integrity_report: Integrity check results
            strict_mode: If True, WARN -> ABORT
        """
        failures = integrity_report.get_failures()
        warnings = integrity_report.get_warnings()

        # ABORT if any critical failures
        if failures:
            result.decision = RunDecision.ABORT
            result.can_proceed = False
            result.detailed_reasons = [
                f"FAIL: {f.message}" for f in failures
            ]

            # Check if we can recover with degraded mode
            degraded_report = run_integrity_checks(
                registry=self.registry,
                city=result.city,
                target_start=result.target_start,
                target_end=result.target_end,
                run_mode="degraded_run",
            )

            if degraded_report.can_proceed:
                result.decision = RunDecision.DEGRADED_RUN
                result.can_proceed = True
                result.detailed_reasons.append(
                    "RECOVERY: Falling back to degraded run mode"
                )
                logger.warning(
                    "Full run not possible, falling back to DEGRADED_RUN"
                )
            return

        # Strict mode: WARN -> ABORT
        if strict_mode and warnings:
            result.decision = RunDecision.ABORT
            result.can_proceed = False
            result.detailed_reasons = [
                f"STRICT_MODE: {w.message}" for w in warnings
            ]
            return

        # DEGRADED_RUN if warnings
        if warnings:
            result.decision = RunDecision.DEGRADED_RUN
            result.can_proceed = True
            result.detailed_reasons = [
                f"WARN: {w.message}" for w in warnings
            ]

            # Collect inference impacts
            for w in warnings:
                if w.impact_on_inference:
                    result.inference_impacts.append(w.impact_on_inference)

            return

        # FULL_RUN if all clear
        result.decision = RunDecision.FULL_RUN
        result.can_proceed = True
        result.detailed_reasons = ["All integrity checks passed"]

    def _calculate_uncertainty_adjustments(
        self,
        result: GateResult,
    ) -> None:
        """Calculate uncertainty multipliers for degraded scenarios.

        Args:
            result: Gate result to populate
        """
        if result.decision == RunDecision.FULL_RUN:
            result.uncertainty_adjustments = {"base": 1.0}
            return

        adjustments = {"base": 1.0}

        # Missing datasets
        for missing in result.datasets_missing:
            key = f"missing_{missing}"
            if key in self.UNCERTAINTY_MULTIPLIERS:
                adjustments[key] = self.UNCERTAINTY_MULTIPLIERS[key]
            else:
                adjustments[f"missing_{missing}"] = 1.1  # Default

        # Degraded components
        for degraded in result.degraded_components:
            if "quality" in degraded.lower():
                dtype = degraded.split(":")[0]
                key = f"low_quality_{dtype}"
                adjustments[key] = self.UNCERTAINTY_MULTIPLIERS.get(key, 1.15)
            elif "temporal" in degraded.lower():
                adjustments["limited_temporal_coverage"] = (
                    self.UNCERTAINTY_MULTIPLIERS["limited_temporal_coverage"]
                )
            elif "spatial" in degraded.lower():
                adjustments["limited_spatial_coverage"] = (
                    self.UNCERTAINTY_MULTIPLIERS["limited_spatial_coverage"]
                )

        # Calculate combined multiplier
        combined = 1.0
        for key, mult in adjustments.items():
            if key != "base":
                combined *= mult

        adjustments["combined"] = round(combined, 2)
        result.uncertainty_adjustments = adjustments

    def _generate_explanation(self, result: GateResult) -> None:
        """Generate human-readable explanation.

        Args:
            result: Gate result to populate
        """
        lines = []

        if result.decision == RunDecision.FULL_RUN:
            lines.append(
                f"[OK] FULL RUN APPROVED for {result.city}"
            )
            lines.append(
                f"All {len(result.datasets_used)} required datasets available "
                "with acceptable quality."
            )

        elif result.decision == RunDecision.DEGRADED_RUN:
            lines.append(
                f"[WARN] DEGRADED RUN APPROVED for {result.city}"
            )
            lines.append("Proceeding with documented limitations:")
            for reason in result.detailed_reasons:
                lines.append(f"  - {reason}")

            if result.inference_impacts:
                lines.append("\nImpact on inference:")
                for impact in result.inference_impacts:
                    lines.append(f"  -> {impact}")

            if result.uncertainty_adjustments.get("combined", 1.0) > 1.0:
                lines.append(
                    f"\nUncertainty multiplier: "
                    f"{result.uncertainty_adjustments['combined']:.2f}x"
                )

        else:  # ABORT
            lines.append(
                f"[FAIL] RUN ABORTED for {result.city}"
            )
            lines.append("Cannot proceed due to critical issues:")
            for reason in result.detailed_reasons:
                lines.append(f"  - {reason}")

            if result.datasets_missing:
                lines.append("\nMissing datasets:")
                for missing in result.datasets_missing:
                    lines.append(f"  - {missing}")

        result.explanation = "\n".join(lines)

    def _log_decision(self, result: GateResult) -> None:
        """Log the gate decision.

        Args:
            result: Gate result
        """
        log_func = {
            RunDecision.FULL_RUN: logger.info,
            RunDecision.DEGRADED_RUN: logger.warning,
            RunDecision.ABORT: logger.error,
        }.get(result.decision, logger.info)

        msg = (
            "\n" + "=" * 60 + "\n"
            f"GATE DECISION: {result.decision.value.upper()}\n"
            f"{result.explanation}\n"
            + "=" * 60
        )
        log_func(msg)

    def _save_report(self, result: GateResult) -> None:
        """Save gate result as run report.

        Args:
            result: Gate result
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"run_{timestamp}.json"
        result.save(report_path)

    # ==========================================================================
    # QUERY METHODS
    # ==========================================================================

    def get_last_result(self) -> Optional[GateResult]:
        """Get the last gate evaluation result."""
        return self._last_result

    def is_ready(self) -> bool:
        """Check if system is ready to run (based on last evaluation)."""
        if not self._last_result:
            return False
        return self._last_result.can_proceed

    def get_decision(self) -> Optional[RunDecision]:
        """Get the last decision (based on last evaluation)."""
        if not self._last_result:
            return None
        return self._last_result.decision

    def get_degraded_flags(self) -> Dict[str, bool]:
        """Get flags indicating which components are degraded.

        Returns:
            Dict mapping component name to degraded status
        """
        if not self._last_result:
            return {}

        flags = {}
        for ds_type in DatasetType:
            flags[f"{ds_type.value}_available"] = ds_type.value not in (
                self._last_result.datasets_missing
            )
            flags[f"{ds_type.value}_degraded"] = any(
                ds_type.value in d for d in self._last_result.degraded_components
            )

        flags["is_degraded_run"] = (
            self._last_result.decision == RunDecision.DEGRADED_RUN
        )

        return flags


# ==============================================================================
# GATE DECORATOR
# ==============================================================================

def require_gate_pass(func):
    """Decorator to require gate pass before function execution.

    Usage:
        @require_gate_pass
        def run_analysis(gate: RunGate, ...):
            ...
    """
    def wrapper(*args, **kwargs):
        # Find gate in args or kwargs
        gate = kwargs.get("gate")
        if gate is None:
            for arg in args:
                if isinstance(arg, RunGate):
                    gate = arg
                    break

        if gate is None:
            raise ValueError(
                "require_gate_pass decorator requires RunGate instance"
            )

        if not gate.is_ready():
            decision = gate.get_decision()
            raise RuntimeError(
                f"Gate not passed. Decision: {decision}. "
                "Run gate.evaluate() first."
            )

        return func(*args, **kwargs)

    return wrapper


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def evaluate_run_readiness(
    registry: DataRegistry,
    city: str,
    target_start: datetime,
    target_end: datetime,
    target_bounds: Optional[Tuple[float, float, float, float]] = None,
    strict_mode: bool = False,
) -> GateResult:
    """Evaluate run readiness.

    Args:
        registry: Data registry
        city: Target city
        target_start: Start of analysis period
        target_end: End of analysis period
        target_bounds: (min_lat, min_lon, max_lat, max_lon)
        strict_mode: If True, warnings cause abort

    Returns:
        GateResult
    """
    gate = RunGate(registry)
    return gate.evaluate(
        city=city,
        target_start=target_start,
        target_end=target_end,
        target_bounds=target_bounds,
        strict_mode=strict_mode,
    )


def generate_run_report(result: GateResult) -> str:
    """Generate human-readable run report.

    Args:
        result: Gate result

    Returns:
        Report text
    """
    lines = [
        "=" * 70,
        "RUN-READINESS REPORT",
        "=" * 70,
        "",
        f"City: {result.city}",
        f"Period: {result.target_start:%Y-%m-%d} to {result.target_end:%Y-%m-%d}",
        f"Timestamp: {result.timestamp:%Y-%m-%d %H:%M:%S} UTC",
        "",
        "-" * 70,
        f"DECISION: {result.decision.value.upper()}",
        f"Can Proceed: {'YES' if result.can_proceed else 'NO'}",
        "-" * 70,
        "",
    ]

    # Explanation
    lines.extend(["EXPLANATION", "-" * 40])
    lines.append(result.explanation)
    lines.append("")

    # Datasets used
    if result.datasets_used:
        lines.extend(["DATASETS USED", "-" * 40])
        for ds in result.datasets_used:
            lines.append(f"  ✓ {ds}")
        lines.append("")

    # Datasets missing
    if result.datasets_missing:
        lines.extend(["DATASETS MISSING", "-" * 40])
        for ds in result.datasets_missing:
            lines.append(f"  ✗ {ds}")
        lines.append("")

    # Degraded components
    if result.degraded_components:
        lines.extend(["DEGRADED COMPONENTS", "-" * 40])
        for comp in result.degraded_components:
            lines.append(f"  ⚠ {comp}")
        lines.append("")

    # Inference impacts
    if result.inference_impacts:
        lines.extend(["INFERENCE IMPACTS", "-" * 40])
        for impact in result.inference_impacts:
            lines.append(f"  → {impact}")
        lines.append("")

    # Uncertainty adjustments
    if result.uncertainty_adjustments:
        lines.extend(["UNCERTAINTY ADJUSTMENTS", "-" * 40])
        for key, value in result.uncertainty_adjustments.items():
            if key != "base":
                lines.append(f"  {key}: {value:.2f}x")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def create_run_summary(
    result: GateResult,
    output_path: Optional[Path] = None,
) -> Dict:
    """Create run summary artifact.

    This is the MANDATORY transparency output that includes:
    - datasets used
    - datasets missing
    - degraded components
    - expected impact on inference reliability

    Args:
        result: Gate result
        output_path: Path to save summary (default: data/run_reports/)

    Returns:
        Summary dictionary
    """
    summary = {
        "meta": {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "gate_version": result.gate_version,
        },
        "context": {
            "city": result.city,
            "target_start": result.target_start.isoformat() if result.target_start else None,
            "target_end": result.target_end.isoformat() if result.target_end else None,
        },
        "decision": {
            "result": result.decision.value,
            "can_proceed": result.can_proceed,
            "explanation": result.explanation,
        },
        "datasets": {
            "used": result.datasets_used,
            "missing": result.datasets_missing,
            "count_used": len(result.datasets_used),
            "count_missing": len(result.datasets_missing),
        },
        "degradation": {
            "components": result.degraded_components,
            "is_degraded": result.decision == RunDecision.DEGRADED_RUN,
        },
        "inference_reliability": {
            "impacts": result.inference_impacts,
            "uncertainty_adjustments": result.uncertainty_adjustments,
            "combined_uncertainty_multiplier": (
                result.uncertainty_adjustments.get("combined", 1.0)
            ),
        },
        "detailed_reasons": result.detailed_reasons,
    }

    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/run_reports/run_{timestamp}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Run summary saved to %s", output_path)

    return summary
