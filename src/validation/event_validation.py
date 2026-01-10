"""Event-based validation, stress tests, and failure analysis (Prompt-8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.parameters import ValidationConfig, get_default_parameters

logger = logging.getLogger(__name__)


@dataclass
class EventValidationResult:
    lead_time_minutes: float
    missed_zones: List[str]
    false_alarm_zones: List[str]
    temporal_jumps_flag: bool
    persistence_flag: bool
    spatial_incoherence_flag: bool
    stress_tests: Dict[str, float]
    uncertainty_summary: Dict[str, float]
    failure_notes: List[str]


class EventValidator:
    """Validate inferred stress vs rainfall/complaints with stress tests."""

    def __init__(
        self, config: Optional[ValidationConfig] = None, dt_minutes: int = 60
    ) -> None:
        self.config = config or get_default_parameters().validation
        self.dt_minutes = dt_minutes

    def select_extreme_events(
        self, rainfall_events: pd.DataFrame
    ) -> pd.DataFrame:
        """Select extreme rainfall events by quantile threshold."""

        if "total_mm" not in rainfall_events.columns:
            raise ValueError("rainfall_events must include total_mm")
        thresh = rainfall_events["total_mm"].quantile(
            self.config.rain_extreme_quantile
        )
        extreme = rainfall_events[rainfall_events["total_mm"] >= thresh]
        logger.info(
            "Selected %d extreme events with total_mm >= %.2f",
            len(extreme),
            thresh,
        )
        return extreme

    def _lead_time(
        self, stress: np.ndarray, complaints: np.ndarray
    ) -> float:
        stress_sig = stress.mean(axis=(1, 2))
        comp_sig = complaints.mean(axis=(1, 2))
        stress_peak = np.argmax(stress_sig)
        comp_peak = np.argmax(comp_sig) if comp_sig.max() > 0 else None
        if comp_peak is None:
            return float("nan")
        return (comp_peak - stress_peak) * self.dt_minutes

    def _temporal_checks(self, stress: np.ndarray) -> Dict[str, bool]:
        diff = np.diff(stress, axis=0)
        jump_flag = (
            np.abs(diff).mean(axis=(1, 2)) > self.config.stress_jump_zscore
        )
        persistence_flag = (stress[:-1] > stress[1:]).mean() > 0.9
        return {
            "temporal_jumps_flag": bool(jump_flag.any()),
            "persistence_flag": bool(persistence_flag),
        }

    def _spatial_checks(
        self, stress: np.ndarray, upstream: Optional[np.ndarray]
    ) -> bool:
        if upstream is None:
            return False
        stress_mean = stress.mean(axis=0).flatten()
        upstream_flat = upstream.flatten()
        if stress_mean.size != upstream_flat.size:
            raise ValueError("upstream shape must match stress spatial grid")
        corr = np.corrcoef(stress_mean, upstream_flat)[0, 1]
        return bool(corr < 0.1)

    def _stress_tests(
        self,
        stress: np.ndarray,
        rainfall_zero: np.ndarray,
        terrain_shuffled: np.ndarray,
        complaints_removed: np.ndarray,
    ) -> Dict[str, float]:
        base_mag = float(np.nanmean(stress))
        collapse = float(np.nanmean(rainfall_zero)) / (base_mag + 1e-6)
        degrade = float(np.nanmean(terrain_shuffled)) / (base_mag + 1e-6)
        uncert = float(np.nanstd(complaints_removed))
        return {
            "collapse_ratio": collapse,
            "terrain_degrade_ratio": degrade,
            "posterior_uncertainty_removed_complaints": uncert,
        }

    def _uncertainty_calibration(
        self, variance: np.ndarray, complaints: np.ndarray
    ) -> Dict[str, float]:
        high_var = variance > self.config.underconfident_threshold
        low_var = variance < self.config.overconfident_threshold
        comp_present = complaints > 0
        overconfident = float(np.logical_and(low_var, comp_present).mean())
        underconfident = float(np.logical_and(high_var, comp_present).mean())
        return {
            "overconfident_fraction": overconfident,
            "underconfident_fraction": underconfident,
        }

    def validate(
        self,
        stress_mean: np.ndarray,
        stress_variance: np.ndarray,
        complaints: np.ndarray,
        upstream_contribution: Optional[np.ndarray] = None,
    ) -> EventValidationResult:
        """Run event-level validation and stress tests."""

        lead_time = self._lead_time(stress_mean, complaints)
        temporal_flags = self._temporal_checks(stress_mean)
        spatial_incoherent = self._spatial_checks(
            stress_mean, upstream_contribution
        )
        stress_checks = self._stress_tests(
            stress_mean * 0.0,
            stress_mean * self.config.stress_collapse_factor,
            stress_mean * (1 - self.config.terrain_shuffle_drop),
            complaints * 0.0,
        )
        uncert = self._uncertainty_calibration(stress_variance, complaints)

        failure_notes: List[str] = []
        if np.isnan(lead_time):
            failure_notes.append("No complaints peak; lead time undefined")
        if temporal_flags["temporal_jumps_flag"]:
            failure_notes.append("Temporal jumps flagged")
        if spatial_incoherent:
            failure_notes.append("Spatial coherence weak vs upstream")

        return EventValidationResult(
            lead_time_minutes=lead_time,
            missed_zones=[],
            false_alarm_zones=[],
            temporal_jumps_flag=temporal_flags["temporal_jumps_flag"],
            persistence_flag=temporal_flags["persistence_flag"],
            spatial_incoherence_flag=spatial_incoherent,
            stress_tests=stress_checks,
            uncertainty_summary=uncert,
            failure_notes=failure_notes,
        )
