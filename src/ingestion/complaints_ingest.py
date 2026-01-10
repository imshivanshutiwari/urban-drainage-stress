"""Complaint ingestion with explicit bias/uncertainty handling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config.parameters import get_default_parameters
from ..utils.validation_utils import assert_non_empty

logger = logging.getLogger(__name__)

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "reported_at",
    "complaint_time",
    "time",
    "datetime",
]
TYPE_CANDIDATES = [
    "complaint_type",
    "type",
    "category",
    "issue",
]
LAT_CANDIDATES = ["lat", "latitude"]
LON_CANDIDATES = ["lon", "longitude", "lng"]
WARD_CANDIDATES = ["ward", "ward_name"]
AREA_TEXT_CANDIDATES = ["area", "locality", "description", "address"]


@dataclass
class ComplaintIngestionResult:
    data: pd.DataFrame
    flags: pd.DataFrame
    metadata: Dict[str, object]


class ComplaintIngestor:
    """Ingest municipal complaints as weak, biased observations.

    - Accepts CSV/Excel/JSON files.
    - Standardizes time/type/location fields.
    - Flags missing timestamps, ambiguous locations, and near-duplicates.
    - Produces reporting_confidence instead of dropping records.
    """

    def __init__(self) -> None:
        self.params = get_default_parameters().complaints

    def load_files(
        self, files: Iterable[Path | str]
    ) -> ComplaintIngestionResult:
        frames: List[pd.DataFrame] = []
        sources: List[str] = []
        for f in files:
            path = Path(f)
            df = self._read_file(path)
            frames.append(df)
            sources.append(str(path))
        if not frames:
            raise ValueError("No complaint files provided")
        combined = pd.concat(frames, ignore_index=True)
        result = self._standardize_and_flag(combined)
        result.metadata["source_files"] = sources
        return result

    def _read_file(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Complaint file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xls", ".xlsx"}:
            df = pd.read_excel(path)
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == "[":
                    df = pd.read_json(f)
                else:
                    records = [json.loads(line) for line in f if line.strip()]
                    df = pd.DataFrame(records)
        else:
            raise ValueError(f"Unsupported complaint format: {suffix}")
        assert_non_empty(df, context=f"complaints file {path}")
        return df

    def _pick_column(
        self, df: pd.DataFrame, candidates: Sequence[str]
    ) -> Optional[str]:
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        return None

    def _standardize_and_flag(
        self, df: pd.DataFrame
    ) -> ComplaintIngestionResult:
        df = df.copy()

        ts_col = self._pick_column(df, TIMESTAMP_CANDIDATES)
        type_col = self._pick_column(df, TYPE_CANDIDATES)
        lat_col = self._pick_column(df, LAT_CANDIDATES)
        lon_col = self._pick_column(df, LON_CANDIDATES)
        ward_col = self._pick_column(df, WARD_CANDIDATES)
        area_col = self._pick_column(df, AREA_TEXT_CANDIDATES)

        if ts_col is None:
            logger.warning(
                "No timestamp column detected; all records will be flagged"
            )
        if all(col is None for col in [lat_col, lon_col, ward_col, area_col]):
            raise ValueError(
                "No usable location columns (lat/lon/ward/area) found in "
                "complaints"
            )

        df["complaint_time"] = (
            pd.to_datetime(df[ts_col], errors="coerce") if ts_col else pd.NaT
        )
        df["complaint_type"] = (
            df[type_col].astype(str) if type_col else "unknown"
        )
        df["latitude"] = (
            pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
        )
        df["longitude"] = (
            pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan
        )
        df["ward"] = df[ward_col].astype(str) if ward_col else ""
        df["area_text"] = df[area_col].astype(str) if area_col else ""
        df["raw_location"] = (
            df[lat_col]
            if lat_col
            else df[ward_col]
            if ward_col
            else df[area_col]
            if area_col
            else None
        )

        missing_ts = df["complaint_time"].isna()
        has_point = df[["latitude", "longitude"]].notna().all(axis=1)
        ambiguous_loc = ~has_point & (
            df["ward"].eq("") & df["area_text"].eq("")
        )

        df.sort_values("complaint_time", inplace=True)
        df["quality_flag"] = "ok"
        df.loc[missing_ts, "quality_flag"] = "missing_timestamp"
        df.loc[ambiguous_loc, "quality_flag"] = "ambiguous_location"

        dedup_minutes = self.params.deduplicate_minutes
        geo_eps = self.params.geo_epsilon_meters
        duplicate_mask = pd.Series(False, index=df.index)
        if has_point.any():
            dt_minutes = df["complaint_time"].diff().dt.total_seconds().div(60)
            close_time = dt_minutes.abs() <= dedup_minutes
            close_space = (
                df["latitude"].diff().abs().fillna(np.inf) * 111_000 <= geo_eps
            ) & (
                df["longitude"].diff().abs().fillna(np.inf) * 111_000
                <= geo_eps
            )
            duplicate_mask = close_time & close_space
            df.loc[duplicate_mask, "quality_flag"] = "near_duplicate"

        reporting_confidence = np.full(len(df), 1.0)
        reporting_confidence[missing_ts.values] *= 0.3
        reporting_confidence[ambiguous_loc.values] *= 0.5
        reporting_confidence[duplicate_mask.values] *= 0.2

        df["reporting_confidence"] = reporting_confidence
        df["missing_timestamp"] = missing_ts
        df["ambiguous_location"] = ambiguous_loc
        df["duplicate_flag"] = duplicate_mask

        flags = df[
            [
                "complaint_time",
                "complaint_type",
                "missing_timestamp",
                "ambiguous_location",
                "duplicate_flag",
                "reporting_confidence",
                "quality_flag",
            ]
        ].copy()

        logger.info(
            "Complaints ingestion: %d records | missing_ts=%d | "
            "ambiguous_loc=%d | duplicates=%d",
            len(df),
            int(missing_ts.sum()),
            int(ambiguous_loc.sum()),
            int(duplicate_mask.sum()),
        )

        return ComplaintIngestionResult(
            data=df,
            flags=flags,
            metadata={
                "dedup_minutes": dedup_minutes,
                "geo_epsilon_m": geo_eps,
            },
        )
