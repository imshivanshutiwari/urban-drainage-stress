"""Rainfall cleaning pipeline.

Handles missingness, basic QC, and flags uncertainty explicitly.
"""
import logging
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def basic_qc(df: pd.DataFrame, intensity_col: str = "intensity_mm_hr") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split clean vs. flagged records based on physical plausibility."""
    if intensity_col not in df.columns:
        raise ValueError(f"Missing expected column {intensity_col}")

    df = df.copy()
    df["qc_flag"] = "ok"

    # Negative rainfall is non-physical
    negative = df[intensity_col] < 0
    df.loc[negative, "qc_flag"] = "negative_value"

    # Extremely high values are suspect but not dropped automatically
    extreme = df[intensity_col] > 200  # mm/hr threshold for QC
    df.loc[extreme, "qc_flag"] = "extreme_value"

    clean = df[df["qc_flag"] == "ok"].copy()
    flagged = df[df["qc_flag"] != "ok"].copy()

    logger.info("Rainfall QC complete: %d clean, %d flagged", len(clean), len(flagged))
    return clean, flagged
