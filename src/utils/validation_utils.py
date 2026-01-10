"""Validation helpers to prevent silent failures."""
from typing import Iterable
import pandas as pd
import geopandas as gpd


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str = "") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {context or 'dataframe'}")


def assert_non_empty(df: pd.DataFrame, context: str = "") -> None:
    if df is None or df.empty:
        raise ValueError(f"{context or 'dataframe'} is empty; cannot proceed")


def assert_crs(gdf: gpd.GeoDataFrame, expected: str) -> None:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame missing CRS")
    if gdf.crs.to_string() != expected:
        raise ValueError(f"CRS mismatch: {gdf.crs} != {expected}")
