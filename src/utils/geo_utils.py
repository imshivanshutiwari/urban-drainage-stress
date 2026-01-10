"""Geospatial helper functions with explicit validation."""
from typing import Iterable, Tuple
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS


def ensure_crs(gdf: gpd.GeoDataFrame, target: str) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame if needed."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame missing CRS; cannot reproject.")
    if gdf.crs.to_string() == target:
        return gdf
    return gdf.to_crs(target)


def points_from_records(records: Iterable[Tuple[float, float]], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Create point GeoDataFrame from (lon, lat) iterable."""
    geometries = [Point(lon, lat) for lon, lat in records if _valid_lon_lat(lon, lat)]
    if not geometries:
        raise ValueError("No valid points found; check input coordinates.")
    return gpd.GeoDataFrame(geometry=geometries, crs=CRS.from_string(crs))


def _valid_lon_lat(lon: float, lat: float) -> bool:
    return -180 <= lon <= 180 and -90 <= lat <= 90
