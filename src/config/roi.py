"""Region of Interest (ROI) configuration and validation.

ROI ENFORCEMENT (NON-NEGOTIABLE):
- Every spatial module MUST receive and enforce ROI
- Pipeline MUST ABORT if ROI is missing
- NO DEFAULT GLOBAL EXTENT
- NO ASSUMED GEOGRAPHY

ROI can be defined by:
- City boundary polygon (GeoJSON/Shapefile)
- Explicit bounding box (lat_min, lat_max, lon_min, lon_max)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import box, Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None
    box = None
    Polygon = None

logger = logging.getLogger(__name__)


class ROIError(Exception):
    """Raised when ROI is invalid or missing."""
    pass


class ROIMissingError(ROIError):
    """Raised when ROI is not provided but required."""
    pass


class ROIValidationError(ROIError):
    """Raised when ROI fails validation."""
    pass


@dataclass
class ROIConfig:
    """Region of Interest configuration.
    
    ROI MUST be defined by either:
    - A boundary file (GeoJSON/Shapefile with city polygon)
    - Explicit bounds (lat_min, lat_max, lon_min, lon_max)
    
    If neither is provided, pipeline MUST ABORT.
    """
    
    # Boundary file path (GeoJSON/Shapefile)
    boundary_file: Optional[Path] = None
    
    # Explicit bounding box
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    
    # Coordinate Reference System
    crs: str = "EPSG:4326"
    
    # City name for labeling
    city_name: str = ""
    
    # Buffer for rainfall stations (in degrees for EPSG:4326)
    buffer_degrees: float = 0.05
    
    # Validation settings
    max_area_sq_km: float = 50000.0  # Max reasonable city area
    min_area_sq_km: float = 1.0      # Min reasonable city area
    
    def __post_init__(self):
        """Validate that ROI is properly defined."""
        self._geometry = None
        self._bounds = None
    
    @property
    def is_defined(self) -> bool:
        """Check if ROI is defined."""
        has_boundary = self.boundary_file is not None
        has_bounds = all([
            self.lat_min is not None,
            self.lat_max is not None,
            self.lon_min is not None,
            self.lon_max is not None,
        ])
        return has_boundary or has_bounds
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get ROI bounds as (lon_min, lat_min, lon_max, lat_max)."""
        if self._bounds is not None:
            return self._bounds
        
        if self.boundary_file is not None and self.boundary_file.exists():
            self._load_boundary()
            return self._bounds
        
        if all([self.lon_min, self.lat_min, self.lon_max, self.lat_max]):
            self._bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
            return self._bounds
        
        raise ROIMissingError("ROI bounds not defined")
    
    @property
    def bounds_with_buffer(self) -> Tuple[float, float, float, float]:
        """Get ROI bounds with buffer for rainfall stations."""
        lon_min, lat_min, lon_max, lat_max = self.bounds
        return (
            lon_min - self.buffer_degrees,
            lat_min - self.buffer_degrees,
            lon_max + self.buffer_degrees,
            lat_max + self.buffer_degrees,
        )
    
    @property
    def geometry(self) -> Optional[Any]:
        """Get ROI geometry (Shapely polygon)."""
        if not HAS_GEOPANDAS:
            return None
        
        if self._geometry is not None:
            return self._geometry
        
        if self.boundary_file is not None and self.boundary_file.exists():
            self._load_boundary()
            return self._geometry
        
        # Create box from bounds
        lon_min, lat_min, lon_max, lat_max = self.bounds
        self._geometry = box(lon_min, lat_min, lon_max, lat_max)
        return self._geometry
    
    def _load_boundary(self) -> None:
        """Load boundary from file."""
        if not HAS_GEOPANDAS:
            raise ROIError("GeoPandas required for boundary files")
        
        if not self.boundary_file.exists():
            raise ROIError(f"Boundary file not found: {self.boundary_file}")
        
        try:
            gdf = gpd.read_file(self.boundary_file)
            
            # Reproject to target CRS if needed
            if gdf.crs and gdf.crs.to_string() != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            # Get union of all geometries
            self._geometry = unary_union(gdf.geometry)
            
            # Get bounds
            minx, miny, maxx, maxy = self._geometry.bounds
            self._bounds = (minx, miny, maxx, maxy)
            
            logger.info(
                "Loaded ROI boundary: %s (bounds: %.4f, %.4f, %.4f, %.4f)",
                self.boundary_file.name,
                minx, miny, maxx, maxy
            )
            
        except Exception as e:
            raise ROIError(f"Failed to load boundary: {e}")
    
    def get_area_sq_km(self) -> float:
        """Calculate approximate area in square kilometers."""
        lon_min, lat_min, lon_max, lat_max = self.bounds
        
        # Approximate conversion for lat/lon to km
        mid_lat = (lat_min + lat_max) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(mid_lat))
        
        width_km = (lon_max - lon_min) * km_per_deg_lon
        height_km = (lat_max - lat_min) * km_per_deg_lat
        
        return width_km * height_km
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "city_name": self.city_name,
            "crs": self.crs,
            "bounds": {
                "lon_min": self.bounds[0],
                "lat_min": self.bounds[1],
                "lon_max": self.bounds[2],
                "lat_max": self.bounds[3],
            },
            "boundary_file": str(self.boundary_file) if self.boundary_file else None,
            "area_sq_km": self.get_area_sq_km(),
        }


class ROIValidator:
    """Validates ROI configuration before pipeline execution."""
    
    def __init__(self, roi: ROIConfig):
        self.roi = roi
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """Run all ROI validations. Returns True if valid."""
        self.errors = []
        self.warnings = []
        
        # Check ROI is defined
        if not self.roi.is_defined:
            self.errors.append(
                "ROI is NOT DEFINED. Pipeline MUST have explicit ROI. "
                "Provide either boundary_file or explicit bounds."
            )
            return False
        
        try:
            bounds = self.roi.bounds
        except ROIMissingError as e:
            self.errors.append(str(e))
            return False
        
        # Validate bounds are reasonable
        lon_min, lat_min, lon_max, lat_max = bounds
        
        if lon_min >= lon_max:
            self.errors.append(f"Invalid bounds: lon_min ({lon_min}) >= lon_max ({lon_max})")
        
        if lat_min >= lat_max:
            self.errors.append(f"Invalid bounds: lat_min ({lat_min}) >= lat_max ({lat_max})")
        
        # Check bounds are within valid lat/lon ranges
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            self.errors.append(f"Longitude out of range: [{lon_min}, {lon_max}]")
        
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            self.errors.append(f"Latitude out of range: [{lat_min}, {lat_max}]")
        
        # Check area is reasonable (not global)
        area = self.roi.get_area_sq_km()
        
        if area > self.roi.max_area_sq_km:
            self.errors.append(
                f"ROI too large ({area:.0f} sq km > {self.roi.max_area_sq_km:.0f} max). "
                "This is NOT a valid urban ROI. Specify a city-level region."
            )
        
        if area < self.roi.min_area_sq_km:
            self.warnings.append(
                f"ROI very small ({area:.2f} sq km). Check if bounds are correct."
            )
        
        # Log validation results
        if self.errors:
            for err in self.errors:
                logger.error("ROI VALIDATION FAILED: %s", err)
            return False
        
        for warn in self.warnings:
            logger.warning("ROI validation warning: %s", warn)
        
        logger.info(
            "ROI VALIDATED: %s (%.4f, %.4f) to (%.4f, %.4f), area=%.1f sq km",
            self.roi.city_name or "Unknown",
            lon_min, lat_min, lon_max, lat_max, area
        )
        
        return True


def create_roi_from_bounds(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    city_name: str = "",
    crs: str = "EPSG:4326",
) -> ROIConfig:
    """Create ROI from explicit bounding box."""
    return ROIConfig(
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
        city_name=city_name,
        crs=crs,
    )


def create_roi_from_boundary(
    boundary_file: Union[str, Path],
    city_name: str = "",
    crs: str = "EPSG:4326",
) -> ROIConfig:
    """Create ROI from boundary file (GeoJSON/Shapefile)."""
    return ROIConfig(
        boundary_file=Path(boundary_file),
        city_name=city_name,
        crs=crs,
    )


def load_roi_from_config(config_path: Union[str, Path]) -> ROIConfig:
    """Load ROI from JSON config file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ROIMissingError(f"ROI config file not found: {config_path}")
    
    with open(config_path) as f:
        data = json.load(f)
    
    if "boundary_file" in data and data["boundary_file"]:
        return create_roi_from_boundary(
            boundary_file=data["boundary_file"],
            city_name=data.get("city_name", ""),
            crs=data.get("crs", "EPSG:4326"),
        )
    elif "bounds" in data:
        b = data["bounds"]
        return create_roi_from_bounds(
            lon_min=b["lon_min"],
            lat_min=b["lat_min"],
            lon_max=b["lon_max"],
            lat_max=b["lat_max"],
            city_name=data.get("city_name", ""),
            crs=data.get("crs", "EPSG:4326"),
        )
    else:
        raise ROIMissingError("Config file missing bounds or boundary_file")


# Pre-defined ROIs for common cities
PREDEFINED_ROIS: Dict[str, ROIConfig] = {
    "seattle": create_roi_from_bounds(
        lon_min=-122.42,
        lat_min=47.50,
        lon_max=-122.24,
        lat_max=47.74,
        city_name="Seattle, WA",
    ),
    "mumbai": create_roi_from_bounds(
        lon_min=72.75,
        lat_min=18.88,
        lon_max=72.99,
        lat_max=19.27,
        city_name="Mumbai, India",
    ),
    "tokyo": create_roi_from_bounds(
        lon_min=139.50,
        lat_min=35.52,
        lon_max=139.92,
        lat_max=35.82,
        city_name="Tokyo, Japan",
    ),
    "london": create_roi_from_bounds(
        lon_min=-0.51,
        lat_min=51.28,
        lon_max=0.33,
        lat_max=51.69,
        city_name="London, UK",
    ),
    "new_york": create_roi_from_bounds(
        lon_min=-74.26,
        lat_min=40.49,
        lon_max=-73.70,
        lat_max=40.92,
        city_name="New York, NY",
    ),
}


def get_roi_for_city(city: str, config_dir: Optional[Path] = None) -> ROIConfig:
    """Get ROI for a city.
    
    Priority:
    1. Custom config file (config_dir/{city}_roi.json)
    2. Pre-defined ROI
    
    Raises:
        ROIMissingError: If no ROI found for city
    """
    city_lower = city.lower().replace(" ", "_")
    
    # Check for custom config
    if config_dir:
        config_file = config_dir / f"{city_lower}_roi.json"
        if config_file.exists():
            logger.info("Loading custom ROI from %s", config_file)
            return load_roi_from_config(config_file)
    
    # Check pre-defined ROIs
    if city_lower in PREDEFINED_ROIS:
        logger.info("Using pre-defined ROI for %s", city)
        return PREDEFINED_ROIS[city_lower]
    
    raise ROIMissingError(
        f"No ROI defined for city '{city}'. "
        f"Available pre-defined cities: {list(PREDEFINED_ROIS.keys())}. "
        "Provide custom ROI via config file or explicit bounds."
    )


def validate_crs_consistency(roi: ROIConfig, dataset_crs: str) -> bool:
    """Check CRS consistency between ROI and dataset."""
    roi_crs = roi.crs.upper()
    data_crs = dataset_crs.upper()
    
    # Normalize common variants
    if roi_crs in ["EPSG:4326", "WGS84", "WGS 84"]:
        roi_crs = "EPSG:4326"
    if data_crs in ["EPSG:4326", "WGS84", "WGS 84"]:
        data_crs = "EPSG:4326"
    
    if roi_crs != data_crs:
        logger.warning(
            "CRS mismatch: ROI=%s, dataset=%s. Data will be reprojected.",
            roi_crs, data_crs
        )
        return False
    
    return True


def clip_array_to_roi(
    data: np.ndarray,
    transform: Any,
    roi: ROIConfig,
    nodata: float = np.nan,
) -> Tuple[np.ndarray, Any]:
    """Clip array data to ROI bounds.
    
    Args:
        data: 2D or 3D numpy array (Y, X) or (T, Y, X)
        transform: Rasterio affine transform
        roi: ROI configuration
        nodata: Value for pixels outside ROI
        
    Returns:
        Tuple of (clipped_data, new_transform)
    """
    try:
        from rasterio.windows import from_bounds
        from rasterio.transform import from_origin
        
        lon_min, lat_min, lon_max, lat_max = roi.bounds
        
        # Get window from bounds
        if data.ndim == 3:
            height, width = data.shape[1], data.shape[2]
        else:
            height, width = data.shape
        
        window = from_bounds(
            lon_min, lat_min, lon_max, lat_max,
            transform=transform,
            width=width,
            height=height,
        )
        
        # Round to integer indices
        row_start = max(0, int(window.row_off))
        row_stop = min(height, int(window.row_off + window.height))
        col_start = max(0, int(window.col_off))
        col_stop = min(width, int(window.col_off + window.width))
        
        # Clip data
        if data.ndim == 3:
            clipped = data[:, row_start:row_stop, col_start:col_stop].copy()
        else:
            clipped = data[row_start:row_stop, col_start:col_stop].copy()
        
        # Update transform
        new_transform = from_origin(
            transform.c + col_start * transform.a,
            transform.f + row_start * transform.e,
            transform.a,
            -transform.e,
        )
        
        logger.info(
            "Clipped data to ROI: (%d,%d) to (%d,%d), new shape=%s",
            row_start, col_start, row_stop, col_stop, clipped.shape
        )
        
        return clipped, new_transform
        
    except Exception as e:
        logger.error("Failed to clip to ROI: %s", e)
        return data, transform


def point_in_roi(lon: float, lat: float, roi: ROIConfig) -> bool:
    """Check if a point is within ROI bounds."""
    lon_min, lat_min, lon_max, lat_max = roi.bounds
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def filter_points_to_roi(
    df: "pd.DataFrame",
    roi: ROIConfig,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> "pd.DataFrame":
    """Filter DataFrame to points within ROI.
    
    Args:
        df: DataFrame with coordinate columns
        roi: ROI configuration
        lon_col: Name of longitude column
        lat_col: Name of latitude column
        
    Returns:
        Filtered DataFrame
    """
    import pandas as pd
    
    lon_min, lat_min, lon_max, lat_max = roi.bounds
    
    mask = (
        (df[lon_col] >= lon_min) &
        (df[lon_col] <= lon_max) &
        (df[lat_col] >= lat_min) &
        (df[lat_col] <= lat_max)
    )
    
    n_before = len(df)
    filtered = df[mask].copy()
    n_after = len(filtered)
    
    logger.info(
        "ROI filter: %d/%d points (%.1f%%) within bounds",
        n_after, n_before, 100 * n_after / n_before if n_before > 0 else 0
    )
    
    return filtered
