"""Automated data acquisition for rainfall, complaints, and DEM (Prompt-11)."""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from src.config.paths import RAW_DATA_DIR
from src.data.registry import DataRegistry

logger = logging.getLogger(__name__)

# Default public datasets (small but real)
DEFAULT_RAINFALL_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/"
    "2016-weather-data-seattle.csv"
)
# Use synthetic data for complaints since public 311 datasets are unreliable
DEFAULT_COMPLAINTS_URL: Optional[str] = None
DEFAULT_DEM_URL = (
    "https://raw.githubusercontent.com/OSGeo/gdal/master/autotest/"
    "gcore/data/utmsmall.tif"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    urlretrieve(url, dest)  # noqa: S310 (controlled URLs)
    return dest


def _generate_synthetic_complaints(dest: Path, n_rows: int = 500) -> Path:
    """Generate synthetic 311 complaints data when no URL is available."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic complaints data -> %s", dest)

    random.seed(42)
    base_date = datetime(2016, 1, 1)
    complaint_types = [
        "Flooding", "Sewer Backup", "Storm Drain Clogged",
        "Standing Water", "Drainage Issue", "Street Flooding"
    ]

    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "complaint_id", "timestamp", "latitude", "longitude",
            "complaint_type", "description"
        ])
        for i in range(n_rows):
            ts = base_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            lat = 47.5 + random.uniform(0, 0.2)  # Seattle area approx
            lon = -122.4 + random.uniform(0, 0.2)
            ctype = random.choice(complaint_types)
            writer.writerow([
                f"C{i:06d}",
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                f"{lat:.6f}",
                f"{lon:.6f}",
                ctype,
                f"Sample {ctype.lower()} complaint"
            ])
    return dest


def _generate_synthetic_precipitation(dest: Path, city: str, n_days: int = 365) -> Path:
    """Generate synthetic precipitation data as last-resort fallback.
    
    This creates precipitation data with realistic patterns:
    - Seasonal variation (more rain in winter)
    - Random storm events
    - Many dry days (realistic for most climates)
    
    NOTE: This is SYNTHETIC and should only be used if Open-Meteo API fails.
    """
    import numpy as np
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "FALLBACK: Generating SYNTHETIC precipitation data -> %s "
        "(Real API failed - results will be less accurate)", 
        dest
    )
    
    random.seed(42)
    np.random.seed(42)
    
    # City coordinates for metadata
    city_coords = {
        "seattle": (47.6, -122.33),
        "portland": (45.52, -122.68),
        "new_york": (40.71, -74.01),
        "chicago": (41.88, -87.63),
        "miami": (25.76, -80.19),
    }
    lat, lon = city_coords.get(city.lower(), (47.6, -122.33))
    
    base_date = datetime(2023, 1, 1)
    
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "latitude", "longitude", "precipitation_mm"])
        
        for day in range(n_days):
            ts = base_date + timedelta(days=day)
            
            # Seasonal factor (more rain in winter for Seattle-like climate)
            day_of_year = ts.timetuple().tm_yday
            seasonal = 0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - 30) / 365)
            
            # Base probability of rain
            p_rain = 0.3 * seasonal
            
            if random.random() < p_rain:
                # Generate rain amount (exponential distribution)
                precip = np.random.exponential(scale=5.0 * seasonal)
                # Cap at reasonable maximum
                precip = min(precip, 50.0)
            else:
                precip = 0.0
            
            writer.writerow([
                ts.strftime("%Y-%m-%d"),
                f"{lat:.4f}",
                f"{lon:.4f}",
                f"{precip:.2f}"
            ])
    
    sha = _sha256(dest)
    DataRegistry().record(
        "rainfall", dest, "synthetic_fallback", 
        sha, notes=f"city={city}, WARNING=synthetic_data"
    )
    return dest
    return dest


def _generate_synthetic_dem(dest: Path, size: int = 100) -> Path:
    """Generate a simple synthetic DEM GeoTIFF when no URL is available.
    
    Uses Seattle's actual coordinates to cover real complaint locations:
    - Lat: 47.50 to 47.74 (complaints span 47.5003 to 47.7339)
    - Lon: -122.42 to -122.24 (complaints span -122.4172 to -122.2485)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic DEM data -> %s", dest)

    import numpy as np
    
    # Seattle complaint area bounds
    lon_min, lon_max = -122.42, -122.24
    lat_min, lat_max = 47.50, 47.74
    pixel_width = (lon_max - lon_min) / size   # ~0.0018
    pixel_height = -(lat_max - lat_min) / size  # negative for north-up
    
    # Generate simple elevation surface (Seattle-like terrain)
    np.random.seed(42)  # Reproducibility
    x = np.linspace(0, 2 * np.pi, size)
    y = np.linspace(0, 2 * np.pi, size)
    xx, yy = np.meshgrid(x, y)
    elevation = 50 + 30 * np.sin(xx) * np.cos(yy) + np.random.randn(size, size) * 5
    
    # Try rasterio first (more commonly available)
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, size, size)
        
        with rasterio.open(
            dest,
            'w',
            driver='GTiff',
            height=size,
            width=size,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(elevation.astype(np.float32), 1)
        
        logger.info("Created synthetic DEM with rasterio: %s", dest)
        return dest
        
    except ImportError:
        pass
    
    # Try GDAL as fallback
    try:
        from osgeo import gdal, osr

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(dest), size, size, 1, gdal.GDT_Float32)

        # GeoTransform: (origin_x, pixel_width, 0, origin_y, 0, pixel_height)
        ds.SetGeoTransform((lon_min, pixel_width, 0, lat_max, 0, pixel_height))

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())

        ds.GetRasterBand(1).WriteArray(elevation.astype(np.float32))
        ds.FlushCache()
        ds = None
        
        logger.info("Created synthetic DEM with GDAL: %s", dest)
        return dest
        
    except ImportError:
        # Last resort: create a minimal placeholder file
        logger.warning("Neither rasterio nor GDAL available, creating placeholder DEM")
        dest.write_bytes(b"PLACEHOLDER_DEM")
        return dest


def ensure_rainfall(city: str, url: Optional[str] = None) -> Path:
    """Ensure real precipitation data is available.
    
    CRITICAL: Uses Open-Meteo API to get REAL precipitation data, not temperature.
    The old DEFAULT_RAINFALL_URL points to Seattle weather which has TEMPERATURE,
    not precipitation columns. This caused the pipeline to use synthetic fallback.
    """
    dest = RAW_DATA_DIR / "rainfall" / f"rainfall_{city}.csv"
    
    # Check if we already have REAL precipitation data (not the old temperature file)
    if dest.exists():
        # Validate it's real precipitation, not temperature data
        try:
            import pandas as pd
            df = pd.read_csv(dest, nrows=5)
            if 'precipitation_mm' in df.columns or 'precipitation' in df.columns:
                logger.info("Real precipitation data already present: %s", dest)
                return dest
            else:
                logger.warning(
                    "Existing rainfall file has wrong columns (%s), "
                    "fetching REAL precipitation data",
                    list(df.columns)
                )
                # Remove the bad file
                dest.unlink()
        except Exception as e:
            logger.warning("Error validating rainfall file: %s", e)
            dest.unlink()
    
    # Use real precipitation module
    try:
        from src.data.real_precipitation import ensure_real_rainfall
        path = ensure_real_rainfall(city)
        sha = _sha256(path)
        DataRegistry().record(
            "rainfall", path, "Open-Meteo Archive API (real precipitation)", 
            sha, notes=f"city={city}, source=Open-Meteo"
        )
        return path
    except Exception as e:
        logger.error("Failed to fetch real precipitation: %s", e)
        # Last resort: generate synthetic with precipitation column
        logger.warning("Generating synthetic precipitation as fallback")
        return _generate_synthetic_precipitation(dest, city)


def ensure_complaints(
    city: str, 
    url: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    use_real_data: bool = True,
) -> Path:
    """Ensure complaints data exists, downloading real data if available.
    
    Args:
        city: City name (e.g., 'seattle', 'chicago', 'new_york')
        url: Optional direct URL to download from
        start_date: Start date for data range (default: 1 year ago)
        end_date: End date for data range (default: today)
        use_real_data: If True, try to download real 311 data first
    """
    url = url or DEFAULT_COMPLAINTS_URL
    dest = RAW_DATA_DIR / "complaints" / f"complaints_{city}.csv"
    if dest.exists():
        logger.info("Complaints already present: %s", dest)
        return dest

    # Set default date range
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    # Try to download real data first
    if use_real_data:
        try:
            from src.ingestion.complaints_download import (
                download_seattle_csr,
                download_nyc_311,
                download_chicago_311,
            )
            
            city_lower = city.lower()
            result = None
            
            if city_lower == "seattle":
                logger.info("Downloading REAL Seattle complaints from data.seattle.gov...")
                result = download_seattle_csr(start_date, end_date)
            elif city_lower in ("new_york", "nyc"):
                logger.info("Downloading REAL NYC complaints from data.cityofnewyork.us...")
                result = download_nyc_311(start_date, end_date)
            elif city_lower == "chicago":
                logger.info("Downloading REAL Chicago complaints from data.cityofchicago.org...")
                result = download_chicago_311(start_date, end_date)
            
            if result and result.success and result.file_path:
                # Copy/move downloaded file to expected location
                import shutil
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(result.file_path, dest)
                logger.info(
                    "Downloaded %d REAL complaints from %s (quality=%.1f)",
                    result.record_count, result.source_id, result.quality_score
                )
                sha = _sha256(dest)
                DataRegistry().record(
                    "complaints", dest, result.source_id,
                    sha, notes=f"city={city}, real_data=True, records={result.record_count}"
                )
                return dest
            elif result:
                logger.warning(
                    "Real data download failed: %s. Falling back to synthetic.",
                    result.validation_errors
                )
        except Exception as e:
            logger.warning("Failed to download real complaints: %s. Using synthetic.", e)

    # Fallback to URL or synthetic
    if url:
        path = _download(url, dest)
    else:
        # Generate synthetic data if no URL provided
        path = _generate_synthetic_complaints(dest)

    sha = _sha256(path)
    source = url or "synthetic"
    reg = DataRegistry()
    reg.record("complaints", path, source, sha, notes=f"city={city}")
    return path


def ensure_dem(city: str, url: Optional[str] = None) -> Path:
    """Ensure DEM is available, using synthetic Seattle-coordinates DEM if needed.
    
    For Seattle, we generate a synthetic DEM covering the actual complaint area
    (lat 47.50-47.74, lon -122.42 to -122.24) to ensure coordinate alignment.
    """
    dest = RAW_DATA_DIR / "terrain" / f"dem_{city}.tif"
    if dest.exists():
        logger.info("DEM already present: %s", dest)
        return dest

    # For Seattle, always use synthetic to ensure correct coordinates
    # (Real DEM from URLs may have different CRS/bounds)
    if city.lower() == "seattle":
        logger.info("Generating Seattle DEM with real complaint coordinates...")
        path = _generate_synthetic_dem(dest)
        url = None
    elif url:
        try:
            path = _download(url, dest)
        except Exception as e:
            logger.warning("DEM download failed (%s), generating synthetic", e)
            path = _generate_synthetic_dem(dest)
            url = None
    else:
        url = url or DEFAULT_DEM_URL
        try:
            path = _download(url, dest)
        except Exception as e:
            logger.warning("DEM download failed (%s), generating synthetic", e)
            path = _generate_synthetic_dem(dest)
            url = None

    sha = _sha256(path)
    source = url or "synthetic"
    DataRegistry().record("dem", path, source, sha, notes=f"city={city}")
    return path
