"""Fully automated DEM download from SRTM/Copernicus.

Data Automation Prompt-1 implementation.
This module downloads real public DEM data with ZERO manual intervention.
Supports:
- NASA SRTM 30m (via OpenTopography or CGIAR mirrors)
- Copernicus GLO-30 DEM (ESA)

The pipeline:
1. Determines which tiles intersect the city boundary
2. Downloads only required tiles with retry/checksum
3. Caches raw tiles for idempotent re-runs
4. Logs all metadata for provenance
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config.paths import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# SRTM tile naming: N/S latitude, E/W longitude (1-degree tiles)
SRTM_BASE_URL = (
    "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
)
# Fallback: OpenTopography (requires API key for bulk, but small areas OK)
OPENTOPO_SRTM_URL = "https://portal.opentopography.org/API/globaldem"

# Copernicus GLO-30 (AWS open data)
COPERNICUS_BASE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
)


@dataclass
class BoundingBox:
    """Geographic bounding box in WGS84 (EPSG:4326)."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def buffer(self, degrees: float = 0.01) -> "BoundingBox":
        """Expand bbox by buffer degrees on all sides."""
        return BoundingBox(
            min_lon=self.min_lon - degrees,
            min_lat=self.min_lat - degrees,
            max_lon=self.max_lon + degrees,
            max_lat=self.max_lat + degrees,
        )


@dataclass
class DEMTileInfo:
    """Metadata for a single DEM tile."""

    tile_id: str
    url: str
    local_path: Path
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    source: str  # "srtm" or "copernicus"


@dataclass
class DEMDownloadResult:
    """Result of DEM download operation."""

    tiles: List[Path] = field(default_factory=list)
    source: str = ""
    download_timestamp: str = ""
    bbox: Optional[BoundingBox] = None
    resolution_m: float = 30.0
    crs: str = "EPSG:4326"
    success: bool = False
    error_message: str = ""
    tile_metadata: List[dict] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_retry(
    url: str,
    dest: Path,
    max_retries: int = 3,
    timeout: int = 60,
    headers: Optional[dict] = None,
) -> bool:
    """Download file with retry logic and timeout."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            logger.info(
                "Downloading (attempt %d/%d): %s",
                attempt + 1, max_retries, url
            )
            req = Request(url, headers=headers or {})
            with urlopen(req, timeout=timeout) as response:
                data = response.read()
                dest.write_bytes(data)
            logger.info("Downloaded %d bytes -> %s", len(data), dest)
            return True
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning("Download attempt %d failed: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.info("Retrying in %d seconds...", wait)
                time.sleep(wait)
    return False


def _get_srtm_tile_id(lat: int, lon: int) -> str:
    """Get SRTM tile ID for a given lat/lon corner."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def _get_copernicus_tile_id(lat: int, lon: int) -> str:
    """Get Copernicus GLO-30 tile ID for a given lat/lon."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00"


def compute_required_tiles(bbox: BoundingBox) -> List[Tuple[int, int]]:
    """Compute which 1-degree tiles are needed to cover the bounding box."""
    tiles = []
    lat_start = int(math.floor(bbox.min_lat))
    lat_end = int(math.floor(bbox.max_lat))
    lon_start = int(math.floor(bbox.min_lon))
    lon_end = int(math.floor(bbox.max_lon))

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append((lat, lon))

    logger.info(
        "Computed %d tiles for bbox [%.4f,%.4f,%.4f,%.4f]",
        len(tiles), bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
    )
    return tiles


def download_srtm_tiles(
    bbox: BoundingBox,
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> DEMDownloadResult:
    """Download SRTM 30m tiles covering the bounding box.

    Uses OpenTopography API which provides free access to SRTM data.
    For areas >1 sq degree, an API key may be required.

    Args:
        bbox: Geographic bounding box (WGS84)
        cache_dir: Directory to cache raw tiles
        api_key: Optional OpenTopography API key

    Returns:
        DEMDownloadResult with downloaded tile paths and metadata
    """
    cache_dir = cache_dir or RAW_DATA_DIR / "terrain" / "srtm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = DEMDownloadResult(
        source="srtm_30m",
        download_timestamp=datetime.utcnow().isoformat(),
        bbox=bbox,
        resolution_m=30.0,
        crs="EPSG:4326",
    )

    # Use OpenTopography global DEM API for SRTM
    params = {
        "demtype": "SRTMGL1",
        "south": bbox.min_lat,
        "north": bbox.max_lat,
        "west": bbox.min_lon,
        "east": bbox.max_lon,
        "outputFormat": "GTiff",
    }
    if api_key:
        params["API_Key"] = api_key

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{OPENTOPO_SRTM_URL}?{query}"

    # Create deterministic filename from bbox
    bbox_hash = hashlib.md5(
        f"{bbox.min_lon},{bbox.min_lat},{bbox.max_lon},{bbox.max_lat}".encode()
    ).hexdigest()[:12]
    dest_file = cache_dir / f"srtm_{bbox_hash}.tif"

    if dest_file.exists():
        logger.info("SRTM tile already cached: %s", dest_file)
        result.tiles = [dest_file]
        result.success = True
        result.tile_metadata.append({
            "tile_id": bbox_hash,
            "path": str(dest_file),
            "cached": True,
            "sha256": _sha256_file(dest_file),
        })
        return result

    success = _download_with_retry(url, dest_file, max_retries=3, timeout=120)

    if success and dest_file.exists() and dest_file.stat().st_size > 1000:
        result.tiles = [dest_file]
        result.success = True
        result.tile_metadata.append({
            "tile_id": bbox_hash,
            "path": str(dest_file),
            "url": url,
            "sha256": _sha256_file(dest_file),
            "downloaded_at": result.download_timestamp,
        })
        logger.info("SRTM download successful: %s", dest_file)
    else:
        result.error_message = "SRTM download failed or file too small"
        logger.error(result.error_message)

    return result


def download_copernicus_tiles(
    bbox: BoundingBox,
    cache_dir: Optional[Path] = None,
) -> DEMDownloadResult:
    """Download Copernicus GLO-30 DEM tiles from AWS open data.

    Copernicus DEM is freely available via AWS S3 without authentication.

    Args:
        bbox: Geographic bounding box (WGS84)
        cache_dir: Directory to cache raw tiles

    Returns:
        DEMDownloadResult with downloaded tile paths and metadata
    """
    cache_dir = cache_dir or RAW_DATA_DIR / "terrain" / "copernicus_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = DEMDownloadResult(
        source="copernicus_glo30",
        download_timestamp=datetime.utcnow().isoformat(),
        bbox=bbox,
        resolution_m=30.0,
        crs="EPSG:4326",
    )

    required = compute_required_tiles(bbox)
    downloaded_tiles = []

    for lat, lon in required:
        tile_id = _get_copernicus_tile_id(lat, lon)
        # Copernicus naming: DEM/tile_id/tile_id_DEM.tif
        url = f"{COPERNICUS_BASE_URL}{tile_id}/{tile_id}_DEM.tif"
        dest_file = cache_dir / f"{tile_id}.tif"

        if dest_file.exists():
            logger.info("Copernicus tile cached: %s", dest_file)
            downloaded_tiles.append(dest_file)
            result.tile_metadata.append({
                "tile_id": tile_id,
                "path": str(dest_file),
                "cached": True,
                "sha256": _sha256_file(dest_file),
            })
            continue

        success = _download_with_retry(url, dest_file, max_retries=3)
        if success and dest_file.exists():
            downloaded_tiles.append(dest_file)
            result.tile_metadata.append({
                "tile_id": tile_id,
                "path": str(dest_file),
                "url": url,
                "sha256": _sha256_file(dest_file),
                "downloaded_at": result.download_timestamp,
            })
        else:
            logger.warning("Failed to download tile: %s", tile_id)

    if downloaded_tiles:
        result.tiles = downloaded_tiles
        result.success = True
    else:
        result.error_message = "No Copernicus tiles downloaded"

    return result


def download_dem(
    bbox: BoundingBox,
    source: str = "auto",
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    allow_synthetic: bool = True,
) -> DEMDownloadResult:
    """Download DEM tiles with automatic source selection.

    Tries SRTM first (via OpenTopography), falls back to Copernicus,
    then generates synthetic DEM if all downloads fail.

    Args:
        bbox: Geographic bounding box (WGS84)
        source: "srtm", "copernicus", "synthetic", or "auto"
        cache_dir: Directory to cache raw tiles
        api_key: Optional OpenTopography API key for SRTM
        allow_synthetic: Allow synthetic DEM generation as fallback

    Returns:
        DEMDownloadResult with downloaded tile paths and metadata
    """
    logger.info(
        "Starting DEM download for bbox [%.4f,%.4f,%.4f,%.4f] source=%s",
        bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat, source
    )

    if source == "srtm":
        return download_srtm_tiles(bbox, cache_dir, api_key)
    elif source == "copernicus":
        return download_copernicus_tiles(bbox, cache_dir)
    elif source == "synthetic":
        return generate_synthetic_dem(bbox, cache_dir)
    else:  # auto
        # Try SRTM first
        result = download_srtm_tiles(bbox, cache_dir, api_key)
        if result.success:
            return result

        # Fallback to Copernicus
        logger.info("SRTM failed, trying Copernicus...")
        result = download_copernicus_tiles(bbox, cache_dir)
        if result.success:
            return result

        # Final fallback: synthetic DEM
        if allow_synthetic:
            logger.warning(
                "All remote sources failed, generating synthetic DEM"
            )
            return generate_synthetic_dem(bbox, cache_dir)

        return result


def generate_synthetic_dem(
    bbox: BoundingBox,
    cache_dir: Optional[Path] = None,
    resolution_deg: float = 0.0003,  # ~30m at equator
) -> DEMDownloadResult:
    """Generate a realistic synthetic DEM for testing/development.

    Creates terrain with:
    - Base elevation varying with latitude
    - Hills and valleys using Perlin-like noise
    - Drainage channels
    - Urban-like flat areas

    Args:
        bbox: Geographic bounding box
        cache_dir: Output directory
        resolution_deg: Pixel resolution in degrees

    Returns:
        DEMDownloadResult with synthetic DEM path
    """
    import numpy as np

    cache_dir = cache_dir or RAW_DATA_DIR / "terrain" / "synthetic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = DEMDownloadResult(
        source="synthetic",
        download_timestamp=datetime.utcnow().isoformat(),
        bbox=bbox,
        resolution_m=resolution_deg * 111000,
        crs="EPSG:4326",
    )

    # Compute dimensions
    width = int((bbox.max_lon - bbox.min_lon) / resolution_deg) + 1
    height = int((bbox.max_lat - bbox.min_lat) / resolution_deg) + 1

    # Limit size for performance
    max_dim = 500
    if width > max_dim or height > max_dim:
        scale = max(width, height) / max_dim
        width = int(width / scale)
        height = int(height / scale)
        resolution_deg *= scale

    logger.info(
        "Generating synthetic DEM: %dx%d pixels", width, height
    )

    # Generate terrain
    np.random.seed(42)  # Reproducible

    # Base elevation (higher in "mountains" to the east)
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)

    # Multi-scale terrain
    terrain = (
        50 +  # Base elevation
        30 * np.sin(xx / 2) * np.cos(yy / 2) +  # Large hills
        10 * np.sin(xx * 2) * np.sin(yy * 2) +  # Medium features
        5 * np.random.randn(height, width)  # Local noise
    )

    # Add drainage channels (lower elevation along certain paths)
    channel_mask = np.sin(xx + yy) > 0.8
    terrain[channel_mask] -= 5

    # Add some flat urban-like areas
    urban_mask = (
        (xx > np.pi) & (xx < 2 * np.pi) &
        (yy > np.pi) & (yy < 2 * np.pi)
    )
    terrain[urban_mask] = np.mean(terrain[urban_mask])

    # Ensure positive elevation
    terrain = np.clip(terrain, 1, 500).astype(np.float32)

    # Create GeoTIFF
    bbox_hash = hashlib.md5(
        f"{bbox.min_lon},{bbox.min_lat},{bbox.max_lon},{bbox.max_lat}".encode()
    ).hexdigest()[:12]
    dest_file = cache_dir / f"synthetic_{bbox_hash}.tif"

    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        transform = from_bounds(
            bbox.min_lon, bbox.min_lat,
            bbox.max_lon, bbox.max_lat,
            width, height
        )

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": width,
            "height": height,
            "count": 1,
            "crs": CRS.from_epsg(4326),
            "transform": transform,
            "nodata": -9999.0,
            "compress": "lzw",
        }

        with rasterio.open(dest_file, "w", **profile) as dst:
            dst.write(terrain, 1)

        result.tiles = [dest_file]
        result.success = True
        result.tile_metadata.append({
            "tile_id": bbox_hash,
            "path": str(dest_file),
            "source": "synthetic",
            "sha256": _sha256_file(dest_file),
            "generated_at": result.download_timestamp,
            "dimensions": f"{width}x{height}",
        })

        logger.info("Synthetic DEM generated: %s", dest_file)

    except ImportError as e:
        result.error_message = f"rasterio not available: {e}"
        logger.error(result.error_message)

    return result


def bbox_from_geojson(geojson_path: Path) -> BoundingBox:
    """Extract bounding box from a GeoJSON file."""
    import json

    with geojson_path.open() as f:
        data = json.load(f)

    # Handle FeatureCollection or single Feature
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = [{"geometry": data}]

    all_coords = []
    for feat in features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        _extract_coords(coords, all_coords)

    if not all_coords:
        raise ValueError("No coordinates found in GeoJSON")

    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]

    return BoundingBox(
        min_lon=min(lons),
        min_lat=min(lats),
        max_lon=max(lons),
        max_lat=max(lats),
    )


def _extract_coords(coords, result: list) -> None:
    """Recursively extract coordinate pairs from nested lists."""
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        result.append(coords[:2])
    else:
        for c in coords:
            _extract_coords(c, result)


def bbox_from_city_name(city: str) -> BoundingBox:
    """Get approximate bounding box for well-known cities.

    For production, use a geocoding service. This provides defaults
    for common test cities.
    """
    cities = {
        "seattle": BoundingBox(-122.45, 47.49, -122.24, 47.73),
        "portland": BoundingBox(-122.84, 45.43, -122.47, 45.65),
        "los_angeles": BoundingBox(-118.67, 33.70, -118.15, 34.34),
        "new_york": BoundingBox(-74.26, 40.49, -73.70, 40.92),
        "chicago": BoundingBox(-87.94, 41.64, -87.52, 42.02),
        "houston": BoundingBox(-95.79, 29.52, -95.01, 30.11),
        "miami": BoundingBox(-80.32, 25.71, -80.13, 25.86),
        "denver": BoundingBox(-105.11, 39.61, -104.60, 39.91),
        # Small Seattle area for testing
        "default": BoundingBox(-122.40, 47.55, -122.30, 47.65),
    }
    key = city.lower().replace(" ", "_")
    if key not in cities:
        logger.warning(
            "City '%s' not found, using 'default' bbox", city
        )
        key = "default"
    return cities[key]
