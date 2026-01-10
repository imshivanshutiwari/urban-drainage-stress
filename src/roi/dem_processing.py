"""DEM ingestion, mosaic, clip, validation, and flow derivation.

Extended for Data Automation Prompt-1 with:
- Multi-tile mosaic
- Boundary clipping
- Geospatial validation
- Provenance tracking
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.mask import mask as rasterio_mask
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box, mapping, shape

from ..config.parameters import get_default_parameters

logger = logging.getLogger(__name__)


@dataclass
class DEMMetadata:
    crs: str
    resolution: Tuple[float, float]
    nodata: Optional[float]
    spike_mask_fraction: float
    flat_mask_fraction: float
    notes: List[str]


def load_dem(
    path: Path | str,
) -> Tuple[np.ndarray, DatasetReader, DEMMetadata]:
    """Load DEM, validate metadata, and detect spikes/flats.

    Returns the DEM array, the dataset, and derived metadata.
    
    CRITICAL: Always cast DEM to float64 to avoid dtype casting errors
    during arithmetic operations (e.g., sink filling, flow direction).
    """

    params = get_default_parameters().terrain
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DEM file not found: {path}")

    ds = rasterio.open(path)
    dem = ds.read(1, masked=False)
    
    # CRITICAL FIX: Cast to float64 to avoid dtype errors
    # Original dtype may be uint8, int16, etc. which causes:
    # "Cannot cast ufunc 'add' output from dtype('float64') to dtype('uint8')"
    original_dtype = dem.dtype
    dem = dem.astype(np.float64)
    
    logger.info(
        "DEM loaded: shape=%s, original_dtype=%s, cast to float64",
        dem.shape, original_dtype
    )

    notes: List[str] = []
    notes.append(f"Original dtype: {original_dtype}, cast to float64")
    
    if ds.crs is None:
        raise ValueError("DEM is missing CRS; cannot proceed")
    if ds.res is None:
        raise ValueError("DEM resolution missing; cannot proceed")

    nodata = ds.nodata
    if nodata is None:
        notes.append("No explicit nodata; treating all values as valid")
    else:
        dem = np.where(dem == nodata, np.nan, dem)
        
    # Handle scale factor if present (common in some DEMs)
    if hasattr(ds, 'scales') and ds.scales and ds.scales[0] != 1.0:
        scale = ds.scales[0]
        dem = dem * scale
        notes.append(f"Applied scale factor: {scale}")
        logger.info("Applied DEM scale factor: %s", scale)
    
    if hasattr(ds, 'offsets') and ds.offsets and ds.offsets[0] != 0.0:
        offset = ds.offsets[0]
        dem = dem + offset
        notes.append(f"Applied offset: {offset}")
        logger.info("Applied DEM offset: %s", offset)

    spike_mask, flat_mask = detect_artifacts(
        dem,
        spike_z=params.spike_zscore_threshold,
        flat_deg=params.flat_slope_threshold,
        transform=ds.transform,
    )

    if spike_mask.mean() > 0.01:
        logger.warning(
            "DEM spike fraction %.3f exceeds 1%%; consider manual QC",
            spike_mask.mean(),
        )
    if flat_mask.mean() > 0.05:
        logger.warning(
            "DEM flat fraction %.3f exceeds 5%%; urban flattening likely",
            flat_mask.mean(),
        )

    meta = DEMMetadata(
        crs=str(ds.crs),
        resolution=ds.res,
        nodata=nodata,
        spike_mask_fraction=float(spike_mask.mean()),
        flat_mask_fraction=float(flat_mask.mean()),
        notes=notes,
    )

    return dem, ds, meta


def detect_artifacts(
    dem: np.ndarray, spike_z: float, flat_deg: float, transform: Affine
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect extreme spikes and near-flat artifacts."""

    if dem.ndim != 2:
        raise ValueError("DEM must be 2D")

    # Spikes: z-score vs neighborhood median (3x3)
    padded = np.pad(dem, 1, mode="edge")
    window = np.stack(
        [
            padded[i: i + dem.shape[0], j: j + dem.shape[1]]
            for i in range(3)
            for j in range(3)
        ]
    )
    neighborhood_med = np.nanmedian(window, axis=0)
    neighborhood_mad = (
        np.nanmedian(np.abs(window - neighborhood_med), axis=0) + 1e-6
    )
    zscore = (dem - neighborhood_med) / neighborhood_mad
    spike_mask = np.abs(zscore) > spike_z

    # Flats: slope magnitude below threshold
    slope_mag = slope_magnitude(dem, transform)
    flat_mask = slope_mag < flat_deg

    return spike_mask, flat_mask


def slope_magnitude(dem: np.ndarray, transform: Affine) -> np.ndarray:
    """Compute slope magnitude (degrees) via finite differences."""

    xres = abs(transform.a)
    yres = abs(transform.e)
    dzdx = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * xres)
    dzdy = (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0)) / (2 * yres)
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


def preprocess_dem(
    dem: np.ndarray,
    fill_sinks: bool = True,
    sink_fill_max_depth_m: float = 0.5,
    slope_smooth_kernel: int = 3,
) -> Tuple[np.ndarray, dict]:
    """Conservative preprocessing with metadata flags.

    - Sink filling limited by depth to avoid over-flattening.
    - Optional slope smoothing with small kernel.
    """

    meta = {
        "sink_filled": False,
        "smoothing_applied": False,
        "sink_fill_max_depth_m": sink_fill_max_depth_m,
        "smoothing_kernel": slope_smooth_kernel,
    }

    out = dem.copy()

    if fill_sinks:
        out, filled_fraction = _fill_sinks_limited(out, sink_fill_max_depth_m)
        meta["sink_filled"] = True
        meta["sink_filled_fraction"] = filled_fraction
        logger.info(
            "Sink filling applied (max depth %.2fm); cells adjusted: %.3f%%",
            sink_fill_max_depth_m,
            filled_fraction * 100,
        )

    if slope_smooth_kernel and slope_smooth_kernel > 1:
        out = _bounded_mean_filter(out, slope_smooth_kernel)
        meta["smoothing_applied"] = True
        logger.info(
            "Slope smoothing applied with kernel %d",
            slope_smooth_kernel,
        )

    return out, meta


def _fill_sinks_limited(
    dem: np.ndarray, max_depth: float
) -> Tuple[np.ndarray, float]:
    """Very conservative single-pass sink fill capped by max_depth."""

    filled = dem.copy()
    # For each cell, ensure it is not more than max_depth below min neighbor.
    neighbors_min = _min_neighbor(filled)
    adjustment = np.clip(neighbors_min - filled, 0, max_depth)
    filled += adjustment
    filled_fraction = np.mean(adjustment > 0)
    return filled, filled_fraction


def _min_neighbor(arr: np.ndarray) -> np.ndarray:
    stacks = [
        np.roll(np.roll(arr, i, axis=0), j, axis=1)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if not (i == 0 and j == 0)
    ]
    return np.minimum.reduce(stacks)


def _bounded_mean_filter(arr: np.ndarray, kernel: int) -> np.ndarray:
    if kernel % 2 == 0:
        raise ValueError("Smoothing kernel must be odd")
    pad = kernel // 2
    padded = np.pad(arr, pad, mode="edge")
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            window = padded[i: i + kernel, j: j + kernel]
            out[i, j] = np.nanmean(window)
    return out


def flow_direction_d8(dem: np.ndarray) -> np.ndarray:
    """Compute D8 flow direction (0-7 codes) toward steepest descent neighbor.

    Codes follow clockwise from north: 0=N,1=NE,2=E,3=SE,4=S,5=SW,6=W,7=NW.
    """

    dirs = np.full(dem.shape, -1, dtype=np.int8)
    offsets = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]
    for i in range(1, dem.shape[0] - 1):
        for j in range(1, dem.shape[1] - 1):
            dz = []
            for code, (di, dj) in enumerate(offsets):
                dz.append(dem[i, j] - dem[i + di, j + dj])
            dz = np.array(dz)
            if np.all(dz <= 0):
                continue  # pit or flat
            dirs[i, j] = int(np.argmax(dz))
    return dirs


def flow_accumulation_index(
    dem: np.ndarray, flow_dir: np.ndarray
) -> np.ndarray:
    """Approximate flow accumulation using D8 routing (structural index)."""

    offsets = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]
    h, w = dem.shape
    # Each cell contributes 1 to itself
    acc = np.ones((h, w), dtype=np.float32)

    # Process cells from high to low elevation to approximate routing order.
    flat_inds = np.argsort(dem, axis=None)[::-1]
    for idx in flat_inds:
        i, j = np.unravel_index(idx, dem.shape)
        d = flow_dir[i, j]
        if d < 0:
            continue
        di, dj = offsets[d]
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            acc[ni, nj] += acc[i, j]
    return acc


def upstream_contribution_index(
    acc: np.ndarray, artifacts_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize accumulation to [0,1] and attach confidence proxy."""

    norm = (acc - np.nanmin(acc)) / (np.nanmax(acc) - np.nanmin(acc) + 1e-6)
    confidence = np.ones_like(norm, dtype=np.float32)
    if artifacts_mask is not None:
        confidence = np.where(artifacts_mask, 0.5, 1.0)
    params = get_default_parameters().terrain
    confidence = np.clip(confidence, params.accumulation_confidence_floor, 1.0)
    return norm.astype(np.float32), confidence.astype(np.float32)


def query_upstream_contribution(
    contribution: np.ndarray,
    transform: Affine,
    points: Iterable[Tuple[float, float]],
) -> List[float]:
    """Sample contribution index at arbitrary lon/lat points."""

    results: List[float] = []
    for lon, lat in points:
        col, row = ~transform * (lon, lat)
        r, c = int(round(row)), int(round(col))
        if 0 <= r < contribution.shape[0] and 0 <= c < contribution.shape[1]:
            results.append(float(contribution[r, c]))
        else:
            results.append(np.nan)
    return results


def aggregate_contribution(
    contribution: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Aggregate contribution over grid (mean), respecting optional mask."""

    if mask is not None:
        vals = contribution[mask]
    else:
        vals = contribution
    return float(np.nanmean(vals))


def dem_pipeline(path: Path | str) -> dict:
    """End-to-end DEM processing returning artifacts, flow, and contribution.

    This is intentionally approximate; outputs are structural priors with
    documented uncertainty, not hydraulic truth.
    """

    params = get_default_parameters().terrain
    dem, ds, meta = load_dem(path)

    dem_proc, pre_meta = preprocess_dem(
        dem,
        fill_sinks=params.fill_sinks,
        sink_fill_max_depth_m=params.sink_fill_max_depth_m,
        slope_smooth_kernel=params.slope_smooth_kernel,
    )

    flow_dir = flow_direction_d8(dem_proc)
    acc = flow_accumulation_index(dem_proc, flow_dir)
    spike_mask, flat_mask = detect_artifacts(
        dem_proc,
        params.spike_zscore_threshold,
        params.flat_slope_threshold,
        ds.transform,
    )
    contribution, confidence = upstream_contribution_index(
        acc,
        artifacts_mask=(spike_mask | flat_mask),
    )

    return {
        "dem": dem,
        "dem_processed": dem_proc,
        "flow_direction": flow_dir,
        "flow_accumulation_index": acc,
        "upstream_contribution": contribution,
        "contribution_confidence": confidence,
        "metadata": meta,
        "preprocess_metadata": pre_meta,
        "artifact_masks": {"spike": spike_mask, "flat": flat_mask},
        "transform": ds.transform,
        "crs": ds.crs,
    }


# ============================================================================
# MOSAIC, CLIP, AND VALIDATION (Data Automation Prompt-1)
# ============================================================================


@dataclass
class DEMValidationResult:
    """Result of DEM validation checks."""

    is_valid: bool = True
    crs_valid: bool = True
    resolution_valid: bool = True
    coverage_valid: bool = True
    artifacts_acceptable: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def mosaic_tiles(
    tile_paths: List[Path],
    output_path: Path,
    nodata: float = -9999.0,
) -> Path:
    """Mosaic multiple DEM tiles into a single raster.

    This function:
    - Opens all input tiles
    - Merges them with proper handling of overlaps (uses first/last)
    - Writes to output with preserved metadata
    - NO silent resampling is performed

    Args:
        tile_paths: List of paths to input DEM tiles
        output_path: Path for mosaicked output
        nodata: Nodata value to use

    Returns:
        Path to mosaicked DEM
    """
    if not tile_paths:
        raise ValueError("No tiles provided for mosaic")

    if len(tile_paths) == 1:
        # Single tile, just copy
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tile_paths[0], output_path)
        logger.info("Single tile, copied to %s", output_path)
        return output_path

    logger.info("Mosaicking %d tiles...", len(tile_paths))

    datasets = [rasterio.open(p) for p in tile_paths]

    try:
        # Verify all tiles have same CRS
        crs_set = {str(ds.crs) for ds in datasets}
        if len(crs_set) > 1:
            raise ValueError(f"CRS mismatch in tiles: {crs_set}")

        # Merge tiles
        mosaic_arr, mosaic_transform = merge(
            datasets,
            nodata=nodata,
            method="first",  # Use first valid value in overlaps
        )

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,
            count=1,
            dtype=mosaic_arr.dtype,
            nodata=nodata,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic_arr[0], 1)

        logger.info(
            "Mosaic complete: %s (%dx%d)",
            output_path, mosaic_arr.shape[2], mosaic_arr.shape[1]
        )
        return output_path

    finally:
        for ds in datasets:
            ds.close()


def clip_to_boundary(
    dem_path: Path,
    boundary: Union[dict, Path],
    output_path: Path,
    buffer_m: float = 100.0,
) -> Path:
    """Clip DEM to a city/region boundary with optional buffer.

    Args:
        dem_path: Path to input DEM
        boundary: GeoJSON dict, shapely geometry, or path to GeoJSON file
        output_path: Path for clipped output
        buffer_m: Buffer in meters to add around boundary

    Returns:
        Path to clipped DEM
    """
    logger.info("Clipping DEM to boundary...")

    # Load boundary geometry
    if isinstance(boundary, Path):
        with boundary.open() as f:
            boundary = json.load(f)

    if isinstance(boundary, dict):
        if boundary.get("type") == "FeatureCollection":
            geoms = [
                shape(f["geometry"]) for f in boundary.get("features", [])
            ]
            from shapely.ops import unary_union
            geom = unary_union(geoms)
        elif boundary.get("type") == "Feature":
            geom = shape(boundary["geometry"])
        else:
            geom = shape(boundary)
    else:
        geom = boundary

    with rasterio.open(dem_path) as src:
        # Buffer in degrees (approximate for WGS84)
        if buffer_m > 0:
            buf_deg = buffer_m / 111000.0  # ~111km per degree
            geom = geom.buffer(buf_deg)

        # Clip to geometry
        out_image, out_transform = rasterio_mask(
            src,
            [mapping(geom)],
            crop=True,
            nodata=src.nodata or -9999.0,
        )

        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="lzw",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out_image[0], 1)

    logger.info(
        "Clipped DEM: %s (%dx%d)",
        output_path, out_image.shape[2], out_image.shape[1]
    )
    return output_path


def clip_to_bbox(
    dem_path: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    output_path: Path,
) -> Path:
    """Clip DEM to a simple bounding box."""
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    return clip_to_boundary(
        dem_path,
        mapping(bbox_geom),
        output_path,
        buffer_m=0,
    )


def reproject_dem(
    dem_path: Path,
    output_path: Path,
    target_crs: str = "EPSG:4326",
    target_resolution: Optional[Tuple[float, float]] = None,
) -> Path:
    """Reproject DEM to target CRS with optional resolution.

    Args:
        dem_path: Input DEM path
        output_path: Output path
        target_crs: Target CRS (default WGS84)
        target_resolution: Optional (x_res, y_res) in target CRS units

    Returns:
        Path to reprojected DEM
    """
    with rasterio.open(dem_path) as src:
        dst_crs = CRS.from_string(target_crs)

        if str(src.crs) == target_crs and target_resolution is None:
            # Already in target CRS, just copy
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(dem_path, output_path)
            return output_path

        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution,
        )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            compress="lzw",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

    logger.info("Reprojected DEM to %s: %s", target_crs, output_path)
    return output_path


def validate_dem(
    dem_path: Path,
    expected_crs: Optional[str] = None,
    expected_resolution_m: Optional[float] = None,
    expected_bbox: Optional[Tuple[float, float, float, float]] = None,
    max_artifact_fraction: float = 0.05,
) -> DEMValidationResult:
    """Comprehensive DEM validation.

    Validates:
    - CRS correctness
    - Pixel resolution
    - Bounding box coverage
    - Absence of extreme elevation artifacts

    Args:
        dem_path: Path to DEM file
        expected_crs: Expected CRS string (e.g., "EPSG:4326")
        expected_resolution_m: Expected resolution in meters
        expected_bbox: Expected (min_lon, min_lat, max_lon, max_lat)
        max_artifact_fraction: Maximum acceptable artifact fraction

    Returns:
        DEMValidationResult with validation outcomes
    """
    result = DEMValidationResult()

    if not dem_path.exists():
        result.is_valid = False
        result.errors.append(f"DEM file not found: {dem_path}")
        return result

    try:
        with rasterio.open(dem_path) as ds:
            # CRS validation
            if ds.crs is None:
                result.crs_valid = False
                result.errors.append("DEM is missing CRS")
            elif expected_crs:
                actual_crs = str(ds.crs)
                if actual_crs != expected_crs:
                    result.crs_valid = False
                    result.errors.append(
                        f"CRS mismatch: expected {expected_crs}, "
                        f"got {actual_crs}"
                    )
                result.metrics["crs"] = actual_crs

            # Resolution validation
            if ds.res:
                res_deg = (abs(ds.res[0]) + abs(ds.res[1])) / 2
                res_m = res_deg * 111000  # Approximate for WGS84
                result.metrics["resolution_deg"] = res_deg
                result.metrics["resolution_m_approx"] = res_m

                if expected_resolution_m:
                    tolerance = expected_resolution_m * 0.2  # 20% tolerance
                    if abs(res_m - expected_resolution_m) > tolerance:
                        result.resolution_valid = False
                        result.warnings.append(
                            f"Resolution {res_m:.1f}m differs from "
                            f"expected {expected_resolution_m}m"
                        )
            else:
                result.resolution_valid = False
                result.errors.append("DEM resolution not available")

            # Coverage validation
            if expected_bbox:
                emin_lon, emin_lat, emax_lon, emax_lat = expected_bbox
                actual = ds.bounds
                result.metrics["bounds"] = {
                    "left": actual.left,
                    "bottom": actual.bottom,
                    "right": actual.right,
                    "top": actual.top,
                }

                if (actual.left > emin_lon or actual.bottom > emin_lat or
                        actual.right < emax_lon or actual.top < emax_lat):
                    result.coverage_valid = False
                    result.errors.append(
                        f"DEM does not fully cover expected bbox: "
                        f"expected [{emin_lon},{emin_lat},"
                        f"{emax_lon},{emax_lat}] "
                        f"got [{actual.left:.4f},{actual.bottom:.4f},"
                        f"{actual.right:.4f},{actual.top:.4f}]"
                    )

            # Artifact detection
            dem_arr = ds.read(1, masked=True)
            nodata_frac = np.sum(dem_arr.mask) / dem_arr.size
            result.metrics["nodata_fraction"] = float(nodata_frac)

            if nodata_frac > 0.5:
                result.artifacts_acceptable = False
                result.errors.append(
                    f"Too much nodata: {nodata_frac:.1%}"
                )

            # Check for extreme values
            valid = dem_arr.compressed()
            if len(valid) > 0:
                result.metrics["elevation_min"] = float(np.min(valid))
                result.metrics["elevation_max"] = float(np.max(valid))
                result.metrics["elevation_mean"] = float(np.mean(valid))
                result.metrics["elevation_std"] = float(np.std(valid))

                # Extreme artifact check
                z = (valid - np.mean(valid)) / (np.std(valid) + 1e-6)
                extreme_frac = np.sum(np.abs(z) > 5) / len(valid)
                result.metrics["extreme_artifact_fraction"] = float(
                    extreme_frac
                )

                if extreme_frac > max_artifact_fraction:
                    result.artifacts_acceptable = False
                    result.warnings.append(
                        f"High artifact fraction: {extreme_frac:.1%}"
                    )

            result.metrics["width"] = ds.width
            result.metrics["height"] = ds.height
            result.metrics["size_bytes"] = dem_path.stat().st_size

    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Failed to read DEM: {e}")
        return result

    # Final validity
    result.is_valid = (
        result.crs_valid and
        result.resolution_valid and
        result.coverage_valid and
        result.artifacts_acceptable and
        len(result.errors) == 0
    )

    if result.is_valid:
        logger.info("DEM validation passed: %s", dem_path)
    else:
        logger.error(
            "DEM validation failed: %s - errors: %s",
            dem_path, result.errors
        )

    return result


def register_dem_metadata(
    dem_path: Path,
    source: str,
    validation: DEMValidationResult,
    registry_path: Optional[Path] = None,
) -> Path:
    """Register DEM metadata in YAML registry.

    Args:
        dem_path: Path to DEM file
        source: Data source (e.g., "srtm", "copernicus")
        validation: Validation result
        registry_path: Path to registry YAML file

    Returns:
        Path to registry file
    """
    import yaml
    from src.config.paths import DATA_DIR

    registry_path = registry_path or DATA_DIR / "registry" / "datasets.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing registry
    if registry_path.exists():
        with registry_path.open() as f:
            registry = yaml.safe_load(f) or {}
    else:
        registry = {}

    if "dem" not in registry:
        registry["dem"] = []

    # Create entry
    entry = {
        "path": str(dem_path),
        "source": source,
        "download_date": datetime.utcnow().isoformat(),
        "resolution_m": validation.metrics.get("resolution_m_approx"),
        "crs": validation.metrics.get("crs"),
        "bounds": validation.metrics.get("bounds"),
        "elevation_stats": {
            "min": validation.metrics.get("elevation_min"),
            "max": validation.metrics.get("elevation_max"),
            "mean": validation.metrics.get("elevation_mean"),
        },
        "validation": {
            "is_valid": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
        },
        "limitations": [
            "30m resolution; not suitable for parcel-level analysis",
            "Vertical accuracy ~5-10m; relative accuracy better",
            "Urban areas may have building artifacts",
        ],
    }

    registry["dem"].append(entry)

    with registry_path.open("w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    logger.info("DEM registered in %s", registry_path)
    return registry_path


def automated_dem_pipeline(
    city: str,
    boundary: Optional[Union[dict, Path]] = None,
    target_crs: str = "EPSG:4326",
    target_resolution_m: Optional[float] = None,
    dem_source: str = "auto",
    output_dir: Optional[Path] = None,
) -> dict:
    """Fully automated DEM acquisition, processing, and registration.

    This is the main entry point for Data Automation Prompt-1.
    Zero manual steps required.

    Args:
        city: City name (used for bbox lookup if no boundary provided)
        boundary: Optional GeoJSON boundary dict or path
        target_crs: Target coordinate reference system
        target_resolution_m: Optional target resolution
        dem_source: "srtm", "copernicus", or "auto"
        output_dir: Output directory for processed DEM

    Returns:
        Dict with paths, metadata, and validation results
    """
    from src.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
    from src.ingestion.dem_download import (
        bbox_from_city_name,
        bbox_from_geojson,
        download_dem,
    )

    output_dir = output_dir or PROCESSED_DATA_DIR / "terrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting automated DEM pipeline for city=%s", city)

    # 1. Determine bounding box
    if boundary:
        if isinstance(boundary, Path):
            bbox = bbox_from_geojson(boundary)
        else:
            # Assume it's a GeoJSON dict
            # Write to temp file and read
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".geojson", delete=False
            ) as f:
                json.dump(boundary, f)
                bbox = bbox_from_geojson(Path(f.name))
    else:
        bbox = bbox_from_city_name(city)

    # Add small buffer
    bbox = bbox.buffer(0.01)

    # 2. Download DEM tiles
    download_result = download_dem(
        bbox,
        source=dem_source,
        cache_dir=RAW_DATA_DIR / "terrain" / "cache",
    )

    if not download_result.success:
        raise RuntimeError(
            f"DEM download failed: {download_result.error_message}"
        )

    # 3. Mosaic tiles if multiple
    mosaic_path = output_dir / f"dem_{city}_mosaic.tif"
    mosaic_tiles(download_result.tiles, mosaic_path)

    # 4. Clip to boundary/bbox
    clipped_path = output_dir / f"dem_{city}_clipped.tif"
    clip_to_bbox(
        mosaic_path,
        bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat,
        clipped_path,
    )

    # 5. Reproject if needed
    final_path = output_dir / f"dem_{city}.tif"
    res = None
    if target_resolution_m:
        res = (
            target_resolution_m / 111000,
            target_resolution_m / 111000
        )
    reproject_dem(clipped_path, final_path, target_crs, res)

    # 6. Validate
    validation = validate_dem(
        final_path,
        expected_crs=target_crs,
        expected_resolution_m=target_resolution_m or 30.0,
        expected_bbox=(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
    )

    if not validation.is_valid:
        logger.error("DEM validation failed: %s", validation.errors)
        raise ValueError(
            f"DEM validation failed: {validation.errors}"
        )

    # 7. Register in dataset registry
    registry_path = register_dem_metadata(
        final_path,
        download_result.source,
        validation,
    )

    return {
        "dem_path": final_path,
        "mosaic_path": mosaic_path,
        "clipped_path": clipped_path,
        "download_result": download_result,
        "validation": validation,
        "registry_path": registry_path,
        "bbox": {
            "min_lon": bbox.min_lon,
            "min_lat": bbox.min_lat,
            "max_lon": bbox.max_lon,
            "max_lat": bbox.max_lat,
        },
    }
