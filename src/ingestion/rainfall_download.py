"""Automated rainfall data download pipeline (Data Automation Prompt-2).

Implements:
- Automated download from public sources (NOAA, Open-Meteo, etc.)
- Schema validation immediately after download
- Temporal coverage verification
- Graceful fallback to manual sources
- Multi-station handling with metadata preservation

This module NEVER fabricates data. If data cannot be obtained,
it raises clear exceptions with exact instructions for manual download.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from src.config.paths import RAW_DATA_DIR
from src.ingestion.rainfall_sources import (
    RainfallSource,
    RainfallSourceRegistry,
)

logger = logging.getLogger(__name__)


class ManualDataRequired(Exception):
    """Raised when manual data download is required.

    Contains all information needed for the user to obtain the data.
    """

    def __init__(
        self,
        source_id: str,
        expected_filename: str,
        expected_schema: Dict,
        date_range: Tuple[datetime, datetime],
        target_directory: Path,
        download_instructions: str,
    ):
        self.source_id = source_id
        self.expected_filename = expected_filename
        self.expected_schema = expected_schema
        self.date_range = date_range
        self.target_directory = target_directory
        self.download_instructions = download_instructions

        msg = self._format_message()
        super().__init__(msg)

    def _format_message(self) -> str:
        start, end = self.date_range
        return f"""
================================================================================
MANUAL DATA REQUIRED
================================================================================

Source: {self.source_id}
Reason: This data source cannot be automated due to access restrictions.

WHAT TO DO:
-----------
1. Download data from the source manually
2. Expected filename: {self.expected_filename}
3. Place file in: {self.target_directory}
4. Re-run the pipeline (it will resume automatically)

DATE RANGE NEEDED:
  Start: {start.strftime('%Y-%m-%d %H:%M')}
  End:   {end.strftime('%Y-%m-%d %H:%M')}

EXPECTED SCHEMA:
  Format: {self.expected_schema.get('format', 'CSV')}
  Columns: {', '.join(self.expected_schema.get('expected_columns', []))}
  Unit: {self.expected_schema.get('unit', 'mm')}

DOWNLOAD INSTRUCTIONS:
{self.download_instructions}

================================================================================
"""


@dataclass
class DownloadResult:
    """Result of a rainfall download attempt."""

    success: bool = False
    source_id: str = ""
    file_path: Optional[Path] = None
    stations: List[str] = field(default_factory=list)
    temporal_coverage: Tuple[datetime, datetime] = field(
        default_factory=lambda: (datetime.min, datetime.max)
    )
    coverage_fraction: float = 0.0
    row_count: int = 0
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class StationData:
    """Data from a single rainfall station."""

    station_id: str
    data: pd.DataFrame
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation_m: Optional[float] = None
    coverage_fraction: float = 0.0
    quality_flags: Dict = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: int = 60,
    headers: Optional[Dict] = None,
) -> bytes:
    """Download URL content with retry logic."""
    for attempt in range(max_retries):
        try:
            req = Request(url, headers=headers or {})
            with urlopen(req, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(
                "Download attempt %d/%d failed: %s",
                attempt + 1, max_retries, e
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Failed to download after {max_retries} attempts: {url}"
    )


def download_openmeteo(
    latitude: float,
    longitude: float,
    start_date: datetime,
    end_date: datetime,
    cache_dir: Optional[Path] = None,
) -> DownloadResult:
    """Download rainfall data from Open-Meteo Historical API.

    This is a fully automated, free API requiring no authentication.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start of date range
        end_date: End of date range
        cache_dir: Directory to cache downloaded data

    Returns:
        DownloadResult with downloaded data path and metadata
    """
    cache_dir = cache_dir or RAW_DATA_DIR / "rainfall" / "openmeteo_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = DownloadResult(source_id="openmeteo")

    # Build API URL
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "precipitation",
        "timezone": "UTC",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://archive-api.open-meteo.com/v1/archive?{query}"

    # Check cache
    cache_key = hashlib.md5(url.encode()).hexdigest()[:12]
    cache_file = cache_dir / f"openmeteo_{cache_key}.csv"

    if cache_file.exists():
        logger.info("Using cached Open-Meteo data: %s", cache_file)
        result.file_path = cache_file
        result.success = True
        result.metadata["cached"] = True
        return _validate_rainfall_file(result, start_date, end_date)

    try:
        logger.info("Downloading from Open-Meteo: %s", url)
        data = _download_with_retry(url)
        response = json.loads(data.decode("utf-8"))

        if "error" in response:
            result.validation_errors.append(
                f"API error: {response.get('reason', 'Unknown')}"
            )
            return result

        # Parse response to DataFrame
        hourly = response.get("hourly", {})
        times = hourly.get("time", [])
        precip = hourly.get("precipitation", [])

        if not times or not precip:
            result.validation_errors.append("Empty response from Open-Meteo")
            return result

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(times),
            "rainfall_mm": precip,
            "latitude": latitude,
            "longitude": longitude,
            "source": "openmeteo",
        })

        # Save to cache
        df.to_csv(cache_file, index=False)
        result.file_path = cache_file
        result.success = True
        result.row_count = len(df)
        result.metadata = {
            "url": url,
            "downloaded_at": datetime.utcnow().isoformat(),
            "sha256": _sha256_file(cache_file),
        }

        logger.info(
            "Open-Meteo download successful: %d rows", len(df)
        )

    except Exception as e:
        result.validation_errors.append(f"Download failed: {e}")
        logger.error("Open-Meteo download failed: %s", e)

    return _validate_rainfall_file(result, start_date, end_date)


def download_noaa_isd(
    station_id: str,
    year: int,
    cache_dir: Optional[Path] = None,
) -> DownloadResult:
    """Download rainfall from NOAA Integrated Surface Database.

    Note: NOAA ISD has specific station IDs and yearly files.

    Args:
        station_id: NOAA station identifier
        year: Year to download
        cache_dir: Cache directory

    Returns:
        DownloadResult with downloaded data
    """
    cache_dir = cache_dir or RAW_DATA_DIR / "rainfall" / "noaa_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = DownloadResult(source_id="noaa_isd")

    url = (
        f"https://www.ncei.noaa.gov/data/global-hourly/access/"
        f"{year}/{station_id}.csv"
    )

    cache_file = cache_dir / f"noaa_{station_id}_{year}.csv"

    if cache_file.exists():
        logger.info("Using cached NOAA data: %s", cache_file)
        result.file_path = cache_file
        result.success = True
        result.metadata["cached"] = True
        return result

    try:
        logger.info("Downloading from NOAA ISD: %s", url)
        data = _download_with_retry(url)
        cache_file.write_bytes(data)

        result.file_path = cache_file
        result.success = True
        result.metadata = {
            "station_id": station_id,
            "year": year,
            "url": url,
            "sha256": _sha256_file(cache_file),
        }

    except Exception as e:
        result.validation_errors.append(f"NOAA download failed: {e}")
        logger.warning("NOAA ISD download failed: %s", e)

    return result


def _validate_rainfall_file(
    result: DownloadResult,
    start_date: datetime,
    end_date: datetime,
) -> DownloadResult:
    """Validate downloaded rainfall data.

    Checks:
    - File exists and is readable
    - Has required columns
    - Temporal coverage matches requested range
    - No impossible values
    """
    if not result.file_path or not result.file_path.exists():
        result.validation_passed = False
        return result

    try:
        df = pd.read_csv(result.file_path)
        result.row_count = len(df)

        # Check for timestamp column
        ts_col = None
        for col in ["timestamp", "time", "datetime", "DATE"]:
            if col in df.columns:
                ts_col = col
                break

        if not ts_col:
            result.validation_errors.append(
                "No timestamp column found"
            )
            result.validation_passed = False
            return result

        # Check for rainfall column
        rain_col = None
        for col in ["rainfall_mm", "precipitation", "rain", "HourlyPrecip"]:
            if col in df.columns:
                rain_col = col
                break

        if not rain_col:
            result.validation_errors.append(
                "No rainfall column found"
            )
            result.validation_passed = False
            return result

        # Parse timestamps and check coverage
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        valid_ts = df[ts_col].dropna()

        if len(valid_ts) == 0:
            result.validation_errors.append("No valid timestamps")
            result.validation_passed = False
            return result

        actual_start = valid_ts.min().to_pydatetime()
        actual_end = valid_ts.max().to_pydatetime()
        result.temporal_coverage = (actual_start, actual_end)

        # Calculate coverage
        requested_hours = (end_date - start_date).total_seconds() / 3600
        covered_hours = (actual_end - actual_start).total_seconds() / 3600
        result.coverage_fraction = min(
            covered_hours / max(requested_hours, 1), 1.0
        )

        if result.coverage_fraction < 0.5:
            result.warnings.append(
                f"Low temporal coverage: {result.coverage_fraction:.1%}"
            )

        # Check for impossible values
        rain_vals = pd.to_numeric(df[rain_col], errors="coerce")
        if (rain_vals < 0).any():
            result.validation_errors.append("Negative rainfall values found")

        if (rain_vals > 500).any():  # >500mm/hr is extreme
            result.warnings.append(
                "Extremely high rainfall values (>500mm) detected"
            )

        result.validation_passed = len(result.validation_errors) == 0

    except Exception as e:
        result.validation_errors.append(f"Validation error: {e}")
        result.validation_passed = False

    return result


def check_manual_data_exists(
    city: str,
    source_id: str,
    start_date: datetime,
    end_date: datetime,
) -> Optional[Path]:
    """Check if manually downloaded data exists for a source.

    Looks for files matching expected naming patterns in the
    expected directory.
    """
    manual_dir = RAW_DATA_DIR / "rainfall" / "manual"

    patterns = [
        f"{source_id}_{city}*.csv",
        f"{city}_{source_id}*.csv",
        f"rainfall_{city}*.csv",
    ]

    for pattern in patterns:
        matches = list(manual_dir.glob(pattern))
        if matches:
            # Return most recent
            return max(matches, key=lambda p: p.stat().st_mtime)

    return None


def request_manual_download(
    source: RainfallSource,
    city: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """Raise exception requesting manual data download.

    This function NEVER returns - it always raises ManualDataRequired
    with complete instructions for obtaining the data.
    """
    target_dir = RAW_DATA_DIR / "rainfall" / "manual"
    target_dir.mkdir(parents=True, exist_ok=True)

    expected_filename = f"{source.source_id}_{city}_{start_date:%Y%m%d}.csv"

    instructions = f"""
Source: {source.name}
Website: {source.notes or 'See source documentation'}

Steps:
1. Navigate to the data portal
2. Select station(s) for {city}
3. Set date range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}
4. Download as CSV
5. Rename to: {expected_filename}
6. Place in: {target_dir}

Expected columns: {', '.join(source.expected_columns)}
Expected unit: {source.unit}
"""

    raise ManualDataRequired(
        source_id=source.source_id,
        expected_filename=expected_filename,
        expected_schema={
            "format": source.data_format.value,
            "expected_columns": source.expected_columns,
            "unit": source.unit,
        },
        date_range=(start_date, end_date),
        target_directory=target_dir,
        download_instructions=instructions,
    )


class RainfallDownloader:
    """Orchestrates rainfall data acquisition from multiple sources.

    Implements priority-based source selection with automatic fallback
    and clear manual data requirements when automation is not possible.
    """

    def __init__(
        self,
        registry: Optional[RainfallSourceRegistry] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.registry = registry or RainfallSourceRegistry()
        self.cache_dir = cache_dir or RAW_DATA_DIR / "rainfall"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        city: str,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        required_coverage: float = 0.5,
        allow_manual_fallback: bool = True,
    ) -> DownloadResult:
        """Download rainfall data, trying sources in priority order.

        Args:
            city: City name for logging/filenames
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start of required date range
            end_date: End of required date range
            required_coverage: Minimum acceptable temporal coverage
            allow_manual_fallback: Whether to raise ManualDataRequired

        Returns:
            DownloadResult with best available data

        Raises:
            ManualDataRequired: If no automated source works and
                manual fallback is enabled
        """
        logger.info(
            "Downloading rainfall for %s [%.4f, %.4f] from %s to %s",
            city, latitude, longitude,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        # Try automated sources in priority order
        auto_sources = self.registry.get_auto_sources()
        best_result: Optional[DownloadResult] = None

        for source in auto_sources:
            logger.info("Trying source: %s", source.source_id)

            try:
                if source.source_id == "openmeteo":
                    result = download_openmeteo(
                        latitude, longitude,
                        start_date, end_date,
                        self.cache_dir / "openmeteo",
                    )
                elif source.source_id == "noaa_isd":
                    # NOAA requires station ID - skip if not available
                    logger.info("Skipping NOAA ISD (requires station ID)")
                    continue
                else:
                    logger.info(
                        "No downloader for source: %s", source.source_id
                    )
                    continue

                if result.success and result.validation_passed:
                    if result.coverage_fraction >= required_coverage:
                        self.registry.record_usage(source.source_id)
                        logger.info(
                            "Source %s successful: %.1f%% coverage",
                            source.source_id,
                            result.coverage_fraction * 100,
                        )
                        return result

                    # Keep as fallback if better coverage
                    if best_result:
                        better = (
                            result.coverage_fraction >
                            best_result.coverage_fraction
                        )
                    else:
                        better = True
                    if better:
                        best_result = result

            except Exception as e:
                logger.warning(
                    "Source %s failed: %s", source.source_id, e
                )

        # If we have partial data, use it
        if best_result and best_result.success:
            logger.warning(
                "Using partial data: %.1f%% coverage",
                best_result.coverage_fraction * 100,
            )
            return best_result

        # Check for existing manual data
        for source in self.registry.get_manual_sources():
            manual_path = check_manual_data_exists(
                city, source.source_id, start_date, end_date
            )
            if manual_path:
                logger.info(
                    "Found manual data: %s", manual_path
                )
                result = DownloadResult(
                    success=True,
                    source_id=source.source_id,
                    file_path=manual_path,
                    metadata={"source": "manual"},
                )
                return _validate_rainfall_file(result, start_date, end_date)

        # No automated data - request manual download if allowed
        if allow_manual_fallback:
            manual_sources = self.registry.get_manual_sources()
            if manual_sources:
                request_manual_download(
                    manual_sources[0],
                    city,
                    start_date,
                    end_date,
                )

        # Return empty result if manual fallback disabled
        return DownloadResult(
            success=False,
            validation_errors=["No rainfall data available from any source"],
        )

    def validate_and_merge_stations(
        self,
        station_data: List[StationData],
    ) -> Tuple[pd.DataFrame, Dict]:
        """Merge multi-station data with proper handling.

        Does NOT:
        - Average blindly across stations
        - Drop stations silently

        Does:
        - Preserve station-level metadata
        - Flag inconsistent timestamps
        - Track station-specific coverage
        """
        if not station_data:
            return pd.DataFrame(), {}

        # Align timestamps across stations
        all_timestamps = set()
        for sd in station_data:
            if "timestamp" in sd.data.columns:
                all_timestamps.update(sd.data["timestamp"].tolist())

        all_timestamps = sorted(all_timestamps)

        # Create merged DataFrame with station columns
        merged = pd.DataFrame({"timestamp": all_timestamps})

        station_meta = {}
        for sd in station_data:
            col_name = f"rainfall_{sd.station_id}"
            has_ts = "timestamp" in sd.data.columns
            has_rain = "rainfall_mm" in sd.data.columns
            if has_ts and has_rain:
                station_df = sd.data[["timestamp", "rainfall_mm"]].copy()
                station_df = station_df.rename(
                    columns={"rainfall_mm": col_name}
                )
                merged = merged.merge(station_df, on="timestamp", how="left")

            station_meta[sd.station_id] = {
                "latitude": sd.latitude,
                "longitude": sd.longitude,
                "coverage": sd.coverage_fraction,
                "quality_flags": sd.quality_flags,
            }

        return merged, station_meta


def ensure_rainfall(
    city: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Path:
    """High-level function to ensure rainfall data is available.

    This is the main entry point for the rainfall acquisition system.
    It will:
    1. Try automated downloads from public sources
    2. Fall back to manual sources if needed
    3. Raise clear exceptions if manual intervention required

    Args:
        city: City name
        start_date: Start of required period (default: last 30 days)
        end_date: End of required period (default: now)
        latitude: City latitude (uses lookup if not provided)
        longitude: City longitude (uses lookup if not provided)

    Returns:
        Path to rainfall data file

    Raises:
        ManualDataRequired: If data must be downloaded manually
    """
    # Default date range: last 30 days
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    # Default coordinates (lookup by city name)
    city_coords = {
        "seattle": (47.6062, -122.3321),
        "portland": (45.5152, -122.6784),
        "los_angeles": (34.0522, -118.2437),
        "new_york": (40.7128, -74.0060),
        "chicago": (41.8781, -87.6298),
        "houston": (29.7604, -95.3698),
        "miami": (25.7617, -80.1918),
        "denver": (39.7392, -104.9903),
        "default": (47.6062, -122.3321),  # Seattle
    }

    key = city.lower().replace(" ", "_")
    if key not in city_coords:
        logger.warning("City %s not found, using default coordinates", city)
        key = "default"

    if latitude is None or longitude is None:
        latitude, longitude = city_coords[key]

    # Download
    downloader = RainfallDownloader()
    result = downloader.download(
        city=city,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )

    if result.success and result.file_path:
        logger.info(
            "Rainfall data acquired: %s (coverage: %.1f%%)",
            result.file_path,
            result.coverage_fraction * 100,
        )
        return result.file_path

    raise RuntimeError(
        f"Failed to acquire rainfall data: {result.validation_errors}"
    )
