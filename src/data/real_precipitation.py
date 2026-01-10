"""Download REAL precipitation data from Open-Meteo API.

This module fetches actual hourly precipitation data for a given location,
NOT temperature data. Open-Meteo is free and doesn't require an API key.

Reference: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd

logger = logging.getLogger(__name__)


# Seattle coordinates
DEFAULT_LAT = 47.6062
DEFAULT_LON = -122.3321

# Open-Meteo API base URL
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_real_precipitation(
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch REAL hourly precipitation data from Open-Meteo.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with columns: timestamp, latitude, longitude, 
        precipitation_mm, rain_mm, snowfall_mm
    """
    logger.info(
        "Fetching real precipitation data: lat=%.4f, lon=%.4f, %s to %s",
        latitude, longitude, start_date, end_date
    )
    
    # Build API URL
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,rain,snowfall,weathercode",
        "timezone": "UTC",
    }
    
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{OPEN_METEO_URL}?{query_string}"
    
    logger.info("API URL: %s", url)
    
    try:
        with urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
    except URLError as e:
        logger.error("Failed to fetch precipitation data: %s", e)
        raise RuntimeError(f"Failed to fetch precipitation data: {e}") from e
    
    # Parse response
    if "hourly" not in data:
        raise ValueError("No hourly data in API response")
    
    hourly = data["hourly"]
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "latitude": latitude,
        "longitude": longitude,
        "precipitation_mm": hourly.get("precipitation", [0] * len(hourly["time"])),
        "rain_mm": hourly.get("rain", [0] * len(hourly["time"])),
        "snowfall_mm": hourly.get("snowfall", [0] * len(hourly["time"])),
        "weather_code": hourly.get("weathercode", [0] * len(hourly["time"])),
    })
    
    # Add derived columns for compatibility
    df["rainfall_mm"] = df["rain_mm"]  # Alias for ingestion
    
    # Log statistics
    total_precip = df["precipitation_mm"].sum()
    rainy_hours = (df["precipitation_mm"] > 0).sum()
    max_hourly = df["precipitation_mm"].max()
    
    logger.info(
        "Fetched %d hours of data: total_precip=%.1f mm, "
        "rainy_hours=%d, max_hourly=%.1f mm",
        len(df), total_precip, rainy_hours, max_hourly
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved precipitation data to %s", output_path)
    
    return df


def fetch_multi_station_precipitation(
    stations: list[dict],
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch precipitation for multiple stations.
    
    Args:
        stations: List of dicts with 'name', 'lat', 'lon' keys
        start_date: Start date
        end_date: End date
        output_path: Optional path to save combined CSV
        
    Returns:
        Combined DataFrame for all stations
    """
    all_data = []
    
    for station in stations:
        logger.info("Fetching data for station: %s", station.get("name", "Unknown"))
        try:
            df = fetch_real_precipitation(
                latitude=station["lat"],
                longitude=station["lon"],
                start_date=start_date,
                end_date=end_date,
            )
            df["station_name"] = station.get("name", f"Station_{len(all_data)}")
            all_data.append(df)
        except Exception as e:
            logger.warning("Failed to fetch station %s: %s", station.get("name"), e)
    
    if not all_data:
        raise RuntimeError("Failed to fetch data for any station")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        logger.info("Saved combined precipitation data to %s", output_path)
    
    return combined


def ensure_real_rainfall(city: str, force_download: bool = False) -> Path:
    """Ensure real rainfall data exists for a city.
    
    This replaces the fake temperature data with actual precipitation.
    """
    from src.config.paths import RAW_DATA_DIR
    
    output_path = RAW_DATA_DIR / "rainfall" / f"rainfall_{city}_real.csv"
    
    if output_path.exists() and not force_download:
        logger.info("Real rainfall data already exists: %s", output_path)
        return output_path
    
    # City coordinates
    city_coords = {
        "seattle": (47.6062, -122.3321),
        "portland": (45.5152, -122.6784),
        "san_francisco": (37.7749, -122.4194),
        "new_york": (40.7128, -74.0060),
        "chicago": (41.8781, -87.6298),
        "houston": (29.7604, -95.3698),
        "miami": (25.7617, -80.1918),
    }
    
    lat, lon = city_coords.get(city.lower(), (47.6062, -122.3321))
    
    # Fetch last year of data
    end_date = datetime.now() - timedelta(days=7)  # API has ~1 week lag
    start_date = end_date - timedelta(days=365)
    
    df = fetch_real_precipitation(
        latitude=lat,
        longitude=lon,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        output_path=output_path,
    )
    
    return output_path


if __name__ == "__main__":
    # Test: fetch Seattle precipitation
    logging.basicConfig(level=logging.INFO)
    
    df = fetch_real_precipitation(
        start_date="2023-06-01",
        end_date="2023-06-30",
    )
    print(df.head(20))
    print(f"\nTotal precipitation: {df['precipitation_mm'].sum():.1f} mm")
    print(f"Max hourly: {df['precipitation_mm'].max():.1f} mm")
