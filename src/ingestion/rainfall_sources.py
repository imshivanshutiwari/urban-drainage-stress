"""Rainfall data source registry (Data Automation Prompt-2).

Maintains a registry of supported rainfall data sources with:
- Automation capability (auto/manual)
- Data resolution and format
- Known biases and limitations
- Priority-based source selection

This is NOT mock data. Each source represents real government
hydrometeorological data with documented provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.config.paths import DATA_DIR

logger = logging.getLogger(__name__)


class AutomationLevel(Enum):
    """Automation capability of a data source."""

    FULLY_AUTO = "auto"  # Can be downloaded programmatically
    SEMI_AUTO = "semi"  # Requires one-time manual setup
    MANUAL = "manual"  # Must be downloaded manually each time


class DataFormat(Enum):
    """Supported data formats."""

    CSV = "csv"
    EXCEL = "excel"
    NETCDF = "netcdf"
    JSON = "json"


@dataclass
class RainfallSource:
    """Metadata for a rainfall data source."""

    source_id: str
    name: str
    automation: AutomationLevel
    data_format: DataFormat
    temporal_resolution_minutes: int
    spatial_type: str  # "point" (gauge) or "grid" (radar)
    url_template: Optional[str] = None
    api_endpoint: Optional[str] = None
    requires_auth: bool = False
    coverage_region: str = ""
    known_biases: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    priority: int = 100  # Lower = higher priority
    expected_columns: List[str] = field(default_factory=list)
    unit: str = "mm"
    notes: str = ""


# Default source registry
DEFAULT_SOURCES: Dict[str, RainfallSource] = {
    # Fully automatable sources
    "noaa_isd": RainfallSource(
        source_id="noaa_isd",
        name="NOAA Integrated Surface Database",
        automation=AutomationLevel.FULLY_AUTO,
        data_format=DataFormat.CSV,
        temporal_resolution_minutes=60,
        spatial_type="point",
        url_template=(
            "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/"
        ),
        requires_auth=False,
        coverage_region="Global",
        known_biases=[
            "Hourly resolution may miss short intense bursts",
            "Station density varies by region",
        ],
        limitations=[
            "Data latency of 1-2 days",
            "Some stations have gaps",
        ],
        priority=10,
        expected_columns=["DATE", "HourlyPrecipitation"],
        unit="inch",  # NOAA uses inches
    ),
    "openmeteo": RainfallSource(
        source_id="openmeteo",
        name="Open-Meteo Historical API",
        automation=AutomationLevel.FULLY_AUTO,
        data_format=DataFormat.JSON,
        temporal_resolution_minutes=60,
        spatial_type="grid",
        api_endpoint="https://archive-api.open-meteo.com/v1/archive",
        requires_auth=False,
        coverage_region="Global",
        known_biases=[
            "Model-derived, not direct observation",
            "May underestimate convective peaks",
        ],
        limitations=[
            "Historical only (not real-time)",
            "Resolution ~10km",
        ],
        priority=20,
        expected_columns=["time", "precipitation"],
        unit="mm",
    ),
    "seattle_rain": RainfallSource(
        source_id="seattle_rain",
        name="Seattle Rain Gauge Network",
        automation=AutomationLevel.FULLY_AUTO,
        data_format=DataFormat.CSV,
        temporal_resolution_minutes=15,
        spatial_type="point",
        url_template=(
            "https://data.seattle.gov/resource/ye5r-uc9w.csv"
        ),
        requires_auth=False,
        coverage_region="Seattle, WA",
        known_biases=[
            "Urban heat island may affect readings",
        ],
        limitations=[
            "Seattle area only",
        ],
        priority=5,
        expected_columns=["datetime", "rain_gauge"],
        unit="inch",
    ),
    # Semi-automatable sources (one-time setup)
    "imd_aws": RainfallSource(
        source_id="imd_aws",
        name="IMD Automatic Weather Stations",
        automation=AutomationLevel.MANUAL,
        data_format=DataFormat.CSV,
        temporal_resolution_minutes=15,
        spatial_type="point",
        url_template=None,  # No public API
        requires_auth=True,
        coverage_region="India",
        known_biases=[
            "Station maintenance varies",
            "Urban stations may have obstructions",
        ],
        limitations=[
            "Must request data from IMD portal",
            "Data not publicly automatable",
        ],
        priority=15,
        expected_columns=[
            "timestamp", "station_id", "rainfall_mm"
        ],
        unit="mm",
        notes="Download from https://mausam.imd.gov.in/",
    ),
    "imd_radar": RainfallSource(
        source_id="imd_radar",
        name="IMD Doppler Radar",
        automation=AutomationLevel.MANUAL,
        data_format=DataFormat.NETCDF,
        temporal_resolution_minutes=10,
        spatial_type="grid",
        url_template=None,
        requires_auth=True,
        coverage_region="India (radar coverage)",
        known_biases=[
            "Beam blockage in hilly terrain",
            "Z-R relationship uncertainty",
            "Ground clutter near radar",
        ],
        limitations=[
            "Radar data requires manual download",
            "NetCDF format needs preprocessing",
        ],
        priority=25,
        expected_columns=[],  # NetCDF variables
        unit="mm/hr",
    ),
}


class RainfallSourceRegistry:
    """Registry for managing rainfall data sources.

    Supports:
    - Listing available sources by automation level
    - Priority-based source selection
    - Checking data availability
    - Recording which sources were used
    """

    def __init__(
        self,
        sources: Optional[Dict[str, RainfallSource]] = None,
        registry_path: Optional[Path] = None,
    ):
        """Initialize registry with default or custom sources.

        Args:
            sources: Custom source definitions (uses defaults if None)
            registry_path: Path to persist registry state
        """
        self.sources = sources or DEFAULT_SOURCES.copy()
        self.registry_path = (
            registry_path or DATA_DIR / "registry" / "rainfall_sources.yaml"
        )
        self._used_sources: List[str] = []

    def list_sources(
        self,
        automation: Optional[AutomationLevel] = None,
        region: Optional[str] = None,
    ) -> List[RainfallSource]:
        """List sources, optionally filtered by automation level or region.

        Args:
            automation: Filter by automation capability
            region: Filter by coverage region (substring match)

        Returns:
            List of matching sources, sorted by priority
        """
        result = []
        for src in self.sources.values():
            if automation and src.automation != automation:
                continue
            if region and region.lower() not in src.coverage_region.lower():
                continue
            result.append(src)
        return sorted(result, key=lambda s: s.priority)

    def get_source(self, source_id: str) -> Optional[RainfallSource]:
        """Get a specific source by ID."""
        return self.sources.get(source_id)

    def get_auto_sources(self) -> List[RainfallSource]:
        """Get all fully automatable sources, sorted by priority."""
        return self.list_sources(automation=AutomationLevel.FULLY_AUTO)

    def get_manual_sources(self) -> List[RainfallSource]:
        """Get sources requiring manual intervention."""
        manual_levels = (AutomationLevel.MANUAL, AutomationLevel.SEMI_AUTO)
        return [
            s for s in self.sources.values()
            if s.automation in manual_levels
        ]

    def register_source(self, source: RainfallSource) -> None:
        """Add or update a source in the registry."""
        self.sources[source.source_id] = source
        logger.info("Registered rainfall source: %s", source.source_id)

    def record_usage(self, source_id: str) -> None:
        """Record that a source was used in data acquisition."""
        if source_id not in self._used_sources:
            self._used_sources.append(source_id)

    def get_used_sources(self) -> List[str]:
        """Get list of sources used in this session."""
        return self._used_sources.copy()

    def save_registry(self) -> Path:
        """Persist registry to YAML file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "sources": {
                sid: {
                    "name": s.name,
                    "automation": s.automation.value,
                    "format": s.data_format.value,
                    "resolution_minutes": s.temporal_resolution_minutes,
                    "spatial_type": s.spatial_type,
                    "url": s.url_template or s.api_endpoint,
                    "region": s.coverage_region,
                    "priority": s.priority,
                    "unit": s.unit,
                    "biases": s.known_biases,
                    "limitations": s.limitations,
                }
                for sid, s in self.sources.items()
            },
            "used_sources": self._used_sources,
        }

        with self.registry_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Rainfall source registry saved: %s", self.registry_path)
        return self.registry_path

    def get_expected_schema(self, source_id: str) -> Dict:
        """Get expected schema for a source (for manual fallback guidance)."""
        src = self.get_source(source_id)
        if not src:
            return {}

        return {
            "source_id": source_id,
            "format": src.data_format.value,
            "expected_columns": src.expected_columns,
            "unit": src.unit,
            "temporal_resolution_min": src.temporal_resolution_minutes,
            "notes": src.notes,
        }
