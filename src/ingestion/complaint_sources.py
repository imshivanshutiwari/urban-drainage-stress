"""Complaint data source registry (Data Automation Prompt-3).

Maintains a registry of flood/waterlogging complaint data sources with:
- Automation capability (auto/manual)
- Spatial resolution (point/ward/area)
- Known reporting biases (critical for scientific honesty)
- City-specific source mappings

THIS DATA IS MESSY, BIASED, AND INCONSISTENT BY NATURE.
The registry documents these limitations explicitly.
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

    FULLY_AUTO = "auto"
    SEMI_AUTO = "semi"
    MANUAL = "manual"


class SpatialResolution(Enum):
    """Spatial granularity of complaint data."""

    POINT = "point"  # Exact coordinates available
    WARD = "ward"  # Ward/zone level only
    AREA = "area"  # Free-text area descriptions
    MIXED = "mixed"  # Combination of above


@dataclass
class ReportingBias:
    """Documents known biases in complaint reporting.

    These biases are CRITICAL for downstream inference.
    Ignoring them leads to scientifically dishonest results.
    """

    name: str
    description: str
    impact: str  # "underreporting", "overreporting", "spatial_bias", etc.
    severity: str  # "low", "medium", "high"


@dataclass
class ComplaintSource:
    """Metadata for a complaint data source."""

    source_id: str
    name: str
    automation: AutomationLevel
    spatial_resolution: SpatialResolution
    city: str
    url_template: Optional[str] = None
    api_endpoint: Optional[str] = None
    requires_auth: bool = False
    data_format: str = "csv"
    expected_columns: List[str] = field(default_factory=list)
    known_biases: List[ReportingBias] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    priority: int = 100
    notes: str = ""
    typical_delay_hours: float = 24.0  # Reporting delay


# Pre-defined reporting biases (reusable)
BIAS_UNDERREPORTING_POOR = ReportingBias(
    name="socioeconomic_underreporting",
    description="Low-income areas underreport due to lack of awareness/access",
    impact="underreporting",
    severity="high",
)

BIAS_OVERREPORTING_VOCAL = ReportingBias(
    name="vocal_minority",
    description="Politically active areas generate more complaints",
    impact="overreporting",
    severity="medium",
)

BIAS_SPATIAL_WARD = ReportingBias(
    name="ward_aggregation",
    description="Point locations unknown; ward centroids used",
    impact="spatial_bias",
    severity="medium",
)

BIAS_TEMPORAL_DELAY = ReportingBias(
    name="reporting_delay",
    description="Complaints filed hours/days after actual event",
    impact="temporal_bias",
    severity="medium",
)

BIAS_SELECTION = ReportingBias(
    name="selection_bias",
    description="Only severe cases reported; minor flooding ignored",
    impact="underreporting",
    severity="high",
)

BIAS_DUPLICATE = ReportingBias(
    name="duplicate_reporting",
    description="Same incident reported multiple times by neighbors",
    impact="overreporting",
    severity="medium",
)


# Default source registry
DEFAULT_SOURCES: Dict[str, ComplaintSource] = {
    # US Open Data Sources (Automated)
    "nyc_311": ComplaintSource(
        source_id="nyc_311",
        name="NYC 311 Service Requests",
        automation=AutomationLevel.FULLY_AUTO,
        spatial_resolution=SpatialResolution.POINT,
        city="new_york",
        api_endpoint=(
            "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
        ),
        requires_auth=False,
        data_format="csv",
        expected_columns=[
            "created_date", "complaint_type", "latitude", "longitude"
        ],
        known_biases=[
            BIAS_OVERREPORTING_VOCAL,
            BIAS_TEMPORAL_DELAY,
            BIAS_DUPLICATE,
        ],
        limitations=[
            "Filtering needed for water/flooding complaints only",
            "High volume requires date filtering",
        ],
        priority=5,
    ),
    "seattle_csr": ComplaintSource(
        source_id="seattle_csr",
        name="Seattle Customer Service Requests",
        automation=AutomationLevel.FULLY_AUTO,
        spatial_resolution=SpatialResolution.POINT,
        city="seattle",
        api_endpoint=(
            "https://data.seattle.gov/resource/5ngg-rpne.csv"
        ),
        requires_auth=False,
        data_format="csv",
        expected_columns=[
            "created_date", "request_type", "latitude", "longitude"
        ],
        known_biases=[
            BIAS_OVERREPORTING_VOCAL,
            BIAS_UNDERREPORTING_POOR,
        ],
        limitations=[
            "Must filter for drainage/flooding types",
        ],
        priority=5,
    ),
    "chicago_311": ComplaintSource(
        source_id="chicago_311",
        name="Chicago 311 Service Requests",
        automation=AutomationLevel.FULLY_AUTO,
        spatial_resolution=SpatialResolution.POINT,
        city="chicago",
        api_endpoint=(
            "https://data.cityofchicago.org/resource/v6vf-nfxy.csv"
        ),
        requires_auth=False,
        data_format="csv",
        expected_columns=[
            "created_date", "sr_type", "latitude", "longitude"
        ],
        known_biases=[
            BIAS_OVERREPORTING_VOCAL,
            BIAS_SELECTION,
        ],
        priority=5,
    ),
    # Indian Municipal Data (Manual)
    "bmc_mumbai": ComplaintSource(
        source_id="bmc_mumbai",
        name="BMC Mumbai Waterlogging Complaints",
        automation=AutomationLevel.MANUAL,
        spatial_resolution=SpatialResolution.WARD,
        city="mumbai",
        url_template=None,
        requires_auth=True,
        data_format="excel",
        expected_columns=[
            "date", "ward", "complaint_type", "status"
        ],
        known_biases=[
            BIAS_UNDERREPORTING_POOR,
            BIAS_SPATIAL_WARD,
            BIAS_TEMPORAL_DELAY,
            BIAS_SELECTION,
        ],
        limitations=[
            "Ward-level only; no point coordinates",
            "Manual download from portal required",
            "Data quality varies by ward",
        ],
        priority=10,
        notes="Download from https://portal.mcgm.gov.in/",
        typical_delay_hours=48.0,
    ),
    "bbmp_bangalore": ComplaintSource(
        source_id="bbmp_bangalore",
        name="BBMP Bangalore Sahaaya Complaints",
        automation=AutomationLevel.MANUAL,
        spatial_resolution=SpatialResolution.WARD,
        city="bangalore",
        requires_auth=True,
        data_format="csv",
        expected_columns=[
            "complaint_date", "ward_name", "category", "description"
        ],
        known_biases=[
            BIAS_UNDERREPORTING_POOR,
            BIAS_SPATIAL_WARD,
            BIAS_SELECTION,
        ],
        limitations=[
            "Ward-level aggregation only",
            "Requires RTI or portal access",
        ],
        priority=10,
        notes="Download from BBMP Sahaaya portal or via RTI",
    ),
    # Generic/Synthetic for testing
    "synthetic": ComplaintSource(
        source_id="synthetic",
        name="Synthetic Complaint Data",
        automation=AutomationLevel.FULLY_AUTO,
        spatial_resolution=SpatialResolution.POINT,
        city="default",
        expected_columns=[
            "timestamp", "latitude", "longitude", "complaint_type"
        ],
        known_biases=[],
        limitations=["Synthetic data for testing only"],
        priority=100,
    ),
}


class ComplaintSourceRegistry:
    """Registry for managing complaint data sources.

    Key features:
    - City-specific source lookup
    - Bias documentation for each source
    - Priority-based source selection
    - Tracks which sources were used
    """

    def __init__(
        self,
        sources: Optional[Dict[str, ComplaintSource]] = None,
        registry_path: Optional[Path] = None,
    ):
        self.sources = sources or DEFAULT_SOURCES.copy()
        self.registry_path = (
            registry_path or DATA_DIR / "registry" / "complaint_sources.yaml"
        )
        self._used_sources: List[str] = []

    def list_sources(
        self,
        city: Optional[str] = None,
        automation: Optional[AutomationLevel] = None,
    ) -> List[ComplaintSource]:
        """List sources filtered by city and/or automation level."""
        result = []
        for src in self.sources.values():
            if city and src.city.lower() != city.lower():
                continue
            if automation and src.automation != automation:
                continue
            result.append(src)
        return sorted(result, key=lambda s: s.priority)

    def get_source(self, source_id: str) -> Optional[ComplaintSource]:
        """Get a specific source by ID."""
        return self.sources.get(source_id)

    def get_sources_for_city(self, city: str) -> List[ComplaintSource]:
        """Get all sources for a specific city, sorted by priority."""
        return self.list_sources(city=city)

    def get_auto_sources(
        self, city: Optional[str] = None
    ) -> List[ComplaintSource]:
        """Get fully automatable sources."""
        sources = self.list_sources(
            city=city, automation=AutomationLevel.FULLY_AUTO
        )
        return sources

    def get_manual_sources(
        self, city: Optional[str] = None
    ) -> List[ComplaintSource]:
        """Get sources requiring manual intervention."""
        result = []
        for src in self.sources.values():
            if city and src.city.lower() != city.lower():
                continue
            if src.automation in (
                AutomationLevel.MANUAL, AutomationLevel.SEMI_AUTO
            ):
                result.append(src)
        return sorted(result, key=lambda s: s.priority)

    def register_source(self, source: ComplaintSource) -> None:
        """Add or update a source in the registry."""
        self.sources[source.source_id] = source
        logger.info("Registered complaint source: %s", source.source_id)

    def record_usage(self, source_id: str) -> None:
        """Record that a source was used."""
        if source_id not in self._used_sources:
            self._used_sources.append(source_id)

    def get_biases_for_source(self, source_id: str) -> List[ReportingBias]:
        """Get documented biases for a source."""
        src = self.get_source(source_id)
        return src.known_biases if src else []

    def get_bias_summary(self, source_id: str) -> Dict:
        """Get a summary of biases for downstream uncertainty handling."""
        src = self.get_source(source_id)
        if not src:
            return {}

        return {
            "source_id": source_id,
            "spatial_resolution": src.spatial_resolution.value,
            "typical_delay_hours": src.typical_delay_hours,
            "biases": [
                {
                    "name": b.name,
                    "impact": b.impact,
                    "severity": b.severity,
                }
                for b in src.known_biases
            ],
            "underreporting_risk": any(
                b.impact == "underreporting" for b in src.known_biases
            ),
            "spatial_uncertainty": src.spatial_resolution != SpatialResolution.POINT,
        }

    def save_registry(self) -> Path:
        """Persist registry to YAML file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "sources": {
                sid: {
                    "name": s.name,
                    "automation": s.automation.value,
                    "spatial_resolution": s.spatial_resolution.value,
                    "city": s.city,
                    "url": s.api_endpoint or s.url_template,
                    "format": s.data_format,
                    "priority": s.priority,
                    "expected_columns": s.expected_columns,
                    "biases": [
                        {"name": b.name, "impact": b.impact, "severity": b.severity}
                        for b in s.known_biases
                    ],
                    "limitations": s.limitations,
                    "typical_delay_hours": s.typical_delay_hours,
                }
                for sid, s in self.sources.items()
            },
            "used_sources": self._used_sources,
        }

        with self.registry_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Complaint source registry saved: %s", self.registry_path)
        return self.registry_path

    def get_expected_schema(self, source_id: str) -> Dict:
        """Get expected schema for manual fallback guidance."""
        src = self.get_source(source_id)
        if not src:
            return {}

        return {
            "source_id": source_id,
            "format": src.data_format,
            "expected_columns": src.expected_columns,
            "spatial_resolution": src.spatial_resolution.value,
            "notes": src.notes,
        }
