"""Central Data Registry (Data Automation Prompt-4).

The system's backbone for tracking all datasets:
- Records provenance, quality, and validation status
- Persists state to disk for durability
- Provides queryable interface for pipeline decisions

WITHOUT THIS MODULE, THE SYSTEM IS "JUST CODE FILES".
WITH IT, THE SYSTEM KNOWS WHAT DATA IT HAS AND WHAT'S MISSING.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterator
import hashlib
import threading

import yaml

logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMERATIONS
# ==============================================================================

class DatasetType(str, Enum):
    """Types of datasets tracked by the registry."""
    DEM = "dem"
    RAINFALL = "rainfall"
    COMPLAINTS = "complaints"
    RADAR = "radar"
    LAND_USE = "land_use"
    DRAINAGE_NETWORK = "drainage_network"
    POPULATION = "population"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class AutomationMode(str, Enum):
    """How the dataset was acquired."""
    AUTO = "auto"           # Fully automated download
    SEMI_AUTO = "semi_auto" # Automated with manual trigger
    MANUAL = "manual"       # Manual download required
    SYNTHETIC = "synthetic" # Generated for testing
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    """Validation state of a dataset."""
    VALID = "valid"             # Passed all checks
    VALID_WITH_WARNINGS = "valid_with_warnings"  # Passed with concerns
    DEGRADED = "degraded"       # Usable but limited
    INVALID = "invalid"         # Failed critical checks
    PENDING = "pending"         # Not yet validated
    STALE = "stale"             # Validation expired


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class SpatialCoverage:
    """Spatial extent of a dataset."""
    min_lat: float = 0.0
    max_lat: float = 0.0
    min_lon: float = 0.0
    max_lon: float = 0.0
    crs: str = "EPSG:4326"
    resolution_m: Optional[float] = None
    coverage_area_km2: Optional[float] = None
    city: str = ""
    region: str = ""

    def overlaps(self, other: "SpatialCoverage") -> bool:
        """Check if spatial extents overlap."""
        return not (
            self.max_lat < other.min_lat or
            self.min_lat > other.max_lat or
            self.max_lon < other.min_lon or
            self.min_lon > other.max_lon
        )

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within coverage."""
        return (
            self.min_lat <= lat <= self.max_lat and
            self.min_lon <= lon <= self.max_lon
        )

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
            "crs": self.crs,
            "resolution_m": self.resolution_m,
            "coverage_area_km2": self.coverage_area_km2,
            "city": self.city,
            "region": self.region,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SpatialCoverage":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TemporalCoverage:
    """Temporal extent of a dataset."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    resolution_hours: Optional[float] = None
    gaps: List[Tuple[datetime, datetime]] = field(default_factory=list)
    is_static: bool = False  # For DEM, land use, etc.

    def covers(self, target_start: datetime, target_end: datetime) -> bool:
        """Check if temporal extent covers target range."""
        if self.is_static:
            return True
        if self.start is None or self.end is None:
            return False
        return self.start <= target_start and self.end >= target_end

    def overlap_fraction(
        self, target_start: datetime, target_end: datetime
    ) -> float:
        """Calculate fraction of target period covered."""
        if self.is_static:
            return 1.0
        if self.start is None or self.end is None:
            return 0.0

        overlap_start = max(self.start, target_start)
        overlap_end = min(self.end, target_end)

        if overlap_start >= overlap_end:
            return 0.0

        target_duration = (target_end - target_start).total_seconds()
        overlap_duration = (overlap_end - overlap_start).total_seconds()

        return overlap_duration / target_duration if target_duration > 0 else 0.0

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "resolution_hours": self.resolution_hours,
            "gaps": [
                (g[0].isoformat(), g[1].isoformat()) for g in self.gaps
            ],
            "is_static": self.is_static,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TemporalCoverage":
        """Deserialize from dictionary."""
        return cls(
            start=datetime.fromisoformat(data["start"]) if data.get("start") else None,
            end=datetime.fromisoformat(data["end"]) if data.get("end") else None,
            resolution_hours=data.get("resolution_hours"),
            gaps=[
                (datetime.fromisoformat(g[0]), datetime.fromisoformat(g[1]))
                for g in data.get("gaps", [])
            ],
            is_static=data.get("is_static", False),
        )


@dataclass
class DataQualityMetrics:
    """Quality metrics for a dataset."""
    completeness: float = 0.0      # 0-100%
    accuracy: float = 0.0          # 0-100%
    consistency: float = 0.0       # 0-100%
    timeliness_score: float = 0.0  # 0-100%
    overall_score: float = 0.0     # 0-100%
    grade: str = "F"               # A-F
    bias_indicators: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "DataQualityMetrics":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetMetadata:
    """Complete metadata for a registered dataset."""

    # Identity
    dataset_id: str = ""
    dataset_type: DatasetType = DatasetType.OTHER
    name: str = ""
    description: str = ""

    # Source
    source_name: str = ""
    source_url: str = ""
    automation_mode: AutomationMode = AutomationMode.UNKNOWN
    download_timestamp: Optional[datetime] = None

    # Location
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    file_size_bytes: Optional[int] = None

    # Coverage
    spatial_coverage: SpatialCoverage = field(default_factory=SpatialCoverage)
    temporal_coverage: TemporalCoverage = field(default_factory=TemporalCoverage)

    # Quality
    quality_metrics: DataQualityMetrics = field(default_factory=DataQualityMetrics)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_timestamp: Optional[datetime] = None
    validation_errors: List[str] = field(default_factory=list)

    # Provenance
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    parent_dataset_id: Optional[str] = None  # For derived datasets
    processing_history: List[Dict] = field(default_factory=list)

    # Pipeline flags
    is_required: bool = False
    is_available: bool = False
    degraded_reason: Optional[str] = None

    def __post_init__(self):
        """Generate dataset_id if not provided."""
        if not self.dataset_id:
            self.dataset_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique dataset ID."""
        components = [
            self.dataset_type.value,
            self.source_name or "unknown",
            self.name or "unnamed",
            datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        ]
        base = "_".join(components)
        return hashlib.sha256(base.encode()).hexdigest()[:12]

    def mark_valid(self, warnings: Optional[List[str]] = None) -> None:
        """Mark dataset as valid."""
        if warnings:
            self.validation_status = ValidationStatus.VALID_WITH_WARNINGS
            self.quality_metrics.warnings.extend(warnings)
        else:
            self.validation_status = ValidationStatus.VALID
        self.validation_timestamp = datetime.utcnow()
        self.is_available = True
        self.updated_at = datetime.utcnow()

    def mark_invalid(self, errors: List[str]) -> None:
        """Mark dataset as invalid."""
        self.validation_status = ValidationStatus.INVALID
        self.validation_errors.extend(errors)
        self.validation_timestamp = datetime.utcnow()
        self.is_available = False
        self.updated_at = datetime.utcnow()

    def mark_degraded(self, reason: str) -> None:
        """Mark dataset as degraded but usable."""
        self.validation_status = ValidationStatus.DEGRADED
        self.degraded_reason = reason
        self.validation_timestamp = datetime.utcnow()
        self.is_available = True
        self.updated_at = datetime.utcnow()

    def add_processing_step(self, step_name: str, details: Dict) -> None:
        """Record a processing step in history."""
        self.processing_history.append({
            "step": step_name,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
        })
        self.updated_at = datetime.utcnow()
        self.version += 1

    def to_dict(self) -> Dict:
        """Serialize to dictionary for persistence."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_type": self.dataset_type.value,
            "name": self.name,
            "description": self.description,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "automation_mode": self.automation_mode.value,
            "download_timestamp": (
                self.download_timestamp.isoformat()
                if self.download_timestamp else None
            ),
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_size_bytes": self.file_size_bytes,
            "spatial_coverage": self.spatial_coverage.to_dict(),
            "temporal_coverage": self.temporal_coverage.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "validation_status": self.validation_status.value,
            "validation_timestamp": (
                self.validation_timestamp.isoformat()
                if self.validation_timestamp else None
            ),
            "validation_errors": self.validation_errors,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "parent_dataset_id": self.parent_dataset_id,
            "processing_history": self.processing_history,
            "is_required": self.is_required,
            "is_available": self.is_available,
            "degraded_reason": self.degraded_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetMetadata":
        """Deserialize from dictionary."""
        return cls(
            dataset_id=data.get("dataset_id", ""),
            dataset_type=DatasetType(data.get("dataset_type", "other")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            source_name=data.get("source_name", ""),
            source_url=data.get("source_url", ""),
            automation_mode=AutomationMode(
                data.get("automation_mode", "unknown")
            ),
            download_timestamp=(
                datetime.fromisoformat(data["download_timestamp"])
                if data.get("download_timestamp") else None
            ),
            file_path=data.get("file_path"),
            file_hash=data.get("file_hash"),
            file_size_bytes=data.get("file_size_bytes"),
            spatial_coverage=SpatialCoverage.from_dict(
                data.get("spatial_coverage", {})
            ),
            temporal_coverage=TemporalCoverage.from_dict(
                data.get("temporal_coverage", {})
            ),
            quality_metrics=DataQualityMetrics.from_dict(
                data.get("quality_metrics", {})
            ),
            validation_status=ValidationStatus(
                data.get("validation_status", "pending")
            ),
            validation_timestamp=(
                datetime.fromisoformat(data["validation_timestamp"])
                if data.get("validation_timestamp") else None
            ),
            validation_errors=data.get("validation_errors", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at") else datetime.utcnow()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at") else datetime.utcnow()
            ),
            version=data.get("version", 1),
            parent_dataset_id=data.get("parent_dataset_id"),
            processing_history=data.get("processing_history", []),
            is_required=data.get("is_required", False),
            is_available=data.get("is_available", False),
            degraded_reason=data.get("degraded_reason"),
        )


# ==============================================================================
# REGISTRY CONFIGURATION
# ==============================================================================

@dataclass
class RegistryConfig:
    """Configuration for the data registry."""

    # Persistence
    registry_path: Path = field(
        default_factory=lambda: Path("data/registry/registry.yaml")
    )
    backup_enabled: bool = True
    backup_count: int = 5

    # Required datasets by type
    required_datasets: Dict[str, List[DatasetType]] = field(
        default_factory=lambda: {
            "full_run": [
                DatasetType.DEM,
                DatasetType.RAINFALL,
                DatasetType.COMPLAINTS,
            ],
            "degraded_run": [
                DatasetType.DEM,
                DatasetType.RAINFALL,
            ],
            "minimal_run": [
                DatasetType.DEM,
            ],
        }
    )

    # Quality thresholds
    min_quality_score: float = 50.0
    min_temporal_coverage: float = 0.7  # 70% coverage required
    min_spatial_overlap: float = 0.5    # 50% overlap required

    # Staleness
    validation_ttl_hours: int = 24  # Re-validate after 24 hours

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "registry_path": str(self.registry_path),
            "backup_enabled": self.backup_enabled,
            "backup_count": self.backup_count,
            "required_datasets": {
                k: [d.value for d in v]
                for k, v in self.required_datasets.items()
            },
            "min_quality_score": self.min_quality_score,
            "min_temporal_coverage": self.min_temporal_coverage,
            "min_spatial_overlap": self.min_spatial_overlap,
            "validation_ttl_hours": self.validation_ttl_hours,
        }


# ==============================================================================
# DATA REGISTRY
# ==============================================================================

class DataRegistry:
    """Central registry for all pipeline datasets.

    This is the BACKBONE of the data-aware pipeline:
    - Tracks all datasets with full metadata
    - Persists state to disk for durability
    - Provides queryable interface for pipeline decisions
    - Supports auditing and provenance tracking
    """

    def __init__(
        self,
        config: Optional[RegistryConfig] = None,
        auto_load: bool = True,
    ):
        """Initialize the registry.

        Args:
            config: Registry configuration
            auto_load: Load existing registry from disk
        """
        self.config = config or RegistryConfig()
        self._datasets: Dict[str, DatasetMetadata] = {}
        self._lock = threading.RLock()
        self._dirty = False

        # Ensure registry directory exists
        self.config.registry_path.parent.mkdir(parents=True, exist_ok=True)

        if auto_load and self.config.registry_path.exists():
            self.load()

        logger.info(
            "DataRegistry initialized with %d datasets",
            len(self._datasets),
        )

    # ==========================================================================
    # CRUD OPERATIONS
    # ==========================================================================

    def register(
        self,
        metadata: DatasetMetadata,
        overwrite: bool = False,
    ) -> str:
        """Register a new dataset.

        Args:
            metadata: Dataset metadata
            overwrite: Allow overwriting existing dataset

        Returns:
            Dataset ID

        Raises:
            ValueError: If dataset exists and overwrite=False
        """
        with self._lock:
            if metadata.dataset_id in self._datasets and not overwrite:
                raise ValueError(
                    f"Dataset {metadata.dataset_id} already exists. "
                    "Use overwrite=True to replace."
                )

            metadata.updated_at = datetime.utcnow()
            self._datasets[metadata.dataset_id] = metadata
            self._dirty = True

            logger.info(
                "Registered dataset: %s (%s) from %s",
                metadata.dataset_id,
                metadata.dataset_type.value,
                metadata.source_name,
            )

            return metadata.dataset_id

    def update(
        self,
        dataset_id: str,
        updates: Dict[str, Any],
    ) -> DatasetMetadata:
        """Update dataset metadata.

        Args:
            dataset_id: Dataset to update
            updates: Fields to update

        Returns:
            Updated metadata

        Raises:
            KeyError: If dataset not found
        """
        with self._lock:
            if dataset_id not in self._datasets:
                raise KeyError(f"Dataset not found: {dataset_id}")

            metadata = self._datasets[dataset_id]

            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

            metadata.updated_at = datetime.utcnow()
            metadata.version += 1
            self._dirty = True

            logger.debug("Updated dataset %s: %s", dataset_id, list(updates.keys()))

            return metadata

    def get(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by ID."""
        return self._datasets.get(dataset_id)

    def remove(self, dataset_id: str) -> bool:
        """Remove a dataset from registry.

        Args:
            dataset_id: Dataset to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if dataset_id in self._datasets:
                del self._datasets[dataset_id]
                self._dirty = True
                logger.info("Removed dataset: %s", dataset_id)
                return True
            return False

    # ==========================================================================
    # QUERY OPERATIONS
    # ==========================================================================

    def list_all(self) -> List[DatasetMetadata]:
        """List all registered datasets."""
        return list(self._datasets.values())

    def query(
        self,
        dataset_type: Optional[DatasetType] = None,
        validation_status: Optional[ValidationStatus] = None,
        is_available: Optional[bool] = None,
        is_required: Optional[bool] = None,
        source_name: Optional[str] = None,
        city: Optional[str] = None,
    ) -> List[DatasetMetadata]:
        """Query datasets with filters.

        Args:
            dataset_type: Filter by type
            validation_status: Filter by validation status
            is_available: Filter by availability
            is_required: Filter by required flag
            source_name: Filter by source
            city: Filter by city in spatial coverage

        Returns:
            List of matching datasets
        """
        results = []

        for ds in self._datasets.values():
            if dataset_type and ds.dataset_type != dataset_type:
                continue
            if validation_status and ds.validation_status != validation_status:
                continue
            if is_available is not None and ds.is_available != is_available:
                continue
            if is_required is not None and ds.is_required != is_required:
                continue
            if source_name and ds.source_name != source_name:
                continue
            if city and ds.spatial_coverage.city.lower() != city.lower():
                continue
            results.append(ds)

        return results

    def get_by_type(self, dataset_type: DatasetType) -> List[DatasetMetadata]:
        """Get all datasets of a specific type."""
        return self.query(dataset_type=dataset_type)

    def get_available(self) -> List[DatasetMetadata]:
        """Get all available datasets."""
        return self.query(is_available=True)

    def get_required(self) -> List[DatasetMetadata]:
        """Get all required datasets."""
        return self.query(is_required=True)

    def get_best_for_type(
        self,
        dataset_type: DatasetType,
        city: Optional[str] = None,
    ) -> Optional[DatasetMetadata]:
        """Get the best available dataset of a type.

        Selection criteria (in order):
        1. Validation status (valid > degraded > others)
        2. Quality score
        3. Recency

        Args:
            dataset_type: Dataset type
            city: Optional city filter

        Returns:
            Best dataset or None
        """
        candidates = self.query(
            dataset_type=dataset_type,
            is_available=True,
            city=city,
        )

        if not candidates:
            return None

        # Score candidates
        def score_dataset(ds: DatasetMetadata) -> Tuple[int, float, datetime]:
            status_score = {
                ValidationStatus.VALID: 4,
                ValidationStatus.VALID_WITH_WARNINGS: 3,
                ValidationStatus.DEGRADED: 2,
                ValidationStatus.PENDING: 1,
            }.get(ds.validation_status, 0)

            return (
                status_score,
                ds.quality_metrics.overall_score,
                ds.updated_at,
            )

        candidates.sort(key=score_dataset, reverse=True)
        return candidates[0]

    def has_type(self, dataset_type: DatasetType) -> bool:
        """Check if any dataset of type is available."""
        return any(
            ds.is_available
            for ds in self._datasets.values()
            if ds.dataset_type == dataset_type
        )

    def get_missing_required(
        self,
        run_mode: str = "full_run",
    ) -> List[DatasetType]:
        """Get list of required dataset types that are missing.

        Args:
            run_mode: One of 'full_run', 'degraded_run', 'minimal_run'

        Returns:
            List of missing dataset types
        """
        required_types = self.config.required_datasets.get(run_mode, [])
        missing = []

        for dtype in required_types:
            if not self.has_type(dtype):
                missing.append(dtype)

        return missing

    # ==========================================================================
    # COVERAGE QUERIES
    # ==========================================================================

    def check_temporal_coverage(
        self,
        target_start: datetime,
        target_end: datetime,
        dataset_type: Optional[DatasetType] = None,
    ) -> Dict[str, float]:
        """Check temporal coverage for target period.

        Args:
            target_start: Start of target period
            target_end: End of target period
            dataset_type: Filter by type (None for all)

        Returns:
            Dict mapping dataset_id to coverage fraction
        """
        results = {}

        for ds in self._datasets.values():
            if dataset_type and ds.dataset_type != dataset_type:
                continue
            if not ds.is_available:
                continue

            coverage = ds.temporal_coverage.overlap_fraction(
                target_start, target_end
            )
            results[ds.dataset_id] = coverage

        return results

    def check_spatial_coverage(
        self,
        bounds: Tuple[float, float, float, float],
        dataset_type: Optional[DatasetType] = None,
    ) -> Dict[str, bool]:
        """Check spatial coverage for target bounds.

        Args:
            bounds: (min_lat, min_lon, max_lat, max_lon)
            dataset_type: Filter by type (None for all)

        Returns:
            Dict mapping dataset_id to overlap boolean
        """
        results = {}
        min_lat, min_lon, max_lat, max_lon = bounds

        target_coverage = SpatialCoverage(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
        )

        for ds in self._datasets.values():
            if dataset_type and ds.dataset_type != dataset_type:
                continue
            if not ds.is_available:
                continue

            overlaps = ds.spatial_coverage.overlaps(target_coverage)
            results[ds.dataset_id] = overlaps

        return results

    # ==========================================================================
    # PERSISTENCE
    # ==========================================================================

    def save(self, force: bool = False) -> None:
        """Save registry to disk.

        Args:
            force: Save even if no changes
        """
        if not self._dirty and not force:
            return

        with self._lock:
            # Create backup
            if self.config.backup_enabled and self.config.registry_path.exists():
                self._create_backup()

            # Serialize
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "config": self.config.to_dict(),
                "datasets": {
                    k: v.to_dict() for k, v in self._datasets.items()
                },
            }

            # Write
            with open(self.config.registry_path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

            self._dirty = False
            logger.info(
                "Registry saved: %d datasets to %s",
                len(self._datasets),
                self.config.registry_path,
            )

    def load(self) -> None:
        """Load registry from disk."""
        if not self.config.registry_path.exists():
            logger.warning("Registry file not found: %s", self.config.registry_path)
            return

        with self._lock:
            with open(self.config.registry_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return

            # Load datasets
            datasets_data = data.get("datasets", {})
            self._datasets = {
                k: DatasetMetadata.from_dict(v)
                for k, v in datasets_data.items()
            }

            self._dirty = False
            logger.info(
                "Registry loaded: %d datasets from %s",
                len(self._datasets),
                self.config.registry_path,
            )

    def _create_backup(self) -> None:
        """Create backup of registry file."""
        backup_dir = self.config.registry_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"registry_{timestamp}.yaml"

        import shutil
        shutil.copy(self.config.registry_path, backup_path)

        # Clean old backups
        backups = sorted(backup_dir.glob("registry_*.yaml"))
        while len(backups) > self.config.backup_count:
            backups[0].unlink()
            backups.pop(0)

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def __len__(self) -> int:
        return len(self._datasets)

    def __contains__(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets

    def __iter__(self) -> Iterator[DatasetMetadata]:
        return iter(self._datasets.values())

    def summary(self) -> Dict:
        """Generate registry summary."""
        by_type = {}
        for dtype in DatasetType:
            datasets = self.get_by_type(dtype)
            by_type[dtype.value] = {
                "total": len(datasets),
                "available": sum(1 for d in datasets if d.is_available),
                "valid": sum(
                    1 for d in datasets
                    if d.validation_status in [
                        ValidationStatus.VALID,
                        ValidationStatus.VALID_WITH_WARNINGS,
                    ]
                ),
            }

        return {
            "total_datasets": len(self._datasets),
            "available": len(self.get_available()),
            "required": len(self.get_required()),
            "by_type": by_type,
            "last_updated": max(
                (d.updated_at for d in self._datasets.values()),
                default=None,
            ),
        }

    def generate_report(self) -> str:
        """Generate human-readable registry report."""
        lines = [
            "=" * 60,
            "DATA REGISTRY REPORT",
            "=" * 60,
            "",
            f"Total Datasets: {len(self._datasets)}",
            f"Available: {len(self.get_available())}",
            "",
        ]

        for dtype in DatasetType:
            datasets = self.get_by_type(dtype)
            if datasets:
                lines.append(f"[{dtype.value.upper()}]")
                for ds in datasets:
                    status = "✓" if ds.is_available else "✗"
                    lines.append(
                        f"  {status} {ds.name} ({ds.source_name}) "
                        f"- {ds.validation_status.value}"
                    )
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_registry(
    config_path: Optional[Path] = None,
) -> DataRegistry:
    """Get or create the singleton registry instance.

    Args:
        config_path: Path to registry file

    Returns:
        DataRegistry instance
    """
    config = RegistryConfig()
    if config_path:
        config.registry_path = config_path

    return DataRegistry(config=config)


def register_dataset(
    dataset_type: DatasetType,
    name: str,
    source_name: str,
    file_path: Path,
    automation_mode: AutomationMode = AutomationMode.AUTO,
    spatial_coverage: Optional[SpatialCoverage] = None,
    temporal_coverage: Optional[TemporalCoverage] = None,
    is_required: bool = False,
) -> str:
    """Convenience function to register a dataset.

    Args:
        dataset_type: Type of dataset
        name: Dataset name
        source_name: Source identifier
        file_path: Path to data file
        automation_mode: How dataset was acquired
        spatial_coverage: Spatial extent
        temporal_coverage: Temporal extent
        is_required: Is this dataset required

    Returns:
        Dataset ID
    """
    registry = get_registry()

    metadata = DatasetMetadata(
        dataset_type=dataset_type,
        name=name,
        source_name=source_name,
        file_path=str(file_path),
        automation_mode=automation_mode,
        download_timestamp=datetime.utcnow(),
        is_required=is_required,
    )

    if spatial_coverage:
        metadata.spatial_coverage = spatial_coverage
    if temporal_coverage:
        metadata.temporal_coverage = temporal_coverage

    # Calculate file hash
    if file_path.exists():
        metadata.file_size_bytes = file_path.stat().st_size
        with open(file_path, "rb") as f:
            metadata.file_hash = hashlib.md5(f.read()).hexdigest()

    dataset_id = registry.register(metadata)
    registry.save()

    return dataset_id
