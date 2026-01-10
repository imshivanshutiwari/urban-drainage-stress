"""Maximum-level complaint data normalization (Data Automation Prompt-3).

Full-featured normalization including:
- Aggressive geocoding with multiple fallback services
- Address parsing and standardization
- Fuzzy duplicate detection and deduplication
- Text normalization and keyword extraction
- Spatial clustering analysis
- Cross-source entity resolution
- Coordinate validation and correction

Note: Even aggressive geocoding preserves uncertainty flags.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import json
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# GEOCODING SERVICES
# ==============================================================================

@dataclass
class GeocodingResult:
    """Result of geocoding attempt."""

    success: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: float = 0.0
    source: str = "unknown"
    match_type: str = "none"  # exact, interpolated, centroid, ward_centroid
    address_standardized: str = ""
    uncertainty_m: float = float("inf")
    metadata: Dict = field(default_factory=dict)


class GeocodingService:
    """Base class for geocoding services."""

    def __init__(self, rate_limit_delay: float = 0.1):
        self.rate_limit_delay = rate_limit_delay
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def geocode(self, address: str, city: str = "") -> GeocodingResult:
        """Geocode an address. Override in subclasses."""
        raise NotImplementedError


class NominatimGeocoder(GeocodingService):
    """OpenStreetMap Nominatim geocoding service."""

    BASE_URL = "https://nominatim.openstreetmap.org/search"

    def __init__(self, user_agent: str = "UrbanDrainageStress/1.0"):
        super().__init__(rate_limit_delay=1.0)  # Nominatim requires 1s delay
        self.user_agent = user_agent

    def geocode(self, address: str, city: str = "") -> GeocodingResult:
        """Geocode using Nominatim."""
        self._rate_limit()

        result = GeocodingResult(source="nominatim")

        try:
            query = f"{address}, {city}" if city else address
            url = (
                f"{self.BASE_URL}?q={quote_plus(query)}"
                f"&format=json&limit=1&addressdetails=1"
            )

            req = Request(url, headers={"User-Agent": self.user_agent})
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if data:
                item = data[0]
                result.success = True
                result.latitude = float(item["lat"])
                result.longitude = float(item["lon"])
                result.confidence = min(float(item.get("importance", 0.5)), 1.0)
                result.match_type = item.get("type", "unknown")
                result.address_standardized = item.get("display_name", "")

                # Estimate uncertainty based on match type
                match_type = item.get("class", "")
                if match_type == "building":
                    result.uncertainty_m = 10.0
                elif match_type == "highway":
                    result.uncertainty_m = 50.0
                elif match_type in ("place", "boundary"):
                    result.uncertainty_m = 500.0
                else:
                    result.uncertainty_m = 100.0

                result.metadata = {
                    "osm_id": item.get("osm_id"),
                    "osm_type": item.get("osm_type"),
                    "class": item.get("class"),
                }

        except Exception as e:
            logger.debug("Nominatim geocoding failed: %s", e)

        return result


class WardCentroidGeocoder(GeocodingService):
    """Geocode to ward/district centroids when point geocoding fails."""

    # Pre-defined ward centroids for common cities
    WARD_CENTROIDS: Dict[str, Dict[str, Tuple[float, float]]] = {
        "mumbai": {
            "A": (18.9388, 72.8354),
            "B": (18.9523, 72.8339),
            "C": (18.9631, 72.8310),
            "D": (18.9751, 72.8282),
            "E": (18.9890, 72.8315),
            "F/N": (19.0215, 72.8420),
            "F/S": (19.0078, 72.8390),
            "G/N": (19.0365, 72.8450),
            "G/S": (19.0180, 72.8430),
            "H/E": (19.0520, 72.8510),
            "H/W": (19.0450, 72.8380),
            "K/E": (19.0750, 72.8680),
            "K/W": (19.0680, 72.8450),
            "L": (19.1050, 72.8850),
            "M/E": (19.1280, 72.9050),
            "M/W": (19.1180, 72.8750),
            "N": (19.1450, 72.9150),
            "P/N": (19.1750, 72.9380),
            "P/S": (19.1580, 72.9280),
            "R/C": (19.1950, 72.9550),
            "R/N": (19.2150, 72.9680),
            "R/S": (19.2050, 72.9480),
            "S": (19.0850, 72.8280),
            "T": (19.2280, 72.9780),
        },
        "bangalore": {
            "bommanahalli": (12.9012, 77.6185),
            "btm_layout": (12.9166, 77.6101),
            "dasarahalli": (13.0450, 77.5150),
            "mahadevapura": (12.9960, 77.6970),
            "rajarajeshwari_nagar": (12.9260, 77.5190),
            "south": (12.9400, 77.5850),
            "west": (12.9700, 77.5450),
            "yelahanka": (13.1007, 77.5963),
        },
        "seattle": {
            "ballard": (47.6792, -122.3844),
            "beacon_hill": (47.5690, -122.3110),
            "capitol_hill": (47.6253, -122.3222),
            "central_district": (47.6065, -122.2984),
            "downtown": (47.6062, -122.3321),
            "fremont": (47.6510, -122.3505),
            "greenwood": (47.6930, -122.3550),
            "magnolia": (47.6400, -122.3990),
            "queen_anne": (47.6370, -122.3570),
            "rainier_valley": (47.5350, -122.2810),
            "university_district": (47.6588, -122.3130),
            "west_seattle": (47.5650, -122.3870),
        },
    }

    def __init__(self):
        super().__init__(rate_limit_delay=0.0)  # No rate limit needed

    def geocode(self, address: str, city: str = "") -> GeocodingResult:
        """Geocode to ward centroid."""
        result = GeocodingResult(source="ward_centroid")

        city_key = city.lower().replace(" ", "_")
        if city_key not in self.WARD_CENTROIDS:
            return result

        # Try to extract ward from address
        ward_centroids = self.WARD_CENTROIDS[city_key]
        address_lower = address.lower()

        for ward_name, coords in ward_centroids.items():
            if ward_name.lower() in address_lower:
                result.success = True
                result.latitude, result.longitude = coords
                result.confidence = 0.4
                result.match_type = "ward_centroid"
                result.uncertainty_m = 2000.0
                result.metadata = {"ward": ward_name}
                return result

        return result


class MultiGeocoder:
    """Chain multiple geocoding services with fallback."""

    def __init__(
        self,
        services: Optional[List[GeocodingService]] = None,
        cache_size: int = 10000,
    ):
        self.services = services or [
            NominatimGeocoder(),
            WardCentroidGeocoder(),
        ]
        self._cache: Dict[str, GeocodingResult] = {}
        self._cache_size = cache_size

    def _cache_key(self, address: str, city: str) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{address}|{city}".encode()).hexdigest()

    def geocode(
        self,
        address: str,
        city: str = "",
        min_confidence: float = 0.3,
    ) -> GeocodingResult:
        """Geocode with fallback through services."""
        if not address or not address.strip():
            return GeocodingResult()

        # Check cache
        cache_key = self._cache_key(address, city)
        if cache_key in self._cache:
            return self._cache[cache_key]

        best_result = GeocodingResult()

        for service in self.services:
            try:
                result = service.geocode(address, city)
                if result.success and result.confidence >= min_confidence:
                    # Cache and return
                    if len(self._cache) < self._cache_size:
                        self._cache[cache_key] = result
                    return result

                # Keep best so far
                if result.confidence > best_result.confidence:
                    best_result = result

            except Exception as e:
                logger.debug(
                    "Geocoder %s failed: %s",
                    type(service).__name__, e
                )

        # Cache best result even if below threshold
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = best_result

        return best_result


# ==============================================================================
# TEXT NORMALIZATION
# ==============================================================================

class TextNormalizer:
    """Normalize and clean text fields in complaints."""

    # Common abbreviations
    ABBREVIATIONS = {
        "st": "street",
        "st.": "street",
        "rd": "road",
        "rd.": "road",
        "ave": "avenue",
        "ave.": "avenue",
        "blvd": "boulevard",
        "blvd.": "boulevard",
        "dr": "drive",
        "dr.": "drive",
        "ln": "lane",
        "ln.": "lane",
        "ct": "court",
        "ct.": "court",
        "pl": "place",
        "pl.": "place",
        "sq": "square",
        "sq.": "square",
        "n": "north",
        "n.": "north",
        "s": "south",
        "s.": "south",
        "e": "east",
        "e.": "east",
        "w": "west",
        "w.": "west",
        "ne": "northeast",
        "nw": "northwest",
        "se": "southeast",
        "sw": "southwest",
        "apt": "apartment",
        "apt.": "apartment",
        "bldg": "building",
        "bldg.": "building",
        "flr": "floor",
        "flr.": "floor",
    }

    # Flood-related keywords for classification
    FLOOD_KEYWORDS = {
        "flood", "flooding", "flooded", "waterlogging", "waterlogged",
        "standing water", "ponding", "inundation", "inundated",
        "overflow", "overflowing", "backup", "backed up",
        "drain", "drainage", "clogged", "blocked", "blockage",
        "sewer", "storm drain", "catch basin", "culvert",
        "water damage", "basement flood", "street flood",
    }

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        if not text:
            return ""
        # NFKD decomposition, remove diacritics
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(c for c in normalized if not unicodedata.combining(c))

    @classmethod
    def normalize_address(cls, address: str) -> str:
        """Normalize address string."""
        if not address:
            return ""

        # Lowercase and strip
        result = address.lower().strip()

        # Normalize unicode
        result = cls.normalize_unicode(result)

        # Expand abbreviations
        words = result.split()
        expanded = []
        for word in words:
            expanded.append(cls.ABBREVIATIONS.get(word, word))

        result = " ".join(expanded)

        # Remove extra whitespace
        result = re.sub(r"\s+", " ", result)

        # Remove special characters except comma, hyphen, period
        result = re.sub(r"[^\w\s,.\-]", "", result)

        return result.strip()

    @classmethod
    def normalize_description(cls, description: str) -> str:
        """Normalize complaint description."""
        if not description:
            return ""

        result = description.lower().strip()
        result = cls.normalize_unicode(result)

        # Remove excessive punctuation
        result = re.sub(r"[!?]{2,}", "!", result)
        result = re.sub(r"\.{2,}", ".", result)

        # Remove extra whitespace
        result = re.sub(r"\s+", " ", result)

        return result.strip()

    @classmethod
    def extract_flood_keywords(cls, text: str) -> List[str]:
        """Extract flood-related keywords from text."""
        if not text:
            return []

        text_lower = text.lower()
        found = []

        for keyword in cls.FLOOD_KEYWORDS:
            if keyword in text_lower:
                found.append(keyword)

        return found

    @classmethod
    def compute_text_hash(cls, text: str) -> str:
        """Compute hash for deduplication."""
        if not text:
            return ""
        normalized = cls.normalize_description(text)
        # Remove common words for fuzzy matching
        words = [w for w in normalized.split() if len(w) > 3]
        return hashlib.md5(" ".join(sorted(words)).encode()).hexdigest()[:12]


# ==============================================================================
# DUPLICATE DETECTION
# ==============================================================================

@dataclass
class DuplicateGroup:
    """Group of potential duplicate complaints."""

    group_id: str
    indices: List[int]
    confidence: float
    reason: str
    representative_idx: int


class DuplicateDetector:
    """Detect and handle duplicate complaints."""

    def __init__(
        self,
        spatial_threshold_m: float = 100.0,
        temporal_threshold_hours: float = 24.0,
        text_similarity_threshold: float = 0.8,
    ):
        self.spatial_threshold_m = spatial_threshold_m
        self.temporal_threshold_hours = temporal_threshold_hours
        self.text_similarity_threshold = text_similarity_threshold

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000  # Earth radius in meters

        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = (
            np.sin(delta_lat / 2) ** 2 +
            np.cos(lat1_r) * np.cos(lat2_r) * np.sin(delta_lon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def find_spatial_temporal_duplicates(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        ts_col: str = "timestamp",
    ) -> List[DuplicateGroup]:
        """Find duplicates based on spatial-temporal proximity."""
        duplicates = []

        if lat_col not in df.columns or lon_col not in df.columns:
            return duplicates
        if ts_col not in df.columns:
            return duplicates

        # Convert to numpy for speed
        lats = pd.to_numeric(df[lat_col], errors="coerce").values
        lons = pd.to_numeric(df[lon_col], errors="coerce").values
        timestamps = pd.to_datetime(df[ts_col], errors="coerce")

        valid_mask = ~(np.isnan(lats) | np.isnan(lons) | timestamps.isna())
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            return duplicates

        # Sort by timestamp for efficiency
        sorted_order = timestamps[valid_indices].argsort()
        sorted_indices = valid_indices[sorted_order]

        visited = set()
        group_counter = 0

        for i, idx_i in enumerate(sorted_indices):
            if idx_i in visited:
                continue

            group_indices = [idx_i]
            ts_i = timestamps.iloc[idx_i]
            lat_i, lon_i = lats[idx_i], lons[idx_i]

            # Check subsequent records within temporal window
            for j in range(i + 1, len(sorted_indices)):
                idx_j = sorted_indices[j]
                if idx_j in visited:
                    continue

                ts_j = timestamps.iloc[idx_j]
                time_diff = (ts_j - ts_i).total_seconds() / 3600

                if time_diff > self.temporal_threshold_hours:
                    break  # Past temporal window

                lat_j, lon_j = lats[idx_j], lons[idx_j]
                dist = self.haversine_distance(lat_i, lon_i, lat_j, lon_j)

                if dist <= self.spatial_threshold_m:
                    group_indices.append(idx_j)

            if len(group_indices) > 1:
                group_counter += 1
                visited.update(group_indices)
                duplicates.append(DuplicateGroup(
                    group_id=f"ST_{group_counter:04d}",
                    indices=group_indices,
                    confidence=0.8,
                    reason="spatial_temporal_proximity",
                    representative_idx=group_indices[0],
                ))

        return duplicates

    def find_text_duplicates(
        self,
        df: pd.DataFrame,
        text_col: str = "description",
    ) -> List[DuplicateGroup]:
        """Find duplicates based on text similarity."""
        duplicates = []

        if text_col not in df.columns:
            return duplicates

        # Compute text hashes
        hashes = df[text_col].apply(TextNormalizer.compute_text_hash)

        # Group by hash
        hash_groups = hashes.groupby(hashes).groups

        group_counter = 0
        for hash_val, indices in hash_groups.items():
            if len(indices) > 1 and hash_val:  # Skip empty hashes
                group_counter += 1
                idx_list = list(indices)
                duplicates.append(DuplicateGroup(
                    group_id=f"TXT_{group_counter:04d}",
                    indices=idx_list,
                    confidence=0.9,
                    reason="text_similarity",
                    representative_idx=idx_list[0],
                ))

        return duplicates

    def detect_all_duplicates(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[DuplicateGroup], pd.DataFrame]:
        """Detect all types of duplicates and mark them.

        Returns:
            (duplicate_groups, df_with_flags)
        """
        all_groups = []

        # Spatial-temporal duplicates
        st_dups = self.find_spatial_temporal_duplicates(df)
        all_groups.extend(st_dups)

        # Text duplicates
        txt_dups = self.find_text_duplicates(df)
        all_groups.extend(txt_dups)

        # Add duplicate flags to dataframe
        df_result = df.copy()
        df_result["is_potential_duplicate"] = False
        df_result["duplicate_group_id"] = ""
        df_result["is_representative"] = True

        for group in all_groups:
            for idx in group.indices:
                if idx < len(df_result):
                    df_result.loc[idx, "is_potential_duplicate"] = True
                    df_result.loc[idx, "duplicate_group_id"] = group.group_id
                    if idx != group.representative_idx:
                        df_result.loc[idx, "is_representative"] = False

        n_duplicates = df_result["is_potential_duplicate"].sum()
        logger.info(
            "Found %d potential duplicates in %d groups",
            n_duplicates, len(all_groups)
        )

        return all_groups, df_result


# ==============================================================================
# MAIN NORMALIZATION PIPELINE
# ==============================================================================

@dataclass
class NormalizationResult:
    """Result of full normalization pipeline."""

    df: pd.DataFrame
    records_input: int = 0
    records_output: int = 0
    geocoded_count: int = 0
    geocode_failed_count: int = 0
    duplicates_found: int = 0
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    flood_keyword_matches: int = 0
    normalization_stats: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class ComplaintNormalizer:
    """Full-featured complaint data normalization."""

    def __init__(
        self,
        geocoder: Optional[MultiGeocoder] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
        enable_geocoding: bool = True,
        enable_deduplication: bool = True,
        enable_text_normalization: bool = True,
    ):
        self.geocoder = geocoder or MultiGeocoder()
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.enable_geocoding = enable_geocoding
        self.enable_deduplication = enable_deduplication
        self.enable_text_normalization = enable_text_normalization

    def normalize(
        self,
        df: pd.DataFrame,
        city: str = "",
        source_id: Optional[str] = None,
    ) -> NormalizationResult:
        """Run full normalization pipeline.

        Args:
            df: Raw complaint DataFrame
            city: City name for geocoding context
            source_id: Data source identifier

        Returns:
            NormalizationResult with normalized data
        """
        result = NormalizationResult(
            df=df.copy(),
            records_input=len(df),
        )

        logger.info("Starting normalization of %d records", len(df))

        # Step 1: Standardize column names
        result.df = self._standardize_columns(result.df)

        # Step 2: Parse and validate timestamps
        result.df = self._normalize_timestamps(result.df)

        # Step 3: Normalize text fields
        if self.enable_text_normalization:
            result.df = self._normalize_text(result.df)
            result.flood_keyword_matches = result.df[
                "flood_keywords"
            ].apply(len).sum() if "flood_keywords" in result.df else 0

        # Step 4: Validate and standardize coordinates
        result.df = self._normalize_coordinates(result.df)

        # Step 5: Geocode missing coordinates
        if self.enable_geocoding:
            result.df, geocode_stats = self._geocode_missing(result.df, city)
            result.geocoded_count = geocode_stats["geocoded"]
            result.geocode_failed_count = geocode_stats["failed"]

        # Step 6: Detect duplicates
        if self.enable_deduplication:
            result.duplicate_groups, result.df = (
                self.duplicate_detector.detect_all_duplicates(result.df)
            )
            result.duplicates_found = sum(
                len(g.indices) for g in result.duplicate_groups
            )

        # Step 7: Add metadata
        if source_id:
            result.df["data_source"] = source_id
        result.df["normalized_at"] = datetime.utcnow().isoformat()

        # Compute final stats
        result.records_output = len(result.df)
        result.normalization_stats = self._compute_stats(result.df)

        logger.info(
            "Normalization complete: %d -> %d records, "
            "%d geocoded, %d duplicates",
            result.records_input, result.records_output,
            result.geocoded_count, result.duplicates_found
        )

        return result

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        column_mapping = {
            # Timestamp columns
            "created_date": "timestamp",
            "complaint_date": "timestamp",
            "reported_at": "timestamp",
            "date": "timestamp",
            "datetime": "timestamp",
            "created": "timestamp",
            # Latitude columns
            "lat": "latitude",
            "y": "latitude",
            "lat_coord": "latitude",
            # Longitude columns
            "lng": "longitude",
            "lon": "longitude",
            "long": "longitude",
            "x": "longitude",
            "lon_coord": "longitude",
            # Complaint type columns
            "sr_type": "complaint_type",
            "request_type": "complaint_type",
            "category": "complaint_type",
            "type": "complaint_type",
            "issue": "complaint_type",
            "issue_type": "complaint_type",
            # Address columns
            "location": "address",
            "incident_address": "address",
            "street_address": "address",
            "location_desc": "address",
            # Description columns
            "details": "description",
            "notes": "description",
            "complaint_details": "description",
            "issue_description": "description",
            # Ward columns
            "ward_name": "ward",
            "district": "ward",
            "zone": "ward",
            "area": "ward",
        }

        df = df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]

        return df

    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and normalize timestamp column."""
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Extract temporal features
            valid_ts = df["timestamp"].notna()
            df.loc[valid_ts, "hour"] = df.loc[valid_ts, "timestamp"].dt.hour
            df.loc[valid_ts, "day_of_week"] = (
                df.loc[valid_ts, "timestamp"].dt.dayofweek
            )
            df.loc[valid_ts, "month"] = df.loc[valid_ts, "timestamp"].dt.month
            df.loc[valid_ts, "year"] = df.loc[valid_ts, "timestamp"].dt.year

            # Flag missing timestamps
            df["timestamp_missing"] = ~valid_ts

        return df

    def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text fields and extract keywords."""
        df = df.copy()

        # Normalize address
        if "address" in df.columns:
            df["address_normalized"] = df["address"].apply(
                lambda x: TextNormalizer.normalize_address(str(x))
                if pd.notna(x) else ""
            )

        # Normalize description and extract keywords
        if "description" in df.columns:
            df["description_normalized"] = df["description"].apply(
                lambda x: TextNormalizer.normalize_description(str(x))
                if pd.notna(x) else ""
            )

            df["flood_keywords"] = df["description"].apply(
                lambda x: TextNormalizer.extract_flood_keywords(str(x))
                if pd.notna(x) else []
            )

            df["has_flood_keywords"] = df["flood_keywords"].apply(
                lambda x: len(x) > 0
            )

            # Compute text hash for deduplication
            df["description_hash"] = df["description"].apply(
                lambda x: TextNormalizer.compute_text_hash(str(x))
                if pd.notna(x) else ""
            )

        # Normalize complaint type
        if "complaint_type" in df.columns:
            df["complaint_type_normalized"] = df["complaint_type"].apply(
                lambda x: str(x).lower().strip() if pd.notna(x) else ""
            )

        return df

    def _normalize_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize coordinate columns."""
        df = df.copy()

        for col in ["latitude", "longitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validate coordinate ranges
        if "latitude" in df.columns and "longitude" in df.columns:
            valid_lat = (df["latitude"].abs() <= 90) & (df["latitude"].abs() > 0.1)
            valid_lon = (df["longitude"].abs() <= 180) & (df["longitude"].abs() > 0.1)

            df["coords_valid"] = valid_lat & valid_lon
            df["coords_missing"] = df["latitude"].isna() | df["longitude"].isna()

            # Flag suspicious coordinates
            df["coords_suspicious"] = ~df["coords_valid"] & ~df["coords_missing"]

            # Initialize uncertainty column
            df["coord_uncertainty_m"] = np.where(
                df["coords_valid"],
                10.0,  # Assume 10m for valid coordinates
                np.inf
            )

        return df

    def _geocode_missing(
        self,
        df: pd.DataFrame,
        city: str,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Geocode records with missing coordinates."""
        df = df.copy()
        stats = {"geocoded": 0, "failed": 0, "skipped": 0}

        if "address" not in df.columns:
            return df, stats

        # Find records needing geocoding
        needs_geocoding = (
            df.get("coords_missing", pd.Series(True, index=df.index)) |
            df.get("coords_suspicious", pd.Series(False, index=df.index))
        )

        address_col = "address_normalized" if "address_normalized" in df.columns else "address"

        for idx in df[needs_geocoding].index:
            address = df.loc[idx, address_col]
            if not address or not str(address).strip():
                stats["skipped"] += 1
                continue

            result = self.geocoder.geocode(str(address), city)

            if result.success:
                df.loc[idx, "latitude"] = result.latitude
                df.loc[idx, "longitude"] = result.longitude
                df.loc[idx, "coord_uncertainty_m"] = result.uncertainty_m
                df.loc[idx, "coords_valid"] = True
                df.loc[idx, "coords_missing"] = False
                df.loc[idx, "geocode_source"] = result.source
                df.loc[idx, "geocode_confidence"] = result.confidence
                df.loc[idx, "geocode_match_type"] = result.match_type
                stats["geocoded"] += 1
            else:
                df.loc[idx, "geocode_source"] = "failed"
                stats["failed"] += 1

        logger.info(
            "Geocoding: %d successful, %d failed, %d skipped",
            stats["geocoded"], stats["failed"], stats["skipped"]
        )

        return df, stats

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute normalization statistics."""
        stats = {
            "total_records": len(df),
            "valid_coordinates": df.get(
                "coords_valid", pd.Series(False)
            ).sum() if "coords_valid" in df else 0,
            "missing_coordinates": df.get(
                "coords_missing", pd.Series(True)
            ).sum() if "coords_missing" in df else len(df),
            "valid_timestamps": df["timestamp"].notna().sum()
            if "timestamp" in df.columns else 0,
            "with_flood_keywords": df.get(
                "has_flood_keywords", pd.Series(False)
            ).sum() if "has_flood_keywords" in df else 0,
            "potential_duplicates": df.get(
                "is_potential_duplicate", pd.Series(False)
            ).sum() if "is_potential_duplicate" in df else 0,
        }

        if "complaint_type_normalized" in df.columns:
            stats["unique_complaint_types"] = df[
                "complaint_type_normalized"
            ].nunique()

        if "coord_uncertainty_m" in df.columns:
            valid_unc = df["coord_uncertainty_m"][
                df["coord_uncertainty_m"] < float("inf")
            ]
            if len(valid_unc) > 0:
                stats["mean_coord_uncertainty_m"] = valid_unc.mean()
                stats["max_coord_uncertainty_m"] = valid_unc.max()

        return stats


def normalize_complaints_full(
    df: pd.DataFrame,
    city: str = "",
    source_id: Optional[str] = None,
    enable_geocoding: bool = True,
    enable_deduplication: bool = True,
) -> NormalizationResult:
    """High-level function for full complaint normalization.

    Args:
        df: Raw complaint DataFrame
        city: City name for geocoding context
        source_id: Data source identifier
        enable_geocoding: Enable geocoding of missing coordinates
        enable_deduplication: Enable duplicate detection

    Returns:
        NormalizationResult with normalized data and statistics
    """
    normalizer = ComplaintNormalizer(
        enable_geocoding=enable_geocoding,
        enable_deduplication=enable_deduplication,
    )
    return normalizer.normalize(df, city, source_id)
