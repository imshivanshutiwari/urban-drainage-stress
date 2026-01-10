"""Data registry for tracking acquired datasets (Prompt-11)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.config.paths import DATA_DIR

logger = logging.getLogger(__name__)

REGISTRY_PATH = DATA_DIR / "registry.json"


@dataclass
class RegistryEntry:
    name: str
    path: str
    source: str
    downloaded_at: str
    sha256: Optional[str]
    notes: str = ""


class DataRegistry:
    """Lightweight registry that records dataset provenance and hashes."""

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self.path = path
        self._entries: Dict[str, RegistryEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            for name, rec in data.items():
                self._entries[name] = RegistryEntry(**rec)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read registry: %s", exc)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in self._entries.items()}
        self.path.write_text(json.dumps(payload, indent=2))

    def record(
        self,
        name: str,
        path: Path,
        source: str,
        sha256: Optional[str],
        notes: str = "",
    ) -> None:
        entry = RegistryEntry(
            name=name,
            path=str(path),
            source=source,
            downloaded_at=datetime.utcnow().isoformat(),
            sha256=sha256,
            notes=notes,
        )
        self._entries[name] = entry
        self._save()

    def get(self, name: str) -> Optional[RegistryEntry]:
        return self._entries.get(name)

    def all(self) -> Dict[str, RegistryEntry]:
        return dict(self._entries)
