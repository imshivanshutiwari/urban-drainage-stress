"""Path configuration for the drainage stress project.

Centralizes all filesystem paths to avoid hardcoding across modules.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "results"


def ensure_directories() -> None:
    """Create expected directories if they do not exist."""
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR, OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
