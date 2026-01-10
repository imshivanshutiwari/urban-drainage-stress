"""Time utilities for event windowing and validation."""
from datetime import datetime, timedelta
from typing import Iterable, List


def is_monotonic(timestamps: Iterable[datetime]) -> bool:
    """Check monotonicity; returns False if any decrease is found."""
    prev = None
    for ts in timestamps:
        if prev and ts < prev:
            return False
        prev = ts
    return True


def build_event_windows(start: datetime, end: datetime, step_minutes: int) -> List[tuple]:
    """Return list of (window_start, window_end) tuples."""
    step = timedelta(minutes=step_minutes)
    windows = []
    current = start
    while current < end:
        windows.append((current, min(end, current + step)))
        current += step
    return windows
