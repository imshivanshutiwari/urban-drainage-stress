"""Central Data Registry Module.

This module provides the backbone for data-aware pipeline execution:
- Central registry for all datasets
- Integrity checks for scientific validity
- Run-readiness gate for pipeline control
"""

from src.data_registry.registry import (
    DataRegistry,
    DatasetMetadata,
    DatasetType,
    AutomationMode,
    ValidationStatus,
)
from src.data_registry.integrity_checks import (
    IntegrityChecker,
    CheckResult,
    CheckStatus,
)
from src.data_registry.run_gate import (
    RunGate,
    RunDecision,
    GateResult,
)

__all__ = [
    "DataRegistry",
    "DatasetMetadata",
    "DatasetType",
    "AutomationMode",
    "ValidationStatus",
    "IntegrityChecker",
    "CheckResult",
    "CheckStatus",
    "RunGate",
    "RunDecision",
    "GateResult",
]
