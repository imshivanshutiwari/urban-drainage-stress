"""Guards module for scale-free DL enforcement."""
from .input_sanity_checks import (
    validate_model_inputs,
    assert_scale_free,
    InputSanityConfig,
    SanityCheckResult,
)
from .uncertainty_guards import (
    validate_uncertainty_additive,
    assert_uncertainty_additive,
    combine_uncertainties_safe,
    UncertaintyGuardConfig,
    UncertaintyGuardResult,
)

__all__ = [
    'validate_model_inputs',
    'assert_scale_free',
    'InputSanityConfig',
    'SanityCheckResult',
    'validate_uncertainty_additive',
    'assert_uncertainty_additive',
    'combine_uncertainties_safe',
    'UncertaintyGuardConfig',
    'UncertaintyGuardResult',
]
