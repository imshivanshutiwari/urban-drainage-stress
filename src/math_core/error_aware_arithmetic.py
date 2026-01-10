"""
Error-Aware Arithmetic Module

Implements explicit error propagation, guarded division, and epsilon-safe operations.
Tracks numerical error accumulation throughout computations.

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional
import warnings

# Machine epsilon for float64
EPSILON = np.finfo(np.float64).eps
SQRT_EPSILON = np.sqrt(EPSILON)
LOG_EPSILON = np.log(EPSILON)

# Safe bounds for numerical operations
MIN_POSITIVE = np.finfo(np.float64).tiny
MAX_FINITE = np.finfo(np.float64).max
MIN_LOG_ARG = MIN_POSITIVE
MAX_EXP_ARG = 709.0  # log(MAX_FINITE)


@dataclass
class ErrorValue:
    """
    A value paired with its absolute and relative error bounds.
    
    Implements error propagation for all arithmetic operations following
    IEEE 754 floating point error analysis.
    
    Attributes:
        value: The computed value
        abs_error: Absolute error bound (always non-negative)
        rel_error: Relative error bound (always non-negative)
        ulp_error: Error in units of last place
    """
    value: float
    abs_error: float = 0.0
    rel_error: float = 0.0
    ulp_error: float = 0.0
    
    def __post_init__(self):
        """Ensure error bounds are non-negative."""
        self.abs_error = abs(self.abs_error)
        self.rel_error = abs(self.rel_error)
        self.ulp_error = abs(self.ulp_error)
        
        # Update relative error from absolute if value is non-zero
        if self.value != 0 and self.abs_error > 0:
            computed_rel = self.abs_error / abs(self.value)
            self.rel_error = max(self.rel_error, computed_rel)
    
    @classmethod
    def from_measurement(cls, value: float, measurement_error: float) -> 'ErrorValue':
        """Create ErrorValue from a measurement with known error."""
        return cls(
            value=value,
            abs_error=measurement_error,
            rel_error=measurement_error / abs(value) if value != 0 else float('inf'),
            ulp_error=measurement_error / np.spacing(value) if value != 0 else 0.0
        )
    
    @classmethod
    def from_exact(cls, value: float) -> 'ErrorValue':
        """Create ErrorValue from an exact (integer or known) value."""
        return cls(value=value, abs_error=0.0, rel_error=0.0, ulp_error=0.0)
    
    @classmethod
    def from_float(cls, value: float) -> 'ErrorValue':
        """Create ErrorValue from a float with inherent representation error."""
        ulp = np.spacing(value) if np.isfinite(value) else 0.0
        return cls(
            value=value,
            abs_error=0.5 * ulp,  # Rounding to nearest
            rel_error=0.5 * EPSILON if value != 0 else 0.0,
            ulp_error=0.5
        )
    
    def __add__(self, other: Union['ErrorValue', float]) -> 'ErrorValue':
        """Addition with error propagation: |δ(a+b)| ≤ |δa| + |δb| + ε|a+b|"""
        if isinstance(other, (int, float)):
            other = ErrorValue.from_float(float(other))
        
        result = self.value + other.value
        
        # Absolute error: sum of absolute errors plus rounding error
        abs_err = self.abs_error + other.abs_error + EPSILON * abs(result)
        
        # Relative error for addition requires care near cancellation
        if result != 0:
            rel_err = abs_err / abs(result)
        else:
            rel_err = float('inf') if abs_err > 0 else 0.0
        
        return ErrorValue(result, abs_err, rel_err)
    
    def __radd__(self, other: float) -> 'ErrorValue':
        return self.__add__(other)
    
    def __sub__(self, other: Union['ErrorValue', float]) -> 'ErrorValue':
        """Subtraction with error propagation - watch for catastrophic cancellation."""
        if isinstance(other, (int, float)):
            other = ErrorValue.from_float(float(other))
        
        result = self.value - other.value
        
        # Absolute error
        abs_err = self.abs_error + other.abs_error + EPSILON * abs(result)
        
        # Check for catastrophic cancellation
        if abs(result) < SQRT_EPSILON * max(abs(self.value), abs(other.value)):
            # Cancellation detected - relative error may be large
            warnings.warn(
                f"Catastrophic cancellation in subtraction: {self.value} - {other.value}",
                RuntimeWarning
            )
        
        rel_err = abs_err / abs(result) if result != 0 else (float('inf') if abs_err > 0 else 0.0)
        
        return ErrorValue(result, abs_err, rel_err)
    
    def __rsub__(self, other: float) -> 'ErrorValue':
        return ErrorValue.from_float(float(other)).__sub__(self)
    
    def __mul__(self, other: Union['ErrorValue', float]) -> 'ErrorValue':
        """Multiplication: |δ(ab)/ab| ≤ |δa/a| + |δb/b| + ε"""
        if isinstance(other, (int, float)):
            other = ErrorValue.from_float(float(other))
        
        result = self.value * other.value
        
        # Relative error: sum of relative errors plus rounding
        rel_err = self.rel_error + other.rel_error + EPSILON
        
        # Absolute error
        abs_err = abs(result) * rel_err
        
        return ErrorValue(result, abs_err, rel_err)
    
    def __rmul__(self, other: float) -> 'ErrorValue':
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['ErrorValue', float]) -> 'ErrorValue':
        """Division with guarded operation and error propagation."""
        if isinstance(other, (int, float)):
            other = ErrorValue.from_float(float(other))
        
        # Guard against division by zero or near-zero
        if abs(other.value) < MIN_POSITIVE:
            warnings.warn(f"Division by near-zero value: {other.value}", RuntimeWarning)
            return ErrorValue(
                np.sign(self.value) * MAX_FINITE if other.value >= 0 else -np.sign(self.value) * MAX_FINITE,
                MAX_FINITE,
                1.0
            )
        
        result = self.value / other.value
        
        # Relative error: sum of relative errors plus rounding
        rel_err = self.rel_error + other.rel_error + EPSILON
        
        # Additional error from denominator uncertainty
        if other.abs_error > 0:
            denom_contrib = other.abs_error / (abs(other.value) - other.abs_error)
            rel_err += denom_contrib
        
        abs_err = abs(result) * rel_err
        
        return ErrorValue(result, abs_err, rel_err)
    
    def __rtruediv__(self, other: float) -> 'ErrorValue':
        return ErrorValue.from_float(float(other)).__truediv__(self)
    
    def __neg__(self) -> 'ErrorValue':
        return ErrorValue(-self.value, self.abs_error, self.rel_error, self.ulp_error)
    
    def __abs__(self) -> 'ErrorValue':
        return ErrorValue(abs(self.value), self.abs_error, self.rel_error, self.ulp_error)
    
    def __pow__(self, n: float) -> 'ErrorValue':
        """Power with error propagation: δ(x^n)/x^n ≈ n * δx/x"""
        if self.value <= 0 and n != int(n):
            warnings.warn("Non-integer power of non-positive number", RuntimeWarning)
            return ErrorValue(np.nan, np.nan, np.nan)
        
        result = self.value ** n
        rel_err = abs(n) * self.rel_error + EPSILON
        abs_err = abs(result) * rel_err
        
        return ErrorValue(result, abs_err, rel_err)
    
    def sqrt(self) -> 'ErrorValue':
        """Square root with error propagation: δ(√x)/√x = δx/(2x)"""
        if self.value < 0:
            warnings.warn("Square root of negative number", RuntimeWarning)
            return ErrorValue(np.nan, np.nan, np.nan)
        
        if self.value < MIN_POSITIVE:
            return ErrorValue(0.0, np.sqrt(self.abs_error), float('inf'))
        
        result = np.sqrt(self.value)
        rel_err = 0.5 * self.rel_error + EPSILON
        abs_err = result * rel_err
        
        return ErrorValue(result, abs_err, rel_err)
    
    def log(self) -> 'ErrorValue':
        """Natural logarithm with error propagation: δ(ln x) = δx/x"""
        if self.value <= 0:
            warnings.warn("Logarithm of non-positive number", RuntimeWarning)
            return ErrorValue(-np.inf if self.value == 0 else np.nan, np.nan, np.nan)
        
        if self.value < MIN_LOG_ARG:
            return ErrorValue(LOG_EPSILON, self.abs_error / MIN_LOG_ARG, float('inf'))
        
        result = np.log(self.value)
        abs_err = self.rel_error + EPSILON  # δ(ln x) = δx/x ≈ rel_error
        rel_err = abs_err / abs(result) if result != 0 else float('inf')
        
        return ErrorValue(result, abs_err, rel_err)
    
    def exp(self) -> 'ErrorValue':
        """Exponential with error propagation: δ(e^x)/e^x = δx"""
        if self.value > MAX_EXP_ARG:
            warnings.warn(f"Exponential overflow: exp({self.value})", RuntimeWarning)
            return ErrorValue(MAX_FINITE, MAX_FINITE, 1.0)
        
        result = np.exp(self.value)
        rel_err = self.abs_error + EPSILON  # δ(e^x)/e^x = δx
        abs_err = result * rel_err
        
        return ErrorValue(result, abs_err, rel_err)
    
    def confidence_interval(self, sigma: float = 1.0) -> Tuple[float, float]:
        """Return confidence interval as (lower, upper)."""
        half_width = sigma * self.abs_error
        return (self.value - half_width, self.value + half_width)
    
    def is_reliable(self, max_rel_error: float = 0.01) -> bool:
        """Check if the value has acceptable relative error."""
        return self.rel_error <= max_rel_error
    
    def __repr__(self) -> str:
        return f"ErrorValue({self.value:.6g} ± {self.abs_error:.2g}, rel={self.rel_error:.2g})"


# =============================================================================
# ARRAY-LEVEL ERROR-AWARE OPERATIONS
# =============================================================================

@dataclass
class ErrorArray:
    """
    NumPy array with tracked error bounds.
    
    Implements vectorized error-aware arithmetic for efficient computation
    on large spatial grids.
    """
    values: np.ndarray
    abs_errors: np.ndarray
    rel_errors: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize relative errors if not provided."""
        self.values = np.asarray(self.values, dtype=np.float64)
        self.abs_errors = np.asarray(self.abs_errors, dtype=np.float64)
        
        if self.rel_errors is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.rel_errors = np.where(
                    self.values != 0,
                    self.abs_errors / np.abs(self.values),
                    np.where(self.abs_errors > 0, np.inf, 0.0)
                )
        else:
            self.rel_errors = np.asarray(self.rel_errors, dtype=np.float64)
    
    @classmethod
    def from_array(cls, values: np.ndarray) -> 'ErrorArray':
        """Create ErrorArray with inherent floating point error."""
        values = np.asarray(values, dtype=np.float64)
        abs_errors = 0.5 * np.spacing(values)
        return cls(values, abs_errors)
    
    @classmethod
    def from_measurements(cls, values: np.ndarray, errors: np.ndarray) -> 'ErrorArray':
        """Create ErrorArray from measurements with known errors."""
        return cls(np.asarray(values), np.asarray(errors))
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.values.shape
    
    def __add__(self, other: Union['ErrorArray', np.ndarray, float]) -> 'ErrorArray':
        if isinstance(other, ErrorArray):
            result = self.values + other.values
            abs_err = self.abs_errors + other.abs_errors + EPSILON * np.abs(result)
        else:
            other = np.asarray(other)
            result = self.values + other
            abs_err = self.abs_errors + EPSILON * np.abs(result)
        return ErrorArray(result, abs_err)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Union['ErrorArray', np.ndarray, float]) -> 'ErrorArray':
        if isinstance(other, ErrorArray):
            result = self.values - other.values
            abs_err = self.abs_errors + other.abs_errors + EPSILON * np.abs(result)
        else:
            other = np.asarray(other)
            result = self.values - other
            abs_err = self.abs_errors + EPSILON * np.abs(result)
        return ErrorArray(result, abs_err)
    
    def __mul__(self, other: Union['ErrorArray', np.ndarray, float]) -> 'ErrorArray':
        if isinstance(other, ErrorArray):
            result = self.values * other.values
            rel_err = self.rel_errors + other.rel_errors + EPSILON
        else:
            other = np.asarray(other)
            result = self.values * other
            rel_err = self.rel_errors + EPSILON
        abs_err = np.abs(result) * rel_err
        return ErrorArray(result, abs_err, rel_err)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['ErrorArray', np.ndarray, float]) -> 'ErrorArray':
        if isinstance(other, ErrorArray):
            # Guard against division by zero
            safe_denom = np.where(np.abs(other.values) < MIN_POSITIVE, MIN_POSITIVE, other.values)
            result = self.values / safe_denom
            rel_err = self.rel_errors + other.rel_errors + EPSILON
        else:
            other = np.asarray(other)
            safe_denom = np.where(np.abs(other) < MIN_POSITIVE, MIN_POSITIVE, other)
            result = self.values / safe_denom
            rel_err = self.rel_errors + EPSILON
        abs_err = np.abs(result) * rel_err
        return ErrorArray(result, abs_err, rel_err)
    
    def sqrt(self) -> 'ErrorArray':
        """Element-wise square root with error propagation."""
        safe_values = np.maximum(self.values, 0.0)
        result = np.sqrt(safe_values)
        rel_err = 0.5 * self.rel_errors + EPSILON
        abs_err = result * rel_err
        return ErrorArray(result, abs_err, rel_err)
    
    def log(self) -> 'ErrorArray':
        """Element-wise natural log with error propagation."""
        safe_values = np.maximum(self.values, MIN_LOG_ARG)
        result = np.log(safe_values)
        abs_err = self.rel_errors + EPSILON
        return ErrorArray(result, abs_err)
    
    def exp(self) -> 'ErrorArray':
        """Element-wise exponential with error propagation."""
        safe_values = np.minimum(self.values, MAX_EXP_ARG)
        result = np.exp(safe_values)
        rel_err = self.abs_errors + EPSILON
        abs_err = result * rel_err
        return ErrorArray(result, abs_err, rel_err)
    
    def sum(self, axis: Optional[int] = None) -> Union['ErrorArray', 'ErrorValue']:
        """Sum with accumulated error."""
        result = self.values.sum(axis=axis)
        # Error accumulates as sqrt(sum of squared errors) for independent errors
        abs_err = np.sqrt((self.abs_errors ** 2).sum(axis=axis))
        
        if axis is None:
            return ErrorValue(float(result), float(abs_err))
        return ErrorArray(result, abs_err)
    
    def mean(self, axis: Optional[int] = None) -> Union['ErrorArray', 'ErrorValue']:
        """Mean with error propagation."""
        n = self.values.size if axis is None else self.values.shape[axis]
        result = self.values.mean(axis=axis)
        # Standard error of mean
        abs_err = np.sqrt((self.abs_errors ** 2).sum(axis=axis)) / n
        
        if axis is None:
            return ErrorValue(float(result), float(abs_err))
        return ErrorArray(result, abs_err)
    
    def max_error(self) -> float:
        """Return maximum absolute error."""
        return float(np.nanmax(self.abs_errors))
    
    def max_rel_error(self) -> float:
        """Return maximum relative error (excluding inf)."""
        finite_rel = self.rel_errors[np.isfinite(self.rel_errors)]
        return float(np.nanmax(finite_rel)) if len(finite_rel) > 0 else np.inf
    
    def reliability_mask(self, max_rel_error: float = 0.01) -> np.ndarray:
        """Return boolean mask of reliable values."""
        return self.rel_errors <= max_rel_error


# =============================================================================
# SAFE SCALAR OPERATIONS
# =============================================================================

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                fill_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safe division with explicit handling of zeros and near-zeros.
    
    Returns:
        (result, validity_mask) where validity_mask indicates reliable values
    """
    denom = np.asarray(denominator, dtype=np.float64)
    numer = np.asarray(numerator, dtype=np.float64)
    
    # Identify safe divisions
    safe_mask = np.abs(denom) >= SQRT_EPSILON * np.maximum(np.abs(numer), 1.0)
    
    # Perform safe division
    result = np.where(safe_mask, numer / np.where(safe_mask, denom, 1.0), fill_value)
    
    return result, safe_mask


def safe_log(x: np.ndarray, min_arg: float = MIN_LOG_ARG) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safe logarithm with clamping and validity tracking.
    
    Returns:
        (result, validity_mask)
    """
    x = np.asarray(x, dtype=np.float64)
    valid_mask = x > min_arg
    safe_x = np.maximum(x, min_arg)
    result = np.log(safe_x)
    
    return result, valid_mask


def safe_sqrt(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safe square root with handling of negative values.
    
    Returns:
        (result, validity_mask)
    """
    x = np.asarray(x, dtype=np.float64)
    valid_mask = x >= 0
    safe_x = np.maximum(x, 0.0)
    result = np.sqrt(safe_x)
    
    return result, valid_mask


def safe_exp(x: np.ndarray, max_arg: float = MAX_EXP_ARG) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safe exponential with overflow prevention.
    
    Returns:
        (result, validity_mask)
    """
    x = np.asarray(x, dtype=np.float64)
    valid_mask = x <= max_arg
    safe_x = np.minimum(x, max_arg)
    result = np.exp(safe_x)
    
    return result, valid_mask


def propagate_error(
    func_values: np.ndarray,
    jacobian: np.ndarray,
    input_errors: np.ndarray
) -> np.ndarray:
    """
    General first-order error propagation through a function.
    
    For f(x), the error in f is approximately |df/dx| * error(x).
    For multivariate, uses the Jacobian.
    
    Args:
        func_values: Output values (for reference)
        jacobian: Jacobian matrix df/dx, shape (..., n_outputs, n_inputs)
        input_errors: Errors in input values, shape (..., n_inputs)
    
    Returns:
        Propagated errors in outputs
    """
    # For scalar functions: error = |derivative| * input_error
    if jacobian.ndim == input_errors.ndim:
        return np.abs(jacobian) * input_errors
    
    # For vector functions: error = sqrt(sum_i (df/dx_i * error_i)^2)
    # Assuming independent input errors
    jacobian = np.asarray(jacobian)
    input_errors = np.asarray(input_errors)
    
    # Broadcast and compute
    squared_contrib = (jacobian * input_errors[..., np.newaxis, :]) ** 2
    output_variance = squared_contrib.sum(axis=-1)
    
    return np.sqrt(output_variance)


# =============================================================================
# CONDITION NUMBER TRACKING
# =============================================================================

def condition_number_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Estimate condition number for linear combination a*x + b*y.
    
    High condition number indicates sensitivity to input errors.
    """
    a, b = np.asarray(a), np.asarray(b)
    norm_a = np.linalg.norm(a.ravel())
    norm_b = np.linalg.norm(b.ravel())
    
    if norm_a + norm_b < EPSILON:
        return 0.0
    
    return (norm_a + norm_b) / max(EPSILON, abs(norm_a - norm_b))


def track_accumulated_error(operations_log: list) -> float:
    """
    Estimate total accumulated error from a sequence of operations.
    
    Args:
        operations_log: List of (operation_type, relative_error) tuples
    
    Returns:
        Estimated total relative error bound
    """
    total_rel_error = 0.0
    
    for op_type, rel_error in operations_log:
        if op_type in ('add', 'sub'):
            # Additive error accumulation
            total_rel_error = total_rel_error + rel_error
        elif op_type in ('mul', 'div'):
            # Multiplicative error accumulation
            total_rel_error = total_rel_error + rel_error
        elif op_type == 'sqrt':
            total_rel_error = 0.5 * total_rel_error + rel_error
        elif op_type == 'exp':
            # Exponential amplifies absolute error
            total_rel_error = total_rel_error + rel_error
        elif op_type == 'log':
            # Log reduces relative to absolute
            total_rel_error = total_rel_error + rel_error
    
    return total_rel_error
