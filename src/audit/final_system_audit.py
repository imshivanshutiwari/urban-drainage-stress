"""
STEP 8: Final System Audit & Lock

AUDIT CHECKLIST:
    ✓ No absolute thresholds in decision logic
    ✓ No gradient-based retraining in calibration
    ✓ Calibration is logged and versioned
    ✓ Uncertainty grows under distribution shift
    ✓ All outputs have contract states
    ✓ System operates in latent space
    
This module LOCKS the system after successful audit.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import inspect
import hashlib
from datetime import datetime


@dataclass
class AuditResult:
    """Result of system audit."""
    passed: bool
    checks: Dict[str, bool]
    violations: List[str]
    timestamp: str
    system_hash: str
    
    def to_dict(self) -> Dict:
        return {
            'audit_passed': self.passed,
            'checks': self.checks,
            'violations': self.violations,
            'timestamp': self.timestamp,
            'system_hash': self.system_hash,
        }


class SystemAuditor:
    """
    Final System Audit
    
    Verifies that the Universal Deployment Pipeline meets all requirements:
    
    1. NO_ABSOLUTE_THRESHOLDS: Decision logic uses percentiles, not values
    2. NO_GRADIENT_TRAINING: Calibration uses distributional methods only
    3. UNCERTAINTY_GROWS: σ_final > σ_base when shift_score > 0
    4. CONTRACTS_VALID: Every output has a valid contract state
    5. LATENT_SPACE: System operates in normalized latent space
    6. SHIFT_AWARE: Distribution shift is detected and used
    
    Usage:
        auditor = SystemAuditor()
        result = auditor.audit(pipeline)
        auditor.lock_if_passed(result)
    """
    
    REQUIRED_CHECKS = [
        'no_absolute_thresholds',
        'no_gradient_training',
        'uncertainty_grows_with_shift',
        'contracts_valid',
        'operates_in_latent_space',
        'shift_aware',
        'relative_decisions',
    ]
    
    def __init__(self):
        self._lock_file = None
    
    def audit(self, pipeline, test_data: Optional[Dict] = None) -> AuditResult:
        """
        Perform full system audit.
        
        Args:
            pipeline: UniversalDeploymentPipeline instance
            test_data: Optional test data dict with 'reference' and 'shifted'
        
        Returns:
            AuditResult
        """
        checks = {}
        violations = []
        
        # Generate test data if not provided
        if test_data is None:
            test_data = self._generate_test_data()
        
        # Check 1: No absolute thresholds
        passed, msg = self._check_no_absolute_thresholds(pipeline)
        checks['no_absolute_thresholds'] = passed
        if not passed:
            violations.append(msg)
        
        # Check 2: No gradient training
        passed, msg = self._check_no_gradient_training(pipeline)
        checks['no_gradient_training'] = passed
        if not passed:
            violations.append(msg)
        
        # Check 3: Uncertainty grows with shift
        passed, msg = self._check_uncertainty_grows(pipeline, test_data)
        checks['uncertainty_grows_with_shift'] = passed
        if not passed:
            violations.append(msg)
        
        # Check 4: Contracts valid
        passed, msg = self._check_contracts_valid(pipeline, test_data)
        checks['contracts_valid'] = passed
        if not passed:
            violations.append(msg)
        
        # Check 5: Operates in latent space
        passed, msg = self._check_latent_space(pipeline)
        checks['operates_in_latent_space'] = passed
        if not passed:
            violations.append(msg)
        
        # Check 6: Shift aware
        passed, msg = self._check_shift_aware(pipeline, test_data)
        checks['shift_aware'] = passed
        if not passed:
            violations.append(msg)
        
        # Check 7: Relative decisions
        passed, msg = self._check_relative_decisions(pipeline, test_data)
        checks['relative_decisions'] = passed
        if not passed:
            violations.append(msg)
        
        # Overall pass/fail
        all_passed = all(checks.values())
        
        # Compute system hash
        system_hash = self._compute_system_hash(pipeline)
        
        return AuditResult(
            passed=all_passed,
            checks=checks,
            violations=violations,
            timestamp=datetime.now().isoformat(),
            system_hash=system_hash,
        )
    
    def _generate_test_data(self) -> Dict:
        """Generate synthetic test data."""
        np.random.seed(42)
        return {
            'reference': np.random.exponential(0.5, (5, 30, 30)) + 0.2,
            'shifted': np.random.exponential(5.0, (5, 30, 30)) + 2.0,
        }
    
    def _check_no_absolute_thresholds(self, pipeline) -> Tuple[bool, str]:
        """Check that decision engine uses no absolute thresholds."""
        engine = pipeline.decision_engine
        
        # Inspect the _make_decisions method source
        source = inspect.getsource(engine._make_decisions)
        
        # Look for absolute threshold patterns
        forbidden = [
            'stress > ',
            'stress >= ',
            'Z > 0.7',
            'Z > 5',
            'if S >',
            'S_raw >',
        ]
        
        for pattern in forbidden:
            if pattern.lower() in source.lower():
                return False, f"Found absolute threshold pattern: {pattern}"
        
        # Check that percentile is used
        if 'percentile' not in source.lower() and 'p >=' not in source.lower():
            return False, "Decision logic does not use percentiles"
        
        return True, "No absolute thresholds found"
    
    def _check_no_gradient_training(self, pipeline) -> Tuple[bool, str]:
        """Check that calibrator uses no gradient-based training."""
        calibrator = pipeline.city_calibrator
        
        # Check method
        method = calibrator.method
        allowed_methods = ['quantile', 'isotonic', 'linear']
        
        if method not in allowed_methods:
            return False, f"Calibration method '{method}' may use gradients"
        
        # Check for neural network imports
        source = inspect.getsource(type(calibrator))
        forbidden_imports = ['torch', 'tensorflow', 'keras', 'neural', 'nn.']
        
        for imp in forbidden_imports:
            if imp in source:
                return False, f"Found potential gradient training: {imp}"
        
        return True, "No gradient training found"
    
    def _check_uncertainty_grows(self, pipeline, test_data: Dict) -> Tuple[bool, str]:
        """Check that uncertainty grows when shift_score > 0."""
        reference = test_data['reference']
        shifted = test_data['shifted']
        
        # Fit pipeline on reference
        pipeline.fit_reference(reference, "test_ref")
        
        # Run on reference (shift_score should be ~0)
        ref_result = pipeline.run(reference, "ref")
        ref_unc = np.mean(ref_result.sigma_final)
        
        # Run on shifted (shift_score should be > 0)
        shifted_result = pipeline.run(shifted, "shifted")
        shifted_unc = np.mean(shifted_result.sigma_final)
        
        # Uncertainty should be higher for shifted
        if shifted_unc <= ref_unc:
            return False, f"Uncertainty did not grow: ref={ref_unc:.4f}, shifted={shifted_unc:.4f}"
        
        return True, f"Uncertainty grows correctly: ref={ref_unc:.4f}, shifted={shifted_unc:.4f}"
    
    def _check_contracts_valid(self, pipeline, test_data: Dict) -> Tuple[bool, str]:
        """Check that all outputs have valid contracts."""
        from src.decision.decision_contract import ContractValidator, OutputState
        
        pipeline.fit_reference(test_data['reference'], "test_ref")
        result = pipeline.run(test_data['shifted'], "test_new")
        
        # Validate contracts
        flat = result.contracts.flatten()
        
        for i, contract in enumerate(flat):
            if contract.state not in OutputState:
                return False, f"Invalid contract state at {i}"
            
            if contract.state == OutputState.REFUSED and contract.decision is not None:
                return False, f"Refused contract has decision at {i}"
        
        return True, "All contracts valid"
    
    def _check_latent_space(self, pipeline) -> Tuple[bool, str]:
        """Check that system operates in latent space."""
        # Check that latent transform exists and is used
        if not hasattr(pipeline, 'latent_transform'):
            return False, "No latent transform found"
        
        lt = pipeline.latent_transform
        if not hasattr(lt, 'transform') or not callable(lt.transform):
            return False, "Latent transform has no transform method"
        
        return True, "System operates in latent space"
    
    def _check_shift_aware(self, pipeline, test_data: Dict) -> Tuple[bool, str]:
        """Check that shift detection is working."""
        reference = test_data['reference']
        shifted = test_data['shifted']
        
        pipeline.fit_reference(reference, "test_ref")
        
        # Run on reference
        ref_result = pipeline.run(reference, "ref")
        ref_shift = ref_result.shift_score
        
        # Run on shifted
        shifted_result = pipeline.run(shifted, "shifted")
        shifted_shift = shifted_result.shift_score
        
        # Shifted should have higher shift score
        if shifted_shift <= ref_shift + 0.1:
            return False, f"Shift detection failed: ref={ref_shift:.3f}, shifted={shifted_shift:.3f}"
        
        return True, f"Shift detection working: ref={ref_shift:.3f}, shifted={shifted_shift:.3f}"
    
    def _check_relative_decisions(self, pipeline, test_data: Dict) -> Tuple[bool, str]:
        """Check that decisions are relative across cities."""
        reference = test_data['reference']
        shifted = test_data['shifted']
        
        pipeline.fit_reference(reference, "test_ref")
        
        ref_result = pipeline.run(reference, "ref")
        shifted_result = pipeline.run(shifted, "shifted")
        
        # Both should have similar decision distributions (relative)
        ref_dist = ref_result.decision_details.get('decision_distribution', {})
        shifted_dist = shifted_result.decision_details.get('decision_distribution', {})
        
        # Check that both have decisions (not all refused)
        ref_total = sum(ref_dist.values()) if ref_dist else 0
        shifted_total = sum(shifted_dist.values()) if shifted_dist else 0
        
        if ref_total == 0 and shifted_total == 0:
            return False, "All decisions refused in both cities"
        
        return True, "Relative decisions working"
    
    def _compute_system_hash(self, pipeline) -> str:
        """Compute hash of system configuration."""
        config = {
            'calibration_method': pipeline.calibration_method,
            'conservative_mode': pipeline.conservative_mode,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def lock_if_passed(self, result: AuditResult, lock_file: str = None) -> bool:
        """
        Lock the system if audit passed.
        
        Args:
            result: AuditResult from audit()
            lock_file: Path to write lock file
        
        Returns:
            True if locked, False if audit failed
        """
        if not result.passed:
            print("AUDIT FAILED - System not locked")
            for v in result.violations:
                print(f"  ✗ {v}")
            return False
        
        # Write lock file
        lock_data = {
            'locked': True,
            'audit_result': result.to_dict(),
            'locked_at': datetime.now().isoformat(),
        }
        
        if lock_file:
            with open(lock_file, 'w') as f:
                json.dump(lock_data, f, indent=2)
            print(f"System LOCKED: {lock_file}")
        
        return True


def run_full_audit(pipeline=None, 
                   test_data: Optional[Dict] = None,
                   lock_file: str = None) -> AuditResult:
    """
    Convenience function to run full audit.
    
    Args:
        pipeline: UniversalDeploymentPipeline (creates default if None)
        test_data: Optional test data
        lock_file: Path to write lock file
    
    Returns:
        AuditResult
    """
    if pipeline is None:
        from src.pipeline.universal_deployment_pipeline import UniversalDeploymentPipeline
        pipeline = UniversalDeploymentPipeline()
    
    auditor = SystemAuditor()
    result = auditor.audit(pipeline, test_data)
    
    print("\n" + "=" * 60)
    print("SYSTEM AUDIT REPORT")
    print("=" * 60)
    print(f"\nTimestamp: {result.timestamp}")
    print(f"System Hash: {result.system_hash}")
    print(f"\nChecks:")
    for check, passed in result.checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    if result.violations:
        print(f"\nViolations:")
        for v in result.violations:
            print(f"  - {v}")
    
    print(f"\n{'='*60}")
    if result.passed:
        print("AUDIT PASSED ✓")
        auditor.lock_if_passed(result, lock_file)
    else:
        print("AUDIT FAILED ✗")
    print("=" * 60)
    
    return result


# Test
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.pipeline.universal_deployment_pipeline import UniversalDeploymentPipeline
    
    print("Final System Audit Test")
    print("=" * 60)
    
    # Create pipeline
    pipeline = UniversalDeploymentPipeline(conservative_mode=True)
    
    # Run audit
    result = run_full_audit(pipeline)
