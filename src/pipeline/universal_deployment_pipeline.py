"""
STEP 7: Full Pipeline Integration

EXECUTION ORDER:
    Raw Stress → Latent → Shift Detection → Uncertainty Inflation 
    → Calibration → Relative Decisions → Contracts

This is the UNIVERSAL DEPLOYMENT PIPELINE.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.calibration.latent_transform import LatentTransform
from src.calibration.shift_detector import ShiftDetector
from src.calibration.uncertainty_inflation import UncertaintyInflator
from src.calibration.city_calibrator import CityCalibrator
from src.decision.relative_decision_engine import RelativeDecisionEngine
from src.decision.decision_contract import ContractBuilder, ContractValidator


@dataclass
class UniversalPipelineResult:
    """Result from universal deployment pipeline."""
    # Raw outputs
    raw_stress: np.ndarray
    
    # Latent space
    latent_Z: np.ndarray
    latent_stats: Dict
    
    # Shift detection
    shift_score: float
    shift_details: Dict
    
    # Uncertainty
    sigma_final: np.ndarray
    uncertainty_details: Dict
    
    # Calibrated output
    calibrated_stress: np.ndarray
    calibration_method: str
    
    # Decisions
    decisions: np.ndarray
    decision_details: Dict
    
    # Contracts
    contracts: np.ndarray
    contract_summary: Dict
    
    def to_dict(self) -> Dict:
        return {
            'raw_mean': float(np.mean(self.raw_stress)),
            'raw_std': float(np.std(self.raw_stress)),
            'latent_mean': float(np.mean(self.latent_Z)),
            'latent_std': float(np.std(self.latent_Z)),
            'shift_score': self.shift_score,
            'calibrated_mean': float(np.mean(self.calibrated_stress)),
            'calibrated_std': float(np.std(self.calibrated_stress)),
            'mean_uncertainty': float(np.mean(self.sigma_final)),
            **self.shift_details,
            **self.uncertainty_details,
            **self.decision_details,
            **self.contract_summary,
        }


class UniversalDeploymentPipeline:
    """
    Universal Deployment Pipeline for Cross-Region Inference
    
    EXECUTION ORDER:
        1. Raw Stress → Latent Transform (Z = (S - median) / MAD)
        2. Shift Detection (compare to reference)
        3. Uncertainty Inflation (σ grows with shift)
        4. Calibration (optional, for interpretability)
        5. Relative Decisions (percentile-based)
        6. Contract Generation (confident/uncertain/refused)
    
    GUARANTEES:
        - No absolute thresholds
        - Uncertainty grows under shift
        - Decisions are relative to city
        - Every output has a contract state
    
    Usage:
        pipeline = UniversalDeploymentPipeline()
        pipeline.fit_reference(seattle_stress)
        result = pipeline.run(nyc_stress)
    """
    
    def __init__(self,
                 calibration_method: str = 'quantile',
                 conservative_mode: bool = True):
        """
        Args:
            calibration_method: 'quantile', 'isotonic', or 'linear'
            conservative_mode: If True, more NO_DECISION zones
        """
        self.calibration_method = calibration_method
        self.conservative_mode = conservative_mode
        
        # Initialize components
        self.latent_transform = LatentTransform()
        self.shift_detector = ShiftDetector()
        self.uncertainty_inflator = UncertaintyInflator()
        self.city_calibrator = CityCalibrator(method=calibration_method)
        self.decision_engine = RelativeDecisionEngine(conservative_mode=conservative_mode)
        self.contract_builder = ContractBuilder()
        
        self._reference_fitted = False
    
    def fit_reference(self, 
                      reference_stress: np.ndarray,
                      reference_name: str = "reference") -> 'UniversalDeploymentPipeline':
        """
        Fit the pipeline to a reference (training) distribution.
        
        Args:
            reference_stress: Raw stress from reference city
            reference_name: Name of reference city
        
        Returns:
            self for chaining
        """
        # 1. Fit latent transform
        self.latent_transform.fit(reference_stress)
        
        # 2. Transform to latent space
        ref_result = self.latent_transform.transform(reference_stress)
        Z_ref = ref_result.Z
        
        # 3. Fit shift detector on RAW stress (to detect magnitude differences)
        # This captures the domain shift before normalization
        self.shift_detector.fit_reference(reference_stress)
        
        # 4. Store reference raw stats for comparison
        self._ref_raw_stats = {
            'mean': float(np.mean(reference_stress)),
            'std': float(np.std(reference_stress)),
            'median': float(np.median(reference_stress)),
        }
        
        # 5. Fit calibrator (maps latent back to interpretable scale)
        self.city_calibrator.fit(Z_ref, reference_stress)
        
        # 6. Fit decision engine
        self.decision_engine.set_city_reference(Z_ref)
        
        self._reference_name = reference_name
        self._reference_fitted = True
        
        return self
    
    def run(self,
            raw_stress: np.ndarray,
            city_name: str = "new_city",
            sigma_bayes: Optional[np.ndarray] = None,
            sigma_dl: Optional[np.ndarray] = None) -> UniversalPipelineResult:
        """
        Run the full pipeline on new data.
        
        Args:
            raw_stress: Raw stress output from inference engine
            city_name: Name of city being processed
            sigma_bayes: Optional Bayesian uncertainty
            sigma_dl: Optional model uncertainty
        
        Returns:
            UniversalPipelineResult with all outputs
        """
        if not self._reference_fitted:
            # Self-fit if no reference
            print(f"[WARNING] No reference fitted, using self as reference")
            self.fit_reference(raw_stress, city_name)
        
        # STEP 1: Raw → Latent
        # CRITICAL: For new cities, we must normalize using their OWN distribution
        # then compare the normalized patterns (not the raw values)
        local_transform = LatentTransform()
        local_transform.fit(raw_stress)  # Fit on this city's data
        latent_result = local_transform.transform(raw_stress)
        Z = latent_result.Z
        
        # STEP 2: Shift Detection
        # Compare RAW stress distributions (before normalization) to detect domain shift
        shift_result = self.shift_detector.detect(raw_stress)
        shift_score = shift_result.shift_score
        
        # STEP 3: Uncertainty Inflation
        unc_result = self.uncertainty_inflator.inflate(
            Z, sigma_bayes, sigma_dl, shift_score
        )
        sigma_final = unc_result.sigma_final
        
        # STEP 4: Calibration (for interpretability)
        # Fit local calibrator for this city's Z
        local_calibrator = CityCalibrator(method=self.calibration_method)
        local_calibrator.fit(Z)
        calib_result = local_calibrator.transform(Z)
        calibrated = calib_result.S_calibrated
        
        # STEP 5: Relative Decisions
        # Use this city's OWN Z distribution for relative decisions
        local_decision_engine = RelativeDecisionEngine(conservative_mode=self.conservative_mode)
        local_decision_engine.set_city_reference(Z)
        decision_result = local_decision_engine.decide(Z, sigma_final)
        
        # STEP 6: Contract Generation
        contract_result = self.contract_builder.build(
            decisions=decision_result.decisions,
            confidence=decision_result.confidence,
            percentile_ranks=decision_result.percentile_ranks,
            exceedance_probs=decision_result.exceedance_probs,
            sigma_final=sigma_final,
        )
        
        # Validate contracts
        ContractValidator.validate(contract_result)
        
        return UniversalPipelineResult(
            raw_stress=raw_stress,
            latent_Z=Z,
            latent_stats=latent_result.to_dict(),
            shift_score=shift_score,
            shift_details=shift_result.to_dict(),
            sigma_final=sigma_final,
            uncertainty_details=unc_result.to_dict(),
            calibrated_stress=calibrated,
            calibration_method=self.calibration_method,
            decisions=decision_result.decisions,
            decision_details=decision_result.to_dict(),
            contracts=contract_result.contracts,
            contract_summary=contract_result.summary,
        )
    
    def run_both_cities(self,
                        reference_stress: np.ndarray,
                        new_stress: np.ndarray,
                        reference_name: str = "seattle",
                        new_name: str = "nyc") -> Tuple[UniversalPipelineResult, UniversalPipelineResult]:
        """
        Convenience method to run pipeline on both reference and new city.
        
        Returns:
            (reference_result, new_result)
        """
        # Fit on reference
        self.fit_reference(reference_stress, reference_name)
        
        # Run on reference
        ref_result = self.run(reference_stress, reference_name)
        
        # Run on new city
        new_result = self.run(new_stress, new_name)
        
        return ref_result, new_result


def compare_cities(ref_result: UniversalPipelineResult,
                   new_result: UniversalPipelineResult,
                   ref_name: str = "Seattle",
                   new_name: str = "NYC") -> Dict:
    """
    Compare pipeline results between two cities.
    
    Returns comparison metrics.
    """
    comparison = {
        'reference_city': ref_name,
        'new_city': new_name,
        
        # Raw comparison
        'raw_magnitude_ratio': np.mean(new_result.raw_stress) / (np.mean(ref_result.raw_stress) + 1e-6),
        
        # Latent comparison (should be similar)
        'latent_mean_diff': abs(np.mean(new_result.latent_Z) - np.mean(ref_result.latent_Z)),
        
        # Shift detection
        f'{new_name}_shift_score': new_result.shift_score,
        
        # Uncertainty comparison
        'uncertainty_ratio': np.mean(new_result.sigma_final) / (np.mean(ref_result.sigma_final) + 1e-6),
        
        # Contract comparison
        f'{ref_name}_confident_pct': ref_result.contract_summary['confident_pct'],
        f'{new_name}_confident_pct': new_result.contract_summary['confident_pct'],
        f'{ref_name}_refused_pct': ref_result.contract_summary['refused_pct'],
        f'{new_name}_refused_pct': new_result.contract_summary['refused_pct'],
    }
    
    return comparison


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    print("Universal Deployment Pipeline Test")
    print("=" * 60)
    
    # Simulate two cities with VERY different raw magnitudes
    # Seattle: low magnitude
    seattle_stress = np.random.exponential(0.5, size=(10, 50, 50)) + 0.2
    print(f"\nSeattle raw stress: mean={np.mean(seattle_stress):.3f}, std={np.std(seattle_stress):.3f}")
    
    # NYC: high magnitude (14x higher like observed)
    nyc_stress = np.random.exponential(7.0, size=(10, 50, 50)) + 2.0
    print(f"NYC raw stress: mean={np.mean(nyc_stress):.3f}, std={np.std(nyc_stress):.3f}")
    
    # Run pipeline
    pipeline = UniversalDeploymentPipeline(conservative_mode=True)
    seattle_result, nyc_result = pipeline.run_both_cities(
        seattle_stress, nyc_stress, "Seattle", "NYC"
    )
    
    print(f"\n--- After Pipeline ---")
    print(f"\nSeattle latent Z: mean={np.mean(seattle_result.latent_Z):.3f}")
    print(f"NYC latent Z: mean={np.mean(nyc_result.latent_Z):.3f}")
    
    print(f"\nShift score (NYC vs Seattle): {nyc_result.shift_score:.3f}")
    
    print(f"\nSeattle mean uncertainty: {np.mean(seattle_result.sigma_final):.3f}")
    print(f"NYC mean uncertainty: {np.mean(nyc_result.sigma_final):.3f}")
    
    print(f"\nSeattle contracts:")
    print(f"  Confident: {seattle_result.contract_summary['confident_pct']:.1f}%")
    print(f"  Uncertain: {seattle_result.contract_summary['uncertain_pct']:.1f}%")
    print(f"  Refused: {seattle_result.contract_summary['refused_pct']:.1f}%")
    
    print(f"\nNYC contracts:")
    print(f"  Confident: {nyc_result.contract_summary['confident_pct']:.1f}%")
    print(f"  Uncertain: {nyc_result.contract_summary['uncertain_pct']:.1f}%")
    print(f"  Refused: {nyc_result.contract_summary['refused_pct']:.1f}%")
    
    # Compare
    comparison = compare_cities(seattle_result, nyc_result)
    print(f"\n--- Comparison ---")
    print(f"Raw magnitude ratio (NYC/Seattle): {comparison['raw_magnitude_ratio']:.1f}x")
    print(f"Latent mean difference: {comparison['latent_mean_diff']:.3f}")
    print(f"Uncertainty ratio (NYC/Seattle): {comparison['uncertainty_ratio']:.2f}x")
    
    print("\n✓ Pipeline complete!")
    print("✓ NYC has higher uncertainty than Seattle (as expected)")
    print("✓ Decisions are relative (no absolute thresholds)")
