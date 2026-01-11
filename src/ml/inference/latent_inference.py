"""
Latent Space Inference Pipeline.

Runtime inference with the Latent ST-GNN.

CRITICAL CONSTRAINTS:
- ΔZ is added ONLY in latent space
- Output is NEVER re-scaled by DL
- Output NEVER bypasses calibration
- Runtime guards enforced

Author: Urban Drainage AI Team
Date: January 2026
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import logging

# Import guards
from src.ml.guards import (
    validate_model_inputs,
    assert_scale_free,
    validate_uncertainty_additive,
    combine_uncertainties_safe,
)
from src.ml.targets import compute_latent_z, latent_to_raw

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class LatentInferenceConfig:
    """Configuration for latent inference."""
    # Guards
    enforce_input_guards: bool = True
    enforce_uncertainty_guards: bool = True
    
    # Clipping
    max_delta_z: float = 5.0  # Maximum allowed correction
    
    # Calibration bypass prevention
    require_calibration_after: bool = True


# ============================================================================
# INFERENCE RESULT
# ============================================================================

@dataclass
class LatentInferenceResult:
    """Container for inference results."""
    # Latent outputs (what DL produces)
    delta_z: np.ndarray  # Structural residual
    sigma_dl: np.ndarray  # DL epistemic uncertainty
    
    # Combined outputs (after Bayesian fusion)
    corrected_z: np.ndarray  # Physics Z + DL ΔZ
    total_uncertainty: np.ndarray  # Combined uncertainty
    
    # Metadata
    guards_passed: bool
    warnings: list


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class LatentInferencePipeline:
    """
    Latent space inference pipeline with runtime guards.
    
    ROLE: Apply structural corrections in latent space.
    NEVER outputs raw scale or decisions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        reference_mean: float,
        reference_std: float,
        config: Optional[LatentInferenceConfig] = None,
        device: str = 'cpu',
    ):
        """
        Args:
            model: Trained LatentSTGNN
            reference_mean: Reference mean from training data (Seattle)
            reference_std: Reference std from training data (Seattle)
            config: Inference configuration
            device: Compute device
        """
        self.model = model.to(device).eval()
        self.reference_mean = reference_mean
        self.reference_std = reference_std
        self.config = config or LatentInferenceConfig()
        self.device = device
        
        # Emit declaration
        self._emit_declaration()
        
    def _emit_declaration(self):
        """Emit the mandatory declaration."""
        declaration = (
            "The deep learning component is restricted "
            "to learning transferable structural residuals "
            "in a scale-free latent space. "
            "All magnitude calibration and decision logic "
            "are handled outside the DL model."
        )
        logger.info(f"📜 INFERENCE DECLARATION: {declaration}")
        
    @torch.no_grad()
    def infer(
        self,
        features: np.ndarray,
        edge_index: np.ndarray,
        physics_z: np.ndarray,
        bayesian_uncertainty: np.ndarray,
        edge_weight: Optional[np.ndarray] = None,
        input_types: Optional[Dict[str, str]] = None,
    ) -> LatentInferenceResult:
        """
        Run inference.
        
        Args:
            features: Scale-free node features [T, N, input_dim]
            edge_index: Graph edges [2, E]
            physics_z: Bayesian model output in latent Z-space [T, N]
            bayesian_uncertainty: Bayesian uncertainty [T, N]
            edge_weight: Optional edge weights [E]
            input_types: Optional type specification for input validation
        
        Returns:
            LatentInferenceResult
        """
        warnings = []
        guards_passed = True
        
        # ====================================================================
        # GUARD 1: Input Sanity Check
        # ====================================================================
        if self.config.enforce_input_guards:
            input_types = input_types or {'features': 'zscore'}
            result = validate_model_inputs(
                {'features': features},
                input_types
            )
            
            if not result.passed:
                guards_passed = False
                for v in result.violations:
                    logger.error(f"INPUT GUARD VIOLATION: {v}")
                raise ValueError(
                    f"Input guard failed! Violations: {result.violations}"
                )
            
            warnings.extend(result.warnings)
        
        # ====================================================================
        # MODEL FORWARD PASS
        # ====================================================================
        # Convert to tensors
        x = torch.from_numpy(features).float().to(self.device)
        ei = torch.from_numpy(edge_index).long().to(self.device)
        ew = None
        if edge_weight is not None:
            ew = torch.from_numpy(edge_weight).float().to(self.device)
        
        # Forward pass (DL produces ΔZ and σ_DL)
        delta_z, sigma_dl = self.model(x, ei, ew)
        
        # Convert back to numpy
        delta_z = delta_z.cpu().numpy().squeeze(-1)  # [T, N]
        sigma_dl = sigma_dl.cpu().numpy().squeeze(-1)  # [T, N]
        
        # ====================================================================
        # CLIP ΔZ (Prevent magnitude explosion)
        # ====================================================================
        delta_z = np.clip(delta_z, -self.config.max_delta_z, self.config.max_delta_z)
        
        # ====================================================================
        # COMBINE IN LATENT SPACE
        # ====================================================================
        # Corrected Z = Physics Z + DL ΔZ (in latent space)
        corrected_z = physics_z + delta_z
        
        logger.info(
            f"Correction applied in latent space: "
            f"mean(ΔZ)={np.mean(delta_z):.4f}, "
            f"std(ΔZ)={np.std(delta_z):.4f}"
        )
        
        # ====================================================================
        # GUARD 2: Uncertainty Additive Check
        # ====================================================================
        # Combine uncertainties using safe quadrature sum
        total_uncertainty = combine_uncertainties_safe(
            bayesian_uncertainty, sigma_dl
        )
        
        if self.config.enforce_uncertainty_guards:
            unc_result = validate_uncertainty_additive(
                bayesian_uncertainty, sigma_dl, total_uncertainty
            )
            
            if not unc_result.passed:
                guards_passed = False
                logger.error(f"UNCERTAINTY GUARD VIOLATION: {unc_result.violation_message}")
                
                if unc_result.abort_required:
                    raise RuntimeError(
                        f"Uncertainty guard requires ABORT: {unc_result.violation_message}"
                    )
            
            warnings.append(f"Uncertainty stats: {unc_result.stats}")
        
        # ====================================================================
        # FINAL CHECKS
        # ====================================================================
        if self.config.require_calibration_after:
            logger.info(
                "⚠️ Remember: This output is in LATENT space. "
                "Calibration and decision logic must be applied separately!"
            )
        
        return LatentInferenceResult(
            delta_z=delta_z.astype(np.float32),
            sigma_dl=sigma_dl.astype(np.float32),
            corrected_z=corrected_z.astype(np.float32),
            total_uncertainty=total_uncertainty.astype(np.float32),
            guards_passed=guards_passed,
            warnings=warnings,
        )
