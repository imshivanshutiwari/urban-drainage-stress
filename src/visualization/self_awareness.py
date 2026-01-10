"""System self-awareness and honest reporting.

This module implements the "SYSTEM KNOWS WHAT IT DOESN'T KNOW" requirement:
- Reports learned parameters WITH confidence intervals
- Explicitly states what is NOT identifiable
- Disclaimers about model limitations
- Run diagnostics and metadata

Reference: Projectfile.md - "System self-awareness"

NO HIDING LIMITATIONS. HONEST SCIENCE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from pathlib import Path
import json
import platform

import numpy as np

logger = logging.getLogger(__name__)


class IdentifiabilityStatus(Enum):
    """Whether a parameter is identifiable from data."""
    IDENTIFIABLE = "identifiable"
    WEAKLY_IDENTIFIABLE = "weakly_identifiable"
    NOT_IDENTIFIABLE = "not_identifiable"
    UNKNOWN = "unknown"


@dataclass
class ParameterReport:
    """Report for a single learned parameter."""
    name: str
    value: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    identifiability: IdentifiabilityStatus
    fit_metric: Optional[float] = None
    warning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "ci_95": [self.ci_lower, self.ci_upper],
            "ci_width": self.ci_width,
            "identifiability": self.identifiability.value,
            "fit_metric": self.fit_metric,
            "warning": self.warning,
        }


@dataclass
class LimitationDisclaimer:
    """A specific limitation disclaimer."""
    category: str
    description: str
    severity: str  # "critical", "moderate", "minor"
    recommendation: str


@dataclass
class RunMetadata:
    """Metadata about the run."""
    timestamp: str
    duration_seconds: float
    python_version: str
    numpy_version: str
    platform_info: str
    
    # Data characteristics
    n_timesteps: int
    grid_shape: tuple
    n_observations: int
    observation_density: float
    
    # Computational info
    n_parameters_learned: int
    convergence_achieved: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "platform": self.platform_info,
            "data": {
                "n_timesteps": self.n_timesteps,
                "grid_shape": self.grid_shape,
                "n_observations": self.n_observations,
                "observation_density": self.observation_density,
            },
            "computation": {
                "n_parameters_learned": self.n_parameters_learned,
                "convergence_achieved": self.convergence_achieved,
            },
        }


@dataclass
class SelfAwarenessReport:
    """Complete self-awareness report for a run.
    
    This report is MANDATORY for every run. It contains:
    1. What the model learned (parameters with CIs)
    2. What the model doesn't know (limitations)
    3. Where the model is uncertain (high variance regions)
    4. Run metadata and diagnostics
    """
    
    # Learned parameters
    parameters: List[ParameterReport]
    
    # Limitations and disclaimers
    limitations: List[LimitationDisclaimer]
    
    # Identifiability summary
    identifiable_count: int
    weakly_identifiable_count: int
    not_identifiable_count: int
    
    # Spatial uncertainty summary
    high_uncertainty_fraction: float
    no_decision_fraction: float
    
    # Fit quality
    overall_fit_metric: float
    fit_quality: str  # "excellent", "good", "acceptable", "poor"
    
    # Validation
    validation_passed: bool
    integrity_score: float
    
    # Metadata
    metadata: RunMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "learned_parameters": [p.to_dict() for p in self.parameters],
            "limitations": [
                {
                    "category": L.category,
                    "description": L.description,
                    "severity": L.severity,
                    "recommendation": L.recommendation,
                }
                for L in self.limitations
            ],
            "identifiability_summary": {
                "identifiable": self.identifiable_count,
                "weakly_identifiable": self.weakly_identifiable_count,
                "not_identifiable": self.not_identifiable_count,
            },
            "uncertainty_summary": {
                "high_uncertainty_fraction": self.high_uncertainty_fraction,
                "no_decision_fraction": self.no_decision_fraction,
            },
            "fit_quality": {
                "metric": self.overall_fit_metric,
                "rating": self.fit_quality,
            },
            "validation": {
                "passed": self.validation_passed,
                "integrity_score": self.integrity_score,
            },
            "metadata": self.metadata.to_dict(),
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved self-awareness report to %s", path)
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "="*70)
        print("🔬 SCIENTIFIC SELF-AWARENESS REPORT")
        print("="*70)
        
        print(f"\n📅 Run: {self.metadata.timestamp}")
        print(f"⏱️  Duration: {self.metadata.duration_seconds:.1f}s")
        
        print(f"\n📊 Data: {self.metadata.n_timesteps} timesteps, "
              f"{self.metadata.grid_shape[0]}x{self.metadata.grid_shape[1]} grid")
        print(f"👁️  Observations: {self.metadata.n_observations} "
              f"({self.metadata.observation_density:.1%} coverage)")
        
        print(f"\n✅ Validation: {'PASSED' if self.validation_passed else '❌ FAILED'}")
        print(f"🎯 Integrity Score: {self.integrity_score:.0f}/100")
        print(f"📈 Fit Quality: {self.fit_quality.upper()} ({self.overall_fit_metric:.3f})")
        
        print(f"\n🎲 WHAT THE MODEL LEARNED:")
        print("-"*50)
        for p in self.parameters:
            status = "✓" if p.identifiability == IdentifiabilityStatus.IDENTIFIABLE else \
                    "~" if p.identifiability == IdentifiabilityStatus.WEAKLY_IDENTIFIABLE else "?"
            print(f"  {status} {p.name}: {p.value:.4f} [{p.ci_lower:.4f}, {p.ci_upper:.4f}]")
            if p.warning:
                print(f"    ⚠️  {p.warning}")
        
        print(f"\n❓ WHAT THE MODEL DOESN'T KNOW:")
        print("-"*50)
        print(f"  • {self.not_identifiable_count} parameters not identifiable")
        print(f"  • {self.weakly_identifiable_count} parameters weakly identifiable")
        print(f"  • {self.high_uncertainty_fraction:.1%} of space has high uncertainty")
        print(f"  • {self.no_decision_fraction:.1%} labeled as NO_DECISION")
        
        print(f"\n⚠️  LIMITATIONS:")
        print("-"*50)
        for L in self.limitations:
            severity_icon = "🔴" if L.severity == "critical" else \
                           "🟡" if L.severity == "moderate" else "🟢"
            print(f"  {severity_icon} [{L.category}] {L.description}")
            print(f"     → {L.recommendation}")
        
        print("\n" + "="*70)


class SelfAwarenessEngine:
    """Engine for generating self-awareness reports."""
    
    def __init__(self) -> None:
        self.start_time = datetime.now()
    
    def generate_report(
        self,
        # Learned parameters
        weights: Optional[Dict[str, float]] = None,
        weights_ci: Optional[Dict[str, tuple]] = None,
        memory_decay: float = 0.0,
        memory_decay_ci: Optional[tuple] = None,
        fit_metrics: Optional[Dict[str, float]] = None,
        
        # Inference results
        posterior_variance: Optional[np.ndarray] = None,
        risk_levels: Optional[np.ndarray] = None,
        
        # Validation
        validation_passed: bool = True,
        integrity_score: float = 100.0,
        
        # Data info
        n_timesteps: int = 0,
        grid_shape: tuple = (0, 0),
        n_observations: int = 0,
        complaints: Optional[np.ndarray] = None,
    ) -> SelfAwarenessReport:
        """Generate comprehensive self-awareness report."""
        
        # Build parameter reports
        parameters = self._build_parameter_reports(
            weights, weights_ci, memory_decay, memory_decay_ci, fit_metrics
        )
        
        # Assess identifiability
        id_counts = self._count_identifiability(parameters)
        
        # Build limitations
        limitations = self._build_limitations(
            parameters, n_observations, posterior_variance, fit_metrics
        )
        
        # Uncertainty summary
        if posterior_variance is not None:
            std = np.sqrt(posterior_variance)
            high_threshold = np.percentile(std, 75)
            high_unc_frac = np.mean(std >= high_threshold)
        else:
            high_unc_frac = 0.0
        
        # No-decision fraction
        if risk_levels is not None:
            unique, counts = np.unique(risk_levels, return_counts=True)
            no_dec_count = sum(c for u, c in zip(unique, counts) 
                             if 'no_decision' in str(u).lower())
            no_dec_frac = no_dec_count / risk_levels.size
        else:
            no_dec_frac = 0.0
        
        # Fit quality
        if fit_metrics and 'r_squared' in fit_metrics:
            r2 = fit_metrics['r_squared']
            if r2 >= 0.8:
                fit_quality = "excellent"
            elif r2 >= 0.6:
                fit_quality = "good"
            elif r2 >= 0.4:
                fit_quality = "acceptable"
            else:
                fit_quality = "poor"
            overall_fit = r2
        else:
            fit_quality = "unknown"
            overall_fit = 0.0
        
        # Observation density
        if complaints is not None:
            obs_density = np.mean(complaints > 0)
            n_obs = int(np.sum(complaints > 0))
        else:
            obs_density = n_observations / (n_timesteps * grid_shape[0] * grid_shape[1] + 1)
            n_obs = n_observations
        
        # Metadata
        end_time = datetime.now()
        metadata = RunMetadata(
            timestamp=self.start_time.isoformat(),
            duration_seconds=(end_time - self.start_time).total_seconds(),
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            platform_info=platform.platform(),
            n_timesteps=n_timesteps,
            grid_shape=grid_shape,
            n_observations=n_obs,
            observation_density=obs_density,
            n_parameters_learned=len(parameters),
            convergence_achieved=fit_quality in ["excellent", "good", "acceptable"],
        )
        
        return SelfAwarenessReport(
            parameters=parameters,
            limitations=limitations,
            identifiable_count=id_counts['identifiable'],
            weakly_identifiable_count=id_counts['weakly'],
            not_identifiable_count=id_counts['not'],
            high_uncertainty_fraction=high_unc_frac,
            no_decision_fraction=no_dec_frac,
            overall_fit_metric=overall_fit,
            fit_quality=fit_quality,
            validation_passed=validation_passed,
            integrity_score=integrity_score,
            metadata=metadata,
        )
    
    def _build_parameter_reports(
        self,
        weights: Optional[Dict[str, float]],
        weights_ci: Optional[Dict[str, tuple]],
        memory_decay: float,
        memory_decay_ci: Optional[tuple],
        fit_metrics: Optional[Dict[str, float]],
    ) -> List[ParameterReport]:
        """Build parameter reports with identifiability assessment."""
        parameters = []
        
        if weights:
            for name, value in weights.items():
                if weights_ci and name in weights_ci:
                    ci_lower, ci_upper = weights_ci[name]
                else:
                    ci_lower, ci_upper = value * 0.5, value * 1.5
                
                ci_width = ci_upper - ci_lower
                
                # Assess identifiability from CI width
                rel_width = ci_width / (abs(value) + 1e-9)
                if rel_width < 0.5:
                    identifiability = IdentifiabilityStatus.IDENTIFIABLE
                    warning = None
                elif rel_width < 1.0:
                    identifiability = IdentifiabilityStatus.WEAKLY_IDENTIFIABLE
                    warning = f"Wide CI ({rel_width:.0%} of value)"
                else:
                    identifiability = IdentifiabilityStatus.NOT_IDENTIFIABLE
                    warning = f"Very wide CI ({rel_width:.0%}) - not identifiable from data"
                
                parameters.append(ParameterReport(
                    name=name,
                    value=value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    ci_width=ci_width,
                    identifiability=identifiability,
                    fit_metric=fit_metrics.get('r_squared') if fit_metrics else None,
                    warning=warning,
                ))
        
        # Memory decay parameter
        if memory_decay_ci:
            ci_lower, ci_upper = memory_decay_ci
        else:
            ci_lower, ci_upper = memory_decay * 0.5, memory_decay * 1.5
        
        ci_width = ci_upper - ci_lower
        rel_width = ci_width / (abs(memory_decay) + 1e-9)
        
        if rel_width < 0.5:
            id_status = IdentifiabilityStatus.IDENTIFIABLE
        elif rel_width < 1.0:
            id_status = IdentifiabilityStatus.WEAKLY_IDENTIFIABLE
        else:
            id_status = IdentifiabilityStatus.NOT_IDENTIFIABLE
        
        parameters.append(ParameterReport(
            name="memory_decay",
            value=memory_decay,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_width,
            identifiability=id_status,
        ))
        
        return parameters
    
    def _count_identifiability(
        self, parameters: List[ParameterReport]
    ) -> Dict[str, int]:
        """Count parameters by identifiability status."""
        counts = {'identifiable': 0, 'weakly': 0, 'not': 0}
        for p in parameters:
            if p.identifiability == IdentifiabilityStatus.IDENTIFIABLE:
                counts['identifiable'] += 1
            elif p.identifiability == IdentifiabilityStatus.WEAKLY_IDENTIFIABLE:
                counts['weakly'] += 1
            else:
                counts['not'] += 1
        return counts
    
    def _build_limitations(
        self,
        parameters: List[ParameterReport],
        n_observations: int,
        posterior_variance: Optional[np.ndarray],
        fit_metrics: Optional[Dict[str, float]],
    ) -> List[LimitationDisclaimer]:
        """Build list of limitation disclaimers."""
        limitations = []
        
        # Data limitations
        if n_observations < 100:
            limitations.append(LimitationDisclaimer(
                category="Data",
                description=f"Only {n_observations} observations available",
                severity="critical" if n_observations < 20 else "moderate",
                recommendation="Collect more observations or use wider priors",
            ))
        
        # Parameter identifiability
        not_id = sum(1 for p in parameters 
                    if p.identifiability == IdentifiabilityStatus.NOT_IDENTIFIABLE)
        if not_id > 0:
            limitations.append(LimitationDisclaimer(
                category="Identifiability",
                description=f"{not_id} parameters cannot be reliably estimated",
                severity="moderate",
                recommendation="Consider fixing these parameters or using informative priors",
            ))
        
        # Fit quality
        if fit_metrics and fit_metrics.get('r_squared', 0) < 0.5:
            limitations.append(LimitationDisclaimer(
                category="Model Fit",
                description=f"Poor fit (R² = {fit_metrics['r_squared']:.2f})",
                severity="moderate",
                recommendation="Consider additional covariates or model structure",
            ))
        
        # Always include standard disclaimers
        limitations.append(LimitationDisclaimer(
            category="Spatial Resolution",
            description="30m DEM resolution limits detection of small-scale features",
            severity="minor",
            recommendation="Interpret results at neighborhood, not parcel level",
        ))
        
        limitations.append(LimitationDisclaimer(
            category="Temporal",
            description="Daily aggregation may miss sub-daily dynamics",
            severity="minor",
            recommendation="Results valid for daily-scale decisions only",
        ))
        
        limitations.append(LimitationDisclaimer(
            category="Missing Variables",
            description="Infrastructure age, condition, capacity not modeled",
            severity="moderate",
            recommendation="Treat outputs as physical susceptibility, not failure risk",
        ))
        
        return limitations


def generate_self_awareness_report(
    weights: Optional[Dict[str, float]] = None,
    weights_ci: Optional[Dict[str, tuple]] = None,
    memory_decay: float = 0.0,
    memory_decay_ci: Optional[tuple] = None,
    fit_metrics: Optional[Dict[str, float]] = None,
    posterior_variance: Optional[np.ndarray] = None,
    risk_levels: Optional[np.ndarray] = None,
    validation_passed: bool = True,
    integrity_score: float = 100.0,
    n_timesteps: int = 0,
    grid_shape: tuple = (0, 0),
    n_observations: int = 0,
    complaints: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    print_summary: bool = True,
) -> SelfAwarenessReport:
    """Main entry point for generating self-awareness report.
    
    This should be called at the END of every pipeline run.
    """
    engine = SelfAwarenessEngine()
    report = engine.generate_report(
        weights=weights,
        weights_ci=weights_ci,
        memory_decay=memory_decay,
        memory_decay_ci=memory_decay_ci,
        fit_metrics=fit_metrics,
        posterior_variance=posterior_variance,
        risk_levels=risk_levels,
        validation_passed=validation_passed,
        integrity_score=integrity_score,
        n_timesteps=n_timesteps,
        grid_shape=grid_shape,
        n_observations=n_observations,
        complaints=complaints,
    )
    
    if print_summary:
        report.print_summary()
    
    if output_path:
        report.save(output_path)
    
    return report
