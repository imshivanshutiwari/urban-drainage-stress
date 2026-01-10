"""
Mathematical Core Module - State-of-the-Art Formulations

This module provides MAX-LEVEL mathematical implementations:
1. Error-Aware Arithmetic
2. Sparse Linear Operators
3. Multi-Resolution Grids
4. Robust Computational Geometry
5. Anisotropic Distance
6. Non-Stationary Kernels
7. Hierarchical Probabilistic Models
8. Mixture Distributions
9. State-Space Inference
10. Epistemic/Aleatoric Uncertainty
11. Continuous-Time Dynamics
12. Distribution-Aware Scaling
13. Formal Stability Controls
14. Robust Optimization
15. Utility-Based Decisions
16. Probabilistic Logic
17. Hierarchical Aggregation

NO HEURISTICS. NO SIMPLIFICATIONS. NO PLACEHOLDERS.
"""

# 1. Error-Aware Arithmetic
from .error_aware_arithmetic import (
    ErrorValue, ErrorArray, EPSILON, SQRT_EPSILON,
    safe_divide, safe_log, safe_sqrt, safe_exp
)

# 2. Sparse Linear Operators
from .sparse_linear_operators import (
    SparseStressOperator, SparseTemporalOperator,
    build_spatial_laplacian, build_diffusion_operator, create_weight_matrix
)

# 3. Multi-Resolution Grids
from .multi_resolution_grid import (
    GridCell, AdaptiveQuadtree, MultiResolutionGrid, compute_optimal_resolution
)

# 4. Robust Computational Geometry
from .robust_geometry import (
    BoundingBox, RTreeNode, RTree, Polygon,
    orient2d, incircle2d, compute_voronoi_with_uncertainty, compute_delaunay_with_uncertainty
)

# 5. Anisotropic Distance
from .anisotropic_distance import (
    MetricTensor, DistanceResult, AnisotropicDistanceField, DrainageNetworkDistance,
    compute_anisotropic_kriging_distances
)

# 6. Non-Stationary Kernels
from .nonstationary_kernels import (
    KernelParameters, SquaredExponentialKernel, MaternKernel,
    NonStationaryKernel, LocallyAdaptiveKernel, ProcessConvolutionKernel,
    estimate_local_length_scales, build_nonstationary_covariance_matrix
)

# 7. Hierarchical Probabilistic Models
from .hierarchical_model import (
    Prior, GaussianPrior, GammaPrior, InverseGammaPrior, BetaPrior,
    Likelihood, HierarchicalModel, StressHierarchicalModel,
    compute_credible_interval, compute_highest_density_interval, bayes_factor
)

# 8. Mixture Distributions
from .mixture_distributions import (
    MixtureComponent, GaussianMixtureModel, MixtureOfExperts, FiniteMixture,
    fit_mixture_to_uncertainty, mixture_kalman_update
)

# 9. State-Space Inference
from .state_space_inference import (
    StateEstimate, LinearStateSpaceModel, KalmanFilter,
    ExtendedKalmanFilter, UnscentedKalmanFilter, DrainageStateSpaceModel
)

# 10. Epistemic/Aleatoric Uncertainty
from .uncertainty_decomposition import (
    DecomposedUncertainty, EnsemblePredictor, HeteroscedasticModel, 
    GaussianProcessUncertainty, DeepUncertaintyEstimator,
    propagate_decomposed_uncertainty, information_gain, optimal_sampling_location
)

# 11. Continuous-Time Dynamics
from .continuous_dynamics import (
    SDEState, LinearSDE, NonlinearSDE, SemiImplicitIntegrator,
    AdaptiveTimestepper, ConservativeIntegrator, DrainageStressDynamics
)

# 12. Distribution-Aware Scaling
from .distribution_scaling import (
    TransformResult, QuantileTransformer, BoxCoxTransformer, YeoJohnsonTransformer,
    RankPreservingScaler, RobustScaler, GaussianCopulaTransformer,
    select_best_transform, propagate_uncertainty_through_transform
)

# 13. Formal Stability Controls
from .stability_controls import (
    StabilityReport, LipschitzMonitor, SpectralRadiusMonitor,
    LyapunovAnalyzer, ConditionMonitor, GradientController,
    analyze_system_stability, ensure_stability
)

# 14. Robust Optimization
from .robust_optimization import (
    OptimizationResult, CVaROptimizer, DistributionallyRobustOptimizer,
    ChanceConstrainedOptimizer, RobustCounterpartSolver, compute_risk_spectrum
)

# 15. Utility-Based Decisions
from .utility_decisions import (
    DecisionResult, UtilityFunction, CRRAUtility, CARAUtility,
    ProspectTheoryUtility, MultiAttributeUtility, BayesianDecisionMaker,
    compute_certainty_equivalent_distribution, expected_utility_optimization
)

# 16. Probabilistic Logic
from .probabilistic_logic import (
    LogicalAtom, PSLRule, PSLModel, DempsterShaferCombiner,
    FuzzyLogic, BeliefPropagation, combine_probabilistic_evidence
)

# 17. Hierarchical Aggregation
from .hierarchical_aggregation import (
    AggregationType, AggregatedValue, HierarchyNode, HierarchicalStructure,
    UncertaintyAwareAggregator, HierarchicalAggregator,
    SpatialHierarchy, TemporalHierarchy, create_aggregation_matrix
)

__all__ = [
    # Error-aware arithmetic
    'ErrorValue', 'ErrorArray', 'EPSILON', 'SQRT_EPSILON',
    'safe_divide', 'safe_log', 'safe_sqrt', 'safe_exp',
    
    # Sparse operators
    'SparseStressOperator', 'SparseTemporalOperator',
    'build_spatial_laplacian', 'build_diffusion_operator', 'create_weight_matrix',
    
    # Multi-resolution
    'GridCell', 'AdaptiveQuadtree', 'MultiResolutionGrid', 'compute_optimal_resolution',
    
    # Geometry
    'BoundingBox', 'RTreeNode', 'RTree', 'Polygon',
    'orient2d', 'incircle2d', 'compute_voronoi_with_uncertainty', 'compute_delaunay_with_uncertainty',
    
    # Distance
    'MetricTensor', 'DistanceResult', 'AnisotropicDistanceField', 'DrainageNetworkDistance',
    'compute_anisotropic_kriging_distances',
    
    # Kernels
    'KernelParameters', 'SquaredExponentialKernel', 'MaternKernel',
    'NonStationaryKernel', 'LocallyAdaptiveKernel', 'ProcessConvolutionKernel',
    'estimate_local_length_scales', 'build_nonstationary_covariance_matrix',
    
    # Hierarchical prob
    'Prior', 'GaussianPrior', 'GammaPrior', 'InverseGammaPrior', 'BetaPrior',
    'Likelihood', 'HierarchicalModel', 'StressHierarchicalModel',
    'compute_credible_interval', 'compute_highest_density_interval', 'bayes_factor',
    
    # Mixtures
    'MixtureComponent', 'GaussianMixtureModel', 'MixtureOfExperts', 'FiniteMixture',
    'fit_mixture_to_uncertainty', 'mixture_kalman_update',
    
    # State-space
    'StateEstimate', 'LinearStateSpaceModel', 'KalmanFilter',
    'ExtendedKalmanFilter', 'UnscentedKalmanFilter', 'DrainageStateSpaceModel',
    
    # Uncertainty
    'DecomposedUncertainty', 'EnsemblePredictor', 'HeteroscedasticModel',
    'GaussianProcessUncertainty', 'DeepUncertaintyEstimator',
    'propagate_decomposed_uncertainty', 'information_gain', 'optimal_sampling_location',
    
    # Continuous time
    'SDEState', 'LinearSDE', 'NonlinearSDE', 'SemiImplicitIntegrator',
    'AdaptiveTimestepper', 'ConservativeIntegrator', 'DrainageStressDynamics',
    
    # Scaling
    'TransformResult', 'QuantileTransformer', 'BoxCoxTransformer', 'YeoJohnsonTransformer',
    'RankPreservingScaler', 'RobustScaler', 'GaussianCopulaTransformer',
    'select_best_transform', 'propagate_uncertainty_through_transform',
    
    # Stability
    'StabilityReport', 'LipschitzMonitor', 'SpectralRadiusMonitor',
    'LyapunovAnalyzer', 'ConditionMonitor', 'GradientController',
    'analyze_system_stability', 'ensure_stability',
    
    # Robust opt
    'OptimizationResult', 'CVaROptimizer', 'DistributionallyRobustOptimizer',
    'ChanceConstrainedOptimizer', 'RobustCounterpartSolver', 'compute_risk_spectrum',
    
    # Utility
    'DecisionResult', 'UtilityFunction', 'CRRAUtility', 'CARAUtility',
    'ProspectTheoryUtility', 'MultiAttributeUtility', 'BayesianDecisionMaker',
    'compute_certainty_equivalent_distribution', 'expected_utility_optimization',
    
    # Probabilistic logic
    'LogicalAtom', 'PSLRule', 'PSLModel', 'DempsterShaferCombiner',
    'FuzzyLogic', 'BeliefPropagation', 'combine_probabilistic_evidence',
    
    # Aggregation
    'AggregationType', 'AggregatedValue', 'HierarchyNode', 'HierarchicalStructure',
    'UncertaintyAwareAggregator', 'HierarchicalAggregator',
    'SpatialHierarchy', 'TemporalHierarchy', 'create_aggregation_matrix',
]
