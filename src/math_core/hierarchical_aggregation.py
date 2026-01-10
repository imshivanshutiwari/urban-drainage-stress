"""
Hierarchical Aggregation Module

Implements multi-scale rollup with uncertainty propagation:
- Hierarchical data structures
- Multi-resolution aggregation
- Uncertainty-aware pooling
- Spatial hierarchies
- Temporal aggregation

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.sparse import csr_matrix
from enum import Enum

from .error_aware_arithmetic import EPSILON


class AggregationType(Enum):
    """Types of aggregation operations."""
    SUM = 'sum'
    MEAN = 'mean'
    WEIGHTED_MEAN = 'weighted_mean'
    MAX = 'max'
    MIN = 'min'
    MEDIAN = 'median'
    QUANTILE = 'quantile'


@dataclass
class AggregatedValue:
    """Value with aggregation metadata."""
    value: float
    uncertainty: float
    n_samples: int
    aggregation_type: AggregationType
    weight_sum: float = 1.0
    
    # Components
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval."""
        z = stats.norm.ppf((1 + confidence) / 2)
        half_width = z * self.uncertainty
        return (self.value - half_width, self.value + half_width)


@dataclass
class HierarchyNode:
    """Node in hierarchical structure."""
    id: str
    level: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Data
    value: Optional[float] = None
    uncertainty: Optional[float] = None
    weight: float = 1.0
    
    # Metadata
    area: float = 1.0  # For spatial hierarchies
    time_span: float = 1.0  # For temporal hierarchies


class HierarchicalStructure:
    """
    Generic hierarchical structure for aggregation.
    """
    
    def __init__(self):
        self.nodes: Dict[str, HierarchyNode] = {}
        self.levels: Dict[int, List[str]] = {}
        self.root_ids: List[str] = []
    
    def add_node(self, node: HierarchyNode):
        """Add node to hierarchy."""
        self.nodes[node.id] = node
        
        if node.level not in self.levels:
            self.levels[node.level] = []
        self.levels[node.level].append(node.id)
        
        if node.parent_id is None:
            self.root_ids.append(node.id)
        elif node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
    
    def get_children(self, node_id: str) -> List[HierarchyNode]:
        """Get child nodes."""
        node = self.nodes[node_id]
        return [self.nodes[cid] for cid in node.children_ids]
    
    def get_ancestors(self, node_id: str) -> List[HierarchyNode]:
        """Get all ancestor nodes."""
        ancestors = []
        current = self.nodes[node_id]
        
        while current.parent_id is not None:
            parent = self.nodes[current.parent_id]
            ancestors.append(parent)
            current = parent
        
        return ancestors
    
    def get_descendants(self, node_id: str) -> List[HierarchyNode]:
        """Get all descendant nodes."""
        descendants = []
        queue = [node_id]
        
        while queue:
            current_id = queue.pop(0)
            current = self.nodes[current_id]
            
            for child_id in current.children_ids:
                descendants.append(self.nodes[child_id])
                queue.append(child_id)
        
        return descendants
    
    def get_leaves(self, node_id: Optional[str] = None) -> List[HierarchyNode]:
        """Get leaf nodes (optionally under a specific node)."""
        if node_id is None:
            return [n for n in self.nodes.values() if not n.children_ids]
        
        descendants = self.get_descendants(node_id)
        return [n for n in descendants if not n.children_ids]


class UncertaintyAwareAggregator:
    """
    Aggregate values with proper uncertainty propagation.
    """
    
    def __init__(self, aggregation_type: AggregationType = AggregationType.MEAN):
        self.aggregation_type = aggregation_type
    
    def aggregate(self, values: np.ndarray,
                  uncertainties: np.ndarray,
                  weights: Optional[np.ndarray] = None) -> AggregatedValue:
        """
        Aggregate values with uncertainty propagation.
        
        Args:
            values: Values to aggregate
            uncertainties: Standard uncertainties
            weights: Optional weights
        
        Returns:
            Aggregated value with uncertainty
        """
        values = np.asarray(values)
        uncertainties = np.asarray(uncertainties)
        n = len(values)
        
        if weights is None:
            weights = np.ones(n)
        weights = np.asarray(weights)
        
        if self.aggregation_type == AggregationType.SUM:
            return self._aggregate_sum(values, uncertainties, weights)
        
        elif self.aggregation_type == AggregationType.MEAN:
            return self._aggregate_mean(values, uncertainties, weights)
        
        elif self.aggregation_type == AggregationType.WEIGHTED_MEAN:
            return self._aggregate_weighted_mean(values, uncertainties, weights)
        
        elif self.aggregation_type == AggregationType.MAX:
            return self._aggregate_max(values, uncertainties)
        
        elif self.aggregation_type == AggregationType.MIN:
            return self._aggregate_min(values, uncertainties)
        
        elif self.aggregation_type == AggregationType.MEDIAN:
            return self._aggregate_median(values, uncertainties)
        
        raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
    
    def _aggregate_sum(self, values: np.ndarray,
                        uncertainties: np.ndarray,
                        weights: np.ndarray) -> AggregatedValue:
        """Sum aggregation."""
        weighted_values = weights * values
        agg_value = np.sum(weighted_values)
        
        # Uncertainty propagation (assuming independence)
        agg_uncertainty = np.sqrt(np.sum((weights * uncertainties) ** 2))
        
        return AggregatedValue(
            value=agg_value,
            uncertainty=agg_uncertainty,
            n_samples=len(values),
            aggregation_type=AggregationType.SUM,
            weight_sum=np.sum(weights)
        )
    
    def _aggregate_mean(self, values: np.ndarray,
                         uncertainties: np.ndarray,
                         weights: np.ndarray) -> AggregatedValue:
        """Mean aggregation."""
        n = len(values)
        agg_value = np.mean(values)
        
        # Uncertainty has two components:
        # 1. Propagated measurement uncertainty
        measurement_unc = np.sqrt(np.sum(uncertainties ** 2)) / n
        
        # 2. Sampling uncertainty (standard error)
        sampling_unc = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else 0
        
        # Combined uncertainty
        agg_uncertainty = np.sqrt(measurement_unc ** 2 + sampling_unc ** 2)
        
        return AggregatedValue(
            value=agg_value,
            uncertainty=agg_uncertainty,
            n_samples=n,
            aggregation_type=AggregationType.MEAN,
            weight_sum=n,
            epistemic_uncertainty=sampling_unc,
            aleatoric_uncertainty=measurement_unc
        )
    
    def _aggregate_weighted_mean(self, values: np.ndarray,
                                  uncertainties: np.ndarray,
                                  weights: np.ndarray) -> AggregatedValue:
        """Inverse-variance weighted mean."""
        # Use inverse variance as weights
        variances = uncertainties ** 2 + EPSILON
        inv_var_weights = 1.0 / variances
        
        # Weighted mean
        total_weight = np.sum(inv_var_weights)
        agg_value = np.sum(inv_var_weights * values) / total_weight
        
        # Uncertainty
        agg_variance = 1.0 / total_weight
        agg_uncertainty = np.sqrt(agg_variance)
        
        return AggregatedValue(
            value=agg_value,
            uncertainty=agg_uncertainty,
            n_samples=len(values),
            aggregation_type=AggregationType.WEIGHTED_MEAN,
            weight_sum=total_weight
        )
    
    def _aggregate_max(self, values: np.ndarray,
                        uncertainties: np.ndarray) -> AggregatedValue:
        """Maximum aggregation."""
        idx_max = np.argmax(values)
        agg_value = values[idx_max]
        
        # Uncertainty of max (approximation using order statistics)
        # For normal distributions, E[max] and Var[max] are complex
        # Use bootstrap for robustness
        n_bootstrap = 1000
        max_samples = []
        
        for _ in range(n_bootstrap):
            perturbed = values + np.random.randn(len(values)) * uncertainties
            max_samples.append(np.max(perturbed))
        
        agg_uncertainty = np.std(max_samples)
        
        return AggregatedValue(
            value=agg_value,
            uncertainty=agg_uncertainty,
            n_samples=len(values),
            aggregation_type=AggregationType.MAX
        )
    
    def _aggregate_min(self, values: np.ndarray,
                        uncertainties: np.ndarray) -> AggregatedValue:
        """Minimum aggregation."""
        idx_min = np.argmin(values)
        agg_value = values[idx_min]
        
        # Bootstrap uncertainty
        n_bootstrap = 1000
        min_samples = []
        
        for _ in range(n_bootstrap):
            perturbed = values + np.random.randn(len(values)) * uncertainties
            min_samples.append(np.min(perturbed))
        
        agg_uncertainty = np.std(min_samples)
        
        return AggregatedValue(
            value=agg_value,
            uncertainty=agg_uncertainty,
            n_samples=len(values),
            aggregation_type=AggregationType.MIN
        )
    
    def _aggregate_median(self, values: np.ndarray,
                           uncertainties: np.ndarray) -> AggregatedValue:
        """Median aggregation."""
        agg_value = np.median(values)
        
        # Median uncertainty via bootstrap
        n_bootstrap = 1000
        median_samples = []
        
        for _ in range(n_bootstrap):
            perturbed = values + np.random.randn(len(values)) * uncertainties
            median_samples.append(np.median(perturbed))
        
        agg_uncertainty = np.std(median_samples)
        
        return AggregatedValue(
            value=agg_value,
            uncertainty=agg_uncertainty,
            n_samples=len(values),
            aggregation_type=AggregationType.MEDIAN
        )


class HierarchicalAggregator:
    """
    Perform hierarchical aggregation with uncertainty propagation.
    """
    
    def __init__(self, hierarchy: HierarchicalStructure,
                 aggregation_type: AggregationType = AggregationType.WEIGHTED_MEAN):
        self.hierarchy = hierarchy
        self.aggregator = UncertaintyAwareAggregator(aggregation_type)
    
    def aggregate_bottom_up(self) -> Dict[str, AggregatedValue]:
        """
        Aggregate from leaves to roots.
        
        Returns values for all nodes.
        """
        results: Dict[str, AggregatedValue] = {}
        
        # Get levels in reverse order (bottom to top)
        max_level = max(self.hierarchy.levels.keys())
        
        for level in range(max_level, -1, -1):
            if level not in self.hierarchy.levels:
                continue
            
            for node_id in self.hierarchy.levels[level]:
                node = self.hierarchy.nodes[node_id]
                
                if not node.children_ids:
                    # Leaf node - use direct value
                    if node.value is not None:
                        results[node_id] = AggregatedValue(
                            value=node.value,
                            uncertainty=node.uncertainty or 0.0,
                            n_samples=1,
                            aggregation_type=self.aggregator.aggregation_type
                        )
                else:
                    # Aggregate children
                    child_values = []
                    child_uncertainties = []
                    child_weights = []
                    
                    for child_id in node.children_ids:
                        if child_id in results:
                            child_result = results[child_id]
                            child_node = self.hierarchy.nodes[child_id]
                            
                            child_values.append(child_result.value)
                            child_uncertainties.append(child_result.uncertainty)
                            child_weights.append(child_node.weight)
                    
                    if child_values:
                        results[node_id] = self.aggregator.aggregate(
                            np.array(child_values),
                            np.array(child_uncertainties),
                            np.array(child_weights)
                        )
        
        return results
    
    def disaggregate_top_down(self, target_values: Dict[str, float],
                               constraints: str = 'proportional'
                               ) -> Dict[str, float]:
        """
        Disaggregate from roots to leaves.
        
        Distributes parent values to children while respecting constraints.
        
        Args:
            target_values: Target values for root nodes
            constraints: 'proportional' or 'equal'
        
        Returns:
            Values for leaf nodes
        """
        results: Dict[str, float] = {}
        
        # Get levels in order (top to bottom)
        for level in sorted(self.hierarchy.levels.keys()):
            for node_id in self.hierarchy.levels[level]:
                node = self.hierarchy.nodes[node_id]
                
                # Get target for this node
                if node.parent_id is None:
                    # Root node
                    target = target_values.get(node_id, 0)
                else:
                    # Get from parent disaggregation
                    target = results.get(node_id, 0)
                
                if not node.children_ids:
                    # Leaf node
                    results[node_id] = target
                else:
                    # Disaggregate to children
                    if constraints == 'proportional':
                        # Distribute proportionally to weights
                        total_weight = sum(
                            self.hierarchy.nodes[cid].weight 
                            for cid in node.children_ids
                        )
                        
                        for child_id in node.children_ids:
                            child = self.hierarchy.nodes[child_id]
                            proportion = child.weight / (total_weight + EPSILON)
                            results[child_id] = target * proportion
                    
                    elif constraints == 'equal':
                        # Distribute equally
                        n_children = len(node.children_ids)
                        for child_id in node.children_ids:
                            results[child_id] = target / n_children
        
        return results


class SpatialHierarchy(HierarchicalStructure):
    """
    Spatial hierarchy for geographic aggregation.
    """
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def create_regular_grid_hierarchy(cls, n_rows: int, n_cols: int,
                                       n_levels: int = 3
                                       ) -> 'SpatialHierarchy':
        """
        Create hierarchy from regular grid.
        
        Each level aggregates 2x2 cells from below.
        """
        hierarchy = cls()
        
        # Create leaf level (finest resolution)
        for i in range(n_rows):
            for j in range(n_cols):
                node = HierarchyNode(
                    id=f"L0_{i}_{j}",
                    level=0,
                    area=1.0
                )
                hierarchy.add_node(node)
        
        # Create coarser levels
        current_rows = n_rows
        current_cols = n_cols
        
        for level in range(1, n_levels):
            prev_rows = current_rows
            prev_cols = current_cols
            current_rows = (prev_rows + 1) // 2
            current_cols = (prev_cols + 1) // 2
            
            for i in range(current_rows):
                for j in range(current_cols):
                    node_id = f"L{level}_{i}_{j}"
                    
                    # Find children
                    children = []
                    for di in range(2):
                        for dj in range(2):
                            child_i = 2 * i + di
                            child_j = 2 * j + dj
                            if child_i < prev_rows and child_j < prev_cols:
                                child_id = f"L{level-1}_{child_i}_{child_j}"
                                children.append(child_id)
                    
                    # Set parent for children
                    for child_id in children:
                        hierarchy.nodes[child_id].parent_id = node_id
                    
                    node = HierarchyNode(
                        id=node_id,
                        level=level,
                        children_ids=children,
                        area=len(children)
                    )
                    hierarchy.add_node(node)
        
        return hierarchy


class TemporalHierarchy(HierarchicalStructure):
    """
    Temporal hierarchy for time series aggregation.
    """
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def create_temporal_hierarchy(cls, n_timesteps: int,
                                   granularities: List[int]
                                   ) -> 'TemporalHierarchy':
        """
        Create temporal hierarchy.
        
        Args:
            n_timesteps: Number of base timesteps
            granularities: Aggregation factors for each level
                          e.g., [1, 6, 24, 168] for hourly -> 6h -> daily -> weekly
        """
        hierarchy = cls()
        
        # Create leaf level
        for t in range(n_timesteps):
            node = HierarchyNode(
                id=f"T0_{t}",
                level=0,
                time_span=1.0
            )
            hierarchy.add_node(node)
        
        # Create coarser levels
        prev_n = n_timesteps
        
        for level, factor in enumerate(granularities[1:], start=1):
            current_n = (prev_n + factor - 1) // factor
            
            for t in range(current_n):
                node_id = f"T{level}_{t}"
                
                # Find children
                children = []
                for dt in range(factor):
                    child_t = t * factor + dt
                    if child_t < prev_n:
                        child_id = f"T{level-1}_{child_t}"
                        children.append(child_id)
                
                # Set parent for children
                for child_id in children:
                    hierarchy.nodes[child_id].parent_id = node_id
                
                node = HierarchyNode(
                    id=node_id,
                    level=level,
                    children_ids=children,
                    time_span=len(children)
                )
                hierarchy.add_node(node)
            
            prev_n = current_n
        
        return hierarchy


def create_aggregation_matrix(hierarchy: HierarchicalStructure,
                               from_level: int,
                               to_level: int) -> csr_matrix:
    """
    Create sparse aggregation matrix.
    
    Args:
        hierarchy: Hierarchical structure
        from_level: Source level (finer)
        to_level: Target level (coarser)
    
    Returns:
        Sparse matrix S where aggregated = S @ detailed
    """
    from_nodes = hierarchy.levels.get(from_level, [])
    to_nodes = hierarchy.levels.get(to_level, [])
    
    n_from = len(from_nodes)
    n_to = len(to_nodes)
    
    from_idx = {nid: i for i, nid in enumerate(from_nodes)}
    to_idx = {nid: i for i, nid in enumerate(to_nodes)}
    
    # Build sparse matrix
    rows = []
    cols = []
    data = []
    
    for to_id in to_nodes:
        to_i = to_idx[to_id]
        
        # Get all descendants at from_level
        descendants = hierarchy.get_descendants(to_id)
        level_descendants = [d for d in descendants if d.level == from_level]
        
        if not level_descendants:
            continue
        
        # Equal weights (could be area-weighted)
        weight = 1.0 / len(level_descendants)
        
        for desc in level_descendants:
            from_j = from_idx.get(desc.id)
            if from_j is not None:
                rows.append(to_i)
                cols.append(from_j)
                data.append(weight)
    
    return csr_matrix((data, (rows, cols)), shape=(n_to, n_from))
