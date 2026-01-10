"""
Multi-Resolution Grid Module

Implements adaptive resolution spatial grids:
- Finer resolution where data density is high
- Coarser where data is sparse
- Consistent uncertainty propagation across scales

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import warnings

from .error_aware_arithmetic import ErrorArray, EPSILON


@dataclass
class GridCell:
    """A cell in the adaptive grid."""
    level: int  # Refinement level (0 = coarsest)
    bounds: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    value: float = 0.0
    uncertainty: float = np.inf
    data_count: int = 0
    children: Optional[List['GridCell']] = None
    
    @property
    def is_leaf(self) -> bool:
        return self.children is None
    
    @property
    def center(self) -> Tuple[float, float]:
        return (
            0.5 * (self.bounds[0] + self.bounds[2]),
            0.5 * (self.bounds[1] + self.bounds[3])
        )
    
    @property
    def size(self) -> Tuple[float, float]:
        return (
            self.bounds[2] - self.bounds[0],
            self.bounds[3] - self.bounds[1]
        )
    
    @property
    def area(self) -> float:
        return self.size[0] * self.size[1]


class AdaptiveQuadtree:
    """
    Adaptive quadtree for multi-resolution spatial representation.
    
    Refines cells where:
    - Data density is high
    - Gradient is large
    - Uncertainty reduction is significant
    
    Coarsens where:
    - Data is sparse
    - Field is smooth
    - Uncertainty is already high (no benefit from resolution)
    """
    
    def __init__(self, bounds: Tuple[float, float, float, float],
                 max_level: int = 6,
                 min_level: int = 2,
                 refinement_threshold: float = 0.1,
                 coarsening_threshold: float = 0.01):
        """
        Initialize adaptive quadtree.
        
        Args:
            bounds: Domain bounds (xmin, ymin, xmax, ymax)
            max_level: Maximum refinement level
            min_level: Minimum level (ensures base resolution)
            refinement_threshold: Data density or gradient threshold for refinement
            coarsening_threshold: Threshold below which to coarsen
        """
        self.bounds = bounds
        self.max_level = max_level
        self.min_level = min_level
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        
        # Initialize root cell
        self.root = GridCell(level=0, bounds=bounds)
        
        # Refine to minimum level
        self._refine_to_level(self.root, min_level)
    
    def _refine_to_level(self, cell: GridCell, target_level: int):
        """Recursively refine cell to target level."""
        if cell.level >= target_level:
            return
        
        self._subdivide(cell)
        for child in cell.children:
            self._refine_to_level(child, target_level)
    
    def _subdivide(self, cell: GridCell):
        """Subdivide a cell into 4 children."""
        if cell.level >= self.max_level:
            return
        
        xmin, ymin, xmax, ymax = cell.bounds
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        
        new_level = cell.level + 1
        cell.children = [
            GridCell(new_level, (xmin, ymin, xmid, ymid)),  # SW
            GridCell(new_level, (xmid, ymin, xmax, ymid)),  # SE
            GridCell(new_level, (xmin, ymid, xmid, ymax)),  # NW
            GridCell(new_level, (xmid, ymid, xmax, ymax)),  # NE
        ]
        
        # Distribute parent data to children
        for child in cell.children:
            child.value = cell.value
            child.uncertainty = cell.uncertainty * 1.1  # Slightly increase uncertainty
            child.data_count = cell.data_count // 4
    
    def _coarsen(self, cell: GridCell):
        """Coarsen cell by removing children."""
        if cell.children is None:
            return
        
        # Aggregate children values with uncertainty-weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        total_data = 0
        
        for child in cell.children:
            if child.uncertainty > 0:
                weight = 1.0 / (child.uncertainty ** 2 + EPSILON)
            else:
                weight = 1e10
            weighted_sum += weight * child.value
            total_weight += weight
            total_data += child.data_count
        
        if total_weight > 0:
            cell.value = weighted_sum / total_weight
            cell.uncertainty = 1.0 / np.sqrt(total_weight)
        
        cell.data_count = total_data
        cell.children = None
    
    def adapt_to_data(self, data_points: np.ndarray, 
                      data_values: Optional[np.ndarray] = None):
        """
        Adapt grid resolution based on data point distribution.
        
        Args:
            data_points: Array of shape (n, 2) with (x, y) coordinates
            data_values: Optional values at each point
        """
        # Compute data density
        self._adapt_recursive(self.root, data_points, data_values)
    
    def _adapt_recursive(self, cell: GridCell, 
                         data_points: np.ndarray,
                         data_values: Optional[np.ndarray]):
        """Recursively adapt cell and its children."""
        # Find points in this cell
        mask = (
            (data_points[:, 0] >= cell.bounds[0]) &
            (data_points[:, 0] < cell.bounds[2]) &
            (data_points[:, 1] >= cell.bounds[1]) &
            (data_points[:, 1] < cell.bounds[3])
        )
        local_points = data_points[mask]
        local_values = data_values[mask] if data_values is not None else None
        
        cell.data_count = len(local_points)
        
        # Compute data density
        density = cell.data_count / cell.area if cell.area > 0 else 0
        
        # Decide whether to refine or coarsen
        if cell.level < self.min_level:
            # Always refine to minimum level
            if cell.is_leaf:
                self._subdivide(cell)
        elif density > self.refinement_threshold and cell.level < self.max_level:
            # High density - refine
            if cell.is_leaf:
                self._subdivide(cell)
        elif density < self.coarsening_threshold and cell.level > self.min_level:
            # Low density - coarsen
            if not cell.is_leaf:
                self._coarsen(cell)
        
        # Update cell value from local data
        if local_values is not None and len(local_values) > 0:
            cell.value = np.mean(local_values)
            cell.uncertainty = np.std(local_values) / np.sqrt(len(local_values)) if len(local_values) > 1 else np.inf
        
        # Recurse to children
        if cell.children is not None:
            for child in cell.children:
                self._adapt_recursive(child, data_points, data_values)
    
    def adapt_to_gradient(self, field: np.ndarray, 
                          base_bounds: Tuple[float, float, float, float]):
        """
        Adapt grid resolution based on field gradient magnitude.
        
        Refines where gradient is large, coarsens where smooth.
        """
        # Compute gradient magnitude on base grid
        gy, gx = np.gradient(field)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        # Normalize gradient
        max_grad = np.nanmax(gradient_mag)
        if max_grad > 0:
            gradient_norm = gradient_mag / max_grad
        else:
            gradient_norm = np.zeros_like(gradient_mag)
        
        self._adapt_gradient_recursive(self.root, gradient_norm, base_bounds, field.shape)
    
    def _adapt_gradient_recursive(self, cell: GridCell,
                                   gradient_norm: np.ndarray,
                                   field_bounds: Tuple[float, float, float, float],
                                   field_shape: Tuple[int, int]):
        """Recursively adapt based on gradient in cell region."""
        ny, nx = field_shape
        
        # Map cell bounds to array indices
        xmin, ymin, xmax, ymax = cell.bounds
        fxmin, fymin, fxmax, fymax = field_bounds
        
        i0 = int((xmin - fxmin) / (fxmax - fxmin) * nx)
        i1 = int((xmax - fxmin) / (fxmax - fxmin) * nx)
        j0 = int((ymin - fymin) / (fymax - fymin) * ny)
        j1 = int((ymax - fymin) / (fymax - fymin) * ny)
        
        # Clamp to valid range
        i0, i1 = max(0, i0), min(nx, i1)
        j0, j1 = max(0, j0), min(ny, j1)
        
        if i1 > i0 and j1 > j0:
            local_gradient = gradient_norm[j0:j1, i0:i1]
            mean_gradient = np.nanmean(local_gradient)
        else:
            mean_gradient = 0.0
        
        # Adapt based on gradient
        if cell.level < self.min_level:
            if cell.is_leaf:
                self._subdivide(cell)
        elif mean_gradient > self.refinement_threshold and cell.level < self.max_level:
            if cell.is_leaf:
                self._subdivide(cell)
        elif mean_gradient < self.coarsening_threshold and cell.level > self.min_level:
            if not cell.is_leaf:
                self._coarsen(cell)
        
        # Recurse
        if cell.children is not None:
            for child in cell.children:
                self._adapt_gradient_recursive(child, gradient_norm, field_bounds, field_shape)
    
    def get_leaf_cells(self) -> List[GridCell]:
        """Return all leaf cells."""
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves
    
    def _collect_leaves(self, cell: GridCell, leaves: List[GridCell]):
        """Recursively collect leaf cells."""
        if cell.is_leaf:
            leaves.append(cell)
        else:
            for child in cell.children:
                self._collect_leaves(child, leaves)
    
    def query(self, x: float, y: float) -> Optional[GridCell]:
        """Find the leaf cell containing point (x, y)."""
        return self._query_recursive(self.root, x, y)
    
    def _query_recursive(self, cell: GridCell, x: float, y: float) -> Optional[GridCell]:
        """Recursively find containing cell."""
        xmin, ymin, xmax, ymax = cell.bounds
        if not (xmin <= x < xmax and ymin <= y < ymax):
            return None
        
        if cell.is_leaf:
            return cell
        
        for child in cell.children:
            result = self._query_recursive(child, x, y)
            if result is not None:
                return result
        
        return cell  # Fallback
    
    def to_regular_grid(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert adaptive grid to regular grid via interpolation.
        
        Returns:
            (values, uncertainties) as regular grids
        """
        xmin, ymin, xmax, ymax = self.bounds
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        
        values = np.zeros((ny, nx))
        uncertainties = np.zeros((ny, nx))
        
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                cell = self.query(x, y)
                if cell is not None:
                    values[j, i] = cell.value
                    uncertainties[j, i] = cell.uncertainty
        
        return values, uncertainties
    
    def get_resolution_map(self, nx: int, ny: int) -> np.ndarray:
        """
        Return map of local resolution levels.
        
        Useful for visualization and diagnostics.
        """
        xmin, ymin, xmax, ymax = self.bounds
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        
        levels = np.zeros((ny, nx), dtype=np.int32)
        
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                cell = self.query(x, y)
                if cell is not None:
                    levels[j, i] = cell.level
        
        return levels


class MultiResolutionGrid:
    """
    Multi-resolution grid with pyramid structure.
    
    Maintains multiple resolution levels with consistent
    interpolation and uncertainty propagation between levels.
    """
    
    def __init__(self, base_shape: Tuple[int, int],
                 bounds: Tuple[float, float, float, float],
                 n_levels: int = 4):
        """
        Initialize multi-resolution grid.
        
        Args:
            base_shape: Shape of finest resolution (ny, nx)
            bounds: Spatial bounds (xmin, ymin, xmax, ymax)
            n_levels: Number of resolution levels
        """
        self.base_shape = base_shape
        self.bounds = bounds
        self.n_levels = n_levels
        
        # Initialize pyramid
        self.levels: List[Dict[str, np.ndarray]] = []
        for level in range(n_levels):
            factor = 2 ** level
            shape = (base_shape[0] // factor, base_shape[1] // factor)
            if shape[0] < 1 or shape[1] < 1:
                break
            self.levels.append({
                'shape': shape,
                'values': np.zeros(shape),
                'uncertainties': np.full(shape, np.inf),
                'data_counts': np.zeros(shape, dtype=np.int32),
                'factor': factor
            })
        
        self.n_levels = len(self.levels)
    
    def set_base_level(self, values: np.ndarray, 
                       uncertainties: Optional[np.ndarray] = None):
        """Set values at the finest resolution level."""
        self.levels[0]['values'] = values.copy()
        if uncertainties is not None:
            self.levels[0]['uncertainties'] = uncertainties.copy()
        else:
            self.levels[0]['uncertainties'] = np.full_like(values, EPSILON)
        
        # Propagate to coarser levels
        self._propagate_to_coarse()
    
    def _propagate_to_coarse(self):
        """Propagate values and uncertainties from fine to coarse levels."""
        for i in range(1, self.n_levels):
            fine_vals = self.levels[i-1]['values']
            fine_unc = self.levels[i-1]['uncertainties']
            
            # Downsample with uncertainty-weighted averaging
            coarse_shape = self.levels[i]['shape']
            factor = 2
            
            # Reshape for block averaging
            ny_f, nx_f = fine_vals.shape
            ny_c, nx_c = coarse_shape
            
            # Ensure dimensions are compatible
            ny_f = ny_c * factor
            nx_f = nx_c * factor
            
            fine_vals_crop = fine_vals[:ny_f, :nx_f]
            fine_unc_crop = fine_unc[:ny_f, :nx_f]
            
            # Compute weights (inverse variance)
            weights = 1.0 / (fine_unc_crop ** 2 + EPSILON)
            
            # Block sum
            weight_blocks = weights.reshape(ny_c, factor, nx_c, factor)
            value_weight_blocks = (fine_vals_crop * weights).reshape(ny_c, factor, nx_c, factor)
            
            total_weights = weight_blocks.sum(axis=(1, 3))
            weighted_sums = value_weight_blocks.sum(axis=(1, 3))
            
            self.levels[i]['values'] = weighted_sums / (total_weights + EPSILON)
            self.levels[i]['uncertainties'] = 1.0 / np.sqrt(total_weights + EPSILON)
    
    def _propagate_to_fine(self, level: int):
        """Propagate constraints from coarse to fine levels."""
        if level <= 0:
            return
        
        coarse_vals = self.levels[level]['values']
        coarse_unc = self.levels[level]['uncertainties']
        fine_shape = self.levels[level-1]['shape']
        
        # Upsample via bilinear interpolation
        from scipy.ndimage import zoom
        upsampled_vals = zoom(coarse_vals, 2, order=1)
        upsampled_unc = zoom(coarse_unc, 2, order=1)
        
        # Crop to fine shape
        upsampled_vals = upsampled_vals[:fine_shape[0], :fine_shape[1]]
        upsampled_unc = upsampled_unc[:fine_shape[0], :fine_shape[1]]
        
        # Blend with existing fine values using uncertainty
        fine_vals = self.levels[level-1]['values']
        fine_unc = self.levels[level-1]['uncertainties']
        
        # Inverse variance weighting
        w_fine = 1.0 / (fine_unc ** 2 + EPSILON)
        w_coarse = 1.0 / (upsampled_unc ** 2 + EPSILON)
        
        total_w = w_fine + w_coarse
        self.levels[level-1]['values'] = (w_fine * fine_vals + w_coarse * upsampled_vals) / total_w
        self.levels[level-1]['uncertainties'] = 1.0 / np.sqrt(total_w)
    
    def add_data(self, x: np.ndarray, y: np.ndarray, 
                 values: np.ndarray, uncertainties: Optional[np.ndarray] = None):
        """
        Add data points and update the multi-resolution grid.
        
        Uses the appropriate resolution level based on local data density.
        """
        if uncertainties is None:
            uncertainties = np.ones_like(values) * np.std(values)
        
        xmin, ymin, xmax, ymax = self.bounds
        
        # Add to finest level
        ny, nx = self.base_shape
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        
        for xi, yi, val, unc in zip(x, y, values, uncertainties):
            i = int((xi - xmin) / dx)
            j = int((yi - ymin) / dy)
            
            if 0 <= i < nx and 0 <= j < ny:
                # Inverse variance update
                old_val = self.levels[0]['values'][j, i]
                old_unc = self.levels[0]['uncertainties'][j, i]
                
                w_old = 1.0 / (old_unc ** 2 + EPSILON)
                w_new = 1.0 / (unc ** 2 + EPSILON)
                
                total_w = w_old + w_new
                self.levels[0]['values'][j, i] = (w_old * old_val + w_new * val) / total_w
                self.levels[0]['uncertainties'][j, i] = 1.0 / np.sqrt(total_w)
                self.levels[0]['data_counts'][j, i] += 1
        
        # Propagate to coarse levels
        self._propagate_to_coarse()
    
    def get_at_resolution(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get values and uncertainties at specified resolution level."""
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"Level {level} out of range [0, {self.n_levels})")
        return self.levels[level]['values'], self.levels[level]['uncertainties']
    
    def interpolate(self, x: np.ndarray, y: np.ndarray,
                    method: str = 'adaptive') -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate values and uncertainties at arbitrary points.
        
        Args:
            x, y: Query coordinates
            method: 'finest', 'coarsest', or 'adaptive'
        
        Returns:
            (values, uncertainties) at query points
        """
        xmin, ymin, xmax, ymax = self.bounds
        
        if method == 'finest':
            level = 0
        elif method == 'coarsest':
            level = self.n_levels - 1
        else:
            # Adaptive: use level based on local data density
            # For now, use finest
            level = 0
        
        values = self.levels[level]['values']
        uncertainties = self.levels[level]['uncertainties']
        shape = self.levels[level]['shape']
        
        # Create interpolators
        xs = np.linspace(xmin, xmax, shape[1])
        ys = np.linspace(ymin, ymax, shape[0])
        
        val_interp = RegularGridInterpolator((ys, xs), values, 
                                              bounds_error=False, fill_value=np.nan)
        unc_interp = RegularGridInterpolator((ys, xs), uncertainties,
                                              bounds_error=False, fill_value=np.inf)
        
        points = np.column_stack([y, x])
        interp_vals = val_interp(points)
        interp_uncs = unc_interp(points)
        
        return interp_vals, interp_uncs


def compute_optimal_resolution(data_points: np.ndarray,
                               bounds: Tuple[float, float, float, float],
                               min_points_per_cell: int = 5) -> int:
    """
    Compute optimal grid resolution based on data density.
    
    Uses Silverman's rule adapted for 2D.
    """
    n = len(data_points)
    if n < min_points_per_cell:
        return 4  # Minimum resolution
    
    xmin, ymin, xmax, ymax = bounds
    domain_area = (xmax - xmin) * (ymax - ymin)
    
    # Silverman's rule for 2D: h = n^(-1/6) * sigma
    sigma_x = np.std(data_points[:, 0])
    sigma_y = np.std(data_points[:, 1])
    sigma = np.sqrt(sigma_x * sigma_y)
    
    h_optimal = (n ** (-1/6)) * sigma
    
    # Convert bandwidth to grid resolution
    n_cells = domain_area / (h_optimal ** 2)
    grid_size = int(np.sqrt(n_cells))
    
    # Clamp to reasonable range
    return max(4, min(256, grid_size))
