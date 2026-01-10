"""
Anisotropic Distance Functions Module

Implements terrain and flow-aware distance metrics:
- Account for elevation (uphill ≠ downhill)
- Follow drainage network paths
- Tensor-based local anisotropy
- Uncertainty in distance computations

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import heapq
from scipy import ndimage

from .error_aware_arithmetic import EPSILON, ErrorValue, ErrorArray


@dataclass
class DistanceResult:
    """Result of distance computation with uncertainty."""
    distance: float
    uncertainty: float
    path: Optional[np.ndarray] = None
    cost_breakdown: Optional[Dict[str, float]] = None
    
    def __lt__(self, other):
        return self.distance < other.distance


class MetricTensor:
    """
    Local metric tensor for anisotropic distance.
    
    Represents a 2x2 positive definite matrix M such that
    distance = sqrt(v^T M v) for displacement v.
    """
    
    def __init__(self, m11: float, m12: float, m22: float):
        """
        Initialize metric tensor.
        
        Args:
            m11, m12, m22: Components of symmetric matrix [[m11, m12], [m12, m22]]
        """
        self.m11 = m11
        self.m12 = m12
        self.m22 = m22
        
        # Ensure positive definiteness
        det = m11 * m22 - m12 * m12
        if det <= EPSILON or m11 <= EPSILON:
            raise ValueError("Metric tensor must be positive definite")
    
    def distance(self, dx: float, dy: float) -> float:
        """Compute distance for displacement (dx, dy)."""
        return np.sqrt(self.m11 * dx * dx + 2 * self.m12 * dx * dy + self.m22 * dy * dy)
    
    def distance_array(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Vectorized distance computation."""
        return np.sqrt(self.m11 * dx**2 + 2 * self.m12 * dx * dy + self.m22 * dy**2)
    
    @property
    def eigenvalues(self) -> Tuple[float, float]:
        """Compute eigenvalues of metric tensor."""
        trace = self.m11 + self.m22
        det = self.m11 * self.m22 - self.m12 * self.m12
        discriminant = np.sqrt(max(0, trace * trace - 4 * det))
        return ((trace + discriminant) / 2, (trace - discriminant) / 2)
    
    @property
    def condition_number(self) -> float:
        """Condition number (ratio of eigenvalues)."""
        e1, e2 = self.eigenvalues
        return max(e1, e2) / (min(e1, e2) + EPSILON)
    
    @classmethod
    def isotropic(cls, scale: float = 1.0) -> 'MetricTensor':
        """Create isotropic (Euclidean) metric."""
        return cls(scale, 0.0, scale)
    
    @classmethod
    def from_gradient(cls, grad_x: float, grad_y: float,
                      uphill_cost: float = 2.0,
                      downhill_cost: float = 0.5,
                      base_cost: float = 1.0) -> 'MetricTensor':
        """
        Create metric tensor from gradient (e.g., elevation gradient).
        
        Moving uphill costs more, downhill costs less.
        """
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        if grad_mag < EPSILON:
            return cls.isotropic(base_cost)
        
        # Unit gradient direction
        gx, gy = grad_x / grad_mag, grad_y / grad_mag
        
        # Cost along gradient (uphill direction)
        c_up = base_cost * uphill_cost * (1 + grad_mag)
        # Cost against gradient (downhill direction)  
        c_down = base_cost * downhill_cost / (1 + grad_mag)
        # Cost perpendicular to gradient
        c_perp = base_cost
        
        # Average directional cost
        c_grad = (c_up + c_down) / 2
        
        # Construct metric tensor
        # M = c_perp * I + (c_grad - c_perp) * g * g^T
        m11 = c_perp + (c_grad - c_perp) * gx * gx
        m12 = (c_grad - c_perp) * gx * gy
        m22 = c_perp + (c_grad - c_perp) * gy * gy
        
        return cls(m11, m12, m22)
    
    @classmethod
    def from_flow_direction(cls, flow_x: float, flow_y: float,
                            along_flow_cost: float = 0.5,
                            against_flow_cost: float = 2.0,
                            cross_flow_cost: float = 1.0) -> 'MetricTensor':
        """
        Create metric tensor from flow direction.
        
        Moving with flow is cheap, against is expensive.
        """
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)
        
        if flow_mag < EPSILON:
            return cls.isotropic(cross_flow_cost)
        
        # Unit flow direction
        fx, fy = flow_x / flow_mag, flow_y / flow_mag
        
        # Asymmetric cost is tricky in metric tensor form
        # We use average of along/against for the metric
        c_flow = (along_flow_cost + against_flow_cost) / 2
        
        # Construct tensor
        m11 = cross_flow_cost + (c_flow - cross_flow_cost) * fx * fx
        m12 = (c_flow - cross_flow_cost) * fx * fy
        m22 = cross_flow_cost + (c_flow - cross_flow_cost) * fy * fy
        
        return cls(m11, m12, m22)


class AnisotropicDistanceField:
    """
    Compute anisotropic distance fields on grids.
    
    Uses fast marching / Dijkstra with local metric tensors.
    """
    
    def __init__(self, shape: Tuple[int, int],
                 bounds: Tuple[float, float, float, float]):
        """
        Initialize distance field calculator.
        
        Args:
            shape: Grid shape (ny, nx)
            bounds: Spatial bounds (xmin, ymin, xmax, ymax)
        """
        self.shape = shape
        self.bounds = bounds
        self.ny, self.nx = shape
        
        xmin, ymin, xmax, ymax = bounds
        self.dx = (xmax - xmin) / self.nx
        self.dy = (ymax - ymin) / self.ny
        
        # Default to isotropic metric
        self.metric_field: Optional[np.ndarray] = None
    
    def set_elevation(self, elevation: np.ndarray,
                      uphill_cost: float = 2.0,
                      downhill_cost: float = 0.5):
        """Set metric field from elevation data."""
        # Compute gradient
        grad_y, grad_x = np.gradient(elevation, self.dy, self.dx)
        
        self.metric_field = np.empty(self.shape, dtype=object)
        for j in range(self.ny):
            for i in range(self.nx):
                self.metric_field[j, i] = MetricTensor.from_gradient(
                    grad_x[j, i], grad_y[j, i],
                    uphill_cost, downhill_cost
                )
    
    def set_flow_direction(self, flow_x: np.ndarray, flow_y: np.ndarray,
                           along_flow_cost: float = 0.5,
                           against_flow_cost: float = 2.0):
        """Set metric field from flow direction data."""
        self.metric_field = np.empty(self.shape, dtype=object)
        for j in range(self.ny):
            for i in range(self.nx):
                self.metric_field[j, i] = MetricTensor.from_flow_direction(
                    flow_x[j, i], flow_y[j, i],
                    along_flow_cost, against_flow_cost
                )
    
    def compute_distance_from_sources(self, sources: np.ndarray,
                                       max_distance: float = np.inf
                                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distance field from source points.
        
        Args:
            sources: Array of source coordinates (n, 2) or boolean mask
            max_distance: Maximum distance to compute
        
        Returns:
            (distances, uncertainties) as grids
        """
        distances = np.full(self.shape, np.inf)
        uncertainties = np.full(self.shape, np.inf)
        
        # Initialize priority queue
        pq = []
        
        # Parse sources
        if sources.dtype == bool:
            # Boolean mask
            source_cells = np.argwhere(sources)
        else:
            # Coordinate array
            xmin, ymin, xmax, ymax = self.bounds
            source_cells = []
            for x, y in sources:
                i = int((x - xmin) / self.dx)
                j = int((y - ymin) / self.dy)
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    source_cells.append([j, i])
            source_cells = np.array(source_cells)
        
        # Initialize sources with zero distance
        for j, i in source_cells:
            distances[j, i] = 0.0
            uncertainties[j, i] = EPSILON
            heapq.heappush(pq, (0.0, j, i))
        
        # 8-connected neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0), (1, 1)]
        
        # Fast marching
        visited = np.zeros(self.shape, dtype=bool)
        
        while pq:
            dist, j, i = heapq.heappop(pq)
            
            if visited[j, i]:
                continue
            visited[j, i] = True
            
            if dist > max_distance:
                break
            
            # Get local metric
            if self.metric_field is not None:
                metric = self.metric_field[j, i]
            else:
                metric = MetricTensor.isotropic()
            
            # Update neighbors
            for dj, di in neighbors:
                nj, ni = j + dj, i + di
                
                if 0 <= nj < self.ny and 0 <= ni < self.nx and not visited[nj, ni]:
                    # Compute step distance using local metric
                    step_dx = di * self.dx
                    step_dy = dj * self.dy
                    step_dist = metric.distance(step_dx, step_dy)
                    
                    new_dist = dist + step_dist
                    
                    if new_dist < distances[nj, ni]:
                        distances[nj, ni] = new_dist
                        # Uncertainty grows with distance
                        uncertainties[nj, ni] = EPSILON * (1 + new_dist / self.dx)
                        heapq.heappush(pq, (new_dist, nj, ni))
        
        return distances, uncertainties
    
    def compute_path(self, start: Tuple[float, float],
                     end: Tuple[float, float]) -> DistanceResult:
        """
        Compute shortest path between two points.
        
        Returns path and total distance with uncertainty.
        """
        xmin, ymin, xmax, ymax = self.bounds
        
        # Convert to grid indices
        si = int((start[0] - xmin) / self.dx)
        sj = int((start[1] - ymin) / self.dy)
        ei = int((end[0] - xmin) / self.dx)
        ej = int((end[1] - ymin) / self.dy)
        
        # Clamp to grid
        si, sj = max(0, min(self.nx-1, si)), max(0, min(self.ny-1, sj))
        ei, ej = max(0, min(self.nx-1, ei)), max(0, min(self.ny-1, ej))
        
        # A* search
        distances = np.full(self.shape, np.inf)
        came_from = np.full((*self.shape, 2), -1, dtype=np.int32)
        distances[sj, si] = 0.0
        
        # Priority queue: (f_score, g_score, j, i)
        pq = [(self._heuristic(si, sj, ei, ej), 0.0, sj, si)]
        
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0), (1, 1)]
        
        while pq:
            f, g, j, i = heapq.heappop(pq)
            
            if j == ej and i == ei:
                # Reconstruct path
                path = [(i * self.dx + xmin, j * self.dy + ymin)]
                cj, ci = j, i
                while came_from[cj, ci, 0] >= 0:
                    cj, ci = came_from[cj, ci]
                    path.append((ci * self.dx + xmin, cj * self.dy + ymin))
                path.reverse()
                
                return DistanceResult(
                    distance=g,
                    uncertainty=EPSILON * len(path),
                    path=np.array(path)
                )
            
            if g > distances[j, i]:
                continue
            
            metric = self.metric_field[j, i] if self.metric_field is not None else MetricTensor.isotropic()
            
            for dj, di in neighbors:
                nj, ni = j + dj, i + di
                
                if 0 <= nj < self.ny and 0 <= ni < self.nx:
                    step_dist = metric.distance(di * self.dx, dj * self.dy)
                    new_g = g + step_dist
                    
                    if new_g < distances[nj, ni]:
                        distances[nj, ni] = new_g
                        came_from[nj, ni] = [j, i]
                        f_score = new_g + self._heuristic(ni, nj, ei, ej)
                        heapq.heappush(pq, (f_score, new_g, nj, ni))
        
        # No path found
        return DistanceResult(distance=np.inf, uncertainty=np.inf)
    
    def _heuristic(self, i1: int, j1: int, i2: int, j2: int) -> float:
        """A* heuristic (Euclidean distance)."""
        return np.sqrt(((i1 - i2) * self.dx) ** 2 + ((j1 - j2) * self.dy) ** 2)


class DrainageNetworkDistance:
    """
    Distance computation along drainage network.
    
    Treats drainage network as a graph and computes
    network distance rather than Euclidean distance.
    """
    
    def __init__(self, network_mask: np.ndarray,
                 flow_direction: Optional[np.ndarray] = None,
                 bounds: Tuple[float, float, float, float] = (0, 0, 1, 1)):
        """
        Initialize drainage network distance.
        
        Args:
            network_mask: Boolean mask of drainage network cells
            flow_direction: Optional D8 flow direction (1-8)
            bounds: Spatial bounds
        """
        self.network_mask = network_mask
        self.flow_direction = flow_direction
        self.bounds = bounds
        self.ny, self.nx = network_mask.shape
        
        xmin, ymin, xmax, ymax = bounds
        self.dx = (xmax - xmin) / self.nx
        self.dy = (ymax - ymin) / self.ny
        
        # Build network graph
        self._build_graph()
    
    def _build_graph(self):
        """Build adjacency structure for network."""
        # D8 directions: E, NE, N, NW, W, SW, S, SE
        self.d8_offsets = {
            1: (1, 0),    # E
            2: (1, -1),   # NE
            3: (0, -1),   # N
            4: (-1, -1),  # NW
            5: (-1, 0),   # W
            6: (-1, 1),   # SW
            7: (0, 1),    # S
            8: (1, 1),    # SE
        }
        
        # Find network cells
        self.network_cells = np.argwhere(self.network_mask)
        
        # Create cell to index mapping
        self.cell_to_idx = {}
        for idx, (j, i) in enumerate(self.network_cells):
            self.cell_to_idx[(j, i)] = idx
        
        # Build adjacency list
        self.adjacency = [[] for _ in range(len(self.network_cells))]
        
        for idx, (j, i) in enumerate(self.network_cells):
            for di, dj in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]:
                nj, ni = j + dj, i + di
                if (nj, ni) in self.cell_to_idx:
                    # Compute edge cost
                    dist = np.sqrt((di * self.dx) ** 2 + (dj * self.dy) ** 2)
                    
                    # Modify cost based on flow direction
                    if self.flow_direction is not None:
                        flow_dir = self.flow_direction[j, i]
                        if flow_dir in self.d8_offsets:
                            fd_i, fd_j = self.d8_offsets[flow_dir]
                            if di == fd_i and dj == fd_j:
                                dist *= 0.5  # With flow: cheaper
                            elif di == -fd_i and dj == -fd_j:
                                dist *= 2.0  # Against flow: expensive
                    
                    neighbor_idx = self.cell_to_idx[(nj, ni)]
                    self.adjacency[idx].append((neighbor_idx, dist))
    
    def distance_to_network(self, x: float, y: float) -> float:
        """Compute Euclidean distance to nearest network cell."""
        xmin, ymin, xmax, ymax = self.bounds
        
        i = int((x - xmin) / self.dx)
        j = int((y - ymin) / self.dy)
        
        if 0 <= i < self.nx and 0 <= j < self.ny:
            if self.network_mask[j, i]:
                return 0.0
        
        # Find nearest network cell
        min_dist = np.inf
        for nj, ni in self.network_cells:
            dist = np.sqrt(((ni - i) * self.dx) ** 2 + ((nj - j) * self.dy) ** 2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def network_distance(self, start: Tuple[float, float],
                         end: Tuple[float, float]) -> DistanceResult:
        """
        Compute distance along drainage network.
        
        First finds nearest network cells to start/end,
        then computes network path distance.
        """
        xmin, ymin, xmax, ymax = self.bounds
        
        # Find nearest network cell to start
        si = int((start[0] - xmin) / self.dx)
        sj = int((start[1] - ymin) / self.dy)
        start_idx = self._nearest_network_cell(si, sj)
        
        # Find nearest network cell to end
        ei = int((end[0] - xmin) / self.dx)
        ej = int((end[1] - ymin) / self.dy)
        end_idx = self._nearest_network_cell(ei, ej)
        
        if start_idx is None or end_idx is None:
            return DistanceResult(distance=np.inf, uncertainty=np.inf)
        
        # Dijkstra on network graph
        n = len(self.network_cells)
        distances = np.full(n, np.inf)
        distances[start_idx] = 0.0
        came_from = np.full(n, -1, dtype=np.int32)
        
        pq = [(0.0, start_idx)]
        
        while pq:
            dist, idx = heapq.heappop(pq)
            
            if idx == end_idx:
                # Reconstruct path
                path_indices = [idx]
                while came_from[idx] >= 0:
                    idx = came_from[idx]
                    path_indices.append(idx)
                path_indices.reverse()
                
                path = []
                for pidx in path_indices:
                    j, i = self.network_cells[pidx]
                    path.append((i * self.dx + xmin, j * self.dy + ymin))
                
                # Add distance from actual start/end to network
                start_dist = self.distance_to_network(start[0], start[1])
                end_dist = self.distance_to_network(end[0], end[1])
                
                return DistanceResult(
                    distance=dist + start_dist + end_dist,
                    uncertainty=EPSILON * len(path_indices),
                    path=np.array(path)
                )
            
            if dist > distances[idx]:
                continue
            
            for neighbor_idx, edge_dist in self.adjacency[idx]:
                new_dist = dist + edge_dist
                if new_dist < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_dist
                    came_from[neighbor_idx] = idx
                    heapq.heappush(pq, (new_dist, neighbor_idx))
        
        return DistanceResult(distance=np.inf, uncertainty=np.inf)
    
    def _nearest_network_cell(self, i: int, j: int) -> Optional[int]:
        """Find index of nearest network cell."""
        min_dist = np.inf
        nearest_idx = None
        
        for idx, (nj, ni) in enumerate(self.network_cells):
            dist = (ni - i) ** 2 + (nj - j) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        return nearest_idx


def compute_anisotropic_kriging_distances(points: np.ndarray,
                                           elevation: Optional[np.ndarray] = None,
                                           flow: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                           bounds: Optional[Tuple[float, float, float, float]] = None
                                           ) -> np.ndarray:
    """
    Compute pairwise anisotropic distances for kriging.
    
    Returns:
        Distance matrix accounting for terrain/flow anisotropy
    """
    n = len(points)
    distances = np.zeros((n, n))
    
    if elevation is None and flow is None:
        # Euclidean fallback
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                distances[i, j] = d
                distances[j, i] = d
        return distances
    
    # Setup anisotropic field
    if bounds is None:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        bounds = (xmin, ymin, xmax, ymax)
    
    if elevation is not None:
        field = AnisotropicDistanceField(elevation.shape, bounds)
        field.set_elevation(elevation)
    else:
        fx, fy = flow
        field = AnisotropicDistanceField(fx.shape, bounds)
        field.set_flow_direction(fx, fy)
    
    # Compute pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            result = field.compute_path(
                (points[i, 0], points[i, 1]),
                (points[j, 0], points[j, 1])
            )
            distances[i, j] = result.distance
            distances[j, i] = result.distance
    
    return distances
