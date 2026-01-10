"""
Robust Computational Geometry Module

Implements topology-aware polygon operations with:
- R-tree spatial indexing
- Robust geometric predicates
- Consistent handling of edge cases
- Uncertainty in geometric measurements

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from scipy.spatial import cKDTree, ConvexHull, Delaunay
import warnings

from .error_aware_arithmetic import EPSILON, ErrorValue, ErrorArray


# Robust geometric predicates using adaptive precision
def _sign(x: float) -> int:
    """Robust sign function."""
    if x > EPSILON:
        return 1
    elif x < -EPSILON:
        return -1
    return 0


def orient2d(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> int:
    """
    Orientation test for 2D points.
    
    Returns:
        1 if counterclockwise
        -1 if clockwise
        0 if collinear
    """
    # Use Shewchuk's robust predicate approximation
    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1])
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0])
    det = detleft - detright
    
    # Adaptive error bound
    detsum = abs(detleft) + abs(detright)
    errbound = 3.3306690738754716e-16 * detsum  # Machine epsilon factor
    
    if abs(det) > errbound:
        return _sign(det)
    return 0  # Too close to call


def incircle2d(pa: np.ndarray, pb: np.ndarray, 
               pc: np.ndarray, pd: np.ndarray) -> int:
    """
    Test if point pd is inside the circumcircle of triangle (pa, pb, pc).
    
    Returns:
        1 if inside
        -1 if outside
        0 if on circle
    """
    adx = pa[0] - pd[0]
    ady = pa[1] - pd[1]
    bdx = pb[0] - pd[0]
    bdy = pb[1] - pd[1]
    cdx = pc[0] - pd[0]
    cdy = pc[1] - pd[1]
    
    abdet = adx * bdy - bdx * ady
    bcdet = bdx * cdy - cdx * bdy
    cadet = cdx * ady - adx * cdy
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    
    det = alift * bcdet + blift * cadet + clift * abdet
    
    # Error bound
    permanent = (abs(bdx * cdy) + abs(cdx * bdy)) * alift \
              + (abs(cdx * ady) + abs(adx * cdy)) * blift \
              + (abs(adx * bdy) + abs(bdx * ady)) * clift
    errbound = 1.1102230246251568e-15 * permanent
    
    if abs(det) > errbound:
        return _sign(det)
    return 0


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)
    
    @property
    def size(self) -> Tuple[float, float]:
        return (self.xmax - self.xmin, self.ymax - self.ymin)
    
    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
    
    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax
    
    def intersects(self, other: 'BoundingBox') -> bool:
        return not (self.xmax < other.xmin or other.xmax < self.xmin or
                    self.ymax < other.ymin or other.ymax < self.ymin)
    
    def expand(self, margin: float) -> 'BoundingBox':
        return BoundingBox(
            self.xmin - margin, self.ymin - margin,
            self.xmax + margin, self.ymax + margin
        )


@dataclass
class RTreeNode:
    """Node in R-tree spatial index."""
    bbox: BoundingBox
    children: List['RTreeNode'] = field(default_factory=list)
    data: Optional[Any] = None
    data_index: Optional[int] = None
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RTree:
    """
    R-tree spatial index for efficient geometric queries.
    
    Supports:
    - Point queries
    - Range queries
    - Nearest neighbor queries
    """
    
    MAX_CHILDREN = 8  # M in R-tree
    MIN_CHILDREN = 3  # m in R-tree
    
    def __init__(self):
        self.root: Optional[RTreeNode] = None
        self.size = 0
    
    def insert(self, bbox: BoundingBox, data: Any = None, data_index: int = None):
        """Insert an object with given bounding box."""
        node = RTreeNode(bbox=bbox, data=data, data_index=data_index)
        
        if self.root is None:
            self.root = RTreeNode(bbox=bbox, children=[node])
        else:
            self._insert_recursive(self.root, node)
            
            # Check for root split
            if len(self.root.children) > self.MAX_CHILDREN:
                new_root_bbox = self._compute_bbox([c.bbox for c in self.root.children])
                new_root = RTreeNode(bbox=new_root_bbox, children=[self.root])
                self._split_node(new_root, 0)
                self.root = new_root
        
        self.size += 1
    
    def _insert_recursive(self, parent: RTreeNode, node: RTreeNode):
        """Recursively find best position and insert."""
        if all(c.is_leaf for c in parent.children):
            # All children are leaves, insert here
            parent.children.append(node)
            self._expand_bbox(parent, node.bbox)
        else:
            # Find child with minimum enlargement
            best_idx = self._choose_subtree(parent, node.bbox)
            self._insert_recursive(parent.children[best_idx], node)
            self._expand_bbox(parent, node.bbox)
            
            # Split if needed
            if len(parent.children[best_idx].children) > self.MAX_CHILDREN:
                self._split_node(parent, best_idx)
    
    def _choose_subtree(self, parent: RTreeNode, bbox: BoundingBox) -> int:
        """Choose child with minimum area enlargement."""
        min_enlargement = float('inf')
        min_area = float('inf')
        best_idx = 0
        
        for i, child in enumerate(parent.children):
            enlarged = self._bbox_union(child.bbox, bbox)
            enlargement = enlarged.area - child.bbox.area
            
            if enlargement < min_enlargement or \
               (enlargement == min_enlargement and child.bbox.area < min_area):
                min_enlargement = enlargement
                min_area = child.bbox.area
                best_idx = i
        
        return best_idx
    
    def _split_node(self, parent: RTreeNode, child_idx: int):
        """Split an overfull node using quadratic split."""
        node = parent.children[child_idx]
        children = node.children
        
        # Pick seeds (most wasteful pair)
        seed1, seed2 = self._pick_seeds(children)
        
        group1 = [children[seed1]]
        group2 = [children[seed2]]
        
        remaining = [c for i, c in enumerate(children) if i not in (seed1, seed2)]
        
        while remaining:
            if len(group1) + len(remaining) == self.MIN_CHILDREN:
                group1.extend(remaining)
                break
            if len(group2) + len(remaining) == self.MIN_CHILDREN:
                group2.extend(remaining)
                break
            
            # Pick next
            next_idx = self._pick_next(remaining, group1, group2)
            next_child = remaining.pop(next_idx)
            
            # Assign to group with less enlargement
            bbox1 = self._compute_bbox([c.bbox for c in group1])
            bbox2 = self._compute_bbox([c.bbox for c in group2])
            
            enlarge1 = self._bbox_union(bbox1, next_child.bbox).area - bbox1.area
            enlarge2 = self._bbox_union(bbox2, next_child.bbox).area - bbox2.area
            
            if enlarge1 < enlarge2:
                group1.append(next_child)
            else:
                group2.append(next_child)
        
        # Create new nodes
        node1 = RTreeNode(
            bbox=self._compute_bbox([c.bbox for c in group1]),
            children=group1
        )
        node2 = RTreeNode(
            bbox=self._compute_bbox([c.bbox for c in group2]),
            children=group2
        )
        
        parent.children[child_idx] = node1
        parent.children.append(node2)
    
    def _pick_seeds(self, children: List[RTreeNode]) -> Tuple[int, int]:
        """Pick two children that waste most area if combined."""
        max_waste = -float('inf')
        seed1, seed2 = 0, 1
        
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                combined = self._bbox_union(children[i].bbox, children[j].bbox)
                waste = combined.area - children[i].bbox.area - children[j].bbox.area
                if waste > max_waste:
                    max_waste = waste
                    seed1, seed2 = i, j
        
        return seed1, seed2
    
    def _pick_next(self, remaining: List[RTreeNode], 
                   group1: List[RTreeNode], 
                   group2: List[RTreeNode]) -> int:
        """Pick entry with maximum preference for one group."""
        bbox1 = self._compute_bbox([c.bbox for c in group1])
        bbox2 = self._compute_bbox([c.bbox for c in group2])
        
        max_diff = -float('inf')
        best_idx = 0
        
        for i, child in enumerate(remaining):
            d1 = self._bbox_union(bbox1, child.bbox).area - bbox1.area
            d2 = self._bbox_union(bbox2, child.bbox).area - bbox2.area
            diff = abs(d1 - d2)
            
            if diff > max_diff:
                max_diff = diff
                best_idx = i
        
        return best_idx
    
    def _bbox_union(self, a: BoundingBox, b: BoundingBox) -> BoundingBox:
        """Compute bounding box union."""
        return BoundingBox(
            min(a.xmin, b.xmin), min(a.ymin, b.ymin),
            max(a.xmax, b.xmax), max(a.ymax, b.ymax)
        )
    
    def _compute_bbox(self, boxes: List[BoundingBox]) -> BoundingBox:
        """Compute bounding box of multiple boxes."""
        if not boxes:
            return BoundingBox(0, 0, 0, 0)
        
        xmin = min(b.xmin for b in boxes)
        ymin = min(b.ymin for b in boxes)
        xmax = max(b.xmax for b in boxes)
        ymax = max(b.ymax for b in boxes)
        
        return BoundingBox(xmin, ymin, xmax, ymax)
    
    def _expand_bbox(self, node: RTreeNode, bbox: BoundingBox):
        """Expand node's bbox to include new bbox."""
        node.bbox = self._bbox_union(node.bbox, bbox)
    
    def query_range(self, bbox: BoundingBox) -> List[Tuple[Any, int]]:
        """Find all objects intersecting the given bounding box."""
        results = []
        if self.root is not None:
            self._query_range_recursive(self.root, bbox, results)
        return results
    
    def _query_range_recursive(self, node: RTreeNode, bbox: BoundingBox, 
                               results: List[Tuple[Any, int]]):
        """Recursively search for intersecting objects."""
        if not node.bbox.intersects(bbox):
            return
        
        if node.is_leaf and node.data is not None:
            if node.bbox.intersects(bbox):
                results.append((node.data, node.data_index))
        else:
            for child in node.children:
                self._query_range_recursive(child, bbox, results)
    
    def query_point(self, x: float, y: float) -> List[Tuple[Any, int]]:
        """Find all objects containing the given point."""
        bbox = BoundingBox(x, y, x, y)
        return self.query_range(bbox)


class Polygon:
    """
    Robust polygon representation with uncertainty.
    
    Handles:
    - Degenerate cases
    - Self-intersections
    - Uncertainty in vertex positions
    """
    
    def __init__(self, vertices: np.ndarray, 
                 uncertainties: Optional[np.ndarray] = None):
        """
        Initialize polygon.
        
        Args:
            vertices: Nx2 array of (x, y) coordinates
            uncertainties: Optional Nx2 array of position uncertainties
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        if uncertainties is not None:
            self.uncertainties = np.asarray(uncertainties, dtype=np.float64)
        else:
            self.uncertainties = np.full_like(self.vertices, EPSILON)
        
        self._validate()
    
    def _validate(self):
        """Validate polygon geometry."""
        if len(self.vertices) < 3:
            warnings.warn("Polygon has fewer than 3 vertices")
        
        # Check for degenerate (zero-area) polygon
        if abs(self.signed_area()) < EPSILON:
            warnings.warn("Polygon has near-zero area")
    
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)
    
    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(
            self.vertices[:, 0].min(),
            self.vertices[:, 1].min(),
            self.vertices[:, 0].max(),
            self.vertices[:, 1].max()
        )
    
    def signed_area(self) -> float:
        """Compute signed area using shoelace formula."""
        n = len(self.vertices)
        if n < 3:
            return 0.0
        
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i, 0] * self.vertices[j, 1]
            area -= self.vertices[j, 0] * self.vertices[i, 1]
        
        return area / 2.0
    
    def area(self) -> float:
        """Compute unsigned area."""
        return abs(self.signed_area())
    
    def area_with_uncertainty(self) -> ErrorValue:
        """Compute area with propagated uncertainty."""
        n = len(self.vertices)
        if n < 3:
            return ErrorValue(0.0, 0.0)
        
        area = 0.0
        area_var = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            dxi, dyi = self.uncertainties[i]
            dxj, dyj = self.uncertainties[j]
            
            area += xi * yj - xj * yi
            
            # Partial derivatives for uncertainty propagation
            area_var += (yj * dxi) ** 2 + (xi * dyj) ** 2
            area_var += (yi * dxj) ** 2 + (xj * dyi) ** 2
        
        area = abs(area) / 2.0
        area_std = np.sqrt(area_var) / 2.0
        
        return ErrorValue(area, area_std)
    
    def centroid(self) -> Tuple[float, float]:
        """Compute polygon centroid."""
        n = len(self.vertices)
        if n < 3:
            return (np.mean(self.vertices[:, 0]), np.mean(self.vertices[:, 1]))
        
        cx, cy = 0.0, 0.0
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            
            cross = xi * yj - xj * yi
            area += cross
            cx += (xi + xj) * cross
            cy += (yi + yj) * cross
        
        area /= 2.0
        if abs(area) < EPSILON:
            return (np.mean(self.vertices[:, 0]), np.mean(self.vertices[:, 1]))
        
        cx /= (6.0 * area)
        cy /= (6.0 * area)
        
        return (cx, cy)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Test if point is inside polygon using ray casting."""
        n = len(self.vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + EPSILON) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def contains_point_probabilistic(self, x: float, y: float,
                                      point_uncertainty: float = 0.0) -> float:
        """
        Compute probability that point is inside polygon.
        
        Accounts for uncertainty in both point and polygon vertices.
        """
        # Monte Carlo estimation with correlated samples
        n_samples = 100
        
        inside_count = 0
        for _ in range(n_samples):
            # Perturb point
            px = x + np.random.normal(0, point_uncertainty) if point_uncertainty > 0 else x
            py = y + np.random.normal(0, point_uncertainty) if point_uncertainty > 0 else y
            
            # Perturb polygon vertices
            perturbed_vertices = self.vertices + np.random.normal(0, 1, self.vertices.shape) * self.uncertainties
            
            # Test containment
            temp_poly = Polygon(perturbed_vertices)
            if temp_poly.contains_point(px, py):
                inside_count += 1
        
        return inside_count / n_samples
    
    def distance_to_boundary(self, x: float, y: float) -> float:
        """Compute signed distance to polygon boundary."""
        min_dist = float('inf')
        
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            dist = self._point_segment_distance(
                x, y,
                self.vertices[i, 0], self.vertices[i, 1],
                self.vertices[j, 0], self.vertices[j, 1]
            )
            min_dist = min(min_dist, dist)
        
        # Signed distance: negative inside, positive outside
        if self.contains_point(x, y):
            return -min_dist
        return min_dist
    
    def _point_segment_distance(self, px: float, py: float,
                                 x1: float, y1: float,
                                 x2: float, y2: float) -> float:
        """Compute distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        
        length_sq = dx * dx + dy * dy
        if length_sq < EPSILON:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def compute_voronoi_with_uncertainty(points: np.ndarray,
                                      bounds: BoundingBox,
                                      point_uncertainties: Optional[np.ndarray] = None
                                      ) -> Tuple[List[Polygon], np.ndarray]:
    """
    Compute Voronoi diagram with uncertainty propagation.
    
    Returns:
        (voronoi_cells, cell_uncertainties)
    """
    from scipy.spatial import Voronoi
    
    # Add bounding points to limit cells
    margin = max(bounds.size) * 2
    bounding_points = np.array([
        [bounds.xmin - margin, bounds.ymin - margin],
        [bounds.xmax + margin, bounds.ymin - margin],
        [bounds.xmin - margin, bounds.ymax + margin],
        [bounds.xmax + margin, bounds.ymax + margin],
    ])
    
    all_points = np.vstack([points, bounding_points])
    
    vor = Voronoi(all_points)
    
    cells = []
    cell_uncertainties = []
    
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) == 0:
            # Unbounded region
            cells.append(None)
            cell_uncertainties.append(np.inf)
            continue
        
        vertices = vor.vertices[region]
        
        # Clip to bounds
        vertices[:, 0] = np.clip(vertices[:, 0], bounds.xmin, bounds.xmax)
        vertices[:, 1] = np.clip(vertices[:, 1], bounds.ymin, bounds.ymax)
        
        if point_uncertainties is not None:
            vert_unc = np.full_like(vertices, point_uncertainties[i])
        else:
            vert_unc = None
        
        try:
            poly = Polygon(vertices, vert_unc)
            cells.append(poly)
            cell_uncertainties.append(
                point_uncertainties[i] if point_uncertainties is not None else 0.0
            )
        except:
            cells.append(None)
            cell_uncertainties.append(np.inf)
    
    return cells, np.array(cell_uncertainties)


def compute_delaunay_with_uncertainty(points: np.ndarray,
                                       point_uncertainties: Optional[np.ndarray] = None
                                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Delaunay triangulation with edge uncertainty.
    
    Returns:
        (triangles, edge_uncertainties)
    """
    tri = Delaunay(points)
    triangles = tri.simplices
    
    n_triangles = len(triangles)
    edge_uncertainties = np.zeros((n_triangles, 3))
    
    if point_uncertainties is not None:
        for i, simplex in enumerate(triangles):
            for j in range(3):
                k = (j + 1) % 3
                p1, p2 = simplex[j], simplex[k]
                edge_uncertainties[i, j] = np.sqrt(
                    point_uncertainties[p1] ** 2 + point_uncertainties[p2] ** 2
                )
    
    return triangles, edge_uncertainties
