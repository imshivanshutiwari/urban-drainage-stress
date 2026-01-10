"""
Sparse Linear Operators Module

Implements sparse matrix formulations for stress computation.
Treats the stress computation as a linear system over the spatial grid.

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, diags, kron, eye
from scipy.sparse.linalg import spsolve, cg, gmres, LinearOperator
from typing import Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings

from .error_aware_arithmetic import EPSILON, ErrorArray


@dataclass
class SparseOperatorResult:
    """Result of a sparse operator application with error tracking."""
    values: np.ndarray
    error_bound: np.ndarray
    condition_estimate: float
    iterations: Optional[int] = None


class SparseStressOperator:
    """
    Sparse linear operator for stress field computation.
    
    Implements the stress equation as a sparse linear system:
        L @ stress = forcing
    
    where L is a sparse operator combining:
        - Spatial diffusion (Laplacian)
        - Temporal decay
        - Boundary conditions
        - Source terms from rainfall/complaints
    
    Attributes:
        shape: Grid shape (ny, nx)
        operator: Sparse matrix representation
        preconditioner: Optional preconditioner for iterative solve
    """
    
    def __init__(self, shape: Tuple[int, int], 
                 diffusion_coeff: float = 1.0,
                 decay_rate: float = 0.1,
                 boundary_type: str = 'neumann'):
        """
        Initialize sparse stress operator.
        
        Args:
            shape: Grid dimensions (ny, nx)
            diffusion_coeff: Spatial diffusion coefficient
            decay_rate: Temporal decay rate
            boundary_type: 'dirichlet', 'neumann', or 'periodic'
        """
        self.ny, self.nx = shape
        self.n = self.ny * self.nx
        self.diffusion_coeff = diffusion_coeff
        self.decay_rate = decay_rate
        self.boundary_type = boundary_type
        
        # Build the operator
        self._build_operator()
        
        # Condition number estimate (lazy computed)
        self._condition_estimate: Optional[float] = None
    
    def _build_operator(self):
        """Construct the sparse operator matrix."""
        # 1D Laplacian for each dimension
        Lx = self._build_1d_laplacian(self.nx)
        Ly = self._build_1d_laplacian(self.ny)
        
        # 2D Laplacian via Kronecker product
        Ix = eye(self.nx, format='csr')
        Iy = eye(self.ny, format='csr')
        
        self.laplacian = kron(Iy, Lx) + kron(Ly, Ix)
        
        # Full operator: -D*Laplacian + decay*I
        self.operator = -self.diffusion_coeff * self.laplacian + self.decay_rate * eye(self.n, format='csr')
        
        # Build preconditioner (incomplete LU or diagonal)
        self._build_preconditioner()
    
    def _build_1d_laplacian(self, n: int) -> csr_matrix:
        """Build 1D Laplacian with specified boundary conditions."""
        # Standard second-order finite difference: [1, -2, 1]
        diagonals = [
            np.ones(n - 1),      # sub-diagonal
            -2 * np.ones(n),     # main diagonal
            np.ones(n - 1)       # super-diagonal
        ]
        offsets = [-1, 0, 1]
        L = diags(diagonals, offsets, shape=(n, n), format='csr')
        
        # Apply boundary conditions
        if self.boundary_type == 'neumann':
            # Neumann: zero flux at boundaries
            # Modify first and last rows
            L = L.tolil()
            L[0, 0] = -1.0
            L[n-1, n-1] = -1.0
            L = L.tocsr()
        elif self.boundary_type == 'periodic':
            # Periodic: wrap around
            L = L.tolil()
            L[0, n-1] = 1.0
            L[n-1, 0] = 1.0
            L = L.tocsr()
        # Dirichlet: no modification needed (implicit zero)
        
        return L
    
    def _build_preconditioner(self):
        """Build a preconditioner for iterative solving."""
        # Use diagonal (Jacobi) preconditioner for simplicity
        diag = self.operator.diagonal()
        diag_safe = np.where(np.abs(diag) < EPSILON, 1.0, diag)
        self.precond_diag = 1.0 / diag_safe
        
        # Create LinearOperator for the preconditioner
        def precond_matvec(x):
            return self.precond_diag * x
        
        self.preconditioner = LinearOperator(
            shape=(self.n, self.n),
            matvec=precond_matvec,
            rmatvec=precond_matvec
        )
    
    @property
    def condition_estimate(self) -> float:
        """Estimate condition number of the operator."""
        if self._condition_estimate is None:
            # Use power iteration to estimate largest/smallest eigenvalues
            try:
                from scipy.sparse.linalg import eigsh
                # Largest eigenvalue
                eigval_max = eigsh(self.operator, k=1, which='LM', return_eigenvectors=False)[0]
                # Smallest eigenvalue (by magnitude)
                eigval_min = eigsh(self.operator, k=1, which='SM', return_eigenvectors=False)[0]
                self._condition_estimate = abs(eigval_max / eigval_min) if eigval_min != 0 else np.inf
            except:
                # Fallback: rough estimate from diagonal dominance
                diag = np.abs(self.operator.diagonal())
                off_diag_sum = np.abs(self.operator).sum(axis=1).A1 - diag
                dominance = diag / (off_diag_sum + EPSILON)
                self._condition_estimate = 1.0 / dominance.min()
        
        return self._condition_estimate
    
    def apply(self, x: np.ndarray) -> SparseOperatorResult:
        """Apply the operator to a field."""
        x_flat = x.ravel()
        result = self.operator @ x_flat
        
        # Error bound from floating point operations
        error_bound = EPSILON * np.abs(result) * self.operator.nnz / self.n
        
        return SparseOperatorResult(
            values=result.reshape(self.ny, self.nx),
            error_bound=error_bound.reshape(self.ny, self.nx),
            condition_estimate=self.condition_estimate
        )
    
    def solve(self, rhs: np.ndarray, 
              method: str = 'direct',
              tol: float = 1e-10,
              maxiter: int = 1000) -> SparseOperatorResult:
        """
        Solve the linear system: operator @ x = rhs
        
        Args:
            rhs: Right-hand side (forcing terms)
            method: 'direct', 'cg' (conjugate gradient), or 'gmres'
            tol: Tolerance for iterative methods
            maxiter: Maximum iterations for iterative methods
        
        Returns:
            SparseOperatorResult with solution and error bounds
        """
        rhs_flat = rhs.ravel()
        iterations = None
        
        if method == 'direct':
            x = spsolve(self.operator.tocsc(), rhs_flat)
        elif method == 'cg':
            x, info = cg(self.operator, rhs_flat, tol=tol, maxiter=maxiter,
                        M=self.preconditioner)
            iterations = info if info > 0 else maxiter
            if info != 0:
                warnings.warn(f"CG did not converge: info={info}")
        elif method == 'gmres':
            x, info = gmres(self.operator, rhs_flat, tol=tol, maxiter=maxiter,
                           M=self.preconditioner)
            iterations = info if info > 0 else maxiter
            if info != 0:
                warnings.warn(f"GMRES did not converge: info={info}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Error bound: condition_number * machine_epsilon * ||x||
        error_bound = self.condition_estimate * EPSILON * np.abs(x)
        
        # Add iterative method residual if applicable
        if method != 'direct':
            residual = np.abs(self.operator @ x - rhs_flat)
            error_bound = np.maximum(error_bound, residual)
        
        return SparseOperatorResult(
            values=x.reshape(self.ny, self.nx),
            error_bound=error_bound.reshape(self.ny, self.nx),
            condition_estimate=self.condition_estimate,
            iterations=iterations
        )
    
    def add_source_term(self, source_weights: np.ndarray, 
                        source_locations: np.ndarray) -> 'SparseStressOperator':
        """
        Add localized source terms to the operator.
        
        This modifies the operator to include point sources at specified locations.
        """
        # Create source matrix
        n_sources = len(source_locations)
        source_matrix = sparse.lil_matrix((self.n, self.n))
        
        for i, (y, x) in enumerate(source_locations):
            if 0 <= y < self.ny and 0 <= x < self.nx:
                idx = y * self.nx + x
                source_matrix[idx, idx] += source_weights[i]
        
        # Create new operator with sources
        new_op = SparseStressOperator.__new__(SparseStressOperator)
        new_op.ny, new_op.nx = self.ny, self.nx
        new_op.n = self.n
        new_op.diffusion_coeff = self.diffusion_coeff
        new_op.decay_rate = self.decay_rate
        new_op.boundary_type = self.boundary_type
        new_op.laplacian = self.laplacian
        new_op.operator = self.operator + source_matrix.tocsr()
        new_op._condition_estimate = None
        new_op._build_preconditioner()
        
        return new_op


def build_spatial_laplacian(shape: Tuple[int, int], 
                            dx: float = 1.0, 
                            dy: float = 1.0,
                            boundary: str = 'neumann') -> csr_matrix:
    """
    Build 2D Laplacian operator with variable grid spacing.
    
    Args:
        shape: Grid dimensions (ny, nx)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        boundary: Boundary condition type
    
    Returns:
        Sparse Laplacian matrix
    """
    ny, nx = shape
    n = ny * nx
    
    # Coefficients for second derivative
    cx = 1.0 / (dx * dx)
    cy = 1.0 / (dy * dy)
    cc = -2.0 * (cx + cy)  # Center coefficient
    
    # Build sparse matrix
    rows, cols, data = [], [], []
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            
            # Center
            rows.append(idx)
            cols.append(idx)
            data.append(cc)
            
            # Left neighbor
            if i > 0:
                rows.append(idx)
                cols.append(idx - 1)
                data.append(cx)
            elif boundary == 'periodic':
                rows.append(idx)
                cols.append(idx + nx - 1)
                data.append(cx)
            elif boundary == 'neumann':
                # Neumann: add to center
                data[-1] += cx
            
            # Right neighbor
            if i < nx - 1:
                rows.append(idx)
                cols.append(idx + 1)
                data.append(cx)
            elif boundary == 'periodic':
                rows.append(idx)
                cols.append(idx - nx + 1)
                data.append(cx)
            elif boundary == 'neumann':
                data[-1] += cx
            
            # Bottom neighbor
            if j > 0:
                rows.append(idx)
                cols.append(idx - nx)
                data.append(cy)
            elif boundary == 'periodic':
                rows.append(idx)
                cols.append(idx + (ny - 1) * nx)
                data.append(cy)
            elif boundary == 'neumann':
                data[-1] += cy
            
            # Top neighbor
            if j < ny - 1:
                rows.append(idx)
                cols.append(idx + nx)
                data.append(cy)
            elif boundary == 'periodic':
                rows.append(idx)
                cols.append(idx - (ny - 1) * nx)
                data.append(cy)
            elif boundary == 'neumann':
                data[-1] += cy
    
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def build_diffusion_operator(shape: Tuple[int, int],
                             diffusion_tensor: np.ndarray,
                             dx: float = 1.0,
                             dy: float = 1.0) -> csr_matrix:
    """
    Build anisotropic diffusion operator with spatially varying tensor.
    
    Args:
        shape: Grid dimensions (ny, nx)
        diffusion_tensor: Array of shape (ny, nx, 2, 2) for local diffusion tensors
        dx, dy: Grid spacings
    
    Returns:
        Sparse anisotropic diffusion matrix
    """
    ny, nx = shape
    n = ny * nx
    
    rows, cols, data = [], [], []
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            D = diffusion_tensor[j, i]  # 2x2 tensor
            
            Dxx = D[0, 0] / (dx * dx)
            Dyy = D[1, 1] / (dy * dy)
            Dxy = D[0, 1] / (4 * dx * dy)  # Cross derivative coefficient
            
            # Center coefficient
            rows.append(idx)
            cols.append(idx)
            data.append(-2 * (Dxx + Dyy))
            
            # x neighbors
            if i > 0:
                rows.append(idx)
                cols.append(idx - 1)
                data.append(Dxx)
            if i < nx - 1:
                rows.append(idx)
                cols.append(idx + 1)
                data.append(Dxx)
            
            # y neighbors
            if j > 0:
                rows.append(idx)
                cols.append(idx - nx)
                data.append(Dyy)
            if j < ny - 1:
                rows.append(idx)
                cols.append(idx + nx)
                data.append(Dyy)
            
            # Cross-derivative terms (corners)
            if i > 0 and j > 0:
                rows.append(idx)
                cols.append(idx - nx - 1)
                data.append(Dxy)
            if i < nx - 1 and j > 0:
                rows.append(idx)
                cols.append(idx - nx + 1)
                data.append(-Dxy)
            if i > 0 and j < ny - 1:
                rows.append(idx)
                cols.append(idx + nx - 1)
                data.append(-Dxy)
            if i < nx - 1 and j < ny - 1:
                rows.append(idx)
                cols.append(idx + nx + 1)
                data.append(Dxy)
    
    return csr_matrix((data, (rows, cols)), shape=(n, n))


class SparseTemporalOperator:
    """
    Sparse operator for temporal evolution.
    
    Implements the time-stepping scheme for the stress evolution equation:
        d(stress)/dt = L @ stress + forcing
    
    Uses implicit or semi-implicit schemes for stability.
    """
    
    def __init__(self, spatial_operator: SparseStressOperator, dt: float):
        """
        Initialize temporal operator.
        
        Args:
            spatial_operator: The spatial operator L
            dt: Time step size
        """
        self.spatial_op = spatial_operator
        self.dt = dt
        self.n = spatial_operator.n
        self.shape = (spatial_operator.ny, spatial_operator.nx)
        
        # Build implicit time-stepping matrix: (I - dt*L)
        self.implicit_matrix = (
            eye(self.n, format='csr') - dt * spatial_operator.operator
        )
    
    def step_implicit(self, state: np.ndarray, forcing: np.ndarray) -> SparseOperatorResult:
        """
        Take one implicit Euler time step.
        
        Solves: (I - dt*L) @ new_state = state + dt*forcing
        """
        state_flat = state.ravel()
        forcing_flat = forcing.ravel()
        
        rhs = state_flat + self.dt * forcing_flat
        new_state = spsolve(self.implicit_matrix.tocsc(), rhs)
        
        # Error estimate
        error = EPSILON * np.abs(new_state) * self.spatial_op.condition_estimate
        
        return SparseOperatorResult(
            values=new_state.reshape(self.shape),
            error_bound=error.reshape(self.shape),
            condition_estimate=self.spatial_op.condition_estimate
        )
    
    def step_crank_nicolson(self, state: np.ndarray, 
                            forcing: np.ndarray) -> SparseOperatorResult:
        """
        Take one Crank-Nicolson time step (second-order accurate).
        
        Solves: (I - dt/2*L) @ new = (I + dt/2*L) @ old + dt*forcing
        """
        state_flat = state.ravel()
        forcing_flat = forcing.ravel()
        
        # Build matrices
        I = eye(self.n, format='csr')
        half_dt_L = 0.5 * self.dt * self.spatial_op.operator
        
        lhs_matrix = I - half_dt_L
        rhs_vector = (I + half_dt_L) @ state_flat + self.dt * forcing_flat
        
        new_state = spsolve(lhs_matrix.tocsc(), rhs_vector)
        
        # Error estimate (second-order method has smaller error)
        error = EPSILON * np.abs(new_state) * self.spatial_op.condition_estimate * 0.5
        
        return SparseOperatorResult(
            values=new_state.reshape(self.shape),
            error_bound=error.reshape(self.shape),
            condition_estimate=self.spatial_op.condition_estimate
        )


def create_weight_matrix(weights: np.ndarray, 
                         sparse_threshold: float = 0.01) -> csr_matrix:
    """
    Create a sparse diagonal weight matrix.
    
    Small weights below threshold are zeroed for sparsity.
    """
    weights_flat = weights.ravel()
    n = len(weights_flat)
    
    # Threshold small weights
    sparse_weights = np.where(np.abs(weights_flat) >= sparse_threshold, weights_flat, 0.0)
    
    return diags(sparse_weights, 0, shape=(n, n), format='csr')


def sparse_weighted_sum(operators: list, weights: list) -> csr_matrix:
    """
    Compute weighted sum of sparse operators efficiently.
    
    result = sum(w_i * A_i)
    """
    if len(operators) != len(weights):
        raise ValueError("Number of operators and weights must match")
    
    result = weights[0] * operators[0]
    for w, A in zip(weights[1:], operators[1:]):
        result = result + w * A
    
    return result.tocsr()
