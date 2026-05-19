"""
GFDM Strategy Pattern: Modular Differential Operators and Boundary Handlers.

This module implements the Strategy Pattern for GFDM (Generalized Finite Difference
Method) to cleanly separate:
1. **DifferentialOperator**: Geometric derivative weight computation
2. **BoundaryHandler**: Boundary condition enforcement (Row Replacement vs Ghost)

Architecture Overview:
---------------------
    DifferentialOperator (ABC)
    ├── TaylorOperator     - Standard GFDM (Taylor polynomial + WLS)
    ├── UpwindOperator     - Flow-biased stencils for advection
    └── RBFOperator        - RBF-FD variant (future)

    BoundaryHandler (ABC)
    ├── DirectCollocationHandler  - Row Replacement (recommended)
    └── GhostNodeHandler          - Legacy ghost particles

Usage:
------
    from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import (
        create_operator,
        create_bc_handler,
        TaylorOperator,
        DirectCollocationHandler,
    )

    # Factory pattern (recommended)
    operator = create_operator(points, delta=0.1, method="direct")
    bc_handler = create_bc_handler(method="direct")

    # Direct instantiation
    operator = TaylorOperator(points, delta=0.1, taylor_order=2)
    bc_handler = DirectCollocationHandler()

    # In solver
    grad = operator.gradient(u)
    lap = operator.laplacian(u)
    bc_handler.apply_to_matrix(A, b, boundary_indices, operator, bc_config)

References:
----------
- docs/development/GFDM_BC_INVESTIGATION_REPORT.md (Sections 11, 12)
- GitHub Issue #524

Author: MFGarchon Development Team
Created: 2025-12-18
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from mfgarchon.utils.numerical.kernels import create_kernel

# =============================================================================
# Numerical Tolerance Constants
# =============================================================================
# Centralized tolerance values for consistent numerical behavior across the module.
# These values form a hierarchy from machine precision to user-facing thresholds.

# Near-zero checks (distances, denominators)
ZERO_TOL = 1e-14

# SVD singular value cutoff for rank determination
SVD_TOL = 1e-12

# Regularization added to matrices for stability
REGULARIZATION = 1e-12

# Pseudoinverse rcond parameter
PINV_RCOND = 1e-10

# Condition number threshold for "ill-conditioned" warning
COND_THRESHOLD = 1e12

# PHS singularity epsilon (added to r for r^m)
PHS_EPSILON = 1e-14

# =============================================================================
# Abstract Base Classes
# =============================================================================


class DifferentialOperator(ABC):
    """
    Abstract Strategy for computing spatial derivatives on point clouds.

    This interface is generic for all **collocation methods** (Strong Form):
    - FDM (Finite Difference Method)
    - GFDM (Generalized FDM)
    - RBF-FD (Radial Basis Function - Finite Difference)
    - Spectral Collocation

    NOT for weak-form methods (FEM, FVM) which require element stiffness matrices.

    The key property is that derivatives are computed as weighted sums:
        Lu(x_i) = sum_j w_ij u(x_j)

    Subclasses must implement:
    - gradient(u): Compute gradient at all points
    - laplacian(u): Compute Laplacian at all points
    - get_derivative_weights(point_idx): Return weights for Jacobian assembly
    """

    @property
    @abstractmethod
    def n_points(self) -> int:
        """Number of collocation points."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension."""
        ...

    @property
    @abstractmethod
    def points(self) -> np.ndarray:
        """Collocation points array, shape (n_points, dimension)."""
        ...

    @abstractmethod
    def gradient(self, u: np.ndarray) -> np.ndarray:
        """
        Compute gradient at all points.

        Args:
            u: Function values, shape (n_points,)

        Returns:
            Gradient array, shape (n_points, dimension)
        """
        ...

    @abstractmethod
    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian at all points.

        Args:
            u: Function values, shape (n_points,)

        Returns:
            Laplacian array, shape (n_points,)
        """
        ...

    @abstractmethod
    def get_derivative_weights(self, point_idx: int) -> dict | None:
        """
        Get derivative sensitivity weights for matrix assembly.

        Returns the weights that map perturbations in neighbor u values to
        perturbations in derivatives at point i:
            d(du/dx_d)|_i = sum_j w^grad_{d,j} du_j
            d(Delta u)|_i = sum_j w^lap_j du_j

        Args:
            point_idx: Index of the center point

        Returns:
            Dictionary with:
            - "neighbor_indices": indices of neighbors
            - "grad_weights": shape (dimension, n_neighbors), gradient sensitivity
            - "lap_weights": shape (n_neighbors,), Laplacian sensitivity
            Or None if data not available.
        """
        ...


class BoundaryHandler(ABC):
    """
    Abstract Strategy for enforcing boundary conditions.

    Boundary handlers modify the linear system (A, b) to enforce BCs at
    boundary points. Different strategies:

    - **DirectCollocationHandler** (Row Replacement):
      Clear PDE row, insert BC equation. Recommended for robustness.

    - **GhostNodeHandler** (Legacy):
      Use ghost particles to enforce BC implicitly. Can corrupt derivatives.

    The handler receives:
    - The system matrix A and RHS vector b
    - Boundary point indices
    - The differential operator (for derivative weights)
    - BC configuration (type, values, normals)
    """

    @abstractmethod
    def apply_to_matrix(
        self,
        A: np.ndarray,
        b: np.ndarray,
        boundary_indices: np.ndarray | set[int],
        operator: DifferentialOperator,
        bc_config: dict,
    ) -> None:
        """
        Apply boundary conditions to system matrix (in-place).

        Args:
            A: System matrix, shape (n_points, n_points), modified in-place
            b: RHS vector, shape (n_points,), modified in-place
            boundary_indices: Indices of boundary points
            operator: Differential operator for derivative weights
            bc_config: BC configuration with keys:
                - "type": "dirichlet", "neumann", "no_flux"
                - "values": BC values at boundary points (dict or scalar)
                - "normals": Outward normal vectors (n_boundary, dimension)
        """
        ...

    @abstractmethod
    def apply_to_residual(
        self,
        residual: np.ndarray,
        u: np.ndarray,
        boundary_indices: np.ndarray | set[int],
        operator: DifferentialOperator,
        bc_config: dict,
    ) -> None:
        """
        Apply boundary conditions to residual vector (in-place).

        MUST be consistent with apply_to_matrix for Newton iteration.

        Args:
            residual: Residual vector, shape (n_points,), modified in-place
            u: Current solution, shape (n_points,)
            boundary_indices: Indices of boundary points
            operator: Differential operator for derivative computation
            bc_config: BC configuration (same as apply_to_matrix)
        """
        ...


# =============================================================================
# TaylorOperator: Standard GFDM with Taylor Polynomial Basis
# =============================================================================


class TaylorOperator(DifferentialOperator):
    """
    Standard GFDM operator using Taylor polynomial basis and weighted least squares.

    This is the core GFDM implementation WITHOUT ghost particle logic.
    Ghost particles are handled separately by BoundaryHandler.

    Mathematical Background:
    -----------------------
    Taylor expansion around x_i:
        f(x_j) = f(x_i) + grad f|_i . (x_j - x_i) + (1/2)(x_j - x_i)^T H|_i (x_j - x_i) + ...

    Weighted least squares problem:
        min_coeffs sum_j w(r_ij) [f(x_j) - sum_alpha c_alpha phi_alpha(x_j - x_i)]^2

    Result: Derivative as linear combination of neighbor values
        df/dx|_i = sum_j w_ij f(x_j)

    Row Sum Properties:
    ------------------
    For a properly constructed stencil, the gradient weights should sum to zero:
        sum_j w^grad_{k,j} = 0  for each dimension k

    This is the "translation invariance" property: adding a constant to u
    should not change the gradient.

    **At boundaries**: This property may be violated when the neighborhood is
    asymmetric (truncated by the boundary). The row sum violation magnitude
    indicates how much the constant-preservation property is lost.

    For Laplacian weights, the row sum has no universal value (it depends on
    the mesh and weights), but should be consistent across similar points.

    Use `get_weights_at_point(i)` to inspect weights and check row sums.

    Boundary Conditioning (Issue #529):
    -----------------------------------
    Near domain boundaries, stencils become poorly conditioned due to:
    - Asymmetric neighbor distribution (truncated by boundary)
    - Fewer available neighbors
    - Translation invariance loss (row sum violation)

    Use `get_boundary_conditioning_analysis(domain_bounds)` to diagnose if
    boundary points cause ill-conditioning. If so, consider:
    - Increasing delta (neighborhood radius)
    - Using adaptive neighborhoods
    - Adding ghost nodes via BoundaryHandler

    Attributes:
        _points: Collocation points, shape (n_points, dimension)
        _n_points: Number of collocation points
        _dimension: Spatial dimension
        delta: Neighborhood radius
        taylor_order: Order of Taylor expansion (1 or 2)
        multi_indices: List of derivative multi-indices
    """

    def __init__(
        self,
        points: np.ndarray,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        weight_scale: float = 1.0,
        k_neighbors: int | None = None,
        neighborhood_mode: str = "hybrid",
        geometry: object | None = None,
        adaptive_params: tuple[np.ndarray, np.ndarray] | None = None,
        obstacle_sdf: object | None = None,
        visibility_samples: int = 10,
        visibility_margin: float = 0.0,
    ):
        """
        Initialize Taylor operator with precomputed structure.

        Args:
            points: Collocation points, shape (n_points, dimension)
            delta: Neighborhood radius for finding neighbors (scalar, or ignored
                if adaptive_params provided)
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("wendland", "gaussian", "uniform")
            weight_scale: Scale parameter for weight function
            k_neighbors: Number of neighbors for k-NN mode (auto-computed if None,
                or ignored if adaptive_params provided)
            neighborhood_mode: Neighborhood selection strategy:
                - "radius": Use all points within delta
                - "knn": Use exactly k nearest neighbors
                - "hybrid": Use delta, but ensure at least k neighbors (default)
                - "adaptive": Use per-point (k, delta) from adaptive_params
            geometry: Geometry object implementing SupportsPeriodic protocol for
                periodic domains (e.g., Hyperrectangle with periodic_dims).
                If provided and periodic, enables wrap-around neighbor search.
                Issue #711.
            adaptive_params: Optional tuple (k_arr, delta_arr) of per-point arrays
                from compute_adaptive_gfdm_params(). When provided, enables
                per-point adaptive neighborhoods where each point has its own
                (k, delta) values. This provides better boundary/corner handling.
                Sets neighborhood_mode="adaptive" automatically.
            obstacle_sdf: Optional callable ``f(x) -> NDArray`` giving the signed
                distance to the obstacle region (Issue #1124). Convention:
                ``obstacle_sdf(x) < 0`` means x is INSIDE obstacle. When provided,
                neighbors whose line of sight to the center crosses an obstacle
                are excluded from the stencil. This makes ``D_lap`` / ``D_grad``
                respect domain connectivity in thin-wall geometries where
                ``delta`` exceeds wall thickness. Same convention as
                ``NeighborhoodBuilder.obstacle_sdf`` — pass the SDF of the
                obstacle region (e.g., ``UnionDomain(...).signed_distance``),
                not the navigable domain.
            visibility_samples: Number of interior samples per stencil edge for
                visibility check. Default 10 (sufficient for convex obstacles).
            visibility_margin: Safety margin for visibility filter; edges with
                any sample at ``obstacle_sdf < margin`` are blocked. Default 0.0.
        """
        self._points = np.asarray(points)
        self._n_points, self._dimension = self._points.shape
        self.taylor_order = taylor_order
        self.weight_function = weight_function
        self.weight_scale = weight_scale
        self._geometry = geometry
        self._obstacle_sdf = obstacle_sdf
        self._visibility_samples = int(visibility_samples)
        self._visibility_margin = float(visibility_margin)

        # Handle adaptive parameters mode
        if adaptive_params is not None:
            k_arr, delta_arr = adaptive_params
            if len(k_arr) != self._n_points or len(delta_arr) != self._n_points:
                raise ValueError(
                    f"adaptive_params arrays must have length {self._n_points}, "
                    f"got k_arr={len(k_arr)}, delta_arr={len(delta_arr)}"
                )
            self._k_arr = np.asarray(k_arr, dtype=int)
            self._delta_arr = np.asarray(delta_arr)
            self.neighborhood_mode = "adaptive"
            # Store representative scalars for compatibility (mean values)
            self.delta = float(np.mean(self._delta_arr))
            self.k_neighbors = int(np.mean(self._k_arr))
        else:
            self._k_arr = None
            self._delta_arr = None
            self.neighborhood_mode = neighborhood_mode
            self.delta = delta
            # Compute k_neighbors from Taylor order if not provided
            # Formula: k = ρ × n_derivatives where n_derivatives = C(d+p, p) - 1
            # Uses unified design with overdetermination ratio ρ = 3.0
            n_monomials = self._count_monomials(self._dimension, taylor_order)
            n_derivatives = n_monomials - 1  # Exclude constant term
            if k_neighbors is None:
                overdetermination = 3.0  # Unified design parameter
                self.k_neighbors = int(overdetermination * n_derivatives)
            else:
                # User-provided k must be at least n_derivatives + 1 for overdetermination
                self.k_neighbors = max(k_neighbors, n_derivatives + 1)

        # Check if geometry supports periodicity (Issue #711)
        # Use isinstance() with Protocol per CLAUDE.md standards (no hasattr duck typing)
        from mfgarchon.geometry.protocols import SupportsPeriodic

        self._is_periodic = isinstance(geometry, SupportsPeriodic) and len(geometry.periodic_dimensions) > 0

        # Build multi-index set for Taylor expansion
        self.multi_indices = self._build_multi_indices()
        self.n_derivatives = len(self.multi_indices)

        # Precompute neighbor structure (NO ghost particles)
        self._build_neighborhoods()

        # Precompute Taylor matrices and validate stencil quality
        self._build_taylor_matrices()
        self._validate_stencils()

        # Pre-assemble global sparse derivative matrices (Issue #932)
        # Runtime gradient/laplacian become single sparse matmuls
        self._gradient_matrices: list | None = None
        self._laplacian_matrix = None
        self._preassemble_sparse_matrices()

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def points(self) -> np.ndarray:
        return self._points

    def _count_monomials(self, dimension: int, order: int) -> int:
        """
        Count number of monomials for Taylor expansion of given order.

        Formula: C(d+p, p) = (d+p)! / (d! * p!)

        Examples:
            d=2, p=2: C(4,2) = 6 monomials (1, x, y, x², xy, y²)
            d=2, p=3: C(5,3) = 10 monomials
            d=3, p=2: C(5,2) = 10 monomials
        """
        from math import comb

        return comb(dimension + order, order)

    def _count_derivatives(self, dimension: int, order: int) -> int:
        """Count number of derivatives for given dimension and order (excludes constant)."""
        return self._count_monomials(dimension, order) - 1

    def _build_multi_indices(self) -> list[tuple[int, ...]]:
        """Generate multi-index set B(d,p) = {beta in N^d : 0 < |beta| <= p}."""
        d, p = self._dimension, self.taylor_order
        multi_indices: list[tuple[int, ...]] = []

        def generate(current: list[int], remaining_dims: int, remaining_order: int):
            if remaining_dims == 0:
                if 0 < sum(current) <= p:
                    multi_indices.append(tuple(current))
                return
            for i in range(remaining_order + 1):
                generate([*current, i], remaining_dims - 1, remaining_order - i)

        generate([], d, p)
        multi_indices.sort(key=lambda beta: (sum(beta), beta))
        return multi_indices

    # =========================================================================
    # Periodic Domain Support (Issue #711)
    # Delegates to geometry.SupportsPeriodic protocol methods
    # =========================================================================

    def _wrap_displacement(self, delta_x: np.ndarray) -> np.ndarray:
        """Wrap displacement vector using geometry's wrap_displacement method."""
        if not self._is_periodic:
            return delta_x
        return self._geometry.wrap_displacement(delta_x)

    def _get_augmented_points_for_tree(self) -> tuple[np.ndarray, np.ndarray]:
        """Get augmented point cloud with ghost copies for periodic tree search."""
        if not self._is_periodic:
            return self._points, np.arange(self._n_points)

        # Issue #711: Call periodic utility (parallel to enforcement.py)
        # Geometry provides protocol info (bounds, periodic_dims), utility does work
        from mfgarchon.geometry.boundary.periodic import create_periodic_ghost_points

        bounds = self._geometry.bounds
        periodic_dims = self._geometry.periodic_dimensions
        return create_periodic_ghost_points(self._points, bounds, periodic_dims)

    def _build_neighborhoods(self):
        """Build neighborhood structure for all points."""
        # For periodic domains, use augmented points for tree search
        augmented_points, original_indices = self._get_augmented_points_for_tree()
        tree = cKDTree(augmented_points)

        self.neighborhoods: list[dict] = []
        self._hybrid_expanded_count = 0
        self._visibility_filtered_count = 0
        self._visibility_filtered_min_kept = self._n_points  # tracks worst-case stencil size

        for i in range(self._n_points):
            # Get per-point k and delta for adaptive mode
            if self.neighborhood_mode == "adaptive":
                k_i = self._k_arr[i]
                delta_i = self._delta_arr[i]
            else:
                k_i = self.k_neighbors
                delta_i = self.delta

            if self.neighborhood_mode == "knn":
                distances, aug_neighbor_indices = tree.query(self._points[i], k=k_i)
                aug_neighbor_indices = np.array(aug_neighbor_indices)
                distances = np.array(distances)
            elif self.neighborhood_mode == "radius":
                aug_neighbor_indices = tree.query_ball_point(self._points[i], delta_i)
                aug_neighbor_indices = np.array(aug_neighbor_indices)
                aug_neighbor_points = augmented_points[aug_neighbor_indices]
                distances = np.linalg.norm(aug_neighbor_points - self._points[i], axis=1)
            elif self.neighborhood_mode == "adaptive":
                # Adaptive mode: use per-point (k, delta) with hybrid strategy
                aug_neighbor_indices = tree.query_ball_point(self._points[i], delta_i)
                aug_neighbor_indices = np.array(aug_neighbor_indices)

                if len(aug_neighbor_indices) < k_i:
                    # Fallback to k-NN if not enough neighbors in delta
                    distances, aug_neighbor_indices = tree.query(self._points[i], k=k_i)
                    aug_neighbor_indices = np.array(aug_neighbor_indices)
                    distances = np.array(distances)
                    # Expand delta to include all k neighbors (for weight computation)
                    delta_i = distances.max() * 1.01
                    self._hybrid_expanded_count += 1
                else:
                    aug_neighbor_points = augmented_points[aug_neighbor_indices]
                    distances = np.linalg.norm(aug_neighbor_points - self._points[i], axis=1)
            else:
                # Hybrid mode (default)
                aug_neighbor_indices = tree.query_ball_point(self._points[i], delta_i)
                aug_neighbor_indices = np.array(aug_neighbor_indices)

                if len(aug_neighbor_indices) < k_i:
                    # k-NN fallback: find k neighbors beyond original delta
                    distances, aug_neighbor_indices = tree.query(self._points[i], k=k_i)
                    aug_neighbor_indices = np.array(aug_neighbor_indices)
                    distances = np.array(distances)
                    # Expand delta to include all k neighbors (for weight computation)
                    delta_i = distances.max() * 1.01
                    self._hybrid_expanded_count += 1
                else:
                    aug_neighbor_points = augmented_points[aug_neighbor_indices]
                    distances = np.linalg.norm(aug_neighbor_points - self._points[i], axis=1)

            # Map augmented indices back to original point indices
            neighbor_indices = original_indices[aug_neighbor_indices]

            # Get neighbor points from augmented cloud (preserves ghost positions for delta_x)
            neighbor_points = augmented_points[aug_neighbor_indices]

            # Issue #1124: visibility filter at operator level. Without this,
            # ``get_derivative_weights(i)`` returns weights on a stencil whose
            # edges may cross obstacles, and the assembled ``D_lap`` / ``D_grad``
            # couple values through walls. The NeighborhoodBuilder layered on
            # top re-applies this filter, but the pre-assembled operator-side
            # sparse matrices come straight from these stencils.
            if self._obstacle_sdf is not None and len(neighbor_indices) > 0:
                from mfgarchon.geometry.visibility import filter_visible_neighbors

                visible_mask = filter_visible_neighbors(
                    center=self._points[i],
                    candidates=neighbor_points,
                    obstacle_sdf=self._obstacle_sdf,
                    n_samples=self._visibility_samples,
                    margin=self._visibility_margin,
                )
                n_blocked = int((~visible_mask).sum())
                if n_blocked > 0:
                    neighbor_indices = neighbor_indices[visible_mask]
                    neighbor_points = neighbor_points[visible_mask]
                    distances = distances[visible_mask]
                    self._visibility_filtered_count += n_blocked
                    self._visibility_filtered_min_kept = min(self._visibility_filtered_min_kept, len(neighbor_indices))

            self.neighborhoods.append(
                {
                    "indices": neighbor_indices,  # Original indices (for value lookup)
                    "points": neighbor_points,  # Augmented points (for delta_x)
                    "distances": distances,
                    "size": len(neighbor_indices),
                    "delta": delta_i,  # Store per-point delta for weight computation
                }
            )

        if self.neighborhood_mode in ("hybrid", "adaptive") and self._hybrid_expanded_count > 0:
            import warnings

            pct = 100.0 * self._hybrid_expanded_count / self._n_points
            mode_str = "Adaptive" if self.neighborhood_mode == "adaptive" else "Hybrid"
            k_str = (
                f"k_arr[min={self._k_arr.min()},max={self._k_arr.max()}]"
                if self._k_arr is not None
                else f"k={self.k_neighbors}"
            )
            delta_str = (
                f"delta_arr[min={self._delta_arr.min():.4f},max={self._delta_arr.max():.4f}]"
                if self._delta_arr is not None
                else f"delta={self.delta:.4f}"
            )
            warnings.warn(
                f"{mode_str} neighborhood: {self._hybrid_expanded_count}/{self._n_points} points ({pct:.1f}%) "
                f"had fewer than {k_str} neighbors within {delta_str}. "
                f"Used k-NN fallback.",
                UserWarning,
                stacklevel=2,
            )

        if self._obstacle_sdf is not None and self._visibility_filtered_count > 0:
            import warnings

            min_kept = self._visibility_filtered_min_kept
            if min_kept < self.n_derivatives:
                warnings.warn(
                    f"Operator-level visibility filter (Issue #1124) reduced at "
                    f"least one stencil to {min_kept} neighbors, below the "
                    f"Taylor LSQ minimum {self.n_derivatives}. Taylor matrix "
                    f"construction will fall back to None at those points. "
                    f"Consider increasing delta or relaxing visibility_margin.",
                    UserWarning,
                    stacklevel=2,
                )

    def _build_taylor_matrices(self):
        """Precompute Taylor expansion matrices for all points."""
        self.taylor_matrices: list[dict | None] = []

        for i in range(self._n_points):
            neighborhood = self.neighborhoods[i]
            n_neighbors = neighborhood["size"]

            if n_neighbors < self.n_derivatives:
                self.taylor_matrices.append(None)
                continue

            # Build Taylor expansion matrix A
            A = np.zeros((n_neighbors, self.n_derivatives))
            center_point = self._points[i]
            neighbor_points = neighborhood["points"]

            for j, neighbor_point in enumerate(neighbor_points):
                # Issue #711: Use wrapped displacement for periodic domains
                # For non-periodic: delta_x = neighbor - center (standard)
                # For periodic: delta_x wraps to [-L/2, L/2] (shortest path)
                delta_x = self._wrap_displacement(neighbor_point - center_point)

                for k, beta in enumerate(self.multi_indices):
                    term = 1.0
                    factorial = 1.0
                    for dim in range(self._dimension):
                        if beta[dim] > 0:
                            term *= delta_x[dim] ** beta[dim]
                            factorial *= math.factorial(beta[dim])
                    A[j, k] = term / factorial

            # Compute weights (use per-point delta for adaptive mode)
            delta_i = neighborhood.get("delta", self.delta)
            weights = self._compute_weights(neighborhood["distances"], delta=delta_i)
            W = np.diag(weights)
            sqrt_W = np.sqrt(W)

            # SVD decomposition for numerical stability
            WA = sqrt_W @ A
            try:
                U, S, Vt = np.linalg.svd(WA, full_matrices=False)
                rank = np.sum(SVD_TOL < S)

                # Handle rank=0 as failure (all singular values below threshold)
                # This prevents empty arrays in subsequent computations
                if rank == 0:
                    self.taylor_matrices.append(None)
                    continue

                # Compute condition number from singular values
                # κ = σ_max / σ_min (only consider non-zero singular values)
                if S[rank - 1] > ZERO_TOL:
                    cond = S[0] / S[rank - 1]
                else:
                    cond = np.inf

                self.taylor_matrices.append(
                    {
                        "A": A,
                        "W": W,
                        "sqrt_W": sqrt_W,
                        "U": U[:, :rank],
                        "S": S[:rank],
                        "Vt": Vt[:rank, :],
                        "rank": rank,
                        "condition_number": cond,
                    }
                )
            except np.linalg.LinAlgError:
                self.taylor_matrices.append(None)

    def _validate_stencils(self, weight_threshold: float = 0.01, cond_threshold: float = COND_THRESHOLD) -> bool:
        """
        Validate stencil quality after construction.

        Checks that all points have sufficient rank in their Taylor matrices
        to compute the requested derivatives. Also checks for ill-conditioned
        stencils. Emits warnings if issues are found.

        Args:
            weight_threshold: Neighbors with weight < threshold * max_weight
                are considered ineffective. Default 0.01 (1% of max).
            cond_threshold: Condition numbers above this are considered
                ill-conditioned. Default 1e12.

        Returns:
            True if all stencils are valid, False if any are degenerate or
            ill-conditioned.

        Notes:
            Degenerate stencils occur when effective data points < n_derivatives.
            "Effective" means neighbors with weight above threshold (compact
            support kernels like Wendland zero out distant neighbors).

            Minimum data points needed for taylor_order p in dimension d:
              n_min = C(d+p, p) = (d+p)! / (d! * p!)

            Examples (taylor_order=2):
              1D: 3 points,  2D: 6 points,  3D: 10 points
        """
        import warnings

        degenerate_points = []
        ill_conditioned_points = []

        for i in range(self._n_points):
            taylor_data = self.taylor_matrices[i]
            if taylor_data is None:
                n_neighbors = self.neighborhoods[i]["size"]
                degenerate_points.append((i, "no_data", 0, n_neighbors, 0, np.inf))
                continue

            rank = taylor_data.get("rank", 0)
            cond = taylor_data.get("condition_number", np.inf)

            if rank < self.n_derivatives:
                # Count effective neighbors
                W = taylor_data.get("W")
                if W is not None:
                    weights = np.diag(W)
                    max_w = weights.max() if len(weights) > 0 else 0
                    n_effective = int(np.sum(weights > weight_threshold * max_w))
                else:
                    n_effective = 0
                n_total = self.neighborhoods[i]["size"]
                degenerate_points.append((i, "rank_deficient", rank, n_total, n_effective, cond))
            elif cond > cond_threshold:
                # Stencil is full rank but ill-conditioned
                n_total = self.neighborhoods[i]["size"]
                ill_conditioned_points.append((i, cond, n_total))

        valid = True

        if degenerate_points:
            valid = False
            n_issues = len(degenerate_points)
            examples = degenerate_points[:3]
            msg_lines = [
                f"TaylorOperator: {n_issues}/{self._n_points} points have degenerate stencils.",
                f"  Minimum data points needed: {self.n_derivatives + 1} "
                f"(for {self.n_derivatives} derivatives in {self._dimension}D)",
            ]
            for idx, issue_type, rank, n_total, n_effective, cond in examples:
                if issue_type == "no_data":
                    msg_lines.append(f"  Point {idx}: {n_total} neighbors, insufficient for Taylor expansion")
                else:
                    cond_str = f", cond={cond:.1e}" if np.isfinite(cond) else ""
                    msg_lines.append(
                        f"  Point {idx}: {n_effective}/{n_total} effective neighbors, "
                        f"rank={rank} < {self.n_derivatives}{cond_str}"
                    )
            if n_issues > 3:
                msg_lines.append(f"  ... and {n_issues - 3} more")
            msg_lines.append(
                "  Fix: Increase k_neighbors, use weight_function='gaussian' (no compact "
                "support), or use LocalRBFOperator."
            )
            warnings.warn("\n".join(msg_lines), UserWarning, stacklevel=3)

        if ill_conditioned_points:
            # Don't mark as invalid for ill-conditioning alone, just warn
            n_issues = len(ill_conditioned_points)
            examples = ill_conditioned_points[:3]
            cond_stats = self.get_condition_numbers()
            msg_lines = [
                f"TaylorOperator: {n_issues}/{self._n_points} points have ill-conditioned stencils "
                f"(cond > {cond_threshold:.0e}).",
                f"  Condition number stats: min={cond_stats['min']:.1e}, max={cond_stats['max']:.1e}, "
                f"median={cond_stats['median']:.1e}",
            ]
            for idx, cond, n_total in examples:
                msg_lines.append(f"  Point {idx}: cond={cond:.1e} ({n_total} neighbors)")
            if n_issues > 3:
                msg_lines.append(f"  ... and {n_issues - 3} more")
            msg_lines.append(
                "  Info: High condition numbers may indicate points near boundaries or "
                "anisotropic neighbor distributions. Results may be less accurate."
            )
            warnings.warn("\n".join(msg_lines), UserWarning, stacklevel=3)

        return valid

    def _compute_weights(self, distances: np.ndarray, delta: float | None = None) -> np.ndarray:
        """
        Compute weights based on distance using kernel infrastructure.

        Integrates with mfgarchon.utils.numerical.kernels for a unified
        kernel interface. Supported kernels:
        - "wendland_c0", "wendland_c2", "wendland_c4", "wendland_c6"
        - "gaussian"
        - "cubic_spline", "quintic_spline"
        - "cubic", "quartic"
        - "uniform" (special case: all weights = 1)
        - "phs" (Polyharmonic Spline: r^m, no shape parameter)

        For backward compatibility, "wendland" maps to "wendland_c4".

        Args:
            distances: Distance array to neighbors
            delta: Support radius for kernel. If None, uses self.delta.
                For adaptive mode, pass the per-point delta.
        """
        if delta is None:
            delta = self.delta

        # Uniform weights (special case, not a kernel)
        if self.weight_function == "uniform":
            return np.ones_like(distances)

        # Polyharmonic Spline: φ(r) = r^m (no shape parameter to tune)
        if self.weight_function.startswith("phs"):
            # Parse order: "phs" -> m=3, "phs3" -> m=3, "phs5" -> m=5
            if self.weight_function == "phs":
                m = 3
            else:
                try:
                    m = int(self.weight_function[3:])
                except ValueError:
                    m = 3
            # Add small epsilon to avoid r=0 singularity
            return (distances + PHS_EPSILON) ** m

        # Map backward-compatible names
        kernel_name = self.weight_function
        if kernel_name == "wendland":
            kernel_name = "wendland_c4"  # C^4 is the default (good balance)

        # Create kernel and evaluate
        try:
            kernel = create_kernel(kernel_name, dimension=self._dimension)
            return kernel(distances, h=delta)
        except ValueError as e:
            raise ValueError(
                f"Unknown weight function: '{self.weight_function}'. "
                f"Valid options: 'uniform', 'gaussian', 'wendland', 'wendland_c2', "
                f"'wendland_c4', 'wendland_c6', 'phs', 'phs3', 'phs5'."
            ) from e

    def _approximate_derivatives_at_point(self, u: np.ndarray, point_idx: int) -> dict[tuple[int, ...], float]:
        """Compute all Taylor derivatives at a single point."""
        taylor_data = self.taylor_matrices[point_idx]
        if taylor_data is None:
            return {}

        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]

        # Function values (NO ghost particle handling - that's BoundaryHandler's job)
        u_center = u[point_idx]
        u_neighbors = u[neighbor_indices]
        b = u_neighbors - u_center

        # Solve using SVD
        sqrt_W = taylor_data["sqrt_W"]
        U = taylor_data["U"]
        S = taylor_data["S"]
        Vt = taylor_data["Vt"]

        Wb = sqrt_W @ b
        UT_Wb = U.T @ Wb
        S_inv_UT_Wb = UT_Wb / S
        coeffs = Vt.T @ S_inv_UT_Wb

        return {beta: coeffs[k] for k, beta in enumerate(self.multi_indices)}

    def approximate_derivatives_at_point(self, u: np.ndarray, point_idx: int) -> dict[tuple[int, ...], float]:
        """
        Public accessor for derivative computation at a single point.

        This is the core GFDM computation. Returns a dict mapping multi-indices
        to derivative values at the specified point.

        Args:
            u: Function values at all points
            point_idx: Index of point where derivatives are computed

        Returns:
            Dict mapping multi-indices (e.g., (1,0) for ∂/∂x) to derivative values
        """
        return self._approximate_derivatives_at_point(u, point_idx)

    def _preassemble_sparse_matrices(self) -> None:
        """Pre-assemble global sparse gradient and Laplacian matrices (Issue #932).

        Each row i of the sparse matrix encodes:
            deriv[i] = sum_j w_ij * (u[j] - u[i])
                     = sum_j w_ij * u[j] - u[i] * sum_j w_ij

        So: diagonal = -sum(weights), off-diagonal = weights.
        Runtime gradient/laplacian become single sparse matmuls: ``W @ u``.
        """
        import scipy.sparse

        N = self._n_points
        d = self._dimension

        # Collect COO entries for gradient matrices (one per dimension)
        grad_rows: list[list[int]] = [[] for _ in range(d)]
        grad_cols: list[list[int]] = [[] for _ in range(d)]
        grad_vals: list[list[float]] = [[] for _ in range(d)]

        # COO entries for Laplacian matrix
        lap_rows: list[int] = []
        lap_cols: list[int] = []
        lap_vals: list[float] = []

        for i in range(N):
            dw = self.get_derivative_weights(i)
            if dw is None:
                continue

            neighbor_indices = dw["neighbor_indices"]
            g_weights = dw["grad_weights"]  # (dimension, n_neighbors)
            l_weights = dw["lap_weights"]  # (n_neighbors,)

            # Gradient: for each dimension, row i has weights on neighbors
            # and -sum(weights) on diagonal (from b = u_neighbor - u_center)
            for dim in range(d):
                w = g_weights[dim]
                diag_val = -w.sum()
                for j_local, j_global in enumerate(neighbor_indices):
                    if j_global == i:
                        # Neighbor is center point — combine with diagonal correction
                        grad_rows[dim].append(i)
                        grad_cols[dim].append(i)
                        grad_vals[dim].append(w[j_local] + diag_val)
                    else:
                        grad_rows[dim].append(i)
                        grad_cols[dim].append(j_global)
                        grad_vals[dim].append(w[j_local])
                # If center not in neighbors, add pure diagonal
                if i not in neighbor_indices:
                    grad_rows[dim].append(i)
                    grad_cols[dim].append(i)
                    grad_vals[dim].append(diag_val)

            # Laplacian: same pattern
            diag_val = -l_weights.sum()
            for j_local, j_global in enumerate(neighbor_indices):
                if j_global == i:
                    lap_rows.append(i)
                    lap_cols.append(i)
                    lap_vals.append(l_weights[j_local] + diag_val)
                else:
                    lap_rows.append(i)
                    lap_cols.append(j_global)
                    lap_vals.append(l_weights[j_local])
            if i not in neighbor_indices:
                lap_rows.append(i)
                lap_cols.append(i)
                lap_vals.append(diag_val)

        self._gradient_matrices = [
            scipy.sparse.csr_matrix((grad_vals[dim], (grad_rows[dim], grad_cols[dim])), shape=(N, N))
            for dim in range(d)
        ]
        self._laplacian_matrix = scipy.sparse.csr_matrix((lap_vals, (lap_rows, lap_cols)), shape=(N, N))

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Compute gradient at all points via pre-assembled sparse matrices."""
        if self._gradient_matrices is not None:
            return np.column_stack([G @ u for G in self._gradient_matrices])

        # Fallback: per-point loop (should not be reached after init)
        grad = np.zeros((self._n_points, self._dimension))
        for i in range(self._n_points):
            derivs = self._approximate_derivatives_at_point(u, i)
            for d in range(self._dimension):
                multi_idx = tuple(1 if j == d else 0 for j in range(self._dimension))
                grad[i, d] = derivs.get(multi_idx, 0.0)
        return grad

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """Compute Laplacian at all points via pre-assembled sparse matrix."""
        if self._laplacian_matrix is not None:
            return self._laplacian_matrix @ u

        # Fallback: per-point loop
        lap = np.zeros(self._n_points)
        for i in range(self._n_points):
            derivs = self._approximate_derivatives_at_point(u, i)
            for d in range(self._dimension):
                multi_idx = tuple(2 if j == d else 0 for j in range(self._dimension))
                lap[i] += derivs.get(multi_idx, 0.0)
        return lap

    def hessian(self, u: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix at all points."""
        hess = np.zeros((self._n_points, self._dimension, self._dimension))

        for i in range(self._n_points):
            derivs = self._approximate_derivatives_at_point(u, i)
            for d1 in range(self._dimension):
                for d2 in range(self._dimension):
                    multi_idx_list = [0] * self._dimension
                    multi_idx_list[d1] += 1
                    multi_idx_list[d2] += 1
                    multi_idx = tuple(multi_idx_list)
                    hess[i, d1, d2] = derivs.get(multi_idx, 0.0)

        return hess

    def get_derivative_weights(self, point_idx: int) -> dict | None:
        """Get derivative sensitivity weights for matrix assembly."""
        taylor_data = self.taylor_matrices[point_idx]
        if taylor_data is None:
            return None

        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]
        n_neighbors = len(neighbor_indices)

        sqrt_W = taylor_data["sqrt_W"]
        U = taylor_data["U"]
        S = taylor_data["S"]
        Vt = taylor_data["Vt"]

        # M maps b (neighbor deviations) to Taylor coefficients
        M = Vt.T @ np.diag(1.0 / S) @ U.T @ sqrt_W

        # Gradient weights
        grad_weights = np.zeros((self._dimension, n_neighbors))
        for d in range(self._dimension):
            multi_idx = tuple(1 if j == d else 0 for j in range(self._dimension))
            if multi_idx in self.multi_indices:
                k = self.multi_indices.index(multi_idx)
                grad_weights[d, :] = M[k, :]

        # Laplacian weights
        lap_weights = np.zeros(n_neighbors)
        for d in range(self._dimension):
            multi_idx = tuple(2 if j == d else 0 for j in range(self._dimension))
            if multi_idx in self.multi_indices:
                k = self.multi_indices.index(multi_idx)
                lap_weights += M[k, :]

        # Find center index in neighbors
        center_in_neighbors = -1
        for idx, n_idx in enumerate(neighbor_indices):
            if n_idx == point_idx:
                center_in_neighbors = idx
                break

        return {
            "neighbor_indices": neighbor_indices,
            "grad_weights": grad_weights,
            "lap_weights": lap_weights,
            "center_idx_in_neighbors": center_in_neighbors,
            "weight_matrix": M,
        }

    def get_neighborhood(self, point_idx: int) -> dict:
        """Get neighborhood data for a point."""
        return self.neighborhoods[point_idx]

    def get_taylor_data(self, point_idx: int) -> dict | None:
        """Get Taylor matrix data for a point."""
        return self.taylor_matrices[point_idx]

    def get_condition_numbers(self) -> dict:
        """
        Get condition number statistics across all stencils.

        Returns:
            Dictionary with:
            - 'values': array of condition numbers (inf for failed points)
            - 'min': minimum condition number
            - 'max': maximum finite condition number
            - 'mean': mean of finite condition numbers
            - 'median': median of finite condition numbers
            - 'n_ill_conditioned': count where κ > 1e12
            - 'n_failed': count where κ = inf (SVD failed or rank=0)

        Notes:
            Condition number κ = σ_max / σ_min measures sensitivity to
            perturbations. For GFDM stencils:
            - κ < 1e4: Well-conditioned
            - 1e4 < κ < 1e8: Acceptable but monitor
            - 1e8 < κ < 1e12: Poorly conditioned, results may be unreliable
            - κ > 1e12: Ill-conditioned, results likely unreliable
        """
        cond_numbers = []
        for taylor_data in self.taylor_matrices:
            if taylor_data is None:
                cond_numbers.append(np.inf)
            else:
                cond_numbers.append(taylor_data.get("condition_number", np.inf))

        values = np.array(cond_numbers)
        finite_values = values[np.isfinite(values)]

        return {
            "values": values,
            "min": float(np.min(finite_values)) if len(finite_values) > 0 else np.inf,
            "max": float(np.max(finite_values)) if len(finite_values) > 0 else np.inf,
            "mean": float(np.mean(finite_values)) if len(finite_values) > 0 else np.inf,
            "median": float(np.median(finite_values)) if len(finite_values) > 0 else np.inf,
            "n_ill_conditioned": int(np.sum(values > 1e12)),
            "n_failed": int(np.sum(np.isinf(values))),
        }

    def get_boundary_conditioning_analysis(
        self,
        domain_bounds: np.ndarray | None = None,
        boundary_tolerance: float | None = None,
    ) -> dict:
        """
        Analyze condition numbers by boundary proximity (Issue #529).

        Near domain boundaries, stencils have asymmetric neighbor distributions
        which often causes ill-conditioning. This method helps diagnose whether
        boundary points are the source of conditioning issues.

        Args:
            domain_bounds: Domain bounds, shape (dimension, 2) with [min, max] per axis.
                          If None, inferred from point cloud bounding box.
            boundary_tolerance: Distance threshold for "near boundary" classification.
                              If None, defaults to 1.5 * delta.

        Returns:
            Dictionary with:
            - 'interior_stats': Condition stats for interior points
            - 'boundary_stats': Condition stats for boundary-adjacent points
            - 'boundary_indices': Indices of points near boundary
            - 'interior_indices': Indices of interior points
            - 'boundary_fraction': Fraction of points near boundary
            - 'boundary_ill_conditioned_rate': Rate of ill-conditioning at boundary
            - 'interior_ill_conditioned_rate': Rate of ill-conditioning in interior
            - 'diagnosis': String describing the conditioning pattern

        Example:
            >>> operator = TaylorOperator(points, delta=0.1)
            >>> analysis = operator.get_boundary_conditioning_analysis()
            >>> if analysis['boundary_ill_conditioned_rate'] > 0.5:
            ...     print("Consider increasing delta near boundaries")
        """
        if boundary_tolerance is None:
            boundary_tolerance = 1.5 * self.delta

        # Determine domain bounds
        if domain_bounds is None:
            # Infer from point cloud with small margin
            mins = np.min(self._points, axis=0)
            maxs = np.max(self._points, axis=0)
            domain_bounds = np.column_stack([mins, maxs])
        else:
            domain_bounds = np.asarray(domain_bounds)
            if domain_bounds.shape[1] != 2:
                domain_bounds = domain_bounds.T  # Handle (2, d) format

        # Classify points as boundary-adjacent or interior
        boundary_mask = np.zeros(self._n_points, dtype=bool)
        for d in range(self._dimension):
            near_min = self._points[:, d] - domain_bounds[d, 0] < boundary_tolerance
            near_max = domain_bounds[d, 1] - self._points[:, d] < boundary_tolerance
            boundary_mask |= near_min | near_max

        boundary_indices = np.where(boundary_mask)[0]
        interior_indices = np.where(~boundary_mask)[0]

        # Get condition numbers
        cond_stats = self.get_condition_numbers()
        cond_values = cond_stats["values"]

        # Compute stats for each group
        def compute_group_stats(indices: np.ndarray, values: np.ndarray) -> dict:
            if len(indices) == 0:
                return {
                    "count": 0,
                    "min": np.inf,
                    "max": np.inf,
                    "mean": np.inf,
                    "median": np.inf,
                    "n_ill_conditioned": 0,
                    "ill_conditioned_rate": 0.0,
                }
            group_values = values[indices]
            finite_values = group_values[np.isfinite(group_values)]
            n_ill = int(np.sum(group_values > 1e12))
            return {
                "count": len(indices),
                "min": float(np.min(finite_values)) if len(finite_values) > 0 else np.inf,
                "max": float(np.max(finite_values)) if len(finite_values) > 0 else np.inf,
                "mean": float(np.mean(finite_values)) if len(finite_values) > 0 else np.inf,
                "median": float(np.median(finite_values)) if len(finite_values) > 0 else np.inf,
                "n_ill_conditioned": n_ill,
                "ill_conditioned_rate": n_ill / len(indices) if len(indices) > 0 else 0.0,
            }

        boundary_stats = compute_group_stats(boundary_indices, cond_values)
        interior_stats = compute_group_stats(interior_indices, cond_values)

        # Generate diagnosis
        diagnosis_parts = []
        if boundary_stats["count"] > 0:
            b_rate = boundary_stats["ill_conditioned_rate"]
            i_rate = interior_stats["ill_conditioned_rate"] if interior_stats["count"] > 0 else 0.0

            if b_rate > 0.5 and b_rate > 2 * i_rate:
                diagnosis_parts.append(
                    f"Boundary stencils are primary issue ({b_rate:.0%} ill-conditioned vs {i_rate:.0%} interior). "
                    "Consider: (1) increasing delta near boundaries, (2) using adaptive delta, "
                    "(3) adding explicit boundary layer points."
                )
            elif b_rate > 0.2:
                diagnosis_parts.append(
                    f"Moderate boundary conditioning issues ({b_rate:.0%} ill-conditioned). "
                    "Monitor for numerical instabilities."
                )

        if interior_stats["count"] > 0 and interior_stats["ill_conditioned_rate"] > 0.1:
            diagnosis_parts.append(
                f"Interior points also have issues ({interior_stats['ill_conditioned_rate']:.0%} ill-conditioned). "
                "Consider: (1) increasing base delta, (2) using RBF-FD instead of Taylor."
            )

        if not diagnosis_parts:
            diagnosis_parts.append("Stencil conditioning is acceptable.")

        return {
            "interior_stats": interior_stats,
            "boundary_stats": boundary_stats,
            "boundary_indices": boundary_indices,
            "interior_indices": interior_indices,
            "boundary_fraction": len(boundary_indices) / self._n_points if self._n_points > 0 else 0.0,
            "boundary_ill_conditioned_rate": boundary_stats["ill_conditioned_rate"],
            "interior_ill_conditioned_rate": interior_stats["ill_conditioned_rate"],
            "diagnosis": " ".join(diagnosis_parts),
        }


# =============================================================================
# UpwindOperator: Flow-Biased Stencils for Advection
# =============================================================================


class UpwindOperator(TaylorOperator):
    """
    GFDM operator with flow-biased stencils for advection-dominated problems.

    For HJB equations with Hamiltonian H(grad u) or FP equations with drift,
    standard central stencils cause oscillations. Upwind stencils bias the
    neighborhood selection toward the upstream direction.

    Note: This is a STRICT upwind scheme. For smoother solutions, consider
    a "biased central" variant with downwind neighbors at lower weights.
    """

    def __init__(
        self,
        points: np.ndarray,
        velocity_field: np.ndarray,
        delta: float = 0.1,
        taylor_order: int = 2,
        **kwargs,
    ):
        """
        Initialize upwind operator.

        Args:
            points: Collocation points, shape (n_points, dimension)
            velocity_field: Velocity/drift at each point, shape (n_points, dimension)
            delta: Neighborhood radius
            taylor_order: Order of Taylor expansion
            **kwargs: Additional arguments passed to TaylorOperator
        """
        self.velocity_field = np.asarray(velocity_field)
        super().__init__(points, delta=delta, taylor_order=taylor_order, **kwargs)

    def _build_neighborhoods(self):
        """Build neighborhoods with upwind bias."""
        tree = cKDTree(self._points)

        self.neighborhoods: list[dict] = []
        self._hybrid_expanded_count = 0

        for i in range(self._n_points):
            # Get candidate neighbors within larger radius
            search_radius = 1.5 * self.delta
            candidate_indices = tree.query_ball_point(self._points[i], search_radius)

            # Filter based on velocity (keep upstream neighbors)
            v = self.velocity_field[i]
            v_norm = np.linalg.norm(v)

            valid_neighbors = []
            for j in candidate_indices:
                dx = self._points[j] - self._points[i]
                # Keep if:
                # 1. Same point (center)
                # 2. Neighbor is "behind" the flow (dot product <= 0)
                # 3. Very close (within core radius)
                dist = np.linalg.norm(dx)
                if j == i:
                    valid_neighbors.append(j)
                elif dist < 0.3 * self.delta:
                    # Core region: include regardless of direction
                    valid_neighbors.append(j)
                elif v_norm < 1e-10:
                    # No flow: use standard stencil
                    if dist < self.delta:
                        valid_neighbors.append(j)
                else:
                    # Upwind check: dot(dx, v) <= 0 means upstream
                    if np.dot(dx, v) <= 1e-10 * v_norm * dist:
                        valid_neighbors.append(j)

            neighbor_indices = np.array(valid_neighbors)

            # Ensure minimum neighbors (fallback to k-NN if needed)
            if len(neighbor_indices) < self.k_neighbors:
                distances, neighbor_indices = tree.query(self._points[i], k=self.k_neighbors)
                neighbor_indices = np.array(neighbor_indices)
                self._hybrid_expanded_count += 1

            neighbor_points = self._points[neighbor_indices]
            distances = np.linalg.norm(neighbor_points - self._points[i], axis=1)

            self.neighborhoods.append(
                {
                    "indices": neighbor_indices,
                    "points": neighbor_points,
                    "distances": distances,
                    "size": len(neighbor_indices),
                }
            )

        if self._hybrid_expanded_count > 0:
            import warnings

            pct = 100.0 * self._hybrid_expanded_count / self._n_points
            warnings.warn(
                f"Upwind stencil: {self._hybrid_expanded_count}/{self._n_points} points ({pct:.1f}%) "
                f"had insufficient upstream neighbors. Used k-NN fallback.",
                UserWarning,
                stacklevel=2,
            )


# =============================================================================
# LocalRBFOperator: RBF-FD with PHS + Polynomial Augmentation
# =============================================================================


class LocalRBFOperator(DifferentialOperator):
    """
    Local RBF-FD operator with Polyharmonic Splines and polynomial augmentation.

    This is the modern RBF-FD approach (Fornberg & Flyer, 2015):
    - Uses LOCAL stencils (like GFDM), not global interpolation
    - PHS basis: phi(r) = r^m (no shape parameter to tune)
    - Polynomial augmentation ensures polynomial reproduction up to degree p

    Mathematical Formulation:
    ------------------------
    For each point x_i with k neighbors, solve the augmented system:

        [Phi  P ] [lambda]   [u    ]
        [P^T  0 ] [  c   ] = [0    ]

    where:
        Phi_jk = phi(||x_j - x_k||) = r_jk^m  (PHS matrix)
        P_jl = p_l(x_j)                       (polynomial terms)

    Derivative weights come from differentiating the interpolant:
        s(x) = sum_j lambda_j phi(||x - x_j||) + sum_l c_l p_l(x)

    Key Properties:
    - No shape parameter (unlike Gaussian/Wendland)
    - Polynomial reproduction up to degree p
    - Conditionally positive definite (requires augmentation)
    - Optimal convergence: O(h^{p+1}) for smooth functions

    Row Sum Properties:
    ------------------
    RBF-FD gradient weights should sum to zero (translation invariance).
    At boundary points with asymmetric stencils, this may be violated.

    Unlike TaylorOperator, RBF-FD computes individual Hessian diagonal
    components (d²u/dx_k²) directly, not by splitting the Laplacian.
    This gives correct second derivatives even for anisotropic functions.

    Common Configurations:
    - PHS m=3, poly p=2: Cubic accuracy (recommended for 2D)
    - PHS m=5, poly p=4: Quintic accuracy (higher cost)
    - Rule: p >= (m-1)/2 for positive definiteness

    References:
    - Fornberg, B., Flyer, N. (2015). "A Primer on Radial Basis Functions"
    - Flyer, N., et al. (2016). "On the role of polynomials in RBF-FD"

    Example:
        >>> op = LocalRBFOperator(points, phs_order=3, poly_degree=2)
        >>> grad = op.gradient(u)
        >>> lap = op.laplacian(u)
    """

    def __init__(
        self,
        points: np.ndarray,
        delta: float = 0.1,
        kernel: str = "phs3",
        poly_degree: int = 2,
        k_neighbors: int | None = None,
        neighborhood_mode: str = "hybrid",
        phs_order: int | None = None,  # Deprecated, use kernel="phs3" etc.
    ):
        """
        Initialize local RBF-FD operator.

        Args:
            points: Collocation points, shape (n_points, dimension)
            delta: Neighborhood radius (also shape parameter for non-PHS kernels)
            kernel: RBF kernel type:
                - "phs1", "phs3", "phs5", "phs7": Polyharmonic splines (no shape param)
                - "gaussian": Gaussian RBF
                - "wendland_c0", "wendland_c2", "wendland_c4", "wendland_c6": Wendland
                - "multiquadric": sqrt(1 + (r/delta)^2)
            poly_degree: Polynomial augmentation degree p (required for PHS,
                         optional but recommended for others)
            k_neighbors: Number of neighbors (auto-computed if None)
            neighborhood_mode: "radius", "knn", or "hybrid"
            phs_order: DEPRECATED - use kernel="phs3" instead
        """
        self._points = np.asarray(points)
        self._n_points, self._dimension = self._points.shape
        self.delta = delta
        self.poly_degree = poly_degree
        self.neighborhood_mode = neighborhood_mode

        # Handle deprecated phs_order parameter
        if phs_order is not None:
            import warnings

            warnings.warn(
                "phs_order is deprecated, use kernel='phs3' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            kernel = f"phs{phs_order}"

        self.kernel_name = kernel

        # Create kernel object (unified interface via factory)
        self._kernel = create_kernel(kernel, dimension=self._dimension)

        # Count polynomial terms: C(d+p, p) = (d+p)! / (d! p!)
        self.n_poly_terms = self._count_poly_terms(self._dimension, poly_degree)

        # Minimum neighbors = PHS stencil size + polynomial terms
        min_neighbors = self.n_poly_terms + 1
        if k_neighbors is None:
            self.k_neighbors = max(min_neighbors + 2, 2 * self.n_poly_terms)
        else:
            self.k_neighbors = max(k_neighbors, min_neighbors)

        # Build polynomial multi-indices
        self.poly_indices = self._build_poly_indices()

        # Precompute neighborhoods
        self._build_neighborhoods()

        # Precompute RBF-FD weights and validate
        self._build_rbf_weights()
        self._validate_stencils()

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def multi_indices(self) -> list[tuple[int, ...]]:
        """Return multi-indices for derivatives that can be computed.

        LocalRBFOperator computes first derivatives (gradient) and second
        derivatives (Laplacian). Returns indices for order 1 and 2.
        """
        indices: list[tuple[int, ...]] = []
        # First derivatives: (1,0,0...), (0,1,0...), etc.
        for k in range(self._dimension):
            multi_idx = tuple(1 if j == k else 0 for j in range(self._dimension))
            indices.append(multi_idx)
        # Second derivatives: (2,0,0...), (0,2,0...), etc.
        for k in range(self._dimension):
            multi_idx = tuple(2 if j == k else 0 for j in range(self._dimension))
            indices.append(multi_idx)
        return indices

    def _count_poly_terms(self, d: int, p: int) -> int:
        """Count polynomial terms: C(d+p, p)."""
        from math import comb

        return comb(d + p, p)

    def _build_poly_indices(self) -> list[tuple[int, ...]]:
        """Generate polynomial multi-indices up to degree p."""
        d, p = self._dimension, self.poly_degree
        indices: list[tuple[int, ...]] = []

        def generate(current: list[int], remaining_dims: int, remaining_order: int):
            if remaining_dims == 0:
                if sum(current) <= p:
                    indices.append(tuple(current))
                return
            for i in range(remaining_order + 1):
                generate([*current, i], remaining_dims - 1, remaining_order - i)

        generate([], d, p)
        indices.sort(key=lambda beta: (sum(beta), beta))
        return indices

    def _build_neighborhoods(self):
        """Build neighborhood structure (same as TaylorOperator)."""
        tree = cKDTree(self._points)
        self.neighborhoods: list[dict] = []
        self._hybrid_expanded_count = 0

        for i in range(self._n_points):
            if self.neighborhood_mode == "knn":
                distances, neighbor_indices = tree.query(self._points[i], k=self.k_neighbors)
                neighbor_indices = np.array(neighbor_indices)
                distances = np.array(distances)
            elif self.neighborhood_mode == "radius":
                neighbor_indices = tree.query_ball_point(self._points[i], self.delta)
                neighbor_indices = np.array(neighbor_indices)
                neighbor_points = self._points[neighbor_indices]
                distances = np.linalg.norm(neighbor_points - self._points[i], axis=1)
            else:
                # Hybrid mode
                neighbor_indices = tree.query_ball_point(self._points[i], self.delta)
                neighbor_indices = np.array(neighbor_indices)

                if len(neighbor_indices) < self.k_neighbors:
                    distances, neighbor_indices = tree.query(self._points[i], k=self.k_neighbors)
                    neighbor_indices = np.array(neighbor_indices)
                    distances = np.array(distances)
                    self._hybrid_expanded_count += 1
                else:
                    neighbor_points = self._points[neighbor_indices]
                    distances = np.linalg.norm(neighbor_points - self._points[i], axis=1)

            neighbor_points = self._points[neighbor_indices]
            self.neighborhoods.append(
                {
                    "indices": neighbor_indices,
                    "points": neighbor_points,
                    "distances": distances,
                    "size": len(neighbor_indices),
                }
            )

    def _build_rbf_weights(self):
        """Precompute RBF-FD weights for gradient and Laplacian."""
        self.rbf_weights: list[dict | None] = []

        for i in range(self._n_points):
            neighborhood = self.neighborhoods[i]
            n_neighbors = neighborhood["size"]
            n_poly = self.n_poly_terms

            if n_neighbors < n_poly:
                self.rbf_weights.append(None)
                continue

            neighbor_points = neighborhood["points"]
            center = self._points[i]

            # Shift to local coordinates
            local_coords = neighbor_points - center

            # Build RBF matrix Phi (n x n)
            Phi = self._build_rbf_matrix(local_coords)

            # Build polynomial matrix P (n x m)
            P = self._build_poly_matrix(local_coords)

            # Augmented system: [Phi, P; P^T, 0]
            n = n_neighbors
            m = n_poly
            A_aug = np.zeros((n + m, n + m))
            A_aug[:n, :n] = Phi
            A_aug[:n, n:] = P
            A_aug[n:, :n] = P.T

            # Regularization for stability
            A_aug[:n, :n] += REGULARIZATION * np.eye(n)

            try:
                # Compute condition number before taking pseudoinverse
                # Use SVD for accurate condition number
                S = np.linalg.svd(A_aug, compute_uv=False)
                if S[-1] > ZERO_TOL:
                    cond = S[0] / S[-1]
                else:
                    cond = np.inf

                A_inv = np.linalg.pinv(A_aug, rcond=PINV_RCOND)
            except np.linalg.LinAlgError:
                self.rbf_weights.append(None)
                continue

            # Compute derivative weights at center (x=0)
            # For gradient: d/dx_k [phi(r)] at r=0, d/dx_k [p_l(x)] at x=0
            # For Laplacian: Delta[phi(r)] at r=0, Delta[p_l(x)] at x=0

            grad_weights = np.zeros((self._dimension, n_neighbors))
            hess_diag_weights = np.zeros((self._dimension, n_neighbors))
            lap_weights = np.zeros(n_neighbors)

            # Precompute kernel derivatives at all neighbor distances
            distances = neighborhood["distances"]
            dphi_dr = np.zeros(n_neighbors)
            d2phi_dr2 = np.zeros(n_neighbors)

            for j in range(n_neighbors):
                r = distances[j]
                if r > ZERO_TOL:
                    _, dphi = self._kernel.evaluate_with_derivative(r, self.delta)
                    dphi_dr[j] = float(dphi)
                    # Compute d²φ/dr² from Laplacian: Δφ = d²φ/dr² + (d-1)/r * dφ/dr
                    lap_phi = self._kernel.laplacian(r, self.delta, self._dimension)
                    d2phi_dr2[j] = float(lap_phi) - (self._dimension - 1) / r * dphi_dr[j]

            for k in range(self._dimension):
                # RHS for d/dx_k: derivative of RBF and polynomials at center
                rhs_grad = np.zeros(n + m)

                # d/dx_k phi(||x - x_j||) at x=0
                # d phi/dx_k = d phi/dr * (x_k - x_j_k) / r = d phi/dr * (-local_j_k) / r
                for j in range(n_neighbors):
                    r = distances[j]
                    if r > ZERO_TOL:
                        rhs_grad[j] = dphi_dr[j] * (-local_coords[j, k]) / r

                # d/dx_k p_idx(x) at x=0
                for poly_idx, alpha in enumerate(self.poly_indices):
                    if alpha[k] >= 1:
                        # Derivative of x^alpha w.r.t. x_k at origin
                        # Only non-zero if alpha = e_k (unit vector)
                        if sum(alpha) == 1 and alpha[k] == 1:
                            rhs_grad[n + poly_idx] = 1.0

                # Solve for gradient weights
                weights = A_inv @ rhs_grad
                grad_weights[k, :] = weights[:n_neighbors]

                # RHS for d²/dx_k²: second derivative of RBF and polynomials at center
                # d²φ/dx_k² = d²φ/dr² * (x_k/r)² + dφ/dr * (r² - x_k²) / r³
                rhs_hess = np.zeros(n + m)

                for j in range(n_neighbors):
                    r = distances[j]
                    if r > ZERO_TOL:
                        xk = local_coords[j, k]
                        rhs_hess[j] = d2phi_dr2[j] * (xk / r) ** 2 + dphi_dr[j] * (r**2 - xk**2) / r**3

                # d²/dx_k² p_idx(x) at x=0: only non-zero if alpha = 2*e_k
                for poly_idx, alpha in enumerate(self.poly_indices):
                    if alpha[k] == 2 and sum(alpha) == 2:
                        rhs_hess[n + poly_idx] = 2.0

                # Solve for Hessian diagonal weights
                weights_hess = A_inv @ rhs_hess
                hess_diag_weights[k, :] = weights_hess[:n_neighbors]

            # Laplacian = sum of Hessian diagonal (verify consistency)
            lap_weights = hess_diag_weights.sum(axis=0)

            # Cross-derivative weights (off-diagonal Hessian)
            # For d²u/dx_k dx_dim2 (k != dim2):
            # d²φ/dx_k dx_dim2 = d²φ/dr² * (x_k/r)(x_dim2/r) + dφ/dr * (-x_k x_dim2 / r³)
            hess_cross_weights = {}  # Map (k, dim2) -> weights array, k < dim2

            for k in range(self._dimension):
                for dim2 in range(k + 1, self._dimension):
                    rhs_cross = np.zeros(n + m)

                    for j in range(n_neighbors):
                        r = distances[j]
                        if r > ZERO_TOL:
                            xk = local_coords[j, k]
                            x2 = local_coords[j, dim2]
                            rhs_cross[j] = d2phi_dr2[j] * (xk / r) * (x2 / r) + dphi_dr[j] * (-xk * x2 / r**3)

                    # d²/dx_k dx_dim2 p(x) at x=0: only non-zero if alpha has 1 in both k and dim2
                    for idx_poly, alpha in enumerate(self.poly_indices):
                        if alpha[k] == 1 and alpha[dim2] == 1 and sum(alpha) == 2:
                            rhs_cross[n + idx_poly] = 1.0

                    weights_cross = A_inv @ rhs_cross
                    hess_cross_weights[(k, dim2)] = weights_cross[:n_neighbors]

            # Find center index in neighbors
            center_idx = -1
            for idx, n_idx in enumerate(neighborhood["indices"]):
                if n_idx == i:
                    center_idx = idx
                    break

            self.rbf_weights.append(
                {
                    "neighbor_indices": neighborhood["indices"],
                    "grad_weights": grad_weights,
                    "hess_diag_weights": hess_diag_weights,
                    "hess_cross_weights": hess_cross_weights,
                    "lap_weights": lap_weights,
                    "center_idx_in_neighbors": center_idx,
                    "A_inv": A_inv,
                    "condition_number": cond,
                }
            )

    def _build_rbf_matrix(self, local_coords: np.ndarray) -> np.ndarray:
        """Build RBF matrix Phi_ij = phi(||x_i - x_j||) using unified kernel."""
        n = len(local_coords)
        Phi = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                r = np.linalg.norm(local_coords[i] - local_coords[j])
                Phi[i, j] = self._kernel(r, self.delta)

        return Phi

    def _build_poly_matrix(self, local_coords: np.ndarray) -> np.ndarray:
        """Build polynomial matrix P_ij = p_j(x_i)."""
        n = len(local_coords)
        m = self.n_poly_terms
        P = np.zeros((n, m))
        for i in range(n):
            for poly_idx, alpha in enumerate(self.poly_indices):
                term = 1.0
                for d in range(self._dimension):
                    if alpha[d] > 0:
                        term *= local_coords[i, d] ** alpha[d]
                P[i, poly_idx] = term
        return P

    def _validate_stencils(self) -> bool:
        """
        Validate stencil quality after RBF-FD weight construction.

        Checks that all points have valid RBF-FD weights. Unlike TaylorOperator,
        LocalRBFOperator uses PHS kernels without compact support, so weight
        computation rarely fails. Issues typically indicate insufficient neighbors.

        Returns:
            True if all stencils are valid, False if any failed.

        Notes:
            Minimum neighbors needed: n_poly_terms = C(d+p, p)
            where d = dimension, p = poly_degree.

            Examples (poly_degree=2):
              1D: 3 neighbors,  2D: 6 neighbors,  3D: 10 neighbors
        """
        import warnings

        failed_points = []
        for i in range(self._n_points):
            if self.rbf_weights[i] is None:
                n_neighbors = self.neighborhoods[i]["size"]
                failed_points.append((i, n_neighbors))

        if failed_points:
            n_issues = len(failed_points)
            examples = failed_points[:3]
            msg_lines = [
                f"LocalRBFOperator: {n_issues}/{self._n_points} points failed weight computation.",
                f"  Minimum neighbors needed: {self.n_poly_terms} "
                f"(for poly_degree={self.poly_degree} in {self._dimension}D)",
            ]
            for idx, n_neighbors in examples:
                msg_lines.append(f"  Point {idx}: {n_neighbors} neighbors < {self.n_poly_terms} required")
            if n_issues > 3:
                msg_lines.append(f"  ... and {n_issues - 3} more")
            msg_lines.append("  Fix: Increase k_neighbors or delta to capture more neighbors.")

            warnings.warn("\n".join(msg_lines), UserWarning, stacklevel=3)
            return False

        return True

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Compute gradient at all points."""
        grad = np.zeros((self._n_points, self._dimension))

        for i in range(self._n_points):
            weights = self.rbf_weights[i]
            if weights is None:
                continue

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]
            u_neighbors = u[neighbor_indices]

            for k in range(self._dimension):
                grad[i, k] = np.dot(grad_weights[k], u_neighbors)

        return grad

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """Compute Laplacian at all points."""
        lap = np.zeros(self._n_points)

        for i in range(self._n_points):
            weights = self.rbf_weights[i]
            if weights is None:
                continue

            neighbor_indices = weights["neighbor_indices"]
            lap_weights = weights["lap_weights"]
            u_neighbors = u[neighbor_indices]

            lap[i] = np.dot(lap_weights, u_neighbors)

        return lap

    def get_derivative_weights(self, point_idx: int) -> dict | None:
        """Get derivative weights for matrix assembly."""
        return self.rbf_weights[point_idx]

    def get_neighborhood(self, point_idx: int) -> dict:
        """Get neighborhood data for a point."""
        return self.neighborhoods[point_idx]

    def get_taylor_data(self, point_idx: int) -> dict | None:
        """Get Taylor-like data for compatibility with HJBGFDMSolver.

        LocalRBFOperator doesn't use Taylor expansion, but provides this method
        for API compatibility. Returns RBF weights formatted as Taylor-like data.
        """
        weights = self.rbf_weights[point_idx]
        if weights is None:
            return None

        # Return in Taylor-compatible format (minimal data needed)
        return {
            "A": None,  # No Taylor matrix
            "W": None,
            "sqrt_W": None,
            "U": None,
            "S": np.array([1.0]),  # Dummy singular values
            "Vt": None,
            "rank": 1,
        }

    def approximate_derivatives_at_point(self, u: np.ndarray, point_idx: int) -> dict[tuple[int, ...], float]:
        """
        Compute derivatives at a single point using RBF-FD weights.

        Returns first derivatives (gradient), diagonal second derivatives
        (Hessian diagonal), and cross-derivatives (off-diagonal Hessian).

        Args:
            u: Function values at all points
            point_idx: Index of point where derivatives are computed

        Returns:
            Dict mapping multi-indices to derivative values:
            - (1,0,...) for ∂u/∂x₁
            - (0,1,...) for ∂u/∂x₂
            - (2,0,...) for ∂²u/∂x₁²
            - (0,2,...) for ∂²u/∂x₂²
            - (1,1,...) for ∂²u/∂x₁∂x₂ (cross-derivative)
        """
        weights = self.rbf_weights[point_idx]
        if weights is None:
            return {}

        neighbor_indices = weights["neighbor_indices"]
        u_neighbors = u[neighbor_indices]
        grad_weights = weights["grad_weights"]
        hess_diag_weights = weights["hess_diag_weights"]
        hess_cross_weights = weights.get("hess_cross_weights", {})

        result: dict[tuple[int, ...], float] = {}

        # First derivatives (gradient components)
        for k in range(self._dimension):
            multi_idx = tuple(1 if j == k else 0 for j in range(self._dimension))
            result[multi_idx] = float(np.dot(grad_weights[k], u_neighbors))

        # Second derivatives (Hessian diagonal components)
        for k in range(self._dimension):
            multi_idx = tuple(2 if j == k else 0 for j in range(self._dimension))
            result[multi_idx] = float(np.dot(hess_diag_weights[k], u_neighbors))

        # Cross-derivatives (off-diagonal Hessian components)
        for (k, dim2), cross_weights in hess_cross_weights.items():
            # Build multi-index with 1 at positions k and dim2
            multi_idx_list = [0] * self._dimension
            multi_idx_list[k] = 1
            multi_idx_list[dim2] = 1
            multi_idx = tuple(multi_idx_list)
            result[multi_idx] = float(np.dot(cross_weights, u_neighbors))

        return result

    def get_condition_numbers(self) -> dict:
        """
        Get condition number statistics across all stencils.

        Returns:
            Dictionary with:
            - 'values': array of condition numbers (inf for failed points)
            - 'min': minimum condition number
            - 'max': maximum finite condition number
            - 'mean': mean of finite condition numbers
            - 'median': median of finite condition numbers
            - 'n_ill_conditioned': count where cond > 1e12
            - 'n_failed': count where cond = inf (construction failed)

        Notes:
            For RBF-FD, condition number measures the sensitivity of the
            augmented RBF+polynomial system. High values indicate:
            - Near-singular RBF matrix (neighbors too far or too close)
            - Polynomial terms nearly linearly dependent on RBF terms
        """
        cond_numbers = []
        for weights in self.rbf_weights:
            if weights is None:
                cond_numbers.append(np.inf)
            else:
                cond_numbers.append(weights.get("condition_number", np.inf))

        values = np.array(cond_numbers)
        finite_values = values[np.isfinite(values)]

        return {
            "values": values,
            "min": float(np.min(finite_values)) if len(finite_values) > 0 else np.inf,
            "max": float(np.max(finite_values)) if len(finite_values) > 0 else np.inf,
            "mean": float(np.mean(finite_values)) if len(finite_values) > 0 else np.inf,
            "median": float(np.median(finite_values)) if len(finite_values) > 0 else np.inf,
            "n_ill_conditioned": int(np.sum(values > 1e12)),
            "n_failed": int(np.sum(np.isinf(values))),
        }


# =============================================================================
# DirectCollocationHandler: Row Replacement for BC Enforcement
# =============================================================================


class DirectCollocationHandler(BoundaryHandler):
    """
    Boundary handler using Row Replacement (Direct Collocation) pattern.

    For each boundary point:
    1. Clear the PDE row in the matrix
    2. Insert the BC equation:
       - Dirichlet: A[i, i] = 1, b[i] = g
       - Neumann: A[i, :] = normal derivative weights, b[i] = g

    This is the CORRECT approach for GFDM boundary conditions.
    Ghost particles should NOT be used for derivative computation.
    """

    def apply_to_matrix(
        self,
        A: np.ndarray,
        b: np.ndarray,
        boundary_indices: np.ndarray | set[int],
        operator: DifferentialOperator,
        bc_config: dict,
    ) -> None:
        """Apply Row Replacement to system matrix."""
        bc_type = bc_config.get("type", "neumann")
        bc_values = bc_config.get("values", {})
        normals = bc_config.get("normals")
        dimension = operator.dimension

        # Convert to list if set
        if isinstance(boundary_indices, set):
            boundary_indices = np.array(list(boundary_indices))

        for local_idx, global_idx in enumerate(boundary_indices):
            # Clear the row
            A[global_idx, :] = 0.0

            if bc_type == "dirichlet":
                # Dirichlet: u = g
                A[global_idx, global_idx] = 1.0
                if isinstance(bc_values, dict):
                    b[global_idx] = bc_values.get(global_idx, 0.0)
                else:
                    b[global_idx] = bc_values

            elif bc_type in ("neumann", "no_flux"):
                # Neumann: du/dn = g (usually g=0 for no-flux)
                weights = operator.get_derivative_weights(global_idx)
                if weights is None:
                    # Fallback: simple identity
                    A[global_idx, global_idx] = 1.0
                    b[global_idx] = 0.0
                    continue

                neighbor_indices = weights["neighbor_indices"]
                grad_weights = weights["grad_weights"]  # Shape: (dim, n_neighbors)

                # Get normal vector
                if normals is not None:
                    normal = normals[local_idx]
                else:
                    # Default: cannot compute without normals
                    raise ValueError(
                        "Neumann BC requires normal vectors. "
                        "Pass normals in bc_config or use BoundaryConditions.get_outward_normal()."
                    )

                # Normal derivative: du/dn = n . grad(u)
                # Weights: sum over dimensions of n[d] * grad_weights[d, :]
                center_weight = 0.0
                for k, j in enumerate(neighbor_indices):
                    if j >= 0 and j != global_idx:
                        weight = sum(normal[d] * grad_weights[d, k] for d in range(dimension))
                        A[global_idx, j] = weight
                        center_weight -= weight

                A[global_idx, global_idx] = center_weight

                # RHS: usually 0 for no-flux
                if isinstance(bc_values, dict):
                    b[global_idx] = bc_values.get(global_idx, 0.0)
                else:
                    b[global_idx] = bc_values if bc_values != {} else 0.0

    def apply_to_residual(
        self,
        residual: np.ndarray,
        u: np.ndarray,
        boundary_indices: np.ndarray | set[int],
        operator: DifferentialOperator,
        bc_config: dict,
    ) -> None:
        """
        Apply BC to residual (MUST match apply_to_matrix for Newton consistency).
        """
        bc_type = bc_config.get("type", "neumann")
        bc_values = bc_config.get("values", {})
        normals = bc_config.get("normals")

        if isinstance(boundary_indices, set):
            boundary_indices = np.array(list(boundary_indices))

        # Compute gradient once if needed for Neumann
        if bc_type in ("neumann", "no_flux"):
            grad_u = operator.gradient(u)

        for local_idx, global_idx in enumerate(boundary_indices):
            if bc_type == "dirichlet":
                # Residual = u - g
                if isinstance(bc_values, dict):
                    target = bc_values.get(global_idx, 0.0)
                else:
                    target = bc_values
                residual[global_idx] = u[global_idx] - target

            elif bc_type in ("neumann", "no_flux"):
                # Residual = du/dn - g
                if normals is None:
                    raise ValueError("Neumann BC requires normal vectors.")

                normal = normals[local_idx]
                du_dn = np.dot(grad_u[global_idx], normal)

                if isinstance(bc_values, dict):
                    target = bc_values.get(global_idx, 0.0)
                else:
                    target = bc_values if bc_values != {} else 0.0

                residual[global_idx] = du_dn - target


# =============================================================================
# GhostNodeHandler: Legacy Ghost Particle BC (for backward compatibility)
# =============================================================================


class GhostNodeHandler(BoundaryHandler):
    """
    Legacy boundary handler using ghost particles.

    WARNING: Ghost particles can corrupt derivative computation at boundary
    points. Use DirectCollocationHandler instead for new code.

    This handler is provided for backward compatibility only.
    """

    def __init__(self, domain_bounds: list[tuple[float, float]] | None = None):
        """
        Initialize ghost node handler.

        Args:
            domain_bounds: Domain bounds for ghost particle placement
        """
        self.domain_bounds = domain_bounds

    def apply_to_matrix(
        self,
        A: np.ndarray,
        b: np.ndarray,
        boundary_indices: np.ndarray | set[int],
        operator: DifferentialOperator,
        bc_config: dict,
    ) -> None:
        """
        Ghost particle BC: No matrix modification needed.

        Ghost particles are handled during derivative computation by
        mirroring function values. This method is a no-op.
        """
        # Ghost particle method doesn't modify matrix - it modifies
        # how function values are computed during derivative evaluation.
        # This is handled in get_neighbor_values() of the operator.

    def apply_to_residual(
        self,
        residual: np.ndarray,
        u: np.ndarray,
        boundary_indices: np.ndarray | set[int],
        operator: DifferentialOperator,
        bc_config: dict,
    ) -> None:
        """
        Ghost particle BC: Residual uses mirrored values.

        No explicit residual modification - the BC is enforced implicitly
        through ghost particle values.
        """


# =============================================================================
# Configuration Dataclass for Clean Parameter Management
# =============================================================================


@dataclass
class OperatorConfig:
    """
    Configuration for differential operator creation.

    This dataclass encapsulates all parameters needed by create_operator(),
    making it easy to pass configuration through solver initialization.

    Attributes:
        method: Operator type ("direct", "upwind", "rbf")
        delta: Neighborhood radius
        taylor_order: Order of Taylor expansion
        weight_function: Weight function type
        k_neighbors: Minimum neighbors (None for auto)
        neighborhood_mode: "radius", "knn", or "hybrid"
        velocity_field: Required for "upwind" method

    Example:
        # In HJBGFDMSolver
        config = OperatorConfig(method="direct", delta=0.1)
        self.operator = config.create_operator(points)

        # Later, switch to upwind when velocity is known
        config.method = "upwind"
        config.velocity_field = optimal_velocity
        self.operator = config.create_operator(points)
    """

    method: str = "direct"
    delta: float = 0.1
    taylor_order: int = 2
    weight_function: str = "wendland"
    k_neighbors: int | None = None
    neighborhood_mode: str = "hybrid"
    velocity_field: np.ndarray | None = None

    def create_operator(self, points: np.ndarray) -> DifferentialOperator:
        """Create operator from this configuration."""
        return create_operator(
            points,
            delta=self.delta,
            method=self.method,
            velocity_field=self.velocity_field,
            taylor_order=self.taylor_order,
            weight_function=self.weight_function,
            k_neighbors=self.k_neighbors,
            neighborhood_mode=self.neighborhood_mode,
        )

    def with_velocity(self, velocity_field: np.ndarray) -> OperatorConfig:
        """Return a new config with velocity_field set (for upwind)."""
        return OperatorConfig(
            method=self.method,
            delta=self.delta,
            taylor_order=self.taylor_order,
            weight_function=self.weight_function,
            k_neighbors=self.k_neighbors,
            neighborhood_mode=self.neighborhood_mode,
            velocity_field=velocity_field,
        )


@dataclass
class BCConfig:
    """
    Configuration for boundary condition handling.

    Attributes:
        method: BC handling method ("direct" or "ghost")
        bc_type: BC type at boundary ("dirichlet", "neumann", "no_flux")
        bc_values: BC values (scalar or dict mapping point_idx -> value)
        normals: Outward normal vectors at boundary points
        domain_bounds: Domain bounds for ghost particles

    Example:
        bc_config = BCConfig(
            method="direct",
            bc_type="neumann",
            normals=normals_array,
        )
        bc_handler = bc_config.create_handler()
        bc_handler.apply_to_matrix(A, b, boundary_idx, operator, bc_config.to_dict())
    """

    method: str = "direct"
    bc_type: str = "neumann"
    bc_values: float | dict[int, float] = 0.0
    normals: np.ndarray | None = None
    domain_bounds: list[tuple[float, float]] | None = None

    def create_handler(self) -> BoundaryHandler:
        """Create BC handler from this configuration."""
        return create_bc_handler(self.method, domain_bounds=self.domain_bounds)

    def to_dict(self) -> dict:
        """Convert to dict format expected by BoundaryHandler.apply_*()."""
        return {
            "type": self.bc_type,
            "values": self.bc_values,
            "normals": self.normals,
        }


# =============================================================================
# Parameter Recommendation Functions
# =============================================================================


def compute_adaptive_gfdm_params(
    points: np.ndarray,
    taylor_order: int = 2,
    overdetermination: float = 3.0,
    safety_factor: float = 1.2,
    fill_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-point adaptive (k, delta) for GFDM with unified design.

    Uses the same overdetermination ratio ρ as compute_gfdm_parameters().
    For each point, ensures at least k = ρ × n_monomials neighbors by
    expanding δ at boundary/corner points where the base δ is insufficient.

    Args:
        points: Collocation points, shape (n_points, dimension)
        taylor_order: Order of Taylor expansion (default 2)
        overdetermination: Target ratio ρ = n_neighbors / n_monomials.
            Default 3.0.
        safety_factor: Multiplier for δ_base. Default 1.2.
        fill_distance: Pre-computed h. If None, estimated from points.

    Returns:
        k_arr: Per-point k_neighbors, shape (n_points,)
        delta_arr: Per-point support radius, shape (n_points,)

    Mathematical Basis (Unified Design):
        - k_target = ρ × n_monomials (same for all points)
        - δ_base = h × (k / V_d)^(1/d) × safety_factor
        - Interior: δ = δ_base, k = actual neighbors in δ
        - Boundary/corner: δ expands until k_target neighbors

    Example:
        >>> k_arr, delta_arr = compute_adaptive_gfdm_params(points)
        >>> operator = TaylorOperator(points, adaptive_params=(k_arr, delta_arr))
    """
    from math import comb, pi

    from scipy.spatial import cKDTree

    points = np.asarray(points)
    n_points, dimension = points.shape

    # Estimate fill distance
    if fill_distance is None:
        ranges = np.ptp(points, axis=0)
        domain_volume = np.prod(ranges)
        fill_distance = (domain_volume / n_points) ** (1.0 / dimension)

    # Compute parameters from unified design
    # n_derivatives = unknowns to solve (excludes constant term)
    n_monomials = comb(dimension + taylor_order, taylor_order)
    n_derivatives = n_monomials - 1
    k_target = int(overdetermination * n_derivatives)

    # Compute δ_base from k_target (same formula as compute_gfdm_parameters)
    # V_d = π^(d/2) / Γ(d/2 + 1) is unit ball volume in d dimensions
    from math import gamma

    v_d = (pi ** (dimension / 2)) / gamma(dimension / 2 + 1)
    delta_base = fill_distance * (k_target / v_d) ** (1.0 / dimension) * safety_factor

    # Build KD-tree
    tree = cKDTree(points)

    # Per-point adaptive parameters
    k_arr = np.zeros(n_points, dtype=int)
    delta_arr = np.zeros(n_points)

    for i in range(n_points):
        # Count neighbors in delta_base
        neighbors_in_delta = len(tree.query_ball_point(points[i], delta_base)) - 1

        if neighbors_in_delta >= k_target:
            # Sufficient neighbors: use delta_base and all neighbors within
            k_arr[i] = neighbors_in_delta
            delta_arr[i] = delta_base
        else:
            # Insufficient neighbors (boundary/corner): expand delta
            distances, _ = tree.query(points[i], k=k_target + 1)
            delta_arr[i] = distances[-1] * 1.1  # 10% margin
            k_arr[i] = k_target

    return k_arr, delta_arr


def wendland_c2_effective_ratio(dimension: int) -> float:
    """
    Compute the effective weight ratio for Wendland C2 kernel in d dimensions.

    In a d-dimensional ball, distances follow PDF p(r) ~ r^(d-1) (shell effect).
    Most points concentrate near the boundary where Wendland weights are small.

    This function computes eta(d) = E[w(r)] analytically using Beta functions:
        eta(d) = d × [B(d,5) + 4×B(d+1,5)]

    For integer d, this simplifies to:
        eta(d) = d × 24 × [1/prod(d..d+4) + 4/prod(d+1..d+5)]

    Args:
        dimension: Spatial dimension d (1, 2, 3, ...)

    Returns:
        Effective weight ratio eta(d):
        - 1D: 0.333 (1/3)
        - 2D: 0.143 (1/7)
        - 3D: 0.071 (1/14)

    Usage:
        To achieve rho_effective overdetermination, set:
        k_nominal = ceil(rho_effective × n_derivatives / eta(d))
    """
    # For integer dimensions, use factorial form (exact, avoids gamma function)
    # B(d,5) = 4! / [d(d+1)(d+2)(d+3)(d+4)]
    # B(d+1,5) = 4! / [(d+1)(d+2)(d+3)(d+4)(d+5)]
    d = dimension
    prod1 = d * (d + 1) * (d + 2) * (d + 3) * (d + 4)
    prod2 = (d + 1) * (d + 2) * (d + 3) * (d + 4) * (d + 5)

    B_d_5 = 24.0 / prod1
    B_d1_5 = 24.0 / prod2

    return d * (B_d_5 + 4 * B_d1_5)


def compute_gfdm_parameters(
    points: np.ndarray,
    taylor_order: int = 2,
    overdetermination: float | None = None,
    effective_overdetermination: float = 1.5,
    safety_factor: float = 1.2,
    fill_distance: float | None = None,
) -> dict:
    """
    Compute recommended GFDM parameters based on point cloud geometry.

    Uses a DIMENSION-AWARE design that accounts for the Wendland weight decay
    in high dimensions (hypersphere shell effect). The key insight is that
    in d dimensions, most neighbors lie near the delta boundary where weights
    are small, so we need more nominal neighbors to achieve the same effective
    overdetermination.

    Args:
        points: Collocation points, shape (n_points, dimension)
        taylor_order: Order of Taylor expansion (default 2)
        overdetermination: DEPRECATED. Use effective_overdetermination instead.
            If provided, this is treated as nominal rho (old behavior).
        effective_overdetermination: Target EFFECTIVE overdetermination ratio.
            This is the ratio of effective weighted neighbors to unknowns.
            Default 1.5 (validated stable for MFG problems).
            The function automatically computes the nominal k needed to achieve
            this effective ratio based on dimension.
        safety_factor: Multiplier for delta_base to ensure enough neighbors.
            Default 1.2 (20% margin).
        fill_distance: Pre-computed fill distance h. If None, estimated from
            point cloud as h ~ (domain_volume / n_points)^(1/d).

    Returns:
        Dictionary with recommended parameters:
        - 'delta': Support radius (derived from k and geometry)
        - 'k_neighbors': Target number of neighbors (dimension-adjusted)
        - 'fill_distance': Estimated or provided h
        - 'n_monomials': Number of monomials C(d+p, p)
        - 'n_derivatives': Number of unknowns (n_monomials - 1)
        - 'effective_overdetermination': The target effective ratio
        - 'nominal_overdetermination': The nominal ratio (k / n_derivatives)
        - 'wendland_effective_ratio': eta(d), the weight efficiency factor

    Example:
        >>> points = np.random.rand(300, 2)  # 300 points in [0,1]^2
        >>> params = compute_gfdm_parameters(points, taylor_order=2)
        >>> # For 2D: eta=0.143, so k ~ 1.5 * 5 / 0.143 ~ 52
        >>> operator = TaylorOperator(points, **params)

    Mathematical Basis (Dimension-Aware Design):
        - n_derivatives = C(d+p, p) - 1 (unknowns, excluding constant)
        - eta(d) = Wendland C2 effective weight ratio (accounts for shell effect)
        - k_nominal = ceil(effective_overdetermination × n_derivatives / eta(d))
        - delta = base_length × (k / V_d)^(1/d) × safety_factor

    Typical values (effective_overdetermination=1.5, taylor_order=2):
        - 1D: eta=0.333, k~7,  delta ~ 3.8 × base
        - 2D: eta=0.143, k~52, delta ~ 5.0 × base (validated in run 160942)
        - 3D: eta=0.071, k~189, delta ~ 5.8 × base

    Note: k-NN fallback provides additional safety for boundary/corner points
    where the delta-ball extends outside the domain.
    """
    import warnings
    from math import ceil, comb, gamma, pi

    points = np.asarray(points)
    n_points, dimension = points.shape

    # Compute cell_size (average spacing) - determines neighbor density in uniform regions
    ranges = np.ptp(points, axis=0)  # max - min per dimension
    domain_volume = np.prod(ranges)
    cell_size = (domain_volume / n_points) ** (1.0 / dimension)

    # fill_distance (if provided) captures maximum gap in sparse regions
    if fill_distance is None:
        fill_distance = cell_size  # Default: assume quasi-uniform mesh

    # Use max(h, cell_size) to ensure coverage in sparse regions
    base_length = max(fill_distance, cell_size)

    # Compute number of monomials and derivatives
    n_monomials = comb(dimension + taylor_order, taylor_order)
    n_derivatives = n_monomials - 1

    # Dimension-aware k computation using Wendland effective weight ratio
    eta = wendland_c2_effective_ratio(dimension)

    if overdetermination is not None:
        # Legacy mode: user specified nominal overdetermination directly
        warnings.warn(
            "overdetermination parameter is deprecated. "
            "Use effective_overdetermination instead for dimension-aware design.",
            DeprecationWarning,
            stacklevel=2,
        )
        nominal_overdetermination = overdetermination
        k_neighbors = int(nominal_overdetermination * n_derivatives)
        # Back-compute effective overdetermination for reporting
        effective_overdetermination = nominal_overdetermination * eta
    else:
        # New dimension-aware mode: compute nominal k from effective target
        # k_nominal = effective_overdetermination × n_derivatives / eta(d)
        k_neighbors = ceil(effective_overdetermination * n_derivatives / eta)
        nominal_overdetermination = k_neighbors / n_derivatives

    # Compute delta so that a ball of radius delta contains ~k neighbors
    # Volume of unit ball in d dimensions: V_d = pi^(d/2) / Gamma(d/2 + 1)
    v_d = (pi ** (dimension / 2)) / gamma(dimension / 2 + 1)
    delta = base_length * (k_neighbors / v_d) ** (1.0 / dimension) * safety_factor

    return {
        "delta": delta,
        "k_neighbors": k_neighbors,
        "fill_distance": fill_distance,
        "cell_size": cell_size,
        "base_length": base_length,
        "n_monomials": n_monomials,
        "n_derivatives": n_derivatives,
        "taylor_order": taylor_order,
        "dimension": dimension,
        "effective_overdetermination": effective_overdetermination,
        "nominal_overdetermination": nominal_overdetermination,
        "wendland_effective_ratio": eta,
        "safety_factor": safety_factor,
        # Legacy alias
        "overdetermination": nominal_overdetermination,
    }


# =============================================================================
# Factory Functions
# =============================================================================


def create_operator(
    points: np.ndarray,
    delta: float = 0.1,
    method: str = "direct",
    velocity_field: np.ndarray | None = None,
    **kwargs,
) -> DifferentialOperator:
    """
    Factory function for creating differential operators.

    This factory handles the different parameter requirements for each operator type:

    - **TaylorOperator** ("direct"/"taylor"): Standard GFDM, no special parameters
    - **UpwindOperator** ("upwind"): Requires velocity_field for stencil bias
    - **LocalRBFOperator** ("rbf"/"rbf-fd"): Local RBF-FD with PHS + poly augmentation

    Args:
        points: Collocation points, shape (n_points, dimension)
        delta: Neighborhood radius
        method: Operator type:
            - "direct" or "taylor": Standard GFDM (TaylorOperator)
            - "upwind": Flow-biased stencils (UpwindOperator)
            - "rbf" or "rbf-fd": Local RBF-FD (LocalRBFOperator)
        velocity_field: Velocity/drift at each point, shape (n_points, dimension).
            REQUIRED for "upwind" method. The velocity determines which neighbors
            are "upstream" and should receive higher weights.
        **kwargs: Additional arguments passed to operator constructor:
            - taylor_order: Order of Taylor expansion (1 or 2, default 2)
            - weight_function: "wendland", "gaussian", or "uniform"
            - kernel: RBF kernel for "rbf" method (default "phs3"):
                "phs1", "phs3", "phs5", "phs7", "gaussian", "wendland_c2",
                "wendland_c4", "multiquadric"
            - poly_degree: Polynomial degree for "rbf" method (default 2)
            - k_neighbors: Min neighbors (auto-computed if None)
            - neighborhood_mode: "radius", "knn", or "hybrid"

    Returns:
        DifferentialOperator instance

    Raises:
        ValueError: If method is unknown or required parameters are missing

    Example:
        # Standard GFDM
        op = create_operator(points, delta=0.1, method="direct")

        # Upwind for advection (requires velocity)
        velocity = compute_optimal_control(u)  # From HJB solve
        op = create_operator(points, delta=0.1, method="upwind",
                            velocity_field=velocity)

    Note:
        For HJBGFDMSolver integration, the velocity_field typically comes from
        the optimal control computation: v*(x) = -D_p H(x, Du(x)). The solver
        should recompute the upwind operator when the velocity field changes
        significantly during Newton iteration.
    """
    if method in ("direct", "taylor"):
        return TaylorOperator(points, delta=delta, **kwargs)

    elif method == "upwind":
        if velocity_field is None:
            raise ValueError(
                "UpwindOperator requires velocity_field parameter. "
                "This is the velocity/drift at each point, shape (n_points, dimension). "
                "For HJB problems, use v*(x) = -D_p H(x, Du(x))."
            )
        return UpwindOperator(points, velocity_field, delta=delta, **kwargs)

    elif method in ("rbf", "rbf-fd", "local_rbf"):
        # Extract RBF-specific kwargs
        kernel = kwargs.pop("kernel", "phs3")
        poly_degree = kwargs.pop("poly_degree", 2)
        # Handle deprecated phs_order
        if "phs_order" in kwargs:
            phs_order = kwargs.pop("phs_order")
            kernel = f"phs{phs_order}"
        return LocalRBFOperator(
            points,
            delta=delta,
            kernel=kernel,
            poly_degree=poly_degree,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown operator method: '{method}'. Valid options: 'direct', 'taylor', 'upwind', 'rbf', 'rbf-fd'."
        )


def create_bc_handler(
    method: str = "direct",
    domain_bounds: list[tuple[float, float]] | None = None,
) -> BoundaryHandler:
    """
    Factory function for creating boundary handlers.

    Args:
        method: BC handling method:
            - "direct": Row Replacement (DirectCollocationHandler) - RECOMMENDED
            - "ghost": Legacy ghost particles (GhostNodeHandler)
        domain_bounds: Domain bounds for ghost particles (only for "ghost" method)

    Returns:
        BoundaryHandler instance
    """
    if method == "direct":
        return DirectCollocationHandler()

    elif method == "ghost":
        return GhostNodeHandler(domain_bounds=domain_bounds)

    else:
        raise ValueError(f"Unknown BC method: {method}. Use 'direct' or 'ghost'.")


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for GFDM strategies."""
    print("Testing GFDM Strategies...")

    # Test 1: TaylorOperator
    print("\n[1] Testing TaylorOperator...")
    x_2d = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x_2d, x_2d)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    operator = TaylorOperator(points, delta=0.2, taylor_order=2)
    print(f"    Points: {operator.n_points}, Dimension: {operator.dimension}")

    # f(x,y) = x^2 + y^2 -> gradient = [2x, 2y], laplacian = 4
    u = points[:, 0] ** 2 + points[:, 1] ** 2
    grad = operator.gradient(u)
    lap = operator.laplacian(u)

    # Check interior points
    interior_mask = (points[:, 0] > 0.15) & (points[:, 0] < 0.85) & (points[:, 1] > 0.15) & (points[:, 1] < 0.85)

    grad_error = np.mean(np.abs(grad[interior_mask, 0] - 2 * points[interior_mask, 0]))
    lap_error = np.mean(np.abs(lap[interior_mask] - 4.0))
    print(f"    Gradient error: {grad_error:.2e}")
    print(f"    Laplacian error: {lap_error:.2e}")
    assert grad_error < 0.1, f"Gradient error too large: {grad_error}"
    assert lap_error < 0.5, f"Laplacian error too large: {lap_error}"

    # Test 2: get_derivative_weights
    print("\n[2] Testing get_derivative_weights...")
    weights = operator.get_derivative_weights(50)  # Middle point
    assert weights is not None, "Weights should not be None"
    assert "neighbor_indices" in weights, "Missing neighbor_indices"
    assert "grad_weights" in weights, "Missing grad_weights"
    assert "lap_weights" in weights, "Missing lap_weights"
    print(f"    Neighbors: {len(weights['neighbor_indices'])}")
    print(f"    Grad weights shape: {weights['grad_weights'].shape}")

    # Test 3: DirectCollocationHandler
    print("\n[3] Testing DirectCollocationHandler...")
    bc_handler = DirectCollocationHandler()

    # Create simple system
    n = operator.n_points
    A = np.eye(n)
    b = np.zeros(n)

    # Identify boundary points
    boundary_idx = []
    for i in range(n):
        x, y = points[i]
        if abs(x) < 1e-10 or abs(x - 1) < 1e-10 or abs(y) < 1e-10 or abs(y - 1) < 1e-10:
            boundary_idx.append(i)
    boundary_idx = np.array(boundary_idx)
    print(f"    Boundary points: {len(boundary_idx)}")

    # Compute normals (simple axis-aligned)
    normals = np.zeros((len(boundary_idx), 2))
    for local_idx, global_idx in enumerate(boundary_idx):
        x, y = points[global_idx]
        if abs(x) < 1e-10:
            normals[local_idx] = [-1, 0]
        elif abs(x - 1) < 1e-10:
            normals[local_idx] = [1, 0]
        elif abs(y) < 1e-10:
            normals[local_idx] = [0, -1]
        elif abs(y - 1) < 1e-10:
            normals[local_idx] = [0, 1]

    # Apply Neumann BC
    bc_config = {"type": "neumann", "values": 0.0, "normals": normals}
    bc_handler.apply_to_matrix(A, b, boundary_idx, operator, bc_config)

    # Check that boundary rows were modified
    modified_rows = np.sum(np.abs(A[boundary_idx] - np.eye(n)[boundary_idx]) > 1e-10)
    print(f"    Modified rows: {modified_rows}")
    assert modified_rows > 0, "BC handler should modify boundary rows"

    # Test 4: Factory functions
    print("\n[4] Testing factory functions...")
    op1 = create_operator(points, delta=0.2, method="direct")
    assert isinstance(op1, TaylorOperator), "direct should create TaylorOperator"

    handler1 = create_bc_handler(method="direct")
    assert isinstance(handler1, DirectCollocationHandler), "direct should create DirectCollocationHandler"

    handler2 = create_bc_handler(method="ghost")
    assert isinstance(handler2, GhostNodeHandler), "ghost should create GhostNodeHandler"

    print("\nAll GFDM strategy smoke tests passed!")
