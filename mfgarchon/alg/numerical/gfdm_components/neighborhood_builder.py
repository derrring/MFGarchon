"""
Neighborhood Builder Component for GFDM Solvers.

This component provides stencil construction and Taylor expansion functionality:
- Neighborhood structure building with adaptive delta enlargement
- Reverse neighborhood mapping for sparse Jacobian
- Taylor matrix construction with SVD/QR decomposition
- Weight function computation (Wendland, Gaussian, cubic spline)
- Derivative weight extraction from Taylor coefficients

Extracted from GFDMStencilMixin as part of Issue #545 (mixin refactoring).
Uses composition pattern instead of inheritance for better testability and reusability.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.gfdm_components.boundary_handler import BoundaryHandler


class NeighborhoodBuilder:
    """
    Handles neighborhood and Taylor matrix construction for GFDM solvers.

    This component encapsulates:
    - Delta-neighborhood building with adaptive enlargement
    - Reverse neighborhood mapping (for sparse Jacobian)
    - Taylor expansion matrix construction (SVD/QR)
    - Weight function computation (smoothing kernels)
    - Derivative weight extraction

    Composition Pattern (Issue #545):
        This component is injected into HJBGFDMSolver instead of using mixin inheritance.
        Benefits: Testable independently, reusable for FDM/FEM solvers, clear dependencies.

    Parameters
    ----------
    collocation_points : np.ndarray
        Collocation points, shape (n_points, dimension)
    dimension : int
        Spatial dimension
    delta : float
        Neighborhood radius (base delta)
    taylor_order : int
        Order of Taylor expansion (1 or 2)
    weight_function : str
        Weight function: 'gaussian', 'wendland', 'cubic_spline', 'inverse_distance', 'uniform'
    weight_scale : float
        Smoothing length for Gaussian kernel
    k_min : int
        Minimum number of neighbors required
    adaptive_neighborhoods : bool
        Enable adaptive delta enlargement
    max_delta_multiplier : float
        Maximum delta multiplier for adaptive enlargement
    boundary_indices : np.ndarray
        Indices of boundary points
    n_derivatives : int
        Number of derivatives in Taylor expansion
    multi_indices : list[tuple[int, ...]]
        Multi-indices for Taylor expansion
    gfdm_operator : Any
        GFDMOperator instance for base neighborhoods
    use_local_coordinate_rotation : bool
        Whether LCR is enabled (affects Taylor matrix construction)
    boundary_handler : BoundaryHandler | None
        BoundaryHandler instance for deep neighbor counting (optional)

    Attributes
    ----------
    collocation_points : np.ndarray
        Collocation points (n_points, dimension)
    n_points : int
        Number of collocation points
    dimension : int
        Spatial dimension
    delta : float
        Base neighborhood radius
    taylor_order : int
        Taylor expansion order
    weight_function : str
        Weight function name
    weight_scale : float
        Smoothing length for weights
    k_min : int
        Minimum neighbors required
    adaptive_neighborhoods : bool
        Adaptive delta enabled flag
    max_delta_multiplier : float
        Max delta multiplier
    boundary_indices : np.ndarray
        Boundary point indices
    n_derivatives : int
        Number of derivatives
    multi_indices : list[tuple[int, ...]]
        Taylor multi-indices
    adaptive_stats : dict
        Statistics for adaptive delta enlargement
    neighborhoods : dict[int, dict]
        Neighborhood data for each point
    taylor_matrices : dict[int, dict]
        Taylor expansion matrices for each point
    _reverse_neighborhoods : dict[int, np.ndarray]
        Reverse neighborhood map for sparse Jacobian

    Examples
    --------
    >>> # Create neighborhood builder for 2D GFDM solver
    >>> collocation = np.random.uniform(0, 1, (500, 2))
    >>> boundary_indices = np.array([0, 1, 2, 3])  # Assume first 4 points are boundary
    >>> multi_indices = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]  # 2nd order
    >>>
    >>> builder = NeighborhoodBuilder(
    ...     collocation_points=collocation,
    ...     dimension=2,
    ...     delta=0.1,
    ...     taylor_order=2,
    ...     weight_function='wendland',
    ...     weight_scale=0.1,
    ...     k_min=10,
    ...     adaptive_neighborhoods=True,
    ...     max_delta_multiplier=3.0,
    ...     boundary_indices=boundary_indices,
    ...     n_derivatives=6,
    ...     multi_indices=multi_indices,
    ...     gfdm_operator=operator,
    ...     use_local_coordinate_rotation=False,
    ...     boundary_handler=None
    ... )
    >>>
    >>> # Build neighborhoods and Taylor matrices
    >>> builder.build_neighborhood_structure()
    >>> builder.build_reverse_neighborhoods()
    >>> builder.build_taylor_matrices()
    >>>
    >>> # Access neighborhood data
    >>> neighborhood = builder.neighborhoods[0]
    >>> print(f"Point 0 has {neighborhood['size']} neighbors")
    >>>
    >>> # Get affected rows for sparse Jacobian
    >>> affected = builder.get_affected_rows(j=10)
    """

    def __init__(
        self,
        collocation_points: np.ndarray,
        dimension: int,
        delta: float,
        taylor_order: int,
        weight_function: str,
        weight_scale: float,
        k_min: int,
        adaptive_neighborhoods: bool,
        max_delta_multiplier: float,
        boundary_indices: np.ndarray,
        n_derivatives: int,
        multi_indices: list[tuple[int, ...]],
        gfdm_operator: Any,
        use_local_coordinate_rotation: bool,
        boundary_handler: BoundaryHandler | None = None,
        obstacle_sdf: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        visibility_samples: int = 10,
        visibility_margin: float = 0.0,
    ):
        """Initialize neighborhood builder.

        Args:
            obstacle_sdf: Signed distance function of obstacle regions. When provided,
                neighbors whose line of sight to the center point passes through the
                obstacle (SDF < 0) are excluded. This prevents cross-wall stencils in
                narrow channels between obstacles.

                **Sign convention (Issue #1038)**: ``obstacle_sdf(x) < 0`` must mean
                "x is INSIDE the obstacle (to be filtered out of stencil
                line-of-sight)". This matches a single-obstacle ``Hypersphere`` /
                ``Hyperrectangle`` ``.signed_distance``, but does **NOT** match a
                ``DifferenceDomain(box, obstacle).signed_distance`` — the latter
                follows the standard navigable-region convention (sd<0 inside
                navigable, sd>0 outside), opposite of what is needed here.

                For a single-obstacle problem, pass the obstacle's own
                ``.signed_distance`` directly::

                    obstacle = Hypersphere(center=..., radius=...)
                    domain = DifferenceDomain(box, obstacle)
                    HJBGFDMSolver(...,
                        obstacle_sdf=obstacle.signed_distance,  # ✓ correct
                        # NOT domain.signed_distance — that has inverted convention
                    )

            visibility_samples: Number of interior samples along each segment for
                obstacle intersection testing. Default 10.
            visibility_margin: Safety margin for obstacle proximity. Neighbors with
                line-of-sight passing within this distance of an obstacle are excluded.
                Default 0.0.
        """
        self.collocation_points = np.asarray(collocation_points)
        self.n_points = len(collocation_points)
        self.dimension = dimension
        self.delta = delta
        self.taylor_order = taylor_order
        self.weight_function = weight_function
        self.weight_scale = weight_scale
        self.k_min = k_min
        self.adaptive_neighborhoods = adaptive_neighborhoods
        self.max_delta_multiplier = max_delta_multiplier
        self.boundary_indices = np.asarray(boundary_indices)
        self.n_derivatives = n_derivatives
        self.multi_indices = multi_indices
        self._gfdm_operator = gfdm_operator
        self._use_local_coordinate_rotation = use_local_coordinate_rotation
        self._boundary_handler = boundary_handler
        self._obstacle_sdf = obstacle_sdf
        self._visibility_samples = visibility_samples
        self._visibility_margin = visibility_margin

        # Initialize adaptive stats
        self.adaptive_stats: dict[str, Any] = {
            "n_adapted": 0,
            "max_delta_used": delta,
            "adaptive_enlargements": [],
        }

        # Data structures (populated by build methods)
        self.neighborhoods: dict[int, dict] = {}
        self.taylor_matrices: dict[int, dict] = {}
        self._reverse_neighborhoods: dict[int, np.ndarray] = {}

    def build_neighborhood_structure(self) -> None:
        """
        Build delta-neighborhood structure for all collocation points.

        Uses GFDMOperator's neighborhoods (without ghost particles - pure one-sided stencils)
        and only extends for points needing adaptive delta enlargement.

        For boundary points, also ensures sufficient "deep" neighbors exist
        (neighbors with adequate depth in the normal direction) to prevent
        degenerate "pancake" stencils that cannot capture normal derivatives.
        """
        self.neighborhoods = {}

        # For adaptive delta, we need pairwise distances
        if self.adaptive_neighborhoods:
            distances = cdist(self.collocation_points, self.collocation_points)
        else:
            distances = None

        # Parameters for deep neighbor check (Issue #531: GFDM boundary stencil degeneracy)
        # Minimum depth: fraction of delta that neighbors must extend into interior
        min_depth_fraction = 0.3  # Neighbors must be at least 0.3*delta into interior
        # Minimum number of deep neighbors required for boundary points
        k_deep = max(2, self.dimension)  # At least 2 (or dimension) deep neighbors

        # Pre-compute boundary set for fast lookup
        boundary_set = set(self.boundary_indices) if len(self.boundary_indices) > 0 else set()

        # Visibility filtering stats
        visibility_stats = {"n_filtered": 0, "total_removed": 0}

        for i in range(self.n_points):
            # Start with GFDMOperator's neighborhood (pure one-sided stencils, no ghost particles)
            base_neighborhood = self._gfdm_operator.get_neighborhood(i)

            # Visibility filtering: remove neighbors whose line of sight crosses an obstacle
            if self._obstacle_sdf is not None:
                base_neighborhood = self._apply_visibility_filter(i, base_neighborhood)
                if base_neighborhood.get("visibility_removed", 0) > 0:
                    visibility_stats["n_filtered"] += 1
                    visibility_stats["total_removed"] += base_neighborhood["visibility_removed"]

            n_neighbors = base_neighborhood["size"]

            # Check if this is a boundary point
            is_boundary = i in boundary_set

            # For boundary points, also check depth quality
            needs_depth_adaptation = False
            if is_boundary and self.adaptive_neighborhoods and distances is not None:
                min_depth = min_depth_fraction * self.delta
                # Use boundary_handler if available
                if self._boundary_handler is not None:
                    deep_count = self._boundary_handler.count_deep_neighbors(i, base_neighborhood["points"], min_depth)
                else:
                    # Fallback: count neighbors with sufficient depth manually
                    deep_count = self._count_deep_neighbors_fallback(i, base_neighborhood["points"], min_depth)
                needs_depth_adaptation = deep_count < k_deep

            # Check if adaptive delta enlargement is needed (count OR depth insufficient)
            needs_adaptation = (
                self.adaptive_neighborhoods
                and distances is not None
                and (n_neighbors < self.k_min or needs_depth_adaptation)
            )

            if needs_adaptation:
                # Adaptive delta enlargement for insufficient neighbors or depth
                neighbor_indices = base_neighborhood["indices"].copy()
                neighbor_points = base_neighborhood["points"].copy()
                neighbor_distances = base_neighborhood["distances"].copy()

                # Only count real neighbors (not ghost particles) for k_min check
                real_neighbor_count = np.sum(neighbor_indices >= 0)

                delta_current = self.delta
                delta_multiplier = 1.0
                was_adapted = False
                max_delta = self.delta * self.max_delta_multiplier

                # Track deep neighbor count for boundary points
                deep_count = 0
                if is_boundary:
                    min_depth = min_depth_fraction * delta_current
                    if self._boundary_handler is not None:
                        deep_count = self._boundary_handler.count_deep_neighbors(i, neighbor_points, min_depth)
                    else:
                        deep_count = self._count_deep_neighbors_fallback(i, neighbor_points, min_depth)

                # Continue expanding if count OR depth is insufficient
                count_ok = real_neighbor_count >= self.k_min
                depth_ok = (not is_boundary) or (deep_count >= k_deep)

                while (not count_ok or not depth_ok) and delta_current < max_delta:
                    # Enlarge delta by 20% increments
                    delta_multiplier *= 1.2
                    delta_current = self.delta * delta_multiplier

                    # Recompute neighborhood with enlarged delta
                    neighbor_mask = distances[i, :] < delta_current
                    neighbor_indices = np.where(neighbor_mask)[0]

                    # Apply visibility filter to expanded neighbors
                    if self._obstacle_sdf is not None and len(neighbor_indices) > 0:
                        from mfgarchon.geometry.visibility import filter_visible_neighbors

                        expanded_points = self.collocation_points[neighbor_indices]
                        vis_mask = filter_visible_neighbors(
                            self.collocation_points[i],
                            expanded_points,
                            self._obstacle_sdf,
                            n_samples=self._visibility_samples,
                            margin=self._visibility_margin,
                        )
                        neighbor_indices = neighbor_indices[vis_mask]

                    real_neighbor_count = len(neighbor_indices)
                    was_adapted = True

                    # Recompute deep neighbors for boundary points
                    if is_boundary:
                        neighbor_points_temp = self.collocation_points[neighbor_indices]
                        min_depth = min_depth_fraction * delta_current
                        if self._boundary_handler is not None:
                            deep_count = self._boundary_handler.count_deep_neighbors(i, neighbor_points_temp, min_depth)
                        else:
                            deep_count = self._count_deep_neighbors_fallback(i, neighbor_points_temp, min_depth)

                    # Update loop conditions
                    count_ok = real_neighbor_count >= self.k_min
                    depth_ok = (not is_boundary) or (deep_count >= k_deep)

                # Update neighborhood data for enlarged delta
                if was_adapted:
                    neighbor_points = self.collocation_points[neighbor_indices]
                    neighbor_distances = distances[i, neighbor_indices]

                    # Re-add ghost particles if this is a boundary point
                    if base_neighborhood.get("has_ghost", False):
                        ghost_particles, ghost_distances = self._gfdm_operator._create_ghost_particles(i)
                        if ghost_particles:
                            neighbor_points = np.vstack(
                                [neighbor_points] + [gp.reshape(1, -1) for gp in ghost_particles]
                            )
                            neighbor_distances = np.concatenate([neighbor_distances, np.array(ghost_distances)])
                            neighbor_indices = np.concatenate(
                                [neighbor_indices, np.array([-1 - j for j in range(len(ghost_particles))])]
                            )

                # Track maximum delta used
                if delta_current > self.adaptive_stats["max_delta_used"]:
                    self.adaptive_stats["max_delta_used"] = delta_current

                # Record adaptive enlargement
                if was_adapted:
                    self.adaptive_stats["n_adapted"] += 1
                    self.adaptive_stats["adaptive_enlargements"].append(
                        {
                            "point_idx": i,
                            "base_delta": self.delta,
                            "adapted_delta": delta_current,
                            "delta_multiplier": delta_multiplier,
                            "n_neighbors": len(neighbor_indices),
                            "is_boundary": is_boundary,
                            "deep_neighbors": deep_count if is_boundary else None,
                        }
                    )

                # Warn if still insufficient neighbors or depth
                import warnings as _warnings

                if real_neighbor_count < self.k_min:
                    _warnings.warn(
                        f"Point {i}: Could not find {self.k_min} neighbors even with "
                        f"delta={delta_current:.4f} ({delta_multiplier:.2f}x base). "
                        f"Only found {real_neighbor_count} neighbors. GFDM approximation may be poor.",
                        UserWarning,
                        stacklevel=3,
                    )
                elif is_boundary and deep_count < k_deep:
                    _warnings.warn(
                        f"Boundary point {i}: Only {deep_count}/{k_deep} deep neighbors "
                        f"(depth >= {min_depth_fraction}*delta) even with "
                        f"delta={delta_current:.4f} ({delta_multiplier:.2f}x base). "
                        f"Normal derivative approximation may be inaccurate.",
                        UserWarning,
                        stacklevel=3,
                    )

                # Store adapted neighborhood
                self.neighborhoods[i] = {
                    "indices": np.array(neighbor_indices) if isinstance(neighbor_indices, list) else neighbor_indices,
                    "points": np.array(neighbor_points) if isinstance(neighbor_points, list) else neighbor_points,
                    "distances": np.array(neighbor_distances)
                    if isinstance(neighbor_distances, list)
                    else neighbor_distances,
                    "size": len(neighbor_indices),
                    "has_ghost": base_neighborhood.get("has_ghost", False),
                    "ghost_count": base_neighborhood.get("ghost_count", 0),
                    "adapted": True,
                }
            else:
                # Use GFDMOperator's neighborhood directly (no adaptation needed)
                self.neighborhoods[i] = {
                    "indices": base_neighborhood["indices"],
                    "points": base_neighborhood["points"],
                    "distances": base_neighborhood["distances"],
                    "size": base_neighborhood["size"],
                    "has_ghost": base_neighborhood.get("has_ghost", False),
                    "ghost_count": base_neighborhood.get("ghost_count", 0),
                    "adapted": False,
                }

        # Report adaptive neighborhood statistics if enabled
        if self.adaptive_neighborhoods:
            n_adapted = self.adaptive_stats["n_adapted"]
            if n_adapted > 0:
                pct_adapted = 100.0 * n_adapted / self.n_points
                avg_multiplier = np.mean([e["delta_multiplier"] for e in self.adaptive_stats["adaptive_enlargements"]])
                max_multiplier = np.max([e["delta_multiplier"] for e in self.adaptive_stats["adaptive_enlargements"]])

                import warnings

                warnings.warn(
                    f"Adaptive neighborhoods: {n_adapted}/{self.n_points} points ({pct_adapted:.1f}%) "
                    f"required delta enlargement. Base delta: {self.delta:.4f}, "
                    f"Max delta used: {self.adaptive_stats['max_delta_used']:.4f} "
                    f"({max_multiplier:.2f}x base), Avg multiplier: {avg_multiplier:.2f}x. "
                    f"Consider increasing base delta for better theoretical accuracy.",
                    UserWarning,
                    stacklevel=2,
                )

        # Store visibility stats for external access
        self.visibility_stats = visibility_stats

        # Report visibility filtering statistics
        if self._obstacle_sdf is not None and visibility_stats["n_filtered"] > 0:
            pct = 100.0 * visibility_stats["n_filtered"] / self.n_points
            avg_removed = visibility_stats["total_removed"] / visibility_stats["n_filtered"]
            from mfgarchon.utils.mfg_logging import get_logger

            logger = get_logger(__name__)
            logger.info(
                "Visibility filtering: %d/%d points (%.1f%%) had neighbors removed. "
                "Total removed: %d, avg %.1f per affected point.",
                visibility_stats["n_filtered"],
                self.n_points,
                pct,
                visibility_stats["total_removed"],
                avg_removed,
            )

    def _apply_visibility_filter(self, point_idx: int, neighborhood: dict) -> dict:
        """Filter out neighbors whose line of sight to point_idx crosses an obstacle.

        Uses SDF sampling along line segments to detect obstacle crossings.
        Ghost neighbors (negative indices) are preserved unconditionally.

        Args:
            point_idx: Index of the center point.
            neighborhood: Base neighborhood dict from TaylorOperator.

        Returns:
            Filtered neighborhood dict with same structure. Adds 'visibility_removed'
            key tracking how many neighbors were removed.
        """
        from mfgarchon.geometry.visibility import filter_visible_neighbors

        indices = neighborhood["indices"]
        points = neighborhood["points"]
        distances = neighborhood["distances"]

        if len(indices) == 0:
            return {**neighborhood, "visibility_removed": 0}

        # Separate ghost neighbors (negative indices) from real ones
        ghost_mask = indices < 0
        real_mask = ~ghost_mask

        if not np.any(real_mask):
            return {**neighborhood, "visibility_removed": 0}

        center = self.collocation_points[point_idx]
        real_points = points[real_mask]

        # Check visibility for real neighbors only
        visible_mask = filter_visible_neighbors(
            center,
            real_points,
            self._obstacle_sdf,
            n_samples=self._visibility_samples,
            margin=self._visibility_margin,
        )

        n_removed = int(np.sum(~visible_mask))
        if n_removed == 0:
            return {**neighborhood, "visibility_removed": 0}

        # Reconstruct: keep visible real neighbors + all ghost neighbors
        keep_real = np.where(real_mask)[0][visible_mask]
        keep_ghost = np.where(ghost_mask)[0]
        keep = np.concatenate([keep_real, keep_ghost])
        keep.sort()  # Preserve original ordering

        return {
            "indices": indices[keep],
            "points": points[keep],
            "distances": distances[keep],
            "size": len(keep),
            "has_ghost": neighborhood.get("has_ghost", False),
            "ghost_count": neighborhood.get("ghost_count", 0),
            "visibility_removed": n_removed,
        }

    def build_reverse_neighborhoods(self) -> None:
        """
        Build reverse neighborhood map: for each point j, find all points i that have j in their neighborhood.

        This enables sparse Jacobian computation - when perturbing u[j], only residuals at
        points in reverse_neighborhoods[j] are affected.

        Complexity: O(n * k) where k is average neighborhood size.
        """
        self._reverse_neighborhoods = {j: [] for j in range(self.n_points)}

        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            neighbor_indices = neighborhood["indices"]

            # For each neighbor j of point i, add i to j's reverse neighborhood
            for j in neighbor_indices:
                if 0 <= j < self.n_points:  # Exclude ghost particles (negative indices)
                    self._reverse_neighborhoods[j].append(i)

        # Convert to arrays for faster access
        self._reverse_neighborhoods = {j: np.array(rows, dtype=int) for j, rows in self._reverse_neighborhoods.items()}

    def get_affected_rows(self, j: int) -> np.ndarray:
        """
        Get rows affected by perturbing u[j] for sparse Jacobian.

        Perturbing u[j] affects:
        1. Row j (the point itself - its residual depends on its own value)
        2. All rows i where j is in the neighborhood of i (j affects i's derivative approximation)

        Parameters
        ----------
        j : int
            Index of perturbed point

        Returns
        -------
        np.ndarray
            Array of row indices affected by u[j]
        """
        # Get points that have j in their neighborhood
        affected = self._reverse_neighborhoods.get(j, np.array([], dtype=int))

        # Always include j itself (residual at j depends on u[j])
        if j not in affected:
            affected = np.append(affected, j)

        return affected

    def build_taylor_matrices(self) -> None:
        """
        Pre-compute Taylor expansion matrices A for all collocation points.

        Uses GFDMOperator's Taylor matrices when the neighborhood wasn't adapted,
        only rebuilds for points that needed adaptive delta enlargement.
        """
        self.taylor_matrices = {}

        for i in range(self.n_points):
            neighborhood = self.neighborhoods[i]
            n_neighbors_raw = neighborhood["size"]
            n_neighbors = int(n_neighbors_raw) if isinstance(n_neighbors_raw, int | float) else 0

            if n_neighbors < self.n_derivatives:
                self.taylor_matrices[i] = None
                continue

            # Check if we can reuse GFDMOperator's Taylor matrices.
            # Requires: (1) neighborhood not adapted by delta enlargement,
            # (2) no LCR rotation, and (3) neighbor count matches operator's
            # original (may differ if visibility filtering removed neighbors).
            has_lcr_rotation = "rotated_offsets" in neighborhood and self._use_local_coordinate_rotation
            operator_neighborhood = self._gfdm_operator.get_neighborhood(i)
            size_matches = n_neighbors == operator_neighborhood["size"]
            if not neighborhood.get("adapted", False) and not has_lcr_rotation and size_matches:
                base_taylor = self._gfdm_operator.get_taylor_data(i)
                if base_taylor is not None:
                    # Compute condition number safely
                    S = base_taylor["S"]
                    if len(S) > 0 and S[-1] > 1e-15:
                        cond_num = S[0] / S[-1]
                    else:
                        cond_num = np.inf

                    # Wrap GFDMOperator's format to HJBGFDMSolver's expected format
                    self.taylor_matrices[i] = {
                        "A": base_taylor["A"],
                        "W": base_taylor["W"],
                        "sqrt_W": base_taylor["sqrt_W"],
                        "U": base_taylor["U"],
                        "S": S,
                        "Vt": base_taylor["Vt"],
                        "rank": base_taylor["rank"],
                        "condition_number": cond_num,
                        "use_svd": True,
                        "use_qr": False,
                    }
                    continue

            # Need to build Taylor matrices for adapted neighborhoods
            # Build Taylor expansion matrix A
            A = np.zeros((n_neighbors, self.n_derivatives))
            center_point = self.collocation_points[i]
            neighbor_points = neighborhood["points"]

            # For LCR-enabled boundary points, use rotated offsets for better conditioning
            # of normal derivative computation (Issue #531)
            use_rotated = "rotated_offsets" in neighborhood and self._use_local_coordinate_rotation
            rotated_offsets = neighborhood.get("rotated_offsets") if use_rotated else None

            for j, neighbor_point in enumerate(neighbor_points):
                if rotated_offsets is not None:
                    delta_x = rotated_offsets[j]
                else:
                    delta_x = neighbor_point - center_point

                for k, beta in enumerate(self.multi_indices):
                    # Compute (x - x_center)^beta / beta!
                    term = 1.0
                    factorial = 1.0

                    for dim in range(self.dimension):
                        if beta[dim] > 0:
                            term *= delta_x[dim] ** beta[dim]
                            factorial *= math.factorial(beta[dim])

                    A[j, k] = term / factorial

            # Compute weights and store matrices
            weights = self.compute_weights(np.asarray(neighborhood["distances"]))
            W = np.diag(weights)

            # Use SVD or QR decomposition to avoid condition number amplification
            sqrt_W = np.sqrt(W)
            WA = sqrt_W @ A

            # Try SVD first (most robust)
            try:
                # SVD decomposition: WA = U @ S @ V^T
                U, S, Vt = np.linalg.svd(WA, full_matrices=False)

                # Condition number check and regularization
                condition_number = S[0] / S[-1] if S[-1] > 1e-15 else np.inf

                # Truncate small singular values for regularization
                tolerance = 1e-12
                rank = np.sum(tolerance < S)

                self.taylor_matrices[i] = {
                    "A": A,
                    "W": W,
                    "sqrt_W": sqrt_W,
                    "U": U[:, :rank],
                    "S": S[:rank],
                    "Vt": Vt[:rank, :],
                    "rank": rank,
                    "condition_number": condition_number,
                    "use_svd": True,
                    "use_qr": False,
                }

            except np.linalg.LinAlgError:
                # Fallback to QR decomposition if SVD fails
                try:
                    # QR decomposition: WA = Q @ R
                    Q, R = np.linalg.qr(WA)
                    self.taylor_matrices[i] = {
                        "A": A,
                        "W": W,
                        "sqrt_W": sqrt_W,
                        "Q": Q,
                        "R": R,
                        "use_qr": True,
                        "use_svd": False,
                    }
                except np.linalg.LinAlgError:
                    # Final fallback to normal equations if QR also fails
                    self.taylor_matrices[i] = {
                        "A": A,
                        "W": W,
                        "AtW": A.T @ W,
                        "AtWA_inv": None,
                        "use_qr": False,
                        "use_svd": False,
                    }

                    try:
                        AtWA = A.T @ W @ A
                        if np.linalg.det(AtWA) > 1e-12:
                            self.taylor_matrices[i]["AtWA_inv"] = np.linalg.inv(AtWA)
                    except (np.linalg.LinAlgError, FloatingPointError):
                        # Cannot compute inverse - leave AtWA_inv as None
                        pass

    def compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute weights based on distance and weight function using smoothing kernels.

        Uses the unified kernel API from mfgarchon.utils.numerical.kernels.

        For QP optimization (monotonicity constraints), prefer:
        - 'cubic_spline': Compact support with good weight distribution
        - 'wendland': Compact support with C^4 smoothness
        Avoid 'gaussian' for QP - weights decay too fast, causing distant neighbors
        to have near-zero weight which can re-introduce singularity issues.

        Parameters
        ----------
        distances : np.ndarray
            Distances to neighbors

        Returns
        -------
        np.ndarray
            Weights for neighbors
        """
        from mfgarchon.utils.numerical.kernels import (
            CubicSplineKernel,
            GaussianKernel,
            WendlandKernel,
        )

        if self.weight_function == "gaussian":
            # Use GaussianKernel with smoothing length = weight_scale
            kernel = GaussianKernel()
            return kernel(distances, h=self.weight_scale)

        elif self.weight_function == "inverse_distance":
            # Keep legacy inverse distance weights (not a standard kernel)
            return 1.0 / (distances + 1e-12)

        elif self.weight_function == "uniform":
            # Keep legacy uniform weights (trivial case)
            return np.ones_like(distances)

        elif self.weight_function == "wendland":
            # Use Wendland C^4 kernel: (1 - r/h)_+^6 (35q^2 + 18q + 3)
            kernel = WendlandKernel(k=2)  # k=2 -> C^4 continuity
            # Support radius = delta (neighborhood size)
            return kernel(distances, h=self.delta)

        elif self.weight_function == "cubic_spline":
            # Cubic B-spline (M4) kernel - good for QP optimization
            # Compact support (2h) with smoother profile than Wendland
            # Better weight distribution for distant neighbors vs gaussian
            kernel = CubicSplineKernel(dimension=self.dimension)
            return kernel(distances, h=self.delta / 2)  # Support radius = 2h, so h = delta/2

        else:
            raise ValueError(
                f"Unknown weight function: {self.weight_function}. "
                f"Available options: 'gaussian', 'inverse_distance', 'uniform', 'wendland', 'cubic_spline'"
            )

    def compute_derivative_weights_from_taylor(
        self, point_idx: int, boundary_rotations: dict[int, np.ndarray] | None = None
    ) -> dict | None:
        """
        Compute derivative weights from Taylor matrices for a single point.

        Used for LCR boundary points where we need weights computed from
        LCR-rotated Taylor matrices rather than from GFDMOperator.

        Parameters
        ----------
        point_idx : int
            Index of collocation point
        boundary_rotations : dict[int, np.ndarray] | None
            Dictionary mapping point index to rotation matrix (for LCR)

        Returns
        -------
        dict | None
            Dictionary with neighbor_indices, grad_weights, lap_weights,
            or None if Taylor matrix is unavailable.
        """
        taylor_data = self.taylor_matrices.get(point_idx)
        if taylor_data is None:
            return None

        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]
        n_neighbors = len(neighbor_indices)

        # Get SVD components for pseudoinverse
        if not taylor_data.get("use_svd", False):
            return None

        sqrt_W = taylor_data["sqrt_W"]
        U = taylor_data["U"]
        S = taylor_data["S"]
        Vt = taylor_data["Vt"]

        # Compute pseudoinverse: (A^T W A)^{-1} A^T W = V S^{-1} U^T sqrt(W)
        # Each column of this matrix gives weights for one derivative coefficient
        # We need the weights that multiply (u_neighbor - u_center) to get derivatives

        # The derivative operator: coeffs = pinv(sqrt_W @ A) @ sqrt_W @ b
        # where b = u_neighbors - u_center
        # So weights_matrix = V @ diag(1/S) @ U^T @ sqrt_W
        S_inv = 1.0 / S
        weights_matrix = Vt.T @ np.diag(S_inv) @ U.T @ sqrt_W  # shape: (n_derivs, n_neighbors)

        # Extract gradient weights (first-order derivatives)
        grad_weights = np.zeros((self.dimension, n_neighbors))
        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 1:  # First-order derivative
                for d in range(self.dimension):
                    if beta[d] == 1:
                        grad_weights[d, :] = weights_matrix[k, :]

        # Extract Laplacian weights (sum of second-order pure derivatives)
        lap_weights = np.zeros(n_neighbors)
        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 2:  # Second-order derivative
                for d in range(self.dimension):
                    if beta[d] == 2:  # Pure second derivative d^2/dx_d^2
                        lap_weights += weights_matrix[k, :]

        # For LCR boundary points, rotate gradient weights back to original frame
        if self._use_local_coordinate_rotation and boundary_rotations is not None and point_idx in boundary_rotations:
            R = boundary_rotations[point_idx]
            # grad_weights_orig = R^T @ grad_weights_rotated
            grad_weights = R.T @ grad_weights

        return {
            "neighbor_indices": neighbor_indices,
            "grad_weights": grad_weights,
            "lap_weights": lap_weights,
        }

    def _count_deep_neighbors_fallback(self, point_idx: int, neighbor_points: np.ndarray, min_depth: float) -> int:
        """
        Fallback for counting deep neighbors when BoundaryHandler is not available.

        A "deep" neighbor is one that extends sufficiently into the interior
        (at least min_depth in the normal direction from the boundary point).

        This prevents degenerate "pancake" stencils that lie parallel to the boundary.

        Parameters
        ----------
        point_idx : int
            Index of boundary point
        neighbor_points : np.ndarray
            Neighbor collocation points, shape (n_neighbors, dimension)
        min_depth : float
            Minimum depth required (distance into interior)

        Returns
        -------
        int
            Number of deep neighbors
        """
        # Compute outward normal at this boundary point (simplified)
        center_point = self.collocation_points[point_idx]

        # Simple approximation: use direction to domain center as inward normal
        domain_center = np.mean(list(self.domain_bounds), axis=0)
        inward_normal = domain_center - center_point
        inward_normal_mag = np.linalg.norm(inward_normal)
        if inward_normal_mag > 1e-12:
            inward_normal /= inward_normal_mag
        else:
            # Fallback: cannot determine normal, assume all neighbors are deep
            return len(neighbor_points)

        # Count neighbors with sufficient depth
        deep_count = 0
        for neighbor in neighbor_points:
            # Project neighbor offset onto inward normal
            offset = neighbor - center_point
            depth = np.dot(offset, inward_normal)
            if depth >= min_depth:
                deep_count += 1

        return deep_count

    @property
    def domain_bounds(self) -> list[tuple[float, float]]:
        """Get domain bounds (for fallback normal computation)."""
        # Extract from collocation points
        bounds = []
        for d in range(self.dimension):
            x_min = np.min(self.collocation_points[:, d])
            x_max = np.max(self.collocation_points[:, d])
            bounds.append((x_min, x_max))
        return bounds
