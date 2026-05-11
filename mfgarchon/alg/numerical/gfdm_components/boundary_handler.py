"""
Boundary Handling Component for GFDM Solvers.

This component provides boundary-related functionality for GFDM-based HJB solvers:
- Outward normal computation for boundary points
- Local Coordinate Rotation (LCR) for boundary stencil conditioning
- Ghost nodes method for structural Neumann BC enforcement
- Wind-dependent BC for viscosity solutions

Extracted from GFDMBoundaryMixin as part of Issue #545 (mixin refactoring).
Uses composition pattern instead of inheritance for better testability and reusability.

Issue #531: GFDM boundary stencil degeneracy fix.

Author: MFGarchon Development Team
Created: 2026-01-11
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.geometry.boundary.ghost import (
    compute_normal_from_bounds,
    create_reflection_ghost_points,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class BoundaryHandler:
    """
    Handles boundary operations for GFDM solvers (normals, LCR, ghost nodes).

    This component provides all boundary-related functionality needed for GFDM
    solvers, including:
    - Computation of outward normals at boundary points
    - Local Coordinate Rotation (LCR) for improved boundary stencil conditioning
    - Ghost nodes method for structural Neumann BC enforcement
    - Wind-dependent BC handling for Hamilton-Jacobi equations

    Composition Pattern (Issue #545):
        This component is injected into HJBGFDMSolver instead of using mixin inheritance.
        Benefits: Testable independently, reusable for other solvers, clear dependencies.

    Parameters
    ----------
    collocation_points : np.ndarray
        Collocation points, shape (n_points, dimension)
    dimension : int
        Spatial dimension
    domain_bounds : list[tuple[float, float]]
        Domain bounds per dimension: [(xmin, xmax), (ymin, ymax), ...]
    boundary_indices : np.ndarray
        Indices of boundary collocation points
    neighborhoods : dict
        Neighborhood structure for each point
    boundary_conditions : Any
        Boundary conditions object (BoundaryConditions or dict)
    use_ghost_nodes : bool
        Whether to use ghost nodes for Neumann BC
    use_wind_dependent_bc : bool
        Whether to use wind-dependent BC
    gfdm_operator : Any
        GFDM operator instance (for derivative weights)
    bc_property_getter : Callable[[str, Any], Any]
        Function to get BC properties from boundary_conditions
    gradient_computer : Callable[[np.ndarray, int], np.ndarray] | None
        Function to compute gradient at a point (for wind-dependent BC)

    Attributes
    ----------
    collocation_points : np.ndarray
        Collocation points
    dimension : int
        Spatial dimension
    domain_bounds : list[tuple[float, float]]
        Domain bounds
    boundary_indices : np.ndarray
        Boundary point indices
    neighborhoods : dict
        Neighborhood structure
    boundary_conditions : Any
        BC object
    use_ghost_nodes : bool
        Ghost nodes flag
    use_wind_dependent_bc : bool
        Wind-dependent BC flag
    _gfdm_operator : Any
        GFDM operator
    _bc_property_getter : Callable
        BC property getter function
    _gradient_computer : Callable | None
        Gradient computation function
    _boundary_normals : np.ndarray | None
        Cached boundary normals
    _boundary_rotations : dict[int, np.ndarray]
        Rotation matrices for LCR
    _ghost_node_map : dict[int, dict]
        Ghost node mappings

    Examples
    --------
    >>> # Create boundary handler for GFDM solver
    >>> handler = BoundaryHandler(
    ...     collocation_points=points,
    ...     dimension=2,
    ...     domain_bounds=[(0, 1), (0, 1)],
    ...     boundary_indices=boundary_idx,
    ...     neighborhoods=neighborhoods,
    ...     boundary_conditions=bc_obj,
    ...     use_ghost_nodes=True,
    ...     use_wind_dependent_bc=False,
    ...     gfdm_operator=operator,
    ...     bc_property_getter=lambda prop, default: bc_obj.get(prop, default),
    ...     gradient_computer=None,
    ... )
    >>>
    >>> # Compute boundary normals
    >>> normals = handler.compute_boundary_normals()
    >>>
    >>> # Apply ghost nodes
    >>> handler.apply_ghost_nodes_to_neighborhoods()
    """

    def __init__(
        self,
        collocation_points: np.ndarray,
        dimension: int,
        domain_bounds: list[tuple[float, float]],
        boundary_indices: np.ndarray,
        neighborhoods: dict,
        boundary_conditions: Any,
        use_ghost_nodes: bool,
        use_wind_dependent_bc: bool,
        gfdm_operator: Any,
        bc_property_getter: Callable[[str, Any], Any],
        gradient_computer: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ):
        """Initialize boundary handler."""
        self.collocation_points = np.asarray(collocation_points)
        self.dimension = dimension
        self.domain_bounds = domain_bounds
        self.boundary_indices = np.asarray(boundary_indices)
        self.neighborhoods = neighborhoods
        self.boundary_conditions = boundary_conditions
        self.use_ghost_nodes = use_ghost_nodes
        self.use_wind_dependent_bc = use_wind_dependent_bc
        self._gfdm_operator = gfdm_operator
        self._bc_property_getter = bc_property_getter
        self._gradient_computer = gradient_computer

        # Cached data
        self._boundary_normals: np.ndarray | None = None
        self._boundary_rotations: dict[int, np.ndarray] = {}
        self._ghost_node_map: dict[int, dict] = {}

        # Wind BC hyperviscosity parameter
        self._wind_bc_hyperviscosity: float = 0.0

    def compute_outward_normal(self, point_idx: int) -> np.ndarray:
        """
        Compute outward normal vector at a boundary point.

        For rectangular domains, the normal is determined by which boundary
        the point lies on. For corner points (on multiple boundaries), returns
        the average of all boundary normals.

        Delegates to geometry/boundary/ghost.compute_normal_from_bounds().

        Parameters
        ----------
        point_idx : int
            Index of boundary point

        Returns
        -------
        np.ndarray
            Unit outward normal vector, shape (dimension,)
        """
        point = self.collocation_points[point_idx]
        return compute_normal_from_bounds(point, self.domain_bounds)

    def compute_boundary_normals(self, tolerance: float = 1e-6) -> np.ndarray | None:
        """Compute outward normal vectors for all boundary points.

        Resolution order:

        1. **BoundaryConditions.identify_boundary_face** + **outward_normal_for_face**:
           on a rectangular (or hybrid) domain, classify the point to an
           axis-aligned face and emit the face-derived ±1 normal. This is
           the canonical path; aligns with the pre-classification used by
           HJBGFDMSolver and uses the same closed-inequality tolerance.
        2. **BoundaryConditions.get_outward_normal** (SDF-only fallback):
           for genuinely SDF-defined boundaries where no axis-aligned face
           matches. Uses SDF gradient.
        3. **compute_normal_from_bounds** (legacy, last resort): when no
           BoundaryConditions object is attached.

        Previously path 1 was missing and path 2 mis-fired on Difference-
        style domains where ``domain_sdf`` is the *obstacle* SDF rather
        than the outer box's, producing zero or wrong-direction normals
        for outer-wall points. Path 3 had ``tol=1e-10`` which missed
        boundary points placed at ε=1e-6 off-wall by collocation
        generators.

        Parameters
        ----------
        tolerance : float
            Closed-inequality tolerance for face classification (default
            1e-6, matching ``BoundaryConditions.identify_boundary_face``).

        Returns
        -------
        np.ndarray | None
            Array of shape (n_boundary, dimension) with unit normal vectors,
            or None if no boundary points.
        """
        if len(self.boundary_indices) == 0:
            return None

        normals = np.zeros((len(self.boundary_indices), self.dimension))
        bc_obj = self.boundary_conditions
        bounds = np.asarray(self.domain_bounds, dtype=float) if self.domain_bounds is not None else None

        for local_idx, global_idx in enumerate(self.boundary_indices):
            point = self.collocation_points[global_idx]

            # Path 1: classify to axis-aligned face, derive normal from face.
            if bc_obj is not None:
                try:
                    face = bc_obj.identify_boundary_face(point=point, tolerance=tolerance, domain_bounds=bounds)
                except (AttributeError, TypeError):
                    face = None
                if face is not None:
                    try:
                        normals[local_idx] = bc_obj.outward_normal_for_face(face, dimension=self.dimension)
                        continue
                    except AttributeError:
                        pass  # older BC objects without outward_normal_for_face

                # Path 2: SDF-only fallback.
                try:
                    normal = bc_obj.get_outward_normal(point)
                except AttributeError:
                    normal = None
                if normal is not None:
                    normals[local_idx] = normal
                    continue

            # Path 3: axis-aligned from bounds (no BC object).
            normals[local_idx] = self.compute_outward_normal(global_idx)

        return normals

    def create_bc_config(self) -> dict | None:
        """
        Create unified BC configuration dict (single source of truth).

        This ensures consistent BC handling between:
        - _apply_boundary_conditions_to_sparse_system (Jacobian)
        - _apply_boundary_conditions_to_solution (solution)
        - DirectCollocationHandler.apply_to_residual (residual)

        Returns
        -------
        dict | None
            BC configuration dict with keys: type, values, normals
            or None if BC not specified
        """
        # Resolve BC type from boundary_conditions
        bc_type = self._bc_property_getter("type", None)

        if bc_type is None:
            # Try to infer from BC object
            try:
                bc_type = self.boundary_conditions.default_bc.value.lower()
            except AttributeError:
                # No BC specified - return None
                return None

        if isinstance(bc_type, str):
            bc_type = bc_type.lower()

        # Get BC values
        bc_value = self._bc_property_getter("value", None)

        # Build values dict
        if callable(bc_value):
            bc_values = bc_value
        else:
            bc_values = bc_value

        return {
            "type": bc_type,
            "values": bc_values,
            "normals": self._boundary_normals,
        }

    def build_neumann_bc_weights(self) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
        """
        Build GFDM weights for normal derivative at boundary points.

        For Neumann BC (du/dn = 0), we need weights such that:
            du/dn approx sum_j w_j u_j = 0

        Returns
        -------
        dict[int, tuple[np.ndarray, np.ndarray, float]]
            Dictionary mapping boundary point index to (neighbor_indices, weights, center_weight)
        """
        bc_weights: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

        for i in self.boundary_indices:
            weights_data = self._gfdm_operator.get_derivative_weights(i)
            if weights_data is None:
                continue

            neighbor_indices = weights_data["neighbor_indices"]
            grad_weights = weights_data["grad_weights"]  # shape: (d, n_neighbors)

            # Get outward normal at this point
            normal = self.compute_outward_normal(i)

            # Normal derivative weights: du/dn = n . grad_u
            normal_weights = np.zeros(len(neighbor_indices))
            for k in range(len(neighbor_indices)):
                for d in range(self.dimension):
                    normal_weights[k] += normal[d] * grad_weights[d, k]

            # Center contribution
            center_weight = -np.sum(normal_weights[neighbor_indices >= 0])

            bc_weights[i] = (neighbor_indices, normal_weights, center_weight)

        return bc_weights

    def count_deep_neighbors(self, point_idx: int, neighbor_points: np.ndarray, min_depth: float) -> int:
        """
        Count neighbors with sufficient depth in the inward normal direction.

        For boundary points, "deep" neighbors are those that lie sufficiently
        into the interior (in the direction opposite to the outward normal).
        This ensures the stencil can capture normal derivatives accurately.

        Parameters
        ----------
        point_idx : int
            Index of the center point
        neighbor_points : np.ndarray
            Array of neighbor coordinates, shape (n_neighbors, dim)
        min_depth : float
            Minimum required depth (projection onto inward normal)

        Returns
        -------
        int
            Number of neighbors with depth >= min_depth
        """
        center = self.collocation_points[point_idx]
        outward_normal = self.compute_outward_normal(point_idx)

        # Inward normal (direction into the domain)
        inward_normal = -outward_normal

        # Compute depth: projection of (neighbor - center) onto inward normal
        offsets = neighbor_points - center  # shape: (n_neighbors, dim)
        depths = offsets @ inward_normal  # shape: (n_neighbors,)

        return int(np.sum(depths >= min_depth))

    # =========================================================================
    # Local Coordinate Rotation (LCR) Methods
    # =========================================================================

    def build_rotation_matrix(self, normal: np.ndarray) -> np.ndarray:
        """
        Build rotation matrix that aligns first axis with the given normal vector.

        For Local Coordinate Rotation (LCR) at boundary points, this rotation
        transforms neighbor offsets so that:
        - First axis (x') aligns with outward normal
        - Remaining axes (y', z', ...) are tangential to boundary

        Parameters
        ----------
        normal : np.ndarray
            Unit outward normal vector, shape (dimension,)

        Returns
        -------
        np.ndarray
            Rotation matrix R, shape (dimension, dimension).
            To transform: x_rotated = R @ x_original
        """
        dim = len(normal)

        if dim == 1:
            # 1D: trivial - just sign
            return np.array([[np.sign(normal[0]) if abs(normal[0]) > 1e-10 else 1.0]])

        elif dim == 2:
            # 2D: Rotation matrix that maps e_x to normal
            n_x, n_y = normal
            return np.array([[n_x, -n_y], [n_y, n_x]])

        elif dim == 3:
            # 3D: Use Rodrigues' rotation formula
            e_x = np.array([1.0, 0.0, 0.0])

            # Check if normal is already aligned with e_x
            dot = np.dot(e_x, normal)
            if abs(dot - 1.0) < 1e-10:
                return np.eye(3)
            if abs(dot + 1.0) < 1e-10:
                return np.diag([1.0, -1.0, -1.0])

            # General case: Rodrigues' formula
            v = np.cross(e_x, normal)
            s = np.linalg.norm(v)
            c = dot

            v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + v_skew + v_skew @ v_skew * (1 - c) / (s * s)
            return R

        else:
            # Higher dimensions: Use Householder reflection
            e_1 = np.zeros(dim)
            e_1[0] = 1.0

            if np.allclose(normal, e_1):
                return np.eye(dim)
            if np.allclose(normal, -e_1):
                R = np.eye(dim)
                R[0, 0] = -1.0
                return R

            # Householder: R = I - 2*v*v^T / (v^T*v)
            v = e_1 - normal
            v = v / np.linalg.norm(v)
            return np.eye(dim) - 2.0 * np.outer(v, v)

    def apply_local_coordinate_rotation(self) -> None:
        """
        Apply Local Coordinate Rotation (LCR) to boundary point stencils.

        For each boundary point, rotates the neighbor offsets to align with
        the boundary normal before computing Taylor expansion weights.
        This improves accuracy for normal derivatives at boundaries.

        Must be called after neighborhood structure is built and before
        derivative computations.

        Issue #531: GFDM boundary stencil degeneracy fix (part 2).
        """
        if len(self.boundary_indices) == 0:
            return

        boundary_set = set(self.boundary_indices)

        for i in boundary_set:
            normal = self.compute_outward_normal(i)

            # Skip if normal is zero
            if np.linalg.norm(normal) < 1e-10:
                continue

            # Build rotation matrix
            R = self.build_rotation_matrix(normal)
            self._boundary_rotations[i] = R

            # Get neighborhood
            neighborhood = self.neighborhoods[i]
            neighbor_points = neighborhood["points"]
            center = self.collocation_points[i]

            # Rotate neighbor offsets
            offsets = neighbor_points - center
            rotated_offsets = (R @ offsets.T).T  # shape: (n_neighbors, dim)

            # Store rotated offsets
            neighborhood["rotated_offsets"] = rotated_offsets
            neighborhood["rotation_matrix"] = R

    def rotate_derivatives_back(
        self, derivatives: dict[tuple[int, ...], float], R: np.ndarray
    ) -> dict[tuple[int, ...], float]:
        """
        Rotate derivatives from rotated frame back to original coordinates.

        For LCR boundary points, derivatives are computed in a rotated frame
        where the first axis aligns with the boundary normal. This method
        transforms them back to the original coordinate frame.

        Parameters
        ----------
        derivatives : dict[tuple[int, ...], float]
            Dictionary mapping multi-indices to derivative values
        R : np.ndarray
            Rotation matrix used for the forward transformation

        Returns
        -------
        dict[tuple[int, ...], float]
            Rotated derivatives dictionary

        Mathematical transformation:
        - First derivatives (gradient): grad_orig = R^T @ grad_rotated
        - Second derivatives (Hessian): H_orig = R^T @ H_rotated @ R
        """
        if self.dimension != 2:
            # Only 2D transformation is implemented for now
            return derivatives

        rotated = dict(derivatives)
        R_T = R.T

        # Extract first derivatives (gradient)
        u_xp = derivatives.get((1, 0), 0.0)
        u_yp = derivatives.get((0, 1), 0.0)

        # Transform gradient
        grad_rotated = np.array([u_xp, u_yp])
        grad_orig = R_T @ grad_rotated
        rotated[(1, 0)] = float(grad_orig[0])
        rotated[(0, 1)] = float(grad_orig[1])

        # Extract second derivatives (Hessian)
        u_xxp = derivatives.get((2, 0), 0.0)
        u_xyp = derivatives.get((1, 1), 0.0)
        u_yyp = derivatives.get((0, 2), 0.0)

        H_rotated = np.array([[u_xxp, u_xyp], [u_xyp, u_yyp]])

        # Transform Hessian
        H_orig = R_T @ H_rotated @ R
        rotated[(2, 0)] = float(H_orig[0, 0])
        rotated[(1, 1)] = float(H_orig[0, 1])
        rotated[(0, 2)] = float(H_orig[1, 1])

        return rotated

    # =========================================================================
    # Ghost Nodes Methods
    # =========================================================================

    def create_ghost_neighbors(
        self, point_idx: int, neighbor_points: np.ndarray, neighbor_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create ghost neighbors for a boundary point to enforce Neumann BC structurally.

        For Neumann boundary conditions (du/dn = 0), ghost nodes provide a structural
        enforcement by creating mirror-image neighbors outside the domain.

        Delegates position computation to geometry/boundary/ghost.create_reflection_ghost_points().

        Parameters
        ----------
        point_idx : int
            Index of boundary point
        neighbor_points : np.ndarray
            Array of neighbor coordinates (n_neighbors, dim)
        neighbor_indices : np.ndarray
            Array of neighbor point indices (n_neighbors,)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (ghost_points, ghost_indices, mirror_indices)

        Issue #531: Terminal BC compatibility via structural Neumann enforcement
        """
        center = self.collocation_points[point_idx]
        normal = self.compute_outward_normal(point_idx)

        # Select interior neighbors (on interior side of tangent plane)
        offsets = neighbor_points - center
        normal_components = offsets @ normal
        interior_mask = normal_components < -1e-10
        interior_points = neighbor_points[interior_mask]
        interior_indices = neighbor_indices[interior_mask]

        if len(interior_points) == 0:
            return np.zeros((0, self.dimension)), np.array([]), np.array([])

        # Create ghost points by reflection (delegated to geometry utility)
        ghost_points = create_reflection_ghost_points(center, interior_points, normal)

        # Assign negative indices to ghost points (GFDM-specific)
        n_ghosts = len(ghost_points)
        ghost_indices = -(point_idx * 10000 + np.arange(n_ghosts))

        return ghost_points, ghost_indices, interior_indices

    def apply_ghost_nodes_to_neighborhoods(self) -> None:
        """
        Apply ghost nodes method to boundary point neighborhoods.

        For each boundary point with Neumann BC, augments the neighborhood with
        ghost neighbors that enforce du/dn = 0 structurally through symmetry.

        Must be called after neighborhood structure is built and before
        Taylor matrix construction.

        Issue #531: Terminal BC compatibility fix.
        """
        if len(self.boundary_indices) == 0:
            return

        # Get BC type
        bc_type_val = self._bc_property_getter("type", None)

        if bc_type_val is not None:
            bc_type = bc_type_val.lower() if isinstance(bc_type_val, str) else str(bc_type_val).lower()
        else:
            # Try to infer from BoundaryConditions object
            try:
                from mfgarchon.geometry.boundary import BCType

                default_bc = self.boundary_conditions.default_bc
                bc_type = default_bc.value.lower() if isinstance(default_bc, BCType) else str(default_bc).lower()
            except AttributeError:
                return

        # Ghost nodes only apply to Neumann BC
        if bc_type not in ("neumann", "no_flux"):
            return

        boundary_set = set(self.boundary_indices)

        for i in boundary_set:
            # Get existing neighborhood
            neighborhood = self.neighborhoods[i]
            neighbor_points = neighborhood["points"]
            neighbor_indices = neighborhood["indices"]

            # Create ghost neighbors
            ghost_points, ghost_indices, mirror_indices = self.create_ghost_neighbors(
                i, neighbor_points, neighbor_indices
            )

            if len(ghost_points) == 0:
                continue

            # Augment neighborhood
            augmented_points = np.vstack([neighbor_points, ghost_points])
            augmented_indices = np.concatenate([neighbor_indices, ghost_indices])

            center = self.collocation_points[i]
            augmented_distances = np.linalg.norm(augmented_points - center, axis=1)

            # Update neighborhood
            neighborhood["points"] = augmented_points
            neighborhood["indices"] = augmented_indices
            neighborhood["distances"] = augmented_distances
            neighborhood["size"] = len(augmented_points)
            neighborhood["has_ghost"] = True
            neighborhood["ghost_count"] = len(ghost_points)

            # Store ghost node mapping
            ghost_to_mirror = {}
            for ghost_idx, mirror_idx in zip(ghost_indices, mirror_indices, strict=True):
                ghost_to_mirror[int(ghost_idx)] = int(mirror_idx)

            self._ghost_node_map[i] = {
                "ghost_indices": ghost_indices,
                "mirror_indices": mirror_indices,
                "ghost_to_mirror": ghost_to_mirror,
                "n_ghosts": len(ghost_points),
            }

    def get_values_with_ghosts(
        self, u_values: np.ndarray, neighbor_indices: np.ndarray, point_idx: int | None = None
    ) -> np.ndarray:
        """
        Get u values for neighbors, mapping ghost indices to mirror values.

        Ghost points (negative indices) get their values from corresponding mirror points
        (positive indices) to enforce du/dn = 0 symmetrically.

        Parameters
        ----------
        u_values : np.ndarray
            Solution vector at all collocation points
        neighbor_indices : np.ndarray
            Array of neighbor indices (may include negative ghost indices)
        point_idx : int | None
            Index of the point whose neighbors are being queried

        Returns
        -------
        np.ndarray
            Array of u values at neighbor locations
        """
        if not self.use_ghost_nodes or not self._ghost_node_map:
            # No ghost nodes, direct indexing
            return u_values[neighbor_indices]

        # Build global ghost mapping
        ghost_to_mirror_global: dict[int, int] = {}
        for ghost_info in self._ghost_node_map.values():
            ghost_to_mirror_global.update(ghost_info["ghost_to_mirror"])

        # Check wind-dependent BC
        use_wind_check = self.use_wind_dependent_bc and point_idx is not None and point_idx in self._ghost_node_map

        if use_wind_check and self._gradient_computer is not None:
            # Compute gradient at boundary point
            grad_u = self._gradient_computer(u_values, point_idx)
            normal = self.compute_outward_normal(point_idx)
            grad_dot_normal = np.dot(grad_u, normal)
            enforce_bc = grad_dot_normal < 0
        else:
            enforce_bc = True

        # Get values with ghost mapping
        u_neighbors = np.zeros(len(neighbor_indices))
        for i, idx in enumerate(neighbor_indices):
            if idx < 0:
                # Ghost index
                if enforce_bc:
                    # Use mirror
                    mirror_idx = ghost_to_mirror_global.get(int(idx))
                    if mirror_idx is not None:
                        u_neighbors[i] = u_values[mirror_idx]
                    else:
                        u_neighbors[i] = 0.0
                else:
                    # Linear extrapolation
                    mirror_idx = ghost_to_mirror_global.get(int(idx))
                    if mirror_idx is not None and point_idx is not None:
                        u_boundary = u_values[point_idx]
                        u_mirror = u_values[mirror_idx]

                        epsilon = self._wind_bc_hyperviscosity
                        if epsilon > 0:
                            u_neighbors[i] = (2.0 - epsilon) * u_boundary - (1.0 - epsilon) * u_mirror
                        else:
                            u_neighbors[i] = 2.0 * u_boundary - u_mirror
                    elif point_idx is not None:
                        u_neighbors[i] = u_values[point_idx]
                    else:
                        u_neighbors[i] = 0.0
            else:
                # Regular index
                u_neighbors[i] = u_values[int(idx)]

        return u_neighbors

    @property
    def boundary_normals(self) -> np.ndarray | None:
        """Get cached boundary normals (computed during initialization)."""
        return self._boundary_normals

    @boundary_normals.setter
    def boundary_normals(self, normals: np.ndarray | None) -> None:
        """Set boundary normals."""
        self._boundary_normals = normals

    @property
    def boundary_rotations(self) -> dict[int, np.ndarray]:
        """Get boundary rotation matrices (populated by LCR)."""
        return self._boundary_rotations

    @property
    def ghost_node_map(self) -> dict[int, dict]:
        """Get ghost node mapping (populated by ghost nodes)."""
        return self._ghost_node_map
