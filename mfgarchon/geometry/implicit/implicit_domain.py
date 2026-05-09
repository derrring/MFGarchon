"""
Implicit Domain Infrastructure for Meshfree Methods

This module provides dimension-agnostic implicit domain representation via
signed distance functions (SDFs). This enables:

- No mesh generation required (O(d) storage instead of O(N^d))
- Natural obstacle representation via CSG operations
- Dimension-agnostic code (works for 2D, 3D, 4D, ..., 100D)
- Particle-friendly boundary conditions

Key concept: A domain D ⊂ ℝ^d is represented implicitly by a function:
    φ: ℝ^d → ℝ  where  φ(x) < 0 ⟺ x ∈ D

References:
- Osher & Fedkiw (2003): Level Set Methods and Dynamic Implicit Surfaces
- TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md Section 4
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mfgarchon.geometry.base import ImplicitGeometry
from mfgarchon.geometry.protocol import GeometryType
from mfgarchon.geometry.protocols import (
    SupportsBoundaryDistance,
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsLipschitz,
    SupportsManifold,
)
from mfgarchon.geometry.traits import BoundaryDef, ConnectivityType, StructureType
from mfgarchon.utils.deprecation import deprecated
from mfgarchon.utils.mfg_logging import get_logger

# Module logger
logger = get_logger(__name__)


class ImplicitDomain(
    ImplicitGeometry,
    # Boundary traits (Issue #590 Phase 1.2) - SDF-based geometries naturally support these
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsBoundaryDistance,
    # Topology traits (Issue #590 Phase 1.2)
    SupportsManifold,  # SDFs with C² continuity form smooth manifolds
    SupportsLipschitz,  # SDFs are typically 1-Lipschitz
):
    """
    Abstract base class for n-dimensional implicit domains.

    An implicit domain D ⊂ ℝ^d is defined by a signed distance function φ:
        x ∈ D  ⟺  φ(x) < 0   (interior)
        x ∈ ∂D ⟺  φ(x) = 0   (boundary)
        x ∉ D  ⟺  φ(x) > 0   (exterior)

    Advantages over explicit mesh representation:
    - Memory: O(d) vs O(N^d) for mesh
    - Obstacles: Free with CSG operations
    - Dimension-agnostic: Same code for any d
    - Particle methods: Natural boundary handling

    Subclasses must implement:
    - signed_distance(x): Core SDF
    - get_bounding_box(): For efficient sampling

    Example:
        >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))  # [0,1]²
        >>> domain.contains(np.array([0.5, 0.5]))  # True (interior)
        >>> domain.contains(np.array([1.5, 0.5]))  # False (exterior)
        >>> particles = domain.sample_uniform(1000)  # Sample particles
    """

    # GeometryProtocol implementation
    @property
    def geometry_type(self) -> GeometryType:
        """
        Type of geometry (IMPLICIT for all ImplicitDomain subclasses).

        All implicit domains (defined by signed distance functions) return
        GeometryType.IMPLICIT regardless of dimension or specific shape.
        """
        return GeometryType.IMPLICIT

    # --- Trait properties (Issue #732 Tier 1b) ---

    @property
    def connectivity_type(self) -> ConnectivityType:
        """Dynamic: neighbors via runtime spatial search (meshfree)."""
        return ConnectivityType.DYNAMIC

    @property
    def structure_type(self) -> StructureType:
        """Unstructured: scattered point cloud."""
        return StructureType.UNSTRUCTURED

    @property
    def boundary_def(self) -> BoundaryDef:
        """Implicit: boundary defined by SDF phi(x) = 0."""
        return BoundaryDef.IMPLICIT

    @property
    def num_spatial_points(self) -> int:
        """
        Number of discrete spatial points.

        For implicit domains, this is not well-defined as they are continuous
        representations. This method estimates the number of points by computing
        the volume and assuming a typical mesh spacing of 0.1.

        The result is **cached on first call** (Issue #1037): ``compute_volume``
        uses Monte-Carlo sampling without a fixed seed, so subsequent calls would
        return slightly different values. Cached returns guarantee the property
        is deterministic within one process — required by callers like
        ``MFGComponents._setup_custom_initial_density`` that pre-allocate based
        on the value and then iterate the spatial grid.

        Note: For actual discretization, use sample_uniform() or meshfree methods.
        """
        # Issue #1037: cache to avoid MC-volume non-determinism across calls
        if self._num_spatial_points_cached is not None:
            return self._num_spatial_points_cached

        # Estimate number of points based on volume
        # Assume typical mesh spacing h = 0.1
        try:
            volume = self.compute_volume(n_monte_carlo=10000)
            h = 0.1  # Typical mesh spacing
            n_points = int(volume / (h**self.dimension))
            result = max(n_points, 100)  # Minimum of 100 points
        except (ValueError, RuntimeError) as e:
            # Issue #547: Volume computation can fail for complex implicit domains
            logger.warning(
                "Volume computation failed for %dD implicit domain: %s. "
                "Using bounding box approximation (may overestimate point count).",
                self.dimension,
                e,
            )
            # Fallback: use bounding box
            bounds = self.get_bounding_box()
            bbox_volume = np.prod(bounds[:, 1] - bounds[:, 0])
            h = 0.1
            result = max(int(bbox_volume / (h**self.dimension)), 100)

        self._num_spatial_points_cached = result
        return result

    def get_spatial_grid(self) -> NDArray[np.float64]:
        """
        Get spatial grid representation (sampled interior points).

        Since implicit domains are continuous (defined by SDF), we return
        a uniform sample of interior points as the "grid".

        Returns:
            Array of shape (N, dimension) with N sampled interior points
            where N is determined by num_spatial_points.

        Note: This is a sampling-based approximation. For more control over
        point distribution, use sample_uniform() directly.
        """
        n_points = self.num_spatial_points
        return self.sample_uniform(n_points)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension of the domain."""

    @abstractmethod
    def signed_distance(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute signed distance function φ(x).

        Convention:
            φ(x) < 0  ⟺  x inside domain
            φ(x) = 0  ⟺  x on boundary
            φ(x) > 0  ⟺  x outside domain

        Args:
            x: Point(s) to evaluate - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar float or array of shape (N,)

        Notes:
            For exact SDFs, |φ(x)| = distance to boundary.
            For approximations, only sign matters for containment.
        """

    @abstractmethod
    def get_bounding_box(self) -> NDArray[np.float64]:
        """
        Get axis-aligned bounding box containing the domain.

        Returns:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]

        Example:
            >>> domain = Hypersphere(center=[0, 0], radius=1.0)
            >>> bounds = domain.get_bounding_box()
            >>> bounds  # array([[-1, 1], [-1, 1]])
        """

    # Note: subclasses that don't already maintain a `self.bounds` instance
    # attribute (Hyperrectangle does) should expose `.bounds` themselves —
    # see CSG composites (UnionDomain/IntersectionDomain/DifferenceDomain/
    # ComplementDomain) for the Issue #1041 fix that adds them.

    # Note: contains() and sample_uniform() now inherited from ImplicitGeometry

    def project_to_domain(
        self, x: NDArray[np.float64], method: Literal["simple", "gradient"] = "simple"
    ) -> NDArray[np.float64]:
        """
        Project point(s) outside domain back inside.

        Args:
            x: Point(s) - shape (d,) or (N, d)
            method: Projection method
                - "simple": Scale toward centroid of bounding box
                - "gradient": Use SDF gradient (requires smooth SDF)

        Returns:
            Projected point(s) - same shape as x

        Example:
            >>> domain = Hypersphere(center=[0, 0], radius=1.0)
            >>> outside = np.array([2.0, 0.0])
            >>> inside = domain.project_to_domain(outside)
            >>> assert domain.contains(inside)
        """
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape(1, -1)

        # Check which points need projection
        sd = self.signed_distance(x)
        if np.isscalar(sd):
            sd = np.array([sd])

        outside = sd > 0

        if not np.any(outside):
            return x[0] if is_single else x

        # Simple projection: scale toward bounding box center
        if method == "simple":
            bounds = self.get_bounding_box()
            center = bounds.mean(axis=1)

            x_projected = x.copy()
            for i in np.where(outside)[0]:
                # Move point toward center until inside
                direction = center - x[i]
                alpha = 0.0
                step = 0.1

                for _ in range(100):
                    candidate = x[i] + alpha * direction
                    if self.signed_distance(candidate) <= 0:
                        x_projected[i] = candidate
                        break
                    alpha += step
                else:
                    # Fallback: use center
                    x_projected[i] = center

            return x_projected[0] if is_single else x_projected

        else:
            raise ValueError(f"Unknown projection method: {method}")

    @deprecated(
        since="v0.12.0",
        replacement="Use MeshfreeApplicator from mfgarchon.geometry.boundary instead.",
    )
    def apply_boundary_conditions(
        self,
        particles: NDArray[np.float64],
        bc_type: Literal["reflecting", "absorbing", "periodic"] = "reflecting",
    ) -> NDArray[np.float64]:
        """
        Apply boundary conditions to particles that left the domain.

        .. deprecated:: 0.12.0
            Use :class:`MeshfreeApplicator` from ``mfgarchon.geometry.boundary`` instead.
            This provides a unified interface for all geometry types.

        Args:
            particles: Particle positions - shape (N, d)
            bc_type: Boundary condition type
                - "reflecting": Reflect particles back into domain
                - "absorbing": Remove particles outside domain
                - "periodic": Wrap particles to opposite side (box domains only)

        Returns:
            Updated particle positions (may have fewer particles for "absorbing")

        Example:
            >>> # Deprecated usage:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> particles_updated = domain.apply_boundary_conditions(particles, "reflecting")
            >>>
            >>> # New preferred usage:
            >>> from mfgarchon.geometry.boundary import MeshfreeApplicator
            >>> applicator = MeshfreeApplicator(domain)
            >>> particles_updated = applicator.apply_particle_bc(particles, "reflecting")
        """
        if bc_type == "reflecting":
            return self.project_to_domain(particles, method="simple")

        elif bc_type == "absorbing":
            inside = self.contains(particles)
            if np.isscalar(inside):
                inside = np.array([inside])
            return particles[inside]

        elif bc_type == "periodic":
            # Periodic BC requires special handling (only works for hyperrectangles)
            raise NotImplementedError("Periodic BC requires Hyperrectangle-specific implementation")

        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

    def compute_volume(self, n_monte_carlo: int = 100000) -> float:
        """
        Estimate domain volume via Monte Carlo integration.

        Args:
            n_monte_carlo: Number of samples for Monte Carlo

        Returns:
            Estimated volume

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> vol = sphere.compute_volume(n_monte_carlo=1000000)
            >>> assert np.abs(vol - np.pi) < 0.01  # π for unit circle in 2D
        """
        bounds = self.get_bounding_box()
        bbox_volume = np.prod(bounds[:, 1] - bounds[:, 0])

        # Sample uniformly from bounding box
        samples = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n_monte_carlo, self.dimension))

        # Fraction inside domain
        inside = self.contains(samples)
        if np.isscalar(inside):
            inside = np.array([inside])

        fraction_inside = np.mean(inside)

        return bbox_volume * fraction_inside

    def get_boundary_normal(self, x: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
        """
        Compute outward boundary normal at point(s) using SDF gradient.

        The boundary normal is computed as n = ∇φ / ||∇φ|| where φ is the
        signed distance function. For an exact SDF, ||∇φ|| = 1, but we
        normalize to handle approximate SDFs.

        Universal Outward Normal Convention (Issue #661):
            - n points FROM domain interior TO exterior
            - ∂u/∂n > 0 means u increases in the outward direction
            - For SDF φ with convention φ < 0 inside: n = ∇φ / |∇φ|

        Args:
            x: Point(s) to evaluate - shape (d,) or (N, d)
            eps: Finite difference step size for gradient computation

        Returns:
            Unit normal vector(s) pointing outward - same shape as x

        Notes:
            - The normal points outward (direction of increasing φ)
            - For points not exactly on the boundary, this gives the normal
              to the closest boundary point
            - Uses central finite differences for gradient: O(eps²) accuracy
            - Implementation delegates to outward_normal_from_sdf() for
              consistency (Issue #662 consolidation)

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> normal = sphere.get_boundary_normal(np.array([1.0, 0.0]))
            >>> np.allclose(normal, [1.0, 0.0])  # Points radially outward
            True
        """
        # Use canonical implementation from operators/differential (Issue #662)
        from mfgarchon.operators.differential.function_gradient import (
            outward_normal_from_sdf,
        )

        return outward_normal_from_sdf(self.signed_distance, x, eps=eps)

    def project_to_boundary(
        self,
        x: NDArray[np.float64],
        max_iterations: int = 20,
        tol: float = 1e-8,
    ) -> NDArray[np.float64]:
        """
        Project point(s) onto the domain boundary using Newton iteration.

        Uses the iteration: x_{k+1} = x_k - φ(x_k) * n(x_k)
        where φ is the SDF and n is the unit outward normal.

        This converges quickly for smooth boundaries and is exact in one
        step for points where the SDF gradient is constant along the
        projection direction (e.g., spheres, planes).

        Args:
            x: Point(s) to project - shape (d,) or (N, d)
            max_iterations: Maximum Newton iterations
            tol: Convergence tolerance for |φ(x)|

        Returns:
            Projected point(s) on boundary - same shape as x

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> inside = np.array([0.5, 0.0])
            >>> on_boundary = sphere.project_to_boundary(inside)
            >>> np.allclose(on_boundary, [1.0, 0.0])
            True
        """
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape(1, -1)

        x_proj = x.copy()
        n_points = x.shape[0]

        for i in range(n_points):
            xi = x_proj[i].copy()

            for _ in range(max_iterations):
                phi = self.signed_distance(xi)

                # Check convergence
                if np.abs(phi) < tol:
                    break

                # Get outward normal at current point
                normal = self.get_boundary_normal(xi)

                # Newton step: move along normal by -φ
                # (negative because we want to move toward boundary)
                xi = xi - phi * normal

            x_proj[i] = xi

        return x_proj[0] if is_single else x_proj

    def is_on_boundary(self, x: NDArray[np.float64], tol: float = 1e-8) -> bool | NDArray[np.bool_]:
        """
        Check if point(s) are on the domain boundary using SDF.

        A point is on the boundary if |φ(x)| < tol where φ is the SDF.

        Args:
            x: Point(s) to check - shape (d,) or (N, d)
            tol: Tolerance for boundary detection

        Returns:
            Boolean or array of booleans indicating if points are on boundary

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> sphere.is_on_boundary(np.array([1.0, 0.0]))
            True
            >>> sphere.is_on_boundary(np.array([0.5, 0.0]))
            False
        """
        sd = self.signed_distance(x)
        return np.abs(sd) < tol

    # =========================================================================
    # Trait Protocol Implementations (Issue #590 Phase 1.2)
    # =========================================================================

    def get_outward_normal(
        self,
        points: NDArray[np.float64],
        boundary_name: str | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute outward unit normal vectors at boundary points.

        Implements SupportsBoundaryNormal protocol.

        For SDF-based geometries, the outward normal is n = ∇φ / |∇φ| where
        φ is the signed distance function.

        Args:
            points: Points at which to evaluate normal, shape (num_points, dimension)
                    or (dimension,) for single point
            boundary_name: Ignored for implicit domains (SDF defines single boundary)

        Returns:
            Outward unit normals, shape (num_points, dimension) or (dimension,)

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> normal = sphere.get_outward_normal(np.array([1.0, 0.0]))
            >>> assert np.allclose(normal, [1.0, 0.0])  # Points radially outward

        Note:
            This method delegates to get_boundary_normal() which uses finite
            differences to compute the SDF gradient.
        """
        # Delegate to existing implementation
        return self.get_boundary_normal(points)

    def get_signed_distance(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64] | float:
        """
        Compute signed distance to boundary for given points.

        Implements SupportsBoundaryDistance protocol.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,) for single point

        Returns:
            Signed distances, shape (num_points,) or scalar for single point
                - Negative: Inside domain
                - Zero: On boundary
                - Positive: Outside domain

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> points = np.array([[0, 0], [1, 0], [2, 0]])  # Center, boundary, outside
            >>> phi = sphere.get_signed_distance(points)
            >>> assert phi[0] < 0  # Inside
            >>> assert np.isclose(phi[1], 0)  # On boundary
            >>> assert phi[2] > 0  # Outside

        Note:
            This method delegates to signed_distance() which is the core SDF
            implementation provided by subclasses.
        """
        # Delegate to existing implementation
        return self.signed_distance(points)

    def project_to_interior(
        self,
        points: NDArray[np.float64],
        tolerance: float = 1e-10,
    ) -> NDArray[np.float64]:
        """
        Project points from outside domain into interior.

        Implements SupportsBoundaryProjection protocol.

        This moves points outside the domain just inside the boundary by
        projecting to the boundary and then moving inward by tolerance along
        the inward normal.

        Args:
            points: Points to project, shape (num_points, dimension) or (dimension,)
            tolerance: Distance to move inside boundary (for numerical stability)

        Returns:
            Projected points in interior, same shape as input

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> outside = np.array([2.0, 0.0])  # Outside
            >>> inside = sphere.project_to_interior(outside, tolerance=0.01)
            >>> assert sphere.signed_distance(inside) < 0  # Now inside

        Note:
            For points already inside, returns them unchanged.
        """
        is_single = points.ndim == 1
        if is_single:
            points = points.reshape(1, -1)

        # Check which points are outside
        sd = self.signed_distance(points)
        if np.isscalar(sd):
            sd = np.array([sd])

        outside = sd > 0

        if not np.any(outside):
            # All points already inside
            return points[0] if is_single else points

        # Project to boundary, then move inside by tolerance
        points_interior = points.copy()

        # Project outside points to boundary
        boundary_points = self.project_to_boundary(points[outside])

        # Get inward normal (negative of outward normal)
        normals = self.get_outward_normal(boundary_points)

        # Move inward by tolerance
        points_interior[outside] = boundary_points - tolerance * normals

        return points_interior[0] if is_single else points_interior

    @property
    def manifold_dimension(self) -> int:
        """
        Intrinsic dimension of the domain manifold.

        Implements SupportsManifold protocol.

        For ImplicitDomain representing a full domain D subset R^d,
        manifold dimension equals ambient dimension d. The SDF boundary
        dD is (d-1)-dimensional, but the domain itself is d-dimensional.

        Note:
            Subclasses representing embedded surfaces (codimension-1)
            should override to return self.dimension - 1.
        """
        return self.dimension

    def get_metric_tensor(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Get metric tensor for the domain manifold.

        Implements SupportsManifold protocol.

        For ImplicitDomain in Euclidean space, the metric tensor is the
        standard Euclidean metric g_ij = δ_ij (identity matrix).

        Args:
            points: Points at which to evaluate metric, shape (num_points, dimension)
                    or (dimension,) for single point

        Returns:
            Metric tensor(s), shape (num_points, dimension, dimension) or (dimension, dimension)

        Example:
            >>> rect = Hyperrectangle([[0, 1], [0, 1]])
            >>> g = rect.get_metric_tensor(np.array([0.5, 0.5]))
            >>> assert np.allclose(g, np.eye(2))  # Euclidean metric

        Note:
            ImplicitDomains are embedded in Euclidean space, so the metric is
            always Euclidean (flat). Subclasses with Riemannian geometry should
            override this method.
        """
        is_single = points.ndim == 1
        if is_single:
            points = points.reshape(1, -1)

        n_points, d = points.shape

        # Euclidean metric: g_ij = δ_ij at all points
        g = np.tile(np.eye(d), (n_points, 1, 1))

        return g[0] if is_single else g

    def get_lipschitz_constant(
        self,
        function_type: str = "sdf",
    ) -> float:
        """
        Get Lipschitz constant for specified function.

        Implements SupportsLipschitz protocol.

        Args:
            function_type: Type of function to query
                - "sdf": Signed distance function (default)
                - "metric": Metric tensor components
                - "projection": Boundary projection map

        Returns:
            Lipschitz constant L where |f(x) - f(y)| ≤ L|x - y|

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> L = sphere.get_lipschitz_constant("sdf")
            >>> assert L == 1.0  # Exact SDFs are 1-Lipschitz

        Raises:
            ValueError: If function_type not recognized

        Note:
            - Exact SDFs satisfy |∇φ| = 1, so L_sdf = 1.0
            - Metric tensor is constant (Euclidean), so L_metric = 0.0
            - Projection map is generally 1-Lipschitz for convex domains
        """
        if function_type == "sdf":
            # Exact SDFs are 1-Lipschitz: |φ(x) - φ(y)| ≤ |x - y|
            return 1.0

        elif function_type == "metric":
            # Euclidean metric is constant, so derivative is zero
            return 0.0

        elif function_type == "projection":
            # Projection to convex set is 1-Lipschitz (non-expansive)
            # For non-convex sets, this is an upper bound
            return 1.0

        else:
            raise ValueError(f"Unknown function_type '{function_type}'. Valid options: 'sdf', 'metric', 'projection'")

    def get_tangent_space_basis(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute orthonormal basis for tangent space at given points.

        Implements SupportsManifold protocol.

        For ImplicitDomain in flat Euclidean space, the tangent space is R^d
        with canonical basis at every point.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Tangent basis vectors:
                - Single point: (dimension, dimension) canonical basis
                - Multiple points: (num_points, dimension, dimension)
        """
        # Canonical basis = metric tensor for flat space
        return self.get_metric_tensor(points)

    def compute_christoffel_symbols(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute Christoffel symbols for flat Euclidean space.

        Implements SupportsManifold protocol.

        For ImplicitDomain in flat Euclidean space, all Christoffel symbols
        vanish: Gamma^k_{ij} = 0. This reflects zero curvature.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Christoffel symbols (all zeros):
                - Single point: (dimension, dimension, dimension)
                - Multiple points: (num_points, dimension, dimension, dimension)
        """
        is_single = points.ndim == 1
        d = points.shape[-1] if points.ndim > 1 else len(points)

        if is_single:
            return np.zeros((d, d, d))

        n_points = points.shape[0]
        return np.zeros((n_points, d, d, d))

    def validate_lipschitz_regularity(
        self,
        tolerance: float = 1e-6,
    ) -> tuple[bool, str]:
        """
        Validate that boundary satisfies Lipschitz condition.

        Implements SupportsLipschitz protocol.

        For ImplicitDomain, regularity depends on the SDF. Exact SDFs
        (|nabla phi| = 1) are automatically 1-Lipschitz with smooth
        boundaries. Subclasses with non-smooth boundaries (corners, cusps)
        should override.

        Args:
            tolerance: Numerical tolerance for validation checks

        Returns:
            (is_valid, message): True with empty message for smooth SDFs.
        """
        L = self.get_lipschitz_constant("sdf")
        if not np.isfinite(L):
            return False, f"SDF Lipschitz constant is not finite: {L}"
        return True, ""

    # GeometryProtocol methods for solver interface
    def get_grid_shape(self) -> tuple[int]:
        """
        Get discretization shape for implicit domain.

        Returns:
            (N,) where N is an estimated number of points based on volume.

        Notes:
            Implicit domains are continuous - they don't have inherent grid structure.
            This method provides an estimate for compatibility with structured solvers.
            For actual discretization, use sample_uniform() or meshfree methods.
        """
        # Return estimated point count as (N,) tuple
        return (self.num_spatial_points,)

    def get_boundary_conditions(self):
        """
        Get boundary conditions for implicit domain.

        Returns:
            None - implicit domains don't have inherent boundary conditions.
            BCs should be specified by the problem or solver.

        Notes:
            Implicit domains support general BC via signed distance function.
            Specify BCs through problem.boundary_conditions or solver configuration.
        """
        return None

    def get_collocation_points(self) -> NDArray[np.float64]:
        """
        Get collocation points via uniform sampling of the domain.

        Returns:
            Array of shape (N, d) containing uniformly sampled points from domain.

        Notes:
            This samples the domain using the default number of points from
            num_spatial_points. For explicit control over sampling density,
            use sample_uniform(n_points) directly.

        Example:
            >>> rect = Hyperrectangle([[0, 1], [0, 1]])
            >>> points = rect.get_collocation_points()
            >>> points.shape  # (N, 2) where N ≈ volume / 0.1^2
        """
        return self.sample_uniform(self.num_spatial_points)

    # =========================================================================
    # Region Marking (SupportsRegionMarking protocol - Issue #590 Phase 1.3)
    # =========================================================================

    _regions_dict: dict | None = None  # Lazy-initialized by _regions property

    # Issue #1037: cache for num_spatial_points to prevent MC-volume non-determinism
    # across calls within one process. None until first computation.
    _num_spatial_points_cached: int | None = None

    @property
    def _regions(self) -> dict:
        """
        Lazy-initialized storage for named regions.

        For ImplicitDomain (meshfree), regions are stored as predicates/callables,
        not boolean masks (unlike TensorProductGrid which has discrete points).

        Returns:
            Dictionary mapping region names to predicates
        """
        if self._regions_dict is None:
            self._regions_dict = {}
        return self._regions_dict

    def mark_region(
        self,
        name: str,
        predicate: Callable[[NDArray], NDArray[np.bool_]] | None = None,
        sdf_region: Callable[[NDArray], NDArray[np.floating]] | None = None,
        **kwargs,
    ) -> None:
        """
        Mark a named spatial region for later reference.

        For implicit domains (meshfree), regions are specified via predicates
        or SDF functions and evaluated on-demand at query points.

        Args:
            name: Unique name for this region (e.g., "inlet", "obstacle")
            predicate: Function (N, d) → (N,) bool - True where point in region
            sdf_region: Alternative SDF-based specification (φ < 0 means inside region)
            **kwargs: Ignored (for compatibility with TensorProductGrid signature)

        Raises:
            ValueError: If name already exists
            ValueError: If neither predicate nor sdf_region provided

        Example:
            >>> circle = Hypersphere(center=[0.5, 0.5], radius=0.3)
            >>>
            >>> # Mark top half using predicate
            >>> circle.mark_region(
            ...     "top_half",
            ...     predicate=lambda x: x[:, 1] > 0.5
            ... )
            >>>
            >>> # Mark region using SDF
            >>> def sector_sdf(x):
            ...     on_boundary = np.abs(np.linalg.norm(x - [0.5, 0.5], axis=-1) - 0.3) < 0.01
            ...     in_sector = x[:, 1] > 0.65
            ...     return np.where(on_boundary & in_sector, -1.0, 1.0)
            >>> circle.mark_region("exit", sdf_region=sector_sdf)

        Note:
            Issue #549 / #590 Phase 1.3: Enables BC via region_name parameter
        """
        if name in self._regions:
            raise ValueError(f"Region '{name}' already exists. Available regions: {list(self._regions.keys())}")

        # Validate: exactly one specification method
        specified = sum(x is not None for x in [predicate, sdf_region])
        if specified == 0:
            raise ValueError("Must specify one of: predicate or sdf_region")
        if specified > 1:
            raise ValueError("Cannot specify both predicate and sdf_region")

        # Store predicate (or convert SDF to predicate)
        if predicate is not None:
            self._regions[name] = predicate
        elif sdf_region is not None:
            # Convert SDF to predicate: region is where SDF < 0
            self._regions[name] = lambda x: sdf_region(x) < 0

    def get_region_mask(
        self,
        name: str,
        points: NDArray[np.float64] | None = None,
    ) -> NDArray[np.bool_]:
        """
        Get boolean mask for named region.

        For implicit domains, evaluates the stored predicate at given points.

        Args:
            name: Region name (from mark_region call)
            points: Points to evaluate at - shape (N, d)
                    If None, uses get_collocation_points() (sampled discretization)

        Returns:
            Boolean mask of shape (N,) - True at points in region

        Raises:
            KeyError: If region name not found

        Example:
            >>> circle = Hypersphere(center=[0.5, 0.5], radius=0.3)
            >>> circle.mark_region("top", predicate=lambda x: x[:, 1] > 0.5)
            >>>
            >>> # Evaluate at specific points
            >>> pts = np.array([[0.5, 0.6], [0.5, 0.4]])
            >>> mask = circle.get_region_mask("top", points=pts)
            >>> # Returns: [True, False]

        Note:
            Unlike TensorProductGrid (which has fixed grid), implicit domains
            evaluate regions on-demand at provided points.
        """
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found. Available regions: {list(self._regions.keys())}")

        # Get points to evaluate at
        if points is None:
            points = self.get_collocation_points()

        # Ensure 2D array
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Evaluate predicate
        predicate = self._regions[name]
        return predicate(points)

    def intersect_regions(self, *names: str) -> Callable[[NDArray], NDArray[np.bool_]]:
        """
        Get intersection of multiple regions (boolean AND).

        For implicit domains, returns a combined predicate.

        Args:
            *names: Region names to intersect

        Returns:
            Combined predicate: callable (N, d) → (N,) bool

        Raises:
            KeyError: If any region name not found
            ValueError: If no region names provided

        Example:
            >>> circle.mark_region("top", predicate=lambda x: x[:, 1] > 0.5)
            >>> circle.mark_region("right", predicate=lambda x: x[:, 0] > 0.5)
            >>> combined = circle.intersect_regions("top", "right")
            >>> # combined(pts) returns True where BOTH top AND right
        """
        if not names:
            raise ValueError("Must provide at least one region name")

        predicates = [self._regions[name] for name in names]

        # Combine predicates with logical AND
        def combined_predicate(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            result = predicates[0](x)
            for pred in predicates[1:]:
                result &= pred(x)
            return result

        return combined_predicate

    def get_region_predicate(self, name: str) -> Callable[[NDArray], NDArray[np.bool_]]:
        """
        Get predicate function for named region (meshfree helper).

        This is a convenience method for meshfree solvers that work directly
        with predicates rather than boolean masks.

        Args:
            name: Region name

        Returns:
            Predicate function: (N, d) → (N,) bool

        Raises:
            KeyError: If region name not found

        Example:
            >>> circle.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.3)
            >>> inlet_pred = circle.get_region_predicate("inlet")
            >>> particles = np.array([[0.2, 0.5], [0.8, 0.5]])
            >>> in_inlet = inlet_pred(particles)  # [True, False]
        """
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found. Available regions: {list(self._regions.keys())}")
        return self._regions[name]

    def get_region_names(self) -> list[str]:
        """
        Get list of all marked region names.

        Returns:
            List of region names

        Example:
            >>> circle.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.3)
            >>> circle.mark_region("outlet", predicate=lambda x: x[:, 0] > 0.7)
            >>> circle.get_region_names()
            ['inlet', 'outlet']
        """
        return list(self._regions.keys())

    def __repr__(self) -> str:
        """String representation of the domain."""
        return f"{self.__class__.__name__}(dimension={self.dimension})"
