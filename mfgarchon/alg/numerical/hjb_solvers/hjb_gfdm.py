from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import lstsq

# BC types for BoundaryCapable protocol implementation (Issue #527)
from scipy.optimize import approx_fprime

from mfgarchon.alg.numerical.gfdm_components import (
    BoundaryHandler,
    GridCollocationMapper,
    MonotonicityEnforcer,
    NeighborhoodBuilder,
    PrecomputedMonotoneStencils,
)

# GFDM infrastructure (Strategy Pattern)
from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import (
    DirectCollocationHandler,
    TaylorOperator,
    create_operator,
)
from mfgarchon.geometry.boundary.applicator_base import DiscretizationType
from mfgarchon.geometry.boundary.types import BCSegment, BCType, BoundaryFace
from mfgarchon.utils.deprecation import deprecated_parameter, deprecated_value
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.qp_utils import QPCache, QPSolver

from .base_hjb import (
    DEFAULT_NEWTON_MAX_ITERATIONS,
    DEFAULT_NEWTON_TOLERANCE,
    BaseHJBSolver,
)

logger = get_logger(__name__)

# Optional QP solver imports
CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None
OSQP_AVAILABLE = importlib.util.find_spec("osqp") is not None

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.config.mfg_methods import GFDMConfig
    from mfgarchon.core.derivatives import DerivativeTensors
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import BoundaryConditions


class HJBGFDMSolver(BaseHJBSolver):
    """
    Generalized Finite Difference Method (GFDM) solver for HJB equations using collocation.

    This solver implements meshfree collocation for HJB equations using:
    1. δ-neighborhood search for local support
    2. Taylor expansion with weighted least squares for derivative approximation
    3. Newton iteration for nonlinear HJB equations
    4. Support for various boundary conditions
    5. Optional QP constraints for monotonicity preservation

    QP Optimization Levels:
    - "none": GFDM without QP constraints (fastest, no monotonicity guarantee)
    - "auto": Adaptive QP with M-matrix checking (runtime QP when needed)
    - "always": Force QP at every point (slowest, for debugging)
    - "precompute": Precomputed monotone stencils (fast + monotone, recommended)

    Note: Monotonicity and QP constraint functionality is provided by MonotonicityEnforcer component.

    Implements BoundaryCapable protocol for unified BC handling (Issue #527).

    Collocation Point Strategies (Issue #529):
        Use FIXED collocation points throughout the MFG solve. Moving points
        during iteration causes convergence stall due to interpolation noise
        and stencil weight fluctuations.

        IMPORTANT: Fully Lagrangian MFG (moving collocation with the flow)
        is MATHEMATICALLY INVALID because the optimal control alpha* = -grad(u)
        requires grad(u) at FIXED spatial locations.

        See adaptive collocation analysis for detailed discussion
        of three collocation strategies and why only fixed collocation is valid.
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.GFDM

    # BoundaryCapable protocol: Supported BC types
    _SUPPORTED_BC_TYPES: frozenset = frozenset(
        {
            BCType.DIRICHLET,
            BCType.NEUMANN,
            BCType.NO_FLUX,  # Same as Neumann with g=0
        }
    )

    @property
    def supported_bc_types(self) -> frozenset:
        """BC types this solver supports (BoundaryCapable protocol)."""
        return self._SUPPORTED_BC_TYPES

    @property
    def discretization_type(self) -> DiscretizationType:
        """Discretization method (BoundaryCapable protocol)."""
        return DiscretizationType.GFDM

    # Explicitly initialize _neighborhood_builder to None (avoids hasattr)
    _neighborhood_builder: NeighborhoodBuilder | None = None

    @property
    def neighborhoods(self) -> dict:
        """Get neighborhoods from NeighborhoodBuilder or legacy mixin."""
        if self._neighborhood_builder is not None:
            return self._neighborhood_builder.neighborhoods
        # Legacy fallback: direct attribute access
        try:
            return self._neighborhoods
        except AttributeError:
            return {}

    @neighborhoods.setter
    def neighborhoods(self, value: dict) -> None:
        """Set neighborhoods in NeighborhoodBuilder or legacy storage."""
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.neighborhoods = value
        else:
            self._neighborhoods = value

    @property
    def taylor_matrices(self) -> dict:
        """Get Taylor matrices from NeighborhoodBuilder or legacy mixin."""
        if self._neighborhood_builder is not None:
            return self._neighborhood_builder.taylor_matrices
        # Legacy fallback: direct attribute access
        try:
            return self._taylor_matrices
        except AttributeError:
            return {}

    @taylor_matrices.setter
    def taylor_matrices(self, value: dict) -> None:
        """Set Taylor matrices in NeighborhoodBuilder or legacy storage."""
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.taylor_matrices = value
        else:
            self._taylor_matrices = value

    @property
    def adaptive_stats(self) -> dict:
        """Get adaptive neighborhood statistics from NeighborhoodBuilder or legacy mixin."""
        if self._neighborhood_builder is not None:
            return self._neighborhood_builder.adaptive_stats
        # Legacy fallback: direct attribute access
        try:
            return self._adaptive_stats
        except AttributeError:
            return {"n_adapted": 0, "adaptive_enlargements": [], "max_delta_used": 0.0}

    @adaptive_stats.setter
    def adaptive_stats(self, value: dict) -> None:
        """Set adaptive stats in NeighborhoodBuilder or legacy storage."""
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.adaptive_stats = value
        else:
            self._adaptive_stats = value

    @classmethod
    def from_config(
        cls,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        config: GFDMConfig,
        **extra: Any,
    ) -> HJBGFDMSolver:
        """Create solver from GFDMConfig object (Issue #634).

        Converts structured config into constructor kwargs. Additional keyword
        arguments in ``extra`` override config values.

        Args:
            problem: MFG problem instance
            collocation_points: (N_points, d) array of collocation points
            config: Structured GFDM configuration
            **extra: Additional kwargs passed to __init__ (override config)

        Returns:
            Configured HJBGFDMSolver instance
        """
        kwargs: dict[str, Any] = {
            "delta": config.delta,
            "taylor_order": config.taylor_order,
            "weight_function": config.weight_function,
            "weight_scale": config.weight_scale,
            "qp_optimization_level": config.qp.optimization_level,
            "qp_solver": config.qp.solver,
            "qp_warm_start": config.qp.warm_start,
            "qp_constraint_mode": config.qp.constraint_mode,
            "neighborhood_mode": config.neighborhood.mode,
            "k_neighbors": config.neighborhood.k_neighbors,
            "adaptive_neighborhoods": config.neighborhood.adaptive,
            "k_min": config.neighborhood.k_min,
            "max_delta_multiplier": config.neighborhood.max_delta_multiplier,
            "derivative_method": config.derivative.method,
            "rbf_kernel": config.derivative.rbf_kernel,
            "rbf_poly_degree": config.derivative.rbf_poly_degree,
            "use_local_coordinate_rotation": config.boundary_accuracy.local_coordinate_rotation,
            "use_ghost_nodes": config.boundary_accuracy.ghost_nodes,
            "use_wind_dependent_bc": config.boundary_accuracy.wind_dependent_bc,
            "congestion_mode": config.congestion_mode,
        }
        kwargs.update(extra)
        return cls(problem, collocation_points, **kwargs)

    @deprecated_value(
        param_name="qp_optimization_level",
        deprecated_values={"smart": "auto", "tuned": "auto", "basic": "auto"},
        since="v0.17.0",
    )
    @deprecated_parameter(
        param_name="qp_optimization_level",
        since="v0.18.0",
        replacement="monotonicity_scheme",
        removal_blockers=["internal_usage", "equivalence_test", "migration_docs"],
    )
    @deprecated_parameter(
        param_name="NiterNewton",
        since="v0.17.0",
        replacement="max_newton_iterations",
    )
    @deprecated_parameter(
        param_name="l2errBoundNewton",
        since="v0.17.0",
        replacement="newton_tolerance",
    )
    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        weight_scale: float = 1.0,
        max_newton_iterations: int | None = None,
        newton_tolerance: float | None = None,
        # Deprecated parameters for backward compatibility
        NiterNewton: int | None = None,
        l2errBoundNewton: float | None = None,
        boundary_indices: np.ndarray | None = None,
        boundary_conditions: dict | BoundaryConditions | None = None,
        # Monotonicity construction (renamed from qp_optimization_level v0.18.0; Issue #XXXX).
        # Two orthogonal axes:
        #   - monotonicity_scheme: WHICH constraint is enforced
        #   - monotonicity_application: WHEN to enforce it
        # See docstring for full semantics.
        monotonicity_scheme: str | None = None,
        monotonicity_application: str | None = None,
        # Deprecated alias bundling both axes — will be removed v0.25.0
        qp_optimization_level: str | None = None,
        qp_usage_target: float = 0.1,  # Unused, kept for backward compatibility
        qp_solver: str = "osqp",  # "osqp" or "scipy"
        qp_warm_start: bool = True,  # Enable QP warm-starting
        qp_constraint_mode: str = "indirect",  # "indirect" or "hamiltonian"
        # Adaptive neighborhood parameters
        adaptive_neighborhoods: bool = False,
        k_min: int | None = None,
        max_delta_multiplier: float = 5.0,
        # Hybrid neighborhood parameters
        k_neighbors: int | None = None,
        neighborhood_mode: str = "hybrid",
        # New GFDM infrastructure parameters
        derivative_method: str = "taylor",  # "taylor" or "rbf"
        rbf_kernel: str = "phs3",  # For RBF-FD: "phs3", "phs5", "gaussian"
        rbf_poly_degree: int = 2,  # Polynomial augmentation degree for RBF-FD
        use_new_infrastructure: bool = True,  # Use new Strategy Pattern (recommended)
        # Local Coordinate Rotation for boundary accuracy (Issue #531)
        use_local_coordinate_rotation: bool = False,
        # Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        use_ghost_nodes: bool = False,
        # Wind-Dependent BC for viscosity solution compatibility
        use_wind_dependent_bc: bool = False,
        # Congestion mode for Hamiltonian coupling
        congestion_mode: str = "additive",
        # Collocation geometry for periodic domains (Issue #711)
        collocation_geometry: object | None = None,
        # Obstacle-aware visibility filtering for stencil neighbors
        obstacle_sdf: object | None = None,
        visibility_samples: int = 10,
        visibility_margin: float = 0.0,
    ):
        """
        Initialize the GFDM HJB solver.

        Args:
            problem: MFG problem instance
            collocation_points: (N_points, d) array of collocation points
            delta: Neighborhood radius for collocation
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("wendland", "cubic_spline", "gaussian", "inverse_distance", "uniform")
            weight_scale: Scale parameter for weight function
            max_newton_iterations: Maximum Newton iterations (new parameter name)
            newton_tolerance: Newton convergence tolerance (new parameter name)
            NiterNewton: DEPRECATED - use max_newton_iterations
            l2errBoundNewton: DEPRECATED - use newton_tolerance
            boundary_indices: Indices of boundary collocation points
            boundary_conditions: Dictionary or BoundaryConditions object specifying boundary conditions
            use_monotone_constraints: DEPRECATED - Use qp_optimization_level instead.
                If explicitly set to True, will override qp_optimization_level.
            monotonicity_scheme: Which monotonicity construction to enforce on the GFDM
                Laplacian (and, for joint_socp, the per-edge cone on the gradient stencil):
                - "none": no constraints (fastest; no monotonicity guarantee).
                - "qp_m_matrix": classical M-matrix QP — projects unconstrained
                  Wendland-Taylor Laplacian weights onto $L_{ij} \\geq 0$ for $j \\neq i$.
                - "joint_socp": (Phase 1B follow-up) joint SOCP — M-matrix on $-\\Delta_h$
                  + per-edge cone $\\|D_{ij}\\|_2 \\leq C h_i L_{ij}$, closing the discrete
                  comparison principle (audit-major contribution).
                Default: "none".
                Renamed from `qp_optimization_level` in v0.18.0; legacy bundle still
                accepted as deprecated alias.
            monotonicity_application: When the chosen scheme is enforced (only
                meaningful for non-"none" schemes):
                - "adaptive": only at nodes where the unconstrained weights violate
                  the constraint (= legacy "auto"; recommended for qp_m_matrix).
                - "always": at every node, every solve.
                - "precompute": cache feasible weights at construction; reuse for all
                  Picard iterations / time steps. Recommended for joint_socp.
                Default (None): use scheme-recommended default — "adaptive" for
                qp_m_matrix, "precompute" for joint_socp.
            qp_optimization_level: DEPRECATED alias bundling (scheme + application). Will be
                removed in v0.25.0. Mappings: "none"→(none, —), "auto"→(qp_m_matrix, adaptive),
                "always"→(qp_m_matrix, always), "precompute"→(qp_m_matrix, precompute).
                Pass `monotonicity_scheme=` and `monotonicity_application=` instead.
            qp_usage_target: Deprecated parameter, kept for backward compatibility
            qp_solver: QP solver backend (default "osqp"):
                - "osqp": Use OSQP solver (fast convex QP, 5-10× faster than scipy)
                - "scipy": Use scipy.optimize.minimize (SLSQP or L-BFGS-B)
            qp_warm_start: Enable warm-starting for QP solves (default True).
                When True, uses previous QP solution as initial guess for next solve.
                Provides 2-3× additional speedup for OSQP on similar QP problems.
                Only applies to OSQP solver (scipy does not support efficient warm-starting).
            qp_constraint_mode: Type of monotonicity constraints (default "indirect"):
                - "indirect": Constraints on Taylor coefficients (simpler, approximate)
                - "hamiltonian": Direct Hamiltonian gradient constraints dH/du_j >= 0
                  (stricter, better monotonicity guarantees, requires gamma parameter)
            adaptive_neighborhoods: Enable adaptive delta enlargement to guarantee well-posed problems.
                When enabled, points with insufficient neighbors get locally enlarged delta.
                Maintains theoretical soundness while ensuring practical robustness.
                Recommended for irregular particle distributions.
            k_min: Minimum number of neighbors required per point (auto-computed from taylor_order if None).
                For Taylor order p in d dimensions, need C(d+p, p) - 1 derivatives.
            max_delta_multiplier: Maximum allowed delta enlargement factor (default 5.0, conservative).
                Limits delta growth to preserve GFDM locality. For very irregular distributions,
                consider increasing to 10.0 (achieves 98%+ success) or increasing base delta instead.
                Trade-off: Smaller limit = better theory, larger limit = better robustness.
            k_neighbors: Number of neighbors for neighborhood selection (auto-computed if None).
                When None, computed from Taylor order to ensure well-posed least squares.
            neighborhood_mode: Neighborhood selection strategy:
                - "radius": Use all points within delta (classic behavior)
                - "knn": Use exactly k nearest neighbors
                - "hybrid": Use delta, but ensure at least k neighbors (default, most robust)
            derivative_method: Method for computing spatial derivatives:
                - "taylor": Standard GFDM with Taylor polynomial basis (default)
                - "rbf": RBF-FD with polyharmonic splines (better conditioning)
            rbf_kernel: Kernel for RBF-FD method (only used when derivative_method="rbf"):
                - "phs3": r³ polyharmonic spline (most common)
                - "phs5": r⁵ polyharmonic spline (higher accuracy)
                - "gaussian": Gaussian RBF (requires shape parameter tuning)
            rbf_poly_degree: Polynomial augmentation degree for RBF-FD (default 2)
            use_new_infrastructure: Use new Strategy Pattern infrastructure (default True).
                When True, uses TaylorOperator/LocalRBFOperator + DirectCollocationHandler.
                When False, raises ValueError (legacy GFDMOperator removed in v0.17.15).
            use_local_coordinate_rotation: Enable Local Coordinate Rotation (LCR) for
                boundary stencils (default False, Issue #531). When True, rotates
                neighbor offsets at boundary points to align with the boundary normal,
                improving numerical conditioning for normal derivative computation.
                Recommended for domains with complex boundaries or when boundary
                stencils show poor conditioning. Only affects boundary points.
            use_ghost_nodes: Enable Ghost Nodes method for Neumann boundary conditions
                (default False, Issue #531 - Terminal BC compatibility). When True,
                creates mirrored "ghost" neighbors outside the domain for boundary points,
                enforcing ∂u/∂n = 0 structurally through symmetric stencils rather than
                via row replacement. This eliminates terminal cost/BC incompatibility issues
                in MFG problems. Recommended when terminal cost violates Neumann BC
                (e.g., g(x) = ||x - x_exit||² with Neumann BC at walls). Mutually exclusive
                with use_local_coordinate_rotation (ghost nodes take precedence).
            use_wind_dependent_bc: Enable wind-dependent boundary conditions (default False).
                When True (requires use_ghost_nodes=True), ghost nodes are only enforced
                when characteristics flow INTO the boundary (∇u·n > 0). When flow is OUT
                (∇u·n < 0), uses extrapolation instead. This implements the viscosity solution
                approach where BCs are weak constraints, only enforced when the PDE solution
                "wants" to violate them. Recommended for evacuation/exit problems where agents
                need to cross boundaries. Based on Lions & Souganidis theory of discontinuous
                viscosity solutions.
            congestion_mode: Mode for density-velocity coupling (default "additive"):
                - "additive": H = |p|²/(2λ) + γm (standard separable form)
                - "multiplicative": H = (1 + γ|Ω|m)|p|²/(2λ) (velocity reduction by congestion)
                The multiplicative form models agents slowing down in crowded areas, where
                γ|Ω|m ≈ γ × (local_density / average_density). This makes γ dimensionless
                and O(1) for observable effects, unlike additive form where γ ~ 1/|Ω|.
            collocation_geometry: Geometry object for collocation domain (Issue #711).
                If provided and implements SupportsPeriodic (e.g., Hyperrectangle with
                periodic_dims), enables periodic neighbor search for GFDM on torus domains.
                Example: Hyperrectangle(bounds, periodic_dims=(0, 1)) for 2D torus.
            obstacle_sdf: Optional callable ``f(x) -> float`` for visibility-based
                stencil filtering. Convention: ``obstacle_sdf(x) < 0`` means x is INSIDE
                the obstacle. Pass the obstacle's own ``.signed_distance`` directly
                (e.g., ``obstacle_sdf=Hypersphere(...).signed_distance``); do NOT pass
                a ``DifferenceDomain.signed_distance``, which has the opposite convention
                (sd<0 inside the navigable region). See Issue #1038 and the full
                docstring on ``NeighborhoodBuilder.obstacle_sdf``.
            visibility_samples: Number of interior samples along each stencil edge for
                obstacle intersection testing (default 10). Used only when
                ``obstacle_sdf`` is provided.
            visibility_margin: Safety margin for obstacle proximity (default 0.0).
                Stencil edges passing within this distance of an obstacle are filtered.
        """
        super().__init__(problem)

        # --- Resolve (scheme, application) from new API or legacy alias (v0.18.0) ---
        #
        # Two orthogonal axes:
        #   monotonicity_scheme:        WHICH constraint is enforced
        #     "none" | "qp_m_matrix" | "joint_socp"
        #   monotonicity_application:   WHEN it is enforced
        #     "adaptive" | "always" | "precompute"
        #
        # Application defaults per scheme (when application=None):
        #   qp_m_matrix → "adaptive"     (= legacy "auto", recommended runtime check)
        #   joint_socp  → "precompute"   (audit-major default; weights cached at construction)
        #   none        → ignored
        #
        # Legacy `qp_optimization_level` bundles both axes via these mappings:
        #   "none"       → (none, ignored)
        #   "auto"       → (qp_m_matrix, adaptive)
        #   "always"     → (qp_m_matrix, always)
        #   "precompute" → (qp_m_matrix, precompute)
        #
        # Mutual exclusion: pass either the new (scheme + application) or the legacy alias,
        # not both.
        if (
            monotonicity_scheme is not None or monotonicity_application is not None
        ) and qp_optimization_level is not None:
            raise ValueError(
                "Specify at most one of: (monotonicity_scheme=, monotonicity_application=) "
                "or qp_optimization_level=. The latter is the deprecated alias (v0.18.0)."
            )

        # Issue #1034: warn when user defaults to "none" (bare Wendland-Taylor LSQ).
        # This default produces a method whose M-matrix structure is not enforced;
        # boundary stencils can produce oscillatory derivatives that destabilize
        # FP-Particle coupling on long-time-horizon problems (e.g., 1D ToB at T=8
        # with KL=0.098 and 11 spurious modes — see Issue #1034 for full evidence).
        # Validated in mfg-research/.../exp08_towel_2d_validation/_preflight_1d/
        # post_mortem_1d_tob_debug.md.
        if monotonicity_scheme is None and monotonicity_application is None and qp_optimization_level is None:
            import warnings as _w

            _w.warn(
                "HJBGFDMSolver: no `monotonicity_scheme` specified; defaulting to "
                "'none' (no QP correction). This produces bare Wendland-Taylor LSQ "
                "stencils whose M-matrix structure is not enforced — boundary "
                "stencils can produce oscillatory derivatives that destabilize "
                "FP-Particle coupling on long-time-horizon problems. For "
                "paper-canonical monotone behavior, pass "
                "`monotonicity_scheme='joint_socp'` (M-matrix + per-edge cone, "
                "discrete comparison principle) or "
                "`monotonicity_scheme='qp_m_matrix'` (M-matrix only, cheaper). "
                "See Issue #1034. Pass `monotonicity_scheme='none'` explicitly "
                "to suppress this warning if the bare scheme is intentional.",
                UserWarning,
                stacklevel=2,
            )

        if monotonicity_scheme is not None or monotonicity_application is not None:
            # New API path
            scheme = monotonicity_scheme if monotonicity_scheme is not None else "none"
            valid_schemes = ("none", "qp_m_matrix", "joint_socp")
            if scheme not in valid_schemes:
                raise ValueError(f"monotonicity_scheme must be one of {valid_schemes}; got '{scheme}'.")
            valid_apps = ("adaptive", "always", "precompute", None)
            if monotonicity_application not in valid_apps:
                raise ValueError(
                    f"monotonicity_application must be one of {valid_apps}; got '{monotonicity_application}'."
                )
            # Resolve application via scheme-default if unspecified
            if monotonicity_application is None:
                application = {
                    "none": "ignored",
                    "qp_m_matrix": "adaptive",
                    "joint_socp": "precompute",
                }[scheme]
            else:
                application = monotonicity_application
        else:
            # Legacy path
            legacy = qp_optimization_level if qp_optimization_level is not None else "none"
            mapping = {
                "none": ("none", "ignored"),
                "auto": ("qp_m_matrix", "adaptive"),
                "always": ("qp_m_matrix", "always"),
                "precompute": ("qp_m_matrix", "precompute"),
            }
            # Unknown legacy value passes through for solver-internal handling.
            scheme, application = mapping.get(legacy, ("qp_m_matrix", legacy))

        # Canonical storage
        self.monotonicity_scheme = scheme
        self.monotonicity_application = application

        # Reconstruct legacy `qp_optimization_level` for backward-compat internal branches
        # (lines using self.qp_optimization_level == "auto"/"always"/"precompute"/"none"):
        if scheme == "none":
            self.qp_optimization_level = "none"
        elif scheme == "qp_m_matrix":
            self.qp_optimization_level = application  # adaptive→"auto"-like, etc.
            # Note: "adaptive" maps to legacy "auto" semantically, but the legacy code
            # branches check string == "auto", so we need to translate:
            if application == "adaptive":
                self.qp_optimization_level = "auto"
        elif scheme == "joint_socp":
            # joint_socp precomputes weights at __init__ (per-edge cone + M-matrix
            # via SOCP); semantically this IS a precompute application. Setting
            # legacy `qp_optimization_level = "precompute"` selects the per-point
            # HJB Newton path (line ~2425), matching the qp_m_matrix+precompute
            # path. Setting it to "none" instead would route through the batch
            # Hamiltonian path which evaluates H(x,m,p,t) differently and breaks
            # numerical equivalence with the legacy `precompute_socp_weights +
            # patch_operator` workflow used in research code.
            self.qp_optimization_level = "precompute"
            if application not in ("precompute", "ignored"):
                warnings.warn(
                    f"monotonicity_scheme='joint_socp' currently supports only "
                    f"application='precompute'; got '{application}'. Falling back to "
                    f"'precompute'. Adaptive/always strategies are tracked for a "
                    f"follow-up PR.",
                    stacklevel=2,
                )

        # Method name
        if scheme == "none":
            self.hjb_method_name = "GFDM"
        elif scheme == "qp_m_matrix":
            self.hjb_method_name = {
                "adaptive": "GFDM-QP",
                "always": "GFDM-QP-Always",
                "precompute": "GFDM-Precompute",
            }.get(application, f"GFDM-{application}")
        elif scheme == "joint_socp":
            self.hjb_method_name = f"GFDM-JointSOCP-{application}"
        else:
            self.hjb_method_name = f"GFDM-{self.qp_optimization_level}"

        # Handle backward compatibility (warnings issued by @deprecated_parameter decorators)
        if NiterNewton is not None:
            if max_newton_iterations is None:
                max_newton_iterations = NiterNewton

        if l2errBoundNewton is not None:
            if newton_tolerance is None:
                newton_tolerance = l2errBoundNewton

        # Set defaults if still None
        if max_newton_iterations is None:
            max_newton_iterations = DEFAULT_NEWTON_MAX_ITERATIONS
        if newton_tolerance is None:
            newton_tolerance = DEFAULT_NEWTON_TOLERANCE

        # Collocation parameters
        self.collocation_points = collocation_points
        self.n_points = collocation_points.shape[0]
        self.dimension = collocation_points.shape[1]
        self.delta = delta
        self.taylor_order = taylor_order
        self.weight_function = weight_function
        self.weight_scale = weight_scale

        # Newton parameters (store with new names)
        self.max_newton_iterations = max_newton_iterations
        self.newton_tolerance = newton_tolerance

        # Keep old names for backward compatibility (without warnings when accessed)
        self.NiterNewton = max_newton_iterations
        self.l2errBoundNewton = newton_tolerance

        # Boundary condition parameters
        # Auto-detect boundary indices if not provided (Issue #542 fix)
        if boundary_indices is not None:
            self.boundary_indices = boundary_indices
        else:
            # Try to detect boundary points from domain bounds
            self.boundary_indices = self._detect_boundary_indices(collocation_points)
        # Get BC from parameter, or from problem geometry (Issue #542 fix, Issue #527 centralized BC)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            # Use centralized BC resolution from BaseMFGSolver (Issue #527)
            # Checks: cached _boundary_conditions, geometry.boundary_conditions,
            # geometry.get_boundary_conditions(), problem.boundary_conditions,
            # problem.get_boundary_conditions()
            self.boundary_conditions = self.get_boundary_conditions()
        self.interior_indices = np.setdiff1d(np.arange(self.n_points), self.boundary_indices)

        # Monotonicity scheme (single source of truth) — already set above (v0.18.0 rename)
        # self.monotonicity_scheme and self.qp_optimization_level both = resolved monotonicity_scheme

        # QP usage target (deprecated, kept for backward compatibility)
        self.qp_usage_target = qp_usage_target

        # QP solver selection
        self.qp_solver = qp_solver
        self.qp_warm_start = qp_warm_start
        self.qp_constraint_mode = qp_constraint_mode

        # Congestion mode for Hamiltonian coupling
        self.congestion_mode = congestion_mode

        # Collocation geometry for periodic domains (Issue #711)
        self._collocation_geometry = collocation_geometry

        # Initialize QP components (will be fully initialized after neighborhoods are built)
        # Map qp_solver parameter to QPSolver backend
        qp_backend = "auto" if qp_solver == "osqp" else "scipy-slsqp"
        self._qp_cache = QPCache(max_size=1000)
        self._qp_solver_instance = QPSolver(
            backend=qp_backend,
            enable_warm_start=qp_warm_start,
            cache=self._qp_cache,
        )

        # Legacy warm-start cache (kept for backward compatibility, but unused)
        self._qp_warm_start_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

        # Placeholder for MonotonicityEnforcer - will be initialized after neighborhoods built
        self._monotonicity_enforcer: MonotonicityEnforcer | None = None

        # QP stats placeholder (will be aliased to enforcer.stats after initialization)
        self.qp_stats: dict[str, Any] = {}
        self._current_point_idx = 0

        # Adaptive neighborhood parameters
        self.adaptive_neighborhoods = adaptive_neighborhoods
        self.max_delta_multiplier = max_delta_multiplier

        # Cache grid size info from geometry
        self._n_spatial_grid_points = self._compute_n_spatial_grid_points()

        # Cache domain bounds from geometry
        self.domain_bounds = self._get_domain_bounds()

        # Compute k_min from Taylor order if not provided
        from math import comb

        n_derivatives_required = comb(self.dimension + taylor_order, taylor_order) - 1
        if k_min is None:
            self.k_min = n_derivatives_required
        else:
            # Ensure k_min is at least what's required for Taylor expansion
            if k_min < n_derivatives_required:
                warnings.warn(
                    f"k_min={k_min} is less than required for Taylor order {taylor_order} "
                    f"in {self.dimension}D (need {n_derivatives_required}). "
                    f"Using k_min={n_derivatives_required} instead.",
                    UserWarning,
                    stacklevel=2,
                )
                self.k_min = n_derivatives_required
            else:
                self.k_min = k_min

        # Store new infrastructure parameters
        self._use_new_infrastructure = use_new_infrastructure
        self._derivative_method = derivative_method
        self._rbf_kernel = rbf_kernel
        self._rbf_poly_degree = rbf_poly_degree

        # Local Coordinate Rotation for boundary accuracy (Issue #531)
        self._use_local_coordinate_rotation = use_local_coordinate_rotation

        # Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        self._use_ghost_nodes = use_ghost_nodes

        # Wind-Dependent BC for viscosity solution compatibility
        self._use_wind_dependent_bc = use_wind_dependent_bc

        # Hyperviscosity parameter for wind-dependent BC stabilization
        # epsilon > 0 adds damping: u_ghost = 2u_b - u_m - epsilon*(u_b - u_m)
        # Recommended: 0.0 (no damping) to 0.3 (moderate damping)
        self._wind_bc_hyperviscosity = 0.0  # Default: no hyperviscosity

        # Check for mutual exclusivity (ghost nodes takes precedence)
        if self._use_ghost_nodes and self._use_local_coordinate_rotation:
            warnings.warn(
                "Both use_ghost_nodes and use_local_coordinate_rotation are enabled. "
                "Ghost nodes take precedence and LCR will be disabled for boundary points.",
                UserWarning,
                stacklevel=2,
            )

        # Wind-dependent BC requires ghost nodes
        if self._use_wind_dependent_bc and not self._use_ghost_nodes:
            raise ValueError(
                "use_wind_dependent_bc=True requires use_ghost_nodes=True. "
                "Wind-dependent BC is a modification of the ghost nodes method."
            )

        # DEBUG: Print wind-BC configuration once at initialization
        if self._use_wind_dependent_bc:
            import sys

            print(f"\n[Wind-BC INIT] Enabled with {len(boundary_indices)} boundary points", flush=True, file=sys.stderr)

        # Create differential operator using Strategy Pattern
        if use_new_infrastructure:
            # New infrastructure: TaylorOperator or LocalRBFOperator
            if derivative_method == "taylor":
                self._gfdm_operator = TaylorOperator(
                    points=collocation_points,
                    delta=delta,
                    taylor_order=taylor_order,
                    weight_function=weight_function,
                    k_neighbors=k_neighbors,
                    neighborhood_mode=neighborhood_mode,
                    geometry=collocation_geometry,  # Issue #711: periodic support
                )
            elif derivative_method == "rbf":
                self._gfdm_operator = create_operator(
                    points=collocation_points,
                    delta=delta,
                    method="rbf",
                    kernel=rbf_kernel,
                    poly_degree=rbf_poly_degree,
                    k_neighbors=k_neighbors,
                    neighborhood_mode=neighborhood_mode,
                )
            else:
                raise ValueError(f"Unknown derivative_method: {derivative_method}")

            # Initialize BC handler with Row Replacement pattern
            self._bc_handler = DirectCollocationHandler()

            # Initialize BoundaryHandler component (Issue #545: composition over mixins)
            # This component handles boundary normals, LCR, ghost nodes, etc.
            self._boundary_handler = BoundaryHandler(
                collocation_points=collocation_points,
                dimension=self.dimension,
                domain_bounds=self.domain_bounds,
                boundary_indices=self.boundary_indices,
                neighborhoods={},  # Will be populated by _build_neighborhood_structure
                boundary_conditions=self.boundary_conditions,
                use_ghost_nodes=self._use_ghost_nodes,
                use_wind_dependent_bc=self._use_wind_dependent_bc,
                gfdm_operator=self._gfdm_operator,
                bc_property_getter=lambda prop, default=None: self._get_boundary_condition_property(prop) or default,
                gradient_computer=None,  # Will be set later if needed
            )

            # Compute boundary normals for Neumann BC
            self._boundary_normals = self._boundary_handler.compute_boundary_normals()
            # Store in handler for access by other components
            self._boundary_handler.boundary_normals = self._boundary_normals

            # Create unified BC config (single source of truth)
            self._bc_config = self._boundary_handler.create_bc_config()

            # Pre-classify every boundary collocation point to (BoundaryFace,
            # BCSegment) at construction time. Fails fast if any point cannot
            # be matched — better diagnostic than discovering it as a zero
            # Jacobian row 80 Newton iters later.
            self._preclassify_boundary_points()

            # Initialize NeighborhoodBuilder component (Issue #545: composition over mixins)
            # This component handles stencil construction, Taylor matrices, weight functions
            self._neighborhood_builder = NeighborhoodBuilder(
                collocation_points=collocation_points,
                dimension=self.dimension,
                delta=delta,
                taylor_order=taylor_order,
                weight_function=weight_function,
                weight_scale=weight_scale,
                k_min=self.k_min,
                adaptive_neighborhoods=adaptive_neighborhoods,
                max_delta_multiplier=max_delta_multiplier,
                boundary_indices=self.boundary_indices,
                n_derivatives=0,  # Will be set after multi_indices are determined
                multi_indices=[],  # Will be populated after operator initialization
                gfdm_operator=self._gfdm_operator,
                use_local_coordinate_rotation=self._use_local_coordinate_rotation,
                boundary_handler=self._boundary_handler,
                obstacle_sdf=obstacle_sdf,
                visibility_samples=visibility_samples,
                visibility_margin=visibility_margin,
            )
        else:
            raise ValueError(
                "use_new_infrastructure=False is no longer supported (removed in v0.17.15). "
                "Use use_new_infrastructure=True (default) with TaylorOperator."
            )

        # Get multi-indices from operator
        self.multi_indices = self._gfdm_operator.multi_indices
        self.n_derivatives = len(self.multi_indices)

        # Update neighborhood builder with multi_indices (for new infrastructure)
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.multi_indices = self.multi_indices
            self._neighborhood_builder.n_derivatives = self.n_derivatives

        # Store spatial shape for grid<->collocation interpolation
        # This is needed for _map_grid_to_collocation and _map_collocation_to_grid
        # get_grid_shape() returns node counts (Nx+1, Ny+1), not cell counts
        self._output_spatial_shape = tuple(self.problem.geometry.get_grid_shape())

        # Initialize grid-collocation mapper (Issue #545: composition over mixins)
        self._mapper = GridCollocationMapper(
            collocation_points=collocation_points,
            grid_shape=self._output_spatial_shape,
            domain_bounds=self.domain_bounds,
        )

        # Build neighborhood structure - uses GFDMOperator's neighborhoods as base,
        # only extends for points needing adaptive delta enlargement
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.build_neighborhood_structure()
        else:
            # Legacy fallback
            self._build_neighborhood_structure()

        # Update boundary handler neighborhoods reference (after they're built)
        if self._boundary_handler is not None:
            self._boundary_handler.neighborhoods = self.neighborhoods

        # Apply Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        # Ghost nodes take precedence over LCR if both are enabled
        # This must be called BEFORE Taylor matrices are built, since it augments neighborhoods
        if self._use_ghost_nodes:
            if self._boundary_handler is not None:
                self._boundary_handler.apply_ghost_nodes_to_neighborhoods()
            else:
                # Legacy fallback (shouldn't happen with new infrastructure)
                self._apply_ghost_nodes_to_neighborhoods()
        elif self._use_local_coordinate_rotation:
            # Apply Local Coordinate Rotation for boundary stencils (Issue #531)
            # This modifies neighborhoods by adding rotated_offsets for better normal derivatives
            if self._boundary_handler is not None:
                self._boundary_handler.apply_local_coordinate_rotation()
            else:
                # Legacy fallback (shouldn't happen with new infrastructure)
                self._apply_local_coordinate_rotation()

        # Build reverse neighborhood map for sparse Jacobian (point j -> rows affected)
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.build_reverse_neighborhoods()
        else:
            # Legacy fallback
            self._build_reverse_neighborhoods()

        # Build Taylor matrices for extended neighborhoods
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.build_taylor_matrices()
        else:
            # Legacy fallback
            self._build_taylor_matrices()

        # Initialize MonotonicityEnforcer component (Issue #545: composition over mixins)
        # Only create enforcer if QP optimization is enabled
        if self.qp_optimization_level != "none":
            self._monotonicity_enforcer = MonotonicityEnforcer(
                qp_solver=self._qp_solver_instance,
                qp_constraint_mode=self.qp_constraint_mode,
                collocation_points=self.collocation_points,
                neighborhoods=self.neighborhoods,
                multi_indices=self.multi_indices,
                domain_bounds=self.domain_bounds,
                delta=self.delta,
                sigma_function=self._get_sigma_value,
            )
            # Alias qp_stats to enforcer.stats for backward compatibility
            self.qp_stats = self._monotonicity_enforcer.stats
        else:
            self._monotonicity_enforcer = None
            # Initialize empty qp_stats for "none" level
            self.qp_stats = {
                "total_qp_solves": 0,
                "qp_times": [],
                "violations_detected": 0,
                "violation_point_indices": set(),
                "violation_laplacian": 0,
                "violation_gradient": 0,
                "violation_higher_order": 0,
                "points_checked": 0,
                "qp_successes": 0,
                "qp_failures": 0,
                "qp_fallbacks": 0,
                "slsqp_solves": 0,
                "lbfgsb_solves": 0,
                "osqp_solves": 0,
                "osqp_failures": 0,
            }

        # Initialize precomputed joint SOCP stencils first (joint_socp scheme only).
        # Audit-major Phase 1B: enforce M-matrix + per-edge cone at all interior nodes
        # where the joint SOCP is feasible (paper Theorem `thm:joint_socp_feasibility`).
        self._joint_socp_stencils = None
        if self.monotonicity_scheme == "joint_socp":
            from mfgarchon.alg.numerical.gfdm_components.joint_socp import (
                PrecomputedJointSocpStencils,
            )

            interior_indices = np.setdiff1d(np.arange(self.n_points), self.boundary_indices)
            self._joint_socp_stencils = PrecomputedJointSocpStencils(
                operator=self._gfdm_operator,
                points=self.collocation_points,
                interior_indices=interior_indices,
                delta=delta,
                cone_constant_C=1.0,  # within $C_\\star \\in [0.5, 1]$ for Wendland C^2
                eps_pos=0.0,
            )
            stats = self._joint_socp_stencils.stats
            logger.info(
                f"Precomputed joint SOCP stencils: feasible {stats['n_feasible']}/"
                f"{stats['n_interior']} interior "
                f"({stats['n_fast_path']} via Wendland-LSQ fast-path, "
                f"{stats['n_socp']} via CLARABEL SOCP) in {stats['time_ms']:.1f}ms; "
                f"SOCP-infeasible {stats['n_infeasible']} fall back to "
                f"M-matrix QP (Phase 2)"
            )

        # Initialize precomputed M-matrix QP stencils at boundary nodes.
        # Activated under both `qp_optimization_level == "precompute"` (legacy
        # qp_m_matrix scheme) and `monotonicity_scheme == "joint_socp"` (which
        # internally aliases qp_optimization_level to "precompute" — see above).
        # SOCP-infeasible interior nodes fall through to bare Wendland-Taylor;
        # extending the buffer set to cover them was empirically destabilizing
        # (Lap-only correction creates Lap/Grad inconsistency at those nodes).
        self._precomputed_stencils: PrecomputedMonotoneStencils | None = None
        if self.qp_optimization_level == "precompute":
            is_buffer = np.zeros(self.n_points, dtype=bool)
            is_buffer[self.boundary_indices] = True
            self._precomputed_stencils = PrecomputedMonotoneStencils(
                operator=self._gfdm_operator,
                is_boundary=is_buffer,
                tolerance=1e-6,
            )
            logger.info(
                f"Precomputed monotone stencils: {self._precomputed_stencils.stats['n_monotonized']}/{self._precomputed_stencils.stats['n_boundary']} "
                f"buffer points in {self._precomputed_stencils.stats['time_ms']:.1f}ms"
            )

        # Lazy-initialized cache attributes
        # These are expensive to compute and only created when needed
        self._D_grad: list | None = None  # Gradient differentiation matrices
        self._D_lap: Any | None = None  # Laplacian differentiation matrix
        self._potential_at_collocation: np.ndarray | None = None  # Interpolated potential field
        self._cached_derivative_weights: dict | None = None  # Pre-computed GFDM weights
        self._running_cost_fn: Callable[[int], np.ndarray] | None = None  # Running cost f(n) -> (n_points,)
        self._f_potential_warned: bool = False  # One-time warning for unused f_potential (Issue #766)

    def _normalize_running_cost(
        self,
        running_cost: np.ndarray | Callable[[int], np.ndarray] | None,
        n_time_points: int,
    ) -> Callable[[int], np.ndarray] | None:
        """Normalize running cost input to a callable f(n) -> (n_points,).

        Accepts three input forms:
            - None: no running cost
            - 1D array (n_points,): static cost, same at every timestep
            - 2D array (n_time_points, n_points): time-dependent cost
            - Callable: f(time_index) -> (n_points,) array, used directly
        """
        if running_cost is None:
            return None

        # Callable path: validate output shape and return directly
        if callable(running_cost):
            test_output = np.asarray(running_cost(0))
            if test_output.shape != (self.n_points,):
                raise ValueError(f"running_cost callable must return shape ({self.n_points},), got {test_output.shape}")
            return running_cost

        # Array path: normalize to callable
        running_cost = np.asarray(running_cost)
        if running_cost.ndim == 1:
            if running_cost.shape[0] != self.n_points:
                raise ValueError(
                    f"running_cost must have shape ({self.n_points},) or "
                    f"({n_time_points}, {self.n_points}), got {running_cost.shape}"
                )
            rc_static = running_cost.copy()
            return lambda _n: rc_static
        elif running_cost.ndim == 2:
            if running_cost.shape != (n_time_points, self.n_points):
                raise ValueError(
                    f"running_cost must have shape ({n_time_points}, {self.n_points}), got {running_cost.shape}"
                )
            rc_full = running_cost.copy()
            return lambda n: rc_full[n]
        else:
            raise ValueError(f"running_cost must be 1D or 2D array, got {running_cost.ndim}D")

    def _compute_n_spatial_grid_points(self) -> int:
        """Compute total number of spatial grid points from geometry."""
        grid_shape = self.problem.geometry.get_grid_shape()
        return int(np.prod(grid_shape))

    def _get_boundary_condition_property(self, property_name: str) -> Any:
        """Get boundary condition property - returns None if not available.

        BC validation is deferred to solve time to allow testing internal mechanics
        without requiring full BC specification.

        For mixed BCs, returns None with a warning (allows fallback to per-point BC).
        """
        # No BC specified - return None (validation deferred to solve time)
        if self.boundary_conditions is None:
            return None

        # Try dictionary access first (doesn't trigger property)
        if isinstance(self.boundary_conditions, dict):
            return self.boundary_conditions.get(property_name)

        # Try attribute access (may raise ValueError for mixed BC properties)
        try:
            return getattr(self.boundary_conditions, property_name)
        except AttributeError:
            return None
        except ValueError:
            # Mixed BC - warn and return None to allow fallback to default_bc
            # Per-point BC types are not yet supported in HJB GFDM solver.
            # The solver will use default_bc for all boundary points.
            if not getattr(self, "_mixed_bc_warned", False):
                logger.info(
                    f"Mixed BC detected: '{property_name}' is not uniform. "
                    f"Per-point BC types will be applied (DIRICHLET at exits, NEUMANN at walls)."
                )
                self._mixed_bc_warned = True
            return None

    def _get_bc_type_for_point(self, point_idx: int) -> str:
        """Determine BC type for a boundary collocation point.

        Resolution order:

        1. **Pre-classified table** (preferred): for mixed BC, the segment
           was resolved at solver __init__ time and stored in
           ``self._bc_segment_per_point``. O(1) lookup, no re-classification.
        2. **Uniform BC**: read global type from ``_bc_config``.

        For mixed BC where the point was *not* pre-classified, this method
        raises ``ValueError`` rather than silently falling back to
        ``self.boundary_conditions.default_bc`` (which defaults to PERIODIC
        — historically the source of silent zero Jacobian rows). The
        pre-classification at __init__ already raised on unmatched points
        with full diagnostic; reaching here means the boundary_indices set
        was mutated after __init__, which is a programmer error.

        Returns:
            BC type string: "dirichlet", "neumann", or any other BCType
            ``.value.lower()`` for completeness.
        """
        try:
            is_mixed = self.boundary_conditions.is_mixed
        except AttributeError:
            is_mixed = False

        if is_mixed:
            # Pre-classified table is authoritative for mixed BC.
            try:
                segment = self._bc_segment_per_point[point_idx]
            except (AttributeError, KeyError) as exc:
                raise ValueError(
                    f"_get_bc_type_for_point({point_idx}): point not in pre-classified "
                    f"table. boundary_indices was likely mutated after solver __init__, "
                    f"or this method was called before HJBGFDMSolver.__init__ completed. "
                    f"Pre-classified count: "
                    f"{len(getattr(self, '_bc_segment_per_point', {}))}/"
                    f"{len(self.boundary_indices)}."
                ) from exc
            if segment.bc_type == BCType.DIRICHLET:
                return "dirichlet"
            if segment.bc_type in (BCType.NEUMANN, BCType.NO_FLUX):
                return "neumann"
            return segment.bc_type.value.lower()

        # Uniform BC: global type from config.
        bc_type = self._bc_config.get("type") if self._bc_config else None
        if bc_type is None:
            raise ValueError(
                "BC type required but not specified in config (uniform BC path). "
                "Provide boundary_conditions= when constructing HJBGFDMSolver."
            )
        return bc_type

    def _preclassify_boundary_points(self) -> None:
        """Pre-classify every boundary collocation point to a BCSegment + face + normal.

        Called once at __init__ time. Populates three companion maps:

        - ``self._bc_face_per_point[i]``: BoundaryFace the point lies on.
        - ``self._bc_segment_per_point[i]``: BCSegment that applies to ``i``.
        - ``self._bc_normal_per_point[i]``: outward unit normal (axis-aligned,
          from the face — not from any SDF gradient).

        Raises if any boundary point cannot be classified to a face or matched
        to a segment. This converts a class of latent failures — silent
        ``default_bc=PERIODIC`` fallback plus zero Jacobian rows discovered
        80 Newton iterations later — into a loud, diagnosable construction-
        time error.

        Only runs for mixed-BC setups; uniform BC keeps the global-type fast
        path. Skipped entirely if ``len(self.boundary_indices) == 0``.
        """
        self._bc_face_per_point: dict[int, BoundaryFace] = {}
        self._bc_segment_per_point: dict[int, BCSegment] = {}
        self._bc_normal_per_point: dict[int, np.ndarray] = {}

        if len(self.boundary_indices) == 0 or self.boundary_conditions is None:
            return
        try:
            is_mixed = self.boundary_conditions.is_mixed
        except AttributeError:
            return
        if not is_mixed:
            return

        bounds = self._get_domain_bounds_array()
        sorted_segments = sorted(
            self.boundary_conditions.segments,
            key=lambda seg: seg.priority,
            reverse=True,
        )

        # tolerance for classification: 1e-6 covers ε=1e-6 collocation
        # generators (e.g. SDF-clipped boundary points placed at micron
        # distance from the wall). Users with looser collocation can override
        # via a future kwarg; current default is conservative.
        tol = 1e-6
        unmatched: list[tuple[int, np.ndarray, BoundaryFace | None, str]] = []

        for i in self.boundary_indices:
            i = int(i)
            point = self.collocation_points[i]

            face = self.boundary_conditions.identify_boundary_face(
                point=point,
                tolerance=tol,
                domain_bounds=bounds,
            )
            if face is None:
                unmatched.append((i, point, None, "no BoundaryFace match"))
                continue

            matching_segment: BCSegment | None = None
            for seg in sorted_segments:
                if seg.matches_point(
                    point=point,
                    boundary_id=face.to_string(),
                    domain_bounds=bounds,
                ):
                    matching_segment = seg
                    break

            if matching_segment is None:
                unmatched.append((i, point, face, f"BoundaryFace={face!r} not covered by any segment"))
                continue

            self._bc_face_per_point[i] = face
            self._bc_segment_per_point[i] = matching_segment
            self._bc_normal_per_point[i] = self.boundary_conditions.outward_normal_for_face(
                face, dimension=self.dimension
            )

        if unmatched:
            lines = [
                f"HJBGFDMSolver: BC pre-classification failed for "
                f"{len(unmatched)}/{len(self.boundary_indices)} boundary points.",
                "",
                "Common causes:",
                f"  1. Collocation generator places boundary points >{tol:.0e} off the wall "
                "(e.g. ε=1e-4 with default tol=1e-6) → bump tolerance, or shrink ε.",
                "  2. BoundaryConditions.segments don't cover every geometric face → add "
                "a segment for the missing face, or set boundary='all'.",
                "  3. domain_bounds inferred from problem.geometry differ from collocation "
                "extent (e.g. quirky obstacle-clipping geometry).",
                "",
                "Unmatched points (first 5):",
            ]
            for i, point, face, reason in unmatched[:5]:
                lines.append(f"  pt {i} at {point.tolist()}: face={face!r} -- {reason}")
            if len(unmatched) > 5:
                lines.append(f"  ... and {len(unmatched) - 5} more")
            raise ValueError("\n".join(lines))

    def _detect_boundary_indices(self, collocation_points: np.ndarray) -> np.ndarray:
        """Auto-detect boundary point indices from collocation points and domain bounds.

        Points are classified as boundary if they lie within tolerance of any domain boundary.

        Args:
            collocation_points: Array of shape (n_points, dimension) with collocation coordinates.

        Returns:
            Array of boundary point indices. Empty array if bounds cannot be determined.

        Note:
            Issue #542 fix - enables automatic BC enforcement without explicit boundary_indices.
        """
        # Get domain bounds
        bounds = self._get_domain_bounds_for_detection()
        if bounds is None or len(bounds) == 0:
            return np.array([], dtype=int)

        tol = 1e-6
        boundary_mask = np.zeros(len(collocation_points), dtype=bool)

        for d, (d_min, d_max) in enumerate(bounds):
            if d < collocation_points.shape[1]:
                # Points at min or max boundary in this dimension
                at_min = np.abs(collocation_points[:, d] - d_min) < tol
                at_max = np.abs(collocation_points[:, d] - d_max) < tol
                boundary_mask |= at_min | at_max

        return np.where(boundary_mask)[0]

    def _get_domain_bounds_for_detection(self) -> list[tuple[float, float]] | None:
        """Get domain bounds for boundary detection (before full initialization)."""
        # Try geometry interface first
        try:
            geom = self.problem.geometry
            if geom is not None:
                try:
                    bounds_result = geom.get_bounds()
                    if bounds_result is not None:
                        min_coords, max_coords = bounds_result
                        return [(float(min_coords[d]), float(max_coords[d])) for d in range(len(min_coords))]
                except AttributeError:
                    pass
                try:
                    return list(geom.bounds)
                except AttributeError:
                    pass
        except AttributeError:
            pass
        # Fallback to legacy xmin/xmax
        try:
            xmin = self.problem.xmin
            xmax = self.problem.xmax
            return [(float(xmin), float(xmax))]
        except AttributeError:
            return None

    def _get_domain_bounds(self) -> list[tuple[float, float]]:
        """Get domain bounds from geometry or legacy xmin/xmax attributes.

        Returns:
            List of (min, max) tuples for each dimension.

        Note:
            Issue #542 fix - removed hasattr/getattr, using try/except for clearer failure modes.
        """
        # Try geometry interface first (modern API)
        try:
            geom = self.problem.geometry
            if geom is not None:
                try:
                    # Prefer get_bounds() method
                    bounds_result = geom.get_bounds()
                    if bounds_result is not None:
                        min_coords, max_coords = bounds_result
                        return [(float(min_coords[d]), float(max_coords[d])) for d in range(len(min_coords))]
                except AttributeError:
                    pass
                try:
                    # Fallback to .bounds property
                    return list(geom.bounds)
                except AttributeError:
                    pass
        except AttributeError:
            pass

        # Fallback to legacy 1D xmin/xmax
        try:
            xmin = self.problem.xmin
            xmax = self.problem.xmax
            return [(float(xmin), float(xmax))]
        except AttributeError:
            pass

        # Last resort: infer from collocation points
        mins = self.collocation_points.min(axis=0)
        maxs = self.collocation_points.max(axis=0)
        return list(zip(mins.astype(float).tolist(), maxs.astype(float).tolist(), strict=True))

    def _get_domain_bounds_array(self) -> np.ndarray | None:
        """Get domain bounds as numpy array for BCSegment.matches_point().

        Returns:
            Array of shape (dimension, 2) where bounds[i, 0] = min and bounds[i, 1] = max
            for dimension i. Returns None if bounds cannot be determined.

        Note:
            Issue #542 fix - provides bounds in format expected by BCSegment.
        """
        bounds_list = self._get_domain_bounds()
        if not bounds_list:
            return None
        return np.array(bounds_list, dtype=float)

    def _infer_boundary_id(self, point: np.ndarray, domain_bounds: np.ndarray | None, tol: float = 1e-6) -> str | None:
        """Infer boundary identifier for a point on rectangular domain boundary.

        This is optional - BCSegment.matches_point() can work without boundary_id
        using SDF or normal matching. For rectangular domains, providing boundary_id
        enables efficient segment matching via the 'boundary' attribute.

        Args:
            point: Spatial coordinates (dimension,)
            domain_bounds: Domain bounds array (dimension, 2) or None
            tol: Tolerance for boundary detection

        Returns:
            Boundary identifier like "x_min", "y_max", or None if not on axis-aligned boundary.

        Note:
            Issue #542 fix - separated boundary inference from BC matching.
            Returns None for non-rectangular domains or interior/corner points.
        """
        if domain_bounds is None:
            return None

        # Check each axis for boundary proximity
        for axis_idx in range(min(len(point), len(domain_bounds))):
            if abs(point[axis_idx] - domain_bounds[axis_idx, 0]) < tol:
                axis_name = ["x", "y", "z"][axis_idx] if axis_idx < 3 else f"dim{axis_idx}"
                return f"{axis_name}_min"
            elif abs(point[axis_idx] - domain_bounds[axis_idx, 1]) < tol:
                axis_name = ["x", "y", "z"][axis_idx] if axis_idx < 3 else f"dim{axis_idx}"
                return f"{axis_name}_max"

        return None

    def _get_domain_sdf(self) -> callable | None:
        """Get signed distance function from geometry if available.

        Returns:
            SDF callable or None if geometry doesn't provide one.

        Note:
            Issue #542 fix - enables BC matching on SDF-based geometries.
        """
        try:
            return self.problem.geometry.sdf
        except AttributeError:
            return None

    def _compute_domain_volume(self) -> float:
        """Compute the volume (area in 2D) of the domain.

        Used for normalizing density in multiplicative congestion mode,
        where H = (1 + γ|Ω|m)|p|²/(2λ). The |Ω|m factor makes γ dimensionless.

        Returns:
            Domain volume/area as a scalar.
        """
        try:
            return self._domain_volume
        except AttributeError:
            pass

        bounds = self.domain_bounds
        volume = 1.0
        for d_min, d_max in bounds:
            volume *= d_max - d_min

        self._domain_volume = volume
        return volume

    # =========================================================================
    # Boundary Methods: Provided by BoundaryHandler component (Issue #545)
    # compute_boundary_normals, build_rotation_matrix, apply_local_coordinate_rotation
    # rotate_derivatives_back, apply_ghost_nodes_to_neighborhoods
    # build_neumann_bc_weights
    # =========================================================================

    def _compute_gradient_at_point(self, u_values: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Compute gradient ∇u at a single point using GFDM weights.

        Args:
            u_values: Solution vector at all collocation points
            point_idx: Index of point where gradient is computed

        Returns:
            Gradient vector ∇u, shape (dimension,)
        """
        # Get neighborhood for this point
        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]

        # Get Taylor weights for derivatives
        weights = neighborhood["weights"]

        # Extract gradient weights (first-order derivatives: columns 1 to dimension+1)
        # weights structure: [u, u_x, u_y, u_xx, u_xy, u_yy, ...] for 2D
        grad_weights = weights[:, 1 : 1 + self.dimension]  # Shape: (n_neighbors, dimension)

        # Get neighbor values (using standard ghost mirroring, no wind-dependent check)
        # This avoids circular dependency: we need gradient to check wind direction,
        # but we can't check wind direction while computing the gradient!
        # Solution: use standard ghosts here, wind-dependent BC applies at derivative computation
        if self._use_ghost_nodes and self._boundary_handler is not None:
            # Standard ghost mirroring
            ghost_to_mirror = {}
            for ghost_info in self._boundary_handler.ghost_node_map.values():
                ghost_to_mirror.update(ghost_info["ghost_to_mirror"])

            u_neighbors_list = []
            for idx in neighbor_indices:
                if idx < 0:
                    mirror_idx = ghost_to_mirror.get(int(idx))
                    u_neighbors_list.append(u_values[mirror_idx] if mirror_idx is not None else 0.0)
                else:
                    u_neighbors_list.append(u_values[int(idx)])
            u_neighbors = np.array(u_neighbors_list)
        else:
            u_neighbors = u_values[neighbor_indices]

        # Compute gradient: ∇u = Σ w_i * u_i for each component
        grad_u = grad_weights.T @ u_neighbors  # Shape: (dimension,)

        return grad_u

    def _build_differentiation_matrices(self) -> None:
        """
        Pre-compute sparse differentiation matrices for vectorized derivative computation.

        Builds:
        - D_grad: List of sparse matrices (n_points x n_points) for each gradient component
        - D_lap: Sparse matrix (n_points x n_points) for Laplacian

        After this, derivatives can be computed via matrix-vector multiplication:
            grad_u[d] = D_grad[d] @ u
            lap_u = D_lap @ u

        This converts O(n * k^2) per-point computation to O(n * k) matrix multiplication.
        """
        from scipy.sparse import lil_matrix

        n = self.n_points
        d = self.dimension

        # Initialize sparse matrices in LIL format (efficient for construction)
        D_grad_lil = [lil_matrix((n, n)) for _ in range(d)]
        D_lap_lil = lil_matrix((n, n))

        # Pre-compute LCR boundary points set for fast lookup
        lcr_boundary_set = set()
        if self._use_local_coordinate_rotation and self._boundary_handler is not None:
            lcr_boundary_set = set(self._boundary_handler.boundary_rotations.keys())

        for i in range(n):
            # For LCR boundary points, use our Taylor matrices with rotation
            if i in lcr_boundary_set:
                if self._neighborhood_builder is not None:
                    boundary_rotations = self._boundary_handler.boundary_rotations if self._boundary_handler else None
                    weights = self._neighborhood_builder.compute_derivative_weights_from_taylor(i, boundary_rotations)
                else:
                    # Legacy fallback
                    weights = self._compute_derivative_weights_from_taylor(i)
            else:
                weights = self._gfdm_operator.get_derivative_weights(i)

            if weights is None:
                continue

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]  # shape: (d, n_neighbors)
            lap_weights = weights["lap_weights"]  # shape: (n_neighbors,)

            # Override Laplacian weights with M-matrix QP precomputed monotone
            # weights if available. Two activation paths both populate
            # self._precomputed_stencils:
            #   (1) qp_m_matrix scheme + precompute application (legacy path)
            #   (2) joint_socp scheme — applies M-matrix QP at boundary buffer
            #       nodes where joint SOCP is infeasible (Phase 2 fallback per
            #       paper §831). The `joint_socp_stencils` override below takes
            #       priority for SOCP-feasible interior nodes.
            if self._precomputed_stencils is not None and self._precomputed_stencils.has_stencil(i):
                precomputed = self._precomputed_stencils.get_laplacian_weights(i)
                if precomputed is not None:
                    lap_weights = precomputed[0]  # (weights, neighbor_indices)

            # Override BOTH Laplacian and gradient weights with joint SOCP weights at
            # interior nodes where SOCP is feasible (audit-major Phase 1B). This takes
            # priority over qp_m_matrix precompute. Boundary buffer nodes where SOCP
            # is infeasible fall back to qp_m_matrix above (or default Wendland-Taylor
            # if qp_m_matrix precompute also doesn't have a stencil there).
            if self._joint_socp_stencils is not None and self._joint_socp_stencils.has_stencil(i):
                socp_weights = self._joint_socp_stencils.get_weights_dict(i)
                if socp_weights is not None:
                    # Joint SOCP guarantees same neighbor_indices + center as the
                    # operator's stencil (it uses op.get_derivative_weights to build).
                    lap_weights = socp_weights["lap_weights"]
                    grad_weights = socp_weights["grad_weights"]

            # Fill gradient matrices
            for dim in range(d):
                # Neighbor contributions (skip ghost particles with j < 0)
                real_grad_sum = 0.0
                for k, j in enumerate(neighbor_indices):
                    if j >= 0:
                        D_grad_lil[dim][i, j] = grad_weights[dim, k]
                        real_grad_sum += grad_weights[dim, k]
                # Center contribution (sum rule: center weight = -sum of REAL neighbor weights)
                # Note: Must exclude ghost particle weights to maintain row sum = 0
                center_weight = -real_grad_sum
                D_grad_lil[dim][i, i] += center_weight

            # Fill Laplacian matrix (same fix: exclude ghost weights from center)
            real_lap_sum = 0.0
            for k, j in enumerate(neighbor_indices):
                if j >= 0:
                    D_lap_lil[i, j] = lap_weights[k]
                    real_lap_sum += lap_weights[k]
            D_lap_lil[i, i] += -real_lap_sum

        # Convert to CSR format for efficient matrix-vector multiplication
        self._D_grad = [D.tocsr() for D in D_grad_lil]
        self._D_lap = D_lap_lil.tocsr()

    def _compute_derivatives_vectorized(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients and Laplacian for all points via sparse matrix multiplication.

        Args:
            u: Function values at collocation points, shape (n_points,)

        Returns:
            grad_u: Gradient at all points, shape (n_points, dimension)
            lap_u: Laplacian at all points, shape (n_points,)
        """
        # Lazy initialization of differentiation matrices (expensive computation)
        # self._D_grad initialized as None in __init__, computed on first use
        if self._D_grad is None:
            self._build_differentiation_matrices()

        grad_u = np.column_stack([D @ u for D in self._D_grad])
        lap_u = self._D_lap @ u

        # Note: LCR rotation is now applied in _compute_derivative_weights_from_taylor()
        # so gradients are already in the original coordinate frame

        return grad_u, lap_u

    def approximate_derivatives(self, u_values: np.ndarray, point_idx: int) -> dict[tuple[int, ...], float]:
        """
        Approximate derivatives at collocation point using weighted least squares.

        Args:
            u_values: Function values at collocation points
            point_idx: Index of the collocation point

        Returns:
            Dictionary mapping derivative multi-indices to approximated values
        """
        # Track current point for debugging/statistics
        self._current_point_idx = point_idx

        # Get QP level and check for ghost particles
        qp_level = getattr(self, "qp_optimization_level", "none")
        neighborhood = self.neighborhoods[point_idx]
        has_ghost = neighborhood.get("has_ghost", False)

        # Fast path: delegate to GFDMOperator when no ghost particles and no QP needed
        if qp_level == "none" and not has_ghost:
            return self._gfdm_operator.approximate_derivatives_at_point(u_values, point_idx)

        # Slow path: handle ghost particles and/or QP constraints
        if self.taylor_matrices[point_idx] is None:
            return {}

        taylor_data = self.taylor_matrices[point_idx]

        # Extract function values at neighborhood points, handling ghost nodes/particles
        neighbor_indices = neighborhood["indices"]
        u_center = u_values[point_idx]

        # Use ghost-aware value retrieval if ghost nodes method is active
        if self._use_ghost_nodes:
            if self._boundary_handler is not None:
                u_neighbors = self._boundary_handler.get_values_with_ghosts(
                    u_values, neighbor_indices, point_idx=point_idx
                )
            else:
                # Legacy fallback
                u_neighbors = self._get_values_with_ghosts(u_values, neighbor_indices, point_idx=point_idx)
        else:
            # Handle legacy ghost particles based on BC type
            # - Neumann/no-flux: u_ghost = u_center (mirror value)
            # - Dirichlet: u_ghost = BC value (if available)
            bc_type = self._get_boundary_condition_property("type")
            bc_values = self._get_boundary_condition_property("values")

            u_neighbors = []
            for idx in neighbor_indices:  # type: ignore[attr-defined]
                if idx >= 0:
                    # Regular neighbor
                    u_neighbors.append(u_values[idx])
                else:
                    # Legacy ghost particle: value depends on BC type
                    if bc_type == "dirichlet" and bc_values is not None:
                        # Dirichlet BC: use prescribed value
                        # Note: bc_values may be scalar, array, or callable
                        if callable(bc_values):
                            x_pos = self.collocation_points[point_idx]
                            u_neighbors.append(bc_values(x_pos))
                        elif isinstance(bc_values, (list, tuple, np.ndarray)):
                            # Array-like: use value at this point
                            u_neighbors.append(bc_values[point_idx] if point_idx < len(bc_values) else 0.0)
                        else:
                            # Scalar
                            u_neighbors.append(float(bc_values))
                    else:
                        # Neumann/no-flux: mirror u_center
                        u_neighbors.append(u_center)

        u_neighbors = np.array(u_neighbors)  # type: ignore[assignment]

        # Right-hand side: u(x_neighbor) - u(x_center) for Taylor expansion
        # u(x_j) - u(x_0) ≈ ∇u·(x_j - x_0) where A matrix uses (x_j - x_0)
        # For ghost particles: u_ghost = u_center → b = 0, enforcing ∂u/∂n = 0
        b = u_neighbors - u_center

        if qp_level == "always":
            # "always" level: Force QP at every point without checking M-matrix
            derivative_coeffs = self._monotonicity_enforcer.solve_constrained_qp(taylor_data, b, point_idx)  # type: ignore[union-attr]
        elif qp_level == "auto":
            # "auto" level: Adaptive QP with M-matrix checking
            # First try unconstrained solution to check if constraints are needed
            unconstrained_coeffs = self._monotonicity_enforcer._solve_unconstrained_fallback(taylor_data, b)  # type: ignore[union-attr]

            # Check if unconstrained solution violates monotonicity (M-matrix property)
            self.qp_stats["points_checked"] += 1
            needs_constraints = self._monotonicity_enforcer.check_monotonicity_violation(
                unconstrained_coeffs, point_idx
            )  # type: ignore[union-attr]

            if needs_constraints:
                # Apply constrained QP to enforce monotonicity
                self.qp_stats["violations_detected"] += 1
                self.qp_stats["violation_point_indices"].add(point_idx)
                derivative_coeffs = self._monotonicity_enforcer.solve_constrained_qp(taylor_data, b, point_idx)  # type: ignore[union-attr]
            else:
                # Use faster unconstrained solution
                derivative_coeffs = unconstrained_coeffs
        elif taylor_data.get("use_svd", False):  # type: ignore[attr-defined]
            # Use SVD: solve using pseudoinverse with truncated SVD
            sqrt_W = taylor_data["sqrt_W"]
            U = taylor_data["U"]
            S = taylor_data["S"]
            Vt = taylor_data["Vt"]

            # Compute sqrt(W) @ b
            Wb = sqrt_W @ b

            # SVD solution: x = V @ S^{-1} @ U^T @ Wb
            UT_Wb = U.T @ Wb
            S_inv_UT_Wb = UT_Wb / S  # Element-wise division
            derivative_coeffs = Vt.T @ S_inv_UT_Wb

        elif taylor_data.get("use_qr", False):  # type: ignore[attr-defined]
            # Use QR decomposition: solve R @ x = Q^T @ sqrt(W) @ b
            sqrt_W = taylor_data["sqrt_W"]
            Q = taylor_data["Q"]
            R = taylor_data["R"]

            Wb = sqrt_W @ b
            QT_Wb = Q.T @ Wb

            try:
                derivative_coeffs = np.linalg.solve(R, QT_Wb)
            except np.linalg.LinAlgError:
                # Fallback to least squares if R is singular
                A_matrix = taylor_data.get("A")  # type: ignore[attr-defined]
                if A_matrix is not None:
                    lstsq_result = lstsq(A_matrix, b)
                    derivative_coeffs = lstsq_result[0] if lstsq_result is not None else np.zeros(len(b))
                else:
                    derivative_coeffs = np.zeros(len(b))

        elif taylor_data.get("AtWA_inv") is not None:  # type: ignore[attr-defined]
            # Use precomputed normal equations
            derivative_coeffs = taylor_data["AtWA_inv"] @ taylor_data["AtW"] @ b
        else:
            # Final fallback to direct least squares
            A_matrix = taylor_data.get("A")  # type: ignore[attr-defined]
            if A_matrix is not None:
                lstsq_result = lstsq(A_matrix, b)
                derivative_coeffs = lstsq_result[0] if lstsq_result is not None else np.zeros(len(b))
            else:
                derivative_coeffs = np.zeros(len(b))

        # Handle case where coefficient computation failed
        if derivative_coeffs is None:
            derivative_coeffs = np.zeros(len(self.multi_indices))

        # Map coefficients to multi-indices
        derivatives = {}
        for k, beta in enumerate(self.multi_indices):
            derivatives[beta] = derivative_coeffs[k]

        # Apply inverse rotation for LCR boundary points (Issue #531)
        # Derivatives were computed in rotated frame, need to rotate back
        if (
            self._use_local_coordinate_rotation
            and self._boundary_handler is not None
            and point_idx in self._boundary_handler.boundary_rotations
        ):
            derivatives = self._boundary_handler.rotate_derivatives_back(
                derivatives, self._boundary_handler.boundary_rotations[point_idx]
            )

        # Consistency override: when precomputed monotone weights exist for this
        # point, override the corresponding derivative entries so the per-point
        # HJB Newton residual uses the SAME stencil weights as the Jacobian
        # (which is assembled from `_cached_derivative_weights`, populated with
        # SOCP / M-matrix-QP weights at __init__).
        #
        # Without this override, the slow path above computes derivatives via
        # bare Wendland-Taylor LSQ (`taylor_data["AtWA_inv"]` etc.), while the
        # Jacobian uses SOCP-corrected weights. Newton then solves
        #     J · δu = -r
        # with J and r assembled from inconsistent stencil weights, converging
        # to a stationary point of the mongrel system rather than the true
        # discrete-HJB fixed point. Empirically: 12× u_err discrepancy in the
        # exp08 step 4 2D Towel-on-Beach validation at N=100 (joint_socp
        # u_err iter 1 = 48.81 without this fix vs 5.38 with the fix, 9× match
        # to the qp_m_matrix control on the same setup).
        #
        # Precedence (matches `_build_derivative_matrices` / `_cached_derivative_weights`):
        #   joint SOCP at SOCP-feasible interior > M-matrix QP at boundary > bare W-T.
        if self._joint_socp_stencils is not None and self._joint_socp_stencils.has_stencil(point_idx):
            socp = self._joint_socp_stencils.get_weights_dict(point_idx)
            if socp is not None:
                L_w = socp["lap_weights"]  # shape (n_neighbors,)
                D_w = socp["grad_weights"]  # shape (d, n_neighbors)
                # Override gradient: ∂u/∂x_d (i) = sum_j D_w[d, j] * b_j
                for d in range(self.dimension):
                    beta = tuple(1 if k == d else 0 for k in range(self.dimension))
                    derivatives[beta] = float(D_w[d] @ b)
                # Override Laplacian sum (= trace of Hessian) via diagonal split.
                # Preserves bare-WT off-diagonal Hessian entries (e.g. (1,1) in 2D)
                # while enforcing target_lap = L_w · b on the trace.
                target_lap = float(L_w @ b)
                current_lap = sum(
                    float(derivatives.get(beta, 0.0))
                    for beta in self.multi_indices
                    if len(beta) == self.dimension and sum(beta) == 2 and max(beta) == 2
                )
                adjustment = (target_lap - current_lap) / self.dimension
                for d in range(self.dimension):
                    beta = tuple(2 if k == d else 0 for k in range(self.dimension))
                    if beta in derivatives:
                        derivatives[beta] = float(derivatives[beta]) + adjustment
        elif self._precomputed_stencils is not None and self._precomputed_stencils.has_stencil(point_idx):
            precomputed = self._precomputed_stencils.get_laplacian_weights(point_idx)
            if precomputed is not None:
                L_w = precomputed[0]  # shape (n_neighbors,)
                # M-matrix QP only corrects the Laplacian; gradient stays bare W-T.
                target_lap = float(L_w @ b)
                current_lap = sum(
                    float(derivatives.get(beta, 0.0))
                    for beta in self.multi_indices
                    if len(beta) == self.dimension and sum(beta) == 2 and max(beta) == 2
                )
                adjustment = (target_lap - current_lap) / self.dimension
                for d in range(self.dimension):
                    beta = tuple(2 if k == d else 0 for k in range(self.dimension))
                    if beta in derivatives:
                        derivatives[beta] = float(derivatives[beta]) + adjustment

        return derivatives

    def compute_all_derivatives(
        self, u: np.ndarray, use_qp: bool | None = None
    ) -> dict[int, dict[tuple[int, ...], float]]:
        """
        Compute derivatives at all collocation points using precomputed Taylor matrices.

        When to use this vs GFDMOperator:
        - Use this method when you need QP constraints for monotonicity (M-matrix)
        - Use GFDMOperator for general GFDM needs (FP solver, one-off computations)

        Example:
            # For QP-constrained derivatives (HJB specific):
            solver = HJBGFDMSolver(problem, points, monotonicity_scheme="auto")
            derivs = solver.compute_all_derivatives(u, use_qp=True)

            # For general GFDM (simpler, no QP):
            from mfgarchon.utils.numerical import GFDMOperator
            gfdm = GFDMOperator(points, delta=0.1)
            grad = gfdm.gradient(u)
            lap = gfdm.laplacian(u)

        Args:
            u: Function values at collocation points, shape (n_points,)
            use_qp: Override QP constraint behavior for this call.
                None: Use solver's qp_optimization_level setting
                True: Force QP constraints at all points
                False: Disable QP constraints for this call

        Returns:
            Dictionary mapping point index to derivative dictionary.
            derivatives[i] = {(1,): du/dx, (2,): d²u/dx², ...} for 1D
            derivatives[i] = {(1,0): du/dx, (0,1): du/dy, (2,0): d²u/dx², ...} for 2D
        """
        # Optionally override QP level for this computation
        saved_qp_level = None
        if use_qp is not None:
            saved_qp_level = self.qp_optimization_level
            self.qp_optimization_level = "always" if use_qp else "none"

        try:
            all_derivatives: dict[int, dict[tuple[int, ...], float]] = {}
            for i in range(self.n_points):
                all_derivatives[i] = self.approximate_derivatives(u, i)
            return all_derivatives
        finally:
            # Restore QP level if overridden
            if saved_qp_level is not None:
                self.qp_optimization_level = saved_qp_level

    # Note: QP methods moved to MonotonicityEnforcer component (Issue #545)
    # - solve_constrained_qp() (was _solve_monotone_constrained_qp)
    # - _solve_unconstrained_fallback()
    # - check_monotonicity_violation() (was _check_monotonicity_violation)
    # - check_m_matrix() (was _check_m_matrix_property)
    # - print_diagnostics() (was print_qp_diagnostics)
    # - compute_fd_weights_from_taylor() (was _compute_fd_weights_from_taylor)

    def _approximate_all_derivatives_cached(self, u: np.ndarray) -> dict[int, dict[tuple[int, ...], float]]:
        """Compute all derivatives at once (for caching between residual/Jacobian)."""
        all_derivs: dict[int, dict[tuple[int, ...], float]] = {}
        for i in range(self.n_points):
            all_derivs[i] = self.approximate_derivatives(u, i)
        return all_derivs

    def _compute_hjb_residual_vectorized(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        grad_u: np.ndarray,
        lap_u: np.ndarray,
        running_cost: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute HJB residual using vectorized operations (LQ fast path).

        This is the vectorized LQ fast path, only active for non-custom problems
        (is_custom=False) without QP monotonicity enforcement. It assumes:
        - Quadratic control cost: H_control = |p|^2 / (2*lambda)
        - Linear density coupling: f(m) = gamma * |Omega| * m
        - Static potential: V(x) from problem.f_potential (NOT from Hamiltonian class)

        For custom problems with a Hamiltonian class, the per-point path
        _compute_hjb_residual_with_cache() is used instead. See Issue #766.

        Args:
            u_current: Current solution at collocation points
            u_n_plus_1: Solution at next time step
            m_n_plus_1: Density at collocation points
            grad_u: Pre-computed gradient, shape (n_points, dimension)
            lap_u: Pre-computed Laplacian, shape (n_points,)

        Returns:
            Residual vector, shape (n_points,)
        """
        dt = self.problem.T / self.problem.Nt
        u_t = (u_n_plus_1 - u_current) / dt

        # Compute Hamiltonian for all points (vectorized LQ formula)
        # Two modes supported:
        # - additive:       H = |p|²/(2λ) + V + γm (standard separable form)
        # - multiplicative: H = (1 + γ|Ω|m)|p|²/(2λ) + V (velocity reduction by congestion)
        lambda_val = self._get_lambda_value()
        gamma_val = getattr(self.problem, "gamma", 0.0)

        # |grad_u|^2 for all points
        grad_norm_sq = np.sum(grad_u**2, axis=1)

        # Potential term (optional)
        f_potential = getattr(self.problem, "f_potential", None)
        if f_potential is not None:
            # Need to interpolate potential to collocation points
            H_potential = self._interpolate_potential_to_collocation()
        else:
            H_potential = np.zeros(self.n_points)

        # Congestion mode determines how density couples with velocity
        if self.congestion_mode == "multiplicative":
            # Multiplicative (congestion aversion): H = |p|²/(2λ(1 + γ|Ω|m))
            #
            # Legendre transform gives Lagrangian: L = (λ/2)(1 + γ|Ω|m)|v|²
            # - High density m → HIGH running cost (congestion aversion)
            # - Optimal velocity v* = -∇u/[λ(1 + γ|Ω|m)] → slower in crowds
            #
            # The |Ω| normalization makes γ dimensionless and O(1)
            domain_volume = self._compute_domain_volume()
            congestion_factor = 1.0 + gamma_val * domain_volume * m_n_plus_1
            H_kinetic = grad_norm_sq / (2 * lambda_val * congestion_factor)
            H_total = H_kinetic + H_potential
        else:
            # Additive: density-dependent running cost in HJB equation
            # HJB: -∂u/∂t + |∇u|²/(2λ) + γ|Ω|m - (σ²/2)Δu = 0
            #
            # The +γ|Ω|m term increases u in high-density regions,
            # which agents minimize → congestion avoidance (queuing cost)
            # The |Ω| normalization makes γ dimensionless and O(1)
            domain_volume = self._compute_domain_volume()
            H_kinetic = grad_norm_sq / (2 * lambda_val)
            H_interaction = gamma_val * domain_volume * m_n_plus_1
            H_total = H_kinetic + H_potential + H_interaction

        # Running cost L(x) at this timestep (passed explicitly from backward loop)
        if running_cost is not None:
            H_total = H_total + running_cost

        # Diffusion term: (sigma^2 / 2) * Laplacian
        # Issue #1073: use _get_sigma_value (returns σ) instead of confusing
        # `getattr(problem, "diffusion") or getattr(problem, "sigma")` chain.
        # `problem.diffusion` returns σ²/2 (PDE coefficient D), so the old chain
        # treated σ as D, then computed (σ²/2)² = σ⁴/8 instead of σ²/2 — 4-44× too
        # small for σ ∈ {0.3, 0.5, 1.0, 1.414} (only σ=2 happens to be correct).
        sigma = self._get_sigma_value(None)
        diffusion_term = 0.5 * sigma**2 * lap_u

        # HJB residual: -u_t + H - diffusion = 0
        residual = -u_t + H_total - diffusion_term

        return residual

    def _compute_hjb_residual_hamiltonian(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        grad_u: np.ndarray,
        lap_u: np.ndarray,
        H_class: Any,
        current_time: float,
        running_cost: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute HJB residual using batch Hamiltonian class (Issue #775).

        Uses H_class(x, m, p, t) for vectorized evaluation over all collocation
        points. Works with any HamiltonianBase subclass.

        Args:
            u_current: Current solution at collocation points, shape (n_points,)
            u_n_plus_1: Solution at next time step, shape (n_points,)
            m_n_plus_1: Density at collocation points, shape (n_points,)
            grad_u: Pre-computed gradient, shape (n_points, dimension)
            lap_u: Pre-computed Laplacian, shape (n_points,)
            H_class: HamiltonianBase instance with batch-polymorphic __call__
            current_time: Current time value for H(x, m, p, t)

        Returns:
            Residual vector, shape (n_points,)
        """
        dt = self.problem.T / self.problem.Nt
        u_t = (u_n_plus_1 - u_current) / dt

        # Batch Hamiltonian evaluation: H(x, m, p, t) -> (N,)
        x = self.collocation_points  # (N, d)
        H_total = np.asarray(H_class(x, m_n_plus_1, grad_u, t=current_time), dtype=float)

        # Running cost L(x) at this timestep (passed explicitly from backward loop)
        if running_cost is not None:
            H_total = H_total + running_cost

        # Diffusion term: (sigma^2 / 2) * Laplacian
        # Issue #1073: use _get_sigma_value (returns σ) instead of confusing
        # `getattr(problem, "diffusion") or getattr(problem, "sigma")` chain.
        # `problem.diffusion` returns σ²/2 (PDE coefficient D), so the old chain
        # treated σ as D, then computed (σ²/2)² = σ⁴/8 instead of σ²/2 — 4-44× too
        # small for σ ∈ {0.3, 0.5, 1.0, 1.414} (only σ=2 happens to be correct).
        sigma = self._get_sigma_value(None)
        diffusion_term = 0.5 * sigma**2 * lap_u

        # HJB residual: -u_t + H - diffusion = 0
        return -u_t + H_total - diffusion_term

    def _compute_hjb_jacobian_hamiltonian(
        self,
        grad_u: np.ndarray,
        m_n_plus_1: np.ndarray,
        H_class: Any,
        current_time: float,
    ):
        """
        Compute sparse Jacobian using batch H.dp() (Issue #775).

        Uses H_class.dp(x, m, p, t) for vectorized dH/dp computation.
        Jacobian structure: J = (1/dt)I + sum_d diag(dH/dp_d) @ D_grad[d] - (sigma^2/2) D_lap

        Args:
            grad_u: Pre-computed gradient, shape (n_points, dimension)
            m_n_plus_1: Density at collocation points, shape (n_points,)
            H_class: HamiltonianBase instance with batch-polymorphic dp()
            current_time: Current time value for H.dp(x, m, p, t)

        Returns:
            Sparse Jacobian matrix in CSR format
        """
        from scipy.sparse import diags, eye

        # Lazy initialization of differentiation matrices
        if self._D_grad is None:
            self._build_differentiation_matrices()

        n = self.n_points
        d = self.dimension
        dt = self.problem.T / self.problem.Nt
        # Issue #1073: use _get_sigma_value (returns σ) instead of confusing
        # `getattr(problem, "diffusion") or getattr(problem, "sigma")` chain.
        # `problem.diffusion` returns σ²/2 (PDE coefficient D), so the old chain
        # treated σ as D, then computed (σ²/2)² = σ⁴/8 instead of σ²/2 — 4-44× too
        # small for σ ∈ {0.3, 0.5, 1.0, 1.414} (only σ=2 happens to be correct).
        sigma = self._get_sigma_value(None)

        # Batch dH/dp: shape (N, d)
        x = self.collocation_points
        dH_dp = np.asarray(H_class.dp(x, m_n_plus_1, grad_u, t=current_time), dtype=float)

        # J = (1/dt)I + sum_d diag(dH/dp_d) @ D_grad[d] - (sigma^2/2) D_lap
        jacobian = (1.0 / dt) * eye(n, format="csr")
        for dim in range(d):
            jacobian = jacobian + diags(dH_dp[:, dim], format="csr") @ self._D_grad[dim]
        jacobian = jacobian - 0.5 * sigma**2 * self._D_lap

        return jacobian

    def _interpolate_potential_to_collocation(self) -> np.ndarray:
        """
        Interpolate potential field to collocation points (cached).

        Only used by the legacy vectorized LQ path (no hamiltonian_class).
        For problems with a Hamiltonian class, the potential V(x) comes from
        the Hamiltonian via H(x, m, p, t). See Issue #766, #775.

        Handles arbitrary dimensions by building grid axes from bounds.
        """
        # Return cached value if already computed
        # self._potential_at_collocation initialized as None in __init__
        if self._potential_at_collocation is not None:
            return self._potential_at_collocation

        f_potential = getattr(self.problem, "f_potential", None)
        if f_potential is None:
            self._potential_at_collocation = np.zeros(self.n_points)
            return self._potential_at_collocation

        from scipy.interpolate import RegularGridInterpolator

        potential = self.problem.f_potential
        bounds = self.domain_bounds

        # Validate potential shape matches dimension
        if potential.ndim != self.dimension:
            raise ValueError(
                f"Potential array has {potential.ndim} dimensions but problem is "
                f"{self.dimension}D. Potential shape: {potential.shape}"
            )

        # Build grid axes for each dimension
        axes = []
        for d in range(self.dimension):
            xmin, xmax = bounds[d]
            axes.append(np.linspace(xmin, xmax, potential.shape[d]))

        # Handle 1D special case (needs flattening)
        if self.dimension == 1:
            potential = potential.flatten()

        # Create interpolator and evaluate at collocation points
        interp = RegularGridInterpolator(tuple(axes), potential, bounds_error=False, fill_value=0.0)
        self._potential_at_collocation = interp(self.collocation_points)

        return self._potential_at_collocation

    def _compute_hjb_residual_with_cache(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
        cached_derivs: dict[int, dict[tuple[int, ...], float]],
        running_cost: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute HJB residual using pre-computed derivatives (per-point path).

        Per-point path via problem.H(). The potential V(x) comes from the
        Hamiltonian class (e.g., SeparableHamiltonian._potential), NOT from
        problem.f_potential. Active for custom problems (is_custom=True) or
        when QP monotonicity is enabled. See Issue #766.
        """
        from mfgarchon.core.derivatives import from_multi_index_dict

        residual = np.zeros(self.n_points)
        dimension = self.problem.dimension
        dt = self.problem.T / self.problem.Nt
        u_t = (u_n_plus_1 - u_current) / dt

        for i in range(self.n_points):
            x_pos = self.collocation_points[i]
            derivs = cached_derivs[i]
            p_derivs = from_multi_index_dict(derivs, dimension=dimension)
            laplacian = p_derivs.laplacian or 0.0

            H = self.problem.H(i, m_n_plus_1[i], derivs=p_derivs, x_position=x_pos)

            # Running cost L(x) at this timestep (passed explicitly from backward loop)
            if running_cost is not None:
                H = H + running_cost[i]

            sigma_val = self._get_sigma_value(i)
            diffusion_term = 0.5 * sigma_val**2 * laplacian
            residual[i] = -u_t[i] + H - diffusion_term

        return residual

    def _compute_hjb_jacobian_vectorized(
        self,
        grad_u: np.ndarray,
    ):
        """
        Compute sparse Jacobian using vectorized operations with pre-computed differentiation matrices.

        For standard LQ Hamiltonian H = |p|²/(2λ), dH/dp = p/λ.
        Jacobian structure: J = (1/dt)I + (1/λ) * Σ_d diag(p_d) @ D_grad[d] - (σ²/2) * D_lap

        Args:
            grad_u: Pre-computed gradient, shape (n_points, dimension)

        Returns:
            Sparse Jacobian matrix in CSR format
        """
        from scipy.sparse import diags, eye

        # Lazy initialization of differentiation matrices
        if self._D_grad is None:
            self._build_differentiation_matrices()

        n = self.n_points
        d = self.dimension
        dt = self.problem.T / self.problem.Nt
        lambda_val = self._get_lambda_value()
        # Issue #1073: use _get_sigma_value (returns σ) instead of confusing
        # `getattr(problem, "diffusion") or getattr(problem, "sigma")` chain.
        # `problem.diffusion` returns σ²/2 (PDE coefficient D), so the old chain
        # treated σ as D, then computed (σ²/2)² = σ⁴/8 instead of σ²/2 — 4-44× too
        # small for σ ∈ {0.3, 0.5, 1.0, 1.414} (only σ=2 happens to be correct).
        sigma = self._get_sigma_value(None)
        diffusion_coeff = 0.5 * sigma**2

        # Time derivative term: (1/dt) * I
        jacobian = (1.0 / dt) * eye(n, format="csr")

        # Hamiltonian gradient term: (1/λ) * Σ_d diag(p_d) @ D_grad[d]
        # For LQ: dH/dp = p/λ, so ∂(dH/dp · ∇u)/∂u_j = (p/λ) · (∂∇u/∂u_j)
        for dim in range(d):
            p_d = grad_u[:, dim] / lambda_val  # dH/dp_d = p_d / λ
            jacobian = jacobian + diags(p_d, format="csr") @ self._D_grad[dim]

        # Diffusion term: -(σ²/2) * D_lap
        jacobian = jacobian - diffusion_coeff * self._D_lap

        return jacobian

    def _compute_hjb_jacobian_sparse(
        self,
        u_current: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
        cached_derivs: dict[int, dict[tuple[int, ...], float]],
    ):
        """Compute sparse Jacobian using pre-computed derivatives and GFDM weights."""
        from scipy.sparse import lil_matrix

        from mfgarchon.core.derivatives import from_multi_index_dict, to_multi_index_dict

        n = self.n_points
        d = self.problem.dimension
        dt = self.problem.T / self.problem.Nt

        # Lazy initialization: Pre-cache all derivative weights (expensive computation)
        # self._cached_derivative_weights initialized as None in __init__
        #
        # Override precedence (joint SOCP > M-matrix QP > default operator weights):
        # this keeps the per-point HJB Jacobian consistent with the differentiation
        # matrices `_D_lap` / `_D_grad` (which apply the same precedence in
        # `_build_derivative_matrices`). v3 research code achieved this by monkey-
        # patching `_gfdm_operator.get_derivative_weights`; the precomputed-stencil
        # path replaces that hack with explicit dispatch here.
        if self._cached_derivative_weights is None:
            self._cached_derivative_weights = [self._gfdm_operator.get_derivative_weights(i) for i in range(n)]
            if self._joint_socp_stencils is not None:
                for i in range(n):
                    if self._joint_socp_stencils.has_stencil(i):
                        socp_w = self._joint_socp_stencils.get_weights_dict(i)
                        if socp_w is not None:
                            self._cached_derivative_weights[i] = socp_w
            if self._precomputed_stencils is not None:
                for i in range(n):
                    if self._precomputed_stencils.has_stencil(i):
                        pre = self._precomputed_stencils.get_laplacian_weights(i)
                        base = self._cached_derivative_weights[i]
                        if pre is not None and base is not None:
                            base["lap_weights"] = pre[0]

        # Use LIL format for efficient construction
        jacobian = lil_matrix((n, n))

        for i in range(n):
            weights = self._cached_derivative_weights[i]
            if weights is None:
                jacobian[i, i] = 1.0 / dt  # Fallback: identity
                continue

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]
            lap_weights = weights["lap_weights"]

            p_derivs = from_multi_index_dict(cached_derivs[i], dimension=d)

            dH_dp = self.problem.dH_dp(
                x_idx=i,
                m_at_x=m_n_plus_1[i],
                derivs=to_multi_index_dict(p_derivs),
                t_idx=time_idx,
                x_position=self.collocation_points[i],  # Pass actual position for GFDM
            )
            if dH_dp is None:
                dH_dp = self._compute_dH_dp_fd(i, m_n_plus_1[i], p_derivs, time_idx)

            sigma_val = self._get_sigma_value(i)
            diffusion_coeff = 0.5 * sigma_val**2

            # Neighbor contributions
            for k, j in enumerate(neighbor_indices):
                if j < 0:
                    continue  # Skip ghost particles
                val = np.dot(dH_dp, grad_weights[:, k]) - diffusion_coeff * lap_weights[k]
                jacobian[i, j] = val

            # Center point contribution
            center_grad_weight = -np.sum(grad_weights, axis=1)
            center_lap_weight = -np.sum(lap_weights)
            jacobian[i, i] += np.dot(dH_dp, center_grad_weight) - diffusion_coeff * center_lap_weight
            jacobian[i, i] += 1.0 / dt  # Time derivative

        return jacobian.tocsr()

    def _apply_boundary_conditions_to_sparse_system(self, jacobian_sparse, residual: np.ndarray, time_idx: int):
        """Apply boundary conditions to the sparse Jacobian via row replacement.

        Two dispatch paths:

        - **Mixed BC (per-point)**: every boundary point has been pre-classified
          at solver __init__ time to a (BCSegment, outward_normal) pair stored in
          ``self._bc_segment_per_point`` and ``self._bc_normal_per_point``. This
          method consumes those maps for O(1) lookup. If a point is missing from
          the map (only possible if pre-classification raised), we get a KeyError
          rather than a silent zero-row.

        - **Uniform BC**: legacy fast path using ``_bc_config["type"]`` and
          ``_bc_config["normals"]`` arrays indexed by ``local_idx``.

        BC type dispatch is now an exhaustive ``match`` over ``BCType`` with
        ``case _: raise``, so any unhandled enum value surfaces immediately
        rather than silently leaving a cleared zero row. PERIODIC and ROBIN
        raise ``NotImplementedError`` because this solver doesn't support them
        via row replacement (see error messages for alternatives).

        Row construction is **atomic**: we build the full replacement row in a
        local ``np.ndarray`` and assign in one shot, instead of clearing first
        and then conditionally refilling.
        """
        if len(self.boundary_indices) == 0:
            return jacobian_sparse, residual

        jac_lil = jacobian_sparse.tolil()
        residual_bc = residual.copy()

        try:
            use_per_point_bc = self.boundary_conditions.is_mixed
        except AttributeError:
            use_per_point_bc = False

        # Legacy uniform-BC scaffold
        global_bc_type = self._get_boundary_condition_property("type") if not use_per_point_bc else None
        legacy_bc_values = self._get_boundary_condition_property("values") if not use_per_point_bc else None
        legacy_normals = self._bc_config.get("normals", None) if not use_per_point_bc and self._bc_config else None
        bc_str_to_enum = {
            "dirichlet": BCType.DIRICHLET,
            "neumann": BCType.NEUMANN,
            "no_flux": BCType.NO_FLUX,
            "periodic": BCType.PERIODIC,
            "robin": BCType.ROBIN,
        }

        dimension = self.dimension
        n = self.n_points
        # Time coordinate for callable BC values
        current_time = time_idx * (self.problem.T / self.problem.Nt) if getattr(self.problem, "Nt", 0) > 0 else 0.0

        for local_idx, i in enumerate(self.boundary_indices):
            i = int(i)

            # --- Resolve BC for this point ---
            if use_per_point_bc:
                segment = self._bc_segment_per_point[i]
                bc_enum = segment.bc_type
                normal = self._bc_normal_per_point[i]
            else:
                bc_str = (global_bc_type or "neumann").lower()
                if bc_str not in bc_str_to_enum:
                    raise ValueError(
                        f"Unknown BC type {bc_str!r} at boundary point {i} (uniform path). "
                        f"Supported: {tuple(bc_str_to_enum)}."
                    )
                bc_enum = bc_str_to_enum[bc_str]
                segment = None
                if legacy_normals is not None and local_idx < len(legacy_normals):
                    normal = legacy_normals[local_idx]
                elif self._boundary_handler is not None:
                    normal = self._boundary_handler.compute_outward_normal(i)
                else:
                    normal = self._compute_outward_normal(i)

            # --- Ghost-nodes structural BC: keep PDE row intact ---
            if bc_enum in (BCType.NEUMANN, BCType.NO_FLUX):
                if (
                    self._use_ghost_nodes
                    and self._boundary_handler is not None
                    and i in self._boundary_handler.ghost_node_map
                ):
                    # Symmetric ghost stencils enforce BC structurally; leave row.
                    continue

            # --- Build replacement row + rhs (atomic) ---
            match bc_enum:
                case BCType.DIRICHLET:
                    new_row = np.zeros(n)
                    new_row[i] = 1.0
                    new_rhs = self._eval_bc_dirichlet_value(i, segment, legacy_bc_values, current_time)
                case BCType.NEUMANN | BCType.NO_FLUX:
                    new_row, new_rhs = self._build_neumann_bc_row(
                        i, normal, dimension, segment, legacy_bc_values, current_time
                    )
                case BCType.PERIODIC:
                    raise NotImplementedError(
                        f"PERIODIC BC at boundary point {i} not supported by HJBGFDMSolver "
                        f"via row replacement. Use TensorProductGrid + FDM for periodic "
                        f"geometries, or rephrase as paired Dirichlet/Neumann segments."
                    )
                case BCType.ROBIN:
                    raise NotImplementedError(
                        f"ROBIN BC at boundary point {i} not supported by HJBGFDMSolver "
                        f"via row replacement. Use the BCValueProvider pattern in the "
                        f"coupling layer (Issue #625, see AdjointConsistentProvider)."
                    )
                case _:
                    raise ValueError(
                        f"Unhandled BCType {bc_enum!r} at boundary point {i}. "
                        f"This indicates a new BCType value not yet wired into "
                        f"HJBGFDMSolver._apply_boundary_conditions_to_sparse_system."
                    )

            # --- Atomic row replacement ---
            jac_lil[i, :] = new_row
            residual_bc[i] = new_rhs

        return jac_lil.tocsr(), residual_bc

    def _eval_bc_dirichlet_value(
        self,
        point_idx: int,
        segment: BCSegment | None,
        legacy_bc_values,
        current_time: float,
    ) -> float:
        """Resolve a Dirichlet RHS value for boundary point ``point_idx``.

        Prefers ``segment.get_value(point, t)`` when a segment is supplied
        (pre-classified path); falls back to legacy ``bc_values`` (dict,
        callable, or scalar) for the uniform-BC path.
        """
        if segment is not None:
            return float(segment.get_value(self.collocation_points[point_idx], t=current_time))
        if isinstance(legacy_bc_values, dict):
            return float(legacy_bc_values.get(point_idx, 0.0))
        if callable(legacy_bc_values):
            return float(legacy_bc_values(self.collocation_points[point_idx]))
        return float(legacy_bc_values) if legacy_bc_values else 0.0

    def _build_neumann_bc_row(
        self,
        point_idx: int,
        normal: np.ndarray,
        dimension: int,
        segment: BCSegment | None,
        legacy_bc_values,
        current_time: float,
    ) -> tuple[np.ndarray, float]:
        """Build the (row, rhs) for a Neumann / no-flux BC at ``point_idx``.

        Row encodes ``normal · grad(u) ≈ Σ_j (normal · grad_weights[:,k]) u_j``,
        so the linear system row is ``[..., w_j, ..., center_weight, ...]``.
        RHS is the prescribed normal-derivative value (0 for no-flux).

        LCR (local coordinate rotation, Issue #531) and ghost-node paths
        choose the gradient stencil; ghost-node short-circuit happens in the
        caller before this is invoked.
        """
        if (
            self._use_local_coordinate_rotation
            and self._boundary_handler is not None
            and point_idx in self._boundary_handler.boundary_rotations
        ):
            if self._neighborhood_builder is not None:
                boundary_rotations = self._boundary_handler.boundary_rotations
                weights = self._neighborhood_builder.compute_derivative_weights_from_taylor(
                    point_idx, boundary_rotations
                )
            else:
                weights = self._compute_derivative_weights_from_taylor(point_idx)
        else:
            weights = self._gfdm_operator.get_derivative_weights(point_idx)

        n = self.n_points
        new_row = np.zeros(n)

        if weights is None:
            # Degenerate stencil — preserve legacy behavior: pin to identity
            # row with zero RHS rather than producing a zero row. A warning
            # might be more honest, but matches existing semantics.
            new_row[point_idx] = 1.0
            return new_row, 0.0

        neighbor_indices = weights["neighbor_indices"]
        grad_weights = weights["grad_weights"]

        center_weight = 0.0
        for k, j in enumerate(neighbor_indices):
            if j >= 0 and j != point_idx:
                w = sum(normal[d] * grad_weights[d, k] for d in range(dimension))
                new_row[j] = w
                center_weight -= w
        new_row[point_idx] = center_weight

        # RHS: segment.get_value at this point, or legacy dict lookup, else 0
        if segment is not None:
            rhs = float(segment.get_value(self.collocation_points[point_idx], t=current_time))
        elif isinstance(legacy_bc_values, dict):
            rhs = float(legacy_bc_values.get(point_idx, 0.0))
        else:
            rhs = 0.0

        return new_row, rhs

    def _get_lambda_value(self) -> float:
        """
        Get control cost parameter lambda with validation.

        Returns:
            Positive lambda value

        Raises:
            ValueError: If lambda <= 0

        Notes:
            Lambda appears in the Hamiltonian as H = |p|²/(2λ) + ...
            Division by lambda requires λ > 0.
        """
        lambda_val = getattr(self.problem, "lambda_", 1.0)
        if lambda_val is None:
            lambda_val = 1.0
        if lambda_val <= 0:
            raise ValueError(
                f"Control cost parameter lambda_ must be positive, got {lambda_val}. "
                f"Set problem.lambda_ to a positive value."
            )
        return float(lambda_val)

    def _get_sigma_value(self, point_idx: int | None = None) -> float:
        """
        Get diffusion coefficient value, handling both numeric and callable sigma.

        Args:
            point_idx: Collocation point index (for callable sigma evaluation)

        Returns:
            Numeric sigma value

        Handles three cases:
        1. problem.nu exists (legacy attribute)
        2. problem.sigma is callable → evaluate at collocation point
        3. problem.sigma is numeric → use directly (fallback: 1.0)
        """
        # Check for legacy "nu" attribute (optional)
        nu = getattr(self.problem, "nu", None)
        if nu is not None:
            return float(nu)

        sigma = getattr(self.problem, "sigma", None)
        if callable(sigma):
            # Callable sigma: evaluate at current point if available
            if point_idx is not None and point_idx < len(self.collocation_points):
                x = self.collocation_points[point_idx]
                return float(self.problem.sigma(x))
            else:
                # Fallback: use representative value (center of domain)
                return 1.0
        else:
            # Numeric sigma: use directly (with fallback to default)
            return float(getattr(self.problem, "sigma", 1.0))

    # Note: _check_monotonicity_violation moved to MonotonicityEnforcer component
    # Note: _check_m_matrix_property moved to MonotonicityEnforcer component
    # Note: _build_monotonicity_constraints moved to MonotonicityEnforcer component
    # Note: _build_hamiltonian_gradient_constraints moved to MonotonicityEnforcer component

    @deprecated_parameter(
        param_name="M_density_evolution_from_FP",
        since="v0.17.0",
        replacement="M_density",
    )
    @deprecated_parameter(
        param_name="U_final_condition_at_T",
        since="v0.17.0",
        replacement="U_terminal",
    )
    @deprecated_parameter(
        param_name="U_from_prev_picard",
        since="v0.17.0",
        replacement="U_coupling_prev",
    )
    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        show_progress: bool | None = None,
        volatility_field: float | np.ndarray | None = None,
        running_cost: np.ndarray | Callable[[int], np.ndarray] | None = None,
        # Deprecated parameter names for backward compatibility
        M_density_evolution_from_FP: np.ndarray | None = None,
        U_final_condition_at_T: np.ndarray | None = None,
        U_from_prev_picard: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the HJB system using GFDM collocation method.

        Args:
            M_density: (Nt, *spatial_shape) density from FP solver
            U_terminal: (*spatial_shape,) terminal condition u(T,x)
            U_coupling_prev: (Nt, *spatial_shape) previous coupling iteration estimate
            show_progress: Whether to display progress bar for timesteps
            running_cost: Running cost L(x) or L(t,x) at collocation points.
                Static array: shape (n_points,) -- same cost at every backward step.
                Time-dependent array: shape (n_time_points, n_points) -- L(t_n, x) per step.
                Callable: f(time_index) -> (n_points,) array, evaluated per step.
                Added to Hamiltonian: H_total = H(x,p,m) + L(t,x).
            volatility_field: Optional diffusion coefficient override

        Returns:
            (Nt, *spatial_shape) solution array
        """
        # Handle deprecated parameter names (warnings issued by @deprecated_parameter decorators)
        if M_density_evolution_from_FP is not None:
            if M_density is not None:
                raise ValueError("Cannot specify both 'M_density' and deprecated 'M_density_evolution_from_FP'")
            M_density = M_density_evolution_from_FP

        if U_final_condition_at_T is not None:
            if U_terminal is not None:
                raise ValueError("Cannot specify both 'U_terminal' and deprecated 'U_final_condition_at_T'")
            U_terminal = U_final_condition_at_T

        if U_from_prev_picard is not None:
            if U_coupling_prev is not None:
                raise ValueError("Cannot specify both 'U_coupling_prev' and deprecated 'U_from_prev_picard'")
            U_coupling_prev = U_from_prev_picard

        # Validate required parameters
        if U_terminal is None:
            raise ValueError("U_terminal is required")

        # Validate BC specification if boundary points exist
        if len(self.boundary_indices) > 0 and (self._bc_config is None or self._bc_config.get("type") is None):
            raise ValueError(
                f"Boundary conditions required for solving but not specified. "
                f"Found {len(self.boundary_indices)} boundary points. "
                f"Pass boundary_conditions parameter to solver or set BC on problem.geometry."
            )

        # Determine n_time_points from available data or problem configuration
        # n_time_points = Nt + 1 (number of time knots including t=0 and t=T)
        if M_density is not None:
            n_time_points = M_density.shape[0]
        else:
            n_time_points = self.problem.Nt + 1

        # For standalone HJB (no MFG coupling), use defaults
        if M_density is None:
            # Default: uniform density (no coupling effect)
            M_density = np.ones((n_time_points, *U_terminal.shape))
        if U_coupling_prev is None:
            # Default: zero coupling (pure HJB)
            U_coupling_prev = np.zeros((n_time_points, *U_terminal.shape))

        # Store original spatial shape for reshaping output
        self._output_spatial_shape = M_density.shape[1:]

        # Normalize running cost to callable f(n) -> (n_points,)
        # Accepts: None, 1D array, 2D array, or callable
        self._running_cost_fn = self._normalize_running_cost(running_cost, n_time_points)

        # Detect if input is already in collocation format (pure meshfree mode)
        # Grid format: M_density.shape = (Nt, Nx, Ny, ...)
        # Collocation format: M_density.shape = (Nt, n_points)
        is_meshfree_input = M_density.ndim == 2 and M_density.shape[1] == self.n_points

        # For GFDM, we work directly with collocation points
        U_solution_collocation = np.zeros((n_time_points, self.n_points))

        if is_meshfree_input:
            # Pure meshfree mode: input already at collocation points
            M_collocation = M_density.copy()
            # U_terminal should also be at collocation points
            U_solution_collocation[n_time_points - 1, :] = U_terminal.copy()
        else:
            # Hybrid mode: map grid data to collocation points
            M_collocation = self._mapper.map_grid_to_collocation_batch(M_density)
            # Set final condition at t=T (last time index = n_time_points - 1)
            U_solution_collocation[n_time_points - 1, :] = self._mapper.map_grid_to_collocation(U_terminal.flatten())

        # Backward time stepping: Nt steps from index (n_time_points-2) down to 0
        # This covers all Nt intervals in the backward direction (Issue #587 Protocol pattern)
        from mfgarchon.utils.progress import create_progress_bar, should_show_progress

        timestep_range = create_progress_bar(
            range(n_time_points - 2, -1, -1),
            verbose=should_show_progress(show_progress),
            desc="HJB (backward)",
        )

        for n in timestep_range:
            rc_n = self._running_cost_fn(n) if self._running_cost_fn is not None else None

            U_solution_collocation[n, :] = self._solve_timestep(
                U_solution_collocation[n + 1, :],
                M_collocation[n, :],  # FIXED: Use m^n, not m^{n+1} (same-time coupling)
                n,
                running_cost=rc_n,
            )

            # Update progress bar with QP statistics if available (Issue #587 Protocol - no hasattr needed)
            if self.qp_optimization_level in ["auto", "always"]:
                timestep_range.update_metrics(qp_solves=self.qp_stats.get("total_qp_solves", 0))

        # Return format depends on input mode
        if is_meshfree_input:
            # Pure meshfree: return collocation data directly
            return U_solution_collocation
        else:
            # Hybrid mode: map back to grid
            U_solution = self._mapper.map_collocation_to_grid_batch(U_solution_collocation)
            return U_solution

    def _solve_timestep(
        self,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
        running_cost: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve HJB at one time step using Newton iteration with backtracking line search.

        Globalization: each Newton iteration tries the full step `delta_u = -J⁻¹·r`,
        then accepts iff sufficient-decrease in residual norm holds (Armijo with
        c₁=1e-4). Otherwise halves α (geometric backtracking) until accepted or
        α drops below `min_alpha=1e-6`. Replaces the legacy hardcoded `max_step=10`
        cap, which prevented Newton from converging on stiff problems with
        |U|=O(100) (e.g., 2D MFG with strong potential — observed 0/150 timesteps
        converging to 1e-6 tolerance at high Pe).

        Reference: Nocedal-Wright "Numerical Optimization" §3.1 (Armijo condition).
        """
        from scipy.sparse.linalg import spsolve

        u_current = u_n_plus_1.copy()

        # Path selection for HJB residual/Jacobian computation (Issue #766, #775):
        #
        # 1. Hamiltonian batch path (use_hamiltonian_batch=True):
        #    Active when: hamiltonian_class is available AND qp_optimization_level="none"
        #    Uses batch H(x, m, p, t) and H.dp(x, m, p, t) from HamiltonianBase
        #    Works with any Hamiltonian subclass (SeparableHamiltonian, etc.)
        #    Fast: numpy vectorized over all collocation points at once
        #
        # 2. Legacy LQ vectorized path (use_legacy_vectorized=True):
        #    Fallback when: no hamiltonian_class AND is_custom=False AND no QP
        #    Hardcodes LQ formula: H = |p|^2/(2*lambda) + V(x) + gamma*|Omega|*m
        #    Gets V(x) from problem.f_potential via _interpolate_potential_to_collocation()
        #
        # 3. Per-point path (fallback):
        #    Active when: QP mode enabled OR (is_custom=True without hamiltonian_class)
        #    Calls problem.H() per-point -> loops over collocation points
        H_class = getattr(self.problem, "hamiltonian_class", None)
        is_custom = getattr(self.problem, "is_custom", False)
        use_hamiltonian_batch = H_class is not None and self.qp_optimization_level == "none"
        use_legacy_vectorized = not use_hamiltonian_batch and not is_custom and self.qp_optimization_level == "none"

        # Warn if f_potential is set but won't be used (Issue #766)
        if not use_legacy_vectorized and not self._f_potential_warned:
            f_pot = getattr(self.problem, "f_potential", None)
            if f_pot is not None and np.any(f_pot != 0):
                warnings.warn(
                    "f_potential is set but will be ignored because the per-point "
                    "Hamiltonian path is active (is_custom=True or QP mode). "
                    "The potential V(x) comes from the Hamiltonian class instead. "
                    "Use SeparableHamiltonian(potential=...) to set the potential. "
                    "See Issue #766.",
                    UserWarning,
                    stacklevel=2,
                )
                self._f_potential_warned = True

        # Compute actual time for batch Hamiltonian calls
        current_time = time_idx * (self.problem.T / self.problem.Nt)

        # Closure: compute residual norm at any candidate u_trial (used by
        # backtracking line search). Routes through the same path-selection as
        # the Newton step itself, so residual evaluation is consistent.
        def _residual_norm(u_trial: np.ndarray) -> float:
            if use_hamiltonian_batch:
                g_u, l_u = self._compute_derivatives_vectorized(u_trial)
                r = self._compute_hjb_residual_hamiltonian(
                    u_trial,
                    u_n_plus_1,
                    m_n_plus_1,
                    g_u,
                    l_u,
                    H_class,
                    current_time,
                    running_cost=running_cost,
                )
            elif use_legacy_vectorized:
                g_u, l_u = self._compute_derivatives_vectorized(u_trial)
                r = self._compute_hjb_residual_vectorized(
                    u_trial,
                    u_n_plus_1,
                    m_n_plus_1,
                    g_u,
                    l_u,
                    running_cost=running_cost,
                )
            else:
                derivs = self._approximate_all_derivatives_cached(u_trial)
                r = self._compute_hjb_residual_with_cache(
                    u_trial,
                    u_n_plus_1,
                    m_n_plus_1,
                    time_idx,
                    derivs,
                    running_cost=running_cost,
                )
            return float(np.linalg.norm(r))

        # Armijo backtracking parameters (Nocedal-Wright §3.1)
        ARMIJO_C1 = 1e-4  # sufficient-decrease constant
        BACKTRACK_FACTOR = 0.5  # geometric step reduction
        MIN_ALPHA = 1e-6  # give up below this α

        for _newton_iter in range(self.max_newton_iterations):
            if use_hamiltonian_batch:
                # Batch Hamiltonian path: H(x, m, p, t) vectorized (Issue #775)
                grad_u, lap_u = self._compute_derivatives_vectorized(u_current)

                residual = self._compute_hjb_residual_hamiltonian(
                    u_current,
                    u_n_plus_1,
                    m_n_plus_1,
                    grad_u,
                    lap_u,
                    H_class,
                    current_time,
                    running_cost=running_cost,
                )

                if np.linalg.norm(residual) < self.newton_tolerance:
                    break

                jacobian_sparse = self._compute_hjb_jacobian_hamiltonian(
                    grad_u,
                    m_n_plus_1,
                    H_class,
                    current_time,
                )
            elif use_legacy_vectorized:
                # Legacy LQ vectorized path (no hamiltonian_class available)
                grad_u, lap_u = self._compute_derivatives_vectorized(u_current)

                residual = self._compute_hjb_residual_vectorized(
                    u_current,
                    u_n_plus_1,
                    m_n_plus_1,
                    grad_u,
                    lap_u,
                    running_cost=running_cost,
                )

                if np.linalg.norm(residual) < self.newton_tolerance:
                    break

                jacobian_sparse = self._compute_hjb_jacobian_vectorized(grad_u)
            else:
                # Per-point path for QP mode or legacy custom without hamiltonian_class
                all_derivs = self._approximate_all_derivatives_cached(u_current)

                residual = self._compute_hjb_residual_with_cache(
                    u_current,
                    u_n_plus_1,
                    m_n_plus_1,
                    time_idx,
                    all_derivs,
                    running_cost=running_cost,
                )

                if np.linalg.norm(residual) < self.newton_tolerance:
                    break

                jacobian_sparse = self._compute_hjb_jacobian_sparse(u_current, m_n_plus_1, time_idx, all_derivs)

            # Apply boundary conditions (sparse-aware)
            jacobian_bc, residual_bc = self._apply_boundary_conditions_to_sparse_system(
                jacobian_sparse, residual, time_idx
            )

            # Newton update using sparse solver: solve J·δ = -r for δ
            try:
                delta_u = spsolve(jacobian_bc, -residual_bc)
            except Exception as e:
                # Fallback to dense solver
                logger.warning(f"Sparse solver failed in Newton iteration (using dense fallback): {e}")
                delta_u = np.linalg.lstsq(jacobian_bc.toarray(), -residual_bc, rcond=None)[0]

            # Backtracking line search (Armijo). The Newton direction `delta_u`
            # is a descent direction for `½‖r‖²`, but the natural step `α=1`
            # may overshoot on stiff/nonlinear problems. We search for the
            # largest α ∈ {1, 0.5, 0.25, ...} satisfying sufficient-decrease:
            #   ‖r(u + α·δ)‖² ≤ (1 − 2·c₁·α)·‖r(u)‖²
            # which simplifies (for descent direction) to
            #   ‖r(u + α·δ)‖ ≤ (1 − c₁·α)·‖r(u)‖
            # Replaces a hardcoded `max_step=10` cap that was too restrictive
            # for stiff problems with |U|=O(100) and too permissive elsewhere
            # (Issue: HJB Newton non-convergence at high Pe).
            r0_norm = float(np.linalg.norm(residual))
            alpha = 1.0
            u_trial = u_current + alpha * delta_u
            u_trial = self._apply_boundary_conditions_to_solution(u_trial, time_idx)
            r_trial_norm = _residual_norm(u_trial)
            # Guard against NaN/Inf from too-aggressive steps
            if not np.isfinite(r_trial_norm):
                r_trial_norm = float("inf")
            while r_trial_norm > (1.0 - ARMIJO_C1 * alpha) * r0_norm and alpha > MIN_ALPHA:
                alpha *= BACKTRACK_FACTOR
                u_trial = u_current + alpha * delta_u
                u_trial = self._apply_boundary_conditions_to_solution(u_trial, time_idx)
                r_trial_norm = _residual_norm(u_trial)
                if not np.isfinite(r_trial_norm):
                    r_trial_norm = float("inf")

            # Apply accepted update. If line search bottomed out (α<MIN_ALPHA)
            # we still apply the smallest tested step rather than zero, so
            # Newton makes some progress even when sufficient-decrease fails.
            u_current = u_trial

        return u_current

    def _compute_dH_dp_fd(
        self,
        point_idx: int,
        m_at_x: float,
        derivs: DerivativeTensors,
        time_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute dH/dp - analytical for standard LQ Hamiltonian, FD otherwise.

        For standard LQ Hamiltonian H = |∇u|²/(2λ), dH/dp = p/λ analytically.

        Args:
            point_idx: Collocation point index
            m_at_x: Density value at the point
            derivs: DerivativeTensors with current gradient/hessian
            time_idx: Time index (currently unused, kept for API compatibility)

        Returns:
            dH/dp array, shape (dim,)
        """
        p = derivs.grad if derivs.grad is not None else np.zeros(self.problem.dimension)

        # Fast path: for standard LQ Hamiltonian H = |p|²/(2λ), dH/dp = p/λ
        lambda_val = getattr(self.problem, "lambda_", None)
        if lambda_val is not None and lambda_val > 0:
            # Check if using standard (non-custom) Hamiltonian
            is_custom = getattr(self.problem, "is_custom", False)
            if not is_custom:
                return p / lambda_val

        # Fallback: finite differences for custom Hamiltonians using scipy
        x_pos = self.collocation_points[point_idx]
        hess = derivs.hess if derivs.hess is not None else np.zeros((len(p), len(p)))

        def H_of_p(p_vec: np.ndarray) -> float:
            """Hamiltonian as function of momentum p only."""
            from mfgarchon.core.derivatives import DerivativeTensors

            d = DerivativeTensors.from_arrays(grad=p_vec, hess=hess)
            return self.problem.H(point_idx, m_at_x, derivs=d, x_position=x_pos)

        # Use scipy's approx_fprime for gradient computation
        return approx_fprime(p, H_of_p, epsilon=1e-7)

    def _compute_hjb_jacobian_analytic(
        self,
        u_current: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """
        Compute Jacobian using analytic formula with GFDM weights.

        Formula: ∂R_i/∂u_j = (1/dt)δ_{ij} + (∂H/∂p)·(∂p_i/∂u_j) - (σ²/2)·(∂Δu_i/∂u_j)

        Uses user-provided dH_dp if available, otherwise FD on H.
        """
        from mfgarchon.core.derivatives import from_multi_index_dict, to_multi_index_dict

        n = self.n_points
        d = self.problem.dimension
        dt = self.problem.T / self.problem.Nt
        jacobian = np.zeros((n, n))

        for i in range(n):
            # Get derivative weights from GFDM
            weights = self._gfdm_operator.get_derivative_weights(i)
            if weights is None:
                continue  # Skip points without valid Taylor data

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]  # shape (d, n_neighbors)
            lap_weights = weights["lap_weights"]  # shape (n_neighbors,)

            # Compute derivatives at point i
            derivs_dict = self.approximate_derivatives(u_current, i)

            # Build DerivativeTensors using nD infrastructure
            p_derivs = from_multi_index_dict(derivs_dict, dimension=d)

            # Get ∂H/∂p (user-provided or FD fallback)
            dH_dp = self.problem.dH_dp(
                x_idx=i,
                m_at_x=m_n_plus_1[i],
                derivs=to_multi_index_dict(p_derivs),
                t_idx=time_idx,
                x_position=self.collocation_points[i],  # Pass actual position for GFDM
            )

            if dH_dp is None:
                # Fallback: compute via FD on H
                dH_dp = self._compute_dH_dp_fd(i, m_n_plus_1[i], p_derivs, time_idx)

            # Get sigma for diffusion term
            sigma_val = self._get_sigma_value(i)
            diffusion_coeff = 0.5 * sigma_val**2

            # Build row i of Jacobian
            # For neighbors: use GFDM weights directly
            for k, j in enumerate(neighbor_indices):
                if j < 0:
                    continue  # Skip ghost particles

                # ∂R_i/∂u_j = (∂H/∂p) · grad_weights[:, k] - diffusion * lap_weights[k]
                jacobian[i, j] = np.dot(dH_dp, grad_weights[:, k]) - diffusion_coeff * lap_weights[k]

            # For center point contribution: weights are -sum(row weights)
            # because b = u_neighbors - u_center, so ∂b/∂u_center = -1
            center_grad_weight = -np.sum(grad_weights, axis=1)
            center_lap_weight = -np.sum(lap_weights)
            jacobian[i, i] += np.dot(dH_dp, center_grad_weight) - diffusion_coeff * center_lap_weight

            # Time derivative contribution: (1/dt) on diagonal
            jacobian[i, i] += 1.0 / dt

        return jacobian

    def _compute_hjb_jacobian(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,  # FIXED: Added actual u_n_plus_1 parameter
        m_n_plus_1: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """
        Compute Jacobian matrix for Newton iteration.

        Uses analytic Jacobian if dH_dp is available (faster),
        otherwise falls back to finite differences on residual.

        Bug #15 fix: Disable QP in Jacobian computation to reduce QP calls from ~750k to ~7.5k.
        Jacobian only affects Newton convergence rate, not final monotonicity (enforced by residual).
        """
        # Analytic Jacobian uses GFDM weights directly - O(n·k) vs O(n²) for FD
        # Works for any dimension since GFDM weights are dimension-agnostic
        return self._compute_hjb_jacobian_analytic(u_current, m_n_plus_1, time_idx)

    def _apply_boundary_conditions_to_solution(self, u: np.ndarray, time_idx: int) -> np.ndarray:
        """Apply boundary conditions directly to solution array.

        For mixed BC (per-point types), enforces Dirichlet at exit points only.
        """
        if len(self.boundary_indices) == 0:
            return u

        # Check if using per-point BC (mixed BC)
        # Issue #527: Replace hasattr with try/except per CLAUDE.md guidelines
        try:
            use_per_point_bc = self.boundary_conditions.is_mixed
        except AttributeError:
            use_per_point_bc = False

        # Use unified BC config (single source of truth) when using new infrastructure
        if self._use_new_infrastructure and self._bc_config is not None:
            global_bc_type = self._bc_config["type"]
            bc_values = self._bc_config["values"]
        else:
            # Get BC type - will raise error if not specified
            bc_type_val = self._get_boundary_condition_property("type")
            global_bc_type = bc_type_val.lower() if isinstance(bc_type_val, str) else bc_type_val
            bc_values = self._get_boundary_condition_property("value")

        # For per-point BC, apply Dirichlet only at exit points
        if use_per_point_bc:
            for i in self.boundary_indices:
                bc_type = self._get_bc_type_for_point(i)
                if bc_type == "dirichlet":
                    if callable(bc_values):
                        current_time = self.problem.T * time_idx / self.problem.Nt
                        u[i] = bc_values(self.collocation_points[i], current_time)
                    else:
                        u[i] = float(bc_values) if bc_values else 0.0
        elif global_bc_type == "dirichlet":
            # Uniform Dirichlet: apply to all boundary points
            if callable(bc_values):
                current_time = self.problem.T * time_idx / self.problem.Nt
                for i in self.boundary_indices:
                    u[i] = bc_values(self.collocation_points[i], current_time)
            else:
                u[self.boundary_indices] = bc_values
        # For Neumann: no direct solution modification (enforced via residual)

        return u

    def _apply_boundary_conditions_to_system(
        self, jacobian: np.ndarray, residual: np.ndarray, time_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply boundary conditions to the linear system J·δu = -R.

        For Dirichlet BC: Set row to identity and residual to zero.
        For Neumann BC: Row Replacement with normal derivative operator.
        """
        if len(self.boundary_indices) == 0:
            return jacobian, residual

        jacobian_bc = jacobian.copy()
        residual_bc = residual.copy()

        # Use new infrastructure with DirectCollocationHandler
        if self._use_new_infrastructure and self._bc_handler is not None:
            self._bc_handler.apply_to_matrix(
                A=jacobian_bc,
                b=residual_bc,
                boundary_indices=self.boundary_indices,
                operator=self._gfdm_operator,
                bc_config=self._bc_config,
            )
            return jacobian_bc, residual_bc

        # Legacy path (deprecated) - BC required if boundary points exist
        bc_type_val = self._get_boundary_condition_property("type")
        bc_type = bc_type_val.lower() if isinstance(bc_type_val, str) else bc_type_val

        if bc_type == "dirichlet":
            for i in self.boundary_indices:
                jacobian_bc[i, :] = 0.0
                jacobian_bc[i, i] = 1.0
                residual_bc[i] = 0.0

        return jacobian_bc, residual_bc


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBGFDMSolver...")

    import numpy as np

    from mfgarchon import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    # Test 1D problem with uniform collocation points matching problem grid
    geometry_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21])
    problem_1d = MFGProblem(geometry=geometry_1d, T=1.0, Nt=10, sigma=0.1)

    # Use problem grid points as collocation points to avoid index mismatch
    collocation_points = problem_1d.xSpace.reshape(-1, 1)

    solver_1d = HJBGFDMSolver(
        problem_1d,
        collocation_points=collocation_points,
        delta=0.15,
        taylor_order=2,
        weight_function="wendland",
    )

    # Test solver initialization
    assert solver_1d.dimension == 1
    assert solver_1d.n_points == problem_1d.Nx + 1
    assert solver_1d.delta == 0.15
    assert solver_1d.taylor_order == 2
    assert solver_1d.hjb_method_name == "GFDM"
    print("  [1D] Solver initialized")
    print(f"       Collocation points: {solver_1d.n_points}, Delta: {solver_1d.delta}")

    # Test derivative computation API (1D)
    # f(x) = x^2 -> df/dx = 2x, d²f/dx² = 2
    x = collocation_points[:, 0]
    u_1d = x**2

    # Test compute_all_derivatives
    all_derivs_1d = solver_1d.compute_all_derivatives(u_1d)
    assert len(all_derivs_1d) == solver_1d.n_points
    # Interior points should have derivatives
    mid_idx = solver_1d.n_points // 2
    assert (1,) in all_derivs_1d[mid_idx], f"Missing gradient key (1,) at point {mid_idx}"
    print(f"  [1D] compute_all_derivatives: {len(all_derivs_1d)} points")
    print(f"       Multi-indices: {solver_1d.multi_indices}")

    # Test 2D problem
    print("\n  [2D] Testing 2D solver...")

    # Create 2D collocation points (grid)
    Nx_2d = 10
    x_grid = np.linspace(0, 1, Nx_2d)
    y_grid = np.linspace(0, 1, Nx_2d)
    xx, yy = np.meshgrid(x_grid, y_grid)
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])

    geometry_2d = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[Nx_2d, Nx_2d])
    problem_2d = MFGProblem(geometry=geometry_2d, T=1.0, Nt=5, sigma=0.1)

    solver_2d = HJBGFDMSolver(
        problem_2d,
        collocation_points=points_2d,
        delta=0.2,
        taylor_order=2,
        weight_function="wendland",
    )
    print(f"       Collocation points: {solver_2d.n_points}, Delta: {solver_2d.delta}")
    print(f"       Multi-indices: {solver_2d.multi_indices}")

    # f(x,y) = x² + y² -> gradient = [2x, 2y], laplacian = 4
    u_2d = points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2

    # Test compute_all_derivatives
    all_derivs_2d = solver_2d.compute_all_derivatives(u_2d)
    assert len(all_derivs_2d) == solver_2d.n_points

    # Find interior point (center of grid)
    mid_idx_2d = 55  # Center of 10x10 grid
    derivs_mid = all_derivs_2d[mid_idx_2d]
    print(f"  [2D] Derivatives at interior point {mid_idx_2d}:")
    print(f"       Keys: {list(derivs_mid.keys())}")

    # Check expected derivatives for f(x,y) = x² + y² at interior point
    if derivs_mid:
        grad_x = derivs_mid.get((1, 0), 0.0)
        grad_y = derivs_mid.get((0, 1), 0.0)
        lap_xx = derivs_mid.get((2, 0), 0.0)
        lap_yy = derivs_mid.get((0, 2), 0.0)
        print(f"       du/dx = {grad_x:.4f} (expected: {2 * points_2d[mid_idx_2d, 0]:.4f})")
        print(f"       du/dy = {grad_y:.4f} (expected: {2 * points_2d[mid_idx_2d, 1]:.4f})")
        print(f"       d²u/dx² = {lap_xx:.4f} (expected: 2.0)")
        print(f"       d²u/dy² = {lap_yy:.4f} (expected: 2.0)")

    print("\nNote: For gradient/laplacian utilities, use mfgarchon.utils.numerical.gfdm_operators")
    print("Smoke tests passed!")
