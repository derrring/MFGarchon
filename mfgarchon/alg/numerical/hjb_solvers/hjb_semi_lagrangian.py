#!/usr/bin/env python3
"""
Semi-Lagrangian HJB Solver for Mean Field Games

This module implements a semi-Lagrangian method for solving the Hamilton-Jacobi-Bellman
equation in MFG problems. The method follows characteristics backward in time and uses
interpolation to compute values at departure points.

The HJB equation solved is:
    -∂u/∂t + H(x, ∇u, m) - σ²/2 Δu = 0    in [0,T) × Ω
    u(T, x) = g(x)                         at t = T

equivalently ∂u/∂t = H(x, ∇u, m) - σ²/2 Δu (cost-to-go convention; u flows backward
in time from terminal data g). The semi-Lagrangian scheme discretizes this as:
    (u^n - û^{n+1}) / (-Δt) + H(x, ∇û^{n+1}, m^{n+1}) - σ²/2 Δû^{n+1} = 0
    u^n = û^{n+1} - Δt · H(x, ∇û^{n+1}, m^{n+1}) + ... (diffusion handled by chosen method)

where û^{n+1} is the value of u^{n+1} at the characteristic departure point.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator
from mfgarchon.geometry.boundary.applicator_interpolation import InterpolationApplicator
from mfgarchon.geometry.boundary.bc_utils import (
    bc_type_to_geometric_operation,
    get_bc_type_string,
)
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.pde_coefficients import check_adi_compatibility

from .base_hjb import BaseHJBSolver
from .hjb_sl_adi import (
    adi_diffusion_step,
    solve_crank_nicolson_diffusion_1d,
)
from .hjb_sl_characteristics import (
    apply_boundary_conditions_1d,
    apply_boundary_conditions_nd,
    trace_characteristic_backward_1d,
    trace_characteristic_backward_nd,
)
from .hjb_sl_interpolation import (
    interpolate_nearest_neighbor,
    interpolate_value_1d,
    interpolate_value_nd,
    interpolate_value_rbf_fallback,
)

if TYPE_CHECKING:
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary.conditions import BoundaryConditions

from mfgarchon.core.derivatives import DerivativeTensors

logger = get_logger(__name__)
try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class HJBSemiLagrangianSolver(BaseHJBSolver):
    """
    Semi-Lagrangian method for solving Hamilton-Jacobi-Bellman equations.

    The semi-Lagrangian method discretizes the HJB equation by following
    characteristics backward in time and interpolating values at departure points.
    This approach is particularly stable for convection-dominated problems.

    Key features:
    - Stable for large time steps
    - Handles discontinuous solutions well
    - Natural upwind discretization
    - Monotone and conservative

    Dimension support:
    - 1D: Full support (production-ready)
    - nD (2D/3D/4D+): Full support (2025-11-02)
      - Interpolation: RegularGridInterpolator (complete)
      - Diffusion: nD Laplacian (complete)
      - Characteristic tracing: Vector form (complete)
      - Optimal control: Vector optimization (complete)

    Required Geometry Traits (Issue #596 Phase 2.1):
        - SupportsGradient: Provides ∇U operator for optimal control computation

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsGradient trait
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.SL

    def __init__(
        self,
        problem: MFGProblem,
        interpolation_method: str = "linear",
        optimization_method: str = "brent",
        characteristic_solver: str = "explicit_euler",
        diffusion_method: str = "adi",
        use_rbf_fallback: bool = True,
        rbf_kernel: str = "thin_plate_spline",
        use_jax: bool | None = None,
        tolerance: float = 1e-8,
        max_char_iterations: int = 100,
        check_cfl: bool = True,
        enable_adaptive_substepping: bool = True,
        max_substeps: int = 100,
        cfl_target: float = 0.9,
        gradient_clip_threshold: float | None = None,
        enable_gradient_monitoring: bool = True,
        ode_rtol: float = 1e-6,
        ode_atol: float = 1e-8,
    ):
        """
        Initialize semi-Lagrangian HJB solver.

        Args:
            problem: MFG problem instance
            interpolation_method: Method for interpolating values
                - 'linear': Linear interpolation (fastest, C⁰ continuous)
                - 'cubic': Cubic spline interpolation (slower, C² continuous)
                - 'quintic': Quintic interpolation (slowest, highest accuracy, nD only)
                - 'nearest': Nearest neighbor (for debugging)
            optimization_method: Method for Hamiltonian optimization ('brent', 'golden')
            characteristic_solver: Method for solving characteristics
                - 'explicit_euler': First-order explicit Euler (fastest, least accurate)
                - 'rk2': Second-order Runge-Kutta midpoint method
                - 'rk4': Fourth-order Runge-Kutta via scipy.solve_ivp (most accurate)
            diffusion_method: Method for handling diffusion term (default: 'adi')
                - 'adi': ADI (Alternating Direction Implicit) splitting (default)
                - 'explicit': Explicit Laplacian (simple, requires small dt)
                - 'stochastic': Stochastic characteristic with Brownian motion (high-dim friendly)
                - 'none': No diffusion (for testing or zero-diffusion problems)
            use_rbf_fallback: Use RBF interpolation as fallback for boundary cases
            rbf_kernel: RBF kernel function
                - 'thin_plate_spline': Smooth, no free parameters (recommended)
                - 'multiquadric': Good for scattered data
                - 'gaussian': Localized influence
            use_jax: Whether to use JAX acceleration (auto-detect if None)
            tolerance: Convergence tolerance for optimization
            max_char_iterations: Maximum iterations for characteristic solving
            check_cfl: Whether to check CFL condition and issue warnings (default: True).
                CFL = max|grad(u)| * dt / dx. Warns if CFL > 1.0.
            enable_adaptive_substepping: Whether to automatically subdivide time steps
                when CFL > 1.0 to maintain stability (default: True). When enabled,
                the solver will use smaller internal time steps while preserving the
                overall time discretization.
            max_substeps: Maximum number of substeps per time step when adaptive
                substepping is enabled (default: 100). If more substeps are needed,
                a warning is issued and the solver proceeds with max_substeps.
            cfl_target: Target CFL number for adaptive substepping (default: 0.9).
                When CFL > 1.0, the time step is subdivided to achieve CFL ≤ cfl_target.
            gradient_clip_threshold: Safety threshold for gradient clipping (default: None).
                If provided, gradients exceeding this threshold will be clipped to prevent
                overflow in p² terms. Recommended: 1e6 for strong coupling problems.
                When None, no clipping is performed.
            enable_gradient_monitoring: Enable detailed gradient statistics tracking (default: True).
                Records when and where gradient clipping occurs for debugging. Disable for
                performance if clipping monitoring is not needed.
            ode_rtol: Relative tolerance for scipy.solve_ivp when characteristic_solver='rk4'.
                Default 1e-6.
            ode_atol: Absolute tolerance for scipy.solve_ivp when characteristic_solver='rk4'.
                Default 1e-8.
        """
        super().__init__(problem)
        self.hjb_method_name = "Semi-Lagrangian"

        # Solver configuration
        self.interpolation_method = interpolation_method
        self.optimization_method = optimization_method
        self.characteristic_solver = characteristic_solver
        self.diffusion_method = diffusion_method
        self.use_rbf_fallback = use_rbf_fallback
        self.rbf_kernel = rbf_kernel
        self.tolerance = tolerance
        self.max_char_iterations = max_char_iterations
        self.check_cfl = check_cfl
        self.enable_adaptive_substepping = enable_adaptive_substepping
        self.max_substeps = max_substeps
        self.cfl_target = cfl_target
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        # Gradient clipping configuration (Issue #583)
        self.gradient_clip_threshold = gradient_clip_threshold
        self.enable_gradient_monitoring = enable_gradient_monitoring

        # Issue #1049: Carlini-Silva 2014 prove unconditional stability of the
        # deterministic 2-direction averaging SL scheme (here: diffusion_method=
        # "stochastic") **specifically for Q1 (linear, monotone) interpolation**.
        # Cubic interpolation is not covered by the CS 2014 proof and is non-monotone
        # (Issue #1033 documents the exponential blow-up on Towel-on-Beach).
        # The previous validation actively rejected the proof-applicable combination.
        # Now: warn (don't reject) when cubic+stochastic is selected, since that
        # combination violates the monotone-scheme requirement of CS 2014.
        if self.diffusion_method == "stochastic" and self.interpolation_method in ("cubic", "quintic"):
            import warnings

            warnings.warn(
                f"diffusion_method='stochastic' with interpolation_method='{self.interpolation_method}' "
                "is NOT covered by the Carlini-Silva 2014 stability proof, which "
                "requires monotone (Q1/linear) interpolation. Cubic/quintic can "
                "violate the monotone-scheme requirement of Barles-Souganidis and "
                "produce exponential blow-up on stiff problems (see Issue #1033). "
                "Recommended: interpolation_method='linear'. mfgarchon's cubic "
                "path now uses `PchipInterpolator` (monotonic Hermite) which is "
                "more stable than `CubicSpline` but still outside the formal proof.",
                UserWarning,
                stacklevel=2,
            )

        # Gradient clipping statistics tracking
        self._reset_gradient_stats()

        # JAX acceleration
        self.use_jax = use_jax if use_jax is not None else JAX_AVAILABLE
        if self.use_jax and not JAX_AVAILABLE:
            logger.warning("JAX not available, falling back to NumPy")
            self.use_jax = False

        # Detect problem dimension (inherited from BaseNumericalSolver, Issue #633)
        self.dimension = self._detect_dimension()

        # Create boundary condition applicators
        # FDMApplicator: for ghost cell operations (gradient computation)
        self.bc_applicator = FDMApplicator(dimension=self.dimension)
        # InterpolationApplicator: for post-interpolation BC enforcement (Issue #636)
        self.interp_bc_applicator = InterpolationApplicator(dimension=self.dimension)

        # Validate geometry capabilities (Issue #596 Phase 2.1)
        # Semi-Lagrangian solver requires gradient operator for optimal control computation
        from mfgarchon.geometry.protocols import SupportsGradient

        if not isinstance(problem.geometry, SupportsGradient):
            raise TypeError(
                f"HJB Semi-Lagrangian solver requires geometry with SupportsGradient trait for ∇U computation. "
                f"{type(problem.geometry).__name__} does not implement this trait. "
                f"Compatible geometries: TensorProductGrid, ImplicitDomain."
            )

        # Precompute grid and time parameters (dimension-agnostic)
        if self.dimension == 1:
            # 1D problem: Use geometry API
            bounds = problem.geometry.get_bounds()
            xmin, xmax = bounds[0][0], bounds[1][0]
            Nx = problem.geometry.get_grid_shape()[0]
            self.x_grid = np.linspace(xmin, xmax, Nx)
            self.dt = problem.dt
            self.dx = problem.geometry.get_grid_spacing()[0]
            self.grid = None  # 1D uses direct arrays, not grid object
        else:
            # nD problem: Use CartesianGrid interface
            from mfgarchon.geometry.base import CartesianGrid

            if not isinstance(problem.geometry, CartesianGrid):
                raise ValueError(
                    f"Multi-dimensional problem requires CartesianGrid geometry. "
                    f"Got {type(problem.geometry).__name__} (dimension={self.dimension})"
                )
            self.grid = problem.geometry  # Geometry IS the grid
            self.dt = problem.dt
            # Grid spacing: vector of spacings in each dimension
            self.dx = np.array(self.grid.get_grid_spacing())
            # Grid shape: use get_grid_shape() for CartesianGrid interface compatibility
            self._grid_shape = tuple(self.grid.get_grid_shape())
            self._num_points_total = int(np.prod(self._grid_shape))
            self.x_grid = None  # Not used for nD

            # Check ADI compatibility for nD diffusion
            adi_ok, adi_msg = check_adi_compatibility(problem.sigma)
            self._adi_compatible = adi_ok
            if not adi_ok:
                logger.warning(
                    f"Diffusion tensor not ADI-compatible: {adi_msg}. "
                    f"ADI scheme may be inaccurate. Consider using more timesteps "
                    f"or implementing Craig-Sneyd scheme for mixed derivatives."
                )
            else:
                logger.info(f"ADI diffusion enabled for nD solve: {adi_msg}")

        # Setup JAX functions if available
        if self.use_jax:
            self._setup_jax_functions()

    # _detect_dimension() inherited from BaseNumericalSolver (Issue #633)

    def _setup_jax_functions(self):
        """Setup JAX-accelerated functions for performance."""
        if not self.use_jax:
            return

        @jit
        def jax_interpolate_linear(x_points, y_values, x_query):
            """JAX-accelerated linear interpolation."""
            return jnp.interp(x_query, x_points, y_values)

        @jit
        def jax_solve_characteristic_euler(x_current, p_optimal, dt):
            """JAX-accelerated characteristic solving using Euler method."""
            return x_current - p_optimal * dt

        self._jax_interpolate = jax_interpolate_linear
        self._jax_solve_characteristic = jax_solve_characteristic_euler

    def _reset_gradient_stats(self):
        """Reset gradient clipping statistics for new solve (Issue #583)."""
        self.gradient_stats = {
            "count": 0,  # Total number of clipped spatial points
            "max_gradient": 0.0,  # Maximum gradient magnitude encountered
            "locations": [],  # List of {t_idx, spatial_idx, gradient_value, density_value}
            "by_timestep": {},  # {t_idx: count} - clipping events per timestep
        }

    def _log_gradient_clipping_summary(self):
        """Log detailed summary of gradient clipping events (Issue #583)."""
        from mfgarchon.utils.mfg_logging import get_logger

        logger_local = get_logger(__name__)

        if self.gradient_stats["count"] == 0:
            if self.gradient_clip_threshold is not None:
                logger_local.info(
                    f"No gradient clipping required - all gradients remained below threshold "
                    f"({self.gradient_clip_threshold:.2e}). Max gradient: {self.gradient_stats['max_gradient']:.2e}"
                )
            return

        # Gradient clipping occurred
        logger_local.warning("=" * 60)
        logger_local.warning("GRADIENT CLIPPING SUMMARY (Issue #583)")
        logger_local.warning("=" * 60)
        logger_local.warning(f"Total clipped points: {self.gradient_stats['count']}")
        logger_local.warning(f"Max gradient encountered: {self.gradient_stats['max_gradient']:.2e}")
        logger_local.warning(f"Clip threshold: {self.gradient_clip_threshold:.2e}")

        # Temporal distribution
        if self.gradient_stats["by_timestep"]:
            logger_local.warning("\nClipping by timestep (first 10):")
            sorted_timesteps = sorted(self.gradient_stats["by_timestep"].keys())[:10]
            for t_idx in sorted_timesteps:
                count = self.gradient_stats["by_timestep"][t_idx]
                logger_local.warning(f"  t={t_idx}: {count} points clipped")

            if len(self.gradient_stats["by_timestep"]) > 10:
                logger_local.warning(f"  ... and {len(self.gradient_stats['by_timestep']) - 10} more timesteps")

        # Spatial hotspots (if tracked)
        if self.gradient_stats["locations"] and self.enable_gradient_monitoring:
            logger_local.warning("\nFirst few clipping locations:")
            for loc in self.gradient_stats["locations"][:5]:
                density_str = f"{loc['density_value']:.2e}" if loc["density_value"] is not None else "N/A"
                logger_local.warning(
                    f"  t={loc['t_idx']}, x_idx={loc['spatial_idx']}, "
                    f"||∇U||={loc['gradient_value']:.2e}, "
                    f"m={density_str}"
                )

            if len(self.gradient_stats["locations"]) > 5:
                logger_local.warning(f"  ... and {len(self.gradient_stats['locations']) - 5} more locations")

        logger_local.warning("=" * 60)
        logger_local.warning(
            "RECOMMENDATION: Gradient clipping is a SAFETY NET, not a solution. "
            "Enable adaptive Picard damping (adaptive_damping=True in FixedPointIterator) "
            "or use weaker coupling to prevent gradient amplification at the source."
        )

    def _clip_gradient_with_monitoring(
        self,
        grad_u: np.ndarray | tuple[np.ndarray, ...],
        t_idx: int | None = None,
        m_density: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Clip gradients and track where clipping occurs (Issue #583).

        Args:
            grad_u: Gradient array(s) from _compute_gradient
            t_idx: Current timestep index for location tracking (optional)
            m_density: Density values for correlation analysis (optional)

        Returns:
            Clipped gradient with same structure as input
        """
        if self.gradient_clip_threshold is None:
            return grad_u  # No clipping

        # Import logging tools
        from mfgarchon.utils.mfg_logging import get_logger

        logger_local = get_logger(__name__)

        if self.dimension == 1:
            # 1D gradient clipping
            grad_norm = np.abs(grad_u)
            grad_max = np.max(grad_norm)

            # Update max gradient stat
            self.gradient_stats["max_gradient"] = max(self.gradient_stats["max_gradient"], float(grad_max))

            # Identify where clipping is needed
            clip_mask = grad_norm > self.gradient_clip_threshold

            if np.any(clip_mask):
                clip_indices = np.where(clip_mask)[0]
                n_clipped = len(clip_indices)
                self.gradient_stats["count"] += n_clipped

                # Track by timestep
                if t_idx is not None and self.enable_gradient_monitoring:
                    if t_idx not in self.gradient_stats["by_timestep"]:
                        self.gradient_stats["by_timestep"][t_idx] = 0
                    self.gradient_stats["by_timestep"][t_idx] += n_clipped

                    # Store first few locations (avoid memory explosion)
                    if len(self.gradient_stats["locations"]) < 100:
                        for idx in clip_indices[:10]:  # First 10 per timestep
                            self.gradient_stats["locations"].append(
                                {
                                    "t_idx": int(t_idx),
                                    "spatial_idx": int(idx),
                                    "gradient_value": float(grad_norm[idx]),
                                    "density_value": float(m_density[idx]) if m_density is not None else None,
                                }
                            )

                    # Log clipping event with location info
                    x_values = self.x_grid[clip_indices]
                    logger_local.warning(
                        f"Gradient clipped at t={t_idx}: {n_clipped} points, "
                        f"max ||∇U||={grad_max:.2e}, "
                        f"locations: x={x_values[:5].tolist()}"  # First 5 x-coordinates
                    )

                # Perform clipping
                grad_u_clipped = np.clip(grad_u, -self.gradient_clip_threshold, self.gradient_clip_threshold)
                return grad_u_clipped

            return grad_u

        else:
            # nD gradient clipping
            grad_components = list(grad_u)  # Convert tuple to list for modification
            grad = np.stack(grad_components, axis=0)
            grad_norm = np.sqrt(np.sum(grad**2, axis=0))
            grad_max = np.max(grad_norm)

            # Update max gradient stat
            self.gradient_stats["max_gradient"] = max(self.gradient_stats["max_gradient"], float(grad_max))

            # Identify where clipping is needed
            clip_mask = grad_norm > self.gradient_clip_threshold

            if np.any(clip_mask):
                clip_indices = np.argwhere(clip_mask)  # Returns (N, d) array
                n_clipped = len(clip_indices)
                self.gradient_stats["count"] += n_clipped

                # Track by timestep
                if t_idx is not None and self.enable_gradient_monitoring:
                    if t_idx not in self.gradient_stats["by_timestep"]:
                        self.gradient_stats["by_timestep"][t_idx] = 0
                    self.gradient_stats["by_timestep"][t_idx] += n_clipped

                    # Store first few locations
                    if len(self.gradient_stats["locations"]) < 100:
                        for idx_tuple in clip_indices[:10]:
                            idx_tuple_int = tuple(int(i) for i in idx_tuple)
                            self.gradient_stats["locations"].append(
                                {
                                    "t_idx": int(t_idx),
                                    "spatial_idx": idx_tuple_int,
                                    "gradient_value": float(grad_norm[idx_tuple_int]),
                                    "density_value": float(m_density[idx_tuple_int]) if m_density is not None else None,
                                }
                            )

                    # Log clipping event
                    logger_local.warning(
                        f"Gradient clipped at t={t_idx}: {n_clipped} points, max ||∇U||={grad_max:.2e}"
                    )

                # Perform clipping component-wise
                for d in range(self.dimension):
                    grad_components[d] = np.where(
                        clip_mask,
                        grad_components[d] * self.gradient_clip_threshold / (grad_norm + 1e-16),
                        grad_components[d],
                    )

                return tuple(grad_components)

            return grad_u

    def _compute_gradient(
        self,
        u_values: np.ndarray,
        check_cfl: bool = True,
        t_idx: int | None = None,
        m_density: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Compute gradient ∇u for optimal control using trait-based geometry operators (Issue #596 Phase 2.1).

        For standard MFG with quadratic control cost, the optimal control is:
            α*(x,t) = ∇u(x,t)

        Uses geometry.get_gradient_operator() which automatically handles:
        - Boundary conditions via ghost cells
        - Scheme selection (central differences for Semi-Lagrangian)
        - Multi-dimensional stencils

        Args:
            u_values: Value function array
                - 1D: shape (Nx+1,)
                - nD: shape (Nx1+1, Nx2+1, ..., Nxd+1)
            check_cfl: Whether to check CFL condition (default: True)
            t_idx: Current timestep index for gradient clipping monitoring (optional, Issue #583)
            m_density: Density values for gradient clipping correlation analysis (optional, Issue #583)

        Returns:
            gradient: Gradient array(s), optionally clipped if gradient_clip_threshold is set
                - 1D: shape (Nx+1,) - scalar gradient at each point
                - nD: tuple of d arrays, each shape (Nx1+1, ..., Nxd+1)

        Note:
            Uses central differences for characteristic tracing (Semi-Lagrangian scheme).
            Boundary conditions are automatically enforced by gradient operators.
            Issues CFL warning if max|∇u|·dt/dx > 1.
            Gradient clipping (Issue #583): Clips gradients > gradient_clip_threshold to prevent overflow.
        """
        # Get gradient operators from geometry (Issue #596 Phase 2.1)
        # Semi-Lagrangian uses central differences for gradient computation
        grad_ops = self.problem.geometry.get_gradient_operator(scheme="central")

        if self.dimension == 1:
            # 1D gradient computation via operator
            grad_u = grad_ops[0](u_values)

            # Apply gradient clipping (Issue #583)
            grad_u_clipped = self._clip_gradient_with_monitoring(grad_u, t_idx=t_idx, m_density=m_density)

            # CFL check (after clipping to get realistic CFL with clipped gradients)
            if check_cfl and self.check_cfl:
                max_grad = np.max(np.abs(grad_u_clipped))
                cfl = max_grad * self.dt / self.dx
                if cfl > 1.0:
                    logger.warning(
                        f"CFL condition violated: max|∇u|·dt/dx = {cfl:.3f} > 1.0. "
                        f"Consider reducing dt or increasing dx. "
                        f"max|∇u| = {max_grad:.3f}, dt = {self.dt:.6f}, dx = {self.dx:.6f}"
                    )

            return grad_u_clipped

        else:
            # nD gradient computation via operators
            grad_components = []
            for d in range(self.dimension):
                grad_axis = grad_ops[d](u_values)
                grad_components.append(grad_axis)

            # Apply gradient clipping (Issue #583)
            grad_components_clipped = self._clip_gradient_with_monitoring(
                tuple(grad_components), t_idx=t_idx, m_density=m_density
            )

            # CFL check (after clipping)
            if check_cfl and self.check_cfl:
                grad = np.stack(grad_components_clipped, axis=0)
                magnitude = np.sqrt(np.sum(grad**2, axis=0))
                max_grad = np.max(magnitude)
                min_spacing = np.min(self.dx)
                cfl = max_grad * self.dt / min_spacing
                if cfl > 1.0:
                    logger.warning(
                        f"CFL condition violated: max|∇u|·dt/dx_min = {cfl:.3f} > 1.0. "
                        f"Consider reducing dt or increasing grid spacing. "
                        f"max|∇u| = {max_grad:.3f}, dt = {self.dt:.6f}, dx_min = {min_spacing:.6f}"
                    )

            # Return as tuple of arrays (one per dimension)
            return grad_components_clipped

    def _compute_cfl_and_substeps(self, u_values: np.ndarray, dt_target: float) -> tuple[float, int, float]:
        """
        Compute CFL number and determine optimal number of substeps.

        When the CFL condition (CFL = max|grad(u)| * dt / dx) exceeds 1.0,
        this method computes how many substeps are needed to maintain
        CFL <= cfl_target (default 0.9).

        Uses trait-based gradient operators for consistent computation (Issue #596 Phase 2.1).

        Args:
            u_values: Current value function array
            dt_target: Target time step (full time step to subdivide)

        Returns:
            Tuple of (cfl_number, n_substeps, dt_substep):
                - cfl_number: The CFL number with the target dt
                - n_substeps: Number of substeps needed (1 if CFL <= 1.0)
                - dt_substep: Time step to use for each substep
        """
        # Compute gradient using trait-based operators (reuse _compute_gradient with CFL check disabled)
        grad_result = self._compute_gradient(u_values, check_cfl=False)

        if self.dimension == 1:
            # 1D CFL computation
            grad_u = grad_result
            max_grad = np.max(np.abs(grad_u))
            cfl = max_grad * dt_target / self.dx
            dx_eff = self.dx
        else:
            # nD CFL computation
            grad_components = grad_result  # Tuple of gradient arrays
            grad = np.stack(grad_components, axis=0)
            magnitude = np.sqrt(np.sum(grad**2, axis=0))
            max_grad = np.max(magnitude)
            dx_eff = np.min(self.dx)
            cfl = max_grad * dt_target / dx_eff

        # Determine substeps needed
        if cfl <= 1.0 or not self.enable_adaptive_substepping:
            return cfl, 1, dt_target

        # Compute substeps to achieve CFL <= cfl_target
        n_substeps = int(np.ceil(cfl / self.cfl_target))
        n_substeps = min(n_substeps, self.max_substeps)

        if n_substeps >= self.max_substeps:
            logger.warning(
                f"CFL = {cfl:.2f} requires {int(np.ceil(cfl / self.cfl_target))} substeps, "
                f"capped at max_substeps={self.max_substeps}. "
                f"Stability may be compromised. Consider reducing dt or increasing grid resolution."
            )

        dt_substep = dt_target / n_substeps
        actual_cfl = max_grad * dt_substep / dx_eff

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Adaptive substepping: CFL={cfl:.2f} -> {actual_cfl:.2f} ({n_substeps} substeps, dt={dt_substep:.6f})"
            )

        return cfl, n_substeps, dt_substep

    # _get_boundary_conditions() removed (Issue #634): was duplicating
    # BaseMFGSolver.get_boundary_conditions() from base_solver.py:175-234.
    # All callers now use the inherited get_boundary_conditions().

    def _get_bc_type_string(self, bc) -> str | None:
        """
        Extract BC type string from BoundaryConditions object.

        Args:
            bc: BoundaryConditions object or None

        Returns:
            BC type string ("periodic", "dirichlet", "neumann") or None

        Note:
            Issue #545: Replace hasattr pattern for BCType enum value extraction.
            Used in characteristic tracing and diffusion term computation.
        """
        if bc is None:
            return None

        # Try to get default_bc attribute
        try:
            bc_type_enum = bc.default_bc
            if bc_type_enum is None:
                return None

            # Try to get .value attribute (BCType enum)
            try:
                return bc_type_enum.value
            except AttributeError:
                # Fall back to string conversion
                return str(bc_type_enum)
        except AttributeError:
            return None

    def _get_per_boundary_bc_types(self, bc) -> tuple[str | None, str | None]:
        """
        Get BC type strings for each boundary (1D: xmin and xmax).

        For mixed BCs (e.g., Neumann at x=0, Dirichlet at x=L), this method
        queries the BC type at each boundary separately.

        Args:
            bc: BoundaryConditions object or None

        Returns:
            Tuple of (bc_type_at_xmin, bc_type_at_xmax)

        Note:
            For uniform BCs, both values will be the same.
            For mixed BCs, values may differ per boundary.
        """
        if bc is None:
            return (None, None)

        # Try to use get_bc_type_at_boundary method for per-boundary queries
        try:
            bc_type_min_enum = bc.get_bc_type_at_boundary("x_min")
            bc_type_max_enum = bc.get_bc_type_at_boundary("x_max")

            # Extract string values from BCType enums
            bc_type_min = bc_type_min_enum.value if bc_type_min_enum is not None else None
            bc_type_max = bc_type_max_enum.value if bc_type_max_enum is not None else None

            return (bc_type_min, bc_type_max)
        except AttributeError:
            pass

        # Fallback: use uniform BC type for both boundaries
        bc_type = self._get_bc_type_string(bc)
        return (bc_type, bc_type)

    def _enforce_boundary_conditions(self, U: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Enforce boundary conditions on solution array (dimension-agnostic).

        For Semi-Lagrangian, BC enforcement after each timestep ensures:
        - **Neumann** (du/dn=0): 2nd-order extrapolation preserving zero gradient
        - **Dirichlet** (u=g): u[boundary] = g (prescribed value)

        This explicit enforcement is critical because Semi-Lagrangian's
        interpolation-based approach doesn't naturally preserve BCs.

        Uses InterpolationApplicator (Issue #636) for unified BC handling
        across all dimensions.

        Args:
            U: Solution array of shape (Nx,) for 1D, (Ny, Nx) for 2D, etc.
            time: Current time for time-dependent BC values

        Returns:
            Solution with BCs enforced (modified in-place)
        """
        bc = self.get_boundary_conditions()
        if bc is None:
            return U

        # Use InterpolationApplicator for dimension-agnostic BC enforcement
        return self.interp_bc_applicator.enforce_values(U, bc, time=time)

    @deprecated_parameter(param_name="M_density_evolution_from_FP", since="v0.17.0", replacement="M_density")
    @deprecated_parameter(param_name="U_final_condition_at_T", since="v0.17.0", replacement="U_terminal")
    @deprecated_parameter(param_name="U_from_prev_picard", since="v0.17.0", replacement="U_coupling_prev")
    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        volatility_field: float | np.ndarray | None = None,
        # Deprecated parameter names for backward compatibility
        M_density_evolution_from_FP: np.ndarray | None = None,
        U_final_condition_at_T: np.ndarray | None = None,
        U_from_prev_picard: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the HJB system using semi-Lagrangian method.

        The semi-Lagrangian discretization of the HJB equation:
            ∂u/∂t + H(x, ∇u, m) - σ²/2 Δu = 0

        is solved by following characteristics backward in time:
            1. For each grid point x_i at time t^{n+1}
            2. Find optimal control p* that minimizes H(x_i, p, m^{n+1})
            3. Trace characteristic backward: X(t^n) = x_i - p* Δt
            4. Interpolate u^n at departure point X(t^n)
            5. Update: u^{n+1}_i = û^n(X(t^n)) - Δt[H(...) - σ²/2 Δu]

        Args:
            M_density: (Nt, *spatial_shape) density from FP solver
            U_terminal: (*spatial_shape,) terminal condition u(T, x)
            U_coupling_prev: (Nt, *spatial_shape) previous coupling iteration estimate
            volatility_field: Optional diffusion coefficient override

        Returns:
            (Nt, *grid_shape) solution array for value function
        """
        # Handle deprecated parameter names (warnings issued by @deprecated_parameter decorator)
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
        if M_density is None:
            raise ValueError("M_density is required")
        if U_terminal is None:
            raise ValueError("U_terminal is required")
        if U_coupling_prev is None:
            raise ValueError("U_coupling_prev is required")

        # Reset gradient clipping statistics for this solve (Issue #583)
        self._reset_gradient_stats()

        # Handle multi-dimensional grids
        # M_density has shape (Nt_points, *spatial_shape) where Nt_points = Nt + 1
        shape = M_density.shape
        Nt_points = shape[0]  # Number of time points (includes t=0 and t=T)
        grid_shape = shape[1:]  # Remaining dimensions

        # Output shape: (Nt_points, *grid_shape) - same as input
        U_solution = np.zeros((Nt_points, *grid_shape))

        # Set final condition at t=T (last index)
        U_solution[-1] = U_terminal

        total_points = np.prod(grid_shape)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"Starting semi-Lagrangian HJB solve: {Nt_points} time points, {total_points} spatial points ({grid_shape})"
            )
            if self.gradient_clip_threshold is not None:
                logger.info(f"Gradient clipping enabled: threshold = {self.gradient_clip_threshold:.2e}")

        # Solve backward in time using semi-Lagrangian method
        # Loop from second-to-last index down to 0
        total_substeps_used = 0
        for n in range(Nt_points - 2, -1, -1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Solving time step {n}/{Nt_points - 2}")

            # Index for density and coupling arrays
            m_idx = min(n + 1, Nt_points - 1)
            u_prev_idx = min(n, Nt_points - 1)

            # Compute CFL and determine substeps needed for this time step
            # DPP path doesn't use characteristics, so CFL substepping is not needed
            if self._use_dpp:
                cfl, n_substeps, dt_substep = 0.0, 1, self.dt
            else:
                cfl, n_substeps, dt_substep = self._compute_cfl_and_substeps(U_solution[n + 1], self.dt)
            total_substeps_used += n_substeps

            if n_substeps == 1:
                # No substepping needed - use standard time step
                U_solution[n] = self._solve_timestep_semi_lagrangian(
                    U_solution[n + 1],  # u^{n+1} (from output array, always valid)
                    M_density[m_idx],  # m^{n+1} or last available density
                    U_coupling_prev[u_prev_idx],  # u_k^n for coupling terms
                    n,  # time index
                )
            else:
                # Adaptive substepping: subdivide the time step
                U_current = U_solution[n + 1].copy()
                for substep in range(n_substeps):
                    U_current = self._solve_timestep_semi_lagrangian_with_dt(
                        U_current,
                        M_density[m_idx],
                        U_coupling_prev[u_prev_idx],
                        n,
                        dt_substep,
                    )
                    # Check for numerical issues after each substep
                    if np.any(np.isnan(U_current) | np.isinf(U_current)):
                        error_msg = (
                            f"Semi-Lagrangian solver failed at time step {n}/{Nt_points - 2}, "
                            f"substep {substep + 1}/{n_substeps} with NaN/Inf values. "
                            f"CFL was {cfl:.2f}, using {n_substeps} substeps with dt={dt_substep:.6f}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                U_solution[n] = U_current
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Time step {n}: used {n_substeps} substeps (CFL={cfl:.2f})")

            # Check for numerical issues
            if np.any(np.isnan(U_solution[n]) | np.isinf(U_solution[n])):
                error_msg = (
                    f"Semi-Lagrangian solver failed at time step {n}/{Nt_points - 2} with NaN/Inf values. "
                    "Possible causes:\n"
                    "  1. CFL condition violated (try smaller dt or enable adaptive_substepping=True)\n"
                    "  2. Grid too coarse for solution features\n"
                    "  3. Hamiltonian evaluation issues\n"
                    "  4. Interpolation errors near boundaries"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        if logger.isEnabledFor(logging.INFO):
            final_residual = np.linalg.norm(U_solution[1] - U_solution[0])
            logger.info(f"Semi-Lagrangian HJB solve completed. Final residual: {final_residual:.2e}")
            if self.enable_adaptive_substepping and total_substeps_used > Nt_points:
                logger.info(
                    f"Adaptive substepping used {total_substeps_used} total substeps for {Nt_points} time points"
                )

        # Log gradient clipping summary (Issue #583)
        if self.gradient_clip_threshold is not None or self.gradient_stats["count"] > 0:
            self._log_gradient_clipping_summary()

        return U_solution

    def _solve_timestep_semi_lagrangian(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        U_prev_picard: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """
        Solve one timestep using semi-Lagrangian method (supports 1D and nD).

        Args:
            U_next: Value function at next time step
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points
            M_next: Density at next time step (same shape as U_next)
            U_prev_picard: Value from previous Picard iteration (for coupling)
            time_idx: Current time index

        Returns:
            Value function at current time step (same shape as U_next)
        """
        # Issue #909: L-based DPP path for non-smooth Lagrangians
        if self._use_dpp:
            return self._solve_timestep_dpp(U_next, M_next, time_idx)

        # Issue #1026: Carlini-Silva stochastic-characteristic SL bypasses splitting
        if self.diffusion_method == "stochastic":
            return self._solve_timestep_stochastic_sl(U_next, M_next, time_idx, dt=self.dt)

        if self.dimension == 1:
            # 1D solve with operator splitting: characteristics + Crank-Nicolson diffusion
            Nx = len(U_next)

            # Compute gradient for optimal control: α* = ∇u
            # Pass timestep and density for gradient clipping monitoring (Issue #583)
            grad_u = self._compute_gradient(U_next, check_cfl=True, t_idx=time_idx, m_density=M_next)

            # Issue #930: Vectorized advection — batch characteristic tracing + interpolation
            # For explicit_euler/rk2, characteristic is x_departure = x - p*dt (vectorizable)
            if self.characteristic_solver in ("explicit_euler", "rk2"):
                # Step 1a: Batch departure points
                x_departures = self.x_grid - grad_u * self.dt

                # Apply boundary conditions (vectorized)
                bc = self.get_boundary_conditions()
                bc_type_str = get_bc_type_string(bc)
                bc_op = bc_type_to_geometric_operation(bc_type_str)
                bounds = self.problem.geometry.get_bounds()
                xmin, xmax = bounds[0][0], bounds[1][0]
                if bc_op == "reflect":
                    # Reflect: fold back into domain
                    x_departures = np.clip(x_departures, xmin, xmax)
                elif bc_op == "wrap":
                    # Periodic: wrap around
                    L = xmax - xmin
                    x_departures = xmin + (x_departures - xmin) % L

                # Step 1b: Batch interpolation
                from scipy.interpolate import CubicSpline, interp1d

                if self.interpolation_method == "cubic":
                    interp_fn = CubicSpline(self.x_grid, U_next, bc_type="not-a-knot")
                    u_departures = interp_fn(x_departures)
                else:
                    interp_fn = interp1d(self.x_grid, U_next, kind="linear", fill_value="extrapolate")
                    u_departures = interp_fn(x_departures)

                # Step 1c: Batch Hamiltonian evaluation
                x_batch = self.x_grid.reshape(-1, 1)  # (Nx, 1)
                p_batch = grad_u.reshape(-1, 1)  # (Nx, 1)
                H_class = self.problem.hamiltonian_class
                if H_class is not None:
                    H_values = np.asarray(H_class(x_batch, M_next, p_batch, t=time_idx * self.dt), dtype=float).ravel()
                else:
                    H_values = np.zeros(Nx)

                # Step 1d: Advection update (vectorized)
                U_star = u_departures - self.dt * H_values

            else:
                # Fallback: per-point loop for rk4 or other methods
                U_star = np.zeros(Nx)
                for i in range(Nx):
                    try:
                        p_optimal = grad_u[i]
                        x_departure = self._trace_characteristic_backward(self.x_grid[i], p_optimal, self.dt)
                        u_departure = self._interpolate_value(U_next, x_departure)
                        hamiltonian_value = self._evaluate_hamiltonian(self.x_grid[i], p_optimal, M_next[i], time_idx)
                        U_star[i] = u_departure - self.dt * hamiltonian_value
                    except Exception as e:
                        logger.warning(f"Error at grid point {i}: {e}")
                        U_star[i] = U_next[i]

            # Step 2: Diffusion (using configured method)
            U_current = self._apply_diffusion(U_star, self.dt)

            # Step 3: Enforce boundary conditions on solution using the applicator
            bc = self.get_boundary_conditions()
            if bc:
                time = time_idx * self.dt
                U_current = self.bc_applicator.enforce_values(
                    U_current, boundary_conditions=bc, spacing=(self.dx,), time=time
                )

            return U_current

        else:
            # nD solve with operator splitting: advection + ADI diffusion
            # Reshape arrays to grid shape for easier indexing
            if U_next.ndim == 1:
                # Infer grid shape from array size (handles both full grid and interior points)
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))

                if total_points == expected_full:
                    grid_shape = tuple(self._grid_shape)
                else:
                    # Interior points only (num_points - 1 in each dimension)
                    grid_shape = tuple(n - 1 for n in self._grid_shape)

                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            # Step 1: Advection pass - compute u_star for all points
            # u_star = u(X(t-dt)) - dt * H(x, p*, m)
            U_star = np.zeros_like(U_next_shaped)

            # Compute gradient for optimal control: alpha* = grad(u)
            # Returns tuple of gradient components, each with shape grid_shape
            # Pass timestep and density for gradient clipping monitoring (Issue #583)
            grad_components = self._compute_gradient(
                U_next_shaped, check_cfl=True, t_idx=time_idx, m_density=M_next_shaped
            )

            # Track errors for diagnostics
            error_count = 0
            total_points = int(np.prod(grid_shape))

            # Iterate over all grid points for advection
            for multi_idx in np.ndindex(grid_shape):
                # Get spatial coordinates for this grid point
                x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                m_current = M_next_shaped[multi_idx]

                # Extract optimal control from gradient (vector for nD)
                p_optimal = np.array([grad_components[d][multi_idx] for d in range(self.dimension)])

                # Trace characteristic backward (vector operation)
                x_departure = self._trace_characteristic_backward(x_current, p_optimal, self.dt)

                # Interpolate at departure point
                u_departure = self._interpolate_value(U_next_shaped, x_departure)

                # Evaluate Hamiltonian
                hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                # Advection step for backward HJB solve:
                # HJB: -∂u/∂t + H(x,∇u,m) - σ²/2 Δu = 0
                # Rearranging: ∂u/∂t = H - σ²/2 Δu
                # Backward discretization: u^n = u^{n+1} - dt * H (diffusion handled separately)
                # Issue #575: Sign was wrong (+ instead of -)
                u_star_val = u_departure - self.dt * hamiltonian_value

                # Check for numerical issues
                if np.isnan(u_star_val) or np.isinf(u_star_val):
                    error_count += 1
                    if error_count <= 5:
                        logger.warning(
                            f"NaN/Inf at grid point {multi_idx}: "
                            f"u_departure={u_departure:.3e}, H={hamiltonian_value:.3e}"
                        )
                    U_star[multi_idx] = U_next_shaped[multi_idx]  # Fallback
                else:
                    U_star[multi_idx] = u_star_val

            # Report error summary if any occurred in advection
            if error_count > 0:
                error_pct = 100 * error_count / total_points
                if error_pct > 10:
                    raise ValueError(
                        f"Semi-Lagrangian advection failed: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}. Check grid resolution and time step."
                    )
                else:
                    logger.warning(
                        f"Semi-Lagrangian advection: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}"
                    )

            # Step 2: Diffusion pass (using configured method)
            U_current_shaped = self._apply_diffusion(U_star, self.dt)

            # Return flattened if input was flattened
            if U_next.ndim == 1:
                return U_current_shaped.ravel()
            else:
                return U_current_shaped

    def _solve_timestep_semi_lagrangian_with_dt(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        U_prev_picard: np.ndarray,
        time_idx: int,
        dt: float,
    ) -> np.ndarray:
        """
        Solve one timestep using semi-Lagrangian method with custom time step.

        This is the same as _solve_timestep_semi_lagrangian but allows specifying
        a custom dt for adaptive substepping.

        Args:
            U_next: Value function at next time step
            M_next: Density at next time step
            U_prev_picard: Value from previous Picard iteration
            time_idx: Current time index
            dt: Time step to use (allows custom dt for substepping)

        Returns:
            Value function at current time step
        """
        # Issue #909: L-based DPP path for non-smooth Lagrangians
        if self._use_dpp:
            return self._solve_timestep_dpp(U_next, M_next, time_idx, dt=dt)

        # Issue #1026: Carlini-Silva stochastic-characteristic SL bypasses splitting
        if self.diffusion_method == "stochastic":
            return self._solve_timestep_stochastic_sl(U_next, M_next, time_idx, dt=dt)

        if self.dimension == 1:
            # 1D solve with operator splitting
            Nx = len(U_next)
            U_star = np.zeros(Nx)

            # Compute gradient for optimal control
            # Pass timestep and density for gradient clipping monitoring (Issue #583)
            grad_u = self._compute_gradient(U_next, check_cfl=False, t_idx=time_idx, m_density=M_next)

            # Step 1: Advection along characteristics
            for i in range(Nx):
                x_current = self.x_grid[i]
                m_current = M_next[i]

                try:
                    p_optimal = grad_u[i]
                    x_departure = self._trace_characteristic_backward(x_current, p_optimal, dt)
                    u_departure = self._interpolate_value(U_next, x_departure)
                    hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                    # Backward HJB: u^n = u^{n+1} - dt * H (Issue #575: sign fix)
                    U_star[i] = u_departure - dt * hamiltonian_value

                except Exception as e:
                    logger.warning(f"Error at grid point {i}: {e}")
                    U_star[i] = U_next[i]

            # Step 2: Diffusion with custom dt
            U_current = self._apply_diffusion(U_star, dt)

            # Step 3: Enforce boundary conditions on solution
            U_current = self._enforce_boundary_conditions(U_current)

            return U_current

        else:
            # nD solve with operator splitting
            if U_next.ndim == 1:
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))

                if total_points == expected_full:
                    grid_shape = tuple(self._grid_shape)
                else:
                    grid_shape = tuple(n - 1 for n in self._grid_shape)

                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            U_star = np.zeros_like(U_next_shaped)
            # Pass timestep and density for gradient clipping monitoring (Issue #583)
            grad_components = self._compute_gradient(
                U_next_shaped, check_cfl=False, t_idx=time_idx, m_density=M_next_shaped
            )

            error_count = 0
            total_points = int(np.prod(grid_shape))

            for multi_idx in np.ndindex(grid_shape):
                x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                m_current = M_next_shaped[multi_idx]
                p_optimal = np.array([grad_components[d][multi_idx] for d in range(self.dimension)])

                x_departure = self._trace_characteristic_backward(x_current, p_optimal, dt)
                u_departure = self._interpolate_value(U_next_shaped, x_departure)
                hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                # Backward HJB: u^n = u^{n+1} - dt * H (Issue #575: sign fix)
                u_star_val = u_departure - dt * hamiltonian_value

                if np.isnan(u_star_val) or np.isinf(u_star_val):
                    error_count += 1
                    U_star[multi_idx] = U_next_shaped[multi_idx]
                else:
                    U_star[multi_idx] = u_star_val

            if error_count > 0:
                error_pct = 100 * error_count / total_points
                if error_pct > 10:
                    raise ValueError(
                        f"Semi-Lagrangian advection failed: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}."
                    )

            # ADI diffusion with custom dt
            U_current_shaped = self._apply_diffusion(U_star, dt)

            # Enforce boundary conditions (Issue #636 - nD support)
            U_current_shaped = self._enforce_boundary_conditions(U_current_shaped)

            if U_next.ndim == 1:
                return U_current_shaped.ravel()
            else:
                return U_current_shaped

    # === Carlini-Silva stochastic-characteristic SL (Issue #1026) ===

    def _solve_timestep_stochastic_sl(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float | None = None,
    ) -> np.ndarray:
        """Carlini-Silva (2014) semi-Lagrangian step with stochastic characteristics.

        The diffusion enters directly through 2*d stochastic departure points
        (one pair per spatial dimension), eliminating the operator-splitting
        diffusion solve. For separable convex Lagrangian L(x, a, m) =
        (1/2)|a|^2 + f(x, m) the optimal control is alpha* = -nabla u^{n+1}
        and the update is

            U^n_i = (1/(2d)) * sum_{k=1..d} [I[U^{n+1}](y_k^+) + I[U^{n+1}](y_k^-)]
                    - dt * H(x_i, p_i, m_i^n)

        with y_k^pm = x_i + alpha*_i * dt +/- sigma * sqrt(dt) * e_k.

        Validation: see mfg-research/experiments/crowd_evacuation_2d/minors/archive/
        exp14_towel_1d_benchmark/subs/exp14e_solver_comparison/, where the
        decoupled-from-diffusion form (diffusion_method='adi' default) gave
        O(h) convergence on the 1D Boltzmann-Gibbs equilibrium against the
        Carlini-Silva theoretical O(h^2). Reproducing the CS rate requires
        this stochastic-characteristic path.

        References:
            Carlini, E., & Silva, F. J. (2014). A semi-Lagrangian scheme for a
            degenerate second order MFG system. Discrete and Continuous
            Dynamical Systems, 35(9), 4269-4292.

        Args:
            U_next: Value function at next time step (shape (Nx,) for 1D, or
                grid shape / flattened for nD).
            M_next: Density at next time step (matching shape).
            time_idx: Current time index (used in Hamiltonian evaluation).
            dt: Time step. Defaults to self.dt; pass explicitly for adaptive
                substepping.

        Returns:
            Value function at current time step, same shape as U_next.

        Notes:
            Issue #1049: previously rejected interpolation_method="linear" here,
            inverted from CS 2014's stability requirement. The "linear" path is
            now allowed; warning issued at __init__ when cubic/quintic is used
            with stochastic (the unproven combination).
        """
        if dt is None:
            dt = self.dt

        if self.dimension == 1:
            return self._stochastic_sl_step_1d(U_next, M_next, time_idx, dt)
        else:
            return self._stochastic_sl_step_nd(U_next, M_next, time_idx, dt)

    def _stochastic_sl_step_1d(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float,
    ) -> np.ndarray:
        """1D stochastic SL step. See _solve_timestep_stochastic_sl."""
        Nx = len(U_next)
        sigma = self.problem.sigma  # SDE volatility (Sigma in mfg_problem.py:39)
        sqrt_dt = float(np.sqrt(dt))
        diffusion_offset = sigma * sqrt_dt

        # Optimal control alpha* = -p where p = nabla u^{n+1}
        # (existing code uses x_dep = x - p*dt, i.e., assumes alpha* = -p)
        grad_u = self._compute_gradient(U_next, check_cfl=True, t_idx=time_idx, m_density=M_next)

        # Two stochastic departures per node
        x_drift = self.x_grid - grad_u * dt
        y_plus = x_drift + diffusion_offset
        y_minus = x_drift - diffusion_offset

        # Boundary handling — Issue #1048 fix: properly REFLECT characteristic feet
        # for Neumann BC instead of clamping. Clamping collapsed all out-of-bounds
        # feet onto the boundary node, biasing toward the wall value and breaking
        # upwind property near the boundary.
        bc = self.get_boundary_conditions()
        bc_type_str = get_bc_type_string(bc)
        bc_op = bc_type_to_geometric_operation(bc_type_str)
        bounds = self.problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]
        if bc_op == "reflect":
            # Iterated mirror reflection via modular arithmetic:
            #   y' = xmin + |((y − xmin) mod 2L) − L|   where L = xmax − xmin
            # Maps any y ∈ ℝ to [xmin, xmax] via reflections at xmin and xmax.
            # Handles arbitrary numbers of bounces in a single expression.
            L = xmax - xmin
            y_plus = xmin + np.abs(((y_plus - xmin) % (2 * L)) - L)
            y_minus = xmin + np.abs(((y_minus - xmin) % (2 * L)) - L)
        elif bc_op == "wrap":
            L = xmax - xmin
            y_plus = xmin + (y_plus - xmin) % L
            y_minus = xmin + (y_minus - xmin) % L

        # Issue #1033 + #1049: dispatch interpolation by configured method.
        # - "linear": canonical Carlini-Silva 2014 (Q1, monotone, proof applies)
        # - "cubic":  monotone-preserving Hermite (PchipInterpolator) — replaces
        #             non-monotone CubicSpline that blew up on stiff problems
        # - "quintic": fall back to PchipInterpolator (cubic, monotone) here in 1D;
        #             user gets the warning at __init__ that this is outside the proof.
        if self.interpolation_method == "linear":
            u_plus = np.interp(y_plus, self.x_grid, U_next)
            u_minus = np.interp(y_minus, self.x_grid, U_next)
        else:
            # Issue #1033 fix: PchipInterpolator (monotone Hermite) replaces
            # CubicSpline (non-monotone, blew up on stiff problems). The reflect/
            # wrap branch above keeps all feet inside [xmin, xmax], so disabling
            # extrapolation is safe — any out-of-range query indicates a real bug
            # upstream and should propagate as nan, not be silently masked.
            from scipy.interpolate import PchipInterpolator

            interp_fn = PchipInterpolator(self.x_grid, U_next, extrapolate=False)
            u_plus = interp_fn(y_plus)
            u_minus = interp_fn(y_minus)

        # CS update: average over Brownian directions, subtract dt*H
        u_avg = 0.5 * (u_plus + u_minus)

        x_batch = self.x_grid.reshape(-1, 1)
        p_batch = grad_u.reshape(-1, 1)
        H_class = self.problem.hamiltonian_class
        if H_class is not None:
            H_values = np.asarray(H_class(x_batch, M_next, p_batch, t=time_idx * dt), dtype=float).ravel()
        else:
            H_values = np.zeros(Nx)

        U_current = u_avg - dt * H_values

        # Enforce BC on the result
        if bc:
            time = time_idx * dt
            U_current = self.bc_applicator.enforce_values(
                U_current, boundary_conditions=bc, spacing=(self.dx,), time=time
            )

        return U_current

    def _stochastic_sl_step_nd(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float,
    ) -> np.ndarray:
        """nD stochastic SL step.

        Uses the existing per-point _interpolate_value path (handles cubic /
        quintic via RegularGridInterpolator). The Brownian quadrature is over
        2*d departures (one pair per coordinate axis).
        """
        # Reshape to grid form (matches the Strang-splitting nD path)
        if U_next.ndim == 1:
            total_points = U_next.size
            expected_full = int(np.prod(self._grid_shape))
            if total_points == expected_full:
                grid_shape = tuple(self._grid_shape)
            else:
                grid_shape = tuple(n - 1 for n in self._grid_shape)
            U_next_shaped = U_next.reshape(grid_shape)
            M_next_shaped = M_next.reshape(grid_shape)
        else:
            U_next_shaped = U_next
            M_next_shaped = M_next
            grid_shape = U_next_shaped.shape

        sigma = self.problem.sigma
        if isinstance(sigma, np.ndarray):
            # Diagonal volatility: sigma[k] is the volatility along axis k
            sigma_diag = np.asarray(sigma, dtype=float).ravel()
            if sigma_diag.size != self.dimension:
                raise ValueError(
                    f"Diagonal sigma must have {self.dimension} entries, "
                    f"got {sigma_diag.size}. Full-tensor sigma not yet "
                    f"supported by stochastic SL."
                )
        else:
            sigma_diag = np.full(self.dimension, float(sigma))
        sqrt_dt = float(np.sqrt(dt))

        grad_components = self._compute_gradient(U_next_shaped, check_cfl=True, t_idx=time_idx, m_density=M_next_shaped)

        U_current_shaped = np.zeros_like(U_next_shaped)
        d = self.dimension

        for multi_idx in np.ndindex(grid_shape):
            x_current = np.array([self.grid.coordinates[ax][multi_idx[ax]] for ax in range(d)])
            m_current = M_next_shaped[multi_idx]
            p_optimal = np.array([grad_components[ax][multi_idx] for ax in range(d)])
            x_drift = x_current - p_optimal * dt

            # 2d stochastic departures - one pair per axis
            interp_acc = 0.0
            for ax in range(d):
                offset = np.zeros(d)
                offset[ax] = sigma_diag[ax] * sqrt_dt
                y_plus = x_drift + offset
                y_minus = x_drift - offset
                u_plus = self._interpolate_value(U_next_shaped, y_plus)
                u_minus = self._interpolate_value(U_next_shaped, y_minus)
                interp_acc += 0.5 * (u_plus + u_minus)
            u_avg = interp_acc / d

            H_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)
            U_current_shaped[multi_idx] = u_avg - dt * H_value

        # Enforce BC on the result
        U_current_shaped = self._enforce_boundary_conditions(U_current_shaped)

        if U_next.ndim == 1:
            return U_current_shaped.ravel()
        return U_current_shaped

    # === L-based DPP formulation (Issue #909) ===

    def _solve_timestep_dpp(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        time_idx: int,
        dt: float | None = None,
    ) -> np.ndarray:
        """Solve one timestep via L-based Dynamic Programming Principle.

        u^n(x_i) = min_alpha { dt * L(x_i, alpha, m, t) + u^{n+1}(x_i + alpha*dt) }

        This avoids computing grad_u entirely — the optimization is over the
        control alpha, not the momentum p. Handles non-smooth L naturally:
        - Quadratic: closed-form alpha* = -p/lambda (falls back to H-based equivalent)
        - L1 / bang-bang: compare values at alpha in {-1, 0, 1}
        - Bounded: scalar optimization over [-a_max, a_max]
        - Finite action set: compare K candidate values

        Diffusion is handled identically via operator splitting after the
        advection/optimization step.

        Parameters
        ----------
        U_next : np.ndarray
            Value function at next time step. Shape (Nx,) for 1D, grid_shape for nD.
        M_next : np.ndarray
            Density at next time step (same shape).
        time_idx : int
            Current time index.
        dt : float or None
            Time step. If None, uses self.dt.

        Returns
        -------
        np.ndarray
            Value function at current time step.
        """
        if dt is None:
            dt = self.dt

        L_class = self.problem.lagrangian_class
        t_value = time_idx * self.problem.T / self.problem.Nt

        bounds = L_class.control_bounds() or (-10.0, 10.0)

        if self.dimension == 1:
            Nx = len(U_next)
            U_star = np.zeros(Nx)

            # Detect special structure for fast paths
            from mfgarchon.core.hamiltonian import (
                BoundedControlCost,
                L1ControlCost,
                QuadraticControlCost,
                SeparableLagrangian,
            )

            fast_candidates = None
            if isinstance(L_class, SeparableLagrangian):
                cc = L_class.control_cost
                if isinstance(cc, L1ControlCost):
                    # Bang-bang: compare alpha in {-1, 0, 1}
                    fast_candidates = np.array([-1.0, 0.0, 1.0])
                elif isinstance(cc, BoundedControlCost):
                    # Sample endpoints + zero + a few interior points
                    a_max = cc.max_control
                    fast_candidates = np.linspace(-a_max, a_max, 11)
                elif isinstance(cc, QuadraticControlCost):
                    # Quadratic has closed-form: alpha* = -grad_u / lambda
                    # DPP reduces to H-based SL. Use grad_u path for efficiency.
                    fast_candidates = None  # fall through to scalar optimization

            for i in range(Nx):
                x_i = self.x_grid[i]
                x_arr = np.array([x_i])
                m_i = M_next[i]

                if fast_candidates is not None:
                    # Evaluate DPP cost at each candidate
                    best_val = np.inf
                    for alpha in fast_candidates:
                        x_next = x_i + alpha * dt
                        # Apply boundary handling
                        x_next = self._apply_boundary_to_point(x_next)
                        u_next = self._interpolate_value(U_next, x_next)
                        L_val = float(L_class(x_arr, np.array([alpha]), m_i, t_value))
                        cost = dt * L_val + u_next
                        if cost < best_val:
                            best_val = cost
                    U_star[i] = best_val
                else:
                    # Scalar optimization over alpha
                    def dpp_cost(alpha, _xi=x_i, _xa=x_arr, _mi=m_i):
                        x_next = _xi + alpha * dt
                        x_next = self._apply_boundary_to_point(x_next)
                        u_next = self._interpolate_value(U_next, x_next)
                        L_val = float(L_class(_xa, np.array([alpha]), _mi, t_value))
                        return dt * L_val + u_next

                    result = minimize_scalar(
                        dpp_cost, bounds=bounds, method="bounded", options={"xatol": self.tolerance}
                    )
                    U_star[i] = result.fun

            # Diffusion step (same as H-based)
            U_current = self._apply_diffusion(U_star, dt)

            # Enforce boundary conditions
            bc = self.get_boundary_conditions()
            if bc:
                time = time_idx * self.dt
                U_current = self.bc_applicator.enforce_values(
                    U_current, boundary_conditions=bc, spacing=(self.dx,), time=time
                )

            return U_current

        else:
            # nD DPP
            if U_next.ndim == 1:
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))
                grid_shape = (
                    tuple(self._grid_shape) if total_points == expected_full else tuple(n - 1 for n in self._grid_shape)
                )
                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            U_star = np.zeros_like(U_next_shaped)

            for multi_idx in np.ndindex(grid_shape):
                x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                m_current = M_next_shaped[multi_idx]

                def dpp_cost_nd(alpha_vec, _xc=x_current, _mc=m_current):
                    x_next = _xc + alpha_vec * dt
                    # Clip to domain bounds
                    for d in range(self.dimension):
                        x_next[d] = np.clip(x_next[d], self.grid.bounds[0][d], self.grid.bounds[1][d])
                    u_next = self._interpolate_value(U_next_shaped, x_next)
                    L_val = float(L_class(_xc, alpha_vec, _mc, t_value))
                    return dt * L_val + u_next

                alpha0 = np.zeros(self.dimension)
                result = minimize(
                    dpp_cost_nd,
                    alpha0,
                    bounds=[bounds] * self.dimension,
                    method="L-BFGS-B",
                    options={"ftol": self.tolerance, "maxiter": 100},
                )

                U_star[multi_idx] = result.fun if result.success else self._interpolate_value(U_next_shaped, x_current)

            U_current_shaped = self._apply_diffusion(U_star, dt)
            U_current_shaped = self._enforce_boundary_conditions(U_current_shaped)

            return U_current_shaped.ravel() if U_next.ndim == 1 else U_current_shaped

    def _apply_boundary_to_point(self, x: float) -> float:
        """Apply 1D boundary handling to a single point (reflect or clip)."""
        bounds = self.problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]

        bc = self.get_boundary_conditions()
        bc_type = get_bc_type_string(bc)
        bc_op = bc_type_to_geometric_operation(bc_type)

        return apply_boundary_conditions_1d(x, xmin=xmin, xmax=xmax, bc_type=bc_op)

    @property
    def _use_dpp(self) -> bool:
        """Whether to use L-based DPP instead of H-based characteristics.

        Uses DPP when:
        1. problem.lagrangian_class is available, AND
        2. Either no hamiltonian_class, or the Lagrangian's control cost is non-smooth
        """
        L_class = self.problem.lagrangian_class
        if L_class is None:
            return False

        # If no H available, DPP is the only option
        H_class = self.problem.hamiltonian_class
        if H_class is None:
            return True

        # If H is non-smooth, prefer DPP (avoids grad_u at kinks)
        is_smooth = getattr(H_class, "is_smooth", lambda: True)
        return bool(callable(is_smooth) and not is_smooth())

    def _find_optimal_control(self, x: np.ndarray | float, m: float, time_idx: int) -> np.ndarray | float:
        """
        Find optimal control p* that minimizes H(x, p, m) (supports 1D and nD).

        For the standard MFG Hamiltonian H(x, p, m) = |p|²/2 + V(x) + C(x,m),
        the optimal control is p* = 0 in all dimensions.

        For general Hamiltonians:
        - 1D: Uses scalar optimization (minimize_scalar with Brent/Golden search)
        - nD: Uses vector optimization (scipy.optimize.minimize with L-BFGS-B)

        Args:
            x: Spatial position
                - 1D: scalar float
                - nD: array of shape (dimension,)
            m: Density value
            time_idx: Time index

        Returns:
            Optimal control value p*
                - 1D: scalar float
                - nD: array of shape (dimension,)
        """
        # Issue #902: Infer control bounds from Lagrangian or Hamiltonian
        _bounds = (-10.0, 10.0)  # default
        L_class = self.problem.lagrangian_class
        if L_class is not None:
            cb = L_class.control_bounds()
            if cb is not None:
                _bounds = cb

        # Issue #902: Use HamiltonianBase.optimal_control() if available.
        # Note: _find_optimal_control finds p* that MINIMIZES H(x, p, m),
        # NOT the MFG drift alpha*. For quadratic H = |p|^2/2 + ..., p* = 0.
        # This is used for characteristic tracing, not for FP drift.

        # Priority 1: hamiltonian_class — if H has a known minimum (e.g., quadratic at p=0)
        H_class = self.problem.hamiltonian_class
        if H_class is not None:
            # For separable H: minimum of H_control(p) w.r.t. p
            # Quadratic: dp=0 at p=0. L1: H=0 for |p|<=lambda, min at p=0.
            # General: use scipy.
            try:
                _ = H_class.control_cost  # check if separable (has control_cost)
                # ControlCostBase: minimum of evaluate(p) is at p=0 for all standard costs
                if self.dimension == 1:
                    return 0.0
                return np.zeros(self.dimension)
            except AttributeError:
                pass  # Not separable, fall through to optimization

        if self.dimension == 1:
            x_scalar = float(x) if np.ndim(x) > 0 else x

            # Legacy: quadratic shortcut
            try:
                _ = self.problem.coupling_coefficient
                return 0.0
            except AttributeError:
                pass

            # Numerical optimization using _evaluate_hamiltonian
            def hamiltonian_objective(p):
                return self._evaluate_hamiltonian(x_scalar, p, m, time_idx)

            if self.optimization_method == "brent":
                result = minimize_scalar(
                    hamiltonian_objective,
                    bounds=_bounds,
                    method="bounded",
                    options={"xatol": self.tolerance},
                )
            else:
                result = minimize_scalar(
                    hamiltonian_objective,
                    bounds=_bounds,
                    method="golden",
                    options={"xtol": self.tolerance},
                )

            return result.x if result.success else 0.0

        else:
            # Legacy: quadratic shortcut
            try:
                _ = self.problem.coupling_coefficient
                return np.zeros(self.dimension)
            except AttributeError:
                pass

            def hamiltonian_objective(p_vec):
                return self._evaluate_hamiltonian(x, p_vec, m, time_idx)

            p0 = np.zeros(self.dimension)

            try:
                # Use L-BFGS-B for smooth, unconstrained optimization
                result = minimize(
                    hamiltonian_objective,
                    p0,
                    method="L-BFGS-B",
                    options={"ftol": self.tolerance, "maxiter": 100},
                )

                if result.success:
                    return result.x
                else:
                    logger.debug(f"Vector optimization did not converge: {result.message}")
                    return p0

            except Exception as e:
                logger.debug(f"Vector optimization failed at x={x}: {e}")
                return p0

    def _trace_characteristic_backward(
        self, x_current: np.ndarray | float, p_optimal: np.ndarray | float, dt: float
    ) -> np.ndarray | float:
        """
        Trace characteristic backward in time to find departure point (supports 1D and nD).

        Delegates to hjb_sl_characteristics module functions.

        Args:
            x_current: Current spatial position
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [x, y] for 2D
            p_optimal: Optimal control value
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [px, py] for 2D
            dt: Time step size

        Returns:
            Departure point X(t-dt)
                - 1D: scalar float
                - nD: array of shape (dimension,)
        """
        if self.dimension == 1:
            # 1D characteristic tracing
            jax_fn = self._jax_solve_characteristic if self.use_jax else None
            x_departure = trace_characteristic_backward_1d(
                x_current,
                p_optimal,
                dt,
                method=self.characteristic_solver,
                use_jax=self.use_jax,
                jax_solve_fn=jax_fn,
                ode_rtol=self.ode_rtol,
                ode_atol=self.ode_atol,
            )

            # Apply boundary conditions
            # Issue #702: Use centralized bc_utils for consistent BC handling
            bc = self.get_boundary_conditions()
            bc_type = get_bc_type_string(bc)
            bc_op = bc_type_to_geometric_operation(bc_type)

            bounds = self.problem.geometry.get_bounds()
            xmin, xmax = bounds[0][0], bounds[1][0]
            return apply_boundary_conditions_1d(
                x_departure,
                xmin=xmin,
                xmax=xmax,
                bc_type=bc_op,
            )

        else:
            # nD characteristic tracing
            x_departure = trace_characteristic_backward_nd(
                x_current,
                p_optimal,
                dt,
                dimension=self.dimension,
                method=self.characteristic_solver,
                ode_rtol=self.ode_rtol,
                ode_atol=self.ode_atol,
            )

            # Issue #702: Use centralized bc_utils for consistent BC handling
            bc = self.get_boundary_conditions()
            bc_type = get_bc_type_string(bc)
            bc_op = bc_type_to_geometric_operation(bc_type)

            return apply_boundary_conditions_nd(
                x_departure,
                bounds=self.grid.bounds,
                bc_type=bc_op,
            )

    def _interpolate_value(self, U_values: np.ndarray, x_query: np.ndarray | float) -> float:
        """
        Interpolate value function at query point (supports 1D and nD).

        Delegates to hjb_sl_interpolation module functions.

        Args:
            U_values: Value function on grid
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points, e.g., (Nx, Ny) for 2D
            x_query: Query point for interpolation
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [x, y] for 2D

        Returns:
            Interpolated value at query point
        """
        if self.dimension == 1:
            # 1D interpolation
            jax_fn = self._jax_interpolate if self.use_jax else None
            bounds = self.problem.geometry.get_bounds()
            xmin, xmax = bounds[0][0], bounds[1][0]
            return interpolate_value_1d(
                U_values,
                x_query,
                self.x_grid,
                method=self.interpolation_method,
                xmin=xmin,
                xmax=xmax,
                use_jax=self.use_jax,
                jax_interpolate_fn=jax_fn,
            )

        else:
            # nD interpolation
            grid_coords = tuple(self.grid.coordinates)
            grid_shape = tuple(self._grid_shape)

            try:
                return interpolate_value_nd(
                    U_values,
                    x_query,
                    grid_coords,
                    grid_shape,
                    method=self.interpolation_method,
                )
            except Exception as e:
                logger.debug(f"nD interpolation failed at x={x_query}: {e}")

                # Try RBF fallback if enabled
                if self.use_rbf_fallback:
                    try:
                        return interpolate_value_rbf_fallback(
                            U_values,
                            x_query,
                            grid_coords,
                            grid_shape,
                            rbf_kernel=self.rbf_kernel,
                        )
                    except Exception as rbf_error:
                        logger.debug(f"RBF fallback failed: {rbf_error}")

                # Final fallback: nearest neighbor
                return interpolate_nearest_neighbor(
                    U_values,
                    x_query,
                    grid_coords,
                    grid_shape,
                )

    def _compute_diffusion_term(self, U_values: np.ndarray, idx: int | tuple) -> float:
        """
        Compute discrete Laplacian (diffusion term) at grid point (supports 1D and nD).

        1D: Uses standard finite difference (U[i+1] - 2*U[i] + U[i-1]) / dx²
        nD: Computes Laplacian as sum over dimensions: Δu = Σ_d ∂²u/∂x_d²

        Args:
            U_values: Value function array
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points
            idx: Grid point index
                - 1D: scalar integer i
                - nD: tuple of indices, e.g., (i, j) for 2D

        Returns:
            Discrete Laplacian value
        """
        if self.dimension == 1:
            # 1D Laplacian: Use existing logic
            i = int(idx)
            Nx = len(U_values)

            if Nx <= 2:
                return 0.0

            # Handle boundary points - get BC type once
            # Issue #545: Use centralized BC retrieval (NO hasattr)
            bc = self.get_boundary_conditions()
            bc_type = self._get_bc_type_string(bc)

            if i == 0:
                if bc_type == "periodic":
                    laplacian = (U_values[1] - 2 * U_values[0] + U_values[-1]) / self.dx**2
                else:
                    laplacian = (U_values[1] - U_values[0]) / self.dx**2

            elif i == Nx - 1:
                if bc_type == "periodic":
                    laplacian = (U_values[0] - 2 * U_values[-1] + U_values[-2]) / self.dx**2
                else:
                    laplacian = (U_values[-1] - U_values[-2]) / self.dx**2

            else:
                # Central difference for interior points
                laplacian = (U_values[i + 1] - 2 * U_values[i] + U_values[i - 1]) / self.dx**2

            return laplacian

        else:
            # nD Laplacian: Sum of second derivatives in each dimension
            # Δu = ∂²u/∂x₁² + ∂²u/∂x₂² + ...

            # Ensure U_values is reshaped to grid shape
            if U_values.ndim == 1:
                U_shaped = U_values.reshape(self._grid_shape)
            else:
                U_shaped = U_values

            # Get multi-index
            if isinstance(idx, (tuple, list)):
                multi_idx = tuple(idx)
            else:
                # Convert flat index to multi-index
                multi_idx = self.grid.get_multi_index(int(idx))

            laplacian = 0.0

            # Compute second derivative in each dimension
            for d in range(self.dimension):
                # Check if we're at a boundary in this dimension
                at_lower_bound = multi_idx[d] == 0
                at_upper_bound = multi_idx[d] == self._grid_shape[d] - 1

                # Create index tuples for neighbors
                idx_center = list(multi_idx)
                idx_plus = list(multi_idx)
                idx_minus = list(multi_idx)

                if at_lower_bound or at_upper_bound:
                    # Boundary: use one-sided difference (assume Neumann BC)
                    if at_lower_bound:
                        idx_plus[d] = multi_idx[d] + 1
                        u_center = U_shaped[tuple(idx_center)]
                        u_plus = U_shaped[tuple(idx_plus)]
                        # One-sided: (u_plus - u_center) / dx²
                        second_deriv = (u_plus - u_center) / self.dx[d] ** 2
                    else:  # at_upper_bound
                        idx_minus[d] = multi_idx[d] - 1
                        u_center = U_shaped[tuple(idx_center)]
                        u_minus = U_shaped[tuple(idx_minus)]
                        # One-sided: (u_center - u_minus) / dx²
                        second_deriv = (u_center - u_minus) / self.dx[d] ** 2

                else:
                    # Interior: central difference
                    idx_plus[d] = multi_idx[d] + 1
                    idx_minus[d] = multi_idx[d] - 1

                    u_center = U_shaped[tuple(idx_center)]
                    u_plus = U_shaped[tuple(idx_plus)]
                    u_minus = U_shaped[tuple(idx_minus)]

                    # Central: (u_plus - 2*u_center + u_minus) / dx²
                    second_deriv = (u_plus - 2 * u_center + u_minus) / self.dx[d] ** 2

                laplacian += second_deriv

            return float(laplacian)

    def _evaluate_hamiltonian(self, x: np.ndarray | float, p: np.ndarray | float, m: float, time_idx: int) -> float:
        """
        Evaluate Hamiltonian H(x, p, m) at given point (supports 1D and nD).

        Uses DerivativeTensors for consistency with all solvers.
        See docs/NAMING_CONVENTIONS.md "Derivative Tensor Standard" section.

        Args:
            x: Spatial position
                - 1D: scalar float
                - nD: array of shape (dimension,)
            p: Control/momentum value (gradient ∇u)
                - 1D: scalar float
                - nD: array of shape (dimension,)
            m: Density value
            time_idx: Time index

        Returns:
            Hamiltonian value
        """
        # Compute time value
        t_value = time_idx * self.problem.T / self.problem.Nt if time_idx is not None else 0.0

        # Issue #902: Use HamiltonianBase class API first
        H_class = self.problem.hamiltonian_class
        if H_class is not None:
            x_vec = np.atleast_1d(x)
            p_vec = np.atleast_1d(p)
            return float(H_class(x_vec, m, p_vec, t_value))

        # Legacy fallbacks for problems without hamiltonian_class
        derivs = self._build_derivative_tensors(p)
        x_idx = self._position_to_index(x)

        try:
            return self.problem.H(x_idx, m, derivs=derivs, t_idx=time_idx)
        except AttributeError:
            pass

        try:
            return self.problem.hamiltonian(x_idx, m, derivs=derivs, t=t_value)
        except (AttributeError, TypeError):
            pass

        try:
            return self.problem.hamiltonian(np.atleast_1d(x), m, np.atleast_1d(p), t_value)
        except (AttributeError, TypeError) as e:
            logger.debug(f"Legacy Hamiltonian signature failed: {e}")

        return self._default_hamiltonian(derivs, m)

    def _build_derivative_tensors(self, p: np.ndarray | float) -> DerivativeTensors:
        """
        Build DerivativeTensors from gradient array/scalar.

        Args:
            p: Gradient value(s)
                - 1D: scalar float
                - nD: array of shape (dimension,)

        Returns:
            DerivativeTensors with gradient tensor
        """
        if self.dimension == 1:
            p_scalar = float(p) if np.ndim(p) > 0 else p
            grad = np.array([p_scalar])
        else:
            grad = np.atleast_1d(p).astype(float)

        return DerivativeTensors.from_gradient(grad)

    def _position_to_index(self, x: np.ndarray | float) -> int | tuple[int, ...]:
        """
        Convert spatial position to grid index.

        Args:
            x: Spatial position

        Returns:
            Grid index (int for 1D, tuple for nD)
        """
        # Get bounds from geometry (preferred) or legacy xmin/xmax
        # Issue #545: Use try/except instead of hasattr
        try:
            bounds = self.problem.geometry.bounds if self.problem.geometry is not None else None
        except AttributeError:
            bounds = None
        grid_shape = self.problem.geometry.get_grid_shape() if bounds is not None else None

        if self.dimension == 1:
            x_scalar = float(x) if np.ndim(x) > 0 else x
            if bounds is not None:
                xmin = bounds[0][0]
                Nx = grid_shape[0] - 1
            else:
                geom_bounds = self.problem.geometry.get_bounds()
                xmin = geom_bounds[0][0]
                Nx = self.problem.geometry.get_grid_shape()[0] - 1
            # dx is scalar for 1D
            dx = self.dx if np.isscalar(self.dx) else self.dx[0]
            x_idx = int((x_scalar - xmin) / dx)
            return int(np.clip(x_idx, 0, Nx))
        else:
            x_vec = np.atleast_1d(x)
            indices = []
            for i in range(self.dimension):
                if bounds is not None:
                    xmin_i = bounds[i][0]
                    Nx_i = grid_shape[i] - 1
                else:
                    geom_bounds = self.problem.geometry.get_bounds()
                    xmin_i = geom_bounds[0][i]
                    Nx_i = self.problem.geometry.get_grid_shape()[i] - 1
                # dx is array for nD
                dx_i = self.dx[i]
                idx = int((x_vec[i] - xmin_i) / dx_i)
                indices.append(int(np.clip(idx, 0, Nx_i)))
            return tuple(indices)

    def _default_hamiltonian(self, derivs: DerivativeTensors, m: float) -> float:
        """
        Default quadratic Hamiltonian H = |p|²/2 + C*m.

        Args:
            derivs: DerivativeTensors with gradient
            m: Density value

        Returns:
            Hamiltonian value
        """
        coef_CT = getattr(self.problem, "coupling_coefficient", 0.5)
        return 0.5 * derivs.grad_norm_squared + coef_CT * m

    def _solve_crank_nicolson_diffusion(self, U_star: np.ndarray, dt: float, sigma: float) -> np.ndarray:
        """
        Solve diffusion step using Crank-Nicolson (unconditionally stable).

        Delegates to hjb_sl_adi.solve_crank_nicolson_diffusion_1d.

        Args:
            U_star: Intermediate solution after advection step
            dt: Time step size
            sigma: Diffusion coefficient

        Returns:
            Solution after implicit diffusion step
        """
        bc_op = self._get_diffusion_bc_type()
        return solve_crank_nicolson_diffusion_1d(U_star, dt, sigma, self.x_grid, bc_type=bc_op)

    def _get_diffusion_bc_type(self) -> str:
        """Get BC type string for diffusion step ('neumann' or 'periodic')."""
        bc = self.get_boundary_conditions()
        bc_type = get_bc_type_string(bc)
        if bc_type == "periodic":
            return "periodic"
        return "neumann"

    def _adi_diffusion_step(self, U_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply ADI (Alternating Direction Implicit) diffusion for nD grids.

        Delegates to hjb_sl_adi.adi_diffusion_step.

        Args:
            U_star: Intermediate solution after advection step, shape (N1, N2, ..., Nd)
            dt: Time step size

        Returns:
            Solution after ADI diffusion step, same shape as U_star
        """
        if self.dimension == 1:
            # For 1D, use standard Crank-Nicolson
            return self._solve_crank_nicolson_diffusion(U_star, dt, self.problem.sigma)

        bc_op = self._get_diffusion_bc_type()
        return adi_diffusion_step(
            U_star,
            dt,
            self.problem.sigma,
            self.dx,
            tuple(self._grid_shape),
            bc_type=bc_op,
        )

    def _apply_diffusion(self, U_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply diffusion step using the configured method.

        Args:
            U_star: Solution after advection step
            dt: Time step size

        Returns:
            Solution after diffusion step
        """
        if self.diffusion_method == "none":
            # No diffusion - just return advected solution
            return U_star

        elif self.diffusion_method == "explicit":
            # Explicit Laplacian: u^n = u* + dt * σ²/2 * Δu*
            # Simple but requires small dt for stability (dt < dx²/(2*d*σ²))
            return self._explicit_diffusion_step(U_star, dt)

        elif self.diffusion_method == "stochastic":
            # Issue #1026: stochastic SL bypasses _apply_diffusion entirely; the
            # Carlini-Silva update bakes diffusion into the SL averaging step.
            # Reaching this branch indicates broken dispatch in
            # _solve_timestep_semi_lagrangian.
            raise NotImplementedError(
                "_apply_diffusion should not be called when diffusion_method='stochastic'. "
                "The Carlini-Silva 2014 SL update incorporates diffusion via 2d "
                "stochastic departure points, replacing the operator-splitting "
                "diffusion step. Check dispatch in _solve_timestep_semi_lagrangian."
            )

        else:  # "adi" (default)
            if self.dimension == 1:
                return self._solve_crank_nicolson_diffusion(U_star, dt, self.problem.sigma)
            else:
                return self._adi_diffusion_step(U_star, dt)

    def _explicit_diffusion_step(self, U_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply explicit diffusion step using discrete Laplacian.

        Uses central differences for Laplacian:
            Δu ≈ Σ_d (u_{i+1,d} - 2*u_i + u_{i-1,d}) / dx_d²

        Note: This is conditionally stable. Requires:
            dt < dx²/(2*d*σ²) where d is dimension

        Args:
            U_star: Solution after advection step
            dt: Time step size

        Returns:
            Solution after explicit diffusion step
        """
        sigma = self.problem.sigma
        sigma_sq_half = 0.5 * sigma**2

        if self.dimension == 1:
            dx = self.dx
            # 1D Laplacian with Neumann BC
            laplacian = np.zeros_like(U_star)
            laplacian[1:-1] = (U_star[2:] - 2 * U_star[1:-1] + U_star[:-2]) / dx**2
            # Neumann BC: du/dx = 0 at boundaries
            laplacian[0] = (U_star[1] - U_star[0]) / dx**2
            laplacian[-1] = (U_star[-2] - U_star[-1]) / dx**2
        else:
            # nD Laplacian
            laplacian = np.zeros_like(U_star)
            for d in range(self.dimension):
                dx_d = self.dx[d]
                # Second derivative along axis d
                laplacian += np.gradient(np.gradient(U_star, dx_d, axis=d), dx_d, axis=d)

        # Explicit update: u^n = u* + dt * σ²/2 * Δu*
        U_new = U_star + dt * sigma_sq_half * laplacian
        return U_new

    # ═══════════════════════════════════════════════════════════════════
    # BoundaryHandler Protocol (Issue #545)
    # ═══════════════════════════════════════════════════════════════════

    def get_boundary_indices(self) -> np.ndarray:
        """
        Identify boundary points in solver's discretization.

        Returns:
            Array of integer indices identifying boundary grid points.

        Note:
            For Semi-Lagrangian method, boundary points are those on the
            domain boundary where characteristic tracing requires clamping.

        Implementation:
            - 1D: First and last grid points [0, N-1]
            - nD: All points on boundary faces (any coordinate at min/max)
        """
        if self.dimension == 1:
            # 1D: First and last grid points
            Nx = len(self.x_grid)
            return np.array([0, Nx - 1], dtype=np.int64)
        else:
            # nD: All grid points on boundary faces
            # For tensor grid: point is on boundary if any coordinate is at min/max
            boundary_mask = np.zeros(self._grid_shape, dtype=bool)

            # Mark boundary faces in each dimension
            for d in range(self.dimension):
                # Lower boundary face (index 0 along dimension d)
                slices_lower = [slice(None)] * self.dimension
                slices_lower[d] = 0
                boundary_mask[tuple(slices_lower)] = True

                # Upper boundary face (index -1 along dimension d)
                slices_upper = [slice(None)] * self.dimension
                slices_upper[d] = -1
                boundary_mask[tuple(slices_upper)] = True

            # Return flat indices of boundary points
            return np.flatnonzero(boundary_mask.ravel())

    def apply_boundary_conditions(
        self,
        values: np.ndarray,
        bc: BoundaryConditions,
        time: float = 0.0,
    ) -> np.ndarray:
        """
        Apply boundary conditions to solution values.

        For Semi-Lagrangian method, BC enforcement is handled during
        characteristic tracing (clamping departure points to domain).
        This method is a no-op adapter for protocol compliance.

        Args:
            values: Solution values at all grid points
            bc: Boundary conditions object (from mfgarchon.geometry.boundary)
            time: Current time (unused for Semi-Lagrangian)

        Returns:
            Solution values (unchanged, BC enforced during characteristic tracing)

        Note:
            Semi-Lagrangian enforces BCs during characteristic tracing via
            hjb_sl_characteristics.apply_boundary_conditions_1d/nd(), not as
            a post-processing step. This method exists for protocol compliance.
        """
        # Semi-Lagrangian enforces BCs during characteristic tracing,
        # not as a post-processing step. Return values unchanged.
        return values

    def get_bc_type_for_point(self, point_idx: int) -> str:
        """
        Determine BC type for a specific grid point.

        Args:
            point_idx: Index of grid point

        Returns:
            BC type string: "periodic", "dirichlet", "neumann", or "none"

        Note:
            For Semi-Lagrangian solver with uniform BCs, returns the same
            BC type for all boundary points. Mixed BC support would require
            querying BC segments based on point spatial coordinates.
        """
        bc = self.get_boundary_conditions()
        if bc is None:
            return "none"

        # For uniform BC (most common case)
        bc_type_str = self._get_bc_type_string(bc)
        if bc_type_str is not None:
            return bc_type_str

        # For mixed BC, would need to query BC segments
        # (Semi-Lagrangian typically uses uniform BC)
        return "none"

    def get_solver_info(self) -> dict[str, Any]:
        """Return solver configuration information."""
        return {
            "method": "Semi-Lagrangian",
            "interpolation": self.interpolation_method,
            "optimization": self.optimization_method,
            "characteristic_solver": self.characteristic_solver,
            "use_jax": self.use_jax,
            "tolerance": self.tolerance,
            "max_iterations": self.max_char_iterations,
            "adaptive_substepping": self.enable_adaptive_substepping,
            "max_substeps": self.max_substeps,
            "cfl_target": self.cfl_target,
        }


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBSemiLagrangianSolver...")

    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    def _smoke_components_1d():
        """MFGComponents for the 1D smoke tests."""
        return MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
        )

    def _smoke_components_2d():
        """MFGComponents for the 2D smoke tests."""
        return MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
        )

    # Test 1: Solver initialization
    print("\n1. Testing solver initialization...")
    geometry_1d = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[51],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    problem = MFGProblem(
        geometry=geometry_1d,
        T=1.0,
        Nt=100,
        diffusion=0.5 * 0.1**2,
        components=_smoke_components_1d(),
    )
    solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", optimization_method="brent")

    assert solver.dimension == 1
    assert solver.hjb_method_name == "Semi-Lagrangian"
    assert solver.interpolation_method == "linear"
    print("   1D solver initialization: OK")

    # Test 2: 1D Crank-Nicolson diffusion (used by 1D solver)
    print("\n2. Testing 1D Crank-Nicolson diffusion...")
    # Create a smooth test function (Gaussian)
    x = np.linspace(0, 1, 51)
    U_test = np.exp(-50 * (x - 0.5) ** 2)

    # Apply diffusion for one timestep
    dt = 0.01
    sigma = 0.1
    U_diffused = solver._solve_crank_nicolson_diffusion(U_test, dt, sigma)

    assert U_diffused.shape == U_test.shape
    assert not np.any(np.isnan(U_diffused))
    assert not np.any(np.isinf(U_diffused))
    # Diffusion should smooth the peak
    assert U_diffused.max() < U_test.max()
    print(f"   Peak before diffusion: {U_test.max():.4f}")
    print(f"   Peak after diffusion: {U_diffused.max():.4f}")
    print("   1D Crank-Nicolson: OK")

    # Test 3: 2D solver initialization with ADI compatibility check
    print("\n3. Testing 2D solver with ADI...")

    geometry_2d = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[20, 20],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    problem_2d = MFGProblem(
        geometry=geometry_2d,
        T=0.5,
        Nt=50,
        diffusion=0.5 * 0.1**2,
        components=_smoke_components_2d(),
    )

    solver_2d = HJBSemiLagrangianSolver(problem_2d, interpolation_method="linear")

    assert solver_2d.dimension == 2
    # Issue #545: Direct attribute access in test code (will raise AttributeError if missing)
    assert solver_2d._adi_compatible  # Scalar sigma should be ADI compatible
    print(f"   ADI compatible: {solver_2d._adi_compatible}")
    print("   2D solver initialization: OK")

    # Test 4: ADI diffusion step directly
    print("\n4. Testing ADI diffusion step...")
    # Create 2D Gaussian test function
    grid_shape = tuple(solver_2d._grid_shape)
    x = np.linspace(0, 1, grid_shape[0])
    y = np.linspace(0, 1, grid_shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    U_2d_test = np.exp(-50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    # Apply ADI diffusion
    U_2d_diffused = solver_2d._adi_diffusion_step(U_2d_test, dt=0.01)

    assert U_2d_diffused.shape == U_2d_test.shape
    assert not np.any(np.isnan(U_2d_diffused))
    assert not np.any(np.isinf(U_2d_diffused))
    # Diffusion should smooth the peak
    assert U_2d_diffused.max() < U_2d_test.max()
    print(f"   Peak before ADI diffusion: {U_2d_test.max():.4f}")
    print(f"   Peak after ADI diffusion: {U_2d_diffused.max():.4f}")
    print("   ADI diffusion step: OK")

    # Test 5: ADI preserves mass (integral)
    print("\n5. Testing ADI mass conservation...")
    dx_2d = solver_2d.dx  # Grid spacing array
    mass_before = np.sum(U_2d_test) * dx_2d[0] * dx_2d[1]
    mass_after = np.sum(U_2d_diffused) * dx_2d[0] * dx_2d[1]
    mass_error = abs(mass_after - mass_before) / mass_before
    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after: {mass_after:.6f}")
    print(f"   Relative error: {mass_error:.2e}")
    # With Neumann BC, mass should be approximately conserved
    assert mass_error < 0.05, f"Mass error too large: {mass_error}"
    print("   Mass conservation: OK")

    # Test 6: ADI with anisotropic sigma (diagonal tensor) - SKIPPED
    # The diagonal-sigma API was changed to require a spatial field (Nx, Ny) or
    # (Nt, Nx, Ny) rather than a per-axis vector (d,). Restoring this smoke test
    # requires migrating the volatility-field API; out of scope here.
    print("\n6. Anisotropic ADI smoke test SKIPPED (volatility-field API change)")

    # Test 7: BoundaryHandler protocol compliance (Issue #545)
    print("\n7. Testing BoundaryHandler protocol...")
    from mfgarchon.geometry.boundary import validate_boundary_handler

    # Validate protocol compliance
    assert validate_boundary_handler(solver), "1D solver should implement BoundaryHandler"
    assert validate_boundary_handler(solver_2d), "2D solver should implement BoundaryHandler"
    print("   Protocol validation: OK")

    # Test get_boundary_indices()
    boundary_indices_1d = solver.get_boundary_indices()
    assert len(boundary_indices_1d) == 2, "1D should have 2 boundary points"
    assert boundary_indices_1d[0] == 0, "First boundary point should be 0"
    assert boundary_indices_1d[-1] == 50, "Last boundary point should be Nx-1"
    print(f"   1D boundary indices: {boundary_indices_1d}")

    boundary_indices_2d = solver_2d.get_boundary_indices()
    assert len(boundary_indices_2d) > 0, "2D should have boundary points"
    print(f"   2D boundary count: {len(boundary_indices_2d)}")

    # Test get_bc_type_for_point()
    bc_type = solver.get_bc_type_for_point(0)
    valid_bc_types = ["periodic", "dirichlet", "neumann", "no_flux", "robin", "none"]
    assert bc_type in valid_bc_types, f"Invalid BC type: {bc_type}"
    print(f"   BC type for point 0: {bc_type}")

    # Test apply_boundary_conditions() (no-op adapter)
    U_test_1d = np.ones(51)
    U_result = solver.apply_boundary_conditions(U_test_1d, None)
    assert np.array_equal(U_result, U_test_1d), "apply_boundary_conditions should be no-op for Semi-Lagrangian"
    print("   apply_boundary_conditions (no-op): OK")

    print("   BoundaryHandler protocol: OK")

    # Test 8: Gradient clipping (Issue #583)
    print("\n8. Testing gradient clipping (Issue #583)...")
    solver_clipped = HJBSemiLagrangianSolver(
        problem,
        gradient_clip_threshold=1e6,
        enable_gradient_monitoring=True,
    )
    assert solver_clipped.gradient_clip_threshold == 1e6, "Clip threshold not set"
    assert solver_clipped.enable_gradient_monitoring, "Monitoring not enabled"
    print(f"   Clip threshold: {solver_clipped.gradient_clip_threshold:.0e}")

    # Test 1D clipping
    test_grad = np.array([1e5, 2e6, 5e5, 3e6, 1e4])  # 2e6, 3e6 exceed threshold
    solver_clipped._reset_gradient_stats()
    clipped_grad = solver_clipped._clip_gradient_with_monitoring(test_grad, t_idx=0, m_density=np.ones(5))
    assert np.max(np.abs(clipped_grad)) <= 1e6, "1D clipping failed"
    assert solver_clipped.gradient_stats["count"] == 2, "Expected 2 clips"
    print(f"   1D clipping: {solver_clipped.gradient_stats['count']} points clipped, OK")

    # Test 2D clipping with direction preservation (use 2D solver)
    solver_clipped_2d = HJBSemiLagrangianSolver(
        problem_2d,  # Use 2D problem
        gradient_clip_threshold=1e6,
        enable_gradient_monitoring=True,
    )
    grid_shape = (5, 5)
    grad_x = np.ones(grid_shape) * 1e5
    grad_y = np.ones(grid_shape) * 1e5
    grad_x[2, 2] = 3e6
    grad_y[2, 2] = 4e6  # Norm = 5e6 at (2,2)
    solver_clipped_2d._reset_gradient_stats()
    clipped_x, clipped_y = solver_clipped_2d._clip_gradient_with_monitoring(
        (grad_x, grad_y), t_idx=0, m_density=np.ones(grid_shape)
    )
    norm_after = np.sqrt(clipped_x[2, 2] ** 2 + clipped_y[2, 2] ** 2)
    assert norm_after <= 1e6 + 1e-6, "2D clipping failed"
    # Check direction preserved (3:4 ratio)
    assert abs(clipped_x[2, 2] / clipped_y[2, 2] - 0.75) < 1e-10, "Direction not preserved"
    print("   2D clipping with direction preservation: OK")

    # Test no-clip path
    solver_no_clip = HJBSemiLagrangianSolver(problem, gradient_clip_threshold=None)
    result = solver_no_clip._clip_gradient_with_monitoring(test_grad)
    assert np.array_equal(result, test_grad), "No-clip path failed"
    print("   No-clip path (threshold=None): OK")

    print("   Gradient clipping (Issue #583): OK")

    # Test 9: Carlini-Silva stochastic SL (Issue #1026)
    print("\n9. Testing Carlini-Silva stochastic SL (Issue #1026)...")

    # 9a: linear + stochastic rejected at __init__
    try:
        HJBSemiLagrangianSolver(problem, interpolation_method="linear", diffusion_method="stochastic")
        raise AssertionError("Expected ValueError for linear + stochastic")
    except ValueError:
        print("   9a: linear + stochastic rejected at __init__: OK")

    # 9b: cubic + stochastic instantiates and dispatches correctly
    solver_cs = HJBSemiLagrangianSolver(
        problem,
        interpolation_method="cubic",
        diffusion_method="stochastic",
        check_cfl=False,
    )
    assert solver_cs.diffusion_method == "stochastic"
    print("   9b: cubic + stochastic instantiated: OK")

    # 9c: _apply_diffusion raises NotImplementedError under stochastic
    try:
        solver_cs._apply_diffusion(np.zeros(11), 0.01)
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError:
        print("   9c: _apply_diffusion raises under stochastic: OK")

    # 9d: constant terminal -> constant solution (no-drift, no-curvature sanity)
    from mfgarchon.core.hamiltonian import HamiltonianBase, OptimizationSense

    class _ZeroH(HamiltonianBase):
        def __init__(self):
            super().__init__(sense=OptimizationSense.MINIMIZE)

        def __call__(self, x, m, p, t=0.0):
            p_arr = np.atleast_1d(np.asarray(p, dtype=float))
            return np.zeros(p_arr.shape[:-1]) if p_arr.ndim > 0 else 0.0

        def gradient_p(self, x, m, p, t=0.0):
            return np.zeros_like(np.asarray(p, dtype=float))

        def density_derivative(self, x, m, p, t=0.0):
            return 0.0

    grid_const = TensorProductGrid(
        dimension=1,
        bounds=[(-1.0, 1.0)],
        Nx_points=[51],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    components_const = MFGComponents(
        hamiltonian=_ZeroH(),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 1.0,
    )
    problem_const = MFGProblem(
        geometry=grid_const,
        T=0.1,
        Nt=20,
        diffusion=0.045,
        components=components_const,
    )
    solver_const = HJBSemiLagrangianSolver(
        problem_const,
        interpolation_method="cubic",
        diffusion_method="stochastic",
        check_cfl=False,
    )
    U_T_const = np.ones(51)
    M_dummy = np.ones((problem_const.Nt + 1, 51))
    U_const = solver_const.solve_hjb_system(
        M_density=M_dummy,
        U_terminal=U_T_const,
        U_coupling_prev=np.zeros((problem_const.Nt + 1, 51)),
    )
    assert np.allclose(U_const[0], 1.0, atol=1e-10), (
        f"Constant terminal not preserved: U[0] range [{U_const[0].min()}, {U_const[0].max()}]"
    )
    print("   9d: constant terminal preserved (H=0, sigma>0): OK")

    # 9e: stochastic and ADI produce numerically equivalent results on H=0
    #     Gaussian backward heat. Both schemes solve the same equation;
    #     only their discretization paths differ.
    sigma_test = 0.3
    T_test = 0.5
    beta_T = 1.0
    N_test, Nt_test = 100, 200
    grid_g = TensorProductGrid(
        dimension=1,
        bounds=[(-5.0, 5.0)],
        Nx_points=[N_test + 1],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    x_g = grid_g.get_spatial_grid().flatten()
    components_g = MFGComponents(
        hamiltonian=_ZeroH(),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: float(np.exp(-(x[0] ** 2) / (2 * beta_T)) / np.sqrt(2 * np.pi * beta_T)),
    )
    problem_g = MFGProblem(
        geometry=grid_g,
        T=T_test,
        Nt=Nt_test,
        diffusion=sigma_test**2 / 2,
        components=components_g,
    )
    U_T_g = np.exp(-(x_g**2) / (2 * beta_T)) / np.sqrt(2 * np.pi * beta_T)
    M_dummy_g = np.ones((Nt_test + 1, N_test + 1))
    solver_st = HJBSemiLagrangianSolver(
        problem_g,
        interpolation_method="cubic",
        diffusion_method="stochastic",
        check_cfl=False,
    )
    solver_adi = HJBSemiLagrangianSolver(
        problem_g,
        interpolation_method="cubic",
        diffusion_method="adi",
        check_cfl=False,
    )
    U_st = solver_st.solve_hjb_system(
        M_density=M_dummy_g,
        U_terminal=U_T_g,
        U_coupling_prev=np.zeros((Nt_test + 1, N_test + 1)),
    )
    U_adi = solver_adi.solve_hjb_system(
        M_density=M_dummy_g,
        U_terminal=U_T_g,
        U_coupling_prev=np.zeros((Nt_test + 1, N_test + 1)),
    )
    discrepancy = np.max(np.abs(U_st[0] - U_adi[0]))
    assert discrepancy < 5e-3, f"Stochastic and ADI diverge on H=0 Gaussian: max diff = {discrepancy:.3e}"
    print(f"   9e: stochastic vs ADI on H=0 Gaussian, max diff = {discrepancy:.3e}: OK")

    print("   Carlini-Silva stochastic SL (Issue #1026): OK")

    print("\nAll smoke tests passed!")
