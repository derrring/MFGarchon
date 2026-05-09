"""
Finite Difference Method (FDM) for HJB Equation - All Dimensions.

Supports:
    - 1D: Optimized Newton solver from base_hjb
    - 2D/3D/nD: Uses centralized nonlinear solvers

References:
    - Evans (2010): Partial Differential Equations, Ch. 10
    - Achdou & Capuzzo-Dolcetta (2010): Mean field games: numerical methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mfgarchon.geometry.base import CartesianGrid  # nD FDM needs structured grid ABC
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical import FixedPointSolver, NewtonSolver
from mfgarchon.utils.pde_coefficients import CoefficientField

from . import base_hjb
from .base_hjb import BaseHJBSolver

logger = get_logger(__name__)

# Type alias for HJB advection schemes (gradient form only - HJB is not a conservation law)
HJBAdvectionScheme = Literal["gradient_centered", "gradient_upwind"]

# Type alias for Newton failure behavior (Issue #669)
NewtonFailurePolicy = Literal["raise", "warn_and_fallback"]


class ConvergenceError(RuntimeError):
    """Raised when a nonlinear solver fails to converge.

    Attributes:
        solver_type: Type of solver that failed ('newton', 'fixed_point').
        iterations: Number of iterations performed before failure.
        residual: Final residual norm at failure.
    """

    def __init__(self, solver_type: str, iterations: int, residual: float, message: str = ""):
        self.solver_type = solver_type
        self.iterations = iterations
        self.residual = residual
        super().__init__(
            message
            or f"{solver_type} solver failed to converge after {iterations} iterations (residual: {residual:.2e})"
        )


def is_diagonal_tensor(Sigma: NDArray, rtol: float = 1e-10) -> bool:
    """
    Check if tensor is diagonal (off-diagonal elements near zero).

    Args:
        Sigma: Tensor array, either (d, d) or (*shape, d, d)
        rtol: Relative tolerance for off-diagonal elements

    Returns:
        True if diagonal, False otherwise
    """
    # Handle both single tensor and spatially-varying tensors
    if Sigma.ndim == 2:
        # Single (d, d) tensor
        d = Sigma.shape[0]
        off_diag_sum = np.sum(np.abs(Sigma)) - np.sum(np.abs(np.diag(Sigma)))
        diag_sum = np.sum(np.abs(np.diag(Sigma)))
        return off_diag_sum < rtol * diag_sum if diag_sum > 0 else off_diag_sum < rtol
    else:
        # Spatially-varying (*shape, d, d)
        d = Sigma.shape[-1]
        diag_mask = np.eye(d, dtype=bool)
        off_diag_elements = Sigma[..., ~diag_mask]
        diag_elements = Sigma[..., diag_mask]
        off_diag_norm = np.linalg.norm(off_diag_elements)
        diag_norm = np.linalg.norm(diag_elements)
        return off_diag_norm < rtol * diag_norm if diag_norm > 0 else off_diag_norm < rtol


if TYPE_CHECKING:
    from collections.abc import Callable

    import scipy.sparse as sparse
    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary import ConstraintProtocol


class HJBFDMSolver(BaseHJBSolver):
    """
    Finite Difference Method solver for HJB equation (all dimensions).

    Automatically handles 1D, 2D, 3D, and higher-dimensional problems:
        - 1D: Uses optimized Newton solver from base_hjb
        - nD: Uses centralized FixedPointSolver or NewtonSolver

    Recommended: d ≤ 3 due to O(N^d) complexity

    Required Geometry Traits (Issue #596 Phase 2.1):
        - SupportsGradient: Provides ∇U operator for Hamiltonian evaluation H(x, ∇U, m)

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsGradient trait

    Example:
        >>> from mfgarchon import MFGProblem
        >>> from mfgarchon.geometry import TensorProductGrid
        >>>
        >>> grid = TensorProductGrid(bounds=[(0,1), (0,1)], Nx=[50, 50])
        >>> problem = MFGProblem(geometry=grid, ...)
        >>> solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
        >>> U_solution = solver.solve_hjb_system(M_density, U_final)

    Note:
        Solver uses trait-based operators for gradient computation, eliminating
        manual stencil code and enabling geometry-agnostic algorithm design.
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.FDM

    @deprecated_parameter(
        param_name="NiterNewton",
        since="v0.16.0",
        replacement="max_newton_iterations",
    )
    @deprecated_parameter(
        param_name="l2errBoundNewton",
        since="v0.16.0",
        replacement="newton_tolerance",
    )
    @deprecated_parameter(
        param_name="damping_factor",
        since="v0.19.2",
        replacement="relaxation",
    )
    def __init__(
        self,
        problem: MFGProblem,
        solver_type: Literal["fixed_point", "newton"] = "newton",
        advection_scheme: HJBAdvectionScheme = "gradient_upwind",
        relaxation: float = 1.0,
        max_newton_iterations: int | None = None,
        newton_tolerance: float | None = None,
        constraint: ConstraintProtocol | None = None,
        on_newton_failure: NewtonFailurePolicy = "raise",
        # Deprecated parameters (decorator handles warnings)
        NiterNewton: int | None = None,
        l2errBoundNewton: float | None = None,
        damping_factor: float | None = None,
        backend: str | None = None,
    ):
        """
        Initialize FDM solver.

        Args:
            problem: MFG problem (1D or MFGProblem with spatial_bounds for nD)
            solver_type: 'fixed_point' or 'newton' (nD only, 1D always uses Newton)
            advection_scheme: Discretization scheme for advection term:
                - 'gradient_upwind': Godunov upwind (default, monotone, first-order)
                - 'gradient_centered': Central differences (second-order, may oscillate)
                For MFG coupling, use 'gradient_upwind' with FP 'divergence_upwind'.
            relaxation: Under-relaxation factor ω ∈ (0,1] for fixed-point iteration.
                Legacy `damping_factor` kwarg still accepted with DeprecationWarning.
            max_newton_iterations: Max iterations per timestep
            newton_tolerance: Convergence tolerance
            constraint: Variational inequality constraint (Issue #591):
                - ObstacleConstraint: u ≥ ψ or u ≤ ψ (capacity limits, running cost floor)
                - BilateralConstraint: ψ_lower ≤ u ≤ ψ_upper (bounded controls)
                - None: No constraints (default)
                Applied after each timestep solve via projection P_K(u).
            on_newton_failure: Behavior when Newton solver fails (Issue #669).
                Only meaningful when solver_type="newton" and dimension > 1.
                - 'raise': Raise ConvergenceError (default, fail-fast).
                - 'warn_and_fallback': Emit warning and retry with Value Iteration.
            backend: 'numpy', 'torch', or None
        """
        import warnings

        super().__init__(problem)

        # Initialize backend
        from mfgarchon.backends import create_backend

        self.backend = create_backend(backend or "numpy")

        # Validate and store advection scheme
        valid_schemes = {"gradient_centered", "gradient_upwind"}
        if advection_scheme not in valid_schemes:
            raise ValueError(f"Invalid advection_scheme: '{advection_scheme}'. Valid options: {sorted(valid_schemes)}")
        self.advection_scheme = advection_scheme
        self.use_upwind = advection_scheme == "gradient_upwind"

        # Redirect deprecated parameters (decorator handles warnings)
        if NiterNewton is not None:
            max_newton_iterations = max_newton_iterations or NiterNewton
        if l2errBoundNewton is not None:
            newton_tolerance = newton_tolerance or l2errBoundNewton
        if damping_factor is not None:
            relaxation = damping_factor

        # Set defaults (use None check to avoid treating 0 as falsy)
        self.max_newton_iterations = (
            max_newton_iterations if max_newton_iterations is not None else base_hjb.DEFAULT_NEWTON_MAX_ITERATIONS
        )
        self.newton_tolerance = newton_tolerance if newton_tolerance is not None else base_hjb.DEFAULT_NEWTON_TOLERANCE
        self.solver_type = solver_type
        self.relaxation = relaxation
        self.constraint = constraint  # Variational inequality constraint (Issue #591)
        self.on_newton_failure: NewtonFailurePolicy = on_newton_failure  # Issue #669

        # Validate
        if self.max_newton_iterations < 1:
            raise ValueError(f"max_newton_iterations must be >= 1, got {self.max_newton_iterations}")
        if self.newton_tolerance <= 0:
            raise ValueError(f"newton_tolerance must be > 0, got {self.newton_tolerance}")
        if not 0 < relaxation <= 1.0:
            raise ValueError(f"relaxation must be in (0,1], got {relaxation}")

        # Backward compatibility: Store Newton config
        self._newton_config = {
            "max_iterations": self.max_newton_iterations,
            "tolerance": self.newton_tolerance,
        }

        # Detect dimension (inherited from BaseNumericalSolver, Issue #633)
        self.dimension = self._detect_dimension()
        # Backward compatibility: 1D uses "FDM", nD uses "FDM-{d}D-{solver_type}"
        if self.dimension == 1:
            self.hjb_method_name = "FDM"
        else:
            self.hjb_method_name = f"FDM-{self.dimension}D-{solver_type}"

        # Validate geometry capabilities (Issue #596 Phase 2.1)
        # HJB solver requires gradient operator for Hamiltonian evaluation
        from mfgarchon.geometry.protocols import SupportsGradient

        if not isinstance(problem.geometry, SupportsGradient):
            raise TypeError(
                f"HJB FDM solver requires geometry with SupportsGradient trait for ∇U computation. "
                f"{type(problem.geometry).__name__} does not implement this trait. "
                f"Compatible geometries: TensorProductGrid, ImplicitDomain."
            )

        # For nD, extract grid info and create nonlinear solver
        if self.dimension > 1:
            # nD FDM requires a structured grid with get_grid_shape/get_grid_spacing.
            # CartesianGrid ABC guarantees these methods (Issue #732 Tier 1b).
            if not isinstance(problem.geometry, CartesianGrid):
                raise ValueError(
                    f"nD FDM requires CartesianGrid geometry (structured grid). Got {type(problem.geometry).__name__}."
                )

            self.grid = problem.geometry  # Geometry IS the grid
            self.shape = tuple(self.grid.get_grid_shape())
            self.spacing = self.grid.get_grid_spacing()
            self.N_total = int(np.prod(self.shape))
            self.dt = problem.dt

            if self.dimension > 3:
                warnings.warn(
                    f"FDM solver in {self.dimension}D requires {self.N_total:,} grid points. "
                    f"Consider GFDM or sparse methods for d>3.",
                    UserWarning,
                    stacklevel=2,
                )

            # Create nonlinear solver
            if solver_type == "fixed_point":
                self.nonlinear_solver = FixedPointSolver(
                    relaxation=relaxation,
                    max_iterations=self.max_newton_iterations,
                    tolerance=self.newton_tolerance,
                )
            else:  # newton
                self.nonlinear_solver = NewtonSolver(
                    max_iterations=self.max_newton_iterations,
                    tolerance=self.newton_tolerance,
                    sparse=True,
                    jacobian=None,  # Use automatic finite differences
                )

            # Create BC applicator using FDMApplicator (Issue #516)
            from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator

            self.bc_applicator = FDMApplicator(dimension=self.dimension)

            # Get gradient operators from geometry (Issue #596 Phase 2.1)
            # Operators automatically inherit BC from geometry
            scheme = "upwind" if self.use_upwind else "central"
            self._gradient_operators = problem.geometry.get_gradient_operator(scheme=scheme)

        # Initialize warning flags (Issue #545 - NO hasattr pattern)
        self._bc_warning_emitted: bool = False

        # Cached Laplacian operator for nD diffusion (Issue #787)
        self._laplacian_op: object | None = None

    # _detect_dimension() inherited from BaseNumericalSolver (Issue #633)

    @property
    def damping_factor(self) -> float:
        """Deprecated alias for `relaxation` (v0.19.2+). Removal in v0.25.0."""
        return self.relaxation

    def _get_laplacian_op(self):
        """Get (or create) cached Laplacian operator for diffusion term."""
        if self._laplacian_op is None:
            bc = self.get_boundary_conditions()
            self._laplacian_op = self.problem.geometry.get_laplacian_operator(order=2, bc=bc)
        return self._laplacian_op

    def _log_cfl_diagnostic(self, volatility_field: float | None = None) -> None:
        """Log CFL diagnostic for accuracy/convergence guidance (Issue #882, #1052).

        Issue #1052: log once at INFO per solver instance, subsequent calls at
        DEBUG. The CFL parameters don't change between Picard iterations, so
        emitting INFO every iter spammed user output and caused researchers to
        blanket-suppress warnings — masking unrelated DeprecationWarnings.
        """
        try:
            dt = self.problem.dt
            dx = self.problem.geometry.get_grid_spacing()[0]
            sigma = volatility_field if isinstance(volatility_field, (int, float)) else self.problem.sigma
            cfl_diffusive = sigma**2 * dt / dx**2
            if cfl_diffusive > 0.5:
                log_fn = logger.debug if getattr(self, "_cfl_logged", False) else logger.info
                log_fn(
                    "CFL diagnostic (HJB FDM): diffusive=%.2f (sigma=%.3g, dt=%.3g, dx=%.3g). "
                    "Implicit scheme is stable but accuracy may degrade for CFL >> 1.",
                    cfl_diffusive,
                    sigma,
                    dt,
                    dx,
                )
                self._cfl_logged = True
        except (AttributeError, IndexError, TypeError):
            pass  # Not enough info to compute CFL — skip silently

    @deprecated_parameter(
        param_name="M_density_evolution",
        since="v0.17.0",
        replacement="M_density",
    )
    @deprecated_parameter(
        param_name="M_density_evolution_from_FP",
        since="v0.17.0",
        replacement="M_density",
    )
    @deprecated_parameter(
        param_name="U_final_condition",
        since="v0.17.0",
        replacement="U_terminal",
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
    @deprecated_parameter(
        param_name="bc_values",
        since="v0.17.0",
        replacement="BCValueProvider in BoundaryConditions",
    )
    @deprecated_parameter(
        param_name="tensor_volatility_field",
        since="v0.18.7",
        replacement="volatility_field (pass (d,d) array or callable returning (d,d))",
    )
    def solve_hjb_system(
        self,
        M_density: NDArray | None = None,
        U_terminal: NDArray | None = None,
        U_coupling_prev: NDArray | None = None,
        volatility_field: float | NDArray | None = None,
        tensor_volatility_field: NDArray | None = None,
        bc_values: dict[str, float] | None = None,
        progress_callback: Callable[[int], None] | None = None,  # Issue #640
        show_progress: bool | None = None,  # Issue #934
        # MMS verification support
        source_term: Callable | None = None,
        # Deprecated parameter names for backward compatibility (decorators handle warnings)
        M_density_evolution_from_FP: NDArray | None = None,
        U_final_condition_at_T: NDArray | None = None,
        U_from_prev_picard: NDArray | None = None,
        M_density_evolution: NDArray | None = None,
        U_final_condition: NDArray | None = None,
    ) -> NDArray:
        """
        Solve HJB system backward in time.

        Automatically routes to 1D or nD solver based on dimension.

        Args:
            M_density: Density field from FP solver
            U_terminal: Terminal condition u(T,x)
            U_coupling_prev: Previous coupling iteration estimate
            volatility_field: Diffusion coefficient (None uses problem.sigma)
            tensor_volatility_field: Tensor diffusion (Phase 3.0, not yet fully implemented)
            bc_values: DEPRECATED. No longer used (kept for backward compatibility).
                Adjoint-consistent BC is handled via BCValueProvider in BoundaryConditions.

        Note:
            For adjoint-consistent BC, use AdjointConsistentProvider in BCSegment.value
            when constructing BoundaryConditions. The FixedPointIterator resolves
            providers each iteration via problem.using_resolved_bc(state).
            See mfgarchon/geometry/boundary/providers.py for details.
        """
        # Handle deprecated parameter names (decorators handle warnings, keep redirect logic)
        if M_density_evolution is not None:
            if M_density is not None or M_density_evolution_from_FP is not None:
                raise ValueError("Cannot specify M_density_evolution with M_density or M_density_evolution_from_FP")
            M_density = M_density_evolution

        if M_density_evolution_from_FP is not None:
            if M_density is not None:
                raise ValueError("Cannot specify both 'M_density' and deprecated 'M_density_evolution_from_FP'")
            M_density = M_density_evolution_from_FP

        if U_final_condition is not None:
            if U_terminal is not None or U_final_condition_at_T is not None:
                raise ValueError("Cannot specify U_final_condition with U_terminal or U_final_condition_at_T")
            U_terminal = U_final_condition

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
        # Issue #889: merge tensor_volatility_field into volatility_field
        # Deprecation warning issued by @deprecated_parameter decorator
        if tensor_volatility_field is not None:
            if volatility_field is not None:
                raise ValueError(
                    "Cannot specify both volatility_field and tensor_volatility_field. "
                    "Use volatility_field (tensor_volatility_field is deprecated)."
                )
            volatility_field = tensor_volatility_field

        # CFL diagnostic (Issue #882): implicit scheme is unconditionally stable,
        # but large CFL numbers indicate potential accuracy/convergence issues
        self._log_cfl_diagnostic(volatility_field)

        if self.dimension == 1:
            # Extract BC from geometry for Issue #542 fix
            # Issue #527: Use centralized get_boundary_conditions() from BaseMFGSolver
            bc = self.get_boundary_conditions()

            # Extract domain bounds from geometry
            domain_bounds = None
            try:
                bounds = self.problem.geometry.get_bounds()
                # Convert to (1, 2) array for 1D
                domain_bounds = np.array([[bounds[0][0], bounds[1][0]]])
            except AttributeError:
                pass

            # Debug: Log BC being passed (Issue #542 investigation)
            # Changed from logger.info to logger.debug to reduce verbosity (Issue #623)
            from contextlib import suppress

            logger.debug(f"[DEBUG Issue #542] BC passed to solve_hjb_system_backward: {bc}")
            # Log segment count if BC has segments attribute (Issue #545: use contextlib.suppress)
            if bc is not None:
                with suppress(AttributeError):
                    logger.debug(f"[DEBUG Issue #542] BC has {len(bc.segments)} segments")

            # Use optimized 1D solver with BC-aware computation (Issue #542 fix)
            U_solution = base_hjb.solve_hjb_system_backward(
                M_density_from_prev_picard=M_density,
                U_final_condition_at_T=U_terminal,
                U_from_prev_picard=U_coupling_prev,
                problem=self.problem,
                max_newton_iterations=self.max_newton_iterations,
                newton_tolerance=self.newton_tolerance,
                backend=self.backend,
                volatility_field=volatility_field,
                use_upwind=self.use_upwind,
                bc=bc,  # Uses Robin BC from geometry; providers resolved by iterator (Issue #625)
                domain_bounds=domain_bounds,
                source_term=source_term,
            )

            # Apply variational inequality constraint via projection (Issue #591)
            # For 1D path, apply constraint to all timesteps after solving
            if self.constraint is not None:
                for n in range(U_solution.shape[0]):
                    U_solution[n] = self.constraint.project(U_solution[n])

            return U_solution
        else:
            # Use nD solver with centralized nonlinear solver
            return self._solve_hjb_nd(
                M_density,
                U_terminal,
                U_coupling_prev,
                volatility_field,
                progress_callback=progress_callback,
                source_term=source_term,
                show_progress=show_progress,
            )

    def _solve_hjb_nd(
        self,
        M_density: NDArray,
        U_final: NDArray,
        U_prev: NDArray,
        volatility_field: float | NDArray | None = None,
        progress_callback: Callable[[int], None] | None = None,  # Issue #640
        source_term: Callable | None = None,  # MMS verification
        show_progress: bool | None = None,  # Issue #934
    ) -> NDArray:
        """Solve nD HJB using centralized nonlinear solvers with variable diffusion.

        volatility_field accepts scalar, array, (d,d) tensor, or callable.
        Shape-based auto-detection dispatches to scalar or tensor path.
        """
        # Validate shapes
        # n_time_points = problem.Nt + 1 (number of time knots including t=0 and t=T)
        # problem.Nt = number of time intervals
        n_time_points = self.problem.Nt + 1
        expected_shape = (n_time_points, *self.shape)
        if M_density.shape != expected_shape:
            raise ValueError(f"M_density shape {M_density.shape} != {expected_shape}")
        if U_final.shape != self.shape:
            raise ValueError(f"U_final shape {U_final.shape} != {self.shape}")

        # Allocate solution array with shape (n_time_points, *spatial)
        U_solution = np.zeros(expected_shape, dtype=np.float64)
        # Set terminal condition at t=T (last time index)
        U_solution[n_time_points - 1] = U_final.copy()

        if n_time_points <= 1:
            return U_solution

        # Issue #640: Use progress_callback if provided, else show own progress bar
        use_external_progress = progress_callback is not None

        from mfgarchon.utils.progress import create_progress_bar, should_show_progress

        timestep_iter = create_progress_bar(
            range(n_time_points - 2, -1, -1),
            verbose=should_show_progress(show_progress) and not use_external_progress,
            desc=f"HJB {self.dimension}D-FDM ({self.solver_type})",
        )

        # Backward time loop
        for n in timestep_iter:
            U_next = U_solution[n + 1]
            M_next = M_density[n + 1]
            U_guess = U_prev[n]

            # Issue #889: unified volatility_field with shape-based dispatch
            d = self.dimension
            Sigma_at_n = None
            sigma_at_n = None

            if volatility_field is not None and isinstance(volatility_field, np.ndarray):
                # Auto-detect tensor vs scalar from shape
                if volatility_field.ndim >= 2 and volatility_field.shape[-2:] == (d, d):
                    Sigma_at_n = volatility_field
                elif volatility_field.ndim == 1 and len(volatility_field) == d:
                    Sigma_at_n = np.diag(volatility_field)
            elif callable(volatility_field):
                # Callable: evaluate and check shape
                t = n * self.problem.dt
                test_x = np.array([self.grid.coordinates[dd][0] for dd in range(d)])
                test_val = volatility_field(t, test_x, float(M_next.flat[0]))
                test_arr = np.asarray(test_val)
                if test_arr.ndim >= 2 and test_arr.shape[-2:] == (d, d):
                    # Tensor callable — evaluate at all points
                    Sigma_at_n = np.zeros((*self.shape, d, d))
                    for idx in np.ndindex(self.shape):
                        x_coords = np.array([self.grid.coordinates[dd][idx[dd]] for dd in range(d)])
                        Sigma_at_n[idx] = volatility_field(t, x_coords, float(M_next[idx]))

            if Sigma_at_n is None:
                # Scalar path (CoefficientField handles float, array, callable)
                diffusion = CoefficientField(volatility_field, self.problem.sigma, "volatility_field", dimension=d)
                sigma_at_n = diffusion.evaluate_at(
                    timestep_idx=n, grid=self.grid.coordinates, density=M_next, dt=self.problem.dt
                )

            # Compute current time for time-dependent BCs
            t_current = n * self.dt

            # Evaluate source term at current timestep (if provided)
            if source_term is not None:
                x_grid = self.problem.geometry.get_spatial_grid()  # (N, d)
                source_at_n = source_term(t_current, x_grid).reshape(self.shape)
            else:
                source_at_n = None

            # Solve nonlinear system using centralized solver
            U_solution[n] = self._solve_single_timestep(
                U_next,
                M_next,
                U_guess,
                sigma_at_n,
                Sigma_at_n,
                time=t_current,
                constraint=self.constraint,
                source_term=source_at_n,
            )

            # Issue #640: Update external progress if callback provided
            if progress_callback is not None:
                progress_callback(1)

        return U_solution

    def _solve_single_timestep(
        self,
        U_next: NDArray,
        M_next: NDArray,
        U_guess: NDArray,
        sigma_at_n: float | NDArray | None = None,
        Sigma_at_n: NDArray | None = None,
        time: float = 0.0,
        constraint: ConstraintProtocol | None = None,
        source_term: NDArray | None = None,
    ) -> NDArray:
        """
        Solve single HJB timestep using centralized nonlinear solver.

        HJB equation: -∂u/∂t + H(∇u, m) - (σ²/2) Δu = 0

        For fixed-point: Solves u = G(u) where G(u) = u_next - dt·(H - (σ²/2)Δu)
        For Newton: Solves F(u) = 0 where F(u) = (u - u_next)/dt + H - (σ²/2)Δu

        Args:
            U_next: Value function at next timestep
            M_next: Density at next timestep
            U_guess: Initial guess for current timestep
            sigma_at_n: Scalar volatility coefficient (sigma) at current timestep (None, float, or array)
            Sigma_at_n: Tensor diffusion coefficient at current timestep (None or tensor array)
            time: Current time for time-dependent BC values
            constraint: Variational inequality constraint (Issue #591)
                - ObstacleConstraint: u ≥ ψ or u ≤ ψ
                - BilateralConstraint: ψ_lower ≤ u ≤ ψ_upper
                - None: No constraints
        """
        if self.solver_type == "fixed_point":
            # Define fixed-point map G: u → u
            # HJB uses H which includes viscosity term (σ²/2)|∇u|²
            # Fixed-point iteration: u_n = u_{n+1} - dt·(H(∇u_n, m) - S)
            def G(U: NDArray) -> NDArray:
                gradients = self._compute_gradients_nd(U, time=time)
                H_values = self._evaluate_hamiltonian_nd(U, M_next, gradients, sigma_at_n, Sigma_at_n, time=time)
                rhs = H_values
                if source_term is not None:
                    rhs = rhs - source_term
                return U_next - self.dt * rhs

            U_solution, info = self.nonlinear_solver.solve(G, U_guess)

        else:  # newton
            # Define residual F: u → residual
            # F(u) = (u - u_next)/dt + H(∇u, m) - S = 0
            def F(U: NDArray) -> NDArray:
                gradients = self._compute_gradients_nd(U, time=time)
                H_values = self._evaluate_hamiltonian_nd(U, M_next, gradients, sigma_at_n, Sigma_at_n, time=time)
                residual = (U - U_next) / self.dt + H_values
                if source_term is not None:
                    residual = residual - source_term
                return residual

            # Issue #669: Newton-to-Value-Iteration adaptive fallback
            newton_failed = False
            newton_error: Exception | None = None
            try:
                U_solution, info = self.nonlinear_solver.solve(F, U_guess)
                newton_failed = not info.converged
            except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
                newton_failed = True
                newton_error = e
                from mfgarchon.utils.numerical import SolverInfo

                info = SolverInfo(converged=False, iterations=0, residual=float("inf"), residual_history=[])

            if newton_failed:
                if self.on_newton_failure == "warn_and_fallback":
                    # Define fixed-point map G for fallback (same computation as FP path)
                    def G_fallback(U: NDArray) -> NDArray:
                        gradients = self._compute_gradients_nd(U, time=time)
                        H_values = self._evaluate_hamiltonian_nd(
                            U, M_next, gradients, sigma_at_n, Sigma_at_n, time=time
                        )
                        rhs = H_values
                        if source_term is not None:
                            rhs = rhs - source_term
                        return U_next - self.dt * rhs

                    fallback_solver = FixedPointSolver(
                        relaxation=self.relaxation,
                        max_iterations=self.max_newton_iterations * 10,
                        tolerance=self.newton_tolerance,
                    )

                    error_detail = f": {newton_error}" if newton_error else f" (residual: {info.residual:.2e})"
                    import warnings

                    warnings.warn(
                        f"Newton solver failed{error_detail}. "
                        f"Falling back to Value Iteration as explicitly requested "
                        f"(on_newton_failure='warn_and_fallback').",
                        UserWarning,
                        stacklevel=2,
                    )

                    U_solution, info = fallback_solver.solve(G_fallback, U_guess)
                else:  # "raise"
                    if newton_error:
                        raise ConvergenceError(
                            "newton", info.iterations, info.residual, f"Newton solver failed: {newton_error}"
                        ) from newton_error
                    else:
                        raise ConvergenceError("newton", info.iterations, info.residual)

        # Warn if not converged (applies to both FP and post-fallback)
        if not info.converged:
            import warnings

            warnings.warn(
                f"{self.solver_type} did not converge (residual: {info.residual:.2e})",
                UserWarning,
                stacklevel=2,
            )

        # Enforce BC on solution (Issue #542 - nD extension, Issue #527 - centralized BC access)
        # BC-aware gradients use ghost cells for derivatives, but boundary values must be explicitly set
        bc = self.get_boundary_conditions()
        if bc is not None:
            U_solution = self.bc_applicator.enforce_values(
                field=U_solution,
                boundary_conditions=bc,
                spacing=self.spacing,
                time=time,
            )

        # Apply variational inequality constraint via projection (Issue #591)
        # Order: 1) Solve PDE → 2) Enforce BC → 3) Project onto constraint set K
        # This ensures the solution satisfies both BC and constraints
        if constraint is not None:
            U_solution = constraint.project(U_solution)

        return U_solution

    def _compute_gradients_nd(self, U: NDArray, time: float = 0.0) -> dict[int, NDArray]:
        """Compute gradients using trait-based geometry operators (Issue #596 Phase 2.1).

        Uses geometry.get_gradient_operator() which automatically handles:
        - Boundary conditions via ghost cells
        - Scheme selection (upwind vs central)
        - Multi-dimensional stencils

        This replaces ~130 lines of manual gradient computation with a clean trait-based interface.

        Args:
            U: Value function at current timestep
            time: Current time for time-dependent BC values

        Returns:
            Dict mapping dimension index to gradient array for that dimension.
            Key 0 = ∂u/∂x₀, Key 1 = ∂u/∂x₁, etc.
            Also includes special key -1 for the function value U itself.

        Note:
            For time-dependent BCs, operators are created per-timestep with correct time.
            For time-independent BCs, cached operators from __init__() are reused.
        """
        # Store gradients by dimension index for efficient access
        gradients: dict[int, NDArray] = {-1: U}  # -1 = function value

        # Check if we need time-dependent operators
        # For now, always create operators with current time to ensure proper BC handling
        # Potential optimization: cache operators for time-independent BCs
        scheme = "upwind" if self.use_upwind else "central"
        grad_ops = self.problem.geometry.get_gradient_operator(scheme=scheme, time=time)

        # Apply gradient operators in each direction
        for d in range(self.dimension):
            # Operator call syntax: grad_op(U) applies stencil with BC handling
            gradients[d] = grad_ops[d](U)

        return gradients

    def _evaluate_hamiltonian_nd(
        self,
        U: NDArray,
        M: NDArray,
        gradients: dict[int, NDArray],
        sigma_at_n: float | NDArray | None = None,
        Sigma_at_n: NDArray | None = None,
        time: float = 0.0,
    ) -> NDArray:
        """Evaluate Hamiltonian at all grid points with variable diffusion support.

        Uses vectorized batch HamiltonianBase evaluation (Issue #784).
        For scalar diffusion, includes the diffusion term -(sigma^2/2)*Laplacian(U)
        following the 1D pattern in base_hjb.py:774-803 (Issue #787).

        Args:
            U: Value function at current timestep
            M: Density at current timestep
            gradients: Dictionary mapping dimension index to gradient arrays.
                       Key d = dU/dxd.
            sigma_at_n: Scalar volatility coefficient (sigma)
            Sigma_at_n: Tensor diffusion coefficient (diagonal only)
            time: Current time for time-dependent Hamiltonians (default 0.0)
        """
        return self._evaluate_hamiltonian_vectorized(U, M, gradients, sigma_at_n, Sigma_at_n, time=time)

    def _evaluate_hamiltonian_vectorized(
        self,
        U: NDArray,
        M: NDArray,
        gradients: dict[int, NDArray],
        sigma_at_n=None,
        Sigma_at_n=None,
        time: float = 0.0,
    ) -> NDArray:
        """
        Vectorized Hamiltonian evaluation across all grid points.

        Uses batch HamiltonianBase.__call__ for the convective part (Issue #784),
        then adds -(sigma^2/2)*Laplacian(U) for the diffusion term (Issue #787).

        For tensor diffusion, uses the existing diagonal approximation path.

        Args:
            U: Value function at current timestep
            M: Density at current timestep
            gradients: Dictionary mapping dimension index to gradient arrays.
                       Key d = dU/dxd.
            sigma_at_n: Scalar volatility coefficient (sigma)
            Sigma_at_n: Tensor diffusion coefficient (diagonal only)
            time: Current time for time-dependent Hamiltonians

        Returns:
            H_values: Hamiltonian evaluated at all grid points, shape self.shape
        """
        # Build x_grid: (N_total_points, dimension)
        coord_grids = np.meshgrid(*self.grid.coordinates, indexing="ij")
        x_grid = np.stack([g.ravel() for g in coord_grids], axis=-1)  # (N_total, d)

        # Build p_grid: (N_total_points, dimension)
        p_components = []
        for d in range(self.dimension):
            if d in gradients:
                p_components.append(gradients[d].ravel())
            else:
                p_components.append(np.zeros(x_grid.shape[0]))
        p_grid = np.stack(p_components, axis=-1)  # (N_total, d)

        # Flatten density
        m_grid = M.ravel()  # (N_total,)

        if Sigma_at_n is not None:
            # Tensor diffusion mode — batch H_class + anisotropic Laplacian (Issue #784)
            # Extract diagonal weights from diffusion tensor
            if Sigma_at_n.ndim == 2:
                if not is_diagonal_tensor(Sigma_at_n):
                    import warnings

                    warnings.warn(
                        "Non-diagonal tensor diffusion detected. "
                        "Using diagonal approximation (ignoring off-diagonal terms).",
                        UserWarning,
                        stacklevel=3,
                    )
                sigma_diag = np.diag(Sigma_at_n)  # (d,)
            else:
                if not is_diagonal_tensor(Sigma_at_n):
                    import warnings

                    warnings.warn(
                        "Non-diagonal tensor diffusion detected. "
                        "Using diagonal approximation (ignoring off-diagonal terms).",
                        UserWarning,
                        stacklevel=3,
                    )
                # Spatially-varying: extract diagonal, average to constant weights
                sigma_diag = np.diagonal(Sigma_at_n, axis1=-2, axis2=-1)
                if sigma_diag.ndim > 1:
                    sigma_diag = sigma_diag.reshape(-1, self.dimension).mean(axis=0)

            # Convective Hamiltonian via H_class (same pattern as scalar path)
            H_class = self.problem.hamiltonian_class
            H_convective = np.asarray(H_class(x_grid, m_grid, p_grid, t=time), dtype=float)

            # Anisotropic diffusion: -(1/2) * sum_d sigma_d^2 * d^2u/dx_d^2
            from mfgarchon.operators.stencils.finite_difference import weighted_laplacian_with_bc

            bc = self.get_boundary_conditions()
            spacings = list(self.problem.geometry.get_grid_spacing())
            aniso_lap = weighted_laplacian_with_bc(
                U.reshape(self.shape), spacings, axis_weights=sigma_diag, bc=bc, time=time
            ).ravel()

            H_values_flat = H_convective - 0.5 * aniso_lap

        else:
            # Scalar diffusion mode — batch HamiltonianBase (Issue #784)
            # hamiltonian_class is guaranteed by MFGComponents (Issue #670)
            H_class = self.problem.hamiltonian_class
            H_convective = np.asarray(H_class(x_grid, m_grid, p_grid, t=time), dtype=float)

            # Diffusion term: -(sigma^2/2) * Laplacian(U) (Issue #787)
            # Follows 1D pattern in base_hjb.py:774-803
            sigma = sigma_at_n if sigma_at_n is not None else self.problem.sigma
            lap_u = self._get_laplacian_op()(U).ravel()
            H_values_flat = H_convective - 0.5 * sigma**2 * lap_u

        return H_values_flat.reshape(self.shape)

    # =========================================================================
    # Strict Adjoint Mode (Issue #622)
    # =========================================================================

    def build_advection_matrix(
        self,
        U: NDArray,
        coupling_coefficient: float | None = None,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """
        Build upwind advection matrix from value function gradient.

        This matrix encodes the drift velocity v = -coupling_coefficient * ∇U
        using the same upwind discretization that would be used internally.

        For strict adjoint mode (Issue #622), this matrix A_HJB is passed to the
        FP solver which uses A_HJB^T, guaranteeing exact adjoint consistency:
            L_FP = L_HJB^T

        Mathematical Background:
            For MFG with Hamiltonian H = (coupling/2)|∇u|², the optimal control is:
                α* = -coupling_coefficient * ∇u

            The HJB advection term is: α* · ∇u = -coupling * ∇u · ∇u
            This is discretized with upwind differences for stability.

        Args:
            U: Value function at current timestep, shape (*spatial_shape)
            coupling_coefficient: Drift coupling coefficient.
                If None, uses problem.coupling_coefficient
            time: Current time for time-dependent BCs

        Returns:
            Sparse CSR matrix A_advection of shape (N_total, N_total) where
            N_total = prod(spatial_shape). The matrix encodes the advection
            operator using velocity-based linear upwind discretization.

        Example:
            >>> # Build matrix for strict adjoint coupling
            >>> A_hjb = hjb_solver.build_advection_matrix(U_current)
            >>> # FP solver uses transpose
            >>> A_fp = A_hjb.T  # Exact adjoint!

        Note:
            The matrix uses velocity-based linear upwind (same as FP's
            divergence_upwind mode). This ensures the transpose relationship
            holds for mass conservation.

        See Also:
            - Issue #622: Strict Achdou adjoint mode implementation
            - solve_hjb_step_with_matrix(): Uses externally provided matrix
            - FPFDMSolver.solve_fp_step_adjoint_mode(): Uses A^T from this method
        """
        if self.dimension == 1:
            return self._build_advection_matrix_1d(U, coupling_coefficient, time)
        else:
            return self._build_advection_matrix_nd(U, coupling_coefficient, time)

    def _build_advection_matrix_1d(
        self,
        U: NDArray,
        coupling_coefficient: float | None = None,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """Build 1D upwind advection matrix.

        Uses velocity-based linear upwind discretization matching FP divergence form.

        For velocity v = -coupling * ∂U/∂x at point i:
        - If v_i > 0 (flow to right): use backward difference (info from left)
          A[i,i] += v_i/dx, A[i,i-1] -= v_i/dx
        - If v_i < 0 (flow to left): use forward difference (info from right)
          A[i,i] -= v_i/dx, A[i,i+1] += v_i/dx

        This matches the "divergence_upwind" scheme in FP solver.
        """
        import scipy.sparse as sparse

        # Get grid info
        Nx = self.problem.geometry.get_grid_shape()[0]
        dx = self.problem.geometry.get_grid_spacing()[0]

        # Get coupling coefficient
        if coupling_coefficient is None:
            coupling_coefficient = getattr(self.problem, "coupling_coefficient", 1.0)

        # Compute gradient ∂U/∂x using BC-aware computation
        bc = self.get_boundary_conditions()
        grad_U = base_hjb._compute_gradient_array_1d(U, dx, bc=bc, upwind=True, time=time)

        # Compute velocity: v = -coupling * ∂U/∂x
        velocity = -coupling_coefficient * grad_U

        # Build sparse matrix using COO format
        row_indices = []
        col_indices = []
        data_values = []

        for i in range(Nx):
            v_i = velocity[i]

            if abs(v_i) < 1e-14:
                # Zero velocity - no advection contribution
                continue

            if v_i > 0:
                # Flow to right: backward difference (upwind from left)
                # (v * m)' ≈ v_i * (m_i - m_{i-1}) / dx
                # Matrix form: A[i,i] = v_i/dx, A[i,i-1] = -v_i/dx
                row_indices.append(i)
                col_indices.append(i)
                data_values.append(v_i / dx)

                if i > 0:
                    row_indices.append(i)
                    col_indices.append(i - 1)
                    data_values.append(-v_i / dx)
                # Boundary: for i=0, no left neighbor (no-flux BC handles this)
            else:
                # Flow to left: forward difference (upwind from right)
                # (v * m)' ≈ v_i * (m_{i+1} - m_i) / dx
                # Matrix form: A[i,i] = -v_i/dx, A[i,i+1] = v_i/dx
                row_indices.append(i)
                col_indices.append(i)
                data_values.append(-v_i / dx)

                if i < Nx - 1:
                    row_indices.append(i)
                    col_indices.append(i + 1)
                    data_values.append(v_i / dx)
                # Boundary: for i=Nx-1, no right neighbor (no-flux BC handles this)

        # Assemble sparse matrix
        A = sparse.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(Nx, Nx),
        ).tocsr()

        return A

    def _build_advection_matrix_nd(
        self,
        U: NDArray,
        coupling_coefficient: float | None = None,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """Build nD upwind advection matrix.

        Extends 1D logic to multiple dimensions using the same upwind principle
        applied independently in each direction.

        For each dimension d with velocity v_d = -coupling * ∂U/∂x_d:
        - Upwind from appropriate neighbor based on flow direction
        - Matrix entries are summed across all dimensions
        """
        import scipy.sparse as sparse

        # Get coupling coefficient
        if coupling_coefficient is None:
            coupling_coefficient = getattr(self.problem, "coupling_coefficient", 1.0)

        # Compute gradients using trait-based operators
        gradients = self._compute_gradients_nd(U, time=time)

        # Compute velocity in each direction: v_d = -coupling * ∂U/∂x_d
        velocities = {}
        for d in range(self.dimension):
            velocities[d] = -coupling_coefficient * gradients[d]

        # Build sparse matrix using COO format
        row_indices = []
        col_indices = []
        data_values = []

        N_total = int(np.prod(self.shape))

        for flat_idx in range(N_total):
            multi_idx = self.grid.get_multi_index(flat_idx)

            # Process each dimension
            for d in range(self.dimension):
                dx_d = self.spacing[d]
                v_d = velocities[d][multi_idx]

                if abs(v_d) < 1e-14:
                    continue

                if v_d > 0:
                    # Flow in +x_d direction: backward difference
                    row_indices.append(flat_idx)
                    col_indices.append(flat_idx)
                    data_values.append(v_d / dx_d)

                    # Check if neighbor exists
                    if multi_idx[d] > 0:
                        neighbor_idx = list(multi_idx)
                        neighbor_idx[d] -= 1
                        neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat)
                        data_values.append(-v_d / dx_d)
                else:
                    # Flow in -x_d direction: forward difference
                    row_indices.append(flat_idx)
                    col_indices.append(flat_idx)
                    data_values.append(-v_d / dx_d)

                    # Check if neighbor exists
                    if multi_idx[d] < self.shape[d] - 1:
                        neighbor_idx = list(multi_idx)
                        neighbor_idx[d] += 1
                        neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat)
                        data_values.append(v_d / dx_d)

        # Assemble sparse matrix
        A = sparse.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(N_total, N_total),
        ).tocsr()

        return A

    # =========================================================================
    # True Adjoint Mode (Issue #707)
    # =========================================================================

    def build_linearized_operator(
        self,
        U: NDArray,
        M: NDArray,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """
        Build the linearized HJB advection operator (Jacobian) for true adjoint coupling.

        Returns the Hamiltonian advection part of the HJB Jacobian:
            A_adv[i,j] = (dH/dp)_i * (dp_i/dU_j)

        where dH/dp comes from the Hamiltonian class and dp/dU comes from the
        upwind gradient stencil. The transpose A_adv^T is the correct FP
        advection operator (Achdou's structure-preserving discretization).

        This is mathematically different from build_advection_matrix() which uses
        velocity v = -coupling * grad(U). The Jacobian approach is correct for
        arbitrary Hamiltonians, not just quadratic H = (c/2)|p|^2.

        Args:
            U: Value function, shape (*spatial_shape)
            M: Density field, shape (*spatial_shape) -- used for H_p evaluation
            time: Current time for time-dependent problems

        Returns:
            Sparse CSR matrix A_adv of shape (N_total, N_total).
            Transpose A_adv.T gives the correct FP advection operator.

        See Also:
            - Issue #707: True Adjoint Mode
            - Issue #706 (adjoint discretization)
            - compute_hjb_jacobian() in base_hjb.py (1D Newton solver version)
        """
        if self.dimension == 1:
            return self._build_linearized_operator_1d(U, M, time)
        else:
            return self._build_linearized_operator_nd(U, M, time)

    def _build_linearized_operator_1d(
        self,
        U: NDArray,
        M: NDArray,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """Build 1D linearized HJB advection operator.

        Extracts the Hamiltonian advection part of the Jacobian:
            J_D[i] += dH_dp[i] * dp_i/dU_i    (diagonal)
            J_L[i] += dH_dp[i] * dp_i/dU_{i-1} (lower)
            J_U[i] += dH_dp[i] * dp_i/dU_{i+1} (upper)

        where dp/dU depends on the Godunov upwind stencil direction.
        """
        import scipy.sparse as sparse

        Nx = self.problem.geometry.get_grid_shape()[0]
        dx = self.problem.geometry.get_grid_spacing()[0]
        bc = self.get_boundary_conditions()
        H_class = self.problem.hamiltonian_class

        U_flat = np.asarray(U).ravel()
        M_flat = np.asarray(M).ravel()

        J_D = np.zeros(Nx)
        J_L = np.zeros(Nx)
        J_U = np.zeros(Nx)

        if H_class is None:
            raise ValueError(
                "build_linearized_operator requires a Hamiltonian class with dp() method. "
                "Set problem.hamiltonian_class or use build_advection_matrix() for quadratic H."
            )

        # Compute BC-aware upwind gradient
        precomputed_grad = base_hjb._compute_gradient_array_1d(U_flat, dx, bc=bc, upwind=True, time=time)

        # Evaluate dH/dp at all grid points
        x_grid = self.problem.geometry.get_spatial_grid()  # (Nx, 1)
        p_grid = precomputed_grad.reshape(-1, 1)  # (Nx, 1)
        dH_dp = np.asarray(H_class.dp(x_grid, M_flat, p_grid, t=time), dtype=float).ravel()  # (Nx,)

        # Stencil coefficients: dp_i/dU_j depends on Godunov upwind direction
        # p >= 0: backward stencil (U_i - U_{i-1})/dx
        # p < 0:  forward stencil  (U_{i+1} - U_i)/dx
        inv_dx = 1.0 / dx
        backward_mask = precomputed_grad >= 0  # (Nx,)

        # Diagonal: dp_i/dU_i
        J_D = dH_dp * np.where(backward_mask, inv_dx, -inv_dx)
        # Lower: dp_i/dU_{i-1} (backward stencil only)
        J_L = dH_dp * np.where(backward_mask, -inv_dx, 0.0)
        # Upper: dp_i/dU_{i+1} (forward stencil only)
        J_U = dH_dp * np.where(backward_mask, 0.0, inv_dx)

        # Zero out boundary rows (no-flux: boundary nodes have no advection)
        J_D[0] = J_D[-1] = 0.0
        J_L[0] = J_L[-1] = 0.0
        J_U[0] = J_U[-1] = 0.0

        # Assemble tridiagonal sparse matrix
        J_L_shifted = np.roll(J_L, -1) if Nx > 1 else J_L
        J_U_shifted = np.roll(J_U, 1) if Nx > 1 else J_U

        diags = [J_L_shifted, J_D, J_U_shifted] if Nx > 1 else [J_D]
        offsets = [-1, 0, 1] if Nx > 1 else [0]

        return sparse.spdiags(diags, offsets, Nx, Nx, format="csr")

    def _build_linearized_operator_nd(
        self,
        U: NDArray,
        M: NDArray,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """Build nD linearized HJB advection operator.

        Generalizes the 1D approach: for each dimension d, computes
        dH/dp_d * dp_d/dU_j using the upwind stencil in that dimension,
        then sums contributions across all dimensions.
        """
        import scipy.sparse as sparse

        H_class = self.problem.hamiltonian_class
        if H_class is None:
            raise ValueError("build_linearized_operator requires a Hamiltonian class with dp() method.")

        N_total = int(np.prod(self.shape))
        M_flat = np.asarray(M).ravel()

        # Compute upwind gradients in each dimension
        gradients = self._compute_gradients_nd(U, time=time)

        # Stack gradients as (Nx, dim) for H_class.dp()
        x_grid = self.problem.geometry.get_spatial_grid()  # (N_total, dim)
        p_grid = np.column_stack([gradients[d].ravel() for d in range(self.dimension)])

        # Evaluate dH/dp: returns (N_total, dim) -- one component per dimension
        dH_dp_all = np.asarray(H_class.dp(x_grid, M_flat, p_grid, t=time), dtype=float)
        if dH_dp_all.ndim == 1:
            dH_dp_all = dH_dp_all.reshape(-1, 1)

        # Build sparse matrix using COO format
        row_indices = []
        col_indices = []
        data_values = []

        for flat_idx in range(N_total):
            multi_idx = self.grid.get_multi_index(flat_idx)

            # Skip boundary points (no-flux)
            is_boundary = any(multi_idx[d] == 0 or multi_idx[d] == self.shape[d] - 1 for d in range(self.dimension))
            if is_boundary:
                continue

            for d in range(self.dimension):
                dx_d = self.spacing[d]
                inv_dx_d = 1.0 / dx_d
                grad_d = gradients[d][multi_idx]
                dH_dp_d = dH_dp_all[flat_idx, d] if dH_dp_all.shape[1] > 1 else dH_dp_all[flat_idx, 0]

                if abs(dH_dp_d) < 1e-14:
                    continue

                if grad_d >= 0:
                    # Backward stencil: p_d = (U_i - U_{i-1})/dx_d
                    # dp/dU_i = 1/dx, dp/dU_{i-1} = -1/dx
                    row_indices.append(flat_idx)
                    col_indices.append(flat_idx)
                    data_values.append(dH_dp_d * inv_dx_d)

                    neighbor_idx = list(multi_idx)
                    neighbor_idx[d] -= 1
                    neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.shape)
                    row_indices.append(flat_idx)
                    col_indices.append(neighbor_flat)
                    data_values.append(dH_dp_d * (-inv_dx_d))
                else:
                    # Forward stencil: p_d = (U_{i+1} - U_i)/dx_d
                    # dp/dU_i = -1/dx, dp/dU_{i+1} = 1/dx
                    row_indices.append(flat_idx)
                    col_indices.append(flat_idx)
                    data_values.append(dH_dp_d * (-inv_dx_d))

                    neighbor_idx = list(multi_idx)
                    neighbor_idx[d] += 1
                    neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.shape)
                    row_indices.append(flat_idx)
                    col_indices.append(neighbor_flat)
                    data_values.append(dH_dp_d * inv_dx_d)

        A = sparse.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(N_total, N_total),
        ).tocsr()

        return A


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBFDMSolver...")

    # Test 1D problem
    from mfgarchon import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    geometry_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
    problem_1d = MFGProblem(geometry=geometry_1d, T=1.0, Nt=20, sigma=0.1)
    solver_1d = HJBFDMSolver(problem_1d, solver_type="newton")

    # Test solver initialization
    assert solver_1d.dimension == 1
    assert solver_1d.solver_type == "newton"
    assert solver_1d.hjb_method_name == "FDM"

    # Test solve_hjb_system
    import numpy as np

    M_test = np.ones((problem_1d.Nt + 1, problem_1d.Nx + 1)) * 0.5
    U_final = np.zeros(problem_1d.Nx + 1)
    U_prev = np.zeros((problem_1d.Nt + 1, problem_1d.Nx + 1))

    U_solution = solver_1d.solve_hjb_system(
        M_density_evolution_from_FP=M_test,
        U_final_condition_at_T=U_final,
        U_from_prev_picard=U_prev,
    )

    assert U_solution.shape == (problem_1d.Nt + 1, problem_1d.Nx + 1)
    assert not np.any(np.isnan(U_solution))
    assert not np.any(np.isinf(U_solution))

    print("  1D solver converged")
    print(f"  U range: [{U_solution.min():.3f}, {U_solution.max():.3f}]")

    # Test build_advection_matrix (Issue #622 Phase 1)
    print("\nTesting build_advection_matrix (Issue #622)...")
    U_slice = U_solution[10]  # Use a middle timestep
    A_hjb = solver_1d.build_advection_matrix(U_slice)

    import scipy.sparse as sparse

    assert sparse.issparse(A_hjb), "build_advection_matrix should return sparse matrix"
    assert A_hjb.shape == (problem_1d.Nx + 1, problem_1d.Nx + 1), f"Matrix shape mismatch: {A_hjb.shape}"
    assert not np.any(np.isnan(A_hjb.data)), "Matrix contains NaN"
    assert not np.any(np.isinf(A_hjb.data)), "Matrix contains Inf"

    # Verify transpose property: A^T should have same sparsity pattern
    A_hjb_T = A_hjb.T.tocsr()
    assert A_hjb_T.shape == A_hjb.shape, "Transpose should have same shape"

    print(f"  1D advection matrix: shape={A_hjb.shape}, nnz={A_hjb.nnz}")
    print("  build_advection_matrix (1D) passed!")

    # Test 2D problem
    print("\nTesting 2D solver...")
    geometry_2d = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11])
    problem_2d = MFGProblem(geometry=geometry_2d, T=1.0, Nt=5, sigma=0.1)
    solver_2d = HJBFDMSolver(problem_2d, solver_type="newton")

    # Quick 2D build_advection_matrix test
    U_2d = np.zeros((11, 11))
    U_2d[5, 5] = 1.0  # Point source
    A_hjb_2d = solver_2d.build_advection_matrix(U_2d)

    assert sparse.issparse(A_hjb_2d), "2D build_advection_matrix should return sparse matrix"
    assert A_hjb_2d.shape == (11 * 11, 11 * 11), f"2D matrix shape mismatch: {A_hjb_2d.shape}"
    print(f"  2D advection matrix: shape={A_hjb_2d.shape}, nnz={A_hjb_2d.nnz}")
    print("  build_advection_matrix (2D) passed!")

    print("\nAll smoke tests passed!")
