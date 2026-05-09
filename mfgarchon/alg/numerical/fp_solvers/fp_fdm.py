from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.sparse as sparse

from mfgarchon.backends.compat import has_nan_or_inf
from mfgarchon.geometry import BoundaryConditions
from mfgarchon.utils.aux_func import npart, ppart
from mfgarchon.utils.deprecation import deprecated, deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger

from .base_fp import BaseFPSolver
from .fp_fdm_time_stepping import (
    _get_bc_type,
    _get_bc_value,
)
from .fp_fdm_time_stepping import (
    solve_fp_nd_full_system as _solve_fp_nd_full_system,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

# Advection scheme options for FDM (2x2 naming convention)
# Format: {pde_form}_{spatial_scheme}
# - pde_form: "gradient" (v·∇m) or "divergence" (∇·(vm))
# - spatial_scheme: "centered" or "upwind"
AdvectionScheme = Literal[
    "gradient_centered",  # Non-conservative, oscillates for Peclet > 2
    "gradient_upwind",  # Conservative (row sums), stable [DEFAULT]
    "divergence_centered",  # Conservative (telescoping), oscillates for Peclet > 2
    "divergence_upwind",  # Conservative (telescoping), stable
    # Legacy aliases (DEPRECATED, will be removed in v1.0.0)
    "centered",  # -> gradient_centered
    "upwind",  # -> gradient_upwind
    "flux",  # -> divergence_upwind
]


class FPFDMSolver(BaseFPSolver):
    """
    Finite Difference Method (FDM) solver for Fokker-Planck equations.

    Supports general FP equation: dm/dt + div(v*m) = (sigma^2/2) * Laplacian(m)

    Advection Scheme Options (2x2 classification):

        | Scheme             | PDE Form   | Spatial    | Conservative | Stable |
        |--------------------|------------|------------|--------------|--------|
        | gradient_centered  | v·grad(m)  | Central    | NO           | Pe<2   |
        | gradient_upwind    | v·grad(m)  | Upwind     | YES (rows)   | Always |
        | divergence_centered| div(v*m)   | Central    | YES (flux)   | Pe<2   |
        | divergence_upwind  | div(v*m)   | Upwind     | YES (flux)   | Always |

        **gradient_centered**: Non-conservative form with central differences.
            Second-order accurate but oscillates for Peclet > 2.
            Use to demonstrate why conservative schemes are needed.

        **gradient_upwind**: Non-conservative form with upwind differences.
            Mass-conservative via row sums = 1/dt. Stable but first-order.
            WARNING: Has boundary flux bug when flow crosses boundaries.

        **divergence_centered**: Conservative form with centered flux averaging.
            Mass-conservative via flux telescoping. Oscillates for Peclet > 2.
            Demonstrates that conservation alone doesn't guarantee stability.

        **divergence_upwind** [default]: Conservative form with upwind flux selection.
            Mass-conservative via flux telescoping. Stable, first-order.
            Best choice for MFG: handles boundary fluxes correctly.

    Legacy Aliases (DEPRECATED, will be removed in v1.0.0):
        - "centered" -> "gradient_centered"
        - "upwind" -> "gradient_upwind"
        - "flux" -> "divergence_upwind"

    Numerical Scheme:
        - Implicit timestepping for stability
        - Central differences for diffusion terms
        - Supports periodic, Dirichlet, and no-flux boundary conditions

    Required Geometry Traits (Issue #596 Phase 2.2A):
        - SupportsLaplacian: Provides Δm operator for diffusion term (σ²/2) Δm

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsLaplacian

    Note:
        Advection operators currently use manual sparse matrix construction.
        Future work (Issue #597) will integrate trait-based advection operators.
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.FDM

    def __init__(
        self,
        problem: Any,
        boundary_conditions: BoundaryConditions | None = None,
        advection_scheme: AdvectionScheme = "divergence_upwind",
    ) -> None:
        """
        Initialize FDM solver for Fokker-Planck equations.

        Parameters
        ----------
        problem : Any
            MFG problem definition
        boundary_conditions : BoundaryConditions | None
            Boundary condition specification (default: no-flux)
        advection_scheme : str
            Advection term discretization (default: "divergence_upwind").

            Scheme names:
            - "gradient_centered": v·grad(m) with central diff, NOT conservative
            - "gradient_upwind": v·grad(m) with upwind, has boundary flux bug
            - "divergence_centered": div(v*m) with centered flux, conservative (telescoping)
            - "divergence_upwind": div(v*m) with upwind flux, conservative (telescoping) [DEFAULT]

            Legacy names (DEPRECATED, will be removed in v1.0.0):
            - "centered" -> gradient_centered
            - "upwind" -> gradient_upwind
            - "flux" -> divergence_upwind
        """
        import warnings

        super().__init__(problem)
        self.fp_method_name = "FDM"

        # Map legacy scheme names to new names
        scheme_aliases = {
            "centered": "gradient_centered",
            "upwind": "gradient_upwind",
            "flux": "divergence_upwind",
        }

        # Emit deprecation warning for legacy aliases
        if advection_scheme in scheme_aliases:
            new_name = scheme_aliases[advection_scheme]
            warnings.warn(
                f"advection_scheme='{advection_scheme}' is deprecated. "
                f"Use advection_scheme='{new_name}' instead. "
                f"Legacy aliases will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            advection_scheme = new_name

        # Validate scheme name (only new names accepted after mapping)
        valid_schemes = {"gradient_centered", "gradient_upwind", "divergence_centered", "divergence_upwind"}
        if advection_scheme not in valid_schemes:
            raise ValueError(f"Invalid advection_scheme: '{advection_scheme}'. Valid options: {sorted(valid_schemes)}")

        self.advection_scheme = advection_scheme

        # Detect problem dimension first (inherited from BaseNumericalSolver, Issue #633)
        self.dimension = self._detect_dimension()

        # Validate geometry capabilities (Issue #596 Phase 2.2A)
        # FP solver requires Laplacian operator for diffusion term
        from mfgarchon.geometry.protocols import SupportsLaplacian

        if not isinstance(problem.geometry, SupportsLaplacian):
            raise TypeError(
                f"FP FDM solver requires geometry with SupportsLaplacian trait for diffusion term. "
                f"{type(problem.geometry).__name__} does not implement this trait. "
                f"Compatible geometries: TensorProductGrid, ImplicitDomain."
            )

        # Boundary condition resolution hierarchy:
        # Issue #543 Phase 2: Replace hasattr with try/except cascade
        # Issue #527: Align with centralized BC resolution from BaseMFGSolver
        # 1. Explicit boundary_conditions parameter (highest priority)
        # 2. Problem components BC (if available and not None)
        # 3. geometry.boundary_conditions (attribute) - standard path
        # 4. geometry.get_boundary_conditions() (method accessor)
        # 5. Grid geometry boundary handler (legacy, if available)
        # 6. Default no-flux BC (fallback)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            bc_found = False

            # Try components BC
            try:
                if problem.components is not None and problem.components.boundary_conditions is not None:
                    self.boundary_conditions = problem.components.boundary_conditions
                    bc_found = True
            except AttributeError:
                pass  # No components attribute, continue to next option

            # Try geometry.boundary_conditions (standard path - Issue #527)
            if not bc_found:
                try:
                    bc = problem.geometry.boundary_conditions
                    if bc is not None:
                        self.boundary_conditions = bc
                        bc_found = True
                except AttributeError:
                    pass  # No boundary_conditions attribute

            # Try geometry.get_boundary_conditions() (method accessor - Issue #527)
            if not bc_found:
                try:
                    bc = problem.geometry.get_boundary_conditions()
                    if bc is not None:
                        self.boundary_conditions = bc
                        bc_found = True
                except AttributeError:
                    pass  # No get_boundary_conditions method

            # Try geometry BC handler (legacy support)
            if not bc_found:
                try:
                    self.boundary_conditions = problem.geometry.get_boundary_handler()
                    bc_found = True
                except AttributeError:
                    pass  # No geometry BC handler

            # Default to no-flux if no BC found
            if not bc_found:
                from mfgarchon.geometry.boundary import no_flux_bc

                self.boundary_conditions = no_flux_bc(dimension=self.dimension)

    # _detect_dimension() inherited from BaseNumericalSolver (Issue #633)

    def _log_cfl_diagnostic(self, volatility_field: float | None = None) -> None:
        """Log CFL diagnostic for accuracy/convergence guidance (Issue #882, #1052).

        Issue #1052: log once at INFO per solver instance, subsequent calls at
        DEBUG. CFL parameters are static across Picard iterations.
        """
        try:
            dt = self.problem.dt
            dx = self.problem.geometry.get_grid_spacing()[0]
            sigma = volatility_field if isinstance(volatility_field, (int, float)) else self.problem.sigma
            cfl_diffusive = sigma**2 * dt / dx**2
            if cfl_diffusive > 0.5:
                log_fn = logger.debug if getattr(self, "_cfl_logged", False) else logger.info
                log_fn(
                    "CFL diagnostic (FP FDM): diffusive=%.2f (sigma=%.3g, dt=%.3g, dx=%.3g). "
                    "Implicit scheme is stable but accuracy may degrade for CFL >> 1.",
                    cfl_diffusive,
                    sigma,
                    dt,
                    dx,
                )
                self._cfl_logged = True
        except (AttributeError, IndexError, TypeError):
            pass  # Not enough info to compute CFL — skip silently

    @deprecated_parameter(param_name="m_initial_condition", since="v0.17.0", replacement="M_initial")
    @deprecated_parameter(param_name="diffusion_field", since="v0.17.0", replacement="volatility_field")
    @deprecated_parameter(param_name="tensor_diffusion_field", since="v0.17.0", replacement="volatility_field")
    @deprecated_parameter(param_name="volatility_matrix", since="v0.17.0", replacement="volatility_field")
    @deprecated_parameter(param_name="velocity_field", since="v0.18.6", replacement="drift_field")
    @deprecated_parameter(param_name="potential_field", since="v0.18.6", replacement="drift_field")
    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
        progress_callback: Callable[[int], None] | None = None,  # Issue #640
        # Deprecated parameter names for backward compatibility
        m_initial_condition: np.ndarray | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,  # Issue #717: deprecated
        tensor_diffusion_field: np.ndarray | Callable | None = None,  # Issue #717: deprecated
        volatility_matrix: np.ndarray | Callable | None = None,  # Deprecated: use volatility_field
        # Deprecated: velocity_field renamed to drift_field (v0.18.6)
        velocity_field: np.ndarray | None = None,
        # Deprecated: old drift_field (U-potential) renamed to potential_field (v0.18.6)
        potential_field: np.ndarray | None = None,
        # MMS verification support
        source_term: Callable | None = None,
    ) -> np.ndarray:
        """
        Solve FP system forward in time with general drift and diffusion support.

        Implements BaseFPSolver unified API for both drift and diffusion.
        Automatically routes to 1D or nD solver based on problem dimension.

        Parameters
        ----------
        M_initial : np.ndarray
            Initial density m₀(x). Shape: (Nx+1,) for 1D or (N1-1, N2-1, ...) for nD
        m_initial_condition : np.ndarray
            DEPRECATED, use M_initial
        drift_field : np.ndarray or callable, optional
            Drift velocity specification (Issue #573):
            - None: Zero drift (pure diffusion)
            - np.ndarray: Drift velocity α*(t,x), shape (Nt+1, Nx) for 1D or (Nt+1, N1, N2, ...) for nD
              Caller computes α* = -∂_p H(x, ∇U, m) for their Hamiltonian:
                * Quadratic H = (1/2)|p|²: α* = -∇U
                * L1 control H = |p|: α* = -sign(∇U)
                * Quartic H = (1/4)|p|⁴: α* = -sign(∇U) |∇U|^(1/3)
                * Custom H: Any function of ∇U
            - Callable: Custom drift function α(t, x, m) -> drift_vector
              Signature: (t: float, x_coords: list, m: ndarray) -> ndarray
            Default: None
        volatility_field : float, np.ndarray, or callable, optional
            Volatility specification (unified API). Auto-detects scalar vs matrix:
            - None: Use problem.sigma (backward compatible)
            - float: Constant isotropic volatility σ → D = σ²/2
            - (d,) array: Diagonal volatility [σ₀, σ₁, ...] → D = diag(σᵢ²)/2
            - (d, d) array: Full volatility matrix Σ → D = ΣΣᵀ/2
            - (*shape, d, d) array: Spatially varying Σ(x) → D(x) = Σ(x)Σ(x)ᵀ/2
            - Callable: State-dependent σ(t, x, m) or Σ(t, x, m)
            Default: None
        diffusion_field : DEPRECATED, use volatility_field
        tensor_diffusion_field : DEPRECATED, use volatility_field with (d,d) array
        volatility_matrix : DEPRECATED, use volatility_field with (d,d) array
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        np.ndarray
            Density evolution. Shape: (Nt+1, Nx+1) for 1D or
            (Nt+1, N1-1, N2-1, ...) for nD

        Examples
        --------
        Pure diffusion (heat equation):
        >>> M = solver.solve_fp_system(m0)

        MFG optimal control:
        >>> drift = -problem.compute_gradient(U_hjb) / problem.control_cost
        >>> M = solver.solve_fp_system(m0, drift_field=drift)

        Custom diffusion coefficient:
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=0.5)

        Spatially varying diffusion (higher at boundaries):
        >>> Nx = problem.geometry.get_grid_shape()[0]
        >>> x_grid = np.linspace(0, 1, Nx)
        >>> diffusion_array = 0.1 + 0.2 * np.abs(x_grid - 0.5)
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=diffusion_array)

        Spatiotemporal diffusion (time and space dependent):
        >>> Nt, Nx = problem.Nt + 1, problem.geometry.get_grid_shape()[0]
        >>> diffusion_field = np.zeros((Nt, Nx))
        >>> for t in range(Nt):
        ...     diffusion_field[t, :] = 0.1 * (1 + 0.5 * t / Nt)  # Increasing over time
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=diffusion_field)

        State-dependent diffusion (porous medium equation):
        >>> def porous_medium(t, x, m):
        ...     return 0.1 * m  # Diffusion proportional to density
        >>> M = solver.solve_fp_system(m0, diffusion_field=porous_medium)

        Density-dependent diffusion with drift:
        >>> def crowd_diffusion(t, x, m):
        ...     return 0.05 + 0.15 * (1 - m / np.max(m))  # Lower diffusion in crowds
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=crowd_diffusion)

        Pure advection (zero diffusion):
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=0.0)

        Anisotropic volatility (unified API):
        >>> # Diagonal volatility: faster horizontal diffusion
        >>> Sigma = np.diag([0.2, 0.05])  # σ_x=0.2, σ_y=0.05 → D = diag(0.02, 0.00125)
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=Sigma)
        >>>
        >>> # Or pass as 1D array (auto-converted to diagonal matrix):
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=[0.2, 0.05])

        Full tensor with cross-diffusion:
        >>> # 2x2 symmetric tensor
        >>> Sigma = np.array([[0.2, 0.05], [0.05, 0.1]])
        >>> M = solver.solve_fp_system(m0, drift_field=drift, tensor_diffusion_field=Sigma)

        State-dependent tensor diffusion:
        >>> def anisotropic_crowd(t, x, m):
        ...     # Reduce perpendicular diffusion in crowds
        ...     sigma_parallel = 0.2
        ...     sigma_perp = 0.05 * (1 - m / np.max(m))
        ...     return np.diag([sigma_parallel, sigma_perp])
        >>> M = solver.solve_fp_system(m0, tensor_diffusion_field=anisotropic_crowd)

        Non-quadratic Hamiltonians (Issue #573):

        Quadratic control (H = (1/2)|p|²):
        >>> U_hjb = hjb_solver.solve(M_density)
        >>> grad_U = problem.compute_gradient(U_hjb)
        >>> alpha_quadratic = -grad_U  # α* = -∇U for quadratic H
        >>> M = solver.solve_fp_system(m0, drift_field=alpha_quadratic)

        L1 control cost (H = |p|, minimal fuel):
        >>> U_hjb = hjb_solver.solve_hjb_L1(M_density)
        >>> grad_U = problem.compute_gradient(U_hjb)
        >>> alpha_L1 = -np.sign(grad_U)  # α* = -sign(∇U) for L1 H
        >>> M = solver.solve_fp_system(m0, drift_field=alpha_L1)

        Quartic control cost (H = (1/4)|p|⁴):
        >>> U_hjb = hjb_solver.solve_hjb_quartic(M_density)
        >>> grad_U = problem.compute_gradient(U_hjb)
        >>> alpha_quartic = -np.sign(grad_U) * np.abs(grad_U) ** (1/3)  # α* = -(∇U)^(1/3)
        >>> M = solver.solve_fp_system(m0, drift_field=alpha_quartic)
        """
        # Handle deprecated parameter name
        if m_initial_condition is not None:
            if M_initial is not None:
                raise ValueError(
                    "Cannot specify both M_initial and m_initial_condition. "
                    "Use M_initial (m_initial_condition is deprecated)."
                )
            M_initial = m_initial_condition

        # Validate required parameter
        if M_initial is None:
            raise ValueError("M_initial is required")

        # Handle deprecated velocity_field -> drift_field (v0.18.6)
        if velocity_field is not None:
            if drift_field is not None:
                raise ValueError(
                    "Cannot specify both drift_field and velocity_field. "
                    "Use drift_field (velocity_field is deprecated)."
                )
            drift_field = velocity_field

        # Handle deprecated potential_field (old drift_field with U-potential)
        if potential_field is not None:
            if drift_field is not None:
                raise ValueError(
                    "Cannot specify both drift_field and potential_field. "
                    "potential_field is deprecated; pass velocity via drift_field instead."
                )
            # potential_field is U-potential — route through internal U path
            effective_U = potential_field
        elif drift_field is not None:
            # drift_field is velocity α*(t,x) — route through velocity path
            if isinstance(drift_field, np.ndarray):
                effective_U = None  # Not needed when velocity is provided directly
            elif callable(drift_field):
                # Custom drift function - Phase 2
                # Route to unified nD solver (works for all dimensions including 1D)
                return _solve_fp_nd_full_system(
                    m_initial_condition=M_initial,
                    U_solution_for_drift=None,
                    problem=self.problem,
                    boundary_conditions=self.boundary_conditions,
                    show_progress=show_progress,
                    backend=self.backend,
                    diffusion_field=volatility_field,
                    drift_field=drift_field,  # callable velocity → internal drift_field
                    advection_scheme=self.advection_scheme,
                    progress_callback=progress_callback,
                    source_term=source_term,
                )
            else:
                raise TypeError(f"drift_field must be np.ndarray or Callable, got {type(drift_field)}")
        else:
            # Zero drift (pure diffusion): create zero U field for internal use
            try:
                Nt = self.problem.Nt + 1
            except AttributeError as e:
                raise ValueError("Cannot infer time steps. Ensure problem has Nt attribute.") from e

            # Create zero U field with appropriate shape
            if self.dimension == 1:
                Nx_val = getattr(self.problem, "Nx", None)
                Nx = Nx_val + 1 if Nx_val is not None else self.problem.geometry.get_grid_shape()[0]
                effective_U = np.zeros((Nt, Nx))
            else:
                grid_shape = self.problem.geometry.get_grid_shape()
                effective_U = np.zeros((Nt, *grid_shape))

        # Issue #717: Handle deprecated parameter names
        if diffusion_field is not None:
            if volatility_field is not None:
                raise ValueError(
                    "Cannot specify both volatility_field and diffusion_field. "
                    "Use volatility_field (diffusion_field is deprecated)."
                )
            volatility_field = diffusion_field

        # Handle deprecated tensor_diffusion_field → volatility_field
        # Track if input came from tensor-specific parameter (for callable routing)
        _from_tensor_param = False
        if tensor_diffusion_field is not None:
            if volatility_field is not None:
                raise ValueError(
                    "Cannot specify both volatility_field and tensor_diffusion_field. "
                    "Use volatility_field (tensor_diffusion_field is deprecated)."
                )
            volatility_field = tensor_diffusion_field
            _from_tensor_param = True

        # Handle deprecated volatility_matrix → volatility_field
        if volatility_matrix is not None:
            if volatility_field is not None:
                raise ValueError(
                    "Cannot specify both volatility_field and volatility_matrix. "
                    "Use volatility_field (volatility_matrix is deprecated)."
                )
            volatility_field = volatility_matrix
            _from_tensor_param = True

        # Unified volatility_field handling with auto-detection
        # Issue #717: volatility_field is the SDE volatility σ or Σ
        # The solver computes D = σ²/2 (scalar) or D = ΣΣᵀ/2 (matrix) internally
        if volatility_field is None:
            # Use problem.sigma (backward compatible)
            effective_sigma = self.problem.sigma
            is_tensor = False
        elif isinstance(volatility_field, (int, float)):
            # Constant isotropic volatility
            effective_sigma = float(volatility_field)
            is_tensor = False
        elif isinstance(volatility_field, np.ndarray):
            # Auto-detect: scalar field vs matrix volatility
            d = self.dimension
            if volatility_field.ndim == 2 and volatility_field.shape == (d, d):
                # Constant volatility matrix Σ (d × d)
                is_tensor = True
                effective_sigma = volatility_field
            elif volatility_field.ndim >= 2 and volatility_field.shape[-2:] == (d, d):
                # Spatially varying volatility Σ(x) with shape (*spatial, d, d)
                is_tensor = True
                effective_sigma = volatility_field
            elif volatility_field.ndim == 1 and len(volatility_field) == d:
                # Diagonal volatility [σ₀, σ₁, ...] → convert to diag matrix
                is_tensor = True
                effective_sigma = np.diag(volatility_field)
            else:
                # Scalar field (spatial or spatiotemporal varying σ)
                is_tensor = False
                effective_sigma = volatility_field
        elif callable(volatility_field):
            # State-dependent volatility - callable σ(t, x, m) or Σ(t, x, m)
            # Issue #641: Always route to unified nD solver (handles 1D too)
            effective_sigma = volatility_field
            # If came from tensor-specific deprecated param, route to tensor path
            # Otherwise, route to scalar path (runtime detection not yet implemented)
            is_tensor = _from_tensor_param
        else:
            raise TypeError(
                f"volatility_field must be None, float, np.ndarray, or Callable, got {type(volatility_field)}"
            )

        # CFL diagnostic (Issue #882)
        self._log_cfl_diagnostic(volatility_field)

        # Resolve velocity for internal routing:
        # drift_field (after deprecation handling) is velocity when effective_U is None
        _internal_velocity = drift_field if (effective_U is None and isinstance(drift_field, np.ndarray)) else None

        # Route tensor volatility to tensor path
        if is_tensor:
            if self.dimension == 1:
                raise NotImplementedError(
                    "Anisotropic volatility not yet implemented for 1D problems. Use scalar volatility_field for 1D."
                )
            # Route to nD solver with tensor volatility
            return _solve_fp_nd_full_system(
                m_initial_condition=M_initial,
                U_solution_for_drift=effective_U,
                problem=self.problem,
                boundary_conditions=self.boundary_conditions,
                show_progress=show_progress,
                backend=self.backend,
                diffusion_field=None,
                tensor_diffusion_field=effective_sigma,  # Internal API uses old name
                advection_scheme=self.advection_scheme,
                progress_callback=progress_callback,
                source_term=source_term,
                velocity_field=_internal_velocity,
            )

        # Issue #641: Always route to unified nD solver (works for all dimensions)
        # Internal API still uses diffusion_field name for backward compatibility
        return _solve_fp_nd_full_system(
            m_initial_condition=M_initial,
            U_solution_for_drift=effective_U,
            problem=self.problem,
            boundary_conditions=self.boundary_conditions,
            show_progress=show_progress,
            backend=self.backend,
            diffusion_field=effective_sigma if volatility_field is not None else None,
            advection_scheme=self.advection_scheme,
            progress_callback=progress_callback,
            source_term=source_term,
            velocity_field=_internal_velocity,
        )

    # =========================================================================
    # Strict Adjoint Mode (Issue #622)
    # =========================================================================

    def solve_fp_step_adjoint_mode(
        self,
        M_current: np.ndarray,
        A_advection_T: sparse.csr_matrix,
        sigma: float | np.ndarray | None = None,
        time: float = 0.0,
    ) -> np.ndarray:
        """
        Solve single FP timestep using externally provided advection matrix.

        This method is used in strict adjoint mode (Issue #622) where the
        FP solver uses A^T from the HJB solver instead of building its own
        advection matrix. This guarantees exact adjoint consistency:
            L_FP = L_HJB^T

        Mathematical Formulation:
            FP equation: dm/dt + ∇·(vm) = (σ²/2) Δm

            Discretized with A_advection_T (transpose of HJB's matrix):
                (I/dt + A_advection_T + D) m^{k+1} = m^k / dt

            where:
            - A_advection_T: Advection matrix from HJB solver (transposed)
            - D: Diffusion matrix (built internally, symmetric so D = D^T)
            - I: Identity matrix

        Args:
            M_current: Current density at timestep k, shape (*spatial_shape)
            A_advection_T: Transposed advection matrix from HJB solver.
                Shape: (N_total, N_total) where N_total = prod(spatial_shape).
                This is A_hjb.T where A_hjb was built by HJBFDMSolver.build_advection_matrix().
            sigma: Diffusion coefficient (optional).
                - None: Use problem.sigma
                - float: Constant diffusion
                - np.ndarray: Spatially varying diffusion
            time: Current time for time-dependent BCs

        Returns:
            M_next: Density at timestep k+1, shape (*spatial_shape)

        Example:
            >>> # In FixedPointIterator with strict_adjoint=True:
            >>> A_hjb = hjb_solver.build_advection_matrix(U_current)
            >>> M_next = fp_solver.solve_fp_step_adjoint_mode(M_current, A_hjb.T)

        Note:
            The diffusion operator is symmetric (D = D^T), so using this method
            with A_hjb.T ensures the full spatial operator satisfies L_FP = L_HJB^T
            for the advection part while diffusion remains adjoint-consistent by symmetry.

        See Also:
            - Issue #622: Strict Achdou adjoint mode implementation
            - HJBFDMSolver.build_advection_matrix(): Builds the advection matrix
            - FixedPointIterator: Orchestrates matrix passing between solvers
        """
        # Get problem dimensions
        shape = M_current.shape
        N_total = int(np.prod(shape))
        dt = self.problem.dt

        # Validate matrix shape
        if A_advection_T.shape != (N_total, N_total):
            raise ValueError(
                f"A_advection_T shape {A_advection_T.shape} doesn't match expected "
                f"({N_total}, {N_total}) for density shape {shape}"
            )

        # Get diffusion coefficient
        if sigma is None:
            sigma_val = self.problem.sigma
        else:
            sigma_val = sigma

        # Build diffusion matrix using LaplacianOperator
        from mfgarchon.operators.differential.laplacian import LaplacianOperator

        spacing = list(self.problem.geometry.get_grid_spacing())
        bc = self.boundary_conditions

        L_op = LaplacianOperator(spacings=spacing, field_shape=shape, bc=bc)
        L_matrix = L_op.as_scipy_sparse()

        # Compute diffusion coefficient
        if isinstance(sigma_val, np.ndarray):
            # For spatially varying diffusion, use mean (approximation)
            # TODO: Support full spatially varying in matrix form
            D = 0.5 * float(np.mean(sigma_val)) ** 2
        else:
            D = 0.5 * float(sigma_val) ** 2

        # Build full system matrix: (I/dt + A_advection_T - D*Laplacian)
        # Note: Laplacian has negative diagonal, so we SUBTRACT D*L
        identity = sparse.eye(N_total)
        A_system = identity / dt + A_advection_T - D * L_matrix

        # Right-hand side
        b_rhs = M_current.ravel() / dt

        # Solve linear system
        M_next_flat = sparse.linalg.spsolve(A_system, b_rhs)

        # Reshape and ensure non-negativity
        M_next = M_next_flat.reshape(shape)
        M_next = np.maximum(M_next, 0.0)

        return M_next

    @deprecated(since="v0.17.1", replacement="solve_fp_system")
    def _solve_fp_1d(
        self,
        m_initial_condition: np.ndarray,
        U_solution_for_drift: np.ndarray,
        show_progress: bool | None = None,
        progress_callback: Callable[[int], None] | None = None,  # Issue #640
    ) -> np.ndarray:
        """
        Original 1D FP solver implementation.

        .. deprecated:: 0.17.1
            This method is deprecated in favor of `solve_fp_nd_full_system` which
            handles all dimensions including 1D. Will be removed in v1.0.0.

            The unified nD solver provides:
            - Consistent behavior across all dimensions
            - Full BC support (no_flux, neumann, robin, periodic, dirichlet)
            - Cleaner codebase with less code path branching
        """
        # Use geometry-based interface (geometry is always available)
        Nx = self.problem.geometry.get_grid_shape()[0]
        Dx = self.problem.geometry.get_grid_spacing()[0]
        Dt = self.problem.dt

        # Infer number of time points from U_solution shape, not problem.Nt
        # n_time_points = number of time knots (including t=0 and t=T)
        # This allows tests to pass edge cases like n_time_points=0 or n_time_points=1
        n_time_points = U_solution_for_drift.shape[0]
        sigma_base = self.problem.sigma  # Base diffusion (scalar or array)
        coupling_coefficient = getattr(self.problem, "coupling_coefficient", 1.0)

        if n_time_points == 0:
            if self.backend is not None:
                return self.backend.zeros((0, Nx))
            return np.zeros((0, Nx))
        if n_time_points == 1:
            if self.backend is not None:
                m_sol = self.backend.zeros((1, Nx))
            else:
                m_sol = np.zeros((1, Nx))
            m_sol[0, :] = m_initial_condition
            m_sol[0, :] = np.maximum(m_sol[0, :], 0)
            # Apply boundary conditions
            if _get_bc_type(self.boundary_conditions) == "dirichlet":
                m_sol[0, 0] = _get_bc_value(self.boundary_conditions, "x_min")
                m_sol[0, -1] = _get_bc_value(self.boundary_conditions, "x_max")
            return m_sol

        if self.backend is not None:
            m = self.backend.zeros((n_time_points, Nx))
        else:
            m = np.zeros((n_time_points, Nx))
        m[0, :] = m_initial_condition
        m[0, :] = np.maximum(m[0, :], 0)
        # Apply boundary conditions to initial condition
        bc_type = _get_bc_type(self.boundary_conditions)
        if bc_type == "dirichlet":
            m[0, 0] = _get_bc_value(self.boundary_conditions, "x_min")
            m[0, -1] = _get_bc_value(self.boundary_conditions, "x_max")

        # Pre-allocate lists for COO format, then convert to CSR
        row_indices: list[int] = []
        col_indices: list[int] = []
        data_values: list[float] = []

        # Progress bar for forward timesteps
        # Forward FP loop: (n_time_points - 1) steps from index 0 to (n_time_points - 2)
        # Issue #640: When progress_callback is provided (from HierarchicalProgress),
        # suppress internal bar to avoid duplicate progress display
        use_external_progress = progress_callback is not None
        timestep_range = range(n_time_points - 1)
        from mfgarchon.utils.progress import create_progress_bar, should_show_progress

        timestep_range = create_progress_bar(
            timestep_range,
            verbose=should_show_progress(show_progress) and not use_external_progress,
            desc="FP (forward)",
        )

        for k_idx_fp in timestep_range:
            if Dt < 1e-14:
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue
            if Dx < 1e-14 and Nx > 1:
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue

            u_at_tk = U_solution_for_drift[k_idx_fp, :]

            # Extract diffusion coefficient at current timestep
            # Handle scalar (constant) or array (spatially/temporally varying)
            if isinstance(sigma_base, np.ndarray):
                # Array diffusion: shape (Nt, Nx) or broadcastable
                if sigma_base.ndim == 1:
                    # Spatially varying only: sigma.shape = (Nx,)
                    sigma_at_k = sigma_base
                elif sigma_base.ndim == 2:
                    # Spatiotemporal: sigma.shape = (Nt, Nx)
                    sigma_at_k = sigma_base[k_idx_fp, :]
                else:
                    raise ValueError(
                        f"diffusion_field array must be 1D (Nx,) or 2D (Nt, Nx), got shape {sigma_base.shape}"
                    )
            else:
                # Scalar diffusion (constant)
                sigma_at_k = sigma_base

            row_indices.clear()
            col_indices.clear()
            data_values.clear()

            # Handle different boundary conditions
            if bc_type == "periodic":
                # Original periodic boundary implementation
                for i in range(Nx):
                    # Get diffusion at point i (scalar or array)
                    sigma_i = sigma_at_k[i] if isinstance(sigma_at_k, np.ndarray) else sigma_at_k

                    # Diagonal term for m_i^{k+1}
                    val_A_ii = 1.0 / Dt
                    if Nx > 1:
                        val_A_ii += sigma_i**2 / Dx**2
                        # Advection part of diagonal (outflow from cell i)
                        ip1 = (i + 1) % Nx
                        im1 = (i - 1 + Nx) % Nx
                        val_A_ii += float(
                            coupling_coefficient
                            * (npart(u_at_tk[ip1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[im1]))
                            / Dx**2
                        )

                    row_indices.append(i)
                    col_indices.append(i)
                    data_values.append(val_A_ii)

                    if Nx > 1:
                        # Lower diagonal term
                        im1 = (i - 1 + Nx) % Nx  # Previous cell index (periodic)
                        val_A_i_im1 = -(sigma_i**2) / (2 * Dx**2)
                        val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[im1]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(im1)
                        data_values.append(val_A_i_im1)

                        # Upper diagonal term
                        ip1 = (i + 1) % Nx  # Next cell index (periodic)
                        val_A_i_ip1 = -(sigma_i**2) / (2 * Dx**2)
                        val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[ip1] - u_at_tk[i]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(ip1)
                        data_values.append(val_A_i_ip1)

            elif bc_type == "dirichlet":
                # Dirichlet boundary conditions: m[0] = left_value, m[Nx-1] = right_value
                for i in range(Nx):
                    if i == 0 or i == Nx - 1:
                        # Boundary points: identity equation m[i] = boundary_value
                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(1.0)
                    else:
                        # Get diffusion at point i (scalar or array)
                        sigma_i = sigma_at_k[i] if isinstance(sigma_at_k, np.ndarray) else sigma_at_k

                        # Interior points: standard FDM discretization
                        val_A_ii = 1.0 / Dt
                        if Nx > 1:
                            val_A_ii += sigma_i**2 / Dx**2
                            # Advection part (no wrapping for interior points)
                            if i > 0 and i < Nx - 1:
                                val_A_ii += float(
                                    coupling_coefficient
                                    * (npart(u_at_tk[i + 1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[i - 1]))
                                    / Dx**2
                                )

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        if Nx > 1 and i > 0:
                            # Lower diagonal term (flux from left)
                            val_A_i_im1 = -(sigma_i**2) / (2 * Dx**2)
                            val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                        if Nx > 1 and i < Nx - 1:
                            # Upper diagonal term (flux from right)
                            val_A_i_ip1 = -(sigma_i**2) / (2 * Dx**2)
                            val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

            elif bc_type == "no_flux":
                # Two discretization modes based on advection_scheme:
                # - divergence_*: Flux FDM with interface velocities (mass-preserving)
                # - gradient_*: Gradient FDM (original, may lose mass)
                is_conservative_scheme = self.advection_scheme.startswith("divergence")

                if is_conservative_scheme:
                    # Conservative Flux FDM: discretize div(alpha * m) as flux differences
                    # Interface velocity: alpha_{i+1/2} = -coupling * (u[i+1] - u[i]) / Dx
                    # Upwind flux: F_{i+1/2} = alpha * m_upwind
                    # Column sums = 0 for advection part -> exact mass conservation

                    for i in range(Nx):
                        sigma_i = sigma_at_k[i] if isinstance(sigma_at_k, np.ndarray) else sigma_at_k

                        # Start with time derivative and diffusion
                        val_A_ii = 1.0 / Dt + sigma_i**2 / Dx**2
                        val_A_i_im1 = 0.0
                        val_A_i_ip1 = 0.0

                        # Diffusion coupling (symmetric, standard centered)
                        if i > 0:
                            val_A_i_im1 -= sigma_i**2 / (2 * Dx**2)
                        if i < Nx - 1:
                            val_A_i_ip1 -= sigma_i**2 / (2 * Dx**2)

                        # Right interface: F_{i+1/2}
                        if i < Nx - 1:
                            # Interface velocity at x_{i+1/2}
                            alpha_right = -coupling_coefficient * (u_at_tk[i + 1] - u_at_tk[i]) / Dx

                            if alpha_right >= 0:
                                # Flow to the right: upwind from m_i
                                # F_{i+1/2} = alpha_right * m_i
                                # In row i: +alpha_right/Dx (outflow from cell i)
                                val_A_ii += alpha_right / Dx
                            else:
                                # Flow to the left: upwind from m_{i+1}
                                # F_{i+1/2} = alpha_right * m_{i+1}
                                # In row i: coefficient on m_{i+1}
                                val_A_i_ip1 += alpha_right / Dx

                        # Left interface: -F_{i-1/2}
                        if i > 0:
                            # Interface velocity at x_{i-1/2}
                            alpha_left = -coupling_coefficient * (u_at_tk[i] - u_at_tk[i - 1]) / Dx

                            if alpha_left >= 0:
                                # Flow to the right: upwind from m_{i-1}
                                # F_{i-1/2} = alpha_left * m_{i-1}
                                # In row i: -alpha_left/Dx (inflow to cell i)
                                val_A_i_im1 -= alpha_left / Dx
                            else:
                                # Flow to the left: upwind from m_i
                                # F_{i-1/2} = alpha_left * m_i
                                # In row i: -alpha_left/Dx * m_i (outflow from cell i)
                                val_A_ii -= alpha_left / Dx

                        # Boundary treatment: F at domain boundary = 0 (no flux)
                        # This is automatic: we simply don't add flux terms at boundaries
                        # i=0: no left interface flux, only right interface
                        # i=Nx-1: no right interface flux, only left interface

                        # Add matrix entries
                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        if i > 0 and abs(val_A_i_im1) > 1e-15:
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                        if i < Nx - 1 and abs(val_A_i_ip1) > 1e-15:
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

                else:
                    # Non-conservative Gradient FDM (original implementation)
                    # Bug #8 Fix: No-flux boundaries WITH advection
                    # Previous "partial fix" dropped advection at boundaries → mass leaked
                    # New strategy: Include advection with one-sided stencils
                    # Accept ~1-2% FDM discretization error as normal

                    for i in range(Nx):
                        # Get diffusion at point i (scalar or array)
                        sigma_i = sigma_at_k[i] if isinstance(sigma_at_k, np.ndarray) else sigma_at_k

                        if i == 0:
                            # Left boundary: include both diffusion AND advection
                            # Use one-sided (forward) stencil for velocity gradient

                            # Diagonal term: time + diffusion + advection (upwind)
                            val_A_ii = 1.0 / Dt + sigma_i**2 / Dx**2

                            # Add advection contribution (one-sided upwind scheme)
                            # For left boundary, use forward difference for velocity
                            # Only positive part contributes (flux out of domain)
                            if Nx > 1:
                                val_A_ii += float(coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)

                            row_indices.append(i)
                            col_indices.append(i)
                            data_values.append(val_A_ii)

                            # Coupling to m[1]: diffusion + advection
                            val_A_i_ip1 = -(sigma_i**2) / Dx**2
                            if Nx > 1:
                                val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)

                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

                        elif i == Nx - 1:
                            # Right boundary: include both diffusion AND advection
                            # Use one-sided (backward) stencil for velocity gradient

                            # Diagonal term: time + diffusion + advection (upwind)
                            val_A_ii = 1.0 / Dt + sigma_i**2 / Dx**2

                            # Add advection contribution (one-sided upwind scheme)
                            # For right boundary, use backward difference for velocity
                            # Only negative part contributes (flux out of domain)
                            if Nx > 1:
                                val_A_ii += float(coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)

                            row_indices.append(i)
                            col_indices.append(i)
                            data_values.append(val_A_ii)

                            # Coupling to m[N-2]: diffusion + advection
                            val_A_i_im1 = -(sigma_i**2) / Dx**2
                            if Nx > 1:
                                val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)

                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                        else:
                            # Interior points: standard conservative FDM discretization
                            val_A_ii = 1.0 / Dt + sigma_i**2 / Dx**2

                            val_A_ii += float(
                                coupling_coefficient
                                * (npart(u_at_tk[i + 1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[i - 1]))
                                / Dx**2
                            )

                            row_indices.append(i)
                            col_indices.append(i)
                            data_values.append(val_A_ii)

                            # Lower diagonal term
                            val_A_i_im1 = -(sigma_i**2) / (2 * Dx**2)
                            val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                            # Upper diagonal term
                            val_A_i_ip1 = -(sigma_i**2) / (2 * Dx**2)
                            val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

            A_matrix = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(Nx, Nx)).tocsr()

            # Set up right-hand side
            b_rhs = m[k_idx_fp, :] / Dt

            # Apply boundary conditions to RHS
            if bc_type == "dirichlet":
                b_rhs[0] = _get_bc_value(self.boundary_conditions, "x_min")
                b_rhs[-1] = _get_bc_value(self.boundary_conditions, "x_max")
            elif bc_type == "no_flux":
                # For no-flux boundaries, RHS remains as m[k]/Dt
                # The no-flux condition is enforced through the matrix coefficients
                pass

            if self.backend is not None:
                m_next_step_raw = self.backend.zeros((Nx,))
            else:
                m_next_step_raw = np.zeros(Nx, dtype=np.float64)

            if not A_matrix.nnz > 0 and Nx > 0:
                m_next_step_raw[:] = m[k_idx_fp, :]
            else:
                solution = sparse.linalg.spsolve(A_matrix, b_rhs)
                m_next_step_raw[:] = solution

            if has_nan_or_inf(m_next_step_raw, self.backend):
                raise ValueError(f"Fokker-Planck solver produced NaNs at step {k_idx_fp}")

            m[k_idx_fp + 1, :] = m_next_step_raw

            # Ensure boundary conditions are satisfied
            if bc_type == "dirichlet":
                m[k_idx_fp + 1, 0] = _get_bc_value(self.boundary_conditions, "x_min")
                m[k_idx_fp + 1, -1] = _get_bc_value(self.boundary_conditions, "x_max")

            # Issue #880: Enforce non-negativity with diagnostic warning
            min_val = np.min(m[k_idx_fp + 1, :])
            if min_val < 0:
                if min_val < -1e-10:
                    from mfgarchon.utils.mfg_logging import get_logger

                    get_logger(__name__).warning(
                        "FP solver: negative density clipped at timestep %d (min=%.2e)",
                        k_idx_fp + 1,
                        min_val,
                    )
                m[k_idx_fp + 1, :] = np.maximum(m[k_idx_fp + 1, :], 0)

            # Issue #640: Report progress to hierarchical progress bar
            if progress_callback is not None:
                progress_callback(1)

        return m

    def _validate_callable_output(
        self,
        output: np.ndarray | float,
        expected_shape: tuple,
        param_name: str,
        timestep: int | None = None,
    ) -> np.ndarray:
        """
        Validate callable coefficient output.

        Parameters
        ----------
        output : np.ndarray or float
            Output from callable (diffusion or drift)
        expected_shape : tuple
            Expected shape for spatial array
        param_name : str
            Parameter name for error messages
        timestep : int, optional
            Current timestep (for error messages)

        Returns
        -------
        np.ndarray
            Validated array (converts scalar to array if needed)

        Raises
        ------
        ValueError
            If output shape is incorrect or contains NaN/Inf
        TypeError
            If output type is incorrect
        """
        # Convert scalar to array
        if isinstance(output, (int, float)):
            output = np.full(expected_shape, float(output))
        elif isinstance(output, np.ndarray):
            # Validate shape
            if output.shape != expected_shape:
                raise ValueError(
                    f"{param_name} callable returned array with shape {output.shape}, "
                    f"expected {expected_shape} at timestep {timestep}"
                )
        else:
            raise TypeError(
                f"{param_name} callable must return float or np.ndarray, got {type(output)} at timestep {timestep}"
            )

        # Check for NaN/Inf
        if has_nan_or_inf(output, self.backend):
            raise ValueError(f"{param_name} callable returned NaN or Inf at timestep {timestep}")

        return output

    @deprecated(since="v0.17.1", replacement="solve_fp_system")
    def _solve_fp_1d_with_callable(
        self,
        m_initial_condition: np.ndarray,
        drift_field: np.ndarray | None,
        diffusion_callable: callable,
        show_progress: bool | None = None,
    ) -> np.ndarray:
        """
        Solve 1D FP equation with callable (state-dependent) diffusion.

        .. deprecated:: 0.17.1
            This method is deprecated in favor of `solve_fp_nd_full_system` which
            handles callable diffusion for all dimensions. Will be removed in v1.0.0.

        Uses bootstrap strategy: evaluate callable at each timestep using
        the already-computed density m[k] to solve for m[k+1].

        Parameters
        ----------
        m_initial_condition : np.ndarray
            Initial density, shape (Nx,)
        drift_field : np.ndarray or None
            Precomputed drift field, shape (Nt, Nx), or None for zero drift
        diffusion_callable : callable
            Function D(t, x, m) -> diffusion coefficient
            Signature: (float, np.ndarray, np.ndarray) -> float | np.ndarray
        show_progress : bool
            Show progress bar

        Returns
        -------
        np.ndarray
            Density evolution, shape (Nt, Nx)
        """
        from mfgarchon.types.pde_coefficients import DiffusionCallable

        # Validate callable signature using protocol
        if not isinstance(diffusion_callable, DiffusionCallable):
            raise TypeError(
                "diffusion_field callable does not match DiffusionCallable protocol. "
                "Expected signature: (t: float, x: ndarray, m: ndarray) -> float | ndarray"
            )

        # Get problem dimensions from geometry
        Nx = self.problem.geometry.get_grid_shape()[0]
        Dt = self.problem.dt
        bounds = self.problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]

        # Infer Nt from drift_field if provided, else use problem.Nt
        if drift_field is not None:
            Nt = drift_field.shape[0]
        else:
            Nt = self.problem.Nt + 1

        # Create spatial grid for callable evaluation
        x_grid = np.linspace(xmin, xmax, Nx)

        # Allocate solution array
        if self.backend is not None:
            m_solution = self.backend.zeros((Nt, Nx))
        else:
            m_solution = np.zeros((Nt, Nx))

        m_solution[0, :] = m_initial_condition
        m_solution[0, :] = np.maximum(m_solution[0, :], 0)

        # Apply boundary conditions to initial condition
        if _get_bc_type(self.boundary_conditions) == "dirichlet":
            m_solution[0, 0] = _get_bc_value(self.boundary_conditions, "x_min")
            m_solution[0, -1] = _get_bc_value(self.boundary_conditions, "x_max")

        # Progress bar for forward timesteps with callable diffusion
        # n_time_points - 1 steps to go from t=0 to t=T
        from mfgarchon.utils.progress import create_progress_bar, should_show_progress

        timestep_range = create_progress_bar(
            range(Nt - 1),
            verbose=should_show_progress(show_progress),
            desc="FP (callable diffusion)",
        )

        # Bootstrap forward iteration: use m[k] to evaluate callable and compute m[k+1]
        for k in timestep_range:
            t_current = k * Dt
            m_current = m_solution[k, :]

            # Evaluate diffusion callable at current state
            diffusion_at_k = diffusion_callable(t_current, x_grid, m_current)

            # Validate callable output
            diffusion_at_k = self._validate_callable_output(
                diffusion_at_k,
                expected_shape=(Nx,),
                param_name="diffusion_field",
                timestep=k,
            )

            # Temporarily set sigma to evaluated diffusion for this timestep
            original_sigma = self.problem.sigma
            self.problem.sigma = diffusion_at_k

            try:
                # Get drift at current timestep (or zero)
                if drift_field is not None:
                    U_at_k = drift_field[k, :]
                else:
                    U_at_k = np.zeros(Nx)

                # Solve single timestep using _solve_fp_1d machinery
                # Create temporary arrays for single-step solve
                m_temp = np.zeros((2, Nx))
                m_temp[0, :] = m_current
                U_temp = np.zeros((2, Nx))
                U_temp[0, :] = U_at_k

                # Call _solve_fp_1d for single timestep (Nt=2 gives one step)
                # This reuses all the boundary condition logic
                m_result = self._solve_fp_1d(
                    m_initial_condition=m_current,
                    U_solution_for_drift=U_temp,
                    show_progress=False,
                )

                # Extract result at next timestep
                m_solution[k + 1, :] = m_result[1, :]

            finally:
                # Restore original sigma
                self.problem.sigma = original_sigma

        return m_solution

    # NOTE: _solve_fp_1d_with_callable_drift was removed in v0.17.1 (Issue #641)
    # It was dead code - never called from solve_fp_system().
    # Callable drift now routes directly to _solve_fp_nd_full_system() which
    # handles all dimensions including 1D.


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPFDMSolver...")
    print("=" * 60)

    from mfgarchon import MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    # Test 1D problem using geometry-based API (unified nD solver, Issue #641)
    print("\n1. Testing 1D FDM (advection schemes)...")

    # Create 1D grid with TensorProductGrid
    Nx = 40  # Number of cells (grid points = Nx + 1)
    grid_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx + 1])
    dx_1d = grid_1d.get_grid_spacing()[0]

    problem_1d = MFGProblem(
        geometry=grid_1d,
        Nt=25,
        T=1.0,
        sigma=0.1,
        coupling_coefficient=1.0,
    )

    # Create initial density (Gaussian) and drift field
    x = np.array(grid_1d.coordinates[0])  # 1D coordinates
    m_init_1d = np.exp(-((x - 0.5) ** 2) / 0.05)
    m_init_1d /= m_init_1d.sum() * dx_1d  # Normalize using sum*dx (consistent with 2D)

    # Create drift pushing mass to right (advection test)
    Nt = problem_1d.Nt + 1
    U_test_1d = np.zeros((Nt, Nx + 1))
    for t in range(Nt):
        U_test_1d[t] = -x  # Drift to the right (alpha = -dU/dx = +1)

    # Test gradient_upwind (gradient form: v·∇m)
    solver_1d_gu = FPFDMSolver(
        problem_1d,
        boundary_conditions=no_flux_bc(dimension=1),
        advection_scheme="gradient_upwind",
    )
    assert solver_1d_gu.dimension == 1
    assert solver_1d_gu.fp_method_name == "FDM"
    assert solver_1d_gu.advection_scheme == "gradient_upwind"

    M_1d_gu = solver_1d_gu.solve_fp_system(m_init_1d, U_test_1d, show_progress=False)
    assert M_1d_gu.shape == (Nt, Nx + 1)
    assert not has_nan_or_inf(M_1d_gu)

    # Test divergence_upwind (divergence form: ∇·(vm))
    solver_1d_du = FPFDMSolver(
        problem_1d,
        boundary_conditions=no_flux_bc(dimension=1),
        advection_scheme="divergence_upwind",
    )
    assert solver_1d_du.advection_scheme == "divergence_upwind"

    M_1d_du = solver_1d_du.solve_fp_system(m_init_1d, U_test_1d, show_progress=False)
    assert M_1d_du.shape == (Nt, Nx + 1)
    assert not has_nan_or_inf(M_1d_du)

    # Calculate mass drift for both (using sum*dx for consistency with 2D)
    initial_mass_1d = m_init_1d.sum() * dx_1d
    final_mass_1d_gu = M_1d_gu[-1].sum() * dx_1d
    final_mass_1d_du = M_1d_du[-1].sum() * dx_1d
    mass_drift_1d_gu = abs(final_mass_1d_gu - initial_mass_1d) / initial_mass_1d
    mass_drift_1d_du = abs(final_mass_1d_du - initial_mass_1d) / initial_mass_1d

    print(f"   Initial mass: {initial_mass_1d:.6f}")
    print(f"   gradient_upwind:   final={final_mass_1d_gu:.6f}, drift={mass_drift_1d_gu:.2%}")
    print(f"   divergence_upwind: final={final_mass_1d_du:.6f}, drift={mass_drift_1d_du:.2%}")

    # Test 2D problem with advection schemes
    print("\n2. Testing 2D FDM (advection schemes)...")

    # Create 2D problem
    grid_2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],  # [(xmin, xmax), (ymin, ymax)]
        Nx_points=[11, 11],  # (nx+1, ny+1) grid points
    )
    problem_2d = MFGProblem(
        geometry=grid_2d,
        Nt=20,
        T=0.5,
        sigma=0.2,
        coupling_coefficient=1.0,
    )

    # Create Gaussian initial density
    x, y = grid_2d.coordinates
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx, dy = grid_2d.get_grid_spacing()
    cell_volume = dx * dy
    m_init_2d = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.05)
    m_init_2d /= m_init_2d.sum() * cell_volume

    # Create a drift field that pushes mass to corner (advection test)
    Nt = problem_2d.Nt + 1
    U_drift = np.zeros((Nt, *grid_2d.get_grid_shape()))
    # Potential U = -x - y (drift to upper-right corner)
    for t in range(Nt):
        U_drift[t] = -(X + Y)

    # Test gradient_upwind (non-conservative)
    solver_2d_gu = FPFDMSolver(
        problem_2d,
        boundary_conditions=no_flux_bc(dimension=2),
        advection_scheme="gradient_upwind",
    )
    M_2d_gu = solver_2d_gu.solve_fp_system(m_init_2d, U_drift, show_progress=False)

    # Test divergence_upwind (conservative)
    solver_2d_du = FPFDMSolver(
        problem_2d,
        boundary_conditions=no_flux_bc(dimension=2),
        advection_scheme="divergence_upwind",
    )
    M_2d_du = solver_2d_du.solve_fp_system(m_init_2d, U_drift, show_progress=False)

    # Calculate mass drift for both
    initial_mass_2d = m_init_2d.sum() * cell_volume

    final_mass_gu = M_2d_gu[-1].sum() * cell_volume
    mass_drift_gu = abs(final_mass_gu - initial_mass_2d) / initial_mass_2d

    final_mass_du = M_2d_du[-1].sum() * cell_volume
    mass_drift_du = abs(final_mass_du - initial_mass_2d) / initial_mass_2d

    print(f"   Initial mass: {initial_mass_2d:.6f}")
    print(f"   gradient_upwind:   final={final_mass_gu:.6f}, drift={mass_drift_gu:.2%}")
    print(f"   divergence_upwind: final={final_mass_du:.6f}, drift={mass_drift_du:.2%}")

    # Verify solutions are valid
    assert not has_nan_or_inf(M_2d_gu), "gradient_upwind solution has NaN/Inf"
    assert not has_nan_or_inf(M_2d_du), "divergence_upwind solution has NaN/Inf"
    assert np.all(M_2d_gu >= -1e-10), "gradient_upwind: density should be non-negative"
    assert np.all(M_2d_du >= -1e-10), "divergence_upwind: density should be non-negative"

    # Verify advection_scheme is properly set
    assert solver_2d_gu.advection_scheme == "gradient_upwind"
    assert solver_2d_du.advection_scheme == "divergence_upwind"

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
