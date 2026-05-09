from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

# Import MFGComponents and mixins from the dedicated module
from mfgarchon.core.hamiltonian import HamiltonianBase
from mfgarchon.core.mfg_components import (
    ConditionsMixin,
    HamiltonianMixin,
    MFGComponents,
)

# Issue #543: Runtime import for isinstance() checks
from mfgarchon.geometry.protocol import GeometryProtocol  # noqa: TC001

# Deprecation utilities (Issue #616, #666)
from mfgarchon.utils.deprecation import deprecated, deprecated_parameter, validate_kwargs

# Use unified nD-capable BoundaryConditions from conditions.py

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.types.pde_coefficients import DiffusionField, DriftField


# ============================================================================
# Diffusion / Volatility Conversion Helpers (Issue #811)
# ============================================================================
#
# Physics/PDE convention: D = sigma^2/2 = (1/2)*Sigma*Sigma^T
# SDE convention: dX = alpha*dt + Sigma*dW
#
# diffusion= parameter: D (PDE coefficient, appears directly in PDE)
# sigma= parameter: Sigma (SDE volatility, what solvers use internally)


def _diffusion_to_volatility(
    D: float | int | np.ndarray,
) -> float | np.ndarray:
    """Convert PDE diffusion coefficient D to SDE volatility sigma.

    Scalar:   D = sigma^2/2  =>  sigma = sqrt(2D)
    Diagonal: D_i = sigma_i^2/2  =>  sigma_i = sqrt(2 D_i)
    Tensor:   D = (1/2) Sigma Sigma^T  =>  Sigma = cholesky(2D)

    Args:
        D: PDE diffusion coefficient (non-negative scalar, 1D diagonal, or 2D SPD tensor).

    Returns:
        SDE volatility sigma (same shape semantics as input).

    Raises:
        ValueError: If D is negative (scalar) or has unsupported shape.
    """
    if isinstance(D, (int, float)):
        D_f = float(D)
        if D_f < 0:
            raise ValueError(f"Diffusion coefficient must be non-negative, got {D_f}")
        return math.sqrt(2.0 * D_f)

    D_arr = np.asarray(D, dtype=float)
    if D_arr.ndim == 0:
        # 0-d numpy array
        val = float(D_arr)
        if val < 0:
            raise ValueError(f"Diffusion coefficient must be non-negative, got {val}")
        return math.sqrt(2.0 * val)
    if D_arr.ndim == 1:
        # Diagonal: element-wise
        if np.any(D_arr < 0):
            raise ValueError("Diffusion coefficient array must be non-negative")
        return np.sqrt(2.0 * D_arr)
    if D_arr.ndim == 2:
        # Full tensor: D = (1/2) Sigma Sigma^T => Sigma = cholesky(2D)
        return np.linalg.cholesky(2.0 * D_arr)

    raise ValueError(f"Unsupported diffusion shape: {D_arr.shape}")


def _volatility_to_diffusion(
    sigma: float | int | np.ndarray,
) -> float | np.ndarray:
    """Convert SDE volatility sigma to PDE diffusion coefficient D.

    Scalar:   sigma => D = sigma^2/2
    Diagonal: sigma_i => D_i = sigma_i^2/2
    Tensor:   Sigma => D = (1/2) Sigma Sigma^T

    Args:
        sigma: SDE volatility (scalar, 1D diagonal, or 2D matrix).

    Returns:
        PDE diffusion coefficient D.
    """
    if isinstance(sigma, (int, float)):
        return float(sigma) ** 2 / 2.0

    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        return float(sigma_arr) ** 2 / 2.0
    if sigma_arr.ndim == 1:
        return sigma_arr**2 / 2.0
    if sigma_arr.ndim == 2:
        # Full matrix: D = (1/2) Sigma Sigma^T
        return 0.5 * (sigma_arr @ sigma_arr.T)

    raise ValueError(f"Unsupported volatility shape: {sigma_arr.shape}")


# ============================================================================
# Unified MFG Problem Class
# ============================================================================


class MFGProblem(HamiltonianMixin, ConditionsMixin):
    """
    Unified MFG problem class that can handle both predefined and custom formulations.

    This class serves as the single constructor for all MFG problems:
    - Default usage: Uses built-in Hamiltonian (standard MFG formulation)
    - Custom usage: Accepts MFGComponents for full mathematical control

    Inherits from two mixins:

    HamiltonianMixin (mathematical Hamiltonian):
    - H(): Hamiltonian function
    - dH_dm(): Hamiltonian derivative w.r.t. density
    - get_hjb_hamiltonian_jacobian_contrib(): Jacobian for Newton methods
    - get_hjb_residual_m_coupling_term(): Coupling terms
    - get_potential_at_time(): Time-dependent potential accessor

    ConditionsMixin (problem setup):
    - get_boundary_conditions(): Boundary condition accessor
    - _setup_custom_initial_density(): Initial density setup
    - _setup_custom_final_value(): Final value setup
    """

    # Type annotations for geometry attributes (Phase 6 of Issue #435)
    # These are always non-None after __init__ completes
    geometry: GeometryProtocol
    hjb_geometry: GeometryProtocol | None
    fp_geometry: GeometryProtocol | None

    # Type annotations for PDE coefficient fields (Issue #811)
    # sigma: SDE volatility (scalar, used by all solvers as problem.sigma)
    # volatility_field: Full volatility field — stores SDE volatility (sigma), NOT PDE D
    #   (all solvers expect sigma, not D)
    # drift_field: Optional drift (float, array, or callable)
    sigma: float
    volatility_field: DiffusionField
    drift_field: DriftField

    @staticmethod
    def _normalize_to_array(
        value: int | float | list[int] | list[float] | None,
        param_name: str = "parameter",
        warn: bool = True,
    ) -> list[int] | list[float] | None:
        """
        Convert scalar or array to array with optional deprecation warning.

        Args:
            value: Scalar or array value to normalize
            param_name: Parameter name for warning message
            warn: Whether to emit deprecation warning for scalar inputs

        Returns:
            Array form of the value, or None if input is None

        Examples:
            >>> MFGProblem._normalize_to_array(100, "Nx")  # Warns
            [100]
            >>> MFGProblem._normalize_to_array([100], "Nx")  # No warning
            [100]
            >>> MFGProblem._normalize_to_array(None, "Nx")
            None
        """
        import warnings

        if value is None:
            return None

        if isinstance(value, (int, float)):
            if warn:
                warnings.warn(
                    f"Passing scalar {param_name}={value} is deprecated. "
                    f"Use array notation {param_name}=[{value}] instead. "
                    f"Scalar support will be removed in v1.0.0. "
                    f"See docs/development/MATHEMATICAL_NOTATION_STANDARD.md for details.",
                    DeprecationWarning,
                    stacklevel=4,
                )
            return [value]

        # Already a list - return as-is
        return list(value)

    @deprecated_parameter(
        param_name="Lx",
        since="v0.17.1",
        replacement="geometry=TensorProductGrid(...)",
        removal_blockers=["internal_usage", "migration_docs"],
    )
    @deprecated_parameter(
        param_name="Nx",
        since="v0.17.1",
        replacement="geometry=TensorProductGrid(...)",
        removal_blockers=["internal_usage", "migration_docs"],
    )
    @deprecated_parameter(
        param_name="xmax",
        since="v0.17.1",
        replacement="geometry=TensorProductGrid(...)",
        removal_blockers=["internal_usage", "migration_docs"],
    )
    @deprecated_parameter(
        param_name="xmin",
        since="v0.17.1",
        replacement="geometry=TensorProductGrid(...)",
        removal_blockers=["internal_usage", "migration_docs"],
    )
    def __init__(
        self,
        # === API v1.0 parameters (Issue #875) ===
        model: Any | None = None,  # Model instance (game rules)
        domain: GeometryProtocol | None = None,  # Spatial geometry
        conditions: Any | None = None,  # Conditions instance (time + IC/TC)
        constraints: list | None = None,  # Optional constraints (future)
        # === Legacy parameters (all below deprecated in favor of model/domain/conditions) ===
        # Legacy 1D parameters (backward compatible - scalars will be converted to arrays with deprecation warning)
        xmin: float | list[float] | None = None,
        xmax: float | list[float] | None = None,
        Nx: int | list[int] | None = None,
        Lx: float | None = None,  # Alternative to xmin/xmax
        # N-D grid parameters
        spatial_bounds: list[tuple[float, float]] | None = None,
        spatial_discretization: list[int] | None = None,
        # Complex geometry parameters (NEW)
        geometry: GeometryProtocol | None = None,
        obstacles: list | None = None,
        # Dual geometry parameters (Issue #257)
        hjb_geometry: GeometryProtocol | None = None,
        fp_geometry: GeometryProtocol | None = None,
        # Network parameters (NEW)
        network: Any | None = None,  # NetworkGraph
        # Time domain parameters
        T: float | None = None,
        Nt: int | None = None,
        time_domain: tuple[float, int] | None = None,  # Alternative to T/Nt
        # Physical parameters — Issue #811 convention: diffusion = D = sigma^2/2
        diffusion: float | NDArray[np.floating] | Callable | None = None,  # PDE coefficient D = sigma^2/2
        sigma: float | NDArray[np.floating] | Callable | None = None,  # SDE volatility sigma
        volatility: float | NDArray[np.floating] | Callable | None = None,  # Alias for sigma
        drift: float | NDArray[np.floating] | Callable | None = None,  # Optional drift field
        coupling_coefficient: float = 0.5,
        # MFG coupling parameters
        lambda_: float | None = None,  # Control cost (H uses |p|²/(2λ))
        gamma: float = 1.0,  # Density coupling strength (H uses -γm²)
        # Class-based Hamiltonian (Issue #673 - recommended)
        hamiltonian: Any | None = None,  # HamiltonianBase instance
        # Advanced
        components: MFGComponents | None = None,
        suppress_warnings: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MFG problem with support for all spatial dimensions and domain types.

        Supports five initialization modes:
        1. Legacy 1D mode: Specify Nx, xmin, xmax
        2. N-D grid mode: Specify spatial_bounds, spatial_discretization
        3. Geometry mode: Specify geometry object (with optional obstacles)
        4. Network mode: Specify network graph
        5. Custom components: Full mathematical control via MFGComponents

        Args:
            xmin, xmax, Nx, Lx: Legacy 1D spatial domain parameters
            spatial_bounds: List of (min, max) tuples for each dimension
                           Example: [(0, 1), (0, 1)] for 2D unit square
            spatial_discretization: List of grid points per dimension
                                   Example: [50, 50] for 51×51 grid
            geometry: BaseGeometry object for complex domains (unified mode)
            obstacles: List of obstacle geometries
            hjb_geometry: Geometry for HJB solver (dual geometry mode, Issue #257)
            fp_geometry: Geometry for FP solver (dual geometry mode, Issue #257)
                        Note: Both hjb_geometry and fp_geometry must be specified together
            network: NetworkGraph for network MFG problems
            T, Nt, time_domain: Time domain parameters (T, Nt) or tuple (T, Nt)
            diffusion: PDE diffusion coefficient D = sigma^2/2 (Issue #811).
                None -> 0 (deterministic). Internally converted to SDE volatility
                sigma = sqrt(2D) for solver consumption.
                Supports:
                - None: No diffusion (deterministic dynamics)
                - float: Constant isotropic D
                - ndarray: Spatially varying D (element-wise conversion)
                - Callable: State-dependent D(t, x, m) (wrapped with conversion)
                Mutually exclusive with sigma= and volatility=.
            sigma: SDE volatility sigma. Mutually exclusive with diffusion=.
                Direct specification of noise coefficient in dX = alpha dt + sigma dW.
                Supports same types as diffusion= (no conversion applied).
            volatility: Alias for sigma (SDE volatility). Same semantics.
            drift: Drift field α(t, x, m) for FP equation. None → 0 (no drift).
                Supports:
                - None: No drift (no advection)
                - float: Constant drift (same in all directions)
                - ndarray: Precomputed drift array
                - Callable: State-dependent α(t, x, m) -> float | ndarray
            coupling_coefficient: Control cost coefficient
            components: Optional MFGComponents for custom problem definition
            suppress_warnings: Suppress computational feasibility warnings
            **kwargs: Additional parameters

        Examples:
            # Mode 1: 1D legacy (100% backward compatible)
            problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, Nt=100)

            # Mode 2: N-D grid
            problem = MFGProblem(
                spatial_bounds=[(0, 1), (0, 1)],
                spatial_discretization=[50, 50],
                Nt=50
            )

            # Mode 3: Complex geometry with obstacles
            from mfgarchon.geometry import Hyperrectangle, Hypersphere
            domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
            obstacle = Hypersphere(center=[0.5, 0.5], radius=0.1)
            problem = MFGProblem(
                geometry=domain,
                obstacles=[obstacle],
                time_domain=(1.0, 50),
                diffusion=0.1
            )

            # Mode 4: Network MFG
            import networkx as nx
            graph = nx.grid_2d_graph(10, 10)
            problem = MFGProblem(network=graph, time_domain=(1.0, 100))

            # Mode 5: Custom components
            components = MFGComponents(hamiltonian_func=..., ...)
            problem = MFGProblem(
                spatial_bounds=[(0, 1)],
                spatial_discretization=[100],
                Nt=50,
                components=components
            )

            # Mode 6: Dual geometry (Issue #257) - Separate geometries for HJB and FP
            from mfgarchon.geometry import TensorProductGrid
            from mfgarchon.geometry.boundary import no_flux_bc
            hjb_grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[51, 51], boundary_conditions=no_flux_bc(2))
            fp_grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[21, 21], boundary_conditions=no_flux_bc(2))
            problem = MFGProblem(
                hjb_geometry=hjb_grid,
                fp_geometry=fp_grid,
                time_domain=(1.0, 50),
                diffusion=0.1
            )
            # Automatically creates geometry_projector for mapping between geometries

            # Advanced: State-dependent diffusion (callable)
            def density_dependent_diffusion(t, x, m):
                return 0.1 * (1 + m)  # Higher diffusion in dense regions
            problem = MFGProblem(
                geometry=domain,
                sigma=density_dependent_diffusion,
                time_domain=(1.0, 50)
            )

            # Advanced: Spatially varying diffusion (array)
            sigma_array = np.ones((51, 51)) * 0.1  # Base diffusion
            sigma_array[20:30, 20:30] = 0.5  # Higher diffusion in center region
            problem = MFGProblem(
                geometry=domain,
                sigma=sigma_array,
                time_domain=(1.0, 50)
            )

            # Advanced: Custom drift field
            def crowd_avoidance_drift(t, x, m):
                grad_m = np.gradient(m)  # Density gradient
                return -np.stack(grad_m, axis=-1)  # Move down gradient
            problem = MFGProblem(
                geometry=domain,
                drift=crowd_avoidance_drift,
                time_domain=(1.0, 50)
            )
        """
        import warnings

        # =====================================================================
        # API v1.0 path (Issue #875): Model + Domain + Conditions
        # =====================================================================
        from mfgarchon.core.model import Conditions as _Conditions
        from mfgarchon.core.model import Model as _Model

        _new_api = model is not None or domain is not None or conditions is not None
        if _new_api:
            # Validate all three are provided
            if model is None or domain is None or conditions is None:
                _missing = [
                    n for n, v in [("model", model), ("domain", domain), ("conditions", conditions)] if v is None
                ]
                raise ValueError(
                    f"API v1.0 requires all three: model, domain, conditions. Missing: {', '.join(_missing)}"
                )
            if not isinstance(model, _Model):
                raise TypeError(f"model must be a Model instance, got {type(model).__name__}")
            if not isinstance(conditions, _Conditions):
                raise TypeError(f"conditions must be a Conditions instance, got {type(conditions).__name__}")

            # Check for conflicting legacy parameters
            # Note: Nt is allowed with v1.0 API — sets construction default
            # (can be overridden at solve() time)
            _legacy_params = {
                n: v
                for n, v in [
                    ("geometry", geometry),
                    ("xmin", xmin),
                    ("xmax", xmax),
                    ("Nx", Nx),
                    ("spatial_bounds", spatial_bounds),
                    ("spatial_discretization", spatial_discretization),
                    ("sigma", sigma),
                    ("diffusion", diffusion),
                    ("volatility", volatility),
                    ("T", T),
                    ("time_domain", time_domain),
                    ("components", components),
                    ("hamiltonian", hamiltonian),
                ]
                if v is not None
            }
            if _legacy_params:
                raise ValueError(
                    f"Cannot mix API v1.0 (model/domain/conditions) with legacy parameters. "
                    f"Got legacy parameters: {', '.join(_legacy_params.keys())}"
                )

            # Translate to legacy parameters
            geometry = domain
            sigma = model.sigma
            T = conditions.T
            # Nt must be provided — fail fast (no silent defaults)
            if Nt is None:
                raise ValueError(
                    "Nt (time discretization) is required. "
                    "Pass Nt= to MFGProblem() or use problem.solve(Nt=...).\n"
                    "  MFGProblem(model=..., domain=..., conditions=..., Nt=50)"
                )
            # Build MFGComponents from Model + Conditions
            _h = (
                model.effective_hamiltonian if (model.hamiltonian is not None or model.lagrangian is not None) else None
            )
            components = MFGComponents(
                hamiltonian=_h,
                u_terminal=conditions.u_terminal,
                m_initial=conditions.m_initial,
            )
            # Drift-only models: pass drift through
            if model.drift_field is not None:
                drift = model.drift_field
            # Store v1.0 objects for solve() and with_*() methods
            self._v1_model = model
            self._v1_conditions = conditions
            self._v1_constraints = constraints
        else:
            self._v1_model = None
            self._v1_conditions = None
            self._v1_constraints = None

            # Issue #875: Deprecation warning for legacy API path
            warnings.warn(
                "Legacy MFGProblem(geometry=, components=, sigma=, T=, Nt=) is deprecated. "
                "Use the v1.0 API: MFGProblem(model=Model(...), domain=grid, "
                "conditions=Conditions(...), Nt=50). "
                "Legacy support will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Normalize parameter aliases
        if time_domain is not None:
            if T is not None or Nt is not None:
                raise ValueError("Specify EITHER (T, Nt) OR time_domain, not both")
            T, Nt = time_domain

        # --- Diffusion / sigma / volatility: mutual exclusion (Issue #811) ---
        # Convention: diffusion = D = sigma^2/2 (PDE coefficient)
        #             sigma = SDE volatility (what solvers use internally)
        #             volatility = alias for sigma
        _n_phys = sum(x is not None for x in [diffusion, sigma, volatility])
        if _n_phys > 1:
            _names = [
                n for n, v in [("diffusion", diffusion), ("sigma", sigma), ("volatility", volatility)] if v is not None
            ]
            raise ValueError(f"Specify at most one of: diffusion=, sigma=, volatility=. Got: {', '.join(_names)}")

        # Resolve volatility alias
        if volatility is not None:
            sigma = volatility

        # Convert to SDE volatility (the internal representation used by all solvers).
        # After this block, `vola_value` holds the SDE volatility field.
        if sigma is not None:
            # User provided SDE volatility directly — no conversion needed
            vola_value = sigma
        elif diffusion is not None:
            # User provided PDE coefficient D = sigma^2/2.
            # Convert to SDE volatility: sigma = sqrt(2D).
            if callable(diffusion):
                _D_callable = diffusion

                def vola_value(t, x, m, *, _D=_D_callable):  # type: ignore[misc]
                    return _diffusion_to_volatility(_D(t, x, m))
            else:
                vola_value = _diffusion_to_volatility(diffusion)
        else:
            # No physical parameter specified — deterministic (sigma = 0)
            vola_value = 0.0

        # Set defaults for T, Nt if not provided
        if T is None:
            T = 1.0
        if Nt is None:
            Nt = 51

        # drift default
        if drift is None:
            drift = 0.0

        # Store the full volatility field for advanced solvers.
        # Note (Issue #811): volatility_field stores SDE volatility (sigma),
        # not PDE diffusion D. All solvers expect sigma and compute D = sigma^2/2
        # internally.
        # Note (Issue #1085): mfgarchon SDE convention is **Itô**, not Stratonovich.
        # For constant sigma, the two coincide. For callable sigma(t, x, m) with
        # spatial dependence, users with Stratonovich-derived drift must apply
        # the correction `alpha_Ito = alpha_Strat - (1/2) sigma * d_x sigma`
        # before passing the drift. mfgarchon does NOT add this correction.
        self.volatility_field = vola_value
        self.drift_field = drift

        # Extended PDE form fields (Issue #921).
        # These enable generalized MFG equations beyond the classical form:
        #   HJB: -du/dt + H(x,m,Du) - S_hjb = 0
        #   FP:  dm/dt - (sigma^2/2)Dm - div(m*alpha*) - S_fp = 0
        # source_term_hjb/fp: Callable(x, m, v, t) -> array (problem-level signature)
        # nonlocal_operator: LinearOperator for integro-differential terms J[v]
        # obstacle: Callable(x) -> array for variational inequality v >= Psi(x)
        self.source_term_hjb: Callable | None = kwargs.pop("source_term_hjb", None)
        self.source_term_fp: Callable | None = kwargs.pop("source_term_fp", None)
        self.nonlocal_operator: Any | None = kwargs.pop("nonlocal_operator", None)
        self.obstacle: Callable | None = kwargs.pop("obstacle", None)

        # Extract scalar sigma for backward compatibility.
        # If volatility is callable or array, use a representative scalar value.
        if callable(vola_value):
            # Callable: store 1.0 as default, solvers should use volatility_field
            sigma_scalar = 1.0
        elif isinstance(vola_value, np.ndarray):
            # Array: use mean value as representative scalar
            sigma_scalar = float(np.mean(vola_value))
        else:
            # Scalar: use directly
            sigma_scalar = float(vola_value)

        # Normalize spatial parameters to arrays (with deprecation warnings for scalars)
        # This enables dimension-agnostic code while maintaining backward compatibility

        # Issue #544: Deprecate legacy 1D parameters (Nx, xmin, xmax, Lx)
        # Note: Deprecation warnings issued by @deprecated_parameter decorators
        # Detailed migration guide: docs/user/LEGACY_PARAMETERS.md

        if Nx is not None:
            Nx_normalized = self._normalize_to_array(Nx, "Nx")
        else:
            Nx_normalized = None

        if xmin is not None:
            xmin_normalized = self._normalize_to_array(xmin, "xmin")
        else:
            xmin_normalized = None

        if xmax is not None:
            xmax_normalized = self._normalize_to_array(xmax, "xmax")
        else:
            xmax_normalized = None

        # Initialize geometry-related attributes explicitly (Issue #543 - fail-fast principle)
        # These may be set by init methods, but should have explicit defaults
        self.geometry = None  # type: GeometryProtocol | None
        self.hjb_geometry = None  # type: GeometryProtocol | None
        self.fp_geometry = None  # type: GeometryProtocol | None
        self.spatial_shape = None  # type: tuple[int, ...] | None
        self.has_obstacles = False
        self.obstacles = []
        self.geometry_projector = None  # Will be set if dual geometries provided
        self.solver_compatible = {}  # type: dict[str, bool]
        self.solver_recommendations = {}  # type: dict[str, str]

        # Initialize legacy override attributes (Issue #543 - Step 2)
        # These support deprecated parameter API and will be removed in #544
        self._xmin_override = None
        self._xmax_override = None
        self._Lx_override = None
        self._Nx_override = None
        self._dx_override = None
        self._xSpace_override = None
        self._grid_override = None

        if hjb_geometry is not None and fp_geometry is not None:
            # Dual geometry mode: separate geometries for HJB and FP
            if geometry is not None:
                raise ValueError(
                    "Specify EITHER 'geometry' (unified) OR ('hjb_geometry', 'fp_geometry') (dual), not both"
                )
            # Use dual geometries
            final_hjb_geometry = hjb_geometry
            final_fp_geometry = fp_geometry
            # Create projector for mapping between geometries
            from mfgarchon.geometry import GeometryProjector

            self.geometry_projector = GeometryProjector(
                hjb_geometry=hjb_geometry, fp_geometry=fp_geometry, projection_method="auto"
            )
        elif hjb_geometry is not None or fp_geometry is not None:
            # Partial dual geometry specification
            raise ValueError("If using dual geometries, both 'hjb_geometry' AND 'fp_geometry' must be specified")
        elif geometry is not None:
            # Unified geometry mode (backward compatible)
            final_hjb_geometry = geometry
            final_fp_geometry = geometry
        else:
            # No explicit geometry provided - will be handled by mode detection
            final_hjb_geometry = None
            final_fp_geometry = None

        # Detect initialization mode (use normalized Nx for detection)
        # For dual geometry, pass the hjb_geometry to mode detection
        geometry_for_detection = final_hjb_geometry if final_hjb_geometry is not None else geometry
        mode = self._detect_init_mode(
            Nx=Nx_normalized, spatial_bounds=spatial_bounds, geometry=geometry_for_detection, network=network
        )

        # Dispatch to appropriate initializer
        # Note: Pass sigma_scalar (the backward-compatible float value)
        if mode == "1d_legacy":
            # Mode 1: Legacy 1D
            if Lx is not None:
                # Use Lx to set xmin/xmax if provided
                if xmin_normalized is None:
                    xmin_normalized = [0.0]
                xmax_normalized = [xmin_normalized[0] + Lx]
            else:
                if xmin_normalized is None:
                    xmin_normalized = [0.0]
                if xmax_normalized is None:
                    xmax_normalized = [1.0]
            self._init_1d_legacy(
                xmin_normalized, xmax_normalized, Nx_normalized, T, Nt, sigma_scalar, coupling_coefficient
            )

        elif mode == "nd_grid":
            # Mode 2: N-dimensional grid
            self._init_nd(
                spatial_bounds, spatial_discretization, T, Nt, sigma_scalar, coupling_coefficient, suppress_warnings
            )

        elif mode == "geometry":
            # Mode 3: Complex geometry
            self._init_geometry(
                final_hjb_geometry,
                obstacles,
                T,
                Nt,
                sigma_scalar,
                coupling_coefficient,
                lambda_,
                gamma,
                suppress_warnings,
            )
            # For dual geometry mode, store both geometries explicitly
            if self.geometry_projector is not None:
                self.hjb_geometry = final_hjb_geometry
                self.fp_geometry = final_fp_geometry

        elif mode == "network":
            # Mode 4: Network MFG
            self._init_network(network, T, Nt, sigma_scalar, coupling_coefficient, lambda_, gamma)

        elif mode == "default":
            # Default: 1D with default parameters
            warnings.warn(
                "No spatial domain specified. Using default 1D domain: [0, 1] with 51 points.",
                UserWarning,
                stacklevel=2,
            )
            self._init_1d_legacy([0.0], [1.0], [51], T, Nt, sigma_scalar, coupling_coefficient)

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        # Store dual geometries (Issue #257)
        # For unified mode, both point to self.geometry (set by init methods)
        # For dual mode, these were already set above (lines 406-407)
        # Issue #543: Explicit None check instead of hasattr
        if self.hjb_geometry is None:
            self.hjb_geometry = getattr(self, "geometry", None)
            self.fp_geometry = getattr(self, "geometry", None)

        # Note: has_obstacles and obstacles already initialized explicitly (lines 334-335)
        # Specialized init methods may override these defaults

        # Issue #673: Handle class-based Hamiltonian parameter
        # If hamiltonian= provided without components, create MFGComponents
        if hamiltonian is not None and components is None:
            if isinstance(hamiltonian, HamiltonianBase):
                components = MFGComponents(hamiltonian=hamiltonian)
            else:
                raise TypeError(
                    f"hamiltonian must be a HamiltonianBase instance, got {type(hamiltonian).__name__}.\n"
                    "Use class-based Hamiltonian:\n"
                    "  from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost\n"
                    "  H = SeparableHamiltonian(control_cost=QuadraticControlCost(...))\n"
                    "  problem = MFGProblem(hamiltonian=H, ...)"
                )

        # Store custom components if provided
        self.components = components
        self.is_custom = components is not None

        # Merge parameters
        if self.is_custom and self.components is not None:
            all_params = {**self.components.parameters, **kwargs}
        else:
            all_params = kwargs

        # Validate kwargs - fail fast on deprecated/unrecognized parameters (Issue #666)
        self._validate_kwargs(all_params)

        # Initialize arrays (Issue #670: unified naming)
        self.f_potential: NDArray
        self.u_terminal: NDArray  # Terminal condition u(T, x)
        self.m_initialial: NDArray  # Initial density m(0, x)

        # Initialize functions
        self._initialize_functions(**all_params)

        # Validate custom components if provided
        if self.is_custom:
            self._validate_hamiltonian_components()

        # Detect solver compatibility
        self._detect_solver_compatibility()

    @deprecated(
        since="v0.17.0",
        replacement="geometry-first API with TensorProductGrid",
        reason="Manual grid construction is deprecated. See docs/migration/GEOMETRY_PARAMETER_MIGRATION.md",
    )
    def _init_1d_legacy(
        self,
        xmin: list[float],
        xmax: list[float],
        Nx: list[int],
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
    ) -> None:
        """
        Initialize problem in legacy 1D mode (100% backward compatible).

        Args:
            xmin: Lower bound as array (e.g., [-2.0])
            xmax: Upper bound as array (e.g., [2.0])
            Nx: Grid points as array (e.g., [100])
            T: Terminal time
            Nt: Temporal grid points
            sigma: Diffusion coefficient
            coupling_coefficient: Control cost coefficient

        Note:
            This manual grid construction pattern is deprecated. Consider using
            the geometry-first API with TensorProductGrid instead.
            See migration guide: docs/migration/GEOMETRY_PARAMETER_MIGRATION.md
        """
        from mfgarchon.geometry import TensorProductGrid

        # Extract scalar values from arrays for backward compatibility
        xmin_scalar = xmin[0]
        xmax_scalar = xmax[0]
        Nx_scalar = Nx[0]

        # Create TensorProductGrid geometry object (unified internal representation)
        # Use default no_flux_bc for legacy _init_1d_legacy path (Issue #674)
        # Note: This is a deprecated code path; users should use geometry-first API
        from mfgarchon.geometry.boundary import no_flux_bc

        geometry = TensorProductGrid(
            bounds=[(xmin_scalar, xmax_scalar)],
            Nx_points=[Nx_scalar + 1],
            boundary_conditions=no_flux_bc(dimension=1),
        )

        # Store geometry for unified interface
        self.geometry = geometry

        # Set dimension from geometry
        self.dimension = geometry.dimension

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.dt: float = T / Nt if Nt > 0 else 0.0

        # Time grid
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.sigma: float = sigma
        self.coupling_coefficient: float = coupling_coefficient

        # N-D attributes (derived from geometry)
        self.spatial_shape = geometry.get_grid_shape()
        self.spatial_bounds = [(xmin_scalar, xmax_scalar)]
        self.spatial_discretization = [Nx_scalar]

    @deprecated(
        since="v0.17.0",
        replacement="geometry-first API with TensorProductGrid",
        reason="Manual grid construction is deprecated. See docs/migration/GEOMETRY_PARAMETER_MIGRATION.md",
    )
    def _init_nd(
        self,
        spatial_bounds: list[tuple[float, float]],
        spatial_discretization: list[int] | None,
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
        suppress_warnings: bool,
    ) -> None:
        """
        Initialize problem in n-dimensional mode.

        Note:
            This manual grid construction pattern is deprecated. Consider using
            the geometry-first API with TensorProductGrid instead.
            See migration guide: docs/migration/GEOMETRY_PARAMETER_MIGRATION.md
        """
        # Validate inputs
        if not spatial_bounds:
            raise ValueError("spatial_bounds must be a non-empty list of (min, max) tuples")

        dimension = len(spatial_bounds)

        if spatial_discretization is None:
            # Default: 51 points per dimension
            spatial_discretization = [51] * dimension
        elif len(spatial_discretization) != dimension:
            raise ValueError(
                f"spatial_discretization must have {dimension} elements (one per dimension), "
                f"got {len(spatial_discretization)}"
            )

        # Create TensorProductGrid for all dimensions (unified approach)
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        # Convert discretization to Nx_points (add 1 for point count vs intervals)
        Nx_points = [n + 1 for n in spatial_discretization]
        # Use default no_flux_bc for legacy _init_nd path (Issue #674)
        # Note: This is a deprecated code path; users should use geometry-first API
        geometry = TensorProductGrid(
            bounds=spatial_bounds, Nx_points=Nx_points, boundary_conditions=no_flux_bc(dimension=dimension)
        )

        # Store geometry for unified interface
        self.geometry = geometry

        # Set dimension from geometry
        self.dimension = geometry.dimension

        # Store n-D parameters
        self.spatial_bounds = spatial_bounds
        self.spatial_discretization = spatial_discretization

        # Spatial shape from geometry (actual grid size, not discretization)
        # For grids with resolution N, actual points = N+1 per dimension
        if dimension == 1:
            self.spatial_shape = (spatial_discretization[0] + 1,)
        elif dimension == 2 or dimension == 3:
            self.spatial_shape = tuple(n + 1 for n in spatial_discretization)
        else:
            # For TensorProductGrid, use num_spatial_points from geometry
            self.spatial_shape = (geometry.num_spatial_points,)

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.dt: float = T / Nt if Nt > 0 else 0.0
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.sigma: float = sigma
        self.coupling_coefficient: float = coupling_coefficient

        # Check computational feasibility and warn if needed
        if not suppress_warnings:
            self._check_computational_feasibility()

    def _check_computational_feasibility(self) -> None:
        """Warn about computational limits for high-dimensional problems."""
        import warnings

        MAX_PRACTICAL_DIMENSION = 4
        MAX_TOTAL_GRID_POINTS = 10_000_000  # 10 million

        # Calculate total grid points
        total_spatial_points = int(np.prod(self.spatial_shape))
        total_points = total_spatial_points * (self.Nt + 1)
        memory_mb = total_points * 8 / (1024**2)  # Assuming float64

        if self.dimension > MAX_PRACTICAL_DIMENSION:
            warnings.warn(
                f"\n{'=' * 80}\n"
                f"HIGH DIMENSION WARNING\n"
                f"{'=' * 80}\n"
                f"Problem dimension: {self.dimension}D\n"
                f"Practical limit for grid-based FDM: {MAX_PRACTICAL_DIMENSION}D\n"
                f"\n"
                f"Grid-based methods scale as O(N^d), becoming impractical for high dimensions.\n"
                f"Your problem will require:\n"
                f"  - Spatial points: {total_spatial_points:,}\n"
                f"  - Total points (space × time): {total_points:,}\n"
                f"  - Estimated memory: {memory_mb:,.1f} MB per array\n"
                f"\n"
                f"RECOMMENDATION:\n"
                f"For dimension > {MAX_PRACTICAL_DIMENSION}, consider alternative methods:\n"
                f"  - Particle-based collocation methods (algorithms/particle_collocation)\n"
                f"  - Network MFG formulations (for very high dimensions)\n"
                f"  - Dimension reduction techniques\n"
                f"\n"
                f"To suppress this warning: MFGProblem(..., suppress_warnings=True)\n"
                f"{'=' * 80}",
                UserWarning,
                stacklevel=3,
            )
        elif total_points > MAX_TOTAL_GRID_POINTS:
            warnings.warn(
                f"\n{'=' * 80}\n"
                f"MEMORY WARNING\n"
                f"{'=' * 80}\n"
                f"Problem requires {total_points:,} grid points ({memory_mb:,.1f} MB per array).\n"
                f"This may cause memory issues on typical machines.\n"
                f"\n"
                f"Consider:\n"
                f"  - Reducing spatial discretization\n"
                f"  - Reducing time steps\n"
                f"  - Using sparse storage methods\n"
                f"\n"
                f"To suppress this warning: MFGProblem(..., suppress_warnings=True)\n"
                f"{'=' * 80}",
                UserWarning,
                stacklevel=3,
            )

    def _detect_init_mode(
        self,
        Nx: list[int] | None,
        spatial_bounds: list[tuple[float, float]] | None,
        geometry: GeometryProtocol | None,
        network: Any | None,
    ) -> str:
        """
        Detect which initialization mode to use based on provided parameters.

        Args:
            Nx: Normalized array of grid points (or None)
            spatial_bounds: Spatial bounds (or None)
            geometry: Geometry object (or None)
            network: Network object (or None)

        Returns:
            mode: One of "1d_legacy", "nd_grid", "geometry", "network", "default"

        Raises:
            ValueError: If parameters are ambiguous or conflicting
        """
        # Count how many modes are specified
        mode_indicators = {
            "1d_legacy": Nx is not None,
            "nd_grid": spatial_bounds is not None,
            "geometry": geometry is not None,
            "network": network is not None,
        }

        num_modes = sum(mode_indicators.values())

        if num_modes == 0:
            return "default"
        elif num_modes > 1:
            specified = [k for k, v in mode_indicators.items() if v]
            raise ValueError(
                f"Ambiguous initialization: Multiple modes specified: {specified}\n"
                f"Provide ONLY ONE of:\n"
                f"  - Nx (for 1D legacy mode)\n"
                f"  - spatial_bounds (for n-D grid mode)\n"
                f"  - geometry (for complex geometry mode)\n"
                f"  - network (for network MFG mode)"
            )
        else:
            # Exactly one mode specified
            for mode, is_set in mode_indicators.items():
                if is_set:
                    return mode

        # Should never reach here
        return "default"

    def _init_geometry(
        self,
        geometry: GeometryProtocol,
        obstacles: list | None,
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
        lambda_: float | None,
        gamma: float,
        suppress_warnings: bool,
    ) -> None:
        """
        Initialize problem with geometry object implementing GeometryProtocol.

        Accepts any geometry type: TensorProductGrid, BaseGeometry,
        ImplicitDomain, NetworkGeometry, etc.

        Args:
            geometry: Any object implementing GeometryProtocol
            obstacles: List of obstacle geometries (for domain geometries)
            T, Nt: Time domain parameters
            sigma, coupling_coefficient: Physical parameters
            lambda_, gamma: MFG coupling parameters
            suppress_warnings: Suppress warnings
        """
        # Import geometry protocol
        try:
            from mfgarchon.geometry import GeometryProtocol, validate_geometry
        except ImportError as err:
            raise ImportError(
                "Geometry mode requires geometry module. Install with: pip install mfgarchon[geometry]"
            ) from err

        # Validate geometry object implements GeometryProtocol
        if not isinstance(geometry, GeometryProtocol):
            raise TypeError(
                f"geometry must implement GeometryProtocol, got {type(geometry)}. "
                f"Use TensorProductGrid, BaseGeometry, ImplicitDomain, or NetworkGeometry."
            )

        # Validate geometry is properly implemented
        validate_geometry(geometry)

        # Store geometry
        self.geometry = geometry
        self.dimension = geometry.dimension
        self.obstacles = obstacles or []
        self.has_obstacles = len(self.obstacles) > 0

        # Time domain
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt if Nt > 0 else 0.0  # Lowercase (official naming convention)
        self.tSpace = np.linspace(0, T, Nt + 1, endpoint=True)

        # Physical parameters
        self.sigma = sigma  # Already sigma_scalar from __init__ dispatch
        self.coupling_coefficient = coupling_coefficient

        # MFG coupling parameters (for custom Hamiltonians)
        self.lambda_ = lambda_
        self.gamma = gamma

        # Initialize spatial discretization based on geometry type
        from mfgarchon.geometry import GeometryType

        if geometry.geometry_type == GeometryType.CARTESIAN_GRID:
            # CARTESIAN_GRID: Can be TensorProductGrid or AMR mesh
            # Use polymorphic method to get configuration
            config = geometry.get_problem_config()

            # Apply configuration from geometry
            self.num_spatial_points = config["num_spatial_points"]
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config["spatial_bounds"]
            self.spatial_discretization = config["spatial_discretization"]

        elif geometry.geometry_type == GeometryType.UNSTRUCTURED_MESH:
            # BaseGeometry - unstructured mesh via Gmsh
            self.mesh_data = geometry.generate_mesh()
            self.collocation_points = self.mesh_data.vertices
            self.num_spatial_points = len(self.collocation_points)

            # Set spatial shape and bounds
            self.spatial_shape = (self.num_spatial_points,)  # Unstructured
            self.spatial_bounds = None  # Not a regular grid
            self.spatial_discretization = None

        elif geometry.geometry_type == GeometryType.IMPLICIT:
            # ImplicitDomain - point cloud from SDF
            self.num_spatial_points = geometry.num_spatial_points
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = (self.num_spatial_points,)
            self.spatial_bounds = geometry.get_bounding_box()
            self.spatial_discretization = None

        elif geometry.geometry_type in (GeometryType.MAZE, GeometryType.NETWORK):
            # Graph-based geometries (mazes, networks)
            config = geometry.get_problem_config()
            self.num_spatial_points = config["num_spatial_points"]
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config.get("spatial_bounds")
            self.spatial_discretization = config.get("spatial_discretization")

            # Store graph-specific data if available
            if "graph_data" in config:
                self.graph_data = config["graph_data"]

        else:
            # Generic GeometryProtocol object - extract config
            # Issue #557 fix: Extract spatial_bounds from get_problem_config()
            # to support geometries like PointCloudGeometry that provide bounds
            config = geometry.get_problem_config()
            self.num_spatial_points = config["num_spatial_points"]
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config.get("spatial_bounds")
            self.spatial_discretization = config.get("spatial_discretization")

    def _init_network(
        self,
        network: Any,
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
        lambda_: float | None,
        gamma: float,
    ) -> None:
        """
        Initialize problem on network/graph.

        Args:
            network: NetworkGraph or networkx.Graph
            T, Nt: Time domain parameters
            sigma, coupling_coefficient: Physical parameters
        """
        # Import CustomNetwork for geometry-first API
        from mfgarchon.geometry.graph import CustomNetwork

        # Store network
        self.network = network
        self.dimension = "network"  # Special dimension indicator

        # Create CustomNetwork geometry from the network
        try:
            import networkx as nx

            if isinstance(network, nx.Graph):
                # Create geometry from networkx graph
                geometry = CustomNetwork.from_networkx(network)
                self.num_nodes = network.number_of_nodes()
                self.adjacency_matrix = nx.adjacency_matrix(network).toarray()
            else:
                # Assume custom NetworkGraph type with adjacency_matrix attribute
                self.num_nodes = len(network.nodes)
                self.adjacency_matrix = network.adjacency_matrix
                # Create geometry from adjacency matrix
                geometry = CustomNetwork(network.adjacency_matrix)
        except ImportError:
            # NetworkX not available - assume custom type
            self.num_nodes = len(network.nodes)
            self.adjacency_matrix = network.adjacency_matrix
            # Create geometry from adjacency matrix
            geometry = CustomNetwork(network.adjacency_matrix)

        # Store geometry (geometry-first API: never None)
        self.geometry = geometry

        # Time domain
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt if Nt > 0 else 0.0  # Lowercase (official naming convention)
        self.tSpace = np.linspace(0, T, Nt + 1, endpoint=True)

        # Physical parameters
        self.sigma = sigma  # Already sigma_scalar from __init__ dispatch
        self.coupling_coefficient = coupling_coefficient

        # MFG coupling parameters (for custom Hamiltonians)
        self.lambda_ = lambda_
        self.gamma = gamma

        # Spatial discretization (nodes)
        self.spatial_shape = (self.num_nodes,)
        self.num_spatial_points = self.num_nodes  # For networks, spatial points = nodes
        self.spatial_bounds = None
        self.spatial_discretization = None
        self.obstacles = None
        self.has_obstacles = False

    # =========================================================================
    # Geometry Type Helper Properties (Phase 2 of Issue #435)
    # =========================================================================

    @property
    def domain_type(self) -> str:
        """Domain type derived from geometry (Issue #794).

        Returns a string classification of the domain:
        - ``"grid"`` for Cartesian tensor-product grids
        - ``"mesh"`` for unstructured 2D/3D meshes
        - ``"implicit"`` for implicit/SDF-based domains
        - ``"network"``, ``"maze"``, ``"custom"`` for other geometry types
        """
        from mfgarchon.geometry import GeometryType

        gt = self.geometry.geometry_type
        if gt == GeometryType.CARTESIAN_GRID:
            return "grid"
        elif gt == GeometryType.UNSTRUCTURED_MESH:
            return "mesh"
        elif gt == GeometryType.IMPLICIT:
            return "implicit"
        else:
            return str(gt.value)  # "network", "maze", "custom"

    @property
    def is_network(self) -> bool:
        """
        Check if this problem is defined on a network/graph domain.

        Returns:
            True if domain_type is "network", False otherwise.

        Example:
            >>> import networkx as nx
            >>> G = nx.grid_2d_graph(5, 5)
            >>> problem = MFGProblem(network=G, T=1.0, Nt=10)
            >>> problem.is_network
            True
        """
        return self.domain_type == "network"

    @property
    def is_cartesian(self) -> bool:
        """
        Check if this problem is defined on a Cartesian grid domain.

        Returns:
            True if domain_type is "grid", False otherwise.

        Example:
            >>> problem = MFGProblem(Nx=[50], xmin=[0.0], xmax=[1.0], T=1.0, Nt=10)
            >>> problem.is_cartesian
            True
        """
        return self.domain_type == "grid"

    @property
    def is_implicit(self) -> bool:
        """
        Check if this problem uses an implicit/complex geometry.

        Returns:
            True if domain_type is "implicit", False otherwise.

        Example:
            >>> from mfgarchon.geometry import ImplicitDomain
            >>> domain = ImplicitDomain(...)  # Complex geometry
            >>> problem = MFGProblem(geometry=domain, T=1.0, Nt=10)
            >>> problem.is_implicit
            True
        """
        return self.domain_type == "implicit"

    # =========================================================================
    # Physical Parameter Properties (Issue #811)
    # =========================================================================

    @property
    def volatility(self) -> float:
        """SDE volatility sigma. Alias for ``self.sigma``.

        Returns the scalar SDE noise coefficient. For the full field
        (array or callable), use ``self.volatility_field``.
        """
        return self.sigma

    @property
    def diffusion(self) -> float:
        """PDE diffusion coefficient D = sigma^2/2.

        This is the coefficient appearing directly in the PDE:
            dm/dt + div(alpha m) = D Laplacian(m)

        Computed from the scalar ``self.sigma``. For non-scalar or
        state-dependent diffusion, evaluate ``self.volatility_field`` and
        apply the conversion ``D = sigma^2/2`` as needed.
        """
        return self.sigma**2 / 2.0

    @property
    def diffusion_field(self):
        """Deprecated: use volatility_field instead."""
        import warnings

        warnings.warn(
            "diffusion_field is deprecated, use volatility_field. "
            "The field stores SDE volatility (sigma), not PDE diffusion (D).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.volatility_field

    # =========================================================================
    # Hamiltonian Properties (Issue #673)
    # =========================================================================

    @property
    def hamiltonian_class(self) -> Any | None:
        """
        Get the class-based Hamiltonian object if available.

        Returns the HamiltonianBase instance for direct access to:
        - H(x, m, p, t): Hamiltonian value
        - dp(x, m, p, t): ∂H/∂p (optimal control)
        - dm(x, m, p, t): ∂H/∂m (density coupling)
        - optimal_control(x, m, p, t): α* = ±∂H/∂p

        Returns:
            HamiltonianBase instance, or None if using function-based API

        Example:
            >>> from mfgarchon.core.hamiltonian import SeparableHamiltonian
            >>> H = SeparableHamiltonian(...)
            >>> problem = MFGProblem(hamiltonian=H, ...)
            >>> problem.hamiltonian_class.dp(x, m, p, t)  # Direct access
        """
        if self.components is not None:
            return getattr(self.components, "_hamiltonian_class", None)
        return None

    @property
    def lagrangian_class(self) -> Any | None:
        """
        Get the class-based Lagrangian object if available.

        Returns the LagrangianBase instance for direct access to:
        - L(x, alpha, m, t): Running cost value
        - optimal_control(x, m, p, t): alpha* (same as HamiltonianBase)
        - evaluate_hamiltonian(x, m, p, t): H value on-the-fly
        - proximal(tau, z): For ADMM/variational solvers

        Issue #899: LagrangianBase as first-class specification.

        Returns:
            LagrangianBase instance, or None
        """
        if self.components is not None:
            return getattr(self.components, "_lagrangian_class", None)
        return None

    # =========================================================================
    # Time Grid Properties
    # =========================================================================

    @property
    def Nt_points(self) -> int:
        """
        Number of time grid points (Nt + 1).

        Nt is the number of time intervals, while Nt_points is the number
        of time grid points including both endpoints.

        Returns:
            Nt + 1 (number of time points)

        Example:
            >>> problem = MFGProblem(geometry=domain, T=1.0, Nt=10)
            >>> problem.Nt         # 10 intervals
            10
            >>> problem.Nt_points  # 11 points
            11
        """
        return self.Nt + 1

    # =========================================================================
    # Deprecated Legacy Attributes (Computed Properties)
    # Phase 7 of Issue #435: These are computed from geometry for backward
    # compatibility. Access emits DeprecationWarning.
    # =========================================================================

    @property
    @deprecated(since="v0.17.0", replacement="problem.geometry.get_bounds()[0][0]")
    def xmin(self) -> float | None:
        """
        DEPRECATED: Use problem.geometry.get_bounds() instead.

        Returns the minimum x-coordinate for 1D problems.
        """
        # Check for stored value first (for backward compat with tests that set it)
        if getattr(self, "_xmin_override", None) is not None:
            return self._xmin_override
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[0][0])
        return None

    @xmin.setter
    @deprecated(since="v0.17.0", replacement="geometry-first API")
    def xmin(self, value: float | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        self._xmin_override = value

    @property
    @deprecated(since="v0.17.0", replacement="problem.geometry.get_bounds()[1][0]")
    def xmax(self) -> float | None:
        """
        DEPRECATED: Use problem.geometry.get_bounds() instead.

        Returns the maximum x-coordinate for 1D problems.
        """
        if getattr(self, "_xmax_override", None) is not None:
            return self._xmax_override
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[1][0])
        return None

    @xmax.setter
    @deprecated(since="v0.17.0", replacement="geometry-first API")
    def xmax(self, value: float | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        self._xmax_override = value

    @property
    @deprecated(since="v0.17.0", replacement="compute from geometry bounds")
    def Lx(self) -> float | None:
        """
        DEPRECATED: Compute from geometry bounds instead.

        Returns the domain length for 1D problems.
        """
        if self._Lx_override is not None:
            return self._Lx_override
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[1][0] - bounds[0][0])
        return None

    @Lx.setter
    @deprecated(since="v0.17.0", replacement="geometry-first API")
    def Lx(self, value: float | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        self._Lx_override = value

    @property
    @deprecated(since="v0.17.0", replacement="problem.geometry.num_spatial_points - 1")
    def Nx(self) -> int | None:
        """
        DEPRECATED: Use problem.geometry.num_spatial_points instead.

        Returns the number of intervals (not points) for 1D problems.
        """
        if self._Nx_override is not None:
            return self._Nx_override
        if self.geometry is not None and self.dimension == 1:
            # Nx is number of intervals, num_spatial_points is number of points
            return self.geometry.num_spatial_points - 1
        return None

    @Nx.setter
    @deprecated(since="v0.17.0", replacement="geometry-first API")
    def Nx(self, value: int | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        self._Nx_override = value

    @property
    @deprecated(since="v0.17.0", replacement="compute from geometry bounds and num_points")
    def dx(self) -> float | None:
        """
        DEPRECATED: Compute from geometry bounds and num_points instead.

        Returns the grid spacing for 1D problems.
        """
        if self._dx_override is not None:
            return self._dx_override
        if self.geometry is not None and self.dimension == 1:
            from mfgarchon.geometry.base import CartesianGrid

            if isinstance(self.geometry, CartesianGrid):
                return float(self.geometry.get_grid_spacing()[0])
            else:
                # Compute from bounds
                bounds = self.geometry.get_bounds()
                if bounds is not None:
                    n_points = self.geometry.num_spatial_points
                    if n_points > 1:
                        return float((bounds[1][0] - bounds[0][0]) / (n_points - 1))
        return None

    @dx.setter
    @deprecated(since="v0.17.0", replacement="geometry-first API")
    def dx(self, value: float | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        self._dx_override = value

    @property
    @deprecated(since="v0.17.0", replacement="problem.geometry.get_spatial_grid()")
    def xSpace(self) -> np.ndarray | None:
        """
        DEPRECATED: Use problem.geometry.get_spatial_grid() instead.

        Returns the spatial grid array.
        """
        if self._xSpace_override is not None:
            return self._xSpace_override
        if self.geometry is not None:
            return self.geometry.get_spatial_grid()
        return None

    @xSpace.setter
    @deprecated(since="v0.17.0", replacement="geometry-first API")
    def xSpace(self, value: np.ndarray | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        self._xSpace_override = value

    @property
    @deprecated(since="v0.17.0", replacement="problem.geometry")
    def _grid(self) -> Any:
        """
        DEPRECATED: Use problem.geometry instead.

        Returns the geometry object (for backward compatibility).
        """
        if self._grid_override is not None:
            return self._grid_override
        return self.geometry

    @_grid.setter
    @deprecated(since="v0.17.0", replacement="geometry-first API")
    def _grid(self, value: Any) -> None:
        """Allow setting for backward compatibility (with warning)."""
        self._grid_override = value

    # =========================================================================
    # Internal Geometry Helpers (no deprecation warnings)
    # These are for internal use only - external code should use geometry directly
    # =========================================================================

    def _get_domain_length(self) -> float | None:
        """Get domain length for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[1][0] - bounds[0][0])
        return None

    def _get_spacing(self) -> float | None:
        """Get grid spacing for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            from mfgarchon.geometry.base import CartesianGrid

            if isinstance(self.geometry, CartesianGrid):
                return float(self.geometry.get_grid_spacing()[0])
            else:
                bounds = self.geometry.get_bounds()
                if bounds is not None:
                    n_points = self.geometry.num_spatial_points
                    if n_points > 1:
                        return float((bounds[1][0] - bounds[0][0]) / (n_points - 1))
        return None

    def _get_num_intervals(self) -> int | None:
        """Get number of intervals for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            return self.geometry.num_spatial_points - 1
        return None

    def _get_spatial_grid_internal(self) -> np.ndarray | None:
        """Get spatial grid array (internal use, no warning)."""
        if self.geometry is not None:
            return self.geometry.get_spatial_grid()
        return None

    # =========================================================================
    # PDE Coefficient Field Helpers
    # =========================================================================

    def get_diffusion_coefficient_field(self) -> Any:
        """
        Get a CoefficientField wrapper for the diffusion coefficient.

        Returns a CoefficientField that handles scalar, array, and callable
        diffusion coefficients uniformly. Use this in solvers instead of
        directly accessing self.sigma.

        Returns:
            CoefficientField wrapping self.volatility_field with self.sigma as default

        Example:
            >>> diffusion = problem.get_diffusion_coefficient_field()
            >>> sigma_at_t = diffusion.evaluate_at(
            ...     timestep_idx=5,
            ...     grid=x_coords,
            ...     density=m,
            ...     dt=problem.dt
            ... )
        """
        from mfgarchon.utils.pde_coefficients import CoefficientField

        return CoefficientField(
            field=self.volatility_field,
            default_value=self.sigma,
            field_name="diffusion",
            dimension=self.dimension,
        )

    def get_drift_coefficient_field(self) -> Any:
        """
        Get a CoefficientField wrapper for the drift field.

        Returns a CoefficientField that handles array and callable drift
        coefficients uniformly. Use this in solvers instead of directly
        accessing self.drift_field.

        Returns:
            CoefficientField wrapping self.drift_field (default is zero drift)

        Example:
            >>> drift = problem.get_drift_coefficient_field()
            >>> alpha_at_t = drift.evaluate_at(
            ...     timestep_idx=5,
            ...     grid=x_coords,
            ...     density=m,
            ...     dt=problem.dt
            ... )
        """
        from mfgarchon.utils.pde_coefficients import CoefficientField

        # Default drift is zero
        default_drift = 0.0

        return CoefficientField(
            field=self.drift_field,
            default_value=default_drift,
            field_name="drift",
            dimension=self.dimension,
        )

    def has_state_dependent_coefficients(self) -> bool:
        """
        Check if problem has state-dependent (callable) PDE coefficients.

        Solvers may need to handle callable coefficients differently from
        constant/precomputed ones (e.g., re-evaluate at each timestep).

        Returns:
            True if volatility_field or drift_field is callable
        """
        return callable(self.volatility_field) or callable(self.drift_field)

    def __repr__(self) -> str:
        """
        Return string representation using geometry-first API.

        Avoids accessing deprecated attributes to prevent DeprecationWarning
        spam in Jupyter notebooks and debuggers.
        """
        # Use geometry for spatial info, not deprecated attrs
        geom_type = type(self.geometry).__name__ if self.geometry else "None"
        dim = self.dimension

        return f"MFGProblem(geometry={geom_type}, dim={dim}, T={self.T}, Nt={self.Nt}, sigma={self.sigma})"

    def __getstate__(self) -> dict[str, Any]:
        """
        Get state for pickling.

        Returns the instance __dict__ for standard pickle behavior.
        """
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore state from pickle with legacy migration support.

        Handles legacy pickle files where geometry=None but legacy
        attributes (xmin, xmax, Nx) are present. Reconstructs geometry
        from these attributes for backward compatibility.
        """
        # Detect legacy format: geometry=None but has legacy 1D attrs
        if state.get("geometry") is None and state.get("xmin") is not None:
            try:
                from mfgarchon.geometry import TensorProductGrid
                from mfgarchon.geometry.boundary import no_flux_bc

                # Reconstruct geometry from legacy attributes
                xmin = state.get("xmin")
                xmax = state.get("xmax")
                Nx = state.get("Nx")

                if xmin is None or xmax is None or Nx is None:
                    raise KeyError("Missing required legacy fields (xmin, xmax, Nx)")

                # Handle both scalar and list forms
                if isinstance(xmin, (int, float)):
                    bounds = [(float(xmin), float(xmax))]
                    Nx_points = [int(Nx) + 1]
                else:
                    bounds = list(zip(xmin, xmax, strict=True))
                    Nx_points = [n + 1 for n in Nx]

                # Use default no_flux_bc for legacy pickle migration (Issue #674)
                dimension = len(bounds)
                state["geometry"] = TensorProductGrid(
                    bounds=bounds,
                    Nx_points=Nx_points,
                    boundary_conditions=no_flux_bc(dimension=dimension),
                )
            except (KeyError, ImportError) as e:
                import warnings

                warnings.warn(
                    f"Unable to migrate legacy pickle format: {e}. "
                    "This pickle file may be from an incompatible version. "
                    "Consider recreating the MFGProblem.",
                    UserWarning,
                    stacklevel=2,
                )

        self.__dict__.update(state)

    def _detect_solver_compatibility(self) -> None:
        """
        Detect which solver types are compatible with this problem.

        Sets:
            self.solver_compatible: List of compatible solver type strings
            self.solver_recommendations: Dict mapping use cases to solvers

        Called automatically after initialization.
        """
        compatible = []
        recommendations = {}

        # Get problem characteristics
        is_grid = self.domain_type == "grid"
        is_implicit = self.domain_type == "implicit"
        is_network = self.domain_type == "network"
        dim = self.dimension if isinstance(self.dimension, int) else None

        # FDM: Requires regular grid, no complex geometry, works best for dim <= 3
        if is_grid and not self.has_obstacles:
            compatible.append("fdm")
            if dim and dim <= 2:
                recommendations["fast"] = "fdm"
                recommendations["accurate"] = "fdm"

        # Semi-Lagrangian: Works with grids, especially good for higher dimensions
        if is_grid:
            compatible.append("semi_lagrangian")
            if dim and dim >= 3:
                recommendations["fast"] = "semi_lagrangian"

        # GFDM: Works with grids and complex geometry (particle collocation)
        if is_grid or is_implicit:
            compatible.append("gfdm")
            if is_implicit or self.has_obstacles:
                recommendations["obstacles"] = "gfdm"
                recommendations["complex_geometry"] = "gfdm"

        # Particle methods: Work with everything except pure networks
        if not is_network:
            compatible.append("particle")
            if dim and dim >= 4:
                recommendations["high_dimensional"] = "particle"
                recommendations["fast"] = "particle"

        # Network solver: Only for network problems
        if is_network:
            compatible.append("network_solver")
            recommendations["default"] = "network_solver"

        # DGM: Works with grids (experimental)
        if is_grid:
            compatible.append("dgm")

        # PINN: Works with everything (deep learning approach)
        compatible.append("pinn")
        if dim and dim >= 5:
            recommendations["very_high_dimensional"] = "pinn"

        # Set attributes
        self.solver_compatible = compatible
        self.solver_recommendations = recommendations

        # Set default recommendation
        if "default" not in recommendations:
            if is_grid and dim and dim <= 2:
                recommendations["default"] = "fdm"
            elif is_grid and dim and dim == 3:
                recommendations["default"] = "semi_lagrangian"
            elif is_implicit:
                recommendations["default"] = "gfdm"
            elif compatible:
                recommendations["default"] = compatible[0]

    def validate_solver_type(self, solver_type: str) -> None:
        """
        Validate that solver type is compatible with this problem.

        Args:
            solver_type: Solver type identifier (e.g., "fdm", "gfdm", "particle")

        Raises:
            ValueError: If solver type is incompatible with problem configuration

        Note:
            This method is called by solver constructors to provide early
            error detection with helpful messages.
        """
        # Compatibility should already be detected in __init__
        # If empty, initialization failed - raise explicit error
        if not self.solver_compatible:
            raise RuntimeError("Solver compatibility not detected. This indicates __init__ didn't complete properly.")

        if solver_type not in self.solver_compatible:
            # Build helpful error message
            reason = self._get_incompatibility_reason(solver_type)
            suggestion = self._get_solver_suggestion()

            raise ValueError(
                f"Solver type '{solver_type}' is incompatible with this problem.\n\n"
                f"Problem Configuration:\n"
                f"  Domain type: {self.domain_type}\n"
                f"  Dimension: {self.dimension}\n"
                f"  Has obstacles: {self.has_obstacles}\n\n"
                f"Reason: {reason}\n\n"
                f"Compatible solvers: {self.solver_compatible}\n\n"
                f"Suggestion: {suggestion}"
            )

    def _get_incompatibility_reason(self, solver_type: str) -> str:
        """Get human-readable reason why solver is incompatible."""
        reasons = {
            "fdm": {
                "implicit": "FDM requires regular grid, not implicit geometry",
                "network": "FDM requires spatial grid, not network structure",
                "obstacles": "FDM doesn't support obstacles (use GFDM instead)",
            },
            "semi_lagrangian": {
                "implicit": "Semi-Lagrangian requires regular grid",
                "network": "Semi-Lagrangian requires spatial grid",
            },
            "gfdm": {
                "network": "GFDM requires spatial coordinates, not network structure",
            },
            "particle": {
                "network": "Particle methods require spatial domain",
            },
            "network_solver": {
                "grid": "Network solver requires network structure, not spatial grid",
                "implicit": "Network solver requires network structure",
            },
        }

        domain_reasons = reasons.get(solver_type, {})
        return domain_reasons.get(self.domain_type, "Solver not compatible with problem configuration")

    def _get_solver_suggestion(self) -> str:
        """Get helpful suggestion for which solver to use."""
        if not self.solver_recommendations:
            if self.solver_compatible:
                return f"Try using: {self.solver_compatible[0]}"
            return "No compatible solvers found for this configuration"

        # Get default recommendation
        default_solver = self.solver_recommendations.get(
            "default", self.solver_compatible[0] if self.solver_compatible else None
        )

        if not default_solver:
            return "No solver recommendations available"

        # Build recommendation text
        suggestion = f"Use solver '{default_solver}' (recommended for this problem)"

        # Add context-specific recommendations
        additional_recs = []
        if "obstacles" in self.solver_recommendations:
            additional_recs.append(f"obstacles: {self.solver_recommendations['obstacles']}")
        if "fast" in self.solver_recommendations and self.solver_recommendations["fast"] != default_solver:
            additional_recs.append(f"fastest: {self.solver_recommendations['fast']}")
        if "accurate" in self.solver_recommendations and self.solver_recommendations["accurate"] != default_solver:
            additional_recs.append(f"most accurate: {self.solver_recommendations['accurate']}")

        if additional_recs:
            suggestion += f"\n  Alternative recommendations: {', '.join(additional_recs)}"

        suggestion += "\n  Or use create_fast_solver() for automatic selection"

        return suggestion

    def get_solver_info(self) -> dict[str, Any]:
        """
        Get comprehensive solver compatibility information.

        Returns:
            Dictionary with solver compatibility details:
            - compatible: List of compatible solver types
            - recommendations: Dict of use-case specific recommendations
            - dimension: Problem dimension
            - domain_type: Type of spatial domain
            - complexity: Estimated computational complexity
        """
        # Compatibility should already be detected in __init__
        if not self.solver_compatible:
            raise RuntimeError("Solver compatibility not detected. This indicates __init__ didn't complete properly.")

        return {
            "compatible": self.solver_compatible,
            "recommendations": self.solver_recommendations,
            "dimension": self.dimension,
            "domain_type": self.domain_type,
            "has_obstacles": self.has_obstacles,
            "complexity": self._estimate_complexity(),
            "default_solver": self.solver_recommendations.get("default", None),
        }

    def _estimate_complexity(self) -> str:
        """Estimate computational complexity category."""
        if self.domain_type == "network":
            return "O(N_nodes × N_time)"

        if isinstance(self.dimension, int):
            if self.dimension == 1:
                return "O(Nx × Nt)"
            elif self.dimension == 2:
                return "O(Nx × Ny × Nt)"
            elif self.dimension == 3:
                return "O(Nx × Ny × Nz × Nt)"
            else:
                return f"O(N^{self.dimension} × Nt) - curse of dimensionality"

        return "Problem-dependent"

    def get_computational_cost_estimate(self) -> dict:
        """
        Get estimated computational cost for the problem.

        Returns:
            Dictionary with cost estimates:
            - total_spatial_points: Total spatial grid points
            - total_points: Total grid points (space × time)
            - memory_per_array_mb: Memory per solution array (MB)
            - estimated_memory_mb: Total estimated memory (MB)
            - is_feasible: Whether problem is computationally feasible
            - warnings: List of warnings about computational costs
        """
        total_spatial_points = int(np.prod(self.spatial_shape))
        total_points = total_spatial_points * (self.Nt + 1)
        memory_per_array_mb = total_points * 8 / (1024**2)
        estimated_total_mb = memory_per_array_mb * 10  # Rough estimate: ~10 arrays

        warnings_list = []
        is_feasible = True

        if self.dimension > 4:
            warnings_list.append(f"Dimension {self.dimension}D exceeds practical limit (4D)")
            is_feasible = False

        if total_points > 10_000_000:
            warnings_list.append(f"Total points ({total_points:,}) exceeds recommended limit (10M)")
            is_feasible = False

        if estimated_total_mb > 1000:
            warnings_list.append(f"Estimated memory ({estimated_total_mb:.1f} MB) may be excessive")

        return {
            "dimension": self.dimension,
            "spatial_shape": self.spatial_shape,
            "total_spatial_points": total_spatial_points,
            "total_points": total_points,
            "memory_per_array_mb": memory_per_array_mb,
            "estimated_memory_mb": estimated_total_mb,
            "is_feasible": is_feasible,
            "warnings": warnings_list,
        }

    # Issue #670/#671: Legacy default functions removed (Fail Fast principle)
    # - _potential(): Removed - zero potential is now the explicit default
    # - _u_final(): Removed - must be provided via MFGComponents.u_terminal
    # - _m_initial(): Removed - must be provided via MFGComponents.m_initial

    def _initialize_functions(self, **kwargs: Any) -> None:
        """Initialize potential, initial density, and final value functions.

        Issue #670: u_terminal/m_initial must be provided via MFGComponents.
        No silent defaults - Fail Fast principle.
        """
        # Initialize arrays with correct shape for both 1D and n-D
        self.f_potential = np.zeros(self.spatial_shape)
        self.u_terminal = np.zeros(self.spatial_shape)
        self.m_initial = np.zeros(self.spatial_shape)

        # Issue #670: u_terminal and m_initial MUST come from MFGComponents
        has_components = self.components is not None
        has_u_terminal = has_components and self.components.u_terminal is not None
        has_m_initial = has_components and self.components.m_initial is not None

        # Issue #681: Validate IC/BC compatibility before setup
        if has_components and self.geometry is not None:
            from mfgarchon.utils.validation import ValidationError, validate_components

            result = validate_components(
                self.components,
                self.geometry,
                require_m_initial=True,
                require_u_terminal=True,
            )
            if not result.is_valid:
                raise ValidationError(result)

        # Issue #686: Validate custom functions (Hamiltonian, derivatives)
        if has_components and self.geometry is not None:
            from mfgarchon.utils.validation import validate_custom_functions

            h_class = self.components._hamiltonian_class
            if h_class is not None:
                func_result = validate_custom_functions(
                    hamiltonian=h_class,
                    dH_dm=h_class.dm,
                    dH_dp=h_class.dp,
                    geometry=self.geometry,
                    check_consistency=False,
                )
                if not func_result.is_valid:
                    raise ValidationError(func_result)

            # Validate drift if callable
            if callable(self.drift_field):
                from mfgarchon.utils.validation import validate_drift

                drift_result = validate_drift(self.drift_field, self.geometry)
                if not drift_result.is_valid:
                    raise ValidationError(drift_result)

            # Validate potential if callable
            if self.components.potential_func is not None:
                from mfgarchon.utils.validation import validate_running_cost

                pot_result = validate_running_cost(self.components.potential_func, self.geometry)
                if not pot_result.is_valid:
                    raise ValidationError(pot_result)

        # Issue #687: Validate array-type diffusion/drift fields
        if self.geometry is not None and self.spatial_shape is not None:
            from mfgarchon.utils.validation import (
                ValidationResult,
                validate_array_dtype,
                validate_field_shape,
                validate_finite,
            )

            # Validate volatility_field if ndarray
            if isinstance(self.volatility_field, np.ndarray):
                arr_result = ValidationResult()
                for check in [
                    validate_array_dtype(self.volatility_field, "volatility_field"),
                    validate_field_shape(self.volatility_field, self.spatial_shape, "volatility_field"),
                    validate_finite(self.volatility_field, "volatility_field"),
                ]:
                    arr_result.issues.extend(check.issues)
                    if not check.is_valid:
                        arr_result.is_valid = False
                if not arr_result.is_valid:
                    raise ValidationError(arr_result)

            # Validate drift_field if ndarray
            if isinstance(self.drift_field, np.ndarray):
                arr_result = ValidationResult()
                for check in [
                    validate_array_dtype(self.drift_field, "drift_field"),
                    validate_field_shape(self.drift_field, self.spatial_shape, "drift_field"),
                    validate_finite(self.drift_field, "drift_field"),
                ]:
                    arr_result.issues.extend(check.issues)
                    if not check.is_valid:
                        arr_result.is_valid = False
                if not arr_result.is_valid:
                    raise ValidationError(arr_result)

        # === u_terminal: MUST be in MFGComponents (Issue #670: no silent default) ===
        if has_u_terminal:
            self._setup_custom_final_value()
        else:
            raise ValueError(
                "u_terminal (terminal condition) must be provided in MFGComponents. "
                "Example: MFGComponents(u_terminal=lambda x: ..., m_initial=lambda x: ...). "
                "See examples/basic/lq_mfg_classic.py for the classic LQ-MFG setup."
            )

        # === m_initial: MUST be in MFGComponents (Issue #670: no silent default) ===
        if has_m_initial:
            self._setup_custom_initial_density()
        else:
            raise ValueError(
                "m_initial (initial density) must be provided in MFGComponents. "
                "Example: MFGComponents(u_terminal=lambda x: ..., m_initial=lambda x: ...). "
                "See examples/basic/lq_mfg_classic.py for the classic LQ-MFG setup."
            )

        # === Potential: V(x,t) - defaults to zero (Issue #671: explicit default) ===
        # Zero potential is a valid physical choice (many MFG problems have V=0).
        # Unlike m_initial/u_terminal, zero potential doesn't require explicit specification.
        has_potential = has_components and self.components.potential_func is not None
        if has_potential:
            self._setup_custom_potential()
        else:
            # Issue #671: Zero potential is the explicit default (physically meaningful)
            self.f_potential[:] = 0.0

        # Issue #687: Validate computed arrays for NaN/Inf (after setup methods)
        if self.geometry is not None:
            from mfgarchon.utils.validation import ValidationError, validate_finite

            u_result = validate_finite(self.u_terminal, "u_terminal")
            if not u_result.is_valid:
                raise ValidationError(u_result)

            m_result = validate_finite(self.m_initial, "m_initial")
            if not m_result.is_valid:
                raise ValidationError(m_result)

            if has_potential:
                pot_result = validate_finite(self.f_potential, "f_potential")
                if not pot_result.is_valid:
                    raise ValidationError(pot_result)

        # === Issue #672: Validate m_initial before normalization (Fail Fast) ===
        # Check 1: Non-negativity (density must be >= 0)
        if np.any(self.m_initial < 0):
            min_val = np.min(self.m_initial)
            raise ValueError(
                f"m_initial contains negative values (min={min_val:.6e}). "
                "Initial density must be non-negative. "
                "Check your m_initial function in MFGComponents."
            )

        # Check 2: Non-zero mass (must have some mass to normalize)
        if np.sum(self.m_initial) < 1e-15:
            raise ValueError(
                "m_initial has zero or negligible total mass. "
                "Initial density must integrate to a positive value. "
                "Check your m_initial function in MFGComponents."
            )

        # Normalize initial density
        if self.dimension == "network":
            # Network/graph: discrete probability mass, sum = 1
            # No cell volume - just normalize sum to 1
            integral_m_initial = np.sum(self.m_initial)
        elif self.dimension == 1:
            # 1D normalization (original)
            dx = self._get_spacing() or 1.0
            integral_m_initial = np.sum(self.m_initial) * dx
        elif self.spatial_bounds is not None and self.spatial_discretization is not None:
            # n-D normalization (integrate over all dimensions)
            # For tensor product grid: integral = sum(m) * prod(dx_i)
            dx_prod = np.prod(
                [
                    (bounds[1] - bounds[0]) / n
                    for bounds, n in zip(self.spatial_bounds, self.spatial_discretization, strict=False)
                ]
            )
            integral_m_initial = np.sum(self.m_initial) * dx_prod
        else:
            # For unstructured/implicit geometries: use uniform normalization
            # This is a rough approximation - for accurate integration, use proper
            # quadrature rules based on the geometry type
            integral_m_initial = np.sum(self.m_initial) / self.num_spatial_points

        if integral_m_initial > 1e-10:
            self.m_initial /= integral_m_initial

    # Issue #670: _setup_default_initial_density() removed - m_initial must be explicit

    # Methods inherited from HamiltonianMixin:
    # - H(), dH_dm(), get_hjb_hamiltonian_jacobian_contrib()
    # - get_hjb_residual_m_coupling_term(), get_potential_at_time()
    # - _setup_custom_potential(), _validate_hamiltonian_components()
    #
    # Methods inherited from ConditionsMixin:
    # - get_boundary_conditions()
    # - _setup_custom_initial_density(), _setup_custom_final_value()

    def get_u_terminal(self) -> np.ndarray:
        """Get terminal condition u(T, x). Issue #670: unified naming."""
        return self.u_terminal.copy()

    def get_u_final(self) -> np.ndarray:
        """Deprecated: use get_u_terminal() instead.

        .. deprecated:: v0.17.6
            Use :meth:`get_u_terminal` instead. Will be removed in v1.0.0.
        """
        from mfgarchon.utils.deprecation import deprecated

        # Apply decorator dynamically to avoid import cycle at module level
        @deprecated(
            since="v0.17.6",
            replacement="use get_u_terminal() instead",
            reason="Renamed for consistency with MFG literature terminology",
        )
        def _deprecated_get_u_final() -> np.ndarray:
            return self.get_u_terminal()

        return _deprecated_get_u_final()

    def get_m_initial(self) -> np.ndarray:
        """Get initial density m(0, x). Issue #670: unified naming."""
        return self.m_initial.copy()

    # Legacy aliases for backward compatibility
    def get_u_fin(self) -> np.ndarray:
        """Legacy alias for get_u_terminal().

        .. deprecated:: v0.17.6
            Use :meth:`get_u_terminal` instead. Will be removed in v1.0.0.
        """
        from mfgarchon.utils.deprecation import deprecated

        @deprecated(
            since="v0.17.6",
            replacement="use get_u_terminal() instead",
            reason="Shortened alias deprecated in favor of full name",
        )
        def _deprecated_get_u_fin() -> np.ndarray:
            return self.get_u_terminal()

        return _deprecated_get_u_fin()

    def get_m_init(self) -> np.ndarray:
        """Legacy alias for get_m_initial()."""
        return self.get_m_initial()

    def get_final_u(self) -> np.ndarray:
        """Legacy alias for get_u_terminal()."""
        return self.get_u_terminal()

    def get_initial_m(self) -> np.ndarray:
        """Legacy alias for get_m_initial()."""
        return self.get_m_initial()

    def get_problem_info(self) -> dict[str, Any]:
        """Get information about the problem."""
        # Get domain info from geometry (modern API)
        bounds = self.geometry.get_bounds() if self.geometry is not None else None
        domain_info = {
            "dimension": self.dimension,
            "num_spatial_points": self.geometry.num_spatial_points if self.geometry is not None else None,
        }
        if bounds is not None and self.dimension == 1:
            domain_info["xmin"] = float(bounds[0][0])
            domain_info["xmax"] = float(bounds[1][0])
            domain_info["Nx"] = self._get_num_intervals()

        if self.is_custom and self.components is not None:
            return {
                "description": self.components.description,
                "problem_type": self.components.problem_type,
                "is_custom": True,
                "has_custom_hamiltonian": True,
                "has_custom_potential": self.components.potential_func is not None,
                "has_custom_initial": self.components.m_initial is not None,
                "has_custom_final": self.components.u_terminal is not None,
                # Issue #673: jacobian_fd() always available on HamiltonianBase
                "has_jacobian": self.components._hamiltonian_class is not None,
                "parameters": self.components.parameters,
                "domain": domain_info,
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"sigma": self.sigma, "coupling_coefficient": self.coupling_coefficient},
            }
        else:
            return {
                "description": "Default MFG Problem",
                "problem_type": "example",
                "is_custom": False,
                "has_custom_hamiltonian": False,
                "has_custom_potential": False,
                "has_custom_initial": False,
                "has_custom_final": False,
                "has_jacobian": False,
                "has_coupling": False,
                "parameters": {},
                "domain": domain_info,
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"sigma": self.sigma, "coupling_coefficient": self.coupling_coefficient},
            }

    # ============================================================================
    # Kwargs Validation - Fail Fast on Deprecated/Unrecognized Parameters
    # ============================================================================

    # Deprecated kwargs that should use MFGComponents instead (Issue #666, #670)
    _DEPRECATED_KWARGS: ClassVar[dict[str, str]] = {
        "hamiltonian": "MFGComponents.hamiltonian_func",
        "dH_dm": "MFGComponents.hamiltonian_dm_func",
        "dH_dp": "MFGComponents.hamiltonian_dp_func",
        "potential": "MFGComponents.potential_func",
        "running_cost": "MFGComponents.hamiltonian_func",
        "terminal_cost": "MFGComponents.u_terminal",
        # Issue #670: initial/terminal conditions now ONLY via MFGComponents
        "m_initial": "MFGComponents.m_initial",
        "u_final": "MFGComponents.u_terminal",
        "initial_density": "MFGComponents.m_initial",
    }

    # Known valid kwargs that are consumed by _initialize_functions or mixins
    _RECOGNIZED_KWARGS: ClassVar[set[str]] = {
        "boundary_conditions",  # BC object
    }

    def _validate_kwargs(self, kwargs: dict[str, Any]) -> None:
        """
        Validate kwargs - fail fast on deprecated or unrecognized parameters.

        Issue #666: Prevents silent fail where user-provided kwargs are ignored.
        Uses centralized validate_kwargs utility with MFGProblem-specific guidance.

        Raises:
            ValueError: If deprecated kwargs are passed (must use MFGComponents)
            UserWarning: If unrecognized kwargs are passed (probably a typo)
        """
        try:
            validate_kwargs(
                kwargs=kwargs,
                deprecated_kwargs=self._DEPRECATED_KWARGS,
                recognized_kwargs=self._RECOGNIZED_KWARGS,
                context="MFGProblem",
                error_on_deprecated=True,
                warn_on_unrecognized=True,
            )
        except ValueError as e:
            # Enhance error with MFGProblem-specific migration guide
            migration_guide = """

The old kwargs-based Hamiltonian API is no longer supported.
Use MFGComponents for custom problem definitions:

  from mfgarchon.core.mfg_problem import MFGComponents

  components = MFGComponents(
      hamiltonian_func=my_hamiltonian,
      hamiltonian_dm_func=my_dH_dm,
      m_initial=my_m0,
  )

  problem = MFGProblem(
      geometry=my_geometry,
      T=T, Nt=Nt,
      sigma=sigma,
      components=components,
  )

See: docs/migration/HAMILTONIAN_API.md"""
            raise ValueError(str(e) + migration_guide) from None

    # ============================================================================
    # Solve Method - Primary API for solving MFG problems
    # ============================================================================

    def solve(
        self,
        Nt: int | None = None,
        *,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        verbose: bool | None = None,
        config: Any | None = None,
        scheme: Any | None = None,
        hjb_solver: Any | None = None,
        fp_solver: Any | None = None,
    ) -> Any:
        """
        Solve this MFG problem using three-mode API (Issue #580).

        **Three Solving Modes:**

        1. **Safe Mode** (Recommended): Specify validated numerical scheme
            >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
            Automatically creates dual HJB-FP solver pair with duality guarantee.

        2. **Expert Mode**: Manual solver injection for advanced users
            >>> hjb = HJBFDMSolver(problem)
            >>> fp = FPFDMSolver(problem)
            >>> result = problem.solve(hjb_solver=hjb, fp_solver=fp)
            Full control, but duality validation warnings if mismatched.

        3. **Auto Mode**: Intelligent automatic selection (default)
            >>> result = problem.solve()
            Analyzes geometry and selects appropriate scheme automatically.

        Args:
            Nt: Time discretization steps (Issue #875). If provided, overrides
                the Nt stored at construction time. This separates physics (T)
                from numerics (Nt) — different solvers may want different Nt
                for the same physical problem.
            max_iterations: Maximum fixed-point iterations (default: from config or 100)
            tolerance: Convergence tolerance (default: from config or 1e-6)
            verbose: Show solver progress (default: from config or True)
            config: Optional MFGSolverConfig for advanced configuration.
                If config is provided, its values are used as defaults.
                Explicit parameters (max_iterations, tolerance, verbose) override config.
            scheme: NumericalScheme for Safe Mode (FDM_UPWIND, SL_LINEAR, GFDM, etc.)
            hjb_solver: Pre-initialized HJB solver for Expert Mode
            fp_solver: Pre-initialized FP solver for Expert Mode

        Returns:
            SolverResult with U (value function), M (density), convergence info

        Examples:
            >>> # Safe Mode: Automatic dual pairing
            >>> from mfgarchon.types import NumericalScheme
            >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

            >>> # Auto Mode: Intelligent selection
            >>> result = problem.solve()

            >>> # Expert Mode: Full control
            >>> hjb = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")
            >>> fp = FPSLAdjointSolver(problem)
            >>> result = problem.solve(hjb_solver=hjb, fp_solver=fp)

        Note:
            Cannot mix modes: specify either `scheme` OR (`hjb_solver` + `fp_solver`),
            not both. Omit all to use Auto Mode.
        """
        from mfgarchon.alg.numerical.coupling import FixedPointIterator
        from mfgarchon.config import MFGSolverConfig
        from mfgarchon.factory import create_paired_solvers, get_recommended_scheme
        from mfgarchon.utils import check_solver_duality

        # Issue #875: Nt override at solve() time
        if Nt is not None and Nt != self.Nt:
            # Nt deeply affects solver setup. For v1.0 API, reconstruct
            # with the correct Nt and delegate solve to the new problem.
            if self._v1_model is not None:
                new_problem = MFGProblem(
                    model=self._v1_model,
                    domain=self.geometry,
                    conditions=self._v1_conditions,
                    constraints=self._v1_constraints,
                    Nt=Nt,
                )
                return new_problem.solve(
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    verbose=verbose,
                    config=config,
                    scheme=scheme,
                    hjb_solver=hjb_solver,
                    fp_solver=fp_solver,
                )
            else:
                raise ValueError(
                    f"solve(Nt={Nt}) differs from construction Nt={self.Nt}. "
                    "Nt override at solve time is only supported for v1.0 API problems "
                    "(created with model/domain/conditions). For legacy problems, "
                    "reconstruct with the desired Nt."
                )

        # Create or update config
        if config is None:
            config = MFGSolverConfig()

        # Override config only with explicitly passed parameters (not None)
        # This allows config values to be used when parameters are not specified
        if max_iterations is not None:
            config.picard.max_iterations = max_iterations
        if tolerance is not None:
            config.picard.tolerance = tolerance
        if verbose is not None:
            config.picard.verbose = verbose

        # ═══════════════════════════════════════════════════════════════════════
        # Phase 3: Three-Mode API (Issue #580)
        # ═══════════════════════════════════════════════════════════════════════

        # Mode Detection
        safe_mode = scheme is not None
        expert_mode = hjb_solver is not None or fp_solver is not None

        # Mode Validation: Cannot mix modes
        if safe_mode and expert_mode:
            raise ValueError(
                "Cannot mix Safe Mode (scheme parameter) with Expert Mode "
                "(hjb_solver/fp_solver parameters). Use one mode at a time:\n"
                "  • Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)\n"
                "  • Expert Mode: problem.solve(hjb_solver=hjb, fp_solver=fp)\n"
                "  • Auto Mode: problem.solve() [no scheme/solver params]"
            )

        # ─────────────────────────────────────────────────────────────────────
        # Safe Mode: Automatic dual pairing via scheme selection
        # ─────────────────────────────────────────────────────────────────────
        if safe_mode:
            # Convert string to NumericalScheme if needed
            from mfgarchon.types import NumericalScheme

            if isinstance(scheme, str):
                try:
                    scheme = NumericalScheme(scheme)
                except ValueError:
                    raise ValueError(
                        f"Unknown scheme string: {scheme!r}. Valid schemes: {[s.value for s in NumericalScheme]}"
                    ) from None

            # Create validated dual pair (Phase 2 factory)
            hjb_solver, fp_solver = create_paired_solvers(
                problem=self,
                scheme=scheme,
                validate_duality=True,  # Guaranteed dual by construction
            )

            if verbose:
                from mfgarchon.utils.mfg_logging import get_logger

                logger = get_logger(__name__)
                logger.info(f"Safe Mode: Created dual solver pair for {scheme.value}")

        # ─────────────────────────────────────────────────────────────────────
        # Expert Mode: Manual solver injection with duality validation
        # ─────────────────────────────────────────────────────────────────────
        elif expert_mode:
            # Both solvers must be provided
            if hjb_solver is None or fp_solver is None:
                raise ValueError(
                    "Expert Mode requires BOTH hjb_solver and fp_solver. "
                    "You provided only one. Either:\n"
                    "  • Provide both: problem.solve(hjb_solver=hjb, fp_solver=fp)\n"
                    "  • Use Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)\n"
                    "  • Use Auto Mode: problem.solve() [omit both]"
                )

            # Validate duality (educational warnings if mismatched)
            result = check_solver_duality(hjb_solver, fp_solver, warn_on_mismatch=True)

            if verbose and not result.is_valid_pairing():
                from mfgarchon.utils.mfg_logging import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"Expert Mode: Non-dual solver pair detected!\n"
                    f"  HJB: {type(hjb_solver).__name__} ({result.hjb_family})\n"
                    f"  FP: {type(fp_solver).__name__} ({result.fp_family})\n"
                    f"  Status: {result.status.value}\n"
                    f"This may lead to poor convergence or Nash gap issues.\n"
                    f"Consider using Safe Mode for guaranteed duality."
                )

        # ─────────────────────────────────────────────────────────────────────
        # Auto Mode: Intelligent scheme selection (Phase 3 future work)
        # ─────────────────────────────────────────────────────────────────────
        else:  # auto_mode
            # Phase 3 TODO: Implement geometry introspection
            # For now, get_recommended_scheme() returns FDM_UPWIND as safe default
            recommended_scheme = get_recommended_scheme(self)

            hjb_solver, fp_solver = create_paired_solvers(
                problem=self,
                scheme=recommended_scheme,
                validate_duality=True,
            )

            if verbose:
                from mfgarchon.utils.mfg_logging import get_logger

                logger = get_logger(__name__)
                logger.info(f"Auto Mode: Selected {recommended_scheme.value} (geometry-based recommendation)")

        # ─────────────────────────────────────────────────────────────────────
        # Create fixed-point iterator with selected/validated solvers
        # ─────────────────────────────────────────────────────────────────────
        solver = FixedPointIterator(
            problem=self,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            config=config,
        )

        return solver.solve(verbose=verbose)

    # ==========================================================================
    # API v1.0: Parameter Variation Helpers (Issue #875)
    # ==========================================================================

    def with_model(self, model: Any) -> MFGProblem:
        """Return new problem with different model (game rules).

        Requires that this problem was created with the v1.0 API.
        """
        if self._v1_model is None:
            raise ValueError("with_model() requires a problem created with API v1.0 (model/domain/conditions)")
        return MFGProblem(
            model=model,
            domain=self.geometry,
            conditions=self._v1_conditions,
            constraints=self._v1_constraints,
            Nt=self.Nt,
        )

    def with_domain(self, domain: Any) -> MFGProblem:
        """Return new problem with different domain (spatial geometry)."""
        if self._v1_model is None:
            raise ValueError("with_domain() requires a problem created with API v1.0 (model/domain/conditions)")
        return MFGProblem(
            model=self._v1_model,
            domain=domain,
            conditions=self._v1_conditions,
            constraints=self._v1_constraints,
            Nt=self.Nt,
        )

    def with_conditions(self, conditions: Any) -> MFGProblem:
        """Return new problem with different conditions (time + IC/TC)."""
        if self._v1_model is None:
            raise ValueError("with_conditions() requires a problem created with API v1.0 (model/domain/conditions)")
        return MFGProblem(
            model=self._v1_model,
            domain=self.geometry,
            conditions=conditions,
            constraints=self._v1_constraints,
            Nt=self.Nt,
        )

    # No with_sigma() or with_T() — these are premature convenience shortcuts
    # that break orthogonality. sigma lives in Model, T lives in Conditions.
    # Use with_model(Model(hamiltonian=H, sigma=0.2)) or
    # with_conditions(Conditions(u_terminal=..., m_initial=..., T=2.0)) instead.
