from __future__ import annotations

import warnings
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mfgarchon.utils.deprecation import deprecated_parameter

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import BoundaryConditions
    from mfgarchon.geometry.implicit import ImplicitDomain

try:  # pragma: no cover - optional SciPy dependency
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback when SciPy missing
    gaussian_kde = None
    SCIPY_AVAILABLE = False

from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator
from mfgarchon.geometry.boundary.types import BCType

# Issue #625: Migrated from tensor_calculus to operators/stencils
from mfgarchon.operators.stencils.finite_difference import gradient_nd
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.particle import (
    interpolate_grid_to_particles,
    sample_from_density,
)

from .base_fp import BaseFPSolver
from .fp_particle_bc import apply_boundary_conditions as _apply_bc
from .fp_particle_bc import enforce_obstacle_boundary as _enforce_obstacle
from .fp_particle_bc import get_topology_per_dimension as _get_topology
from .fp_particle_bc import needs_segment_aware_bc as _needs_segment_bc
from .fp_particle_density import generate_brownian_increment as _gen_brownian
from .fp_particle_density import normalize_density as _normalize
from .particle_result import FPParticleResult

logger = get_logger(__name__)


class KDENormalization(StrEnum):
    """KDE normalization strategy for particle-based FP solvers."""

    NONE = "none"  # No normalization (raw KDE output)
    INITIAL_ONLY = "initial_only"  # Normalize only at t=0
    ALL = "all"  # Normalize at every time step (default)


class KDEMethod(StrEnum):
    """Density estimation method from particles (Issue #709).

    Different methods handle boundary bias differently:
    - STANDARD: No boundary correction (~50% underestimation at boundaries)
    - REFLECTION: Ghost particles via reflection (has boundary/adjacent redistribution)
    - RENORMALIZATION: Truncated kernel normalization (has boundary spikes)
    - BETA: Beta kernel (Chen 1999) - smooth, optimal MSE, recommended
    - CIC: Cloud-in-Cell deposition - exact mass conservation, O(h^2) accuracy

    Reference:
        Chen, S. X. (1999). "Beta kernel estimators for density functions."
        Computational Statistics & Data Analysis, 31(2), 131-145.
        Hockney & Eastwood (1988). "Computer Simulation Using Particles", Ch. 5.
    """

    STANDARD = "standard"  # Standard KDE, no boundary correction
    REFLECTION = "reflection"  # Reflection/ghost particle method (Schuster 1985)
    RENORMALIZATION = "renormalization"  # Kernel renormalization method
    BETA = "beta"  # Beta kernel (Chen 1999) - recommended for bounded domains
    CIC = "cic"  # Cloud-in-Cell: exact mass conservation (Issue #718)


class FPParticleSolver(BaseFPSolver):
    """
    Particle-based Fokker-Planck solver using Monte Carlo sampling and KDE.

    This solver samples particles from the initial distribution, evolves them
    using SDE dynamics, and reconstructs the density on a grid using KDE.

    For meshfree density evolution on collocation points, use FPGFDMSolver instead.

    Density Modes (Issue #489 - Direct Particle Query):
        - "grid_only" (default): Store only grid density M (backward compatible)
        - "hybrid": Store both grid density M and particle positions for direct queries
        - "query_only": Store only particles (grid density computed on-demand)

        Use hybrid/query_only modes to enable efficient density queries at arbitrary
        points, providing 10-100× speedup for Semi-Lagrangian HJB coupling.

    Boundary Conditions:
        FPParticleSolver requires explicit boundary conditions. Provide via:
        1. boundary_conditions parameter (direct), OR
        2. problem.geometry.get_boundary_conditions() (from geometry)

        No default fallback - explicit BCs required for correctness.
        The solver will fail fast with a clear error message if BCs are missing.

    Composition Pattern (Issue #545):
        This solver uses composition instead of mixins:
        - self._applicator = ParticleApplicator() for BC application
        - self.geometry = problem.geometry for domain information
        - Explicit dependencies, no implicit state sharing

        Template for other solvers: See docs/development/PARTICLE_SOLVER_TEMPLATE.md
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.GENERIC  # Particle methods don't fit standard families

    @deprecated_parameter(param_name="mode", since="v0.17.0", replacement="density_mode")
    @deprecated_parameter(param_name="external_particles", since="v0.17.0", replacement="num_particles")
    @deprecated_parameter(param_name="normalize_kde_output", since="v0.17.0", replacement="kde_normalization")
    @deprecated_parameter(param_name="normalize_only_initial", since="v0.17.0", replacement="kde_normalization")
    def __init__(
        self,
        problem: MFGProblem,
        num_particles: int = 5000,
        kde_bandwidth: Any = "scott",
        kde_normalization: KDENormalization | str = KDENormalization.ALL,
        kde_method: KDEMethod | str = KDEMethod.REFLECTION,
        kde_boundary_smoothing: bool = True,
        density_mode: Literal["grid_only", "hybrid", "query_only"] = "grid_only",
        boundary_conditions: BoundaryConditions | None = None,
        implicit_domain: ImplicitDomain | None = None,
        backend: str | None = None,
        preserve_indices: bool = False,
        # Deprecated parameters (backward compatibility)
        mode: str | None = None,
        external_particles: Any = None,
        normalize_kde_output: bool | None = None,
        normalize_only_initial: bool | None = None,
    ) -> None:
        super().__init__(problem)

        # Handle deprecated 'mode' parameter
        if mode is not None:
            if mode == "collocation":
                raise ValueError(
                    "Collocation mode has been removed from FPParticleSolver. "
                    "Use FPGFDMSolver for GFDM-based FP solving."
                )
            elif mode not in ("hybrid", "grid_only", "query_only"):
                raise ValueError(f"Unknown mode '{mode}'. Valid modes: 'hybrid', 'grid_only', 'query_only'")
            # Map old 'mode' to new 'density_mode' (hybrid was the old name)
            density_mode = mode

        # Handle deprecated normalization parameters
        if normalize_kde_output is not None or normalize_only_initial is not None:
            # Map old parameters to new enum
            if normalize_kde_output is False:
                kde_normalization = KDENormalization.NONE
            elif normalize_only_initial is True:
                kde_normalization = KDENormalization.INITIAL_ONLY
            else:
                kde_normalization = KDENormalization.ALL

        self.num_particles = num_particles
        self.fp_method_name = "Particle"

        self.kde_bandwidth = kde_bandwidth
        self.kde_boundary_smoothing = kde_boundary_smoothing

        # Convert string to enum if needed
        if isinstance(kde_method, str):
            kde_method = KDEMethod(kde_method)
        self.kde_method = kde_method

        # Convert string to enum if needed
        if isinstance(kde_normalization, str):
            kde_normalization = KDENormalization(kde_normalization)
        self.kde_normalization = kde_normalization
        self.M_particles_trajectory: np.ndarray | None = None
        self._time_step_counter = 0  # Track current time step for normalization logic

        # Density mode for direct queries (Issue #489 Phase 2)
        self.density_mode = density_mode
        self._particle_history: list[np.ndarray] | None = None  # Stored if density_mode != "grid_only"

        # Segment-aware BC applicator (Pattern A: solver owns applicator)
        self._applicator = ParticleApplicator()

        # Exit flux tracking for absorbing BC analysis
        self.exit_flux_history: list[int] = []  # Number absorbed per timestep
        self.exit_positions_history: list[np.ndarray] = []  # Where particles exited
        self.total_absorbed: int = 0  # Cumulative absorbed count

        # Implicit domain for obstacle handling (Issue #533)
        # When set, particles entering obstacles are reflected back
        self._implicit_domain = implicit_domain

        # Issue #1119: preserve_indices flag for particle ID tracking
        # When True, absorbed particles are NaN-marked instead of compact-removed,
        # so particle_history[t] has constant shape (num_particles, d) and original
        # indices are preserved across timesteps. Currently supported only in the
        # callable-drift n-D path with segment-aware BC (other paths raise).
        self._preserve_indices: bool = preserve_indices

        # Initialize backend (defaults to NumPy)
        from mfgarchon.backends import create_backend

        if backend is not None:
            self.backend = create_backend(backend)
        else:
            self.backend = create_backend("numpy")  # NumPy fallback

        # Initialize strategy selector for intelligent pipeline selection
        from mfgarchon.backends.strategies.strategy_selector import StrategySelector

        self.strategy_selector = StrategySelector(profiling_mode="silent")
        self.current_strategy = None  # Will be set in solve_fp_system

        # Boundary condition resolution hierarchy (Issue #545):
        # 1. Explicit boundary_conditions parameter (highest priority)
        # 2. Grid geometry boundary conditions (from geometry)
        # 3. Implicit geometry with periodic dimensions (e.g., Hyperrectangle torus)
        # 4. FAIL FAST - no silent fallback (CLAUDE.md principle)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            # Try geometry BC (use try/except, not hasattr - Issue #543)
            try:
                self.boundary_conditions = problem.geometry.get_boundary_conditions()
            except AttributeError as e:
                # Fail fast - no silent fallback to periodic (CLAUDE.md principle)
                raise ValueError(
                    "FPParticleSolver requires explicit boundary conditions. "
                    "Boundary conditions not provided via:\n"
                    "  1. boundary_conditions=... parameter, OR\n"
                    "  2. problem.geometry.get_boundary_conditions()\n\n"
                    f"Original error: {e}"
                ) from e

            # Validate we got BCs (geometry method might return None)
            if self.boundary_conditions is None:
                # Check for fully periodic implicit geometry (e.g., Hyperrectangle torus)
                # Fully periodic = all dimensions are periodic, no BC enforcement needed
                geom = problem.geometry
                periodic_dims = getattr(geom, "periodic_dimensions", ())
                dimension = getattr(geom, "dimension", None)

                if periodic_dims and dimension and len(periodic_dims) == dimension:
                    # Fully periodic domain (torus) - particles wrap, no BC needed
                    # Use "periodic" as a sentinel for BC application logic
                    self.boundary_conditions = "periodic"
                else:
                    raise ValueError(
                        "FPParticleSolver requires boundary conditions. "
                        "problem.geometry.get_boundary_conditions() returned None. "
                        "For implicit geometry, either:\n"
                        "  1. Pass explicit boundary_conditions parameter, OR\n"
                        "  2. Use fully periodic geometry (all dims in periodic_dimensions)"
                    )

    def _get_grid_params(self) -> dict:
        """
        Extract grid parameters from geometry (preferred) or legacy problem API.

        Returns dict with nD-aware parameters:
            - dimension: int, spatial dimension (1, 2, 3, ...)
            - grid_shape: tuple, shape of spatial grid (Nx+1,) or (Nx+1, Ny+1, ...)
            - spacings: list[float], grid spacing per dimension [Dx, Dy, ...]
            - bounds: list[tuple], bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
            - coordinates: list[np.ndarray], 1D coordinate arrays per dimension
            - total_points: int, total number of grid points
            - Nt, Dt, sigma, coupling_coefficient: time/physics parameters

        For 1D backward compatibility, also includes:
            - Nx, Dx, xmin, xmax, Lx, xSpace (aliased from nD params)

        Note: For implicit geometries (e.g., Hyperrectangle) used with initial_particles
        and density_mode="query_only", grid_shape may be estimated and coordinates
        generated from bounds. This is acceptable for meshfree workflows.
        """
        # Try geometry-first API (Issue #543: use try/except instead of hasattr)
        try:
            geom = self.problem.geometry
            if geom is None:
                raise AttributeError("geometry is None")

            # Get dimension first - needed for all paths
            dimension = getattr(geom, "dimension", None)
            if dimension is None:
                # Try to infer from grid_shape
                grid_shape = tuple(geom.get_grid_shape())
                dimension = len(grid_shape)
            else:
                grid_shape = tuple(geom.get_grid_shape())
                # For implicit geometry, get_grid_shape returns (N,) even for nD
                # Check if this is an implicit domain by comparing dimensions
                if len(grid_shape) == 1 and dimension > 1:
                    # Implicit geometry: create synthetic grid shape from dimension
                    # Use sqrt(N) per dimension for 2D, cbrt(N) for 3D, etc.
                    n_per_dim = round(grid_shape[0] ** (1.0 / dimension))
                    grid_shape = tuple([n_per_dim] * dimension)

            # Get bounds per dimension (with fallback chain for legacy interfaces)
            # NOTE: bounds needed before spacing for implicit geometry fallback
            try:
                geom_bounds = geom.bounds
                if geom_bounds is None:
                    raise AttributeError("bounds is None")
                # Handle Hyperrectangle: bounds is np.ndarray of shape (d, 2)
                # Convert to list of tuples: [(xmin, xmax), (ymin, ymax), ...]
                if isinstance(geom_bounds, np.ndarray) and geom_bounds.ndim == 2:
                    # Numpy array format: bounds[d, 0] = min, bounds[d, 1] = max
                    bounds = [(float(geom_bounds[d, 0]), float(geom_bounds[d, 1])) for d in range(geom_bounds.shape[0])]
                else:
                    # List/tuple format - convert to list of tuples
                    bounds = [(float(b[0]), float(b[1])) for b in geom_bounds]
            except AttributeError:
                try:
                    # Legacy 1D geometry
                    bounds = [(geom.xmin, geom.xmax)]
                except AttributeError:
                    try:
                        # Infer from coordinates
                        if len(geom.coordinates) > 0:
                            bounds = [(coords[0], coords[-1]) for coords in geom.coordinates]
                        else:
                            # Issue #1053: fail-fast instead of silent unit-cube
                            # fallback. Empty coordinates on a non-Hyperrectangle
                            # geometry means we genuinely cannot derive bounds.
                            raise TypeError(
                                f"FPParticleSolver._get_grid_params: cannot derive simulation bounds "
                                f"from {type(geom).__name__} (geom.bounds, geom.xmin/.xmax, and "
                                f"geom.coordinates all absent or empty). Implement one of these on "
                                f"your geometry class, or wrap with a Hyperrectangle bbox."
                            )
                    except AttributeError as _e:
                        # Issue #1053: same fail-fast for the AttributeError branch.
                        raise TypeError(
                            f"FPParticleSolver._get_grid_params: cannot derive simulation bounds "
                            f"from {type(geom).__name__} (tried geom.bounds, geom.xmin/.xmax, "
                            f"geom.coordinates — all raised AttributeError). Implement one of "
                            f"these on your geometry class, or wrap with a Hyperrectangle bbox."
                        ) from _e

            # Get spacing per dimension
            # For implicit geometry (e.g., Hyperrectangle), get_grid_spacing() returns None
            # so compute from bounds and grid_shape
            spacing = geom.get_grid_spacing()
            if spacing is not None:
                spacings = list(spacing)
            else:
                # Compute spacing from bounds and grid_shape: dx = (xmax - xmin) / (N - 1)
                spacings = [(bounds[d][1] - bounds[d][0]) / max(grid_shape[d] - 1, 1) for d in range(dimension)]

            # Get coordinate arrays per dimension
            try:
                if len(geom.coordinates) > 0:
                    coordinates = [np.array(c) for c in geom.coordinates]
                else:
                    coordinates = [np.linspace(bounds[d][0], bounds[d][1], grid_shape[d]) for d in range(dimension)]
            except AttributeError:
                coordinates = [np.linspace(bounds[d][0], bounds[d][1], grid_shape[d]) for d in range(dimension)]

        except AttributeError as e:
            # Geometry is always available after MFGProblem initialization
            raise ValueError(
                "FPParticleSolver requires a geometry object. "
                "Create MFGProblem with geometry=TensorProductGrid(...) or with Nx=... parameter."
            ) from e

        # Compute derived quantities
        total_points = int(np.prod(grid_shape))
        domain_lengths = [b[1] - b[0] for b in bounds]

        # Time parameters (always from problem)
        # n_time_points = problem.Nt + 1 (number of time knots including t=0 and t=T)
        # problem.Nt = number of time intervals
        n_time_points = self.problem.Nt + 1
        Dt = (
            self.problem.dt
            if self.problem.dt is not None
            else (self.problem.T / self.problem.Nt if self.problem.Nt > 0 else 0.0)
        )
        sigma = self.problem.sigma if self.problem.sigma is not None else 0.1
        coupling_coefficient = (
            self.problem.coupling_coefficient if self.problem.coupling_coefficient is not None else 1.0
        )

        result = {
            # nD parameters
            "dimension": dimension,
            "grid_shape": grid_shape,
            "spacings": spacings,
            "bounds": bounds,
            "coordinates": coordinates,
            "total_points": total_points,
            "domain_lengths": domain_lengths,
            # Time/physics parameters
            "n_time_points": n_time_points,  # Nt + 1 (number of knots)
            "Nt": n_time_points,  # Backward compatible alias (deprecated)
            "Dt": Dt,
            "sigma": sigma,
            "coupling_coefficient": coupling_coefficient,
        }

        # 1D backward compatibility aliases
        if dimension == 1:
            result["Nx"] = grid_shape[0]
            result["Dx"] = spacings[0]
            result["xmin"] = bounds[0][0]
            result["xmax"] = bounds[0][1]
            result["Lx"] = domain_lengths[0]
            result["xSpace"] = coordinates[0]

        return result

    def _compute_gradient_nd(
        self,
        U_array: np.ndarray,
        spacings: list[float],
        use_backend: bool = False,
    ) -> list:
        """
        Compute spatial gradient using stencils.

        Uses gradient_nd (central differences, no BC handling).
        BC handling for particles is done separately in _apply_boundary_conditions_nd.

        Issue #625: Migrated from tensor_calculus.gradient_simple to stencils.gradient_nd
        """
        # Get array module (numpy or backend-specific like cupy)
        xp = self.backend.array_module if use_backend and self.backend else np
        return gradient_nd(U_array, spacings, xp=xp)

    # =========================================================================
    # DRY Helper Methods (Issue #635 cleanup)
    # =========================================================================

    def _should_normalize_density(self) -> bool:
        """
        Determine if density should be normalized based on KDE normalization strategy.

        Returns True if normalization should be applied for the current time step.
        """
        if self.kde_normalization == KDENormalization.NONE:
            return False
        elif self.kde_normalization == KDENormalization.INITIAL_ONLY:
            return self._time_step_counter == 0
        return True  # KDENormalization.ALL

    def _increment_time_step(self) -> None:
        """Increment time step counter for KDE normalization strategy tracking."""
        self._time_step_counter += 1

    def _apply_kde_method_1d(
        self,
        particles: np.ndarray,
        eval_points: np.ndarray,
        xmin: float,
        xmax: float,
    ) -> np.ndarray:
        """
        Apply selected KDE method for 1D density estimation (Issue #709).

        Args:
            particles: Particle positions, shape (N,)
            eval_points: Grid points for density evaluation, shape (M,)
            xmin: Left domain boundary
            xmax: Right domain boundary

        Returns:
            Density estimates at eval_points, shape (M,)
        """
        from mfgarchon.utils.numerical.particle.kde_boundary import (
            beta_kde,
            reflection_kde,
            renormalization_kde,
        )

        bounds_1d = [(xmin, xmax)]
        bounds_tuple = (xmin, xmax)

        if self.kde_method == KDEMethod.BETA:
            return beta_kde(
                particles,
                eval_points,
                bandwidth=self.kde_bandwidth,
                bounds=bounds_tuple,
            )
        elif self.kde_method == KDEMethod.REFLECTION:
            density = reflection_kde(
                particles,
                eval_points,
                bandwidth=self.kde_bandwidth,
                bounds=bounds_1d,
            )
            # Boundary smoothing: average boundary and adjacent points
            # to fix the reflection KDE redistribution issue (Issue #709)
            if self.kde_boundary_smoothing and len(density) >= 2:
                # Left boundary: average density[0] and density[1]
                avg_left = 0.5 * (density[0] + density[1])
                density[0] = avg_left
                density[1] = avg_left
                # Right boundary: average density[-1] and density[-2]
                avg_right = 0.5 * (density[-1] + density[-2])
                density[-1] = avg_right
                density[-2] = avg_right
            return density
        elif self.kde_method == KDEMethod.RENORMALIZATION:
            return renormalization_kde(
                particles,
                eval_points,
                bandwidth=self.kde_bandwidth,
                bounds=bounds_1d,
            )
        elif self.kde_method == KDEMethod.CIC:
            from mfgarchon.utils.numerical.particle.cic import cic_deposit_nd

            bounds_arr = np.array([[xmin, xmax]])
            n_grid = len(eval_points)
            density_grid = cic_deposit_nd(particles.reshape(-1, 1), bounds_arr, (n_grid,), periodic=False)
            return density_grid.ravel()
        else:  # KDEMethod.STANDARD
            kde = gaussian_kde(particles, bw_method=self.kde_bandwidth)
            density = kde(eval_points)
            density[eval_points < xmin] = 0
            density[eval_points > xmax] = 0
            return density

    def _apply_kde_method_nd(
        self,
        particles: np.ndarray,
        grid_points: np.ndarray,
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Apply selected KDE method for nD density estimation (Issue #709).

        Args:
            particles: Particle positions, shape (N, d)
            grid_points: Grid points for density evaluation, shape (M, d)
            bounds: Domain bounds [(xmin, xmax), (ymin, ymax), ...]

        Returns:
            Density estimates at grid_points, shape (M,)
        """
        from mfgarchon.utils.numerical.particle.kde_boundary import (
            beta_kde,
            reflection_kde,
            renormalization_kde,
        )

        if self.kde_method == KDEMethod.BETA:
            return beta_kde(
                particles,
                grid_points,
                bandwidth=self.kde_bandwidth,
                bounds=bounds,
            )
        elif self.kde_method == KDEMethod.REFLECTION:
            density = reflection_kde(
                particles,
                grid_points,
                bandwidth=self.kde_bandwidth,
                bounds=bounds,
            )
            # Note: nD boundary smoothing is complex (edges, corners)
            # For now, smoothing is only applied in 1D via _apply_kde_method_1d
            # Future work: implement face/edge/corner smoothing for nD
            return density
        elif self.kde_method == KDEMethod.RENORMALIZATION:
            return renormalization_kde(
                particles,
                grid_points,
                bandwidth=self.kde_bandwidth,
                bounds=bounds,
            )
        elif self.kde_method == KDEMethod.CIC:
            from mfgarchon.utils.numerical.particle.cic import cic_deposit_nd

            bounds_arr = np.array(bounds)
            ndim = len(bounds)
            grid_shape = tuple(len(np.unique(grid_points[:, d])) for d in range(ndim))
            # CIC uses reflecting (clipped) BC for bounded domains
            density_grid = cic_deposit_nd(particles, bounds_arr, grid_shape, periodic=False)
            return density_grid.ravel()
        else:  # KDEMethod.STANDARD
            kde = gaussian_kde(particles.T, bw_method=self.kde_bandwidth)
            density = kde(grid_points.T)
            # Clip outside domain to zero
            for d in range(len(bounds)):
                xmin, xmax = bounds[d]
                outside_mask = (grid_points[:, d] < xmin) | (grid_points[:, d] > xmax)
                density[outside_mask] = 0
            return density

    def _compute_total_mass(
        self,
        density: np.ndarray,
        spacing: float | list[float],
        use_backend: bool = False,
    ) -> float:
        """
        Compute total mass from density and grid spacing(s).

        Parameters
        ----------
        density : np.ndarray
            Density array (1D or nD)
        spacing : float or list[float]
            Grid spacing (single value for 1D, list for nD)
        use_backend : bool
            If True, use backend array module

        Returns
        -------
        float
            Total mass (integral of density over domain)
        """
        # Compute volume element
        if isinstance(spacing, (list, tuple)):
            dV = float(np.prod(spacing))
        else:
            dV = float(spacing) if spacing > 1e-14 else 1.0

        if use_backend and self.backend is not None:
            xp = self.backend.array_module
            mass = xp.sum(density) * dV
            try:
                return mass.item()  # PyTorch tensor → scalar
            except AttributeError:
                return float(mass)
        else:
            return float(np.sum(density) * dV)

    def _finalize_particle_solve(
        self,
        M_density_on_grid: np.ndarray,
        particles_trajectory: np.ndarray | list,
        Nt: int,
        dimension: int,
        use_segment_aware_bc: bool = False,
    ) -> np.ndarray | FPParticleResult:
        """
        Finalize particle solve: store trajectory, build history, return result.

        Extracts common post-loop logic from all solve methods.

        Parameters
        ----------
        M_density_on_grid : np.ndarray
            Density evolution on grid
        particles_trajectory : np.ndarray or list
            Particle positions over time (array for uniform BC, list for segment-aware)
        Nt : int
            Number of time points in trajectory
        dimension : int
            Spatial dimension
        use_segment_aware_bc : bool
            Whether segment-aware BC was used (affects trajectory format)

        Returns
        -------
        np.ndarray or FPParticleResult
            Grid density or FPParticleResult with particle history
        """
        # Store trajectory
        self.M_particles_trajectory = particles_trajectory

        # Build particle history for direct query mode (Issue #489)
        if self._particle_history is not None:
            if use_segment_aware_bc:
                # List of arrays with variable particle counts
                for t_particles in particles_trajectory:
                    if t_particles.ndim == 1:
                        self._particle_history.append(t_particles.reshape(-1, 1))
                    else:
                        self._particle_history.append(t_particles)
            elif dimension == 1 and particles_trajectory.ndim == 2:
                # 1D: 2D array (Nt, num_particles) -> List of (num_particles, 1)
                for t in range(Nt):
                    self._particle_history.append(particles_trajectory[t, :].reshape(-1, 1))
            else:
                # nD: 3D array (Nt, num_particles, dimension) -> List of (num_particles, dimension)
                for t in range(Nt):
                    self._particle_history.append(particles_trajectory[t, :, :])

        # Return FPParticleResult if particle history was stored (Issue #489)
        if self._particle_history is not None:
            time_grid = np.linspace(0, self.problem.T, Nt)
            return FPParticleResult(
                M_grid=M_density_on_grid,
                time_grid=time_grid,
                particle_history=self._particle_history,
                bandwidth=self.kde_bandwidth if isinstance(self.kde_bandwidth, (int, float)) else None,
            )

        # Backward compatible: return grid density only
        return M_density_on_grid

    def _create_timestep_range(self, n_steps: int, desc: str = "FP Particle"):
        """
        Create timestep iterator with optional progress bar.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        desc : str
            Progress bar description

        Returns
        -------
        Iterable[int]
            Iterator over timestep indices
        """
        from mfgarchon.utils.progress import create_progress_bar, should_show_progress

        return create_progress_bar(
            range(n_steps),
            verbose=should_show_progress(self._show_progress),
            desc=desc,
        )

    def _normalize_density(self, M_array, Dx: float, use_backend: bool = False):
        """
        Normalize density to unit mass.

        Backend-agnostic helper to reduce code duplication between CPU and GPU pipelines.
        Respects kde_normalization strategy: NONE, INITIAL_ONLY, or ALL.

        Parameters
        ----------
        M_array : np.ndarray or backend tensor
            Density array
        Dx : float
            Grid spacing
        use_backend : bool
            If True, use backend array module; if False, use NumPy

        Returns
        -------
        Normalized density array (same type as input)
        """
        if not self._should_normalize_density():
            return M_array  # Return raw density without normalization

        # Compute total mass using helper
        mass_val = self._compute_total_mass(M_array, Dx, use_backend)

        if mass_val > 1e-9:
            return M_array / mass_val
        else:
            return M_array * 0  # Return zeros

    # =========================================================================
    # nD Helper Methods
    # =========================================================================

    def _sample_particles_from_density_nd(
        self,
        M_initial: np.ndarray,
        coordinates: list[np.ndarray],
        num_particles: int,
    ) -> np.ndarray:
        """
        Sample particles from nD density distribution.

        Delegates to utils.numerical.particle.sample_from_density() for the
        actual sampling. This method provides a consistent interface within
        the solver class.

        Parameters
        ----------
        M_initial : np.ndarray
            Initial density on grid, shape (N1, N2, ..., Nd)
        coordinates : list[np.ndarray]
            List of 1D coordinate arrays per dimension
        num_particles : int
            Number of particles to sample

        Returns
        -------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        """
        return sample_from_density(
            density=M_initial,
            coordinates=coordinates,
            num_samples=num_particles,
            jitter=True,
            seed=None,  # Use global numpy random state for reproducibility with solver
        )

    def _generate_brownian_increment_nd(
        self,
        num_particles: int,
        dimension: int,
        Dt: float,
        sigma: float,
    ) -> np.ndarray:
        """
        Generate d-dimensional Brownian increment for SDE evolution.

        Delegates to unified generate_brownian_increment() from fp_particle_density module.

        Parameters
        ----------
        num_particles : int
            Number of particles
        dimension : int
            Spatial dimension
        Dt : float
            Time step size
        sigma : float
            Diffusion coefficient

        Returns
        -------
        dW : np.ndarray
            Brownian increments, shape (num_particles, dimension)
        """
        result = _gen_brownian(num_particles, dimension, Dt, sigma)
        # Ensure 2D output for backward compatibility (this method always returns 2D)
        if result.ndim == 1:
            return result[:, np.newaxis]
        return result

    def _needs_segment_aware_bc(self) -> bool:
        """
        Check if boundary conditions require segment-aware handling.

        Delegates to unified needs_segment_aware_bc() from fp_particle_bc module.

        Returns True if:
        - BC has multiple segments with different types
        - BC has absorbing (DIRICHLET) segments that should remove particles

        Returns False for uniform periodic/reflecting BC where fast path can be used.
        """
        return _needs_segment_bc(self.boundary_conditions)

    def _get_topology_per_dimension(
        self,
        dimension: int,
    ) -> list[str]:
        """
        Get grid topology for each dimension from boundary conditions.

        Delegates to unified get_topology_per_dimension() from fp_particle_bc module.

        This determines the INDEXING STRATEGY for particles, not the physical BC:
        - "periodic": Space wraps around (particles use modular arithmetic)
        - "bounded": Space has walls (particles reflect at boundaries)

        Parameters
        ----------
        dimension : int
            Number of spatial dimensions

        Returns
        -------
        topologies : list[str]
            Topology per dimension: ["periodic", "bounded", ...]
        """
        return _get_topology(self.boundary_conditions, dimension)

    def _enforce_obstacle_boundary(self, particles: np.ndarray) -> np.ndarray:
        """
        Enforce obstacle boundaries via implicit domain geometry (Issue #533).

        Delegates to unified enforce_obstacle_boundary() from fp_particle_bc module.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)

        Returns
        -------
        particles : np.ndarray
            Updated particle positions with obstacle violations corrected
        """
        return _enforce_obstacle(particles, self._implicit_domain)

    def _infer_reflect_bounds(self, bounds: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
        """
        Issue #1083: infer per-axis bounds for KDE reflection from solver BC.

        Returns the subset of `bounds` for axes whose BC is reflective
        (NO_FLUX / REFLECTING / NEUMANN). Returns None if no axis is reflective
        — caller falls back to standard KDE without ghost reflection.

        For axes with non-reflective BC (DIRICHLET absorbing exit, periodic),
        reflection ghosts are mathematically wrong, so they are excluded.
        """
        bc = self.boundary_conditions
        if bc is None or not getattr(bc, "segments", None):
            # No explicit BC: assume reflecting on all axes (legacy default)
            return list(bounds)

        # Check if any segment is reflective. The simplest correct heuristic
        # for the typical case (uniform reflecting walls + optional Dirichlet
        # exit): if at least one segment is reflective, pass bounds for all
        # axes. Per-axis disambiguation needs the BC framework's face-resolution
        # which is non-trivial; defer until segment-axis mapping is exposed.
        from mfgarchon.geometry.boundary import BCType

        reflective_types = {BCType.NO_FLUX, BCType.REFLECTING, BCType.NEUMANN}
        has_reflective = any(getattr(seg, "bc_type", None) in reflective_types for seg in bc.segments)
        if has_reflective:
            return list(bounds)
        return None

    def _apply_boundary_conditions_nd(
        self,
        particles: np.ndarray,
        bounds: list[tuple[float, float]],
        topology: str | list[str],
    ) -> np.ndarray:
        """
        Apply boundary handling per dimension based on topology.

        Delegates to unified apply_boundary_conditions() from fp_particle_bc module.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        bounds : list[tuple[float, float]]
            Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
        topology : str or list[str]
            Grid topology: "periodic" (wrap) or "bounded" (reflect).
            Can be a single string (same for all dims) or per-dimension list.

        Returns
        -------
        particles : np.ndarray
            Updated particle positions
        """
        return _apply_bc(particles, bounds, topology)

    def _apply_boundary_conditions_segment_aware(
        self,
        particles: np.ndarray,
        bounds: list[tuple[float, float]],
    ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Apply segment-aware boundary conditions using the applicator.

        This method handles mixed BC where different boundaries have different types:
        - DIRICHLET segments: Absorb particles (remove from simulation)
        - REFLECTING/NO_FLUX segments: Bounce particles back
        - PERIODIC segments: Wrap particles

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        bounds : list[tuple[float, float]]
            Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]

        Returns
        -------
        remaining_particles : np.ndarray
            Particles that were not absorbed, shape (M, dimension)
        n_absorbed : int
            Number of particles absorbed this step
        exit_positions : np.ndarray
            Positions where particles exited, shape (K, dimension)
        absorbed_mask : np.ndarray
            Boolean mask of absorbed particles in the INPUT array, shape (N,).
            Issue #1119: exposed for preserve_indices bookkeeping.
        """
        remaining, absorbed_mask, exit_positions = self._applicator.apply(
            particles,
            self.boundary_conditions,
            bounds,
        )

        n_absorbed = int(np.sum(absorbed_mask))

        # Update cumulative tracking
        if n_absorbed > 0:
            self.exit_flux_history.append(n_absorbed)
            self.exit_positions_history.append(exit_positions)
            self.total_absorbed += n_absorbed

        return remaining, n_absorbed, exit_positions, absorbed_mask

    def _apply_boundary_conditions_with_flux_limits(
        self,
        particles: np.ndarray,
        bounds: list[tuple[float, float]],
        flux_limits: dict[str, float],
    ) -> tuple[np.ndarray, int, np.ndarray, dict[str, int]]:
        """
        Apply segment-aware BC with flux-limited absorption.

        Particles at DIRICHLET exits are absorbed only up to the flux capacity.
        When capacity is exceeded, particles are REFLECTED (creating queues).

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        bounds : list[tuple[float, float]]
            Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
        flux_limits : dict[str, float]
            Max particles to absorb per segment this step
            e.g., {"exit_A": 10, "exit_B": 15}

        Returns
        -------
        remaining_particles : np.ndarray
            Particles not absorbed
        n_absorbed : int
            Total particles absorbed this step
        exit_positions : np.ndarray
            Positions where absorbed
        absorbed_per_segment : dict[str, int]
            Absorbed count per segment
        """
        remaining, absorbed_mask, exit_positions, absorbed_per_segment = self._applicator.apply_with_flux_limits(
            particles,
            self.boundary_conditions,
            bounds,
            flux_limits,
        )

        n_absorbed = int(np.sum(absorbed_mask))

        if n_absorbed > 0:
            self.exit_flux_history.append(n_absorbed)
            self.exit_positions_history.append(exit_positions)
            self.total_absorbed += n_absorbed

        return remaining, n_absorbed, exit_positions, absorbed_per_segment

    def _interpolate_grid_to_particles_nd(
        self,
        grid_values: np.ndarray,
        bounds: list[tuple[float, float]],
        particles: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate grid values to particle positions.

        Delegates to utils.numerical.particle.interpolate_grid_to_particles().

        Parameters
        ----------
        grid_values : np.ndarray
            Values on grid, shape (N1, N2, ..., Nd)
        bounds : list[tuple[float, float]]
            Domain bounds per dimension
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)

        Returns
        -------
        values_at_particles : np.ndarray
            Interpolated values, shape (num_particles,)
        """
        # Convert bounds to format expected by utils: tuple of tuples
        grid_bounds = tuple(bounds)
        return interpolate_grid_to_particles(
            grid_values=grid_values,
            grid_bounds=grid_bounds,
            particle_positions=particles,
            method="linear",
        )

    def _estimate_density_from_particles_nd(
        self,
        particles: np.ndarray,
        coordinates: list[np.ndarray],
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Estimate density from particles using KDE on nD grid.

        Delegates to utils.numerical.particle.interpolate_particles_to_grid()
        for the core KDE computation, with additional edge case handling.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        coordinates : list[np.ndarray]
            List of 1D coordinate arrays per dimension
        bounds : list[tuple[float, float]]
            Bounds per dimension

        Returns
        -------
        density : np.ndarray
            Density on grid, shape (N1, N2, ..., Nd)
        """
        dimension = len(coordinates)
        grid_shape = tuple(len(c) for c in coordinates)

        # Edge case: no particles
        if self.num_particles == 0 or len(particles) == 0:
            return np.zeros(grid_shape)

        # Edge case: degenerate particle distribution (all at same location)
        if len(np.unique(particles, axis=0)) < 2:
            density = np.zeros(grid_shape)
            mean_pos = np.mean(particles, axis=0)
            indices = []
            for d in range(dimension):
                idx = np.argmin(np.abs(coordinates[d] - mean_pos[d]))
                indices.append(idx)
            density[tuple(indices)] = self.num_particles
            return density

        try:
            # Issue #709: Select KDE method for boundary correction
            # Create grid points for evaluation
            meshes = np.meshgrid(*coordinates, indexing="ij")
            grid_points = np.column_stack([m.ravel() for m in meshes])

            density_flat = self._apply_kde_method_nd(particles, grid_points, list(bounds))

            density_reshaped = density_flat.reshape(grid_shape)

            # Mass conservation diagnostic (Issue #718)
            spacings = [(bounds[d][1] - bounds[d][0]) / max(len(coordinates[d]) - 1, 1) for d in range(dimension)]
            dV = float(np.prod(spacings))
            raw_mass = float(np.sum(density_reshaped) * dV)
            if abs(raw_mass - 1.0) > 0.05:
                logger.warning(
                    "KDE mass deviation: %.1f%% (method=%s, raw_mass=%.4f). "
                    "Consider kde_method='cic' for exact mass conservation.",
                    (raw_mass - 1.0) * 100,
                    self.kde_method.value,
                    raw_mass,
                )

            return density_reshaped

        except Exception as e:
            warnings.warn(f"KDE failed in nD: {e}. Returning histogram estimate.")
            # Fallback to histogram
            density, _ = np.histogramdd(
                particles,
                bins=[len(c) for c in coordinates],
                range=bounds,
                density=True,
            )
            return density

    def _estimate_density_from_particles(self, particles_at_time_t: np.ndarray) -> np.ndarray:
        # Use geometry-aware parameter extraction
        params = self._get_grid_params()
        Nx = params["Nx"]
        xSpace = params["xSpace"]
        xmin = params["xmin"]
        xmax = params["xmax"]
        Dx = params["Dx"]

        if self.num_particles == 0 or len(particles_at_time_t) == 0:
            return np.zeros(Nx)

        unique_particles = np.unique(particles_at_time_t)
        if len(unique_particles) < 2 or np.std(particles_at_time_t) < 1e-9 * (xmax - xmin):
            m_density_estimated = np.zeros(Nx)
            if len(particles_at_time_t) > 0:
                mean_pos = np.mean(particles_at_time_t)
                closest_idx = np.argmin(np.abs(xSpace - mean_pos))
                if Dx > 1e-14:
                    m_density_estimated[closest_idx] = 1.0 / Dx
                elif Nx == 1:
                    m_density_estimated[closest_idx] = 1.0

            # Normalization logic will apply below if self.normalize_kde_output is True
        else:
            try:
                # Issue #709: Use CPU path with boundary-corrected KDE
                # GPU KDE doesn't support boundary correction, so always use CPU for 1D
                if SCIPY_AVAILABLE and gaussian_kde is not None:
                    m_density_estimated = self._apply_kde_method_1d(particles_at_time_t, xSpace, xmin, xmax)
                else:
                    raise RuntimeError("SciPy not available for KDE")

            except Exception as e:
                error_msg = (
                    f"KDE density estimation failed in FPParticleSolver: {e}\n"
                    f"Number of particles: {len(particles_at_time_t)}\n"
                    f"Grid size: {Nx}\n"
                    f"Bandwidth: {self.kde_bandwidth}\n"
                    "Possible causes:\n"
                    "  1. Too few particles for reliable KDE (need at least 10-20)\n"
                    "  2. Bandwidth selection failed (try fixed bandwidth like 0.1)\n"
                    "  3. Particles outside domain bounds\n"
                    "  4. GPU/SciPy library issues\n"
                    "Suggestions:\n"
                    "  - Increase number of particles (Np > 100 recommended)\n"
                    "  - Use fixed bandwidth: kde_bandwidth=0.1\n"
                    "  - Check particle initialization and drift/diffusion"
                )
                raise RuntimeError(error_msg) from e

        # Normalization step (conditional based on kde_normalization strategy)
        if self._should_normalize_density():
            current_mass = self._compute_total_mass(m_density_estimated, Dx)
            if current_mass > 1e-9:
                return m_density_estimated / current_mass
            else:
                return np.zeros(Nx)
        else:
            return m_density_estimated  # Return raw KDE output on grid

    @deprecated_parameter(param_name="m_initial_condition", since="v0.17.0", replacement="M_initial")
    @deprecated_parameter(param_name="diffusion_field", since="v0.17.0", replacement="volatility_field")
    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
        drift_is_precomputed: bool = False,
        initial_particles: np.ndarray | None = None,
        drift_needs_density: bool = True,
        # Deprecated parameter names for backward compatibility
        m_initial_condition: np.ndarray | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,  # DEPRECATED
        potential_field: np.ndarray | None = None,  # DEPRECATED: use drift_field
    ) -> np.ndarray:
        """
        Solve FP system using particle method with unified API.

        Uses KDE-based particle method: sample own particles, output to grid via KDE.
        Strategy Selection: Automatically selects CPU/GPU/Hybrid based on problem size.

        For meshfree density evolution on scattered points, use FPGFDMSolver instead.

        Args:
            M_initial: Initial density m0(x) on grid. Required unless initial_particles provided.
            m_initial_condition: DEPRECATED, use M_initial
            drift_field: Drift field specification (optional):
                - None: Zero drift (pure diffusion)
                - np.ndarray: If drift_is_precomputed=False (default), this is U(t,x) and gradient will be computed.
                             If drift_is_precomputed=True, this is α(t,x,d) vector field (Nt, *grid_shape, d).
                - Callable: Function alpha(t, x, m) -> drift (Phase 2)
            drift_is_precomputed: If True, drift_field is treated as precomputed drift vector α(t,x).
                                  If False (default), drift_field is treated as value function U(t,x) and
                                  drift is computed as α = -coupling_coefficient * ∇U.
                                  Use True to preserve high-precision gradients from GFDM or other meshfree methods.
            volatility_field: Volatility specification for SDE noise (optional):
                - None: Use problem.sigma (backward compatible)
                - float: Constant isotropic volatility σ (SDE: dX = v dt + σ dW)
                - np.ndarray (d,d): Anisotropic noise matrix Σ (SDE: dX = v dt + Σ dW)
                - Callable: State-dependent σ(t,x,m) or Σ(t,x,m)
                Note: This is the SDE noise coefficient, NOT the PDE diffusion D = σ²/2.
            diffusion_field: DEPRECATED, use volatility_field instead.
            initial_particles: Pre-sampled initial particles, shape (num_particles, dimension).
                              If provided, M_initial is not required (meshfree initialization).
                              Useful with density_mode="query_only" for fully meshfree workflow.
            drift_needs_density: If False, skip density estimation at particle positions during
                                 time evolution. Use when drift_callable doesn't depend on m.
                                 Default True for backward compatibility.
            show_progress: Display progress bar

        Returns:
            M_solution: Density evolution on grid, shape (Nt+1, *grid_shape)
        """
        # Handle deprecated parameter name
        if m_initial_condition is not None:
            if M_initial is not None:
                raise ValueError(
                    "Cannot specify both M_initial and m_initial_condition. "
                    "Use M_initial (m_initial_condition is deprecated)."
                )
            M_initial = m_initial_condition

        # Handle deprecated diffusion_field parameter
        if diffusion_field is not None:
            if volatility_field is not None:
                raise ValueError(
                    "Cannot specify both volatility_field and diffusion_field. "
                    "Use volatility_field (diffusion_field is deprecated)."
                )
            volatility_field = diffusion_field

        # Handle deprecated potential_field -> drift_field
        if potential_field is not None:
            if drift_field is not None:
                raise ValueError(
                    "Cannot specify both drift_field and potential_field. "
                    "Use drift_field (potential_field is deprecated)."
                )
            drift_field = potential_field

        # Validate required parameter - either M_initial or initial_particles
        if M_initial is None and initial_particles is None:
            raise ValueError("Either M_initial or initial_particles is required")

        # Handle drift_field parameter
        if drift_field is None:
            # Zero drift (pure diffusion): create zero U field for internal use
            params = self._get_grid_params()
            Nt = params["Nt"]
            grid_shape = params["grid_shape"]
            effective_U = np.zeros((Nt, *grid_shape))
        elif isinstance(drift_field, np.ndarray):
            # Precomputed drift field (including MFG drift = -∇U/λ)
            effective_U = drift_field
        elif callable(drift_field):
            # Custom drift function - Phase 2
            # Route to callable drift solver
            return self._solve_fp_system_callable_drift(
                M_initial=M_initial,
                drift_callable=drift_field,
                volatility_field=volatility_field,
                show_progress=show_progress,
                initial_particles=initial_particles,
                drift_needs_density=drift_needs_density,
            )
        else:
            raise TypeError(f"drift_field must be None, np.ndarray, or Callable, got {type(drift_field)}")

        # Handle volatility_field parameter (SDE noise coefficient σ or Σ)
        if volatility_field is None:
            # Use problem.sigma (backward compatible)
            effective_sigma = self.problem.sigma
        elif isinstance(volatility_field, (int, float)):
            # Constant isotropic volatility σ
            effective_sigma = float(volatility_field)
        elif isinstance(volatility_field, np.ndarray) or callable(volatility_field):
            # Spatially varying or state-dependent volatility
            # Route to callable drift solver which supports this
            # Note: If drift_field is None, we need to handle pure diffusion case
            if drift_field is None:
                # Pure diffusion with spatially varying coefficient
                # Use callable drift path with zero drift
                def zero_drift(t, x, m):
                    if isinstance(x, np.ndarray) and x.ndim > 1:
                        return np.zeros_like(x)
                    return np.zeros_like(np.atleast_1d(x))

                return self._solve_fp_system_callable_drift(
                    M_initial=M_initial,
                    drift_callable=zero_drift,
                    volatility_field=volatility_field,
                    show_progress=show_progress,
                )
            else:
                # Already routed to callable drift above if drift is callable
                # This handles array drift + varying volatility
                effective_sigma = self.problem.sigma  # Fallback, actual handled in solver
        else:
            raise TypeError(
                f"volatility_field must be None, float, np.ndarray, or Callable, got {type(volatility_field)}"
            )

        # Temporarily override problem.sigma if custom volatility provided
        original_sigma = self.problem.sigma
        if volatility_field is not None:
            self.problem.sigma = effective_sigma

        # Reset time step counter for normalization logic
        self._time_step_counter = 0

        # Store show_progress and drift_is_precomputed for use in methods
        self._show_progress = show_progress
        self._drift_is_precomputed = drift_is_precomputed

        # Initialize particle history for direct query modes (Issue #489).
        # Issue #1119: NB this is the n-D grid-drift entry; preserve_indices is
        # NOT supported here (the segment-aware callsite raises NotImplementedError).
        if self.density_mode in ("hybrid", "query_only"):
            self._particle_history = []

        try:
            # Hybrid mode: particles -> grid (with strategy selection)
            # Determine problem size for strategy selection
            params = self._get_grid_params()
            Nt = params["Nt"]
            dimension = params["dimension"]
            grid_shape = params["grid_shape"]
            total_points = params["total_points"]
            problem_size = (self.num_particles, total_points, Nt)

            # Route based on dimension
            if dimension == 1:
                # 1D: Use existing optimized solvers with strategy selection
                self.current_strategy = self.strategy_selector.select_strategy(
                    backend=self.backend if (self.backend is not None and self.backend.name != "numpy") else None,
                    problem_size=problem_size,
                    strategy_hint="auto",
                )

                if self.current_strategy.name == "cpu":
                    return self._solve_fp_system_cpu(M_initial, effective_U)
                else:
                    return self._solve_fp_system_gpu(M_initial, effective_U)
            else:
                # nD (d >= 2): Use new nD CPU solver
                # GPU nD solver not yet implemented
                return self._solve_fp_system_cpu_nd(M_initial, effective_U)
        finally:
            # Restore original sigma
            self.problem.sigma = original_sigma

    def _solve_fp_system_cpu(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """CPU pipeline - existing NumPy implementation."""
        # Use geometry-aware parameter extraction
        params = self._get_grid_params()
        Nx = params["Nx"]
        Nt = params["Nt"]
        Dx = params["Dx"]
        Dt = params["Dt"]
        sigma = params["sigma"]
        coupling_coefficient = params["coupling_coefficient"]

        # SDE: dX = alpha*dt + sigma*dW
        # Convention: problem.sigma is the SDE noise coefficient directly
        sigma_sde = sigma
        x_grid = params["xSpace"]
        xmin = params["xmin"]
        Lx = params["Lx"]

        if Nt == 0:
            return np.zeros((0, Nx))

        # Check if segment-aware BC is needed (Issue #535 Phase 1)
        use_segment_aware_bc = self._needs_segment_aware_bc()

        # Reset exit flux tracking for this solve
        if use_segment_aware_bc:
            self.exit_flux_history = []
            self.exit_positions_history = []
            self.total_absorbed = 0

        M_density_on_grid = np.zeros((Nt, Nx))

        # For segment-aware BC with absorption, use list storage (variable particle count)
        # For uniform BC, use fixed array (all particles preserved)
        if use_segment_aware_bc:
            particles_list: list[np.ndarray] = [None] * Nt  # type: ignore
            current_M_particles_t = None  # Will be built at end
        else:
            current_M_particles_t = np.zeros((Nt, self.num_particles))
            particles_list = None

        # Sample initial particles
        if Dx > 1e-14 and np.sum(m_initial_condition * Dx) > 1e-9:
            m0_probs_unnormalized = m_initial_condition * Dx
            m0_probs = m0_probs_unnormalized / np.sum(m0_probs_unnormalized)
            try:
                initial_particle_positions = np.random.choice(x_grid, size=self.num_particles, p=m0_probs, replace=True)
            except ValueError:
                initial_particle_positions = np.random.uniform(xmin, xmin + Lx, self.num_particles)
        else:
            initial_particle_positions = (
                np.random.uniform(xmin, xmin + Lx, self.num_particles)
                if Lx > 1e-14
                else np.full(self.num_particles, xmin)
            )

        # Store initial particles
        if use_segment_aware_bc:
            particles_list[0] = initial_particle_positions
            init_particles = particles_list[0]
        else:
            current_M_particles_t[0, :] = initial_particle_positions
            init_particles = current_M_particles_t[0, :]

        M_density_on_grid[0, :] = self._estimate_density_from_particles(init_particles)
        self._increment_time_step()  # Increment after computing density at t=0

        if Nt == 1:
            if use_segment_aware_bc:
                self.M_particles_trajectory = particles_list
            else:
                self.M_particles_trajectory = current_M_particles_t
            return M_density_on_grid

        # Progress bar for forward particle timesteps (n_time_points - 1 steps)
        timestep_range = self._create_timestep_range(Nt - 1, desc="FP (forward)")

        for n_time_idx in timestep_range:
            # Get current particles from appropriate storage
            if use_segment_aware_bc:
                particles_t = particles_list[n_time_idx]
                n_particles_t = len(particles_t)
            else:
                particles_t = current_M_particles_t[n_time_idx, :]
                n_particles_t = self.num_particles

            # Skip if no particles remain (all absorbed)
            if n_particles_t == 0:
                if use_segment_aware_bc:
                    particles_list[n_time_idx + 1] = np.array([])
                M_density_on_grid[n_time_idx + 1, :] = np.zeros(Nx)
                self._increment_time_step()
                continue

            U_at_tn = U_solution_for_drift[n_time_idx, :]

            # Use shared nD gradient method (works for 1D too)
            if Nx > 1:
                gradients = self._compute_gradient_nd(U_at_tn, [Dx], use_backend=False)
                dUdx_grid = gradients[0]  # First (and only) dimension
            else:
                dUdx_grid = np.zeros(Nx)

            if Nx > 1:
                # Interpolate gradient to particle positions using utils
                particles_1d = particles_t.reshape(-1, 1)
                dUdx_at_particles = interpolate_grid_to_particles(
                    grid_values=dUdx_grid,
                    grid_bounds=(xmin, xmin + Lx),
                    particle_positions=particles_1d,
                    method="linear",
                )
            else:
                dUdx_at_particles = np.zeros(n_particles_t)

            alpha_optimal_at_particles = -coupling_coefficient * dUdx_at_particles

            # Generate Brownian motion for current particle count
            dW = np.random.normal(0.0, np.sqrt(Dt), n_particles_t) if Dt > 1e-14 else np.zeros(n_particles_t)

            # Euler-Maruyama update
            new_particles = particles_t + alpha_optimal_at_particles * Dt + sigma_sde * dW

            # Apply boundary conditions (segment-aware or topology-based)
            if use_segment_aware_bc:
                # Segment-aware BC: may absorb particles
                # Convert to 2D for applicator (expects shape (N, d))
                particles_2d = new_particles.reshape(-1, 1)
                if self._preserve_indices:
                    raise NotImplementedError(
                        "preserve_indices=True not supported in 1D path; use callable-drift n-D path (Issue #1119)."
                    )
                remaining_2d, _n_absorbed, _, _ = self._apply_boundary_conditions_segment_aware(
                    particles_2d, [(xmin, xmin + Lx)]
                )
                new_particles = remaining_2d[:, 0]  # Back to 1D
                particles_list[n_time_idx + 1] = new_particles
            else:
                # Uniform BC: topology-based (no absorption)
                particles_2d = new_particles.reshape(-1, 1)
                topologies = self._get_topology_per_dimension(1)
                particles_2d = self._apply_boundary_conditions_nd(particles_2d, [(xmin, xmin + Lx)], topologies)
                new_particles = particles_2d[:, 0]
                current_M_particles_t[n_time_idx + 1, :] = new_particles

            M_density_on_grid[n_time_idx + 1, :] = self._estimate_density_from_particles(new_particles)
            self._increment_time_step()  # Increment after each time step

        # Finalize: store trajectory, build history, return result
        trajectory = particles_list if use_segment_aware_bc else current_M_particles_t
        return self._finalize_particle_solve(
            M_density_on_grid, trajectory, Nt, dimension=1, use_segment_aware_bc=use_segment_aware_bc
        )

    def _solve_fp_system_cpu_nd(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """
        nD CPU pipeline - particle evolution for dimension >= 2.

        Uses the nD helper methods:
        - _sample_particles_from_density_nd() for initial sampling
        - _compute_gradient_nd() for gradient computation
        - _interpolate_grid_to_particles_nd() for drift interpolation
        - _generate_brownian_increment_nd() for vector Brownian motion
        - _apply_boundary_conditions_nd() for per-dimension boundary handling
        - _estimate_density_from_particles_nd() for KDE

        Args:
            m_initial_condition: Initial density on nD grid, shape (N1, N2, ..., Nd)
            U_solution_for_drift: Value function, shape (Nt, N1, N2, ..., Nd)

        Returns:
            Density evolution, shape (Nt, N1, N2, ..., Nd)
        """
        # Extract nD grid parameters
        params = self._get_grid_params()
        dimension = params["dimension"]
        grid_shape = params["grid_shape"]
        spacings = params["spacings"]
        bounds = params["bounds"]
        coordinates = params["coordinates"]
        Nt = params["Nt"]
        Dt = params["Dt"]
        sigma = params["sigma"]
        coupling_coefficient = params["coupling_coefficient"]

        if Nt == 0:
            return np.zeros((0, *tuple(grid_shape)))

        # SDE: dX = alpha*dt + sigma*dW
        # Convention: problem.sigma is the SDE noise coefficient directly
        sigma_sde = sigma

        # Check if we need segment-aware BC (for absorbing boundaries)
        use_segment_aware_bc = self._needs_segment_aware_bc()

        # Reset exit flux tracking for this solve
        self.exit_flux_history = []
        self.exit_positions_history = []
        self.total_absorbed = 0

        # Allocate arrays
        M_density_on_grid = np.zeros((Nt, *tuple(grid_shape)))

        # For segment-aware BC with absorption, use list storage (variable particle count)
        # For uniform BC, use fixed array (all particles preserved)
        if use_segment_aware_bc:
            # List storage: each timestep may have different particle count
            particles_list: list[np.ndarray] = [None] * Nt  # type: ignore
            particles_list[0] = self._sample_particles_from_density_nd(
                m_initial_condition, coordinates, self.num_particles
            )
            current_particles = None  # Will be built at end
        else:
            # Particle positions: (Nt, num_particles, dimension)
            current_particles = np.zeros((Nt, self.num_particles, dimension))
            particles_list = None  # Not used

            # Sample initial particles from density
            current_particles[0] = self._sample_particles_from_density_nd(
                m_initial_condition, coordinates, self.num_particles
            )

        # Get initial particles for density estimation
        init_particles = particles_list[0] if use_segment_aware_bc else current_particles[0]

        # Estimate initial density using KDE
        M_density_on_grid[0] = self._estimate_density_from_particles_nd(init_particles, coordinates, bounds)

        # Normalize if requested
        if self.kde_normalization != KDENormalization.NONE:
            M_density_on_grid[0] = self._normalize_density_nd(M_density_on_grid[0], spacings)

        self._increment_time_step()

        if Nt == 1:
            if use_segment_aware_bc:
                self.M_particles_trajectory = particles_list  # List of arrays
            else:
                self.M_particles_trajectory = current_particles
            return M_density_on_grid

        # Progress bar for forward particle timesteps (n_time_points - 1 steps)
        timestep_range = self._create_timestep_range(Nt - 1, desc=f"FP {dimension}D (forward)")

        # Main time evolution loop
        for t_idx in timestep_range:
            # Get drift field at current time
            drift_or_U_t = U_solution_for_drift[t_idx]

            # Get particles at current timestep
            if use_segment_aware_bc:
                particles_t = particles_list[t_idx]
            else:
                particles_t = current_particles[t_idx]

            n_particles_t = len(particles_t)

            # Skip if no particles remain (all absorbed)
            if n_particles_t == 0:
                if use_segment_aware_bc:
                    particles_list[t_idx + 1] = np.array([]).reshape(0, dimension)
                M_density_on_grid[t_idx + 1] = np.zeros(grid_shape)
                self._increment_time_step()
                continue

            # Check if drift is precomputed or needs to be computed from U
            if self._drift_is_precomputed:
                # Drift is already computed (e.g., from GFDM: α = -D_p H(x,m,∇u))
                # drift_or_U_t has shape (*grid_shape, dimension)
                # Need to interpolate vector field to particle positions
                drift_at_particles = np.zeros((n_particles_t, dimension))

                for d in range(dimension):
                    # Extract d-th component of drift vector field
                    drift_d = drift_or_U_t[..., d]  # Shape: (*grid_shape,)
                    drift_at_particles[:, d] = self._interpolate_grid_to_particles_nd(drift_d, bounds, particles_t)

                # Use precomputed drift directly (no coupling_coefficient multiplication)
                drift = drift_at_particles

            else:
                # Traditional path: drift_or_U_t is value function U, compute gradient
                U_t = drift_or_U_t  # For clarity

                # Compute gradient of U on the grid (list of d arrays, one per dimension)
                gradients = self._compute_gradient_nd(U_t, spacings, use_backend=False)

                # Interpolate gradients to particle positions
                grad_at_particles = np.zeros((n_particles_t, dimension))

                for d in range(dimension):
                    grad_at_particles[:, d] = self._interpolate_grid_to_particles_nd(gradients[d], bounds, particles_t)

                # Compute drift: alpha = -coupling_coefficient * grad(U)
                drift = -coupling_coefficient * grad_at_particles

            # Generate Brownian increments
            dW = self._generate_brownian_increment_nd(n_particles_t, dimension, Dt, sigma_sde)

            # Euler-Maruyama step: X_{t+1} = X_t + drift * dt + sigma * dW
            new_particles = particles_t + drift * Dt + dW

            # Enforce obstacle boundaries if implicit domain is set (Issue #533)
            new_particles = self._enforce_obstacle_boundary(new_particles)

            # Apply boundary conditions
            if use_segment_aware_bc:
                # Segment-aware BC: may absorb particles
                if self._preserve_indices:
                    raise NotImplementedError(
                        "preserve_indices=True not supported in grid-drift n-D path; "
                        "use callable-drift path (Issue #1119)."
                    )
                new_particles, _n_absorbed, _, _ = self._apply_boundary_conditions_segment_aware(new_particles, bounds)
                particles_list[t_idx + 1] = new_particles
            else:
                # Uniform BC: topology-based (no absorption)
                topologies = self._get_topology_per_dimension(dimension)
                new_particles = self._apply_boundary_conditions_nd(new_particles, bounds, topologies)
                current_particles[t_idx + 1] = new_particles

            # Estimate density from particles
            M_density_on_grid[t_idx + 1] = self._estimate_density_from_particles_nd(new_particles, coordinates, bounds)

            # Normalize if requested (respects kde_normalization strategy)
            if self._should_normalize_density():
                M_density_on_grid[t_idx + 1] = self._normalize_density_nd(M_density_on_grid[t_idx + 1], spacings)

            self._increment_time_step()

        # Finalize: store trajectory, build history, return result
        trajectory = particles_list if use_segment_aware_bc else current_particles
        return self._finalize_particle_solve(
            M_density_on_grid, trajectory, Nt, dimension=dimension, use_segment_aware_bc=use_segment_aware_bc
        )

    def _normalize_density_nd(self, density: np.ndarray, spacings: list[float]) -> np.ndarray:
        """
        Normalize density to integrate to 1 for nD grids.

        Delegates to unified normalize_density() from fp_particle_density module.

        Args:
            density: Density array, shape (N1, N2, ..., Nd)
            spacings: Grid spacings [dx1, dx2, ..., dxd]

        Returns:
            Normalized density array
        """
        return _normalize(density, spacings)

    def _solve_fp_system_gpu(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """
        GPU pipeline - full particle evolution on GPU.

        Track B Phase 2.1: Full GPU acceleration including internal KDE.
        Eliminates all GPU↔CPU transfers during evolution loop.

        Expected speedup:
            - Apple Silicon MPS: 1.5-2x for N≥50k particles
            - NVIDIA CUDA: 3-5x (estimated, not tested)
            - See docs/development/TRACK_B_GPU_ACCELERATION_COMPLETE.md

        Note (Issue #535 Phase 1): Segment-aware absorbing BC not yet implemented
        for GPU solver. Use CPU solver for mixed BC with DIRICHLET segments.
        GPU solver currently supports uniform BC only (periodic/reflecting).
        """
        from mfgarchon.alg.numerical.density_estimation import gaussian_kde_gpu_internal
        from mfgarchon.utils.particle_utils import (
            apply_boundary_conditions_gpu,
            interpolate_1d_gpu,
            sample_from_density_gpu,
        )

        # Use geometry-aware parameter extraction
        params = self._get_grid_params()
        Nx = params["Nx"]
        Nt = params["Nt"]
        Dx = params["Dx"]
        Dt = params["Dt"]
        sigma_sde = params["sigma"]
        coupling_coefficient = params["coupling_coefficient"]
        x_grid = params["xSpace"]
        xmin = params["xmin"]
        xmax = params["xmax"]
        Lx = params["Lx"]

        if Nt == 0:
            return np.zeros((0, Nx))

        # Convert inputs to GPU ONCE at start
        x_grid_gpu = self.backend.from_numpy(x_grid)
        U_drift_gpu = self.backend.from_numpy(U_solution_for_drift)

        # Allocate arrays on GPU
        X_particles_gpu = self.backend.zeros((Nt, self.num_particles))
        M_density_gpu = self.backend.zeros((Nt, Nx))

        # Sample initial particles on GPU
        m_initial_gpu = self.backend.from_numpy(m_initial_condition)
        X_particles_gpu[0, :] = sample_from_density_gpu(
            m_initial_gpu, x_grid_gpu, self.num_particles, self.backend, seed=None
        )

        # Compute bandwidth for KDE (do this once on CPU)
        # Convert bandwidth parameter to absolute bandwidth value
        if isinstance(self.kde_bandwidth, str):
            from mfgarchon.alg.numerical.density_estimation import adaptive_bandwidth_selection

            # Need numpy array for bandwidth calculation
            X_init_np = self.backend.to_numpy(X_particles_gpu[0, :])
            bandwidth_absolute = adaptive_bandwidth_selection(X_init_np, method=self.kde_bandwidth)
        else:
            # User provided factor - compute factor * std(particles)
            X_init_np = self.backend.to_numpy(X_particles_gpu[0, :])
            data_std = np.std(X_init_np, ddof=1)
            bandwidth_absolute = float(self.kde_bandwidth) * data_std

        # Estimate initial density using internal GPU KDE (Phase 2.1)
        M_density_gpu[0, :] = gaussian_kde_gpu_internal(
            X_particles_gpu[0, :], x_grid_gpu, bandwidth_absolute, self.backend
        )

        # Normalize based on strategy (use helper function)
        if self.kde_normalization != KDENormalization.NONE:
            M_density_gpu[0, :] = self._normalize_density(M_density_gpu[0, :], Dx, use_backend=True)

        self._increment_time_step()  # Increment after computing density at t=0

        if Nt == 1:
            self.M_particles_trajectory = self.backend.to_numpy(X_particles_gpu)
            return self.backend.to_numpy(M_density_gpu)

        # Main evolution loop - ALL GPU
        for t in range(Nt - 1):
            U_t_gpu = U_drift_gpu[t, :]

            # Compute gradient on grid (use shared nD method)
            if Nx > 1:
                gradients_gpu = self._compute_gradient_nd(U_t_gpu, [Dx], use_backend=True)
                dUdx_gpu = gradients_gpu[0]  # First (and only) dimension
            else:
                dUdx_gpu = self.backend.zeros((Nx,))

            # Interpolate gradient to particle positions (GPU)
            if Nx > 1:
                dUdx_particles_gpu = interpolate_1d_gpu(X_particles_gpu[t, :], x_grid_gpu, dUdx_gpu, self.backend)
            else:
                dUdx_particles_gpu = self.backend.zeros((self.num_particles,))

            # Compute drift (GPU)
            drift_gpu = -coupling_coefficient * dUdx_particles_gpu

            # Random noise (GPU native RNG)
            if Dt > 1e-14:
                # Generate on CPU and transfer (safest approach for now)
                noise_scale = sigma_sde * np.sqrt(Dt)
                noise_np = np.random.randn(self.num_particles) * noise_scale
                noise_gpu = self.backend.from_numpy(noise_np)
            else:
                noise_gpu = self.backend.zeros((self.num_particles,))

            # Euler-Maruyama update (GPU)
            X_particles_gpu[t + 1, :] = X_particles_gpu[t, :] + drift_gpu * Dt + noise_gpu

            # Apply boundary handling (GPU, supports mixed BCs)
            # Map topology to GPU bc_type: "bounded" -> "no_flux" for GPU function
            topology_1d = self._get_topology_per_dimension(1)[0]
            gpu_bc_type = "periodic" if topology_1d == "periodic" else "no_flux"
            if Lx > 1e-14:
                X_particles_gpu[t + 1, :] = apply_boundary_conditions_gpu(
                    X_particles_gpu[t + 1, :], xmin, xmax, gpu_bc_type, self.backend
                )

            # Estimate density using internal GPU KDE (Phase 2.1 - no transfers!)
            M_density_gpu[t + 1, :] = gaussian_kde_gpu_internal(
                X_particles_gpu[t + 1, :], x_grid_gpu, bandwidth_absolute, self.backend
            )

            # Normalize based on strategy (use helper function)
            if self.kde_normalization != KDENormalization.NONE:
                M_density_gpu[t + 1, :] = self._normalize_density(M_density_gpu[t + 1, :], Dx, use_backend=True)

            self._increment_time_step()  # Increment after each time step

        # Store trajectory and convert to NumPy ONCE at end
        X_particles_np = self.backend.to_numpy(X_particles_gpu)
        M_density_np = self.backend.to_numpy(M_density_gpu)

        # Finalize: store trajectory, build history, return result
        # GPU solver is 1D only, uses Nt time points
        return self._finalize_particle_solve(M_density_np, X_particles_np, Nt, dimension=1)

    def _solve_fp_system_callable_drift(
        self,
        M_initial: np.ndarray | None,
        drift_callable: Callable,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
        initial_particles: np.ndarray | None = None,
        drift_needs_density: bool = True,
    ) -> np.ndarray:
        """
        Solve FP equation with callable (state-dependent) drift using particles.

        Evaluates drift at particle positions at each timestep, enabling
        nonlinear PDEs with state-dependent advection.

        Parameters
        ----------
        M_initial : np.ndarray or None
            Initial density on grid. Not required if initial_particles provided.
        drift_callable : callable
            Function α(t, x, m) -> drift velocity
            - t: time (scalar)
            - x: particle positions, shape (N_particles, d)
            - m: density at particle positions, shape (N_particles,)
            Returns: drift velocity, shape (N_particles, d) for nD or (N_particles,) for 1D
        volatility_field : float, np.ndarray, Callable, or None
            Volatility (SDE noise coefficient σ or Σ). Uses problem.sigma if None.
            - float: Constant isotropic volatility σ
            - (d,d) array: Anisotropic noise matrix Σ
            - Callable: State-dependent σ(t,x,m) or Σ(t,x,m)
        show_progress : bool
            Show progress bar
        initial_particles : np.ndarray or None
            Pre-sampled initial particles, shape (num_particles, dimension).
            If provided, M_initial is not used for sampling (meshfree initialization).

        Returns
        -------
        np.ndarray
            Density evolution on grid, shape (Nt+1, *grid_shape)
        """
        from mfgarchon.types.pde_coefficients import DriftCallable

        # Validate callable
        if not isinstance(drift_callable, DriftCallable):
            raise TypeError(
                "drift_field callable does not match DriftCallable protocol. "
                "Expected signature: (t: float, x: ndarray, m: ndarray) -> ndarray"
            )

        # Get parameters
        params = self._get_grid_params()
        Nt = params["Nt"]
        Dt = params["Dt"]
        dimension = params["dimension"]
        grid_shape = params["grid_shape"]
        bounds = params["bounds"]
        spacings = params["spacings"]
        coordinates = params["coordinates"]

        # Initialize particle history for direct query modes (Issue #489 / callable drift path)
        # This must be done here since callable drift returns early from solve_fp_system().
        # Issue #1119: preserve_indices implies user wants per-step trajectories, so force-enable.
        if self.density_mode in ("hybrid", "query_only") or self._preserve_indices:
            self._particle_history = []

        # Get volatility - supports constant, array, or callable
        # For SDE: dX = drift*dt + σ*dW (volatility_field = σ)
        volatility_is_callable = callable(volatility_field)
        volatility_is_array = isinstance(volatility_field, np.ndarray)

        if volatility_field is None:
            base_sigma = self.problem.sigma
        elif isinstance(volatility_field, (int, float)):
            base_sigma = float(volatility_field)
        elif volatility_is_array or volatility_is_callable:
            # Spatially varying or state-dependent volatility
            # Will be evaluated per timestep at particle positions
            base_sigma = None  # Evaluated dynamically
        else:
            raise TypeError(
                f"volatility_field must be None, float, np.ndarray, or Callable, got {type(volatility_field)}"
            )

        # Pre-compute constant sigma_sde if volatility is constant
        # Issue #717 fix: volatility_field IS the SDE volatility σ, use directly
        # The PDE diffusion D = σ²/2 is computed internally when needed
        if base_sigma is not None:
            sigma_sde_constant = base_sigma  # Direct use: σ_sde = σ
        else:
            sigma_sde_constant = None

        # Issue #1042: detect segment-aware BC (e.g. Dirichlet absorbing exits) so we
        # can route through the absorbing-particle path instead of the uniform-topology
        # path. The grid-drift solver (`_solve_fp_system_grid_drift`) does this; the
        # callable-drift path was bypassing it.
        use_segment_aware_bc = self._needs_segment_aware_bc()
        if use_segment_aware_bc:
            # Reset exit-flux tracking for this solve (mirrors grid-drift path)
            self.exit_flux_history = []
            self.exit_positions_history = []
            self.total_absorbed = 0

        # Initialize particles - either from pre-sampled or from density grid.
        # Storage choice: segment-aware BC may absorb particles → variable count per
        # step → list. Uniform BC preserves count → fixed (Nt+1, N, d) array.
        if use_segment_aware_bc:
            particles_list: list[np.ndarray] | None = [None] * (Nt + 1)  # type: ignore[list-item]
            current_particles = None
        else:
            particles_list = None
            current_particles = np.zeros((Nt + 1, self.num_particles, dimension))

        if initial_particles is not None:
            # Meshfree initialization: use pre-sampled particles directly
            if initial_particles.shape[0] != self.num_particles:
                raise ValueError(
                    f"initial_particles has {initial_particles.shape[0]} particles, expected {self.num_particles}"
                )
            if initial_particles.shape[1] != dimension:
                raise ValueError(f"initial_particles has dimension {initial_particles.shape[1]}, expected {dimension}")
            init_p = initial_particles
        else:
            # Standard: sample from grid density
            init_p = self._sample_particles_from_density_nd(M_initial, coordinates, self.num_particles)

        if use_segment_aware_bc:
            particles_list[0] = init_p
        else:
            current_particles[0] = init_p

        # Issue #1119: preserve_indices bookkeeping. Maintain orig_indices mapping
        # live particles → their original index, and a parallel full-size NaN-marked
        # history. Internal SDE/KDE loop still uses compact arrays for performance;
        # NaN-marked array is only built for storage.
        if self._preserve_indices:
            if not use_segment_aware_bc:
                raise NotImplementedError(
                    "preserve_indices=True requires segment-aware BC (Dirichlet absorbing). "
                    "With purely reflecting BC, no particles are absorbed, so the default "
                    "compact array already preserves all indices (Issue #1119)."
                )
            preserve_orig_indices: np.ndarray | None = np.arange(self.num_particles)
            preserve_full_history: list[np.ndarray] | None = [None] * (Nt + 1)
            full_t0 = np.full((self.num_particles, dimension), np.nan)
            full_t0[preserve_orig_indices] = init_p
            preserve_full_history[0] = full_t0
        else:
            preserve_orig_indices = None
            preserve_full_history = None

        # Allocate density array (only used if density_mode != "query_only")
        M_density_on_grid = np.zeros((Nt + 1, *grid_shape))
        if M_initial is not None:
            M_density_on_grid[0] = M_initial.copy()
            # Normalize initial if requested
            if self._should_normalize_density():
                M_density_on_grid[0] = self._normalize_density_nd(M_density_on_grid[0], spacings)

        # Progress bar
        from mfgarchon.utils.progress import create_progress_bar, should_show_progress

        timestep_range = create_progress_bar(
            range(Nt),
            verbose=should_show_progress(show_progress),
            desc="FP Particle (callable drift)",
        )

        # Time evolution with callable drift
        for t_idx in timestep_range:
            t_current = t_idx * Dt
            # Fetch particles at this step (variable count under segment-aware BC)
            particles_t = particles_list[t_idx] if use_segment_aware_bc else current_particles[t_idx]
            n_particles_t = len(particles_t)
            # Skip if all particles absorbed (mirrors grid-drift path, lines ~1606-1613)
            if n_particles_t == 0:
                if use_segment_aware_bc:
                    particles_list[t_idx + 1] = np.empty((0, dimension), dtype=particles_t.dtype)
                # Issue #1119: maintain full-NaN history once all particles absorbed
                if self._preserve_indices:
                    preserve_full_history[t_idx + 1] = np.full((self.num_particles, dimension), np.nan)
                if self.density_mode != "query_only":
                    M_density_on_grid[t_idx + 1] = np.zeros(grid_shape)
                self._increment_time_step()
                continue

            # Estimate density at particle positions for state-dependent drift
            # Skip if drift_needs_density=False (optimization for drifts that don't use m)
            if not drift_needs_density:
                # Drift doesn't depend on density - pass dummy zeros
                m_at_particles = np.zeros(n_particles_t)
            elif M_initial is not None:
                # Standard: interpolate from grid density
                m_at_particles = self._estimate_density_at_particles(
                    particles_t, M_density_on_grid[t_idx], coordinates, bounds
                )
            else:
                # Meshfree: use KDE on current particles (for initial_particles mode)
                # This provides self-consistent density estimate without grid
                # Issue #1083: pass reflect_bounds for axes with reflecting BC
                # so boundary cells aren't underestimated by ~50% (Issue #709 ghosts)
                from mfgarchon.alg.numerical.fp_solvers.particle_density_query import ParticleDensityQuery

                reflect_bounds = self._infer_reflect_bounds(bounds)
                query = ParticleDensityQuery(
                    particles_t,
                    bandwidth_rule="scott",
                    reflect_bounds=reflect_bounds,
                )
                m_at_particles = query.query_density(particles_t, method="hybrid", k=min(50, len(particles_t) - 1))

            # Evaluate drift callable
            # For 1D: x is (N,) and returns (N,)
            # For nD: x is (N, d) and returns (N, d)
            if dimension == 1:
                x_for_callable = particles_t[:, 0]  # Flatten to (N,)
                drift_values = drift_callable(t_current, x_for_callable, m_at_particles)
                # Ensure shape is (N, 1) for consistent processing
                if drift_values.ndim == 1:
                    drift = drift_values[:, np.newaxis]
                else:
                    drift = drift_values
            else:
                drift = drift_callable(t_current, particles_t, m_at_particles)
                if drift.ndim == 1:
                    # Scalar drift applied to all dimensions
                    drift = np.tile(drift[:, np.newaxis], (1, dimension))

            # Generate Brownian increments with per-particle diffusion support
            # Issue #1042: use n_particles_t (current step count) instead of
            # self.num_particles, since count varies under segment-aware BC.
            if sigma_sde_constant is not None:
                # Constant diffusion - use pre-computed value
                dW = self._generate_brownian_increment_nd(n_particles_t, dimension, Dt, sigma_sde_constant)
            else:
                # Spatially varying or callable volatility - evaluate at particle positions
                if volatility_is_callable:
                    # Callable: sigma(t, x, m) -> per-particle sigma
                    if dimension == 1:
                        x_for_callable = particles_t[:, 0]
                    else:
                        x_for_callable = particles_t
                    sigma_at_particles = volatility_field(t_current, x_for_callable, m_at_particles)
                else:
                    # Array: interpolate from grid
                    sigma_at_particles = self._interpolate_field_at_particles(
                        particles_t, volatility_field, coordinates, bounds
                    )

                # Ensure sigma_at_particles is 1D array of shape (n_particles_t,)
                sigma_at_particles = np.atleast_1d(sigma_at_particles).ravel()
                if sigma_at_particles.shape[0] == 1:
                    # Broadcast scalar to all particles
                    sigma_at_particles = np.full(n_particles_t, sigma_at_particles[0])

                # Generate per-particle Brownian increments
                # Issue #717 fix: sigma_at_particles IS the SDE volatility σ
                # SDE: dX = v dt + σ dW, so dW_i = σ_i * N(0, sqrt(dt))
                sigma_sde_particles = sigma_at_particles  # Direct use
                dW = sigma_sde_particles[:, np.newaxis] * np.random.normal(0, np.sqrt(Dt), (n_particles_t, dimension))

            # Euler-Maruyama step
            new_particles = particles_t + drift * Dt + dW

            # Enforce obstacle boundaries if implicit domain is set (Issue #533)
            new_particles = self._enforce_obstacle_boundary(new_particles)

            # Apply boundary conditions — Issue #1042 fix: route to segment-aware
            # path when BC has Dirichlet (absorbing) segments, mirroring grid-drift.
            if use_segment_aware_bc:
                new_particles, _n_absorbed, _, absorbed_mask = self._apply_boundary_conditions_segment_aware(
                    new_particles, bounds
                )
                particles_list[t_idx + 1] = new_particles
                # Issue #1119: update orig_indices + build NaN-marked full array
                if self._preserve_indices:
                    if _n_absorbed > 0:
                        preserve_orig_indices = preserve_orig_indices[~absorbed_mask]
                    full_t = np.full((self.num_particles, dimension), np.nan)
                    full_t[preserve_orig_indices] = new_particles
                    preserve_full_history[t_idx + 1] = full_t
            else:
                topologies = self._get_topology_per_dimension(dimension)
                new_particles = self._apply_boundary_conditions_nd(new_particles, bounds, topologies)
                current_particles[t_idx + 1] = new_particles

            # Estimate density from particles (skip if query_only mode - Issue #711 optimization)
            # When density_mode="query_only", grid density is not used - density is queried
            # at arbitrary points via particle history. Skipping KDE saves ~60s per iteration.
            if self.density_mode != "query_only":
                M_density_on_grid[t_idx + 1] = self._estimate_density_from_particles_nd(
                    new_particles, coordinates, bounds
                )

                # Normalize if requested (respects kde_normalization strategy)
                if self._should_normalize_density():
                    M_density_on_grid[t_idx + 1] = self._normalize_density_nd(M_density_on_grid[t_idx + 1], spacings)

            self._increment_time_step()

        # Finalize: store trajectory, build history, return result
        # Note: callable drift uses Nt+1 time points (0 to Nt inclusive)
        # Issue #1042: trajectory storage and use_segment_aware_bc flag now match
        # the BC routing path taken in the loop.
        # Issue #1119: when preserve_indices=True, swap trajectory to the
        # NaN-marked full-size history (fixed (Nt+1, N, d) shape).
        if self._preserve_indices and use_segment_aware_bc:
            trajectory = preserve_full_history
        else:
            trajectory = particles_list if use_segment_aware_bc else current_particles
        return self._finalize_particle_solve(
            M_density_on_grid,
            trajectory,
            Nt + 1,
            dimension=dimension,
            use_segment_aware_bc=use_segment_aware_bc,
        )

    def _interpolate_field_at_particles(
        self,
        particles: np.ndarray,
        field: np.ndarray,
        coordinates: list[np.ndarray],
        bounds: list[tuple[float, float]],
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """
        Interpolate a grid field to particle positions.

        Delegates to _interpolate_grid_to_particles_nd() which uses the
        utils.numerical.particle.interpolate_grid_to_particles() function.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (N_particles, dimension)
        field : np.ndarray
            Field values on grid, shape (*grid_shape)
        coordinates : list of np.ndarray
            Grid coordinates per dimension (unused, kept for API compatibility)
        bounds : list of tuple
            Domain bounds per dimension
        fill_value : float
            Value for out-of-bounds particles (default: 0.0, handled by utils)

        Returns
        -------
        np.ndarray
            Field values at particle positions, shape (N_particles,)
        """
        return self._interpolate_grid_to_particles_nd(field, bounds, particles)

    def _estimate_density_at_particles(
        self,
        particles: np.ndarray,
        grid_density: np.ndarray,
        coordinates: list[np.ndarray],
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Estimate density at particle positions by interpolating grid density.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (N_particles, dimension)
        grid_density : np.ndarray
            Density on grid, shape (*grid_shape)
        coordinates : list of np.ndarray
            Grid coordinates per dimension
        bounds : list of tuple
            Domain bounds per dimension

        Returns
        -------
        np.ndarray
            Density at particle positions, shape (N_particles,)
        """
        # Use general interpolation with fill_value=0 for out-of-bounds
        return self._interpolate_field_at_particles(particles, grid_density, coordinates, bounds, fill_value=0.0)


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPParticleSolver...")

    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import neumann_bc

    # Minimal components for FP-only testing
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )

    # Test 1D problem with particle solver
    geometry_1d = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[31],
        boundary_conditions=neumann_bc(dimension=1),
    )
    problem = MFGProblem(geometry=geometry_1d, T=1.0, Nt=20, sigma=0.1, components=components)
    solver = FPParticleSolver(problem, num_particles=1000)

    # Test solver initialization
    assert solver.fp_method_name == "Particle"
    assert solver.num_particles == 1000

    # Test solve_fp_system
    import numpy as np

    Nx = problem.geometry.get_grid_shape()[0]
    U_test = np.zeros((problem.Nt + 1, Nx))
    M_init = problem.m_initial  # Issue #670: unified naming

    M_solution = solver.solve_fp_system(M_initial=M_init, drift_field=U_test)

    assert M_solution.shape == (problem.Nt + 1, Nx)
    assert not np.any(np.isnan(M_solution))
    assert not np.any(np.isinf(M_solution))
    assert np.all(M_solution >= 0), "Density must be non-negative"

    print("  Particle solver converged")
    print(f"  Num particles: {solver.num_particles}")
    print(f"  M range: [{M_solution.min():.3f}, {M_solution.max():.3f}]")
    print(f"  KDE bandwidth: {solver.kde_bandwidth}")

    # Test 2D problem with particle solver (nD support)
    print("\nTesting 2D FPParticleSolver...")

    geometry_2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[16, 16],
        boundary_conditions=neumann_bc(dimension=2),
    )
    problem_2d = MFGProblem(geometry=geometry_2d, Nt=10, T=0.5, sigma=0.1, components=components)

    solver_2d = FPParticleSolver(problem_2d, num_particles=500, mode="hybrid")

    # Create 2D test arrays
    # U_test_2d has shape (n_time_points, *spatial) = (Nt + 1, *spatial)
    grid_shape_2d = problem_2d.geometry.get_grid_shape()  # (16, 16)
    U_test_2d = np.zeros((problem_2d.Nt + 1, *tuple(grid_shape_2d)))

    # Create 2D Gaussian initial density
    coords = problem_2d.geometry.coordinates
    X, Y = np.meshgrid(coords[0], coords[1], indexing="ij")
    M_init_2d = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1)
    M_init_2d = M_init_2d / (np.sum(M_init_2d) * (1.0 / 15) ** 2)  # Normalize

    result_2d = solver_2d.solve_fp_system(M_initial=M_init_2d, drift_field=U_test_2d)

    # 2D may return FPParticleResult, extract M_grid array
    M_solution_2d = result_2d.M_grid if hasattr(result_2d, "M_grid") else result_2d

    expected_shape_2d = (problem_2d.Nt + 1, *tuple(grid_shape_2d))  # Nt+1 for t=0...T
    assert M_solution_2d.shape == expected_shape_2d, f"Shape mismatch: {M_solution_2d.shape} vs {expected_shape_2d}"
    assert not np.any(np.isnan(M_solution_2d)), "NaN in 2D solution"
    assert not np.any(np.isinf(M_solution_2d)), "Inf in 2D solution"
    assert np.all(M_solution_2d >= 0), "2D density must be non-negative"

    # Check mass conservation (should be approximately preserved)
    initial_mass = np.sum(M_solution_2d[0]) * (1.0 / 15) ** 2
    final_mass = np.sum(M_solution_2d[-1]) * (1.0 / 15) ** 2
    mass_ratio = final_mass / initial_mass if initial_mass > 1e-10 else 1.0

    print("  2D Particle solver converged")
    print(f"  Grid shape: {grid_shape_2d}")
    print(f"  M shape: {M_solution_2d.shape}")
    print(f"  M range: [{M_solution_2d.min():.3f}, {M_solution_2d.max():.3f}]")
    print(f"  Mass ratio (final/initial): {mass_ratio:.4f}")

    # Test particle trajectory storage for 2D
    assert solver_2d.M_particles_trajectory is not None, "Particle trajectory not stored"
    assert solver_2d.M_particles_trajectory.shape[1] == solver_2d.num_particles, "Particle count mismatch"
    assert solver_2d.M_particles_trajectory.shape[2] == 2, "2D particles should have 2 coordinates"
    print(f"  Particle trajectory shape: {solver_2d.M_particles_trajectory.shape}")

    # Test 3: Absorbing BC (segment-aware)
    print("\nTesting 2D FPParticleSolver with absorbing BC...")
    from mfgarchon.geometry.boundary import BCSegment, mixed_bc

    # Create BC with exit on right wall (DIRICHLET = absorbing for particles)
    bc_absorbing = mixed_bc(
        dimension=2,
        segments=[
            BCSegment(
                name="exit",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="right",  # Right wall is exit (use direction name, not x_max)
            ),
            BCSegment(
                name="walls",
                bc_type=BCType.REFLECTING,
                boundary="all",
                priority=-1,  # Lower priority = fallback
            ),
        ],
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )

    solver_absorbing = FPParticleSolver(
        problem_2d,
        num_particles=200,
        boundary_conditions=bc_absorbing,
    )

    # Drift particles toward the exit (right wall)
    # Use a gradient that pushes particles to the right
    drift_to_right = np.zeros((problem_2d.Nt + 1, *tuple(grid_shape_2d), 2))
    drift_to_right[..., 0] = 0.5  # Positive x-drift (toward x_max)

    M_solution_abs = solver_absorbing.solve_fp_system(
        M_initial=M_init_2d,
        drift_field=drift_to_right,
        drift_is_precomputed=True,
    )

    # Verify some particles were absorbed
    print(f"  Total absorbed: {solver_absorbing.total_absorbed}")
    print(f"  Exit flux history length: {len(solver_absorbing.exit_flux_history)}")

    # With strong rightward drift, particles should hit the exit
    # and be absorbed (mass should decrease)
    initial_mass_abs = np.sum(M_solution_abs[0])
    final_mass_abs = np.sum(M_solution_abs[-1])
    mass_loss = initial_mass_abs - final_mass_abs

    print(f"  Initial mass: {initial_mass_abs:.4f}")
    print(f"  Final mass: {final_mass_abs:.4f}")
    print(f"  Mass loss: {mass_loss:.4f}")

    # With absorbing BC, mass should decrease (particles exiting)
    # Note: This test may show small mass loss due to limited particles/time
    assert not np.any(np.isnan(M_solution_abs)), "NaN in absorbing BC solution"
    print("  Absorbing BC test passed")

    # Test 4: Obstacle geometry (Issue #533)
    print("\nTesting 2D FPParticleSolver with obstacle (Issue #533)...")
    from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere

    # Create 2D domain with circular obstacle
    bounds_rect = np.array([[0.0, 1.0], [0.0, 1.0]])
    base_domain = Hyperrectangle(bounds_rect)
    obstacle = Hypersphere(center=[0.5, 0.5], radius=0.15)
    domain_with_obstacle = DifferenceDomain(base_domain, obstacle)

    # Solver with obstacle handling
    solver_obstacle = FPParticleSolver(
        problem_2d,
        num_particles=500,
        implicit_domain=domain_with_obstacle,
    )

    # Initial density near obstacle
    M_near_obstacle = np.exp(-((X - 0.35) ** 2 + (Y - 0.5) ** 2) / 0.05)
    M_near_obstacle = M_near_obstacle / np.sum(M_near_obstacle)

    # Pure diffusion (no drift) - particles will diffuse toward obstacle
    drift_zero = np.zeros((problem_2d.Nt + 1, *tuple(grid_shape_2d), 2))

    M_solution_obs = solver_obstacle.solve_fp_system(
        M_initial=M_near_obstacle,
        drift_field=drift_zero,
        drift_is_precomputed=True,
        show_progress=False,
    )

    # Check final particle positions
    final_particles = solver_obstacle.M_particles_trajectory[-1]
    inside_valid = domain_with_obstacle.contains(final_particles)
    pct_valid = 100.0 * np.sum(inside_valid) / len(final_particles)

    print(f"  Particles in valid domain: {pct_valid:.1f}%")
    print(f"  Particles inside obstacle: {len(final_particles) - np.sum(inside_valid)}")

    # Most particles should be in valid domain (outside obstacle)
    assert pct_valid > 95.0, f"Too many particles inside obstacle: {100 - pct_valid:.1f}%"
    assert not np.any(np.isnan(M_solution_obs)), "NaN in obstacle solution"
    print("  Obstacle geometry test passed")

    print("\nAll smoke tests passed!")
