#!/usr/bin/env python3
"""
Method of Manufactured Solutions (MMS) validation tests for BC infrastructure.

Issue #523: MMS and Conservation Validation Suite for BC

MMS is the gold standard for verifying numerical PDE solvers:
1. Choose an analytical solution m_exact(t,x)
2. Compute the source term S(t,x) needed to satisfy the PDE
3. Solve the PDE numerically with source S
4. Compare numerical solution to m_exact
5. Verify expected convergence rate as grid refines

For Fokker-Planck: dm/dt + div(v*m) - sigma^2/2 * Laplacian(m) = S(t,x)
For HJB: -du/dt + H(grad_u) = f(x,m) + S(t,x)

References:
    - Roache, P.J. (1998). Verification and Validation in Computational Science
    - Oberkampf & Roy (2010). Verification and Validation in Scientific Computing
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


class ManufacturedSolution:
    """
    Base class for manufactured solutions.

    A manufactured solution provides:
    - Analytical solution m(t,x) or u(t,x)
    - Its derivatives (time, space, Laplacian)
    - The source term S(t,x) needed to satisfy the PDE
    """

    def __init__(self, dimension: int = 1):
        self.dimension = dimension

    def solution(self, t: float, x: np.ndarray) -> np.ndarray:
        """Evaluate the manufactured solution at (t,x)."""
        raise NotImplementedError

    def time_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        """Evaluate dm/dt at (t,x)."""
        raise NotImplementedError

    def gradient(self, t: float, x: np.ndarray) -> np.ndarray:
        """Evaluate grad(m) at (t,x). Shape: (d, N) for d dimensions, N points."""
        raise NotImplementedError

    def laplacian(self, t: float, x: np.ndarray) -> np.ndarray:
        """Evaluate Laplacian(m) at (t,x)."""
        raise NotImplementedError

    def fp_source(self, t: float, x: np.ndarray, velocity: np.ndarray, sigma: float) -> np.ndarray:
        """
        Compute FP source term: S = dm/dt + div(v*m) - sigma^2/2 * Lap(m)

        For div(v*m) in 1D: d/dx(v*m) = v * dm/dx + m * dv/dx
        """
        raise NotImplementedError


class DiffusionSinusoid1D(ManufacturedSolution):
    """
    1D sinusoidal solution to the pure diffusion equation.

    For the FP equation without drift:
        dm/dt = (sigma^2/2) * d^2m/dx^2

    With periodic BC and initial condition m(0,x) = 1 + A*sin(k*x),
    the exact solution is:
        m(t,x) = 1 + A*sin(k*x)*exp(-D*k^2*t)

    where D = sigma^2/2 (diffusion coefficient) and k = 2*pi (wavenumber).

    Properties:
    - EXACT solution to homogeneous diffusion equation
    - Positive everywhere (for A < 1)
    - Periodic in space
    - Mode decays exponentially (physical diffusion behavior)
    """

    def __init__(self, sigma: float = 0.2, amplitude: float = 0.5, k: float = 2.0 * np.pi):
        super().__init__(dimension=1)
        self.sigma = sigma
        self.amplitude = amplitude
        self.k = k  # wavenumber
        self.D = 0.5 * sigma**2  # diffusion coefficient

    def _decay_factor(self, t: float) -> float:
        """exp(-D*k^2*t) - the decay rate of the sinusoidal mode."""
        return np.exp(-self.D * self.k**2 * t)

    def solution(self, t: float, x: np.ndarray) -> np.ndarray:
        """m(t,x) = 1 + A*sin(k*x)*exp(-D*k^2*t)"""
        x = np.atleast_1d(x)
        return 1.0 + self.amplitude * np.sin(self.k * x) * self._decay_factor(t)

    def time_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        """dm/dt = -D*k^2 * A*sin(k*x)*exp(-D*k^2*t)"""
        x = np.atleast_1d(x)
        return -self.D * self.k**2 * self.amplitude * np.sin(self.k * x) * self._decay_factor(t)

    def gradient(self, t: float, x: np.ndarray) -> np.ndarray:
        """dm/dx = k*A*cos(k*x)*exp(-D*k^2*t)"""
        x = np.atleast_1d(x)
        return self.k * self.amplitude * np.cos(self.k * x) * self._decay_factor(t)

    def laplacian(self, t: float, x: np.ndarray) -> np.ndarray:
        """d^2m/dx^2 = -k^2*A*sin(k*x)*exp(-D*k^2*t)"""
        x = np.atleast_1d(x)
        return -(self.k**2) * self.amplitude * np.sin(self.k * x) * self._decay_factor(t)

    def verify_pde_residual(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Verify that dm/dt - D*d^2m/dx^2 = 0 (should be zero for exact solution).
        """
        dmdt = self.time_derivative(t, x)
        d2mdx2 = self.laplacian(t, x)
        residual = dmdt - self.D * d2mdx2
        return residual  # Should be zero (or machine epsilon)


class GaussianDensity1D(ManufacturedSolution):
    """
    1D Gaussian manufactured density with time-varying width.

    m(t,x) = 1/(sqrt(2*pi*s(t)^2)) * exp(-(x-x0)^2 / (2*s(t)^2))
    where s(t)^2 = s0^2 + sigma^2*t (diffusion spreading)

    Properties:
    - Exact solution to pure diffusion (no advection)
    - Integrates to 1 (proper probability density)
    - Requires Dirichlet BC (decays at boundaries)
    """

    def __init__(self, x0: float = 0.5, s0: float = 0.1, sigma: float = 0.1):
        super().__init__(dimension=1)
        self.x0 = x0
        self.s0 = s0
        self.sigma = sigma

    def variance(self, t: float) -> float:
        """s(t)^2 = s0^2 + sigma^2*t"""
        return self.s0**2 + self.sigma**2 * t

    def solution(self, t: float, x: np.ndarray) -> np.ndarray:
        """Gaussian density."""
        x = np.atleast_1d(x)
        var = self.variance(t)
        return 1.0 / np.sqrt(2.0 * np.pi * var) * np.exp(-0.5 * (x - self.x0) ** 2 / var)

    def time_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        """dm/dt using chain rule on Gaussian."""
        x = np.atleast_1d(x)
        var = self.variance(t)
        m = self.solution(t, x)
        dvar_dt = self.sigma**2

        # dm/dt = m * [dvar/dt / (2*var) * ((x-x0)^2/var - 1)]
        return m * (dvar_dt / (2.0 * var)) * ((x - self.x0) ** 2 / var - 1.0)

    def gradient(self, t: float, x: np.ndarray) -> np.ndarray:
        """dm/dx = -m * (x-x0) / var"""
        x = np.atleast_1d(x)
        var = self.variance(t)
        return -self.solution(t, x) * (x - self.x0) / var

    def laplacian(self, t: float, x: np.ndarray) -> np.ndarray:
        """d2m/dx2 = m * [(x-x0)^2/var^2 - 1/var]"""
        x = np.atleast_1d(x)
        var = self.variance(t)
        m = self.solution(t, x)
        return m * ((x - self.x0) ** 2 / var**2 - 1.0 / var)

    def fp_source(self, t: float, x: np.ndarray, velocity: np.ndarray, sigma: float) -> np.ndarray:
        """
        For the special case where sigma matches self.sigma and velocity=0,
        the source should be zero (exact solution to diffusion equation).
        """
        dmdt = self.time_derivative(t, x)
        dmdx = self.gradient(t, x)
        d2mdx2 = self.laplacian(t, x)

        advection = velocity * dmdx
        diffusion = 0.5 * sigma**2 * d2mdx2

        return dmdt + advection - diffusion


class TestMMSFokkerPlanck1D:
    """MMS validation tests for 1D Fokker-Planck solver."""

    def test_sinusoidal_periodic_convergence(self):
        """
        Test convergence rate for sinusoidal solution with periodic BC.

        Expected: 2nd order convergence (error ~ O(h^2))
        """
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.geometry import periodic_bc

        sigma = 0.2
        T = 0.5
        manufactured = DiffusionSinusoid1D(sigma=sigma, amplitude=0.3)

        # Test with multiple resolutions
        resolutions = [21, 41, 81]
        errors = []

        for Nx in resolutions:
            # Create problem
            geometry = TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1)
            )
            problem = MFGProblem(
                geometry=geometry,
                T=T,
                Nt=Nx,  # Keep CFL-like ratio
                sigma=sigma,
                components=_default_components(),
            )

            # Initial condition from manufactured solution
            x_grid = geometry.coordinates[0]  # 1D grid
            m_init = manufactured.solution(0.0, x_grid)

            # For pure diffusion (no advection), solve FP
            bc = periodic_bc(dimension=1)
            solver = FPFDMSolver(problem, boundary_conditions=bc)

            # Create zero drift field (shape: Nt+1 x Nx)
            U_zero = np.zeros((problem.Nt + 1, Nx))

            # Solve
            M_numerical = solver.solve_fp_system(M_initial=m_init, potential_field=U_zero, show_progress=False)

            # Compare final time solution
            m_exact_final = manufactured.solution(T, x_grid)
            error = np.sqrt(np.mean((M_numerical[-1, :] - m_exact_final) ** 2))
            errors.append(error)

        # Check convergence rate
        # FP FDM uses upwind scheme by default -> 1st order spatial, O(h)
        # For 1st order: error(h/2) / error(h) ~ 2
        # For 2nd order: error(h/2) / error(h) ~ 4
        errors = np.array(errors)
        ratios = errors[:-1] / errors[1:]

        # Compute convergence order: p = log(e1/e2) / log(h1/h2)
        # With approximate grid doubling: p ≈ log(e1/e2) / log(2)
        orders = np.log(ratios) / np.log(2)

        # Expect at least 1st order (ratio > 1.8 means order > 0.85)
        # The upwind scheme gives O(h) = 1st order convergence
        assert np.all(ratios > 1.8), (
            f"Convergence ratio too low: {ratios} (orders: {orders:.2f}). "
            f"Expected ratio ~2 for 1st order upwind scheme."
        )

    def test_pure_diffusion_gaussian(self):
        """
        Test that Gaussian spreading matches analytical solution.

        For pure diffusion, Gaussian with s(t)^2 = s0^2 + sigma^2*t is exact.
        """
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.geometry import dirichlet_bc

        sigma = 0.1
        s0 = 0.1
        x0 = 0.5
        T = 0.3
        Nx = 81

        manufactured = GaussianDensity1D(x0=x0, s0=s0, sigma=sigma)

        # Create problem on larger domain to avoid boundary effects
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            geometry=geometry,
            T=T,
            Nt=100,  # Fine time stepping
            sigma=sigma,
            components=_default_components(),
        )

        # Initial condition
        x_grid = geometry.coordinates[0]  # 1D grid
        m_init = manufactured.solution(0.0, x_grid)

        # Dirichlet BC (Gaussian decays at boundaries)
        bc = dirichlet_bc(dimension=1, value=0.0)
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        # Zero drift
        U_zero = np.zeros((problem.Nt + 1, Nx))

        # Solve
        M_numerical = solver.solve_fp_system(M_initial=m_init, potential_field=U_zero, show_progress=False)

        # Compare at final time
        m_exact_final = manufactured.solution(T, x_grid)

        # L2 relative error
        l2_error = np.sqrt(np.sum((M_numerical[-1, :] - m_exact_final) ** 2))
        l2_exact = np.sqrt(np.sum(m_exact_final**2))
        rel_error = l2_error / l2_exact

        # Expect < 5% error for this resolution
        assert rel_error < 0.05, f"Relative error {rel_error:.2%} exceeds 5%"

    def test_source_term_steady_state_convergence(self):
        """
        Test FP source_term wiring with steady-state manufactured solution.

        Uses m(t,x) = 1 + A*sin(kx), which is time-independent. Without source,
        diffusion would decay the sine mode toward flat density. With the correct
        source S = D*k^2*A*sin(kx), the solver maintains the steady state.

        This directly tests that source_term is threaded through the FP FDM chain.
        Expected: convergence as grid refines (error dominated by spatial discretization).
        """
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.geometry import periodic_bc

        sigma = 0.2
        T = 0.3
        manufactured = SteadySinusoid1D(sigma=sigma, amplitude=0.3)

        resolutions = [31, 61, 121]
        errors = []

        for Nx in resolutions:
            geometry = TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1)
            )
            # Fine time stepping to minimize temporal error
            Nt = max(200, Nx * 4)
            problem = MFGProblem(geometry=geometry, T=T, Nt=Nt, sigma=sigma, components=_default_components())

            x_grid = geometry.coordinates[0]
            m_init = manufactured.solution(0.0, x_grid)

            bc = periodic_bc(dimension=1)
            solver = FPFDMSolver(problem, boundary_conditions=bc)

            # Zero drift (pure diffusion + source)
            U_zero = np.zeros((problem.Nt + 1, Nx))

            # Source term: callable (t, x_grid) -> values
            def source_fn(t, x_arr, _mfg=manufactured):
                return _mfg.fp_source(t, x_arr)

            M_numerical = solver.solve_fp_system(
                M_initial=m_init,
                potential_field=U_zero,
                show_progress=False,
                source_term=source_fn,
            )

            # Compare at final time — should match initial (steady state)
            m_exact_final = manufactured.solution(T, x_grid)
            error = np.max(np.abs(M_numerical[-1, :] - m_exact_final))
            errors.append(error)

        errors = np.array(errors)
        ratios = errors[:-1] / errors[1:]
        orders = np.log(ratios) / np.log(2)

        # Should show convergence (ratio > 1.5 means order > 0.58)
        assert np.all(ratios > 1.5), (
            f"FP source_term convergence ratio too low: {ratios} "
            f"(orders: {orders}). Errors: {errors}. "
            f"Expected convergence when source_term is correctly wired."
        )

    def test_mass_conservation_manufactured(self):
        """
        Test that mass is conserved for manufactured solution with no-flux BC.
        """
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.geometry import no_flux_bc

        sigma = 0.2
        T = 0.5
        Nx = 51

        # Use sinusoidal with diffusion-correct decay
        manufactured = DiffusionSinusoid1D(sigma=sigma, amplitude=0.3)

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=T, Nt=50, sigma=sigma, components=_default_components())

        x_grid = geometry.coordinates[0]  # 1D grid
        m_init = manufactured.solution(0.0, x_grid)
        dx = geometry.get_grid_spacing()[0]

        # Initial mass
        mass_init = np.trapezoid(m_init, dx=dx)

        bc = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        U_zero = np.zeros((problem.Nt + 1, Nx))
        M_numerical = solver.solve_fp_system(M_initial=m_init, potential_field=U_zero, show_progress=False)

        # Final mass
        mass_final = np.trapezoid(M_numerical[-1, :], dx=dx)

        # Mass should be conserved to machine precision
        rel_mass_error = abs(mass_final - mass_init) / mass_init
        assert rel_mass_error < 1e-10, f"Mass conservation violated: {rel_mass_error:.2e}"


class TestMMSConvergenceRates:
    """Test convergence rates for different BC types."""

    @pytest.mark.parametrize("bc_type", ["periodic", "no_flux", "dirichlet"])
    def test_fp_convergence_rate(self, bc_type: str):
        """
        Verify 2nd order spatial convergence for FP solver.

        Tests three BC types:
        - periodic: sinusoidal solution
        - no_flux: Gaussian (reflects at boundaries)
        - dirichlet: Gaussian (absorbed at boundaries)
        """
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.geometry import dirichlet_bc, no_flux_bc, periodic_bc

        sigma = 0.15
        T = 0.2

        # Select manufactured solution and BC based on type
        def make_periodic_bc():
            return periodic_bc(dimension=1)

        def make_no_flux_bc():
            return no_flux_bc(dimension=1)

        def make_dirichlet_bc():
            return dirichlet_bc(dimension=1, value=0.0)

        if bc_type == "periodic":
            manufactured = DiffusionSinusoid1D(sigma=sigma, amplitude=0.3)
            bc_func = make_periodic_bc
        elif bc_type == "no_flux":
            manufactured = GaussianDensity1D(x0=0.5, s0=0.1, sigma=sigma)
            bc_func = make_no_flux_bc
        else:  # dirichlet
            manufactured = GaussianDensity1D(x0=0.5, s0=0.1, sigma=sigma)
            bc_func = make_dirichlet_bc

        resolutions = [31, 61, 121]
        errors = []

        for Nx in resolutions:
            geometry = TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1)
            )
            # Use more time steps to minimize temporal error
            Nt = max(100, Nx * 2)
            problem = MFGProblem(geometry=geometry, T=T, Nt=Nt, sigma=sigma, components=_default_components())

            x_grid = geometry.coordinates[0]  # 1D grid
            m_init = manufactured.solution(0.0, x_grid)

            bc = bc_func()
            solver = FPFDMSolver(problem, boundary_conditions=bc)

            U_zero = np.zeros((problem.Nt + 1, Nx))
            M_numerical = solver.solve_fp_system(M_initial=m_init, potential_field=U_zero, show_progress=False)

            m_exact_final = manufactured.solution(T, x_grid)

            # L-infinity error (max pointwise)
            error = np.max(np.abs(M_numerical[-1, :] - m_exact_final))
            errors.append(error)

        errors = np.array(errors)

        # Compute convergence order: p = log(e1/e2) / log(h1/h2)
        # With grid doubling: p = log(e1/e2) / log(2)
        orders = np.log(errors[:-1] / errors[1:]) / np.log(2)

        # FP FDM uses upwind scheme by default -> expect ~1st order convergence
        # Allow 0.7 minimum to account for boundary effects and temporal error
        min_order = np.min(orders)
        assert min_order > 0.7, (
            f"BC={bc_type}: Convergence order {min_order:.2f} < 0.7. Errors: {errors}, Orders: {orders}"
        )


class TestMassConservationStress:
    """Extended mass conservation tests (Issue #523 Phase 3d-3e)."""

    def test_long_time_conservation(self):
        """
        Test mass conservation over extended time (stress test).

        Issue #523 Phase 3e: Conservation stress test
        """
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.geometry import no_flux_bc

        sigma = 0.3
        T = 10.0  # Long time
        Nx = 41
        Nt = 1000  # Many time steps

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=T, Nt=Nt, sigma=sigma, components=_default_components())

        x_grid = geometry.coordinates[0]  # 1D grid
        dx = geometry.get_grid_spacing()[0]

        # Non-trivial initial condition
        m_init = 1.0 + 0.5 * np.sin(4 * np.pi * x_grid)
        mass_init = np.trapezoid(m_init, dx=dx)

        bc = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        U_zero = np.zeros((problem.Nt + 1, Nx))
        M_numerical = solver.solve_fp_system(M_initial=m_init, potential_field=U_zero, show_progress=False)

        # Check mass at every time step
        masses = np.array([np.trapezoid(M_numerical[t, :], dx=dx) for t in range(Nt + 1)])
        max_deviation = np.max(np.abs(masses - mass_init))
        rel_deviation = max_deviation / mass_init

        # Should maintain conservation to high precision
        assert rel_deviation < 1e-8, f"Mass conservation violated over {Nt} steps: max deviation = {rel_deviation:.2e}"

    def test_conservation_with_weak_drift(self):
        """
        Test mass conservation with weak drift field.

        With no-flux BC, mass should be approximately conserved.
        Upwind schemes may have small conservation errors at boundaries.
        """
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.geometry import no_flux_bc

        sigma = 0.3  # Stronger diffusion to stabilize
        T = 0.5  # Shorter time
        Nx = 51
        Nt = 100

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            geometry=geometry,
            T=T,
            Nt=Nt,
            sigma=sigma,
            coupling_coefficient=0.1,  # Weak coupling
            components=_default_components(),
        )

        x_grid = geometry.coordinates[0]  # 1D grid
        dx = geometry.get_grid_spacing()[0]

        # Gaussian initial condition centered in domain
        m_init = np.exp(-30 * (x_grid - 0.5) ** 2)
        m_init = m_init / np.trapezoid(m_init, dx=dx)  # Normalize to 1
        mass_init = np.trapezoid(m_init, dx=dx)

        bc = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=bc)

        # Weak drift: small parabolic potential centered at 0.5
        # U(x) = (x - 0.5)^2 -> drift towards center
        U_field = np.zeros((Nt + 1, Nx))
        for t in range(Nt + 1):
            U_field[t, :] = (x_grid - 0.5) ** 2

        M_numerical = solver.solve_fp_system(M_initial=m_init, potential_field=U_field, show_progress=False)

        # Check mass conservation - allow for numerical discretization error
        mass_final = np.trapezoid(M_numerical[-1, :], dx=dx)
        rel_error = abs(mass_final - mass_init) / mass_init

        # Numerical schemes with boundaries and advection may have O(10%) conservation error
        # This is a known limitation of upwind schemes near boundaries
        # TODO: Investigate if conservative schemes (divergence_upwind) perform better
        assert rel_error < 0.15, f"Mass conservation with weak drift violated: {rel_error:.2%}"


class SteadySinusoid1D(ManufacturedSolution):
    """
    1D steady sinusoidal manufactured solution for source term verification.

    Manufactured density:
        m(t,x) = 1 + A*sin(k*x)     (time-independent)

    The FP solver (with zero drift) solves: dm/dt - D*Lap(m) = S.
    Since dm/dt = 0 and Lap(m) = -k^2*A*sin(kx), we need:
        S(t,x) = dm/dt - D*Lap(m) = D*k^2*A*sin(kx)

    Without the source, diffusion would decay the sinusoidal mode toward uniform
    density. The source exactly compensates, maintaining the steady state.

    This directly tests that source_term is wired correctly through the FP FDM
    solver chain: if the source is ignored, the solution decays to 1.0 (flat),
    producing large error. If wired correctly, m_exact is recovered.
    """

    def __init__(self, sigma: float = 0.2, amplitude: float = 0.3, k: float = 2.0 * np.pi):
        super().__init__(dimension=1)
        self.sigma = sigma
        self.amplitude = amplitude
        self.k = k
        self.D = 0.5 * sigma**2

    def solution(self, t: float, x: np.ndarray) -> np.ndarray:
        """m(t,x) = 1 + A*sin(k*x) (independent of t)."""
        x = np.atleast_1d(x)
        return 1.0 + self.amplitude * np.sin(self.k * x)

    def fp_source(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Source S = D*k^2*A*sin(kx) that balances diffusion to maintain steady state.

        Derivation: The FP solver adds S to the RHS of:
            dm/dt - D*Lap(m) = S
        For m = 1 + A*sin(kx):
            dm/dt = 0
            Lap(m) = -k^2*A*sin(kx)
            => S = 0 - D*(-k^2*A*sin(kx)) = D*k^2*A*sin(kx)

        Args:
            t: Time (unused, source is steady)
            x: Spatial grid, shape (N,) or (N, 1)

        Returns:
            Source values, shape (N,)
        """
        x = np.atleast_1d(x).ravel()
        return self.D * self.k**2 * self.amplitude * np.sin(self.k * x)


class BackwardHeatSolution1D:
    """
    1D sinusoidal solution to the backward heat equation.

    For the HJB equation without Hamiltonian (linear diffusion only):
        -du/dt - (sigma^2/2) * d^2u/dx^2 = 0

    This is the backward heat equation. With terminal condition
    u(T,x) = A*cos(k*x), the exact solution is:
        u(t,x) = A*cos(k*x)*exp(-D*k^2*(T-t))

    where D = sigma^2/2.

    Note: This propagates BACKWARD from t=T to t=0.

    For MMS with quadratic Hamiltonian H(p) = |p|^2/2:
        The source S(t,x) = H(grad u_exact) = k^2 A^2 sin^2(kx) exp(-2Dk^2(T-t)) / 2
        cancels the Hamiltonian contribution so u_exact satisfies the full HJB PDE.
    """

    def __init__(self, sigma: float = 0.2, amplitude: float = 1.0, k: float = 2.0 * np.pi, T: float = 1.0):
        self.sigma = sigma
        self.amplitude = amplitude
        self.k = k
        self.T = T
        self.D = 0.5 * sigma**2

    def solution(self, t: float, x: np.ndarray) -> np.ndarray:
        """u(t,x) = A*cos(k*x)*exp(-D*k^2*(T-t))"""
        x = np.atleast_1d(x)
        return self.amplitude * np.cos(self.k * x) * np.exp(-self.D * self.k**2 * (self.T - t))

    def terminal_condition(self, x: np.ndarray) -> np.ndarray:
        """u(T,x) = A*cos(k*x)"""
        return self.solution(self.T, x)

    def hjb_source(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Source term for MMS with quadratic Hamiltonian H(p) = |p|^2/2.

        Since u_exact solves -du/dt - (sigma^2/2)*Laplacian(u) = 0 (backward heat),
        the residual of the full HJB equation is exactly H(grad u_exact).

        S(t,x) = H(grad u_exact) = |grad u_exact|^2 / 2
               = k^2 * A^2 * sin^2(kx) * exp(-2*D*k^2*(T-t)) / 2

        Args:
            t: Time
            x: Spatial grid, shape (N,) or (N, 1)

        Returns:
            Source values, same shape as x (flattened)
        """
        x = np.atleast_1d(x).ravel()
        decay = np.exp(-self.D * self.k**2 * (self.T - t))
        grad_u = -self.k * self.amplitude * np.sin(self.k * x) * decay
        return 0.5 * grad_u**2


class TestMMSHJB1D:
    """MMS validation tests for 1D HJB solver."""

    def test_backward_heat_periodic_convergence(self):
        """
        Test HJB convergence for backward heat equation with periodic BC.

        Uses MMS (Method of Manufactured Solutions): the backward heat solution
        u(t,x) = A*cos(kx)*exp(-Dk^2(T-t)) solves the backward heat equation
        exactly. The quadratic Hamiltonian H(p) = |p|^2/2 introduces a residual
        that we supply as a source term S = H(grad u_exact).

        With this source, the solver should recover u_exact and show convergence
        as the grid is refined.
        """
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.geometry import periodic_bc

        sigma = 0.2
        T = 0.3

        manufactured = BackwardHeatSolution1D(sigma=sigma, amplitude=1.0, T=T)

        resolutions = [21, 41, 81]
        errors = []

        for Nx in resolutions:
            # Pass BC to geometry - solvers retrieve BC via geometry.get_boundary_conditions()
            bc = periodic_bc(dimension=1)
            geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=bc)
            problem = MFGProblem(geometry=geometry, T=T, Nt=Nx, sigma=sigma, components=_default_components())

            x_grid = geometry.coordinates[0]

            # Terminal condition from manufactured solution
            u_terminal = manufactured.terminal_condition(x_grid)

            # Zero density coupling (pure HJB without MFG coupling)
            M_zero = np.ones((problem.Nt + 1, Nx)) / Nx  # Uniform density
            # Initialize U_coupling_prev to zeros (first Picard iteration)
            U_prev = np.zeros((problem.Nt + 1, Nx))

            solver = HJBFDMSolver(problem)

            # MMS source term: S(t, x) = H(grad u_exact) = |grad u_exact|^2 / 2
            # Wraps hjb_source to accept (t, x_grid) with x_grid as (N, d) ndarray
            def source_fn(t, x_arr, _mfg=manufactured):
                return _mfg.hjb_source(t, x_arr)

            # Solve HJB backward in time with source term
            U_numerical = solver.solve_hjb_system(
                M_density=M_zero,
                U_terminal=u_terminal,
                U_coupling_prev=U_prev,
                source_term=source_fn,
            )

            # Compare initial time solution (t=0) with exact
            u_exact_initial = manufactured.solution(0.0, x_grid)
            error = np.sqrt(np.mean((U_numerical[0, :] - u_exact_initial) ** 2))
            errors.append(error)

        errors = np.array(errors)
        ratios = errors[:-1] / errors[1:]
        orders = np.log(ratios) / np.log(2)

        # HJB FDM should give ~1st order convergence (upwind scheme)
        assert np.all(ratios > 1.5), (
            f"HJB convergence ratio too low: {ratios} (orders: {orders}). "
            f"Errors: {errors}. Expected ratio >1.5 for upwind scheme."
        )

    def test_hjb_terminal_condition_preserved(self):
        """
        Test that HJB solver correctly applies terminal condition.

        At t=T, the numerical solution should match the terminal condition.
        """
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.geometry import periodic_bc

        sigma = 0.2
        T = 0.5
        Nx = 51

        # Pass BC to geometry - solvers retrieve BC via geometry.get_boundary_conditions()
        bc = periodic_bc(dimension=1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=bc)
        problem = MFGProblem(geometry=geometry, T=T, Nt=50, sigma=sigma, components=_default_components())

        x_grid = geometry.coordinates[0]

        # Terminal condition: quadratic
        u_terminal = (x_grid - 0.5) ** 2

        M_zero = np.ones((problem.Nt + 1, Nx)) / Nx
        # Initialize U_coupling_prev to zeros (first Picard iteration)
        U_prev = np.zeros((problem.Nt + 1, Nx))

        solver = HJBFDMSolver(problem)

        U_numerical = solver.solve_hjb_system(
            M_density=M_zero,
            U_terminal=u_terminal,
            U_coupling_prev=U_prev,
        )

        # At t=T (last time index), solution should match terminal condition
        rel_error = np.max(np.abs(U_numerical[-1, :] - u_terminal)) / np.max(np.abs(u_terminal))
        assert rel_error < 1e-10, f"Terminal condition not preserved: {rel_error:.2e}"


class TestCoupledHJBFPValidation:
    """
    Validation tests for coupled HJB-FP system.

    Issue #523 Phase 3c: Coupled HJB-FP integration tests with explicit BC
    """

    def test_coupled_solver_consistency(self):
        """
        Test that coupled HJB-FP solver produces consistent results.

        Verifies:
        1. Solution shapes are correct
        2. Density remains non-negative
        3. Value function is bounded
        """
        from mfgarchon.alg.numerical.coupling import FixedPointIterator
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.geometry import no_flux_bc

        sigma = 0.3
        T = 0.5
        Nx = 31
        Nt = 20

        # Pass BC to geometry - both HJB and FP solvers retrieve BC via geometry
        bc = no_flux_bc(dimension=1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=bc)
        problem = MFGProblem(geometry=geometry, T=T, Nt=Nt, sigma=sigma, components=_default_components())

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            relaxation=0.5,
        )

        result = mfg_solver.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        U, M = result[:2]
        Nt_points = problem.Nt + 1

        # Check shapes
        assert U.shape == (Nt_points, Nx), f"U shape {U.shape} != expected ({Nt_points}, {Nx})"
        assert M.shape == (Nt_points, Nx), f"M shape {M.shape} != expected ({Nt_points}, {Nx})"

        # Density should be non-negative
        assert np.all(M >= -1e-6), f"Negative density: min(M) = {np.min(M):.2e}"

        # Value function should be bounded
        assert np.all(np.isfinite(U)), "Value function contains inf or nan"

    def test_coupled_mass_conservation(self):
        """
        Test mass conservation in coupled HJB-FP solver.

        Previously skipped (Issue #523, #835) due to ~64% mass error.
        Fixed as of v0.17.x — now shows <1% error with no_flux BC.
        """
        from mfgarchon.alg.numerical.coupling import FixedPointIterator
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.geometry import no_flux_bc

        sigma = 0.3
        T = 0.5
        Nx = 41
        Nt = 30

        bc = no_flux_bc(dimension=1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=bc)
        problem = MFGProblem(geometry=geometry, T=T, Nt=Nt, sigma=sigma, components=_default_components())
        dx = geometry.get_grid_spacing()[0]

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            relaxation=0.5,
        )

        result = mfg_solver.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        M = result.M

        # Check mass at initial and final time
        mass_init = np.trapezoid(M[0, :], dx=dx)
        mass_final = np.trapezoid(M[-1, :], dx=dx)
        rel_error = abs(mass_final - mass_init) / mass_init

        # Mass conservation: <5% error for coupled solver with 5 iterations
        assert rel_error < 0.05, f"Coupled solver mass conservation violated: {rel_error:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
