#!/usr/bin/env python3
"""
Integration tests for 2D coupled HJB-FP system via FixedPointIterator.

Tests the full MFG coupling loop in 2D: HJB solver + FP solver + FixedPointIterator.
This fills the gap identified in Issue #762 -- existing tests cover 2D HJB-only
and 1D coupled, but not 2D coupled.

Problem setup: Linear-Quadratic MFG on [-1,1]^2
  - H(x, m, p) = |p|^2/2 + m  (quadratic control + linear coupling)
  - m_initial(x) = exp(-5|x|^2)  (Gaussian IC, provided as ndarray)
  - u_terminal(x) = 0.5|x|^2  (quadratic terminal cost)

Note: m_initial is provided as ndarray to work around Issue #777
(_setup_custom_initial_density broken for 2D callable).

Stability regime for correct ndarray IC with Picard damping=0.3:
  - T=0.2, sigma=0.1: stable up to 8+ iterations (preferred default)
  - T=0.3, sigma>=0.5: stable up to 8+ iterations
  - T=0.3, sigma=0.1: UNSTABLE (diverges at iteration 5)
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid, no_flux_bc

# ---------------------------------------------------------------------------
# Module-level helpers (prefixed with _ per test conventions)
# ---------------------------------------------------------------------------


def _default_hamiltonian():
    """H = |p|^2/2 + m (quadratic control + linear coupling)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _gaussian_ic_2d(N):
    """Build 2D Gaussian initial density on [-1,1]^2 grid with N+1 points per axis."""
    x = np.linspace(-1, 1, N + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.exp(-5 * (X**2 + Y**2))


def _default_components_2d(N):
    """2D Gaussian IC (ndarray) + quadratic terminal cost (callable).

    m_initial provided as ndarray to work around Issue #777:
    _setup_custom_initial_density does not evaluate callable correctly for 2D.
    """
    return MFGComponents(
        m_initial=_gaussian_ic_2d(N),
        u_terminal=lambda x: 0.5 * np.sum(np.asarray(x) ** 2),
        hamiltonian=_default_hamiltonian(),
    )


def _create_2d_problem(N=10, T=0.2, Nt=10, sigma=0.1, bc=None):
    """Create 2D MFG problem using modern API.

    Default parameters chosen for numerical stability of the coupled system
    with correct ndarray IC:
    - N=10: moderate resolution (11x11 grid)
    - T=0.2, Nt=10: fine temporal discretization (dt=0.02), short horizon
    - sigma=0.1: modest diffusion
    """
    if bc is not None:
        geometry = TensorProductGrid(
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            Nx_points=[N + 1, N + 1],
            boundary_conditions=bc,
        )
        return MFGProblem(
            geometry=geometry,
            components=_default_components_2d(N),
            T=T,
            Nt=Nt,
            sigma=sigma,
        )
    return MFGProblem(
        spatial_bounds=[(-1, 1), (-1, 1)],
        spatial_discretization=[N, N],
        T=T,
        Nt=Nt,
        sigma=sigma,
        components=_default_components_2d(N),
    )


def _create_2d_mfg_solver(problem, damping=0.3):
    """Create coupled solver: HJB + FP + FixedPointIterator.

    Uses fixed-point HJB solver (not Newton) for 2D stability -- Newton Jacobians
    become singular during coupled iterations with evolving density.
    Picard damping=0.3 is conservative but stable for 2D coupled problems.
    """
    hjb = HJBFDMSolver(
        problem,
        solver_type="fixed_point",
        damping_factor=0.8,
        max_newton_iterations=50,
    )
    fp = FPFDMSolver(problem)
    return FixedPointIterator(
        problem,
        hjb_solver=hjb,
        fp_solver=fp,
        relaxation=damping,
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestCoupledHJBFP2DBasic:
    """Core 2D coupling functionality."""

    @pytest.mark.slow
    def test_2d_coupled_solve_produces_valid_output(self):
        """Run FixedPointIterator on 2D LQ problem; verify shapes and finiteness."""
        N = 10
        problem = _create_2d_problem(N=N, T=0.2, Nt=10)
        solver = _create_2d_mfg_solver(problem)

        result = solver.solve(max_iterations=8, tolerance=1e-4)

        U, M = result[:2]
        grid_shape = problem.geometry.get_grid_shape()
        expected = (problem.Nt + 1, *grid_shape)

        assert U.shape == expected, f"U shape {U.shape} != expected {expected}"
        assert M.shape == expected, f"M shape {M.shape} != expected {expected}"
        assert np.all(np.isfinite(U)), "U contains non-finite values"
        assert np.all(np.isfinite(M)), "M contains non-finite values"
        assert np.all(M >= -1e-10), "Density contains negative values"

    def test_2d_output_shapes(self):
        """Verify U and M shapes match (Nt+1, Nx, Ny) from geometry."""
        N = 10
        problem = _create_2d_problem(N=N, T=0.2, Nt=6)
        solver = _create_2d_mfg_solver(problem)

        result = solver.solve(max_iterations=3, tolerance=1e-3)

        U, M = result[:2]
        Nx, Ny = problem.geometry.get_grid_shape()
        assert U.shape == (problem.Nt + 1, Nx, Ny)
        assert M.shape == (problem.Nt + 1, Nx, Ny)

    @pytest.mark.slow
    def test_2d_terminal_condition_preserved(self):
        """After coupled solve, U[-1] should match the terminal cost."""
        N = 10
        problem = _create_2d_problem(N=N, T=0.2, Nt=10)
        solver = _create_2d_mfg_solver(problem)

        result = solver.solve(max_iterations=8, tolerance=1e-4)

        U = result[0]

        # Build expected terminal cost on the grid
        x = np.linspace(-1, 1, N + 1)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u_terminal_expected = 0.5 * (X**2 + Y**2)

        assert np.allclose(U[-1], u_terminal_expected, atol=1e-8), (
            f"Terminal condition violated: max diff = {np.max(np.abs(U[-1] - u_terminal_expected))}"
        )

    @pytest.mark.slow
    def test_2d_initial_density_preserved(self):
        """After coupled solve, M[0] should match the (normalized) initial condition."""
        N = 10
        problem = _create_2d_problem(N=N, T=0.2, Nt=10)
        solver = _create_2d_mfg_solver(problem)

        result = solver.solve(max_iterations=8, tolerance=1e-4)

        M = result[1]

        # M[0] is the normalized initial density (MFGProblem normalizes m_initial)
        m_init_from_problem = problem.get_m_initial()
        assert np.allclose(M[0], m_init_from_problem, atol=1e-8), (
            f"Initial density violated: max diff = {np.max(np.abs(M[0] - m_init_from_problem))}"
        )


class TestCoupledHJBFP2DMassConservation:
    """Mass conservation in 2D coupled system."""

    @pytest.mark.slow
    def test_2d_mass_conservation(self):
        """With no-flux BC, total mass should stay within 15% of initial."""
        N = 10
        bc = no_flux_bc(dimension=2)
        problem = _create_2d_problem(N=N, T=0.2, Nt=10, sigma=0.1, bc=bc)
        solver = _create_2d_mfg_solver(problem)

        result = solver.solve(max_iterations=8, tolerance=1e-4)

        M = result[1]

        # 2D trapezoidal integration
        x = np.linspace(-1, 1, N + 1)
        initial_mass = np.trapezoid(np.trapezoid(M[0], x, axis=1), x)

        for t_idx in range(M.shape[0]):
            mass_t = np.trapezoid(np.trapezoid(M[t_idx], x, axis=1), x)
            rel_change = abs(mass_t - initial_mass) / max(abs(initial_mass), 1e-15)
            assert rel_change < 0.15, (
                f"Mass conservation violated at t_idx={t_idx}: "
                f"initial={initial_mass:.6f}, current={mass_t:.6f}, rel_change={rel_change:.4f}"
            )


class TestCoupledHJBFP2DConvergence:
    """Convergence properties of 2D coupled solver."""

    @pytest.mark.slow
    def test_2d_error_decreases(self):
        """Minimum error should be substantially less than initial error."""
        N = 10
        problem = _create_2d_problem(N=N, T=0.2, Nt=10)
        solver = _create_2d_mfg_solver(problem, damping=0.2)

        result = solver.solve(max_iterations=8, tolerance=1e-6)

        err_U = result.error_history_U
        # Minimum error should be < 50% of initial error
        assert np.min(err_U) < 0.5 * err_U[0], (
            f"Error did not decrease enough: initial={err_U[0]:.6e}, min={np.min(err_U):.6e}"
        )

    @pytest.mark.slow
    def test_2d_grid_refinement(self):
        """Both coarse and fine grids should produce finite solutions."""
        # Issue #787: With explicit diffusion -(sigma^2/2)*Laplacian(U), the CFL
        # constraint requires finer time steps. For N=15, sigma=0.5:
        #   D=0.125, sum(1/dx^2)=112.5, need dt < 0.5/(D*sum) = 0.036
        #   Nt=20 gives dt=0.015, CFL=0.21 (stable)
        for N in (10, 15):
            problem = _create_2d_problem(N=N, T=0.3, Nt=20, sigma=0.5)
            solver = _create_2d_mfg_solver(problem)

            result = solver.solve(max_iterations=8, tolerance=1e-4)

            U, M = result[:2]
            assert np.all(np.isfinite(U)), f"U not finite for N={N}"
            assert np.all(np.isfinite(M)), f"M not finite for N={N}"


class TestCoupledHJBFP2DSymmetry:
    """Physical symmetry properties of 2D solutions."""

    @pytest.mark.slow
    def test_2d_solution_symmetry(self):
        """Radially symmetric IC/terminal should produce approximately symmetric U."""
        N = 10
        problem = _create_2d_problem(N=N, T=0.2, Nt=10)
        solver = _create_2d_mfg_solver(problem)

        result = solver.solve(max_iterations=8, tolerance=1e-5)

        U = result[0]

        # Check U symmetry at middle time step (initial time is most affected
        # by coupling iterations; terminal time is exactly symmetric by construction)
        t_mid = U.shape[0] // 2
        arr = U[t_mid]
        norm = np.linalg.norm(arr)

        # y-symmetry: U(x,y) ~= U(x,-y) -- domain and IC are y-symmetric
        y_err = np.linalg.norm(arr - np.flip(arr, axis=1)) / norm
        assert y_err < 0.05, f"U y-symmetry error {y_err:.4f} > 0.05"

        # x-symmetry: U(x,y) ~= U(-x,y) -- domain and IC are x-symmetric
        x_err = np.linalg.norm(arr - np.flip(arr, axis=0)) / norm
        assert x_err < 0.05, f"U x-symmetry error {x_err:.4f} > 0.05"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
