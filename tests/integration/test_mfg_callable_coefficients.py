#!/usr/bin/env python3
"""
Integration tests for MFG with callable (state-dependent) coefficients (Phase 2.3).

Tests the full MFG coupling with state-dependent diffusion and drift,
verifying that callable coefficients work correctly in fixed-point iteration.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


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


class TestMFGCallableCoefficients:
    """Integration tests for MFG with callable coefficients (Phase 2.3).

    Tests the full MFG coupling with state-dependent diffusion and drift,
    verifying that callable coefficients work correctly in fixed-point iteration.
    Both HJB-FDM and FP-FDM now support callable diffusion for 1D problems.
    """

    def test_mfg_with_callable_diffusion(self):
        """Test MFG with state-dependent diffusion: porous medium."""
        # Create problem
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[31]
        )  # Nx=30 intervals
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, sigma=0.1, components=_default_components())

        # Porous medium diffusion: D(m) = σ² m
        def porous_medium_diffusion(t, x, m):
            return 0.05 * m  # Diffusion proportional to density

        # Create solvers
        # Use divergence_upwind for strict positivity preservation (porous medium needs it)
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")

        # Create MFG solver with callable diffusion
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            relaxation=0.5,
            volatility_field=porous_medium_diffusion,
        )

        # Solve
        result = mfg_solver.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Verify result structure
        assert result is not None
        U, M = result[:2]
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        assert U.shape == (Nt_points, Nx_points)
        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)  # divergence_upwind guarantees non-negativity

    def test_mfg_with_density_dependent_diffusion(self):
        """Test MFG with crowd dynamics: D(m) = D0 + D1(1 - m/m_max)."""
        # Create problem
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[31]
        )  # Nx=30 intervals
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, sigma=0.1, components=_default_components())

        # Crowd diffusion: lower diffusion in high-density regions
        def crowd_diffusion(t, x, m):
            m_max = np.max(m) if np.max(m) > 0 else 1.0
            return 0.05 + 0.1 * (1 - m / m_max)

        # Create solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Create MFG solver
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            relaxation=0.5,
            volatility_field=crowd_diffusion,
        )

        # Solve
        result = mfg_solver.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Verify convergence
        U, M = result[:2]
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        assert U.shape == (Nt_points, Nx_points)
        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= -1e-6)  # Allow small numerical noise

    def test_mfg_callable_vs_constant_convergence(self):
        """Test that callable returning constant matches constant diffusion."""
        # Create problem
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[31]
        )  # Nx=30 intervals
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, sigma=0.15, components=_default_components())

        # Callable returning constant
        def constant_diffusion(t, x, m):
            return 0.15

        # Solve with callable
        hjb_solver_callable = HJBFDMSolver(problem)
        fp_solver_callable = FPFDMSolver(problem)
        mfg_solver_callable = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver_callable,
            fp_solver=fp_solver_callable,
            relaxation=0.5,
            volatility_field=constant_diffusion,
        )
        result_callable = mfg_solver_callable.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Solve with constant (None uses problem.sigma)
        hjb_solver_constant = HJBFDMSolver(problem)
        fp_solver_constant = FPFDMSolver(problem)
        mfg_solver_constant = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver_constant,
            fp_solver=fp_solver_constant,
            relaxation=0.5,
            volatility_field=None,  # Use problem.sigma
        )
        result_constant = mfg_solver_constant.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Results should be similar (not exact due to numerical differences)
        U_callable, M_callable = result_callable[:2]
        U_constant, M_constant = result_constant[:2]

        # Check that solutions are reasonably close
        assert np.allclose(U_callable, U_constant, rtol=0.1, atol=1e-2)
        assert np.allclose(M_callable, M_constant, rtol=0.1, atol=1e-2)

    def test_mfg_callable_diffusion_with_array(self):
        """Test MFG with array diffusion (non-callable) for comparison."""
        # Create problem
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[31]
        )  # Nx=30 intervals
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, sigma=0.1, components=_default_components())

        # Spatially varying diffusion (higher at boundaries)
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        bounds = problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]
        x_grid = np.linspace(xmin, xmax, Nx_points)
        diffusion_array = 0.1 + 0.05 * np.abs(x_grid - 0.5)

        # Broadcast to all timesteps
        volatility_field = np.tile(diffusion_array, (Nt_points, 1))

        # Create solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Create MFG solver with array diffusion
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            relaxation=0.5,
            volatility_field=volatility_field,
        )

        # Solve
        result = mfg_solver.solve(max_iterations=5, tolerance=1e-3, verbose=False)

        # Verify
        U, M = result[:2]
        assert U.shape == (Nt_points, Nx_points)
        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= -1e-6)  # Allow small numerical noise

    def test_mfg_callable_with_small_iterations(self):
        """Test that callable diffusion works with few Picard iterations."""
        # Create small problem
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[21]
        )  # Nx=20 intervals
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=10, sigma=0.1, components=_default_components())

        # Simple state-dependent diffusion
        def state_diffusion(t, x, m):
            return 0.08 + 0.02 * m

        # Create solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Create MFG solver
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            relaxation=0.5,
            volatility_field=state_diffusion,
        )

        # Solve with just 2 iterations
        result = mfg_solver.solve(max_iterations=2, tolerance=1e-6, verbose=False)

        # Verify it runs (may not converge, but should execute)
        U, M = result[:2]
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        assert U.shape == (Nt_points, Nx_points)
        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= -1e-6)  # Allow small numerical noise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
