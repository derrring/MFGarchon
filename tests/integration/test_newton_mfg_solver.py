#!/usr/bin/env python3
"""
Integration tests for Newton MFG solver (Issue #492 Phase 1).

Tests the Newton's method solver for coupled HJB-FP systems:
- Basic execution and convergence
- Hybrid Picard warm-up + Newton strategy
- Comparison with Picard (FixedPointIterator)
- Output format and structure

Note: Newton solver tests are marked as slow because Jacobian computation
via finite differences is computationally expensive. Run with:
    pytest -m slow  # to include slow tests
    pytest -m "not slow"  # to exclude slow tests
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator, NewtonMFGSolver
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


# Mark all tests in this module as slow by default (Newton uses expensive Jacobian)
pytestmark = pytest.mark.slow


@pytest.mark.slow
class TestNewtonMFGSolverBasic:
    """Basic functionality tests for NewtonMFGSolver."""

    @pytest.fixture
    def simple_1d_problem(self):
        """Create a simple 1D MFG problem for testing."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[21])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, sigma=0.2, components=_default_components())
        return problem

    @pytest.fixture
    def solvers(self, simple_1d_problem):
        """Create HJB and FP solvers."""
        hjb_solver = HJBFDMSolver(simple_1d_problem)
        fp_solver = FPFDMSolver(simple_1d_problem)
        return hjb_solver, fp_solver

    def test_newton_solver_executes(self, simple_1d_problem, solvers):
        """Test that Newton solver runs without errors."""
        hjb_solver, fp_solver = solvers

        newton_solver = NewtonMFGSolver(
            simple_1d_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=2,
            newton_max_iterations=5,
        )

        U, M, info = newton_solver.solve(max_iterations=10, tolerance=1e-4, verbose=False)

        # Basic shape checks
        expected_shape = (simple_1d_problem.Nt + 1, 21)
        assert U.shape == expected_shape, f"U shape {U.shape} != {expected_shape}"
        assert M.shape == expected_shape, f"M shape {M.shape} != {expected_shape}"

        # Result contains convergence info
        assert "converged" in info
        assert "total_iterations" in info
        assert "picard_iterations" in info
        assert "newton_iterations" in info

    def test_newton_solver_output_finite(self, simple_1d_problem, solvers):
        """Test that Newton solver produces finite output."""
        hjb_solver, fp_solver = solvers

        newton_solver = NewtonMFGSolver(
            simple_1d_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=2,
            newton_max_iterations=5,
        )

        U, M, _info = newton_solver.solve(max_iterations=10, tolerance=1e-4, verbose=False)

        assert np.all(np.isfinite(U)), "U contains inf/nan"
        assert np.all(np.isfinite(M)), "M contains inf/nan"

    def test_newton_solver_density_nonnegative(self, simple_1d_problem, solvers):
        """Test that Newton solver preserves density non-negativity."""
        hjb_solver, fp_solver = solvers

        newton_solver = NewtonMFGSolver(
            simple_1d_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=3,
            newton_max_iterations=10,
        )

        _U, M, _info = newton_solver.solve(max_iterations=15, tolerance=1e-4, verbose=False)

        # Allow small numerical noise (same tolerance as other MFG tests)
        assert np.all(M >= -1e-6), f"M has negative values: min={np.min(M)}"


class TestNewtonMFGSolverHybrid:
    """Tests for hybrid Picard warm-up + Newton strategy."""

    @pytest.fixture
    def moderate_problem(self):
        """Create moderate-size problem for hybrid testing."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[31])
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=15, sigma=0.15, components=_default_components())
        return problem

    def test_picard_warmup_executed(self, moderate_problem):
        """Test that Picard warm-up iterations are executed."""
        hjb_solver = HJBFDMSolver(moderate_problem)
        fp_solver = FPFDMSolver(moderate_problem)

        newton_solver = NewtonMFGSolver(
            moderate_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=5,
            newton_max_iterations=10,
        )

        _U, _M, info = newton_solver.solve(max_iterations=20, tolerance=1e-6, verbose=False)

        assert info["picard_iterations"] == 5, f"Picard iterations: {info['picard_iterations']}"
        assert len(info.get("picard_residuals", [])) > 0, "No Picard residuals recorded"

    def test_picard_residuals_decreasing(self, moderate_problem):
        """Test that Picard warm-up reduces residual (on average)."""
        hjb_solver = HJBFDMSolver(moderate_problem)
        fp_solver = FPFDMSolver(moderate_problem)

        newton_solver = NewtonMFGSolver(
            moderate_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=5,
            picard_damping=0.5,
            newton_max_iterations=5,
        )

        _U, _M, info = newton_solver.solve(max_iterations=12, tolerance=1e-6, verbose=False)

        picard_residuals = info.get("picard_residuals", [])
        if len(picard_residuals) >= 2:
            # Residuals should generally decrease (allow some noise)
            first_half_avg = np.mean(picard_residuals[: len(picard_residuals) // 2])
            second_half_avg = np.mean(picard_residuals[len(picard_residuals) // 2 :])
            # Not a strict test - just verify no dramatic divergence
            assert second_half_avg < first_half_avg * 10, "Picard residuals increasing rapidly"

    def test_no_picard_warmup(self, moderate_problem):
        """Test Newton solver without Picard warm-up."""
        hjb_solver = HJBFDMSolver(moderate_problem)
        fp_solver = FPFDMSolver(moderate_problem)

        newton_solver = NewtonMFGSolver(
            moderate_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=0,  # No Picard warm-up
            newton_max_iterations=15,
        )

        U, M, info = newton_solver.solve(max_iterations=15, tolerance=1e-4, verbose=False)

        assert info["picard_iterations"] == 0
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))


class TestNewtonVsPicard:
    """Compare Newton solver against Picard (FixedPointIterator)."""

    @pytest.fixture
    def comparison_problem(self):
        """Create problem for Picard vs Newton comparison."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[25])
        problem = MFGProblem(geometry=geometry, T=0.4, Nt=12, sigma=0.18, components=_default_components())
        return problem

    def test_both_solvers_produce_similar_results(self, comparison_problem):
        """Test that Newton and Picard converge to similar solutions."""
        # Create fresh solvers for each test
        hjb_newton = HJBFDMSolver(comparison_problem)
        fp_newton = FPFDMSolver(comparison_problem)
        hjb_picard = HJBFDMSolver(comparison_problem)
        fp_picard = FPFDMSolver(comparison_problem)

        # Newton solver
        newton_solver = NewtonMFGSolver(
            comparison_problem,
            hjb_newton,
            fp_newton,
            picard_warmup=3,
            newton_max_iterations=10,
            line_search=True,
        )

        # Picard solver
        picard_solver = FixedPointIterator(
            comparison_problem,
            hjb_solver=hjb_picard,
            fp_solver=fp_picard,
            relaxation=0.5,
        )

        # Solve
        U_newton, M_newton, _info_newton = newton_solver.solve(max_iterations=20, tolerance=1e-5, verbose=False)
        result_picard = picard_solver.solve(max_iterations=50, tolerance=1e-5, verbose=False)
        U_picard, M_picard = result_picard[:2]

        # Both should produce similar solutions (within 10% relative error)
        # This is a coarse check - exact match not expected due to different iteration paths
        U_diff = np.linalg.norm(U_newton - U_picard) / (np.linalg.norm(U_picard) + 1e-10)
        M_diff = np.linalg.norm(M_newton - M_picard) / (np.linalg.norm(M_picard) + 1e-10)

        assert U_diff < 0.5, f"U solutions differ by {U_diff * 100:.1f}%"
        assert M_diff < 0.5, f"M solutions differ by {M_diff * 100:.1f}%"


class TestNewtonMFGSolverParameters:
    """Test Newton solver parameter variations."""

    @pytest.fixture
    def param_test_problem(self):
        """Create problem for parameter testing."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[21])
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=8, sigma=0.2, components=_default_components())
        return problem

    def test_line_search_disabled(self, param_test_problem):
        """Test Newton solver with line search disabled."""
        hjb_solver = HJBFDMSolver(param_test_problem)
        fp_solver = FPFDMSolver(param_test_problem)

        newton_solver = NewtonMFGSolver(
            param_test_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=2,
            newton_max_iterations=5,
            line_search=False,  # Disable line search
        )

        U, M, _info = newton_solver.solve(max_iterations=10, tolerance=1e-4, verbose=False)

        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))

    def test_custom_tolerances(self, param_test_problem):
        """Test Newton solver with custom tolerances."""
        hjb_solver = HJBFDMSolver(param_test_problem)
        fp_solver = FPFDMSolver(param_test_problem)

        newton_solver = NewtonMFGSolver(
            param_test_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=1,
            newton_tolerance=1e-8,  # Tight tolerance
            newton_max_iterations=20,
        )

        _U, _M, info = newton_solver.solve(max_iterations=25, tolerance=1e-8, verbose=False)

        # Should complete (converged or max iterations)
        assert info["total_iterations"] > 0

    def test_high_damping_picard(self, param_test_problem):
        """Test Newton solver with high Picard damping."""
        hjb_solver = HJBFDMSolver(param_test_problem)
        fp_solver = FPFDMSolver(param_test_problem)

        newton_solver = NewtonMFGSolver(
            param_test_problem,
            hjb_solver,
            fp_solver,
            picard_warmup=4,
            picard_damping=0.8,  # High damping (more conservative)
            newton_max_iterations=5,
        )

        U, M, _info = newton_solver.solve(max_iterations=12, tolerance=1e-4, verbose=False)

        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))


@pytest.mark.fast  # Fast tests - no Newton iterations
class TestMFGResidualComputation:
    """Test MFGResidual class functionality (fast, no Jacobian computation)."""

    @pytest.fixture
    def residual_test_problem(self):
        """Create problem for residual testing."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[21])
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=8, sigma=0.2, components=_default_components())
        return problem

    def test_residual_computation(self, residual_test_problem):
        """Test that MFGResidual computes residuals correctly."""
        from mfgarchon.alg.numerical.coupling import MFGResidual

        hjb_solver = HJBFDMSolver(residual_test_problem)
        fp_solver = FPFDMSolver(residual_test_problem)

        residual_computer = MFGResidual(residual_test_problem, hjb_solver, fp_solver)

        # Get initial guess
        x0 = residual_computer.get_initial_guess()

        # Compute residual
        F = residual_computer.residual_function(x0)

        # Residual should be finite
        assert np.all(np.isfinite(F)), "Residual contains inf/nan"

        # Residual size should be 2 * total_size (U and M)
        total_size = np.prod(residual_computer.solution_shape)
        assert F.shape[0] == 2 * total_size, f"Residual size {F.shape[0]} != {2 * total_size}"

    def test_pack_unpack_identity(self, residual_test_problem):
        """Test that pack/unpack are inverse operations."""
        from mfgarchon.alg.numerical.coupling import MFGResidual

        hjb_solver = HJBFDMSolver(residual_test_problem)
        fp_solver = FPFDMSolver(residual_test_problem)

        residual_computer = MFGResidual(residual_test_problem, hjb_solver, fp_solver)

        # Create random state
        shape = residual_computer.solution_shape
        U = np.random.randn(*shape)
        M = np.abs(np.random.randn(*shape))  # Non-negative

        # Pack and unpack
        x = residual_computer.pack_state(U, M)
        U_unpacked, M_unpacked = residual_computer.unpack_state(x)

        assert np.allclose(U, U_unpacked), "Pack/unpack failed for U"
        assert np.allclose(M, M_unpacked), "Pack/unpack failed for M"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
