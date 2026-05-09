#!/usr/bin/env python3
"""
Unit tests for HJBSemiLagrangianSolver.

Tests the semi-Lagrangian method for solving Hamilton-Jacobi-Bellman equations
in Mean Field Games, including characteristic-following schemes and interpolation.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
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


class TestHJBSemiLagrangianInitialization:
    """Test HJBSemiLagrangianSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        assert solver.hjb_method_name == "Semi-Lagrangian"
        assert solver.interpolation_method == "linear"
        assert solver.optimization_method == "brent"
        assert solver.characteristic_solver == "explicit_euler"
        assert solver.tolerance == 1e-8

    def test_custom_interpolation_method(self):
        """Test initialization with custom interpolation method."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")

        assert solver.interpolation_method == "cubic"

    def test_custom_optimization_method(self):
        """Test initialization with custom optimization method."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, optimization_method="golden")

        assert solver.optimization_method == "golden"

    def test_custom_characteristic_solver(self):
        """Test initialization with custom characteristic solver."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2")

        assert solver.characteristic_solver == "rk2"

    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, tolerance=1e-10)

        assert solver.tolerance == 1e-10

    def test_grid_parameters_computed(self):
        """Test that grid parameters are properly computed."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        assert hasattr(solver, "x_grid")
        assert hasattr(solver, "dt")
        assert hasattr(solver, "dx")
        assert len(solver.x_grid) == problem.geometry.get_grid_shape()[0]
        assert np.isclose(solver.dt, problem.dt)
        assert np.isclose(solver.dx, problem.geometry.get_grid_spacing()[0])


class TestHJBSemiLagrangianSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs: Nx, Nt are intervals; knots = intervals + 1
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Output: same shape as input density (Nt+1 time points)
        assert U_solution.shape == (problem.Nt + 1, Nx_points)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs with specific final condition
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - bounds[1][0]) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)

    def test_solve_hjb_system_backward_propagation(self):
        """Test that solution propagates backward in time."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (values at earlier times should be influenced by final condition)
        # Check that solution at t=0 is different from zero
        assert not np.allclose(U_solution[0, :], 0.0)


class TestHJBSemiLagrangianNumericalProperties:
    """Test numerical properties of the semi-Lagrangian method."""

    @pytest.mark.skip(
        reason="Semi-Lagrangian method can have numerical overflow issues with certain configurations (Issue #600)"
    )
    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=40, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) * 0.5
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    @pytest.mark.skip(reason="Semi-Lagrangian method can have numerical overflow issues with certain configurations")
    def test_solution_smoothness(self):
        """Test that solution has reasonable smoothness."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check spatial smoothness - finite differences shouldn't be too large
        U_diff = np.diff(U_solution, axis=1)
        assert np.max(np.abs(U_diff)) < 100.0


class TestHJBSemiLagrangianIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_uniform_density(self):
        """Test solver with uniform density distribution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Uniform density
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)

        # Simple final condition
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = (x_coords - 0.5) ** 2

        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_solver_with_gaussian_density(self):
        """Test solver with Gaussian density distribution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Gaussian density
        bounds = problem.geometry.get_bounds()
        Nx_points = problem.geometry.get_grid_shape()[0]
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_profile = m_profile / np.sum(m_profile)
        M_density = np.tile(m_profile, (problem.Nt + 1, 1))

        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)


class TestHJBSemiLagrangianSolverNotAbstract:
    """Test that HJBSemiLagrangianSolver is concrete (not abstract)."""

    def test_solver_not_abstract(self):
        """Test that HJBSemiLagrangianSolver can be instantiated."""
        import inspect

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())

        # Should not raise TypeError about abstract methods
        solver = HJBSemiLagrangianSolver(problem)
        assert isinstance(solver, HJBSemiLagrangianSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBSemiLagrangianSolver)


class TestCharacteristicTracingMethods:
    """Test different characteristic tracing methods (explicit_euler, rk2, rk4)."""

    def test_explicit_euler_initialization(self):
        """Test that explicit_euler method initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler")

        assert solver.characteristic_solver == "explicit_euler"

    def test_rk2_initialization(self):
        """Test that rk2 method initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2")

        assert solver.characteristic_solver == "rk2"

    def test_rk4_initialization(self):
        """Test that rk4 method initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4")

        assert solver.characteristic_solver == "rk4"

    def test_euler_produces_valid_solution(self):
        """Test that explicit_euler produces valid solution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk2_produces_valid_solution(self):
        """Test that rk2 produces valid solution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk4_produces_valid_solution(self):
        """Test that rk4 with scipy.solve_ivp produces valid solution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk2_consistency_with_euler(self):
        """Test that rk2 produces consistent results with euler on smooth problems."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.2, Nt=20, components=_default_components())

        # Solve with euler
        solver_euler = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_euler = solver_euler.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with rk2
        solver_rk2 = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2", use_jax=False)
        U_rk2 = solver_rk2.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with small dt, should be very similar
        rel_error = np.linalg.norm(U_rk2 - U_euler) / np.linalg.norm(U_euler)
        assert rel_error < 0.1  # Within 10%

    def test_rk4_consistency_with_euler(self):
        """Test that rk4 produces consistent results with euler on smooth problems."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.2, Nt=20, components=_default_components())

        # Solve with euler
        solver_euler = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_euler = solver_euler.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with rk4
        solver_rk4 = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)
        U_rk4 = solver_rk4.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with small dt, should be similar
        rel_error = np.linalg.norm(U_rk4 - U_euler) / np.linalg.norm(U_euler)
        assert rel_error < 0.1  # Within 10%

    def test_trace_characteristic_backward_1d(self):
        """Test _trace_characteristic_backward method directly in 1D."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)

        # Test characteristic tracing
        x_current = 0.5
        p_optimal = 0.1
        dt = 0.01

        x_departure = solver._trace_characteristic_backward(x_current, p_optimal, dt)

        # Should return a scalar
        assert isinstance(x_departure, (float, np.floating))
        # Should be finite
        assert np.isfinite(x_departure)
        # Should be within domain
        bounds = problem.geometry.get_bounds()
        assert bounds[0][0] <= x_departure <= bounds[1][0]


class TestInterpolationMethods:
    """Test different interpolation methods (linear, cubic, quintic)."""

    def test_linear_interpolation_initialization(self):
        """Test that linear interpolation initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear")

        assert solver.interpolation_method == "linear"

    def test_cubic_interpolation_initialization(self):
        """Test that cubic interpolation initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")

        assert solver.interpolation_method == "cubic"

    def test_cubic_produces_valid_solution_1d(self):
        """Test that cubic interpolation produces valid solution in 1D."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_cubic_consistency_with_linear(self):
        """Test that cubic interpolation is consistent with linear on smooth problems."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20, components=_default_components())

        # Solve with linear
        solver_linear = HJBSemiLagrangianSolver(
            problem, interpolation_method="linear", characteristic_solver="rk2", use_jax=False
        )
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_linear = solver_linear.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with cubic
        solver_cubic = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )
        U_cubic = solver_cubic.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with fine grid, should be reasonably similar
        # Note: With gradient-based optimal control (Issue #298 fix), interpolation
        # method has more impact since characteristics now move correctly
        rel_error = np.linalg.norm(U_cubic - U_linear) / np.linalg.norm(U_linear)
        assert rel_error < 0.25  # Within 25% (updated after gradient fix)

    @pytest.mark.xfail(reason="Cubic interpolation produces NaN values - see issue #583")
    def test_cubic_improves_smoothness(self):
        """Test that cubic interpolation produces smoother solutions."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        # Use steep gradients to test interpolation quality
        U_final = np.exp(-20 * (x_coords - 0.5) ** 2)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve with linear
        solver_linear = HJBSemiLagrangianSolver(
            problem, interpolation_method="linear", characteristic_solver="rk2", use_jax=False
        )
        U_linear = solver_linear.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with cubic
        solver_cubic = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )
        U_cubic = solver_cubic.solve_hjb_system(M_density, U_final, U_prev)

        # Measure smoothness via second derivative
        smoothness_linear = np.mean(np.abs(np.diff(U_linear, n=2, axis=1)))
        smoothness_cubic = np.mean(np.abs(np.diff(U_cubic, n=2, axis=1)))

        # Both should be finite
        assert np.isfinite(smoothness_linear)
        assert np.isfinite(smoothness_cubic)
        # Cubic should generally be smoother (smaller second derivatives)
        # This is not always true but should hold for most cases
        # We just check that cubic doesn't make things dramatically worse
        assert smoothness_cubic < smoothness_linear * 2.0


class TestRBFInterpolationFallback:
    """Test RBF interpolation fallback functionality."""

    def test_rbf_fallback_initialization_enabled(self):
        """Test that RBF fallback can be enabled."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline")

        assert solver.use_rbf_fallback is True
        assert solver.rbf_kernel == "thin_plate_spline"

    def test_rbf_fallback_initialization_disabled(self):
        """Test that RBF fallback can be disabled."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=False)

        assert solver.use_rbf_fallback is False

    def test_rbf_kernel_options(self):
        """Test different RBF kernel options."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())

        kernels = ["thin_plate_spline", "multiquadric", "gaussian"]

        for kernel in kernels:
            solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=True, rbf_kernel=kernel)
            assert solver.rbf_kernel == kernel

    @pytest.mark.xfail(reason="Numerical instability with RBF thin_plate_spline on steep gradients - see Issue #583")
    def test_rbf_fallback_produces_valid_solution(self):
        """Test that solver with RBF fallback produces valid solution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline", characteristic_solver="rk2", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        # Use steep gradient to potentially trigger RBF fallback
        U_final = np.exp(-20 * (x_coords - 0.5) ** 2)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rbf_consistency_with_no_fallback(self):
        """Test that RBF fallback doesn't change results on well-behaved problems."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve without RBF
        solver_no_rbf = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=False, characteristic_solver="rk2", use_jax=False
        )
        U_no_rbf = solver_no_rbf.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with RBF
        solver_rbf = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline", characteristic_solver="rk2", use_jax=False
        )
        U_rbf = solver_rbf.solve_hjb_system(M_density, U_final, U_prev)

        # On well-behaved problems, RBF fallback shouldn't trigger
        # Results should be identical or very close
        rel_error = np.linalg.norm(U_rbf - U_no_rbf) / np.linalg.norm(U_no_rbf)
        assert rel_error < 1e-10  # Should be machine precision


class TestEnhancementsIntegration:
    """Test combinations of enhancements working together."""

    def test_rk4_with_cubic_interpolation(self):
        """Test RK4 characteristic tracing with cubic interpolation."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem, characteristic_solver="rk4", interpolation_method="cubic", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk4_with_rbf_fallback(self):
        """Test RK4 characteristic tracing with RBF fallback."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem, characteristic_solver="rk4", use_rbf_fallback=True, rbf_kernel="thin_plate_spline", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_all_enhancements_together(self):
        """Test all enhancements working together: RK4 + cubic + RBF."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="rk4",
            interpolation_method="cubic",
            use_rbf_fallback=True,
            rbf_kernel="thin_plate_spline",
            use_jax=False,
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_enhanced_vs_baseline_consistency(self):
        """Test that enhanced configuration produces consistent results with baseline."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Baseline configuration
        solver_baseline = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="explicit_euler",
            interpolation_method="linear",
            use_rbf_fallback=False,
            use_jax=False,
        )
        U_baseline = solver_baseline.solve_hjb_system(M_density, U_final, U_prev)

        # Enhanced configuration
        solver_enhanced = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="rk4",
            interpolation_method="cubic",
            use_rbf_fallback=True,
            rbf_kernel="thin_plate_spline",
            use_jax=False,
        )
        U_enhanced = solver_enhanced.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with fine grid, should be reasonably consistent
        # Note: With gradient-based optimal control (Issue #298 fix), method differences
        # are more pronounced since characteristics now move correctly
        rel_error = np.linalg.norm(U_enhanced - U_baseline) / np.linalg.norm(U_baseline)
        assert rel_error < 0.20  # Within 20% (updated after gradient fix)


class TestStochasticCharacteristicSL:
    """Issue #1026: Carlini-Silva (2014) stochastic-characteristic SL.

    Tests the diffusion_method="stochastic" branch that incorporates the
    diffusion term into the SL update via 2*d Brownian departure points,
    instead of the operator-splitting (ADI/Crank-Nicolson) default.

    Validation experiment: mfg-research/experiments/crowd_evacuation_2d/
    minors/archive/exp14_towel_1d_benchmark/subs/exp14e_solver_comparison/
    """

    def test_linear_plus_stochastic_accepted(self):
        """Issue #1049: linear+stochastic IS the canonical Carlini-Silva 2014 scheme.

        Previously rejected by validation (`test_linear_plus_stochastic_rejected`).
        That validation was inverted from CS 2014's stability requirement: the
        rejected combination IS the proven-stable canonical scheme, while the
        forced cubic combination is non-monotone (Issue #1033). Test renamed and
        inverted to assert the corrected behavior.
        """
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[51],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        # Should NOT raise — linear+stochastic is now allowed and recommended.
        solver = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="linear",
            diffusion_method="stochastic",
        )
        assert solver.diffusion_method == "stochastic"
        assert solver.interpolation_method == "linear"

    def test_cubic_plus_stochastic_warns(self):
        """Issue #1049: cubic+stochastic emits a UserWarning (CS 2014 proof doesn't apply)."""
        import warnings as _w

        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[51],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            solver = HJBSemiLagrangianSolver(
                problem,
                interpolation_method="cubic",
                diffusion_method="stochastic",
                check_cfl=False,
            )
            cs_warnings = [m for m in caught if "Carlini-Silva" in str(m.message)]

        assert len(cs_warnings) == 1, f"expected 1 CS UserWarning, got {len(cs_warnings)}"
        assert solver.diffusion_method == "stochastic"
        assert solver.interpolation_method == "cubic"

    def test_apply_diffusion_raises_under_stochastic(self):
        """Reaching _apply_diffusion under stochastic dispatch is a programming error."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[51],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="cubic",
            diffusion_method="stochastic",
            check_cfl=False,
        )

        with pytest.raises(NotImplementedError, match="should not be called"):
            solver._apply_diffusion(np.zeros(51), 0.01)

    def test_constant_terminal_preserved(self):
        """H=0 with constant U_T must give constant U[0] (no spurious drift)."""
        from mfgarchon.core.hamiltonian import HamiltonianBase, OptimizationSense

        class ZeroH(HamiltonianBase):
            def __init__(self):
                super().__init__(sense=OptimizationSense.MINIMIZE)

            def __call__(self, x, m, p, t=0.0):
                p_arr = np.atleast_1d(np.asarray(p, dtype=float))
                if p_arr.ndim > 0:
                    return np.zeros(p_arr.shape[:-1])
                return 0.0

            def gradient_p(self, x, m, p, t=0.0):
                return np.zeros_like(np.asarray(p, dtype=float))

            def density_derivative(self, x, m, p, t=0.0):
                return 0.0

        geometry = TensorProductGrid(
            dimension=1,
            bounds=[(-1.0, 1.0)],
            Nx_points=[31],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        components = MFGComponents(
            hamiltonian=ZeroH(),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 1.0,
        )
        problem = MFGProblem(
            geometry=geometry,
            T=0.1,
            Nt=10,
            diffusion=0.045,
            components=components,
        )
        solver = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="cubic",
            diffusion_method="stochastic",
            check_cfl=False,
        )

        Nx = 31
        Nt = problem.Nt
        M_density = np.ones((Nt + 1, Nx))
        U_terminal = np.ones(Nx)

        U = solver.solve_hjb_system(
            M_density=M_density,
            U_terminal=U_terminal,
            U_coupling_prev=np.zeros((Nt + 1, Nx)),
        )

        np.testing.assert_allclose(U[0], 1.0, atol=1e-10)

    def test_consistency_with_default_adi(self):
        """Stochastic and default ADI must converge to the same numerical solution.

        Both schemes solve the same backward HJB; only the discretization
        path differs (Brownian quadrature vs. operator splitting). On a
        smooth Gaussian terminal with H=0, the difference should be
        within a few units of the local truncation error of either scheme.
        """
        from mfgarchon.core.hamiltonian import HamiltonianBase, OptimizationSense

        class ZeroH(HamiltonianBase):
            def __init__(self):
                super().__init__(sense=OptimizationSense.MINIMIZE)

            def __call__(self, x, m, p, t=0.0):
                p_arr = np.atleast_1d(np.asarray(p, dtype=float))
                if p_arr.ndim > 0:
                    return np.zeros(p_arr.shape[:-1])
                return 0.0

            def gradient_p(self, x, m, p, t=0.0):
                return np.zeros_like(np.asarray(p, dtype=float))

            def density_derivative(self, x, m, p, t=0.0):
                return 0.0

        sigma_test = 0.3
        T_test = 0.5
        beta_T = 1.0
        N, Nt = 100, 200

        geometry = TensorProductGrid(
            dimension=1,
            bounds=[(-5.0, 5.0)],
            Nx_points=[N + 1],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        x_grid = geometry.get_spatial_grid().flatten()
        components = MFGComponents(
            hamiltonian=ZeroH(),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: float(np.exp(-(x[0] ** 2) / (2 * beta_T)) / np.sqrt(2 * np.pi * beta_T)),
        )
        problem = MFGProblem(
            geometry=geometry,
            T=T_test,
            Nt=Nt,
            diffusion=sigma_test**2 / 2,
            components=components,
        )

        U_terminal = np.exp(-(x_grid**2) / (2 * beta_T)) / np.sqrt(2 * np.pi * beta_T)
        M_density = np.ones((Nt + 1, N + 1))

        solver_st = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="cubic",
            diffusion_method="stochastic",
            check_cfl=False,
        )
        solver_adi = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="cubic",
            diffusion_method="adi",
            check_cfl=False,
        )

        U_st = solver_st.solve_hjb_system(
            M_density=M_density,
            U_terminal=U_terminal,
            U_coupling_prev=np.zeros((Nt + 1, N + 1)),
        )
        U_adi = solver_adi.solve_hjb_system(
            M_density=M_density,
            U_terminal=U_terminal,
            U_coupling_prev=np.zeros((Nt + 1, N + 1)),
        )

        max_diff = np.max(np.abs(U_st[0] - U_adi[0]))
        # Both schemes are 2nd-order accurate on smooth Gaussians; their
        # difference should be a few units of the local truncation error.
        assert max_diff < 5e-3, f"Stochastic and ADI diverge on smooth Gaussian: max diff = {max_diff:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
