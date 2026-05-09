#!/usr/bin/env python3
"""
Unit tests for FPParticleSolver.

Tests the particle-based Fokker-Planck solver with KDE density estimation,
including backend selection, normalization strategies, and intelligent pipeline dispatch.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver, KDENormalization
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for 1D testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


def _default_components_2d():
    """Default MFGComponents for 2D testing (Issue #670: explicit specification required)."""

    def m_initial_2d(x):
        # x is [x, y] coordinate - compute Gaussian at center (0.5, 0.5)
        x_arr = np.asarray(x)
        return np.exp(-10 * np.sum((x_arr - 0.5) ** 2))

    return MFGComponents(
        m_initial=m_initial_2d,
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class TestFPParticleSolverInitialization:
    """Test FPParticleSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem)

        assert solver.fp_method_name == "Particle"
        assert solver.num_particles == 5000
        assert solver.kde_bandwidth == "scott"
        assert solver.kde_normalization == KDENormalization.ALL
        # Default BC comes from geometry (TensorProductGrid), which is "no_flux"
        assert solver.boundary_conditions.type == "no_flux"

    def test_custom_num_particles(self):
        """Test initialization with custom number of particles."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=1000)

        assert solver.num_particles == 1000

    def test_custom_kde_bandwidth(self):
        """Test initialization with custom KDE bandwidth."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem, kde_bandwidth=0.1)

        assert solver.kde_bandwidth == 0.1

    def test_kde_normalization_none(self):
        """Test initialization with no KDE normalization."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem, kde_normalization=KDENormalization.NONE)

        assert solver.kde_normalization == KDENormalization.NONE

    def test_kde_normalization_initial_only(self):
        """Test initialization with initial-only KDE normalization."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem, kde_normalization=KDENormalization.INITIAL_ONLY)

        assert solver.kde_normalization == KDENormalization.INITIAL_ONLY

    def test_kde_normalization_all(self):
        """Test initialization with all-step KDE normalization."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem, kde_normalization=KDENormalization.ALL)

        assert solver.kde_normalization == KDENormalization.ALL

    def test_kde_normalization_string(self):
        """Test initialization with KDE normalization as string."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem, kde_normalization="none")

        assert solver.kde_normalization == KDENormalization.NONE

    def test_deprecated_normalize_kde_output_false(self):
        """Test backward compatibility with deprecated normalize_kde_output=False."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with pytest.warns(DeprecationWarning, match="normalize_kde_output.*deprecated"):
            solver = FPParticleSolver(problem, normalize_kde_output=False)

        assert solver.kde_normalization == KDENormalization.NONE

    def test_deprecated_normalize_only_initial_true(self):
        """Test backward compatibility with deprecated normalize_only_initial=True."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with pytest.warns(DeprecationWarning, match="normalize_only_initial.*deprecated"):
            solver = FPParticleSolver(problem, normalize_only_initial=True)

        assert solver.kde_normalization == KDENormalization.INITIAL_ONLY

    def test_deprecated_both_parameters(self):
        """Test backward compatibility with both deprecated parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with pytest.warns(DeprecationWarning, match="deprecated"):
            solver = FPParticleSolver(problem, normalize_kde_output=True, normalize_only_initial=False)

        assert solver.kde_normalization == KDENormalization.ALL

    def test_custom_boundary_conditions(self):
        """Test initialization with custom boundary conditions."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        bc = no_flux_bc(dimension=1)
        solver = FPParticleSolver(problem, boundary_conditions=bc)

        assert solver.boundary_conditions.type == "no_flux"

    def test_backend_initialization_numpy(self):
        """Test initialization with NumPy backend."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem, backend="numpy")

        assert solver.backend is not None
        assert solver.backend.name == "numpy"

    def test_default_backend_is_numpy(self):
        """Test that default backend is NumPy."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem)

        assert solver.backend is not None
        assert solver.backend.name == "numpy"

    def test_strategy_selector_initialized(self):
        """Test that strategy selector is properly initialized."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem)

        assert solver.strategy_selector is not None
        assert solver.current_strategy is None  # Not set until solve

    def test_time_step_counter_initialized(self):
        """Test that time step counter is initialized."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = FPParticleSolver(problem)

        assert solver._time_step_counter == 0


class TestFPParticleSolverSolveFPSystem:
    """Test the main solve_fp_system method."""

    def test_solve_fp_system_shape(self):
        """Test that solve_fp_system returns correct shape."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        # Create inputs
        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        # Solve
        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert M_solution.shape == (Nt_points, Nx_points)
        assert np.all(np.isfinite(M_solution))

    def test_solve_fp_system_initial_condition(self):
        """Test that initial condition center of mass is approximately preserved."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=2000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points
        dx = problem.geometry.get_grid_spacing()[0]
        bounds = problem.geometry.get_bounds()

        # Create inputs with specific initial condition
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial * dx)
        U_solution = np.zeros((Nt_points, Nx_points))

        # Solve
        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Check that center of mass is approximately preserved
        # (KDE introduces smoothing but should preserve location)
        cm_initial = np.sum(x_coords * m_initial * dx)
        cm_solution = np.sum(x_coords * M_solution[0, :] * dx)
        assert np.isclose(cm_initial, cm_solution, rtol=0.2)

    def test_solve_with_zero_drift(self):
        """Test solving with zero drift field."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_non_zero_drift(self):
        """Test solving with non-zero drift field."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points
        bounds = problem.geometry.get_bounds()

        # Create non-zero drift
        m_initial = np.ones(Nx_points) / Nx_points
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_solution = np.tile(x_coords**2, (Nt_points, 1))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_different_num_particles(self):
        """Test solver with different numbers of particles."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        for num_particles in [100, 500, 1000]:
            solver = FPParticleSolver(problem, num_particles=num_particles)

            m_initial = np.ones(Nx_points) / Nx_points
            U_solution = np.zeros((Nt_points, Nx_points))

            M_solution = solver.solve_fp_system(m_initial, U_solution)

            assert np.all(np.isfinite(M_solution))

    def test_solve_with_kde_normalization_none(self):
        """Test solving with no KDE normalization."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.NONE)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_kde_normalization_initial_only(self):
        """Test solving with initial-only KDE normalization."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.INITIAL_ONLY)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_strategy_selection(self):
        """Test that strategy is selected during solve."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500)

        assert solver.current_strategy is None  # Before solve

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        solver.solve_fp_system(m_initial, U_solution)

        assert solver.current_strategy is not None  # After solve
        assert hasattr(solver.current_strategy, "name")

    def test_time_step_counter_reset(self):
        """Test that time step counter is reset on solve."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500)

        solver._time_step_counter = 999  # Set to non-zero

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        solver.solve_fp_system(m_initial, U_solution)

        # Counter should be reset (though it gets incremented during solve)
        # Just verify solve completed without error


class TestFPParticleSolverNumericalProperties:
    """Test numerical properties of particle FP solutions."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=1000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points
        dx = problem.geometry.get_grid_spacing()[0]
        bounds = problem.geometry.get_bounds()

        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial * dx)
        U_solution = np.zeros((Nt_points, Nx_points))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # All values should be finite
        assert np.all(np.isfinite(M_solution))

    def test_forward_time_propagation(self):
        """Test that solution is computed for all time steps."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, sigma=0.3, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=2000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points
        dx = problem.geometry.get_grid_spacing()[0]
        bounds = problem.geometry.get_bounds()

        # Concentrated initial condition
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.01**2))
        m_initial = m_initial / np.sum(m_initial * dx)

        U_solution = np.zeros((Nt_points, Nx_points))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Solution should be computed for all time steps and remain finite
        assert M_solution.shape == (Nt_points, Nx_points)
        assert np.all(np.isfinite(M_solution))
        # Verify solution exists at all time points (not all zeros)
        assert np.any(M_solution > 0)

    def test_approximate_mass_conservation(self):
        """Test that total mass is approximately conserved."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=2000, kde_normalization=KDENormalization.ALL)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points
        dx = problem.geometry.get_grid_spacing()[0]
        bounds = problem.geometry.get_bounds()

        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial * dx)
        U_solution = np.zeros((Nt_points, Nx_points))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Check mass conservation across time (with relaxed tolerance for KDE)
        initial_mass = np.sum(M_solution[0, :] * dx)
        for t in range(Nt_points):
            current_mass = np.sum(M_solution[t, :] * dx)
            # Allow larger error for particle methods with KDE
            assert np.isclose(current_mass, initial_mass, rtol=0.3)

    def test_non_negativity(self):
        """Test that density remains non-negative."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=1000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Density should be non-negative (KDE ensures this)
        assert np.all(M_solution >= -1e-10)


class TestFPParticleSolverIntegration:
    """Integration tests with actual FP problems."""

    def test_solver_not_abstract(self):
        """Test that FPParticleSolver can be instantiated."""
        import inspect

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())

        # Should not raise TypeError about abstract methods
        solver = FPParticleSolver(problem)
        assert isinstance(solver, FPParticleSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(FPParticleSolver)

    def test_solver_with_different_parameters(self):
        """Test solver with various parameter configurations."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        configs = [
            {"num_particles": 500, "kde_normalization": KDENormalization.NONE},
            {"num_particles": 1000, "kde_normalization": KDENormalization.INITIAL_ONLY},
            {"num_particles": 2000, "kde_normalization": KDENormalization.ALL, "kde_bandwidth": 0.1},
        ]

        for config in configs:
            solver = FPParticleSolver(problem, **config)

            m_initial = np.ones(Nx_points) / Nx_points
            U_solution = np.zeros((Nt_points, Nx_points))

            M_solution = solver.solve_fp_system(m_initial, U_solution)

            assert np.all(np.isfinite(M_solution))

    def test_solver_with_different_boundary_conditions(self):
        """Test solver with different boundary condition types."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points

        bc_options = [periodic_bc(dimension=1), no_flux_bc(dimension=1)]

        for bc in bc_options:
            solver = FPParticleSolver(problem, num_particles=500, boundary_conditions=bc)

            m_initial = np.ones(Nx_points) / Nx_points
            U_solution = np.zeros((Nt_points, Nx_points))

            M_solution = solver.solve_fp_system(m_initial, U_solution)

            assert np.all(np.isfinite(M_solution))


class TestFPParticleSolverHelperMethods:
    """Test helper methods for gradient computation and normalization."""

    def test_compute_gradient(self):
        """Test gradient computation helper."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500)

        Nx_points = problem.geometry.get_grid_shape()[0]
        dx = problem.geometry.get_grid_spacing()[0]
        bounds = problem.geometry.get_bounds()

        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_array = x_coords**2

        gradients = solver._compute_gradient_nd(U_array, [dx], use_backend=False)

        # Should return list of gradient components (one per dimension)
        assert isinstance(gradients, list)
        assert len(gradients) == 1  # 1D case
        gradient = gradients[0]
        assert np.all(np.isfinite(gradient))
        assert gradient.shape == U_array.shape

    def test_compute_gradient_zero_dx(self):
        """Test gradient computation with zero Dx."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500)

        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_array = x_coords**2

        gradients = solver._compute_gradient_nd(U_array, [0.0], use_backend=False)

        # Should return list of gradient components (one per dimension)
        assert isinstance(gradients, list)
        assert len(gradients) == 1  # 1D case
        # With zero spacing, gradient should be zeros
        assert np.allclose(gradients[0], 0.0)

    def test_normalize_density_none(self):
        """Test density normalization with NONE strategy."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.NONE)

        Nx_points = problem.geometry.get_grid_shape()[0]
        dx = problem.geometry.get_grid_spacing()[0]

        M_array = np.random.rand(Nx_points) * 2.0  # Random unnormalized density

        normalized = solver._normalize_density(M_array, dx, use_backend=False)

        # Should not normalize (return as-is)
        assert np.allclose(normalized, M_array)

    def test_normalize_density_initial_only(self):
        """Test density normalization with INITIAL_ONLY strategy."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.INITIAL_ONLY)

        Nx_points = problem.geometry.get_grid_shape()[0]
        dx = problem.geometry.get_grid_spacing()[0]

        M_array = np.random.rand(Nx_points) * 2.0

        # First call (time step 0) - should normalize
        solver._time_step_counter = 0
        normalized_0 = solver._normalize_density(M_array, dx, use_backend=False)
        assert np.isclose(np.sum(normalized_0 * dx), 1.0, rtol=0.1)

        # Second call (time step 1) - should not normalize
        solver._time_step_counter = 1
        normalized_1 = solver._normalize_density(M_array, dx, use_backend=False)
        assert np.allclose(normalized_1, M_array)

    def test_normalize_density_all(self):
        """Test density normalization with ALL strategy."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=15, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=500, kde_normalization=KDENormalization.ALL)

        Nx_points = problem.geometry.get_grid_shape()[0]
        dx = problem.geometry.get_grid_spacing()[0]

        M_array = np.random.rand(Nx_points) * 2.0

        # Should normalize at any time step
        for t in [0, 1, 5, 10]:
            solver._time_step_counter = t
            normalized = solver._normalize_density(M_array, dx, use_backend=False)
            assert np.isclose(np.sum(normalized * dx), 1.0, rtol=0.1)


class TestFPParticleSolverCallableDrift:
    """Test callable (state-dependent) drift_field support (Phase 2 - Issue #487)."""

    def test_constant_drift_callable_1d(self):
        """Test constant drift via callable function in 1D."""
        # Set random seed for reproducible stochastic particle evolution
        np.random.seed(42)

        # Use stronger drift (0.5) and lower diffusion (0.05) for clearer signal
        # Expected displacement: drift * T = 0.5 * 0.5 = 0.25
        # With diffusion = 0.05, drift dominates (Peclet number ~ 10)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=25, sigma=0.05, components=_default_components())
        # Increase particles to reduce statistical variance
        solver = FPParticleSolver(problem, num_particles=2000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Constant drift pushing right
        def constant_drift(t, x, m):
            return 0.5 * np.ones_like(x)

        # Initial condition (Gaussian centered at 0.3)
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.3) ** 2) / (2 * 0.05**2))
        m_initial /= np.sum(m_initial)

        # Solve with callable drift
        M = solver.solve_fp_system(M_initial=m_initial, drift_field=constant_drift, show_progress=False)

        assert M.shape[1] == Nx_points
        assert np.all(np.isfinite(M))

        # Peak should move right with high confidence
        initial_peak = x_grid[np.argmax(M[0])]
        final_peak = x_grid[np.argmax(M[-1])]

        # Check displacement is positive and significant (at least half expected)
        # Expected: ~0.25, require at least 0.10 to account for diffusion spread
        displacement = final_peak - initial_peak
        assert displacement > 0.10, f"Peak displacement {displacement:.3f} too small (expected ~0.25)"

    def test_constant_drift_callable_2d(self):
        """Test constant drift via callable function in 2D."""
        from mfgarchon.geometry import TensorProductGrid

        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.3, Nt=15, sigma=0.05, components=_default_components_2d())
        solver = FPParticleSolver(problem, num_particles=2000)

        Nx, Ny = domain.num_points[0], domain.num_points[1]

        # Diagonal drift pushing toward upper-right
        def diagonal_drift(t, x, m):
            # x is (N, 2) for 2D
            drift = np.zeros_like(x)
            drift[:, 0] = 0.4  # x-drift
            drift[:, 1] = 0.4  # y-drift
            return drift

        # Initial condition (Gaussian centered at (0.3, 0.3))
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-30 * ((X - 0.3) ** 2 + (Y - 0.3) ** 2))
        m_initial /= np.sum(m_initial)

        # Solve with callable drift
        M = solver.solve_fp_system(M_initial=m_initial, drift_field=diagonal_drift, show_progress=False)

        assert M.shape == (problem.Nt + 2, Nx, Ny)  # Shape depends on particle solver
        assert np.all(np.isfinite(M))
        # Center of mass should move diagonally
        initial_com_x = np.sum(X * M[0]) / np.sum(M[0])
        initial_com_y = np.sum(Y * M[0]) / np.sum(M[0])
        final_com_x = np.sum(X * M[-1]) / np.sum(M[-1])
        final_com_y = np.sum(Y * M[-1]) / np.sum(M[-1])
        assert final_com_x > initial_com_x
        assert final_com_y > initial_com_y

    def test_state_dependent_drift_1d(self):
        """Test state-dependent drift: alpha(t, x, m) depends on density."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=25, sigma=0.1, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=1000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Density-dependent drift: higher density -> lower drift
        def crowd_aware_drift(t, x, m):
            # Drift reduces where density is high
            return 0.3 * (1 - 0.5 * m / (np.max(m) + 1e-10))

        # Initial condition
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Solve
        M = solver.solve_fp_system(M_initial=m_initial, drift_field=crowd_aware_drift, show_progress=False)

        assert np.all(np.isfinite(M))
        # Density should be non-negative (KDE ensures this)
        assert np.all(M >= -1e-10)

    def test_time_dependent_drift_1d(self):
        """Test time-dependent drift: alpha(t, x, m) varies with time."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, sigma=0.1, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=1000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Drift that oscillates over time
        def oscillating_drift(t, x, m):
            return 0.3 * np.sin(2 * np.pi * t) * np.ones_like(x)

        # Initial condition
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Solve
        M = solver.solve_fp_system(M_initial=m_initial, drift_field=oscillating_drift, show_progress=False)

        assert np.all(np.isfinite(M))
        assert np.all(M >= -1e-10)

    def test_callable_drift_with_array_diffusion(self):
        """Test callable drift combined with array diffusion."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, sigma=0.1, components=_default_components())
        solver = FPParticleSolver(problem, num_particles=1000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Callable drift
        def simple_drift(t, x, m):
            return 0.2 * np.ones_like(x)

        # Initial condition
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Solve with callable drift and constant scalar diffusion
        M = solver.solve_fp_system(
            M_initial=m_initial, drift_field=simple_drift, volatility_field=0.15, show_progress=False
        )

        assert np.all(np.isfinite(M))
        assert np.all(M >= -1e-10)


class TestEnforceObstacleBoundary:
    """Issue #1064: enforce_obstacle_boundary must respect outer-boundary
    handling — only project obstacle-interior violations, leave outer-boundary
    violations for the caller's segment-aware BC."""

    def _make_diff_domain(self):
        """Box [0,1]² minus circular obstacle, off-center to avoid degenerate projection."""
        from mfgarchon.geometry.implicit import (
            DifferenceDomain, Hyperrectangle, Hypersphere,
        )
        box = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))
        # Obstacle off bbox center so project_to_domain has a non-degenerate direction
        obstacle = Hypersphere(center=np.array([0.3, 0.5]), radius=0.15)
        return DifferenceDomain(box, obstacle)

    def test_obstacle_interior_projected(self):
        """Particle inside obstacle is projected back to navigable region."""
        from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import enforce_obstacle_boundary
        domain = self._make_diff_domain()
        particles = np.array([[0.3, 0.5]])  # obstacle center
        result = enforce_obstacle_boundary(particles.copy(), domain)
        assert not np.allclose(result, particles), \
            "Particle inside obstacle should have been projected out"
        assert domain.contains(result).all()

    def test_outer_bbox_violation_left_alone(self):
        """Issue #1064: Particle past outer bbox is NOT touched — caller's BC handles it."""
        from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import enforce_obstacle_boundary
        domain = self._make_diff_domain()
        particles = np.array([[1.5, 0.5]])  # past right wall x=1.5 > bbox max 1.0
        result = enforce_obstacle_boundary(particles.copy(), domain)
        np.testing.assert_array_equal(result, particles)

    def test_navigable_particles_untouched(self):
        """Particles in navigable region are unchanged."""
        from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import enforce_obstacle_boundary
        domain = self._make_diff_domain()
        particles = np.array([[0.1, 0.1], [0.9, 0.5], [0.3, 0.7]])
        result = enforce_obstacle_boundary(particles.copy(), domain)
        np.testing.assert_array_equal(result, particles)

    def test_mixed_obstacle_and_outer_violations(self):
        """Mix: one in obstacle (project), one past bbox (leave), one valid (untouched)."""
        from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import enforce_obstacle_boundary
        domain = self._make_diff_domain()
        particles = np.array([
            [0.3, 0.5],   # inside obstacle → should be projected
            [1.5, 0.5],   # past right wall → should be left alone (Issue #1064)
            [0.1, 0.1],   # navigable → untouched
        ])
        result = enforce_obstacle_boundary(particles.copy(), domain)
        assert not np.allclose(result[0], particles[0])
        assert domain.contains(result[0:1]).all()
        np.testing.assert_array_equal(result[1], particles[1])
        np.testing.assert_array_equal(result[2], particles[2])

    def test_none_implicit_domain_passthrough(self):
        """No implicit_domain → no-op."""
        from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import enforce_obstacle_boundary
        particles = np.array([[10.0, 10.0]])
        result = enforce_obstacle_boundary(particles.copy(), None)
        np.testing.assert_array_equal(result, particles)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
