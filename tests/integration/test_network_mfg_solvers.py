#!/usr/bin/env python3
"""
Integration tests for network MFG solvers.

Tests complete MFG problem solving on network/graph structures,
including various network geometries, solver schemes, and coupling methods.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.network_mfg_solver import (
    create_network_mfg_solver,
    create_simple_network_solver,
)
from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import GridNetwork

# Skip all tests if igraph is not available (network backend dependency)
igraph = pytest.importorskip("igraph")


class TestNetworkMFGSolverCreation:
    """Test network MFG solver factory functions."""

    def test_create_network_solver_basic(self):
        """Test basic network MFG solver creation."""
        # Create simple grid network
        network = GridNetwork(width=5, height=5)
        network.create_network()

        # Create network MFG problem
        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        # Create solver
        solver = create_network_mfg_solver(problem)

        assert solver is not None
        assert hasattr(solver, "solve")
        assert solver.problem is problem

    def test_create_network_solver_explicit(self):
        """Test network solver with explicit schemes."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=20,
        )

        solver = create_network_mfg_solver(
            problem,
            hjb_solver_type="explicit",
            fp_solver_type="explicit",
        )

        assert solver is not None

    def test_create_network_solver_implicit(self):
        """Test network solver with implicit schemes."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=20,
        )

        solver = create_network_mfg_solver(
            problem,
            hjb_solver_type="implicit",
            fp_solver_type="implicit",
        )

        assert solver is not None

    def test_create_simple_network_solver(self):
        """Test simplified network solver creation."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = create_simple_network_solver(problem)

        assert solver is not None
        assert hasattr(solver, "solve")

    def test_create_solver_with_custom_damping(self):
        """Test solver creation with custom damping factor."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = create_network_mfg_solver(
            problem,
            damping_factor=0.7,
        )

        assert solver.relaxation == 0.7


class TestNetworkMFGProblemSetup:
    """Test network MFG problem configuration."""

    def test_grid_network_problem(self):
        """Test MFG problem on grid network."""
        network = GridNetwork(width=5, height=5)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=20,
        )

        assert problem.is_network_problem is True
        assert problem.num_nodes == 25
        assert problem.T == 1.0
        assert problem.Nt == 20

    def test_small_grid_network_problem(self):
        """Test MFG problem on small grid."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        assert problem.num_nodes == 9
        assert problem.network_geometry is network

    def test_network_problem_with_components(self):
        """Test network problem with custom components."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        components = NetworkMFGComponents(
            diffusion_coefficient=0.5,
            drift_coefficient=1.0,
        )

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
            components=components,
        )

        assert problem.components.diffusion_coefficient == 0.5
        assert problem.components.drift_coefficient == 1.0


@pytest.mark.skip(
    reason="NetworkGraph geometry is incompatible with GFDM/FDM solvers "
    "(requires CartesianGrid). These tests will be enabled when network-specific "
    "solvers are implemented. See Issue #833."
)
class TestNetworkMFGSolverExecution:
    """Test network MFG solver execution.

    NOTE: All tests in this class are skipped because the current FDM/GFDM
    solvers require CartesianGrid, not NetworkGraph. This is a design limitation,
    not a bug. Network MFG solving needs dedicated graph-based solvers.
    """

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_solve_small_grid_network(self):
        """Test solving MFG on small grid network."""
        # Create 3x3 grid
        network = GridNetwork(width=3, height=3)
        network.create_network()

        # Create problem
        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        # Create solver
        solver = create_simple_network_solver(problem, scheme="implicit")

        # Solve
        result = solver.solve(max_iterations=10, tolerance=1e-4)

        # Verify result structure
        assert result is not None
        U, M = result[:2]

        # Check shapes (Nt+1 time steps, num_nodes spatial points)
        expected_shape = (problem.Nt + 1, problem.num_nodes)
        assert U.shape == expected_shape
        assert M.shape == expected_shape

        # Check values are finite
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_solve_with_explicit_scheme(self):
        """Test solving with explicit time-stepping."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.2,
            Nt=20,
        )

        solver = create_simple_network_solver(
            problem,
            scheme="RK45",
        )

        result = solver.solve(max_iterations=5, tolerance=1e-3)

        assert result is not None
        U, M = result[:2]
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_solve_with_implicit_scheme(self):
        """Test solving with implicit time-stepping."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=15,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=8, tolerance=1e-4)

        assert result is not None
        U, M = result[:2]
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))


class TestNetworkSolutionProperties:
    """Test mathematical properties of network MFG solutions."""

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_mass_conservation(self):
        """Test that total mass is approximately conserved."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=15,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=10, tolerance=1e-4)
        _U, M = result[:2]

        # Check mass conservation across time
        initial_mass = np.sum(M[0, :])
        for t in range(problem.Nt + 1):
            current_mass = np.sum(M[t, :])
            # Allow some numerical error
            assert np.isclose(current_mass, initial_mass, rtol=0.2)

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_density_non_negativity(self):
        """Test that density remains non-negative."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=10, tolerance=1e-4)
        _U, M = result[:2]

        # Density should be non-negative (with small tolerance for numerical errors)
        assert np.all(M >= -1e-10)

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_solution_evolution(self):
        """Test that solution evolves over time."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=20,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=10, tolerance=1e-4)
        U, _M = result[:2]

        # Value function should evolve backward in time
        # (Note: density M may remain constant for symmetric problems with uniform initial conditions)
        assert not np.allclose(U[0, :], U[-1, :])


class TestNetworkGeometryVariations:
    """Test different network geometries."""

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_periodic_grid_network(self):
        """Test MFG on periodic grid network."""
        network = GridNetwork(width=4, height=4, periodic=True)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=8, tolerance=1e-4)

        assert result is not None
        U, M = result[:2]
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_rectangular_grid_network(self):
        """Test MFG on non-square grid."""
        network = GridNetwork(width=6, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=8, tolerance=1e-4)

        assert result is not None
        assert problem.num_nodes == 18


class TestSolverConvergence:
    """Test solver convergence behavior."""

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_convergence_with_tight_tolerance(self):
        """Test convergence with tight tolerance."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.3,
            Nt=10,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=20, tolerance=1e-6)

        # Should converge or reach max iterations
        assert result is not None

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_convergence_with_relaxed_tolerance(self):
        """Test convergence with relaxed tolerance."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=5, tolerance=1e-3)

        assert result is not None


class TestSolverRobustness:
    """Test solver robustness to various configurations."""

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_different_damping_factors(self):
        """Test solver with various damping factors."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        for damping in [0.3, 0.5, 0.7]:
            solver = create_simple_network_solver(problem, damping=damping)

            result = solver.solve(max_iterations=10, tolerance=1e-4)

            assert result is not None

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_different_time_horizons(self):
        """Test solver with different time horizons."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        for T_val in [0.3, 0.5, 1.0]:
            problem = NetworkMFGProblem(
                network_geometry=network,
                T=T_val,
                Nt=10,
            )

            solver = create_simple_network_solver(problem, scheme="implicit")

            result = solver.solve(max_iterations=10, tolerance=1e-4)

            assert result is not None

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_different_network_sizes(self):
        """Test solver with different network sizes."""
        for size in [3, 4, 5]:
            network = GridNetwork(width=size, height=size)
            network.create_network()

            problem = NetworkMFGProblem(
                network_geometry=network,
                T=0.5,
                Nt=10,
            )

            solver = create_simple_network_solver(problem, scheme="implicit")

            result = solver.solve(max_iterations=8, tolerance=1e-4)

            assert result is not None
            assert problem.num_nodes == size * size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
