#!/usr/bin/env python3
"""
Unit tests for runtime safety validation (Issue #688).

Tests that:
- check_finite detects NaN/Inf with location info
- check_bounds detects out-of-range values
- validate_solver_output catches NaN in U/M and negative density
- FixedPointIterator terminates early on NaN (integration)

Follows the pattern of test_array_field_validation.py.
"""

import pytest

import numpy as np

from mfgarchon.utils.validation.runtime import (
    check_bounds,
    check_finite,
    validate_solver_output,
)

# ===========================================================================
# check_finite
# ===========================================================================


@pytest.mark.unit
def test_check_finite_clean_array():
    """Clean array should pass finiteness check."""
    arr = np.linspace(0.0, 1.0, 20)
    result = check_finite(arr, "test", raise_on_error=False)
    assert result.is_valid


@pytest.mark.unit
def test_check_finite_nan_detected():
    """Array with NaN should fail with location info."""
    arr = np.ones((10, 5))
    arr[3, 2] = np.nan
    result = check_finite(arr, "U", location="timestep 3", raise_on_error=False)
    assert not result.is_valid
    assert result.context.get("n_nan") == 1
    assert any("NaN" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_check_finite_raise_on_error():
    """check_finite with raise_on_error=True should raise ValueError."""
    arr = np.ones(10)
    arr[5] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        check_finite(arr, "M", raise_on_error=True)


# ===========================================================================
# check_bounds
# ===========================================================================


@pytest.mark.unit
def test_check_bounds_within():
    """Array within bounds should pass."""
    arr = np.linspace(0.0, 1.0, 20)
    result = check_bounds(arr, "density", lower=0.0, upper=1.0)
    assert result.is_valid


@pytest.mark.unit
def test_check_bounds_violation():
    """Array with values outside bounds should fail."""
    arr = np.array([0.5, 1.5, -0.1, 0.8])
    result = check_bounds(arr, "density", lower=0.0, upper=1.0)
    assert not result.is_valid
    assert any("above" in str(issue).lower() for issue in result.issues)
    assert any("below" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# validate_solver_output
# ===========================================================================


@pytest.mark.unit
def test_validate_solver_output_valid():
    """Clean U and M should pass output validation."""
    Nt, Nx = 10, 20
    U = np.random.randn(Nt, Nx)
    M = np.abs(np.random.randn(Nt, Nx))  # Non-negative
    result = validate_solver_output(U, M)
    assert result.is_valid


@pytest.mark.unit
def test_validate_solver_output_nan_u():
    """NaN in U should be detected."""
    Nt, Nx = 10, 20
    U = np.ones((Nt, Nx))
    U[5, 10] = np.nan
    M = np.ones((Nt, Nx))
    result = validate_solver_output(U, M)
    assert not result.is_valid
    assert any("NaN" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_validate_solver_output_negative_density():
    """Negative density should be detected."""
    Nt, Nx = 10, 20
    U = np.ones((Nt, Nx))
    M = np.ones((Nt, Nx))
    M[3, 5] = -0.5
    result = validate_solver_output(U, M, check_finite=False, check_density_positive=True)
    assert not result.is_valid
    assert any("negative" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Integration: FixedPointIterator NaN early termination
# ===========================================================================


@pytest.mark.unit
def test_fixed_point_nan_early_termination():
    """FixedPointIterator should terminate early when NaN appears in iteration."""
    from unittest.mock import Mock

    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import (
        FixedPointIterator,
    )
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary.conditions import no_flux_bc

    # Real problem setup (Nx=11 for speed)
    Nx = 11
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[Nx],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )
    components = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: x**2,
    )
    problem = MFGProblem(geometry=geom, components=components, Nt=10)

    Nt = problem.Nt
    num_time_steps = Nt + 1
    spatial_shape = problem.spatial_shape

    # Mock HJB solver: returns NaN on second call
    hjb_solver = Mock()
    call_count = [0]

    def hjb_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] >= 2:
            return np.full((num_time_steps, *spatial_shape), np.nan)
        return np.zeros((num_time_steps, *spatial_shape))

    hjb_solver.solve_hjb_system.side_effect = hjb_side_effect

    # Mock FP solver: returns valid density
    fp_solver = Mock()
    fp_solver.solve_fp_system.return_value = np.ones((num_time_steps, *spatial_shape)) / Nx

    # Create iterator and solve
    iterator = FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        relaxation=0.5,
    )
    result = iterator.solve(max_iterations=10, tolerance=1e-6)

    # Verify early termination
    assert not result.converged
    assert result.iterations < 10
    assert result.metadata.get("convergence_reason") == "diverged_nan"

    # Verify output validation detected the NaN
    output_val = result.metadata.get("output_validation")
    assert output_val is not None
    assert output_val["is_valid"] is False
