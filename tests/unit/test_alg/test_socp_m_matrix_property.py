"""Issue #1074: per-stencil M-matrix verification for joint_socp.

The paper's `thm:discrete_comparison` claim hinges on each SOCP-feasible
stencil having M-matrix property:

1. Sum-zero on Laplacian:        ``sum(L) == 0``
2. Off-diagonal non-negative:    ``L[off] >= 0``  (so center is non-positive)
3. Per-edge cone bound:          ``||D[:, j]|| <= (C/h_i) * L[j]``  for each j

These are enforced by `solve_joint_socp_at_stencil` constraints (see
`gfdm_components/joint_socp.py:174`). This test verifies they hold empirically
for representative configurations. If the test fails, the paper claim has an
empirical counter-example.

The full assembled-matrix M-matrix property (Issue #1074 long-term ask)
requires also bounding the advective contribution `(1/lambda)*D_grad*dt`
against the diffusive contribution `(sigma**2/2)*L*dt`, which depends on
the time-step. That's deferred — this test covers the part joint_socp
itself guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# joint_socp requires cvxpy
try:
    import cvxpy  # noqa: F401

    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


pytestmark = pytest.mark.skipif(
    not _HAS_CVXPY, reason="cvxpy not installed; joint_socp tests skipped"
)


def _make_solver(sigma: float, n_x: int = 21):
    """Construct an HJBGFDMSolver with joint_socp + precompute on a 1D grid."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem

    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[n_x],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: np.ones_like(np.asarray(m)),
        ),
    )
    problem = MFGProblem(
        geometry=geometry, T=0.5, Nt=10, sigma=sigma, components=components
    )
    bounds = problem.geometry.get_bounds()
    x_coords = np.linspace(bounds[0][0], bounds[1][0], n_x)
    collocation_points = x_coords.reshape(-1, 1)
    return HJBGFDMSolver(
        problem,
        collocation_points,
        delta=0.15,
        monotonicity_scheme="joint_socp",
        monotonicity_application="precompute",
    )


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_socp_stencil_laplacian_consistency(sigma):
    """Issue #1074: sum(L) == 0 for every SOCP-feasible stencil (Laplacian
    consistency / sum-zero condition)."""
    solver = _make_solver(sigma)
    stencils = solver._joint_socp_stencils
    assert stencils is not None, "joint_socp stencils not precomputed"
    assert len(stencils.stencils) > 0, "no SOCP-feasible stencils produced"

    for idx, stencil in stencils.stencils.items():
        L = stencil.L
        assert np.isclose(L.sum(), 0.0, atol=1e-9), (
            f"point {idx}: sum(L) = {L.sum():.3e}, expected 0 (Laplacian consistency)"
        )


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_socp_stencil_off_diagonal_nonnegative(sigma):
    """Issue #1074: L[j] >= 0 for off-diagonal j (M-matrix property part 1)."""
    solver = _make_solver(sigma)
    stencils = solver._joint_socp_stencils
    assert stencils is not None

    for idx, stencil in stencils.stencils.items():
        L = stencil.L
        center = stencil.center_in_neighbors
        L_off = np.delete(L, center)
        # Allow tiny SOCP slop (eps_pos default = 0)
        assert np.all(L_off >= -1e-9), (
            f"point {idx}: min(L_off) = {L_off.min():.3e}, "
            f"expected >= 0 (M-matrix off-diagonal sign)"
        )


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_socp_stencil_center_nonpositive(sigma):
    """Issue #1074: L[center] <= 0 (M-matrix property part 2, follows from
    sum-zero + off-diagonal non-negative)."""
    solver = _make_solver(sigma)
    stencils = solver._joint_socp_stencils
    assert stencils is not None

    for idx, stencil in stencils.stencils.items():
        L = stencil.L
        center = stencil.center_in_neighbors
        L_center = L[center]
        assert L_center <= 1e-9, (
            f"point {idx}: L_center = {L_center:.3e}, expected <= 0"
        )


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_socp_stencil_per_edge_cone_bound(sigma):
    """Issue #1074: per-edge cone ||D[:, j]|| <= (C/h_i) * L[j] for each off-j.

    This is the core constraint linking gradient stencils to Laplacian stencils
    that closes the discrete comparison principle proof.
    """
    solver = _make_solver(sigma)
    stencils = solver._joint_socp_stencils
    assert stencils is not None

    C = stencils._C  # cone constant used during construction
    points = stencils._points

    n_violations = 0
    max_violation_ratio = 0.0
    for idx, stencil in stencils.stencils.items():
        L = stencil.L
        D = stencil.D  # shape (dimension, n_neighbors)
        # Recompute h_i (median nonzero neighbor distance, matches construction)
        nbr_idx = stencil.neighbor_indices
        offsets = points[nbr_idx] - points[idx]
        dists = np.linalg.norm(offsets, axis=-1)
        nz = dists[dists > 1e-12]
        h_i = float(np.median(nz)) if len(nz) > 0 else stencils._delta
        bound_per_j = (C / h_i) * L  # shape (n,)

        # Per-edge norm of D[:, j]
        D_norms = np.linalg.norm(D, axis=0)  # shape (n,)

        # Off-diagonal indices
        center = stencil.center_in_neighbors
        off_mask = np.ones(len(L), dtype=bool)
        off_mask[center] = False

        # For off-edges, the cone constraint applies
        for j in np.where(off_mask)[0]:
            if L[j] < 1e-12:
                # If L[j] ≈ 0, the bound becomes ≈ 0 — D[:, j] must also be ≈ 0
                if D_norms[j] > 1e-9:
                    n_violations += 1
                    max_violation_ratio = max(max_violation_ratio, D_norms[j] / 1e-9)
            else:
                ratio = D_norms[j] / bound_per_j[j]
                if ratio > 1.0 + 1e-6:  # tolerate tiny SOCP slop
                    n_violations += 1
                    max_violation_ratio = max(max_violation_ratio, ratio)

    assert n_violations == 0, (
        f"sigma={sigma}: {n_violations} per-edge cone violations, "
        f"max ||D[:,j]|| / ((C/h)*L[j]) = {max_violation_ratio:.4f} (expected <= 1)"
    )


def test_socp_at_least_some_feasible():
    """Issue #1074 sanity: at σ=1 (low-Pe regime), most stencils should be
    SOCP-feasible. If 0/many feasible, joint_socp is producing nothing useful."""
    solver = _make_solver(sigma=1.0, n_x=21)
    stencils = solver._joint_socp_stencils
    assert stencils is not None
    n_feasible = stencils.stats["n_feasible"]
    n_infeasible = stencils.stats["n_infeasible"]
    total = n_feasible + n_infeasible
    assert total > 0, "no stencils attempted"
    # At low Pe with σ=1, expect majority feasibility. Stage 1/2 paper
    # baseline reports >85% at N>=75.
    assert n_feasible > 0, (
        f"sigma=1 low-Pe: 0/{total} stencils SOCP-feasible — "
        f"joint_socp scheme producing no usable stencils"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
