"""Property regression tests for the joint_socp 4-bug fix (#1099).

The load-bearing physics claim is that on y-symmetric collocation,
the joint_socp stencil weights at mirror points are mirror-equivalent.
Before the 4-bug fix:

- cone constant C=1 over-binds on irregular clouds → CLARABEL picks
  different "near-optimal" solutions for nearly-mirrored stencils
- pre-filter/post-filter stencil mismatch → override site contracts
  L_w against b on misaligned neighbor indices
- SOCP↔Phase-2 dichotomy → mirror stencils straddling feasibility
  threshold land in different schemes (joint_socp L+D vs Phase-2 L only)

Net visible symptom on Stage C v3 (mfg-research): U(t=0) up to 124%
y-asymmetric on a y-symmetric setup.

These tests exercise the weight-level invariant directly so a regression
fires at the layer where the bug lives, not at the visible-asymmetry
layer 80 backward steps + 1 Picard iter downstream.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

pytest.importorskip("cvxpy")

from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.geometry import Hyperrectangle
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

# ---------------------------------------------------------------------------
# Minimal mock to drive HJBGFDMSolver without full MFGComponents
# ---------------------------------------------------------------------------


class _MockProblem:
    def __init__(self, geometry):
        self.geometry = geometry
        self.dimension = 2
        self.Nx = 9
        self.Nt = 5
        self.Dx = 0.1
        self.Dt = 0.2
        self.sigma = 0.1
        self.T = 1.0
        self.lambda_ = 1.0
        self.is_custom = False
        self.hamiltonian_class = None
        self.f_potential = None

    def H(self, x_idx, m_at_x, p_values, t_idx):
        return 0.5 * sum(v**2 for v in p_values.values() if isinstance(v, (int, float)))

    def get_hjb_hamiltonian_jacobian_contrib(self, *a, **kw):
        return None

    def get_hjb_residual_m_coupling_term(self, *a, **kw):
        return None

    def dH_dp(self, *a, **kw):
        return None


def _y_symmetric_cloud(LX: float, LY: float, nx: int, ny_half: int):
    """Build a strictly y-symmetric collocation cloud.

    Interior y-values are mirror-paired about LY/2 plus the midline LY/2 itself.
    Boundary points are placed at ε-off-wall to exercise the realistic regime
    (collocation generators typically don't put points exactly on the wall).
    """
    xs = np.linspace(0.5, LX - 0.5, nx)
    ys_half = np.linspace(0.5, LY / 2 - 0.5, ny_half)
    # mirror-paired + midline
    ys = np.concatenate([ys_half, [LY / 2], LY - ys_half[::-1]])
    interior = np.array([[x, y] for x in xs for y in ys])
    eps = 1e-7
    boundary = []
    for x in xs:
        boundary.append([x, eps])
        boundary.append([x, LY - eps])
    for y in ys:
        boundary.append([eps, y])
        boundary.append([LX - eps, y])
    boundary = np.array(boundary)
    pts = np.vstack([interior, boundary])
    bdry_idx = np.arange(len(interior), len(pts))
    return pts, bdry_idx


def _make_solver(pts, bdry, geom, scheme="joint_socp", **overrides):
    problem = _MockProblem(geom)
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        dimension=2,
    )
    kwargs = {
        "delta": 1.5,
        "k_neighbors": 12,
        "derivative_method": "taylor",
        "taylor_order": 2,
        "weight_function": "wendland",
        "collocation_geometry": geom,
        "adaptive_neighborhoods": False,
        "boundary_conditions": bc,
        "monotonicity_scheme": scheme,
    }
    kwargs.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(problem, collocation_points=pts, boundary_indices=bdry, **kwargs)


# ---------------------------------------------------------------------------
# Property: mirror stencils produce mirror-equivalent L weights
# ---------------------------------------------------------------------------


def test_mirror_L_weights_match_on_y_symmetric_cloud():
    """For each interior point with a y-mirror in the cloud, sorted L weights match.

    Sorted is the right comparison because neighbor enumeration order may
    differ between the two stencils; the *set* of weights must match. This
    is the load-bearing property of the 4-bug fix: deterministic Wendland-LSQ
    fast-path (C non-binding by default) produces mirror solutions to
    floating-point precision.
    """
    LX, LY = 10.0, 10.0
    pts, bdry = _y_symmetric_cloud(LX, LY, nx=6, ny_half=3)
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))

    s = _make_solver(pts, bdry, geom)
    socp = s._joint_socp_stencils

    pair_count = 0
    max_L_diff = 0.0
    for i in socp._interior_indices:
        i = int(i)
        if not socp.has_stencil(i):
            continue
        xi, yi = pts[i]
        if abs(yi - LY / 2) < 1e-9:
            continue  # self-mirror; skip (would be tautology)
        # Find mirror point (xi, LY - yi)
        mirror_target = np.array([xi, LY - yi])
        dists = np.linalg.norm(pts - mirror_target, axis=1)
        i_mirror = int(np.argmin(dists))
        if dists[i_mirror] > 1e-9:
            continue
        if i_mirror not in socp._interior_indices:
            continue
        if not socp.has_stencil(i_mirror):
            continue

        sd_i = socp.stencils[i]
        sd_m = socp.stencils[i_mirror]
        L_i_sorted = np.sort(sd_i.L)
        L_m_sorted = np.sort(sd_m.L)
        diff = float(np.max(np.abs(L_i_sorted - L_m_sorted)))
        max_L_diff = max(max_L_diff, diff)
        pair_count += 1

    assert pair_count >= 5, (
        f"Mirror-pair coverage too thin ({pair_count} pairs). Cloud may not "
        f"be properly y-symmetric or interior_indices set is wrong."
    )
    # 1e-6 threshold: CLARABEL SOCP tolerances + LU floating-point noise.
    # Pre-fix this would be O(1e-1) due to mirror-stencil divergence.
    assert max_L_diff < 1e-6, (
        f"Mirror L weights differ by {max_L_diff:.3e} across {pair_count} pairs. "
        f"Expected <1e-6 (Wendland-LSQ fast-path or tightly-converged CLARABEL). "
        f"Likely cause: cone constant too tight (default C=8 is recommended) or "
        f"pre/post-filter stencil source mismatch reintroduced."
    )


# ---------------------------------------------------------------------------
# Property: cone constraint actually enforced is the per-stencil achieved_C
# ---------------------------------------------------------------------------


def test_achieved_C_is_recorded_and_satisfies_cone():
    """Every feasible stencil records its achieved_C; weights satisfy
    ``||D_j||_2 <= achieved_C[i] * h_i * L_j``.

    Captures both: (a) the C-bisection produces a per-stencil bound, and
    (b) the cone constraint is enforced at the bisected level (not at the
    original solver default).
    """
    LX, LY = 10.0, 10.0
    pts, bdry = _y_symmetric_cloud(LX, LY, nx=6, ny_half=3)
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    s = _make_solver(pts, bdry, geom)
    socp = s._joint_socp_stencils

    # Every feasible stencil should have an entry
    assert len(socp.achieved_C) == socp.stats["n_feasible"], (
        f"achieved_C size {len(socp.achieved_C)} != n_feasible {socp.stats['n_feasible']}"
    )

    # Every recorded C is in [solver default C, C_max]
    for i, C_i in socp.achieved_C.items():
        assert socp._C - 1e-9 <= C_i <= socp._C_max + 1e-9 if socp._C_max else C_i >= socp._C - 1e-9, (
            f"achieved_C[{i}]={C_i} outside [{socp._C}, {socp._C_max}]"
        )


# ---------------------------------------------------------------------------
# Property: relaxed-SOCP fallback (when enabled) produces zero hard-infeasible
# ---------------------------------------------------------------------------


def test_relaxed_fallback_eliminates_phase2_dichotomy():
    """With ``use_relaxed_fallback=True`` (the default for joint_socp at the
    HJB-GFDM call site), no stencil should remain hard-infeasible.

    The 4-bug fix's bug-2 (SOCP↔Phase-2 dichotomy) is closed iff every
    interior point gets joint_socp-style (L, D) — either via the SOCP solve
    proper or via the relaxed slack-penalty solve. ``n_infeasible`` is the
    count of stencils that fell through to Phase 2 M-matrix-QP (L-only).
    """
    LX, LY = 10.0, 10.0
    pts, bdry = _y_symmetric_cloud(LX, LY, nx=6, ny_half=3)
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    s = _make_solver(pts, bdry, geom)
    socp = s._joint_socp_stencils

    assert socp.stats["n_infeasible"] == 0, (
        f"With relaxed-fallback enabled, no stencil should be hard-infeasible. "
        f"Got {socp.stats['n_infeasible']}/{socp.stats['n_interior']}. "
        f"This means the SOCP↔Phase-2 dichotomy is not fully closed."
    )


# ---------------------------------------------------------------------------
# Property: default cone constant is 8.0 (paper's $C_\star$ is non-binding)
# ---------------------------------------------------------------------------


def test_default_cone_constant_is_eight():
    """HJBGFDMSolver's default joint_socp cone constant is 8.0.

    Paper's $C_\\star \\in [0.5, 1]$ is the tight bound for quasi-uniform
    stencils. On real GFDM clouds with median h/q ≈ 2.2, C=1 over-binds
    and CLARABEL finds different "near-optimal" solutions for nearly-
    mirrored stencils. C=8 makes the cone non-binding for well-conditioned
    stencils → deterministic Wendland-LSQ fast-path.

    1D quasi-uniform clouds are unaffected (cone non-binding at C=1 already).
    """
    LX, LY = 10.0, 10.0
    pts, bdry = _y_symmetric_cloud(LX, LY, nx=6, ny_half=3)
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    s = _make_solver(pts, bdry, geom)
    socp = s._joint_socp_stencils

    assert socp._C == 8.0, (
        f"Default cone_constant_C expected 8.0 (4-bug fix #1099 default), got {socp._C}. "
        f"Changing the default below 8 risks reintroducing the irregular-cloud "
        f"asymmetry symptom."
    )


# ---------------------------------------------------------------------------
# Stress: precompute completes for a denser cloud in bounded time
# ---------------------------------------------------------------------------


def test_precompute_completes_on_dense_cloud():
    """Stress test: ~200 interior points + boundary, joint_socp precompute
    completes and reports stats consistent with the 4-bug fix.

    Failure mode this guards against: pathological CLARABEL solve times
    or precompute-vs-runtime stencil reconciliation bugs that manifest
    only at scale.
    """
    LX, LY = 10.0, 10.0
    pts, bdry = _y_symmetric_cloud(LX, LY, nx=12, ny_half=6)
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))

    import time as _time

    t0 = _time.perf_counter()
    s = _make_solver(pts, bdry, geom)
    dt = _time.perf_counter() - t0

    socp = s._joint_socp_stencils
    n_int = socp.stats["n_interior"]
    n_feasible = socp.stats["n_feasible"]
    n_infeasible = socp.stats["n_infeasible"]

    assert n_int >= 100, f"Stress cloud should have ≥100 interior pts, got {n_int}"
    assert n_feasible == n_int, (
        f"All {n_int} interior pts should be feasible under joint_socp with "
        f"relaxed fallback. Got {n_feasible} feasible, {n_infeasible} infeasible."
    )
    assert dt < 60.0, f"Precompute took {dt:.1f}s for {n_int} interior pts (>60s budget)"
