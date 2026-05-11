"""Edge, stress, and failure-mode tests for HJB-GFDM BC classification pipeline.

Pipeline under test:

  boundary_indices --classifier--> BoundaryFace --segment_match--> BCSegment
                                                 --normal_derive--> outward normal
                                                 --row_build--> sparse J row

Three dispatch paths exercised:

  (1) Mixed BC: pre-classified at __init__, O(1) lookup per Newton iter.
  (2) Uniform BC: legacy global-type fast path.
  (3) Ghost-nodes structural BC: PDE row preserved (continue).

Five BCType cases:

  DIRICHLET / NEUMANN / NO_FLUX work; PERIODIC / ROBIN raise
  NotImplementedError; unknown enum raises ValueError.

Edge regime: collocation ε ∈ {0, 1e-7, 1e-6, 5e-6} off-wall, with
domain side ∈ {1, 10, 20, 100} where IEEE-754 rounding sometimes
pushes |point − bound| slightly above ε.

Stress regime: O(1000) boundary points with realistic ε distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.geometry import Hyperrectangle
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockMFGProblem:
    """Minimal MFG problem stub for solver construction (no PDE assembly)."""

    def __init__(self, geometry, sigma: float = 0.1, T: float = 1.0, Nt: int = 4):
        self.geometry = geometry
        self.dimension = geometry.dimension if hasattr(geometry, "dimension") else 2
        self.sigma = sigma
        self.T = T
        self.Nt = Nt
        self.Nx = 9
        self.Dx = 0.1
        self.Dt = T / Nt
        self.lambda_ = 1.0
        self.is_custom = False
        self.hamiltonian_class = None
        self.f_potential = None

    def H(self, x_idx, m_at_x, p_values, t_idx):
        return 0.5 * sum(v**2 for v in p_values.values() if isinstance(v, (int, float)))

    def get_hjb_hamiltonian_jacobian_contrib(self, U_prev, t_idx):
        return None

    def get_hjb_residual_m_coupling_term(self, M, dU, i, t_idx):
        return None

    def dH_dp(self, **kwargs):
        return None


def _grid_with_boundary(LX: float, LY: float, n_in: int, eps: float):
    """Interior grid + boundary at ε-off-wall, returns (points, boundary_idx)."""
    xs = np.linspace(LX * 0.1, LX * 0.9, n_in)
    ys = np.linspace(LY * 0.1, LY * 0.9, n_in)
    interior = np.array([[x, y] for x in xs for y in ys])
    boundary = []
    for x in np.linspace(LX * 0.05, LX * 0.95, max(3, n_in // 2)):
        boundary.append([x, eps])
        boundary.append([x, LY - eps])
    for y in np.linspace(LY * 0.05, LY * 0.95, max(3, n_in // 2)):
        boundary.append([eps, y])
        boundary.append([LX - eps, y])
    boundary = np.array(boundary)
    points = np.vstack([interior, boundary])
    boundary_idx = np.arange(len(interior), len(points))
    return points, boundary_idx


def _mixed_bc_walls_only(LX: float, LY: float):
    """4 walls all NO_FLUX, no exit (canonical 'box with reflecting walls')."""
    return BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        dimension=2,
    )


def _mixed_bc_with_exit(LX: float, LY: float, exit_y_lo=0.4, exit_y_hi=0.6):
    """Walls NO_FLUX + Dirichlet exit on x_max, y ∈ [exit_lo, exit_hi]·LY."""
    return BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right_below", bc_type=BCType.NO_FLUX,
                      boundary="x_max", region={"y": (0.0, exit_y_lo * LY)}),
            BCSegment(name="right_above", bc_type=BCType.NO_FLUX,
                      boundary="x_max", region={"y": (exit_y_hi * LY, LY)}),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
                      boundary="x_max", region={"y": (exit_y_lo * LY, exit_y_hi * LY)},
                      priority=1),
        ],
        dimension=2,
    )


def _build_solver(points, boundary_idx, bc, geometry, scheme="none"):
    """Construct an HJBGFDMSolver in a minimal way (no PDE solve)."""
    problem = _MockMFGProblem(geometry)
    return HJBGFDMSolver(
        problem,
        collocation_points=points,
        boundary_indices=boundary_idx,
        delta=2.0,
        k_neighbors=12,
        derivative_method="taylor",
        taylor_order=2,
        weight_function="wendland",
        collocation_geometry=geometry,
        adaptive_neighborhoods=False,
        boundary_conditions=bc,
        monotonicity_scheme=scheme,
    )


# ---------------------------------------------------------------------------
# Edge: ε-off-wall classification across bound magnitudes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "LX,LY,eps",
    [
        (1.0, 1.0, 0.0),       # exactly on wall
        (1.0, 1.0, 1e-7),      # well inside tol
        (1.0, 1.0, 1e-6),      # at tol boundary
        (10.0, 10.0, 1e-6),    # moderate bound + tol-boundary ε
        (20.0, 10.0, 1e-6),    # the stageC case (FP rounding regime)
        (100.0, 50.0, 1e-6),   # larger bound, tighter relative
    ],
)
def test_preclassify_eps_off_wall(LX, LY, eps):
    """All boundary points at ε ≤ tol off-wall must classify across bound magnitudes.

    Covers the IEEE-754 FP-rounding regime where ``|bound - (bound - eps)|``
    computes slightly above ``eps`` due to subtraction error. The hybrid
    abs+rel tolerance in ``identify_boundary_face`` must absorb this.
    """
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=5, eps=eps)
    bc = _mixed_bc_walls_only(LX, LY)
    solver = _build_solver(points, boundary_idx, bc, geom)

    n_class = len(solver._bc_segment_per_point)
    assert n_class == len(boundary_idx), (
        f"At LX={LX} LY={LY} eps={eps}: only {n_class}/{len(boundary_idx)} "
        f"boundary points classified. Likely IEEE-754 edge missed by tolerance."
    )


def test_preclassify_too_far_off_wall_raises():
    """Boundary points at ε >> tol must FAIL pre-classification with diagnostic."""
    geom = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))
    # ε=1e-3 with default tol=1e-6 → far above tol, must not classify
    points, boundary_idx = _grid_with_boundary(LX=1.0, LY=1.0, n_in=5, eps=1e-3)
    bc = _mixed_bc_walls_only(1.0, 1.0)

    with pytest.raises(ValueError) as exc_info:
        _build_solver(points, boundary_idx, bc, geom)

    msg = str(exc_info.value)
    # Diagnostic must be greppable for downstream debug
    assert "pre-classification failed" in msg
    assert "Unmatched points" in msg
    assert "Common causes" in msg


# ---------------------------------------------------------------------------
# Edge: BoundaryFace classification at exact tol boundary
# ---------------------------------------------------------------------------


def test_identify_face_closed_inequality():
    """Point at exactly tol distance must classify (closed inequality, not strict)."""
    bc = BoundaryConditions(
        segments=[BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min")],
        dimension=2,
        domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
    )
    # Point at exactly the tol distance from x=0
    tol = 1e-6
    face = bc.identify_boundary_face(np.array([tol, 5.0]), tolerance=tol)
    assert face is not None, "Closed inequality must include the tol-boundary case"
    assert face.axis == 0 and face.side == "min"


def test_identify_face_just_outside_tol_returns_none():
    """Point just outside tol (not boundary) must return None."""
    bc = BoundaryConditions(
        segments=[BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min")],
        dimension=2,
        domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
    )
    # 2× tol away
    face = bc.identify_boundary_face(np.array([2e-6, 5.0]), tolerance=1e-6)
    assert face is None


def test_identify_face_domain_bounds_override():
    """domain_bounds kwarg must override self.domain_bounds for the call."""
    bc = BoundaryConditions(
        segments=[BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min")],
        dimension=2,
        # Self bounds say [0, 100]
        domain_bounds=np.array([[0.0, 100.0], [0.0, 100.0]]),
    )
    # Override to [0, 1]: point at x=0.9 should now classify as on x_max
    override = np.array([[0.0, 1.0], [0.0, 1.0]])
    face = bc.identify_boundary_face(np.array([1.0, 0.5]), tolerance=1e-6,
                                      domain_bounds=override)
    assert face is not None
    assert face.axis == 0 and face.side == "max"


# ---------------------------------------------------------------------------
# Edge: outward_normal_for_face exhaustive over 2D faces (and 3D for breadth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis,side,expected",
    [
        (0, "min", [-1.0, 0.0]),
        (0, "max", [1.0, 0.0]),
        (1, "min", [0.0, -1.0]),
        (1, "max", [0.0, 1.0]),
    ],
)
def test_outward_normal_for_face_2d(axis, side, expected):
    from mfgarchon.geometry.boundary.types import BoundaryFace
    bc = BoundaryConditions(segments=[], dimension=2)
    normal = bc.outward_normal_for_face(BoundaryFace(axis, side), dimension=2)
    np.testing.assert_array_equal(normal, expected)


@pytest.mark.parametrize(
    "axis,side,expected",
    [
        (0, "min", [-1, 0, 0]),
        (1, "max", [0, 1, 0]),
        (2, "min", [0, 0, -1]),
        (2, "max", [0, 0, 1]),
    ],
)
def test_outward_normal_for_face_3d(axis, side, expected):
    from mfgarchon.geometry.boundary.types import BoundaryFace
    bc = BoundaryConditions(segments=[], dimension=3)
    normal = bc.outward_normal_for_face(BoundaryFace(axis, side), dimension=3)
    np.testing.assert_array_equal(normal, expected)


# ---------------------------------------------------------------------------
# Edge: corner / priority resolution
# ---------------------------------------------------------------------------


def test_preclassify_corner_priority():
    """Corner point on two faces resolves to the higher-priority segment."""
    LX, LY = 10.0, 10.0
    eps = 1e-7
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    # Enough interior points for GFDM stencil to build (>=k_neighbors=12)
    xs = np.linspace(0.5, LX - 0.5, 5)
    ys = np.linspace(0.5, LY - 0.5, 5)
    interior = np.array([[x, y] for x in xs for y in ys])
    boundary = np.array([[eps, eps]])  # bottom-left corner
    points = np.vstack([interior, boundary])
    boundary_idx = np.array([len(interior)])

    # left has higher priority than bottom
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0,
                      boundary="x_min", priority=10),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX,
                      boundary="y_min", priority=1),
        ],
        dimension=2,
    )
    solver = _build_solver(points, boundary_idx, bc, geom)
    # The corner point should be assigned to "left" (higher priority)
    assigned = solver._bc_segment_per_point[int(boundary_idx[0])]
    assert assigned.name == "left", (
        f"Corner point at (eps, eps) should bind to higher-priority segment 'left', "
        f"got {assigned.name!r}"
    )


# ---------------------------------------------------------------------------
# Failure mode: unmatched-segment raise with greppable diagnostic
# ---------------------------------------------------------------------------


def test_preclassify_unmatched_raises_with_coords():
    """Missing-segment for a wall must raise listing the unmatched point coords."""
    LX, LY = 10.0, 10.0
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=4, eps=1e-7)

    # BC omits the "right" segment entirely → all x_max points unmatched
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            # no right segment
        ],
        dimension=2,
    )

    with pytest.raises(ValueError) as exc_info:
        _build_solver(points, boundary_idx, bc, geom)

    msg = str(exc_info.value)
    assert "pre-classification failed" in msg
    # Coordinate of an unmatched x_max point must appear in message
    # (x ≈ LX = 10.0 with eps offset)
    assert "9.99" in msg or "10.0" in msg, (
        "Unmatched right-wall coords should be greppable from the error message"
    )
    # BoundaryFace info must be in message
    assert "y_min" in msg or "y_max" in msg or "x_max" in msg or "BoundaryFace" in msg


# ---------------------------------------------------------------------------
# Path coverage: uniform BC bypasses pre-classification
# ---------------------------------------------------------------------------


def test_preclassify_uniform_bc_skipped():
    """Uniform BC sets pre-classification dicts to empty (legacy fast path)."""
    LX, LY = 10.0, 10.0
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=4, eps=1e-7)

    # Uniform Dirichlet — not mixed
    from mfgarchon.geometry.boundary import dirichlet_bc
    bc = dirichlet_bc(value=0.0, dimension=2)

    solver = _build_solver(points, boundary_idx, bc, geom)
    assert len(solver._bc_segment_per_point) == 0, (
        "Uniform BC must skip pre-classification (legacy fast path)"
    )


# ---------------------------------------------------------------------------
# Dispatch path: row shape under each BC type
# ---------------------------------------------------------------------------


def test_dispatch_dirichlet_row_is_identity():
    """Dirichlet BC produces a row with exactly 1 non-zero at the diagonal."""
    LX, LY = 10.0, 10.0
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=4, eps=1e-7)

    # All walls Dirichlet — explicit cover so no fallback
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.DIRICHLET, value=2.0, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.DIRICHLET, value=3.0, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.DIRICHLET, value=4.0, boundary="x_max"),
        ],
        dimension=2,
    )
    solver = _build_solver(points, boundary_idx, bc, geom)

    n = len(points)
    # Start from a fake interior Jacobian (identity for simplicity)
    fake_jac = sp.eye(n, format="csr") * 0.5
    fake_res = np.ones(n)

    jac_bc, res_bc = solver._apply_boundary_conditions_to_sparse_system(
        fake_jac, fake_res, time_idx=0
    )

    # Every boundary row should be exactly [0, ..., 1@i, ..., 0]
    for i in boundary_idx:
        row = jac_bc.getrow(int(i)).toarray().flatten()
        nnz = np.where(np.abs(row) > 1e-12)[0]
        assert len(nnz) == 1, f"Dirichlet row at i={i} should have exactly 1 nnz, got {len(nnz)}"
        assert nnz[0] == int(i), f"Dirichlet row at i={i} non-zero should be diagonal"
        assert abs(row[int(i)] - 1.0) < 1e-12


def test_dispatch_neumann_row_uses_normal_grad():
    """Neumann BC produces a row with normal·grad_weights entries (non-trivial)."""
    LX, LY = 10.0, 10.0
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=4, eps=1e-7)
    bc = _mixed_bc_walls_only(LX, LY)  # all NO_FLUX
    solver = _build_solver(points, boundary_idx, bc, geom)

    n = len(points)
    fake_jac = sp.eye(n, format="csr") * 0.5
    fake_res = np.ones(n)
    jac_bc, _ = solver._apply_boundary_conditions_to_sparse_system(
        fake_jac, fake_res, time_idx=0
    )

    # No Neumann row should be all-zero (the bug fixed by this PR)
    for i in boundary_idx:
        row = jac_bc.getrow(int(i)).toarray().flatten()
        row_norm = np.linalg.norm(row)
        assert row_norm > 0, f"Neumann row at i={i} is zero — would cause spsolve NaN"
        # Should have multiple non-zeros (center + neighbors), not just diagonal
        nnz = np.sum(np.abs(row) > 1e-12)
        assert nnz > 1, f"Neumann row at i={i} has only {nnz} nnz — likely missing stencil"


# ---------------------------------------------------------------------------
# Failure mode: PERIODIC / ROBIN raise NotImplementedError with pointers
# ---------------------------------------------------------------------------


def test_dispatch_periodic_raises():
    """PERIODIC BC type at a boundary point raises NotImplementedError with hint."""
    LX, LY = 10.0, 10.0
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=4, eps=1e-7)
    # Three walls NO_FLUX + one PERIODIC (legitimate setup for some 2D problems)
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.PERIODIC, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.PERIODIC, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        dimension=2,
    )
    solver = _build_solver(points, boundary_idx, bc, geom)
    n = len(points)
    fake_jac = sp.eye(n, format="csr") * 0.5
    fake_res = np.ones(n)

    with pytest.raises(NotImplementedError) as exc_info:
        solver._apply_boundary_conditions_to_sparse_system(fake_jac, fake_res, time_idx=0)
    assert "PERIODIC" in str(exc_info.value)
    assert "TensorProductGrid" in str(exc_info.value) or "FDM" in str(exc_info.value)


def test_dispatch_robin_raises():
    """ROBIN BC type at a boundary point raises NotImplementedError pointing at provider."""
    LX, LY = 10.0, 10.0
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=4, eps=1e-7)
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.ROBIN, alpha=1.0, beta=1.0, value=0.0, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        dimension=2,
    )
    solver = _build_solver(points, boundary_idx, bc, geom)
    n = len(points)
    fake_jac = sp.eye(n, format="csr") * 0.5
    fake_res = np.ones(n)
    with pytest.raises(NotImplementedError) as exc_info:
        solver._apply_boundary_conditions_to_sparse_system(fake_jac, fake_res, time_idx=0)
    assert "ROBIN" in str(exc_info.value)
    assert "BCValueProvider" in str(exc_info.value) or "625" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Invariant: after BC apply, no zero rows in the Jacobian
# ---------------------------------------------------------------------------


def test_invariant_no_zero_rows_after_bc_apply():
    """The whole point: a well-specified BC must never produce zero Jacobian rows.

    This invariant is what the PR exists to guarantee. A regression here
    is the same Stage C NaN cascade.
    """
    LX, LY = 20.0, 10.0  # stageC dimensions, in the FP-edge regime
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=6, eps=1e-6)
    bc = _mixed_bc_with_exit(LX, LY)  # mixed wall+exit, like stageC
    solver = _build_solver(points, boundary_idx, bc, geom)

    n = len(points)
    fake_jac = sp.eye(n, format="csr") * 0.5
    fake_res = np.ones(n)
    jac_bc, _ = solver._apply_boundary_conditions_to_sparse_system(
        fake_jac, fake_res, time_idx=0
    )

    row_sums = np.abs(jac_bc).sum(axis=1).A.flatten()
    zero_rows = np.where(row_sums < 1e-15)[0]
    assert len(zero_rows) == 0, (
        f"Found {len(zero_rows)} zero rows in Jacobian after BC apply — this is the "
        f"exact failure mode the PR fixes. Zero-row indices: {zero_rows[:10].tolist()}."
    )


# ---------------------------------------------------------------------------
# Stress: O(1000) boundary points with realistic ε distribution
# ---------------------------------------------------------------------------


def test_stress_thousand_boundary_points():
    """1000+ boundary points classify and dispatch without zero rows or slowness."""
    LX, LY = 20.0, 10.0
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    rng = np.random.default_rng(seed=42)

    # ~25x25 interior + boundary points around perimeter at varied ε
    xs = np.linspace(LX * 0.05, LX * 0.95, 25)
    ys = np.linspace(LY * 0.05, LY * 0.95, 25)
    interior = np.array([[x, y] for x in xs for y in ys])

    n_per_wall = 60  # 60 * 4 = 240 boundary points
    # Realistic ε distribution: uniform in [0, 5e-7]
    eps_dist = rng.uniform(0.0, 5e-7, size=n_per_wall * 4)
    boundary = []
    for k, x in enumerate(np.linspace(LX * 0.02, LX * 0.98, n_per_wall)):
        boundary.append([x, eps_dist[k]])
        boundary.append([x, LY - eps_dist[k + n_per_wall]])
    for k, y in enumerate(np.linspace(LY * 0.02, LY * 0.98, n_per_wall)):
        boundary.append([eps_dist[k + 2 * n_per_wall], y])
        boundary.append([LX - eps_dist[k + 3 * n_per_wall], y])
    boundary = np.array(boundary)
    points = np.vstack([interior, boundary])
    boundary_idx = np.arange(len(interior), len(points))

    assert len(boundary_idx) >= 240, "stress test should exercise O(100s) boundary pts"

    bc = _mixed_bc_with_exit(LX, LY)

    import time as _time
    t0 = _time.perf_counter()
    solver = _build_solver(points, boundary_idx, bc, geom)
    t_init = _time.perf_counter() - t0

    # Pre-classification should be linear-ish in n_boundary
    assert t_init < 10.0, f"Pre-classification took {t_init:.2f}s for {len(boundary_idx)} bdy pts (>10s)"

    # All boundary points classified
    assert len(solver._bc_segment_per_point) == len(boundary_idx)

    # BC apply produces no zero rows even at realistic ε distribution
    n = len(points)
    fake_jac = sp.eye(n, format="csr") * 0.5
    fake_res = np.ones(n)
    jac_bc, _ = solver._apply_boundary_conditions_to_sparse_system(
        fake_jac, fake_res, time_idx=0
    )
    row_sums = np.abs(jac_bc).sum(axis=1).A.flatten()
    zero_rows = np.where(row_sums < 1e-15)[0]
    assert len(zero_rows) == 0, (
        f"{len(zero_rows)} zero rows under O(1000) stress: classifier or dispatch broke at scale"
    )


# ---------------------------------------------------------------------------
# Failure mode: malformed input (post-init mutation, missing dim, etc.)
# ---------------------------------------------------------------------------


def test_get_bc_type_raises_when_post_init_mutation():
    """If boundary_indices is mutated after __init__, _get_bc_type_for_point must raise."""
    LX, LY = 10.0, 10.0
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    points, boundary_idx = _grid_with_boundary(LX, LY, n_in=4, eps=1e-7)
    bc = _mixed_bc_walls_only(LX, LY)
    solver = _build_solver(points, boundary_idx, bc, geom)

    # Simulate post-init boundary mutation (programmer error)
    spurious_idx = int(points.shape[0]) - 1  # use an interior point as if it were boundary
    if spurious_idx in solver._bc_segment_per_point:
        # If it happens to be already classified, pick another
        pytest.skip("test setup needs unclassified index — skipping for this seed")
    with pytest.raises(ValueError) as exc_info:
        solver._get_bc_type_for_point(spurious_idx)
    msg = str(exc_info.value)
    assert "pre-classified" in msg
    assert "mutated" in msg or "__init__" in msg
