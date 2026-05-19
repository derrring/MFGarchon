"""Direct unit tests for PrecomputedMonotoneStencils.

Pre-#1102, the class accepted only ``operator``: at runtime,
``HJBGFDMSolver.approximate_derivatives`` contracts the precomputed
``L_w`` against ``b = u_neighbors - u_center`` built on
``self.neighborhoods[i]["indices"]``. When ``adaptive_neighborhoods=True``
enlarged the runtime neighborhood (e.g. 53 → 522 at corner buffer points
on a 1200-point Stage C cloud), the two stencils diverged and
``L_w @ b`` raised ``ValueError: matmul: size N is different from K``.

v0.25.0 closed the bug class statically: ``neighborhoods``, ``points``,
``delta`` are required ctor kwargs; the legacy fallback to
``op.get_derivative_weights()`` was deleted. There is no longer a way to
construct a stencil object that silently drifts from runtime neighborhoods.

This suite locks in the v0.25.0 invariants:

1. Required-arg gate: missing ``neighborhoods=`` / ``points=`` / ``delta=``
   raises ``TypeError`` at construction (no silent legacy fallback).
2. Matched-indices path: stencils built on op-equivalent indices produce
   M-matrix-compliant weights.
3. Enlarged-stencil path: ``neighborhoods[i]`` strictly larger than the
   op default produces ``L_w`` of the enlarged length (load-bearing #1102
   invariant).
4. Integration regression: HJBGFDMSolver with ``adaptive_neighborhoods=True``
   and ``monotonicity_scheme="qp_m_matrix"`` constructs end-to-end.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.gfdm_components.precomputed_stencils import (
    PrecomputedMonotoneStencils,
)


def _build_2d_grid_with_neighborhoods(nx: int = 5, ny: int = 5, delta: float = 1.5):
    """Construct a small 2D Cartesian cloud with k-NN-like neighborhoods.

    All boundary points share the same fixed neighbor count, simulating
    the pre-adaptive op state. Returns (points, neighborhoods, boundary_mask).
    """
    from scipy.spatial import cKDTree

    xs = np.linspace(0.0, float(nx - 1), nx)
    ys = np.linspace(0.0, float(ny - 1), ny)
    pts = np.array([[x, y] for x in xs for y in ys])
    n = len(pts)
    tree = cKDTree(pts)
    nh: dict[int, dict] = {}
    for i in range(n):
        _, idx = tree.query(pts[i], k=9)  # includes self
        nh[i] = {"indices": np.asarray(idx, dtype=int)}
    is_boundary = np.zeros(n, dtype=bool)
    for i, p in enumerate(pts):
        if p[0] in (xs[0], xs[-1]) or p[1] in (ys[0], ys[-1]):
            is_boundary[i] = True
    return pts, nh, is_boundary


# ---------------------------------------------------------------------------
# 1. Required-arg gate
# ---------------------------------------------------------------------------


def test_missing_neighborhoods_raises_type_error():
    _, _, is_b = _build_2d_grid_with_neighborhoods()
    with pytest.raises(TypeError):
        PrecomputedMonotoneStencils(is_boundary=is_b)  # type: ignore[call-arg]


def test_missing_points_raises_type_error():
    _, nh, is_b = _build_2d_grid_with_neighborhoods()
    with pytest.raises(TypeError):
        PrecomputedMonotoneStencils(is_boundary=is_b, neighborhoods=nh)  # type: ignore[call-arg]


def test_missing_delta_raises_type_error():
    pts, nh, is_b = _build_2d_grid_with_neighborhoods()
    with pytest.raises(TypeError):
        PrecomputedMonotoneStencils(is_boundary=is_b, neighborhoods=nh, points=pts)  # type: ignore[call-arg]


def test_dimension_3d_rejected():
    pts = np.random.default_rng(0).uniform(size=(20, 3))
    is_b = np.zeros(20, dtype=bool)
    is_b[:5] = True
    nh = {i: {"indices": np.arange(20, dtype=int)} for i in range(20)}
    with pytest.raises(ValueError, match="1D or 2D"):
        PrecomputedMonotoneStencils(
            is_boundary=is_b, neighborhoods=nh, points=pts, delta=1.5
        )


# ---------------------------------------------------------------------------
# 2. Matched-indices path: M-matrix property on op-equivalent stencil
# ---------------------------------------------------------------------------


def test_matched_neighborhoods_produces_m_matrix_compliant_stencils():
    """neighborhoods matching the op's k-NN indices produces stencils
    that are M-matrix compliant after QP (off-diagonals non-negative,
    weights sum to zero modulo tolerance).
    """
    pts, nh, is_b = _build_2d_grid_with_neighborhoods()

    precomp = PrecomputedMonotoneStencils(
        is_boundary=is_b, neighborhoods=nh, points=pts, delta=1.5
    )

    assert precomp.stats["n_boundary"] == int(is_b.sum())
    for i in np.where(is_b)[0]:
        i = int(i)
        sd = precomp.stencils.get(i)
        assert sd is not None, f"missing stencil at boundary point {i}"
        # Same indices as the supplied neighborhoods.
        assert np.array_equal(sd.neighbor_indices, nh[i]["indices"])
        # M-matrix invariants.
        if sd.center_in_neighbors is not None:
            off = np.delete(sd.weights, sd.center_in_neighbors)
            assert np.all(off >= -1e-6), f"point {i}: off-diagonal weights negative"
            assert abs(np.sum(sd.weights)) < 1e-6, f"point {i}: weights do not sum to zero"


# ---------------------------------------------------------------------------
# 3. Enlarged stencil: post-adaptive indices strictly larger than k-NN
# ---------------------------------------------------------------------------


def test_enlarged_neighborhoods_produces_correct_length_weights():
    """When neighborhoods[i] has strictly more indices than the k-NN default
    (simulating adaptive δ-enlargement), L_w must have the enlarged length
    and remain M-matrix compliant after QP. This is the load-bearing #1102
    invariant: precomp L_w aligns with runtime b = u_neighbors - u_center.
    """
    from scipy.spatial import cKDTree

    pts, base_nh, is_b = _build_2d_grid_with_neighborhoods()
    tree = cKDTree(pts)
    enlarged_nh: dict[int, dict] = {}
    for i in range(len(pts)):
        idx = np.asarray(tree.query_ball_point(pts[i], r=3.0), dtype=int)
        enlarged_nh[i] = {"indices": idx}

    precomp = PrecomputedMonotoneStencils(
        is_boundary=is_b, neighborhoods=enlarged_nh, points=pts, delta=1.5
    )

    for i in np.where(is_b)[0]:
        i = int(i)
        sd = precomp.stencils.get(i)
        assert sd is not None, f"missing stencil at boundary point {i}"
        n_enlarged = len(enlarged_nh[i]["indices"])
        n_base = len(base_nh[i]["indices"])
        assert n_enlarged > n_base, (
            f"test fixture broken at point {i}: r=3 not larger than k=8 "
            f"({n_enlarged} <= {n_base})"
        )
        assert len(sd.weights) == n_enlarged, (
            f"point {i}: L_w length {len(sd.weights)} != enlarged stencil "
            f"length {n_enlarged}. #1102 invariant violated."
        )
        assert len(sd.neighbor_indices) == n_enlarged
        if sd.center_in_neighbors is not None:
            off = np.delete(sd.weights, sd.center_in_neighbors)
            assert np.all(off >= -1e-6), f"point {i}: enlarged off-diagonals negative"
            assert abs(np.sum(sd.weights)) < 1e-6, f"point {i}: enlarged weights do not sum to zero"


# ---------------------------------------------------------------------------
# 4. Integration regression — HJBGFDMSolver path
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


def test_solver_constructs_with_adaptive_neighborhoods_and_qp_m_matrix():
    """HJBGFDMSolver(adaptive_neighborhoods=True, monotonicity_scheme="qp_m_matrix")
    must complete construction including the PrecomputedMonotoneStencils
    initialisation, without runtime/precomp size mismatch. Locks in the
    end-to-end #1102 fix.
    """
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
    from mfgarchon.geometry import Hyperrectangle
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    LX, LY = 6.0, 6.0
    rng = np.random.default_rng(0)
    nx, ny = 7, 7
    xs = np.linspace(0.0, LX, nx)
    ys = np.linspace(0.0, LY, ny)
    interior = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if 0 < ix < nx - 1 and 0 < iy < ny - 1:
                interior.append([x + rng.uniform(-0.1, 0.1), y + rng.uniform(-0.1, 0.1)])
    interior = np.asarray(interior)
    eps = 1e-7
    boundary = []
    for x in xs:
        boundary.append([x, eps])
        boundary.append([x, LY - eps])
    for y in ys:
        boundary.append([eps, y])
        boundary.append([LX - eps, y])
    boundary = np.asarray(boundary)
    pts = np.vstack([interior, boundary])
    bdry_idx = np.arange(len(interior), len(pts))

    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        dimension=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = HJBGFDMSolver(
            _MockProblem(geom),
            collocation_points=pts,
            boundary_indices=bdry_idx,
            delta=1.5,
            k_neighbors=12,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=True,
            boundary_conditions=bc,
            monotonicity_scheme="qp_m_matrix",
            monotonicity_application="precompute",
        )

    precomp = s._precomputed_stencils
    assert precomp is not None, "qp_m_matrix + precompute should build stencils"
    for i, sd in precomp.stencils.items():
        runtime_nh = s.neighborhoods[i]["indices"]
        assert len(sd.weights) == len(runtime_nh), (
            f"point {i}: precomp L_w length {len(sd.weights)} != runtime "
            f"neighborhood length {len(runtime_nh)}. #1102 invariant violated."
        )
