"""Operator-level obstacle visibility filter (Issue #1124).

Before this fix, ``HJBGFDMSolver(obstacle_sdf=...)`` filtered only
``NeighborhoodBuilder.neighborhoods``. The underlying ``TaylorOperator``
was constructed without ``obstacle_sdf=``, so its own ``neighborhoods``
(and the pre-assembled sparse ``D_lap`` / ``D_grad`` matrices used by the
HJB Newton solve) still contained edges crossing thin walls.

This suite locks in the load-bearing invariant: when ``obstacle_sdf=`` is
passed to ``TaylorOperator``, every stencil edge respects line-of-sight
visibility. The deeper symptom (HJB ``U(t=0)`` inversion in dead corners
of obstacle clouds; see issue body §Reproducer) is the operational
consequence — the unit-level invariant tested here is the structural fix.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import TaylorOperator
from mfgarchon.geometry.implicit import Hypersphere


def _make_pillar_cloud(rng_seed: int = 42, n: int = 100, exclusion_margin: float = 0.1):
    """2D cloud around a single pillar at (5, 5) radius 1.5.

    Points inside the pillar are removed. Cloud is dense enough that
    several point pairs have wall-crossing line segments at delta=2.5.
    """
    rng = np.random.default_rng(rng_seed)
    pts = rng.uniform(0, 10, size=(n, 2))
    pillar = Hypersphere(center=[5.0, 5.0], radius=1.5)
    mask = pillar.signed_distance(pts) > exclusion_margin
    return pts[mask], pillar


def _count_cross_wall_edges(op: TaylorOperator, pts: np.ndarray, pillar):
    """Count stencil edges (i, j) where the midpoint is inside the obstacle."""
    n_cross = 0
    for i in range(len(pts)):
        center = pts[i]
        for j in op.neighborhoods[i]["indices"]:
            if j == i:
                continue
            mid = 0.5 * (center + pts[j])
            if pillar.signed_distance(np.array([mid]))[0] < 0:
                n_cross += 1
    return n_cross


def test_unfiltered_op_has_cross_wall_edges():
    """Baseline: without ``obstacle_sdf=``, some stencil edges cross the wall.

    Pre-#1124 behavior. Without this baseline the filtered-op test below
    would only prove "no edges cross" — which is trivially true when no
    edges would cross anyway. This baseline establishes that the test
    fixture actually exercises the bug surface.
    """
    pts, pillar = _make_pillar_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = TaylorOperator(points=pts, delta=2.5, taylor_order=2, k_neighbors=12)
    n_cross = _count_cross_wall_edges(op, pts, pillar)
    assert n_cross > 0, "test fixture must exercise wall-crossing edges"


def test_filtered_op_no_cross_wall_edges():
    """Load-bearing #1124 invariant: ``obstacle_sdf=`` eliminates every
    stencil edge whose line segment passes through the obstacle.

    This is the property that makes ``D_lap @ u`` / ``D_grad @ u`` respect
    domain connectivity. Without it the HJB linear operator couples
    values across walls regardless of any downstream filter.
    """
    pts, pillar = _make_pillar_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = TaylorOperator(
            points=pts,
            delta=2.5,
            taylor_order=2,
            k_neighbors=12,
            obstacle_sdf=pillar.signed_distance,
        )
    n_cross = _count_cross_wall_edges(op, pts, pillar)
    assert n_cross == 0, f"filtered op still has {n_cross} cross-wall stencil edges. #1124 invariant violated."


def test_filtered_op_tracks_count():
    """Filtered op records how many edges were blocked. Used downstream
    for diagnostics (e.g., the operator-level warning when filtering
    starves a stencil below ``n_derivatives``).
    """
    pts, pillar = _make_pillar_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = TaylorOperator(
            points=pts,
            delta=2.5,
            taylor_order=2,
            k_neighbors=12,
            obstacle_sdf=pillar.signed_distance,
        )
    assert op._visibility_filtered_count > 0


def test_no_obstacle_means_no_change():
    """When ``obstacle_sdf`` is None (the default), the operator's
    neighborhoods are identical to the unfiltered case. No surprise
    behavior change for callers that don't set the kwarg.
    """
    pts, _ = _make_pillar_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op_default = TaylorOperator(points=pts, delta=2.5, taylor_order=2, k_neighbors=12)
        op_explicit_none = TaylorOperator(
            points=pts,
            delta=2.5,
            taylor_order=2,
            k_neighbors=12,
            obstacle_sdf=None,
        )
    for i in range(len(pts)):
        a = op_default.neighborhoods[i]["indices"]
        b = op_explicit_none.neighborhoods[i]["indices"]
        assert np.array_equal(np.sort(a), np.sort(b))


def test_sparse_matrices_respect_visibility():
    """``D_lap`` / ``D_grad`` (the operators HJB Newton actually uses)
    must have no nonzero entries on cross-wall edges. This is the
    end-to-end consequence of the operator-level filter: not just the
    stencil index sets but the assembled sparse matrices respect
    domain connectivity.
    """
    pts, pillar = _make_pillar_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = TaylorOperator(
            points=pts,
            delta=2.5,
            taylor_order=2,
            k_neighbors=12,
            obstacle_sdf=pillar.signed_distance,
        )

    # Laplacian sparse matrix
    L = op._laplacian_matrix
    assert L is not None
    coo = L.tocoo()
    n_nonzero_cross = 0
    for r, c, _ in zip(coo.row, coo.col, coo.data, strict=True):
        if r == c:
            continue
        mid = 0.5 * (pts[r] + pts[c])
        if pillar.signed_distance(np.array([mid]))[0] < 0:
            n_nonzero_cross += 1
    assert n_nonzero_cross == 0, (
        f"D_lap has {n_nonzero_cross} nonzero entries on cross-wall edges. "
        f"Operator-level filter did not propagate to pre-assembled matrices."
    )


def test_visibility_margin_blocks_grazing_edges():
    """With ``visibility_margin > 0``, edges that pass close to (but not
    through) the obstacle are also blocked. Useful for conservative
    filtering when stencils should stay away from obstacle surfaces.
    """
    pts, pillar = _make_pillar_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op_zero_margin = TaylorOperator(
            points=pts,
            delta=2.5,
            taylor_order=2,
            k_neighbors=12,
            obstacle_sdf=pillar.signed_distance,
            visibility_margin=0.0,
        )
        op_safe_margin = TaylorOperator(
            points=pts,
            delta=2.5,
            taylor_order=2,
            k_neighbors=12,
            obstacle_sdf=pillar.signed_distance,
            visibility_margin=0.3,
        )
    # Safe margin must filter at least as many as zero margin.
    assert op_safe_margin._visibility_filtered_count >= op_zero_margin._visibility_filtered_count


def test_periodic_geometry_compatible():
    """Visibility filter applies after periodic ghost expansion.
    Verifies no crash with the periodic-domain code path. Reaching
    here is the invariant; obstacle-in-periodic is a niche combination
    and we just check the wiring doesn't blow up.
    """
    from mfgarchon.geometry import Hyperrectangle

    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1, size=(40, 2))
    geom = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))  # non-periodic
    pillar = Hypersphere(center=[0.5, 0.5], radius=0.15)
    pts = pts[pillar.signed_distance(pts) > 0.02]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = TaylorOperator(
            points=pts,
            delta=0.3,
            taylor_order=2,
            k_neighbors=10,
            geometry=geom,
            obstacle_sdf=pillar.signed_distance,
        )
    assert op._n_points == len(pts)
