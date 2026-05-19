"""Unit tests for cloud-geodesic distance computation (Issue #1093).

Graduates the research-side implementation at
``mfg-research/experiments/gfdm_monotonicity_audit/minors/exp09_obstacle_navigation_full/geodesic_distance.py``.

Test coverage:

1. Basic correctness: single source, obstacle-free 2D grid → geodesic ≈
   Euclidean (within k-NN approximation tolerance).
2. Triangle inequality: ``d(a, c) <= d(a, b) + d(b, c)`` for arbitrary
   triples on the graph.
3. Multiple sources: returns ``min_s d(., s)``.
4. Obstacle blocks straight path: a "wall" between source and target
   increases geodesic above Euclidean.
5. Unreachable points: enclosed-by-obstacle points return ``np.inf``.
6. Argument validation: bad ``points`` / ``k_neighbors`` / ``sources_idx`` raise.
7. ``build_geodesic_field`` round-trip: evaluating at cloud points returns
   the input distances; out-of-hull queries fall through to nearest.
8. ``build_geodesic_field`` with unreachable points: substitutes the
   ``unreachable_penalty × d_max`` fill, no `nan` leaks out.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import build_geodesic_field, compute_geodesic_distance

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _grid_2d(n: int, lx: float = 4.0, ly: float = 4.0) -> np.ndarray:
    xs = np.linspace(0.0, lx, n)
    ys = np.linspace(0.0, ly, n)
    return np.array([[x, y] for x in xs for y in ys])


# A vertical wall obstacle at x ≈ 2.0, spanning y ∈ [1.0, 3.0].
# SDF convention (mfgarchon): sd <= 0 means INSIDE obstacle.
def _wall_sdf_factory(x_wall: float = 2.0, thickness: float = 0.2, y_lo: float = 1.0, y_hi: float = 3.0):
    half = thickness / 2.0

    def sdf(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p)
        # Distance to wall in x-direction (negative inside).
        dx = np.abs(p[..., 0] - x_wall) - half
        # Distance to wall in y-direction (negative inside [y_lo, y_hi]).
        dy = np.maximum(y_lo - p[..., 1], p[..., 1] - y_hi)
        # 2D AABB SDF: max of per-axis SDFs (negative when both are negative).
        return np.maximum(dx, dy)

    return sdf


# ---------------------------------------------------------------------------
# 1. Basic correctness on obstacle-free grid
# ---------------------------------------------------------------------------


def test_obstacle_free_geodesic_matches_euclidean():
    """No obstacles → geodesic == Euclidean modulo k-NN graph approximation.

    On a 7x7 grid with k=24 neighbours, the graph is dense enough that the
    geodesic should match Euclidean within a few percent. Stronger bounds
    require denser cloud or larger k.
    """
    pts = _grid_2d(7)
    # Source at corner (0, 0); first point in the array.
    d = compute_geodesic_distance(pts, sources_idx=0, k_neighbors=24)
    assert d[0] == 0.0
    euclid = np.linalg.norm(pts - pts[0], axis=1)
    # Geodesic is an upper bound; should be within ~10% of Euclidean.
    assert np.all(d >= euclid - 1e-9)
    # Tight enough to catch sign errors / wrong source handling.
    rel = (d - euclid) / np.maximum(euclid, 1e-6)
    assert rel.max() < 0.15, f"Max relative excess over Euclidean = {rel.max():.4f}"


# ---------------------------------------------------------------------------
# 2. Triangle inequality
# ---------------------------------------------------------------------------


def test_triangle_inequality_holds_on_graph():
    """For any triple (a, b, c) on the cloud: d_geo(a, c) <= d(a, b) + d(b, c).

    Computed by running Dijkstra from b (source), then triangle inequality
    must hold against pre-computed d(a, ·) and d(c, ·).
    """
    pts = _grid_2d(6)
    src_a, src_b, src_c = 0, 17, 35  # arbitrary cloud indices
    d_a = compute_geodesic_distance(pts, sources_idx=src_a, k_neighbors=15)
    d_b = compute_geodesic_distance(pts, sources_idx=src_b, k_neighbors=15)
    # Triangle: d(a, c) <= d(a, b) + d(b, c)
    lhs = d_a[src_c]
    rhs = d_a[src_b] + d_b[src_c]
    assert np.isfinite(lhs)
    assert np.isfinite(rhs)
    # Tolerance for symmetrisation rounding.
    assert lhs <= rhs + 1e-9, f"Triangle ineq violated: d(a,c)={lhs:.4f} > d(a,b)+d(b,c)={rhs:.4f}"


# ---------------------------------------------------------------------------
# 3. Multiple sources
# ---------------------------------------------------------------------------


def test_multiple_sources_returns_min_over_sources():
    """For sources `S`, output is `min_{s ∈ S} d(., s)`."""
    pts = _grid_2d(5)
    sources = np.array([0, 24])  # corners
    d_multi = compute_geodesic_distance(pts, sources_idx=sources, k_neighbors=15)
    d_one = compute_geodesic_distance(pts, sources_idx=0, k_neighbors=15)
    d_two = compute_geodesic_distance(pts, sources_idx=24, k_neighbors=15)
    expected = np.minimum(d_one, d_two)
    np.testing.assert_allclose(d_multi, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Obstacle blocks straight path
# ---------------------------------------------------------------------------


def test_obstacle_inflates_geodesic_above_euclidean():
    """A vertical wall between source and target → geodesic > Euclidean.

    Wall at x ∈ [1.9, 2.1] × y ∈ [1.0, 3.0]. Source at (0.5, 2.0). Target
    at (3.5, 2.0) sits directly across the wall; Euclidean distance 3.0,
    but the geodesic must route around (over y=3.0 or under y=1.0).
    """
    pts = _grid_2d(7)
    # Source ≈ (0.667, 2.0); target ≈ (3.333, 2.0)
    src_idx = int(np.argmin(np.linalg.norm(pts - np.array([0.667, 2.0]), axis=1)))
    tgt_idx = int(np.argmin(np.linalg.norm(pts - np.array([3.333, 2.0]), axis=1)))
    euclid = float(np.linalg.norm(pts[tgt_idx] - pts[src_idx]))

    d_no_obstacle = compute_geodesic_distance(pts, sources_idx=src_idx, k_neighbors=24)[tgt_idx]
    d_with_wall = compute_geodesic_distance(
        pts, sources_idx=src_idx, obstacles_sdf=_wall_sdf_factory(), k_neighbors=24
    )[tgt_idx]

    assert np.isfinite(d_with_wall), "Geodesic with wall should remain finite (routing exists)"
    assert d_with_wall > d_no_obstacle + 0.5, (
        f"Wall failed to inflate geodesic: no-obstacle={d_no_obstacle:.3f} vs with-wall={d_with_wall:.3f}"
    )
    assert d_with_wall > euclid * 1.1, f"Wall inflation only {d_with_wall / euclid:.3f}× Euclidean; expected >1.1×"


# ---------------------------------------------------------------------------
# 5. Unreachable points
# ---------------------------------------------------------------------------


def test_unreachable_points_return_inf():
    """Source on one side of a full vertical wall → cloud on the other side
    is unreachable through navigable region.
    """
    # Wall spans the full y range so no routing exists.
    full_wall_sdf = _wall_sdf_factory(x_wall=2.0, thickness=0.2, y_lo=-10.0, y_hi=10.0)
    # Cloud strictly outside the wall (navigable on both sides).
    pts_left = np.array([[0.5 + 0.5 * i, 1.0 + 0.5 * j] for i in range(3) for j in range(5)])
    pts_right = np.array([[2.5 + 0.5 * i, 1.0 + 0.5 * j] for i in range(3) for j in range(5)])
    pts = np.vstack([pts_left, pts_right])
    # max_edge_length must be short enough that no edge can hop over the wall
    # (the wall is 0.2 wide; max_edge_length=1.0 still might miss segments
    # whose samples all fall in [0.05, 0.95]·edge length on the same side).
    # Use a tight cap: at most ~0.7 (the y-step is 0.5).
    d = compute_geodesic_distance(
        pts,
        sources_idx=0,
        obstacles_sdf=full_wall_sdf,
        k_neighbors=15,
        max_edge_length=0.75,
    )
    # Source at idx 0 is on the LEFT side. RIGHT side (idx 15-29) is unreachable.
    assert d[0] == 0.0
    assert np.all(np.isfinite(d[:15])), "Left side should be reachable"
    assert np.all(np.isinf(d[15:])), "Right side should be unreachable"


# ---------------------------------------------------------------------------
# 6. Argument validation
# ---------------------------------------------------------------------------


def test_compute_validates_points_shape():
    with pytest.raises(ValueError, match=r"points must have shape"):
        compute_geodesic_distance(np.array([]), sources_idx=0)


def test_compute_validates_k_neighbors():
    pts = _grid_2d(3)
    with pytest.raises(ValueError, match=r"k_neighbors must be"):
        compute_geodesic_distance(pts, sources_idx=0, k_neighbors=0)


def test_compute_validates_sources_in_bounds():
    pts = _grid_2d(3)
    with pytest.raises(ValueError, match=r"sources_idx out of bounds"):
        compute_geodesic_distance(pts, sources_idx=999)


# ---------------------------------------------------------------------------
# 7. build_geodesic_field round-trip + out-of-hull fallback
# ---------------------------------------------------------------------------


def test_field_evaluates_to_input_at_cloud_points():
    """Interpolant at cloud points returns the precomputed d_geodesic."""
    pts = _grid_2d(5)
    d = compute_geodesic_distance(pts, sources_idx=0, k_neighbors=20)
    g = build_geodesic_field(pts, d)
    # Interior cloud points should match exactly (linear interp is exact at nodes).
    interior_mask = np.array([0 < p[0] < 4.0 and 0 < p[1] < 4.0 for p in pts])
    cloud_vals = g(pts[interior_mask])
    np.testing.assert_allclose(cloud_vals, d[interior_mask], atol=1e-9)


def test_field_uses_nearest_outside_convex_hull():
    """Queries outside the Delaunay hull fall back to NearestNDInterpolator,
    not NaN.
    """
    pts = _grid_2d(4)
    d = compute_geodesic_distance(pts, sources_idx=0, k_neighbors=12)
    g = build_geodesic_field(pts, d)
    # Far outside the hull [0, 4]^2:
    far_point = np.array([10.0, 10.0])
    val = g(far_point)
    # Hull corner (3.33..., 3.33...) is nearest among the 4x4 grid; its d ≈ Euclidean.
    expected_nearest = float(d[np.argmin(np.linalg.norm(pts - far_point, axis=1))])
    assert np.isfinite(val)
    assert abs(val - expected_nearest) < 1e-9


# ---------------------------------------------------------------------------
# 8. Unreachable points → penalty substitution
# ---------------------------------------------------------------------------


def test_field_replaces_inf_with_penalty_max():
    """build_geodesic_field substitutes np.inf entries with
    `unreachable_penalty × max(finite distances)`.
    """
    # Construct a tiny scenario by hand: 3 reachable + 1 unreachable.
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    d = np.array([0.0, 1.0, 1.0, np.inf])
    g = build_geodesic_field(pts, d, unreachable_penalty=2.0)
    # Evaluate at the unreachable point: expect penalty * max(finite) = 2 * 1 = 2.0
    val = g(pts[3])
    assert np.isfinite(val)
    assert abs(val - 2.0) < 1e-9


def test_field_raises_when_all_unreachable():
    """If every cloud point is unreachable, building a field is meaningless."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    d = np.array([np.inf, np.inf, np.inf])
    with pytest.raises(ValueError, match=r"all d_geodesic entries are non-finite"):
        build_geodesic_field(pts, d)
