"""Geodesic distance on meshfree clouds with obstacle-aware edge filtering.

Computes geodesic distance from each cloud point to the nearest source via
Dijkstra on a k-NN graph where edges crossing obstacles are excluded
(segment-sample SDF check). The companion `build_geodesic_field` wraps the
per-point distances into a callable for off-grid evaluation (terminal cost
fields, coefficients, etc.).

Companion to the structured-grid Eikonal solvers at
`mfgarchon/geometry/level_set/eikonal/` — same problem (distance to source
set, optionally with obstacle constraints), different geometry primitive:
FMM/FSM on `TensorProductGrid` vs Dijkstra on cloud + k-NN connectivity for
`ImplicitDomain`-flavoured meshfree setups.

Issue #1093. Validated in:
    mfg-research/experiments/gfdm_monotonicity_audit/minors/
        exp09_obstacle_navigation_full/geodesic_distance.py

SDF convention (mfgarchon, see NAMING_CONVENTIONS.md § Geometry SDF
Convention): `obstacles_sdf(x) <= 0` means **inside obstacle**;
`obstacles_sdf(x) > 0` means **outside obstacle (navigable)**.

References
----------
- Sethian (1996) ``A Fast Marching Level Set Method for Monotonically
  Advancing Fronts'' (Proc. Natl. Acad. Sci. USA 93) — the structured-grid
  analogue at `mfgarchon/geometry/level_set/eikonal/`.
- Dijkstra (1959) ``A Note on Two Problems in Connexion with Graphs''
  (Numer. Math. 1) — the graph-shortest-path algorithm used here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)


def compute_geodesic_distance(
    points: np.ndarray,
    sources_idx: np.ndarray | int,
    obstacles_sdf: Callable[[np.ndarray], np.ndarray] | None = None,
    k_neighbors: int = 25,
    n_segment_samples: int = 8,
    max_edge_length: float | None = None,
) -> np.ndarray:
    """Compute geodesic distance from each cloud point to the nearest source.

    Builds a k-nearest-neighbor graph on the cloud, optionally filters
    edges whose segments pass through an obstacle (sampled SDF check),
    then runs Dijkstra from the source set.

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
        Cloud point positions. Caller must guarantee all points are in
        the navigable region (`obstacles_sdf(points) > 0` if obstacles
        are provided); points inside obstacles are not filtered out.
    sources_idx : np.ndarray or int
        Index or indices of points with `d_geodesic = 0`. The output
        is `min_{s ∈ sources_idx} d(., s)`.
    obstacles_sdf : callable or None
        `obstacles_sdf(P) -> SDF` for `P` shape `(M, d)`. Following the
        mfgarchon SDF convention, `SDF <= 0` means **inside obstacle**.
        Edges whose mid-segment samples cross into an obstacle (any
        sample with `SDF <= 0`) are excluded from the graph. When None,
        no edge filtering is performed (pure k-NN Dijkstra; geodesic
        equals Euclidean modulo k-NN connectivity approximation).
    k_neighbors : int, default 25
        Connectivity of the k-NN graph. Higher k = more edges = better
        geodesic approximation but slower precompute. The research-side
        validation used k=25.
    n_segment_samples : int, default 8
        Number of samples per edge for the obstacle-crossing test.
        Sample positions are equispaced in `[0.05, 0.95]`; endpoints are
        skipped because cloud points are in the navigable region by
        construction.
    max_edge_length : float or None
        Upper bound on edge length (Euclidean). Edges longer than this
        are dropped. When None, uses `3 × mean(nearest-neighbor distance)`.

    Returns
    -------
    np.ndarray, shape (N,)
        Geodesic distance from each cloud point to the nearest source.
        `np.inf` for points unreachable through the navigable region.

    Notes
    -----
    Complexity: `O(N · k_neighbors)` edge constructions + `O((N + E) log N)`
    Dijkstra. For `N = 1200`, `k = 25` (the validated regime), precompute
    is ~1 s on a typical workstation.

    Examples
    --------
    >>> points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> d = compute_geodesic_distance(points, sources_idx=0)  # no obstacles
    >>> d[0]
    0.0
    >>> # d_geodesic ≈ Euclidean for obstacle-free quadrilateral
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError(f"points must have shape (N, d) with N >= 1; got shape {pts.shape}")
    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be >= 1; got {k_neighbors}")
    if n_segment_samples < 1:
        raise ValueError(f"n_segment_samples must be >= 1; got {n_segment_samples}")

    n = pts.shape[0]
    sources = np.atleast_1d(np.asarray(sources_idx, dtype=int))
    if sources.ndim != 1 or sources.size == 0:
        raise ValueError(f"sources_idx must be a non-empty 1D index array; got {sources}")
    if int(sources.max()) >= n or int(sources.min()) < 0:
        raise ValueError(f"sources_idx out of bounds for {n} points: range = [{sources.min()}, {sources.max()}]")

    # k-NN connectivity. Query k+1 to skip self-edge at distance 0.
    effective_k = min(k_neighbors + 1, n)
    tree = cKDTree(pts)
    distances, indices = tree.query(pts, k=effective_k)

    if max_edge_length is None:
        # Index 1 is the closest non-self neighbour. Use 3× mean for headroom.
        if effective_k >= 2:
            max_edge_length = 3.0 * float(np.mean(distances[:, 1]))
        else:
            max_edge_length = np.inf

    s_lin = np.linspace(0.05, 0.95, n_segment_samples)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    n_filtered_obstacle = 0
    n_filtered_length = 0

    for i in range(n):
        for k in range(1, effective_k):
            j = int(indices[i, k])
            d_ij = float(distances[i, k])
            if d_ij > max_edge_length:
                n_filtered_length += 1
                continue
            if obstacles_sdf is not None:
                seg = pts[i][None, :] + s_lin[:, None] * (pts[j] - pts[i])[None, :]
                sd = np.asarray(obstacles_sdf(seg))
                if (sd <= 0.0).any():
                    n_filtered_obstacle += 1
                    continue
            rows.append(i)
            cols.append(j)
            data.append(d_ij)

    logger.info(
        "cloud_geodesic: %d edges kept (filtered %d for obstacle crossing, %d for length cap %.3f)",
        len(data),
        n_filtered_obstacle,
        n_filtered_length,
        max_edge_length,
    )

    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrise: undirected geodesic graph.
    graph = graph.maximum(graph.T)

    d_geodesic = dijkstra(graph, indices=sources, return_predecessors=False, min_only=True)

    n_unreachable = int(np.sum(~np.isfinite(d_geodesic)))
    if n_unreachable > 0:
        logger.warning(
            "cloud_geodesic: %d/%d points unreachable from sources through navigable region",
            n_unreachable,
            n,
        )
    return d_geodesic


def build_geodesic_field(
    points: np.ndarray,
    d_geodesic: np.ndarray,
    *,
    unreachable_penalty: float = 1.5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap per-point geodesic distances into a callable for off-grid eval.

    Uses scipy `LinearNDInterpolator` on Delaunay triangulation
    (piecewise-linear on the convex hull) with `NearestNDInterpolator`
    fallback for queries outside the hull (rare but possible for HJB ghost
    points near obstacle corners). No RBF overshoot at medial-axis kinks
    where two geodesic trajectories meet.

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
        Cloud point positions (same as fed to `compute_geodesic_distance`).
    d_geodesic : np.ndarray, shape (N,)
        Per-point geodesic distances from `compute_geodesic_distance`.
        Unreachable points (`np.inf` entries) are replaced by
        `unreachable_penalty * max(finite distances)` before interpolation
        so the field stays finite — a strict penalty for "would have to
        go far" rather than an interpolation discontinuity at `inf`.
    unreachable_penalty : float, default 1.5
        Multiplier applied to `max(finite d_geodesic)` for unreachable
        points before interpolation.

    Returns
    -------
    callable
        `g(x)` accepting a single point (shape `(d,)`) or array
        (shape `(M, d)`). Returns a scalar or `(M,)` array respectively.

    Notes
    -----
    Use case: terminal cost `u(T, x) = 0.5 * G_s * g(x) ** 2` where `g`
    is built from `compute_geodesic_distance(..., sources_idx=exit_points)`.
    Bakes obstacle routing into the terminal cost so the HJB solver does
    not need visibility-aware gradient operators or wall potentials to
    encode it (Strategy 3 in exp09_obstacle_navigation_full).
    """
    pts = np.asarray(points)
    d = np.asarray(d_geodesic, dtype=float).copy()
    finite_mask = np.isfinite(d)
    if not finite_mask.all():
        if not finite_mask.any():
            raise ValueError(
                "build_geodesic_field: all d_geodesic entries are non-finite; no source is reachable from any point."
            )
        d_max = float(d[finite_mask].max())
        d = np.where(finite_mask, d, d_max * unreachable_penalty)
        logger.info(
            "cloud_geodesic: substituted %d unreachable points with %.3f × d_max = %.3f",
            int((~finite_mask).sum()),
            unreachable_penalty,
            d_max * unreachable_penalty,
        )

    linear_interp = LinearNDInterpolator(pts, d, fill_value=np.nan)
    nearest_interp = NearestNDInterpolator(pts, d)

    def g_func(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x)
        single = x_arr.ndim == 1
        if single:
            x_arr = x_arr.reshape(1, -1)
        v = linear_interp(x_arr)
        # Fallback to nearest for any out-of-hull queries.
        nan_mask = np.isnan(v)
        if nan_mask.any():
            v[nan_mask] = nearest_interp(x_arr[nan_mask])
        return float(v[0]) if single else v

    return g_func


__all__ = ["compute_geodesic_distance", "build_geodesic_field"]
