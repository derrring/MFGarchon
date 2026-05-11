"""Regression tests for use_ghost_nodes mixed-BC + tolerance fixes (#1110).

Two bugs covered:

- **Bug A** (``apply_ghost_nodes_to_neighborhoods`` early-exit on mixed BC):
  prior to this fix the method read a single global ``bc_type`` from
  ``_bc_property_getter("type")``, which returns ``None`` for mixed BC.
  Fallback hit ``default_bc.value.lower()`` = ``"periodic"``, the global
  check failed, and ghost augmentation silently did nothing on every
  mixed-BC setup.

- **Bug B** (``compute_outward_normal`` single-point tolerance):
  ``create_ghost_neighbors`` called this method, which delegated to
  ``compute_normal_from_bounds`` with default ``tol=1e-10`` and strict
  ``<``. For ε=1e-6 off-wall boundary points, ``|point − wall| > tol``,
  the function returned ``[0, 0]``, and downstream ghost reflection
  silently produced zero ghosts. PR #1097 fixed the bulk-path sibling
  (``compute_boundary_normals``); this issue + fix covers the single-
  point sibling.

The fix routes ``compute_outward_normal`` through
``bc.identify_boundary_face`` + ``outward_normal_for_face`` (the
canonical PR1 path) and adds per-point dispatch to
``apply_ghost_nodes_to_neighborhoods``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mfgarchon.alg.numerical.gfdm_components.boundary_handler import BoundaryHandler
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.geometry import Hyperrectangle
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions


# ---------------------------------------------------------------------------
# Shared helpers
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


def _eps_cloud(LX, LY, eps=1e-6, n_per_side=6, include_exit_band=False):
    """Interior grid + boundary at ε-off-wall — the regime where the bugs fire.

    If ``include_exit_band``, places extra boundary points on x_max within y∈[4,6]
    so the exit segment in the mixed-BC test has matching points.
    """
    xs = np.linspace(LX * 0.1, LX * 0.9, n_per_side)
    ys = np.linspace(LY * 0.1, LY * 0.9, n_per_side)
    interior = np.array([[x, y] for x in xs for y in ys])
    boundary = []
    for x in xs:
        boundary.append([x, eps])
        boundary.append([x, LY - eps])
    for y in ys:
        boundary.append([eps, y])
        boundary.append([LX - eps, y])
    if include_exit_band:
        # Explicit points on the exit band (y ∈ [4, 6] on x_max)
        for y in np.linspace(4.5, 5.5, 3):
            boundary.append([LX - eps, y])
    boundary = np.array(boundary)
    pts = np.vstack([interior, boundary])
    bdry_idx = np.arange(len(interior), len(pts))
    return pts, bdry_idx


# ---------------------------------------------------------------------------
# Bug B: compute_outward_normal single-point tolerance
# ---------------------------------------------------------------------------


def test_compute_outward_normal_eps_off_wall_returns_unit_normal():
    """BoundaryHandler.compute_outward_normal at ε=1e-6 off-wall point returns ±1 unit normal.

    Pre-fix: ``compute_normal_from_bounds`` with tol=1e-10 + strict ``<`` returned
    ``[0, 0]`` because ``|y − 0| = 1e-6 > 1e-10``. Post-fix: routes through
    ``bc.identify_boundary_face`` (PR1 closed-inequality with tol=1e-6) and
    ``bc.outward_normal_for_face`` (axis-aligned exact ±1).
    """
    LX, LY = 10.0, 10.0
    pts, bdry_idx = _eps_cloud(LX, LY, eps=1e-6, n_per_side=4)
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        dimension=2,
    )
    handler = BoundaryHandler(
        collocation_points=pts,
        dimension=2,
        domain_bounds=np.array([[0.0, LX], [0.0, LY]]),
        boundary_indices=bdry_idx,
        neighborhoods={},
        boundary_conditions=bc,
        use_ghost_nodes=False,
        use_wind_dependent_bc=False,
        gfdm_operator=None,
        bc_property_getter=lambda prop, default=None: default,
    )
    # Test on a y=ε bottom-wall point
    bot_idx = bdry_idx[0]  # first boundary point is bottom wall by construction
    bot_pt = pts[bot_idx]
    assert abs(bot_pt[1]) < 1e-3, f"Setup error: pt {bot_idx} not on bottom wall"
    normal = handler.compute_outward_normal(int(bot_idx))
    # Bottom wall outward normal is [0, -1]
    np.testing.assert_allclose(normal, [0.0, -1.0], atol=1e-12)


def test_compute_outward_normal_matches_bulk_method_on_eps_boundary():
    """Single-point ``compute_outward_normal`` agrees with bulk
    ``compute_boundary_normals`` after PR #1097 + this PR.

    Dual-source consistency is the load-bearing property; if these two
    methods produce different normals for the same point, every downstream
    consumer mixing the two is a latent G-013-style bug.
    """
    LX, LY = 10.0, 10.0
    pts, bdry_idx = _eps_cloud(LX, LY, eps=1e-6, n_per_side=4)
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        dimension=2,
    )
    handler = BoundaryHandler(
        collocation_points=pts,
        dimension=2,
        domain_bounds=np.array([[0.0, LX], [0.0, LY]]),
        boundary_indices=bdry_idx,
        neighborhoods={},
        boundary_conditions=bc,
        use_ghost_nodes=False,
        use_wind_dependent_bc=False,
        gfdm_operator=None,
        bc_property_getter=lambda prop, default=None: default,
    )
    bulk_normals = handler.compute_boundary_normals()
    assert bulk_normals is not None
    for local_idx, global_idx in enumerate(bdry_idx):
        single = handler.compute_outward_normal(int(global_idx))
        np.testing.assert_allclose(
            single,
            bulk_normals[local_idx],
            atol=1e-12,
            err_msg=f"Dual-source mismatch at pt {global_idx}: "
            f"single={single}, bulk={bulk_normals[local_idx]}",
        )


# ---------------------------------------------------------------------------
# Bug A: per-point BC dispatch in apply_ghost_nodes_to_neighborhoods
# ---------------------------------------------------------------------------


def test_apply_ghost_nodes_fires_on_mixed_bc_neumann_points():
    """Mixed BC (NO_FLUX walls + DIRICHLET exit): ghost augmentation must fire
    on the NO_FLUX boundary points and skip the DIRICHLET ones.

    Pre-fix: the method early-exited because global ``bc_type`` was not
    uniform, hitting ``default_bc=PERIODIC`` fallback. Zero ghost augmentation
    everywhere.
    """
    LX, LY = 10.0, 10.0
    pts, bdry_idx = _eps_cloud(LX, LY, eps=1e-6, n_per_side=4, include_exit_band=True)
    # NO_FLUX walls + Dirichlet "exit" segment on right wall mid-band
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(
                name="right_below",
                bc_type=BCType.NO_FLUX,
                boundary="x_max",
                region={"y": (0.0, 4.0)},
            ),
            BCSegment(
                name="right_above",
                bc_type=BCType.NO_FLUX,
                boundary="x_max",
                region={"y": (6.0, LY)},
            ),
            BCSegment(
                name="exit",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="x_max",
                region={"y": (4.0, 6.0)},
                priority=1,
            ),
        ],
        dimension=2,
    )
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    problem = _MockProblem(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry_idx,
            delta=2.0,
            k_neighbors=12,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=False,
            use_ghost_nodes=True,
            boundary_conditions=bc,
            monotonicity_scheme="none",
        )

    handler = s._boundary_handler
    # Count points classified as exit (Dirichlet) vs wall (Neumann)
    n_exit = sum(
        1
        for i in bdry_idx
        if s._bc_segment_per_point[int(i)].bc_type == BCType.DIRICHLET
    )
    n_wall = sum(
        1
        for i in bdry_idx
        if s._bc_segment_per_point[int(i)].bc_type in (BCType.NEUMANN, BCType.NO_FLUX)
    )
    # Setup sanity: at least one of each must exist
    assert n_wall > 0
    assert n_exit > 0

    # Ghost augmentation should fire on the n_wall Neumann points
    n_ghosted = sum(
        1
        for i in bdry_idx
        if handler.neighborhoods.get(int(i), {}).get("has_ghost", False)
    )
    assert n_ghosted == n_wall, (
        f"Expected {n_wall} ghost-augmented points (one per NO_FLUX boundary pt), "
        f"got {n_ghosted}. Bug A regression: ghost augmentation no longer fires "
        f"on mixed-BC NO_FLUX points."
    )
    # Exit points must NOT have ghosts
    for i in bdry_idx:
        i = int(i)
        if s._bc_segment_per_point[i].bc_type == BCType.DIRICHLET:
            has_ghost = handler.neighborhoods.get(i, {}).get("has_ghost", False)
            assert not has_ghost, (
                f"Exit point pt {i} has ghost augmentation — should be skipped for Dirichlet"
            )


def test_apply_ghost_nodes_skips_when_periodic_default():
    """With no NEUMANN/NO_FLUX segments at all, no ghost augmentation fires.

    Specifically: a uniformly periodic BC should not produce ghosts.
    """
    LX, LY = 10.0, 10.0
    pts, bdry_idx = _eps_cloud(LX, LY, eps=1e-6, n_per_side=4)
    from mfgarchon.geometry.boundary import periodic_bc

    bc = periodic_bc(dimension=2)
    geom = Hyperrectangle(np.array([[0.0, LX], [0.0, LY]]))
    problem = _MockProblem(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Periodic uniform — joint_socp doesn't matter, ghost should skip
        s = HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry_idx,
            delta=2.0,
            k_neighbors=12,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=False,
            use_ghost_nodes=True,
            boundary_conditions=bc,
            monotonicity_scheme="none",
        )
    handler = s._boundary_handler
    n_ghosted = sum(
        1
        for i in bdry_idx
        if handler.neighborhoods.get(int(i), {}).get("has_ghost", False)
    )
    assert n_ghosted == 0, (
        f"Periodic BC should not trigger ghost augmentation, got {n_ghosted}"
    )


def test_apply_ghost_nodes_legacy_uniform_neumann_still_works():
    """Uniform NEUMANN BC (legacy non-mixed path) still triggers ghost
    augmentation when ``bc_type_for_point`` arg is omitted.

    Backward-compat: the change adds a parameter; existing callers that
    don't pass it fall to the legacy global ``bc_type`` check which now
    works correctly for genuinely uniform BC.
    """
    LX, LY = 10.0, 10.0
    pts, bdry_idx = _eps_cloud(LX, LY, eps=1e-6, n_per_side=4)
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="all", bc_type=BCType.NO_FLUX, boundary="all"),
        ],
        dimension=2,
        default_bc=BCType.NO_FLUX,
    )
    # Build realistic neighborhoods (each boundary point has nearby interior
    # neighbors), so create_ghost_neighbors can actually reflect them.
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    _, idxs = tree.query(pts, k=8)
    neighborhoods = {
        int(i): {
            "indices": idxs[int(i)],
            "points": pts[idxs[int(i)]],
            "distances": np.linalg.norm(pts[idxs[int(i)]] - pts[int(i)], axis=1),
            "size": len(idxs[int(i)]),
        }
        for i in range(len(pts))
    }
    handler = BoundaryHandler(
        collocation_points=pts,
        dimension=2,
        domain_bounds=np.array([[0.0, LX], [0.0, LY]]),
        boundary_indices=bdry_idx,
        neighborhoods=neighborhoods,
        boundary_conditions=bc,
        use_ghost_nodes=True,
        use_wind_dependent_bc=False,
        gfdm_operator=None,
        bc_property_getter=lambda prop, default=None: "no_flux" if prop == "type" else default,
    )
    # Legacy call with no bc_type_for_point — falls back to the bc_property_
    # getter returning "no_flux" globally. Realistic neighborhood lets the
    # reflection proceed without hitting the post-#1113 raise for empty
    # interior-side neighbors.
    handler.apply_ghost_nodes_to_neighborhoods()
    n_ghosted = sum(
        1
        for i in bdry_idx
        if handler.neighborhoods[int(i)].get("has_ghost", False)
    )
    assert n_ghosted > 0, "Legacy uniform-NEUMANN call must trigger ghost augmentation"
