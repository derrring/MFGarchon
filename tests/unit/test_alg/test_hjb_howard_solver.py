"""Direct unit tests for HJBHowardSolver.

HJBHowardSolver graduates the research-side howard_patch.py family
(exp08 1D/2D, exp09, exp11) into mfgarchon proper. Replaces the Newton
inner loop of HJBGFDMSolver._solve_timestep when the Hamiltonian is
strictly convex in p. Resolves Issue #1118 (Newton stalls on |∇u|²
stiffness).

Suite covers:

1. Construction validation: requires SOCP-precomputed stencil_provider.
2. Discretisation enum validation.
3. 1D LQ closed-form: backward sweep reproduces the analytical Riccati
   profile for pure LQ (single-agent, no MFG coupling, no potential).
4. Newton-stall reproducer (Issue #1118): pure LQ regime where Newton
   inner bottoms out at Armijo MIN_ALPHA — Howard converges.
5. Each discretisation option runs to completion (upwind_projection,
   upwind_per_axis, central).
6. 2D smoke test on irregular cloud.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

pytest.importorskip("cvxpy")

from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_howard import HJBHowardSolver
from mfgarchon.geometry import Hyperrectangle
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

# ---------------------------------------------------------------------------
# Minimal mock problem (mirrors test_joint_socp_mirror_symmetry pattern)
# ---------------------------------------------------------------------------


class _MockProblem:
    def __init__(self, geometry, sigma=0.3, T=1.0, Nt=10, dimension=2):
        self.geometry = geometry
        self.dimension = dimension
        self.Nx = 9
        self.Nt = Nt
        self.Dx = 0.1
        self.Dt = T / Nt
        self.sigma = sigma
        self.T = T
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


def _make_2d_cloud(LX=4.0, LY=4.0, nx=5, ny=5, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, LX, nx)
    ys = np.linspace(0.0, LY, ny)
    interior = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if 0 < ix < nx - 1 and 0 < iy < ny - 1:
                interior.append([x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05)])
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
    return pts, bdry_idx, geom


def _make_1d_cloud(LX=2.0, n_int=11):
    interior = np.linspace(0.2, LX - 0.2, n_int).reshape(-1, 1)
    boundary = np.array([[1e-7], [LX - 1e-7]])
    pts = np.vstack([interior, boundary])
    bdry_idx = np.arange(len(interior), len(pts))
    geom = Hyperrectangle(np.array([[0.0, LX]]))
    return pts, bdry_idx, geom


def _make_gfdm_solver(pts, bdry, geom, problem, scheme="joint_socp", k_neighbors=12):
    bc = BoundaryConditions(
        segments=[
            BCSegment(name=f"side_{d}_{end}", bc_type=BCType.NO_FLUX, boundary=f"{ax}_{end}")
            for d in range(problem.dimension)
            for ax in (["x", "y", "z"][d],)
            for end in ("min", "max")
        ],
        dimension=problem.dimension,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            problem,
            collocation_points=pts,
            boundary_indices=bdry,
            delta=1.5,
            k_neighbors=k_neighbors,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            collocation_geometry=geom,
            adaptive_neighborhoods=False,
            boundary_conditions=bc,
            monotonicity_scheme=scheme,
            monotonicity_application="precompute",
        )


# ---------------------------------------------------------------------------
# 1. Construction validation
# ---------------------------------------------------------------------------


def test_construction_requires_joint_socp_stencils():
    """stencil_provider without _joint_socp_stencils raises."""
    _pts, _bdry, geom = _make_2d_cloud()
    problem = _MockProblem(geom)

    class _StubProvider:
        _joint_socp_stencils = None

    with pytest.raises(RuntimeError, match="_joint_socp_stencils"):
        HJBHowardSolver(
            problem,
            stencil_provider=_StubProvider(),
            alpha_star=lambda x, p, m, t: -p,
        )


def test_construction_rejects_unknown_discretisation():
    pts, bdry, geom = _make_2d_cloud()
    problem = _MockProblem(geom)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem)
    with pytest.raises(ValueError, match="discretisation must be one of"):
        HJBHowardSolver(
            problem,
            stencil_provider=gfdm,
            alpha_star=lambda x, p, m, t: -p,
            discretisation="not_a_real_scheme",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 2. 1D LQ closed-form (Riccati profile)
# ---------------------------------------------------------------------------


def test_1d_lq_closed_form_riccati():
    """Pure LQ in 1D, mfgarchon convention.

    HJB residual form is `-u_t + H - (σ²/2)Δu = 0` (NAMING_CONVENTIONS.md
    § HJB Equation Conventions). With u(T, x) = 0.5(x - x_c)² (so terminal
    coefficient P(T) = 0.5), σ = 0, the Riccati ODE reads P'(t) = 2 P².
    Closed form for T = 1: P(t) = 1 / (4 - 2t), so P(0) = 0.25.

    Analytic ratio at off-centre probe ``|x - x_c| = 1``:
        u(0)/u(T) = P(0)/P(T) = 0.25 / 0.5 = 0.5

    See NAMING_CONVENTIONS.md § HJB Equation Conventions § Worked example
    for the full derivation. Probe MUST be off-centre (at centre, the
    quadratic vanishes and the test is uninformative for any P(t)).

    Tolerance is loose (~30%) because:
    - SOCP-corrected D_grad on boundary-buffer interior nodes is
      LSQ-fitted, not exact FD.
    - n_int=11 is a coarse 1D grid.
    - No-flux Neumann-by-extension BC introduces a small wall artifact.

    The load-bearing check is Riccati ordering + coefficient ballpark,
    not high-precision quadrature.
    """
    LX = 4.0
    pts, bdry, geom = _make_1d_cloud(LX=LX, n_int=11)
    problem = _MockProblem(geom, sigma=0.0, T=1.0, Nt=20, dimension=1)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem, k_neighbors=5)

    # Terminal: U(T, x) = 0.5 * x² → G_s = 0.5
    x_pts = pts[:, 0]
    x_c = LX / 2
    U_T = 0.5 * (x_pts - x_c) ** 2

    # Use central D_grad: validates linear-algebra under canonical convention,
    # not the upwind-builder accuracy. Default upwind_projection is first-order
    # LSQ with bias `(1/2)·Σh_j³/Σh_j² · u''`. On this coarse 1D grid
    # (n_int=11, h≈0.36), that bias ≈ 0.6·u'' is comparable to |u'|=1.08,
    # so upwind loses fidelity. On 2D irregular clouds with k=12 averaging
    # over multiple directions + σ>0 diffusion (the regime Howard graduated
    # from per Issue #1118), the bias mitigates and upwind works — that
    # regime is covered by test_each_discretisation_completes and the 2D
    # smoke test below. See `hjb_howard.py` module docstring for the
    # convergence hypothesis (Bokanowski-Maroso-Zidani 2009).
    howard = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,  # H = |p|²/2 → α* = -p
        discretisation="central",
        max_iter=30,
        tol=1e-6,
        volatility_field=0.0,
    )
    U = howard.solve_hjb_system(M_density=None, U_terminal=U_T)

    # Check Riccati at an OFF-center probe. The quadratic vanishes at x_c,
    # so probing the center would always give 0 regardless of P(t). Pick a
    # point with non-zero (x-x_c)² so the Riccati coefficient is observable.
    assert np.all(np.isfinite(U)), "Howard produced non-finite U"
    # Probe at x ≈ x_c - 1.0 (interior, off-center): (x-x_c)² = 1.0
    probe_offset = 1.0
    probe_idx = int(np.argmin(np.abs(x_pts - (x_c - probe_offset))))
    U_T_probe = float(U_T[probe_idx])
    U_0_probe = float(U[0, probe_idx])
    assert U_T_probe > 0.1, f"Test fixture broken: probe at x={x_pts[probe_idx]:.3f} sees U_T={U_T_probe:.4f}"
    # Analytical ratio P(0)/P(T) = 0.25 / 0.5 = 0.5 under mfgarchon's HJB
    # convention `-u_t + H - σ²Δu/2 = 0`. Probe at |x-x_c|=1, T=1, G_s=0.5.
    # See NAMING_CONVENTIONS.md § HJB Equation Conventions § Worked example.
    # Loose bound 0.3-0.85: coarse 1D grid + Neumann-by-extension wall artifact.
    ratio = U_0_probe / U_T_probe
    assert 0.3 < ratio < 0.85, (
        f"Riccati ordering broken: U(0)/U(T) at probe x={x_pts[probe_idx]:.3f} = {ratio:.3f}, "
        f"expected ~0.5 for mfgarchon LQ convention (P(0)/P(T) = 0.25/0.5). "
        f"See NAMING_CONVENTIONS.md § HJB Equation Conventions."
    )


# ---------------------------------------------------------------------------
# 3. Newton-stall reproducer (Issue #1118)
# ---------------------------------------------------------------------------


def test_howard_advances_where_newton_would_stall():
    """The temporal-plateau symptom of Issue #1118 manifests as U(t, x) ≈
    U(T, x) for all t after stall. Howard must produce U that varies
    monotonically backward in time.

    Pure LQ regime, single backward sweep.
    """
    LX = 4.0
    pts, bdry, geom = _make_1d_cloud(LX=LX, n_int=15)
    problem = _MockProblem(geom, sigma=0.0, T=1.0, Nt=10, dimension=1)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem, k_neighbors=5)

    x_pts = pts[:, 0]
    U_T = (x_pts - LX / 2) ** 2  # quadratic terminal cost

    # Use central discretisation for 1D smooth-LQ validation (see the
    # Riccati test docstring for why upwind builders bias on coarse 1D
    # grids; the upwind regime is exercised by the 2D smoke test).
    howard = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,
        discretisation="central",
        volatility_field=0.0,
    )
    U = howard.solve_hjb_system(M_density=None, U_terminal=U_T)

    # Temporal monotonicity at an OFF-center probe: U[Nt] > U[Nt-1] > ... > U[0].
    # The quadratic vanishes at x_c so center-probe would be trivially zero
    # and fail to distinguish plateau from convergence.
    probe_offset = 1.0
    probe_idx = int(np.argmin(np.abs(x_pts - (LX / 2 - probe_offset))))
    profile = np.array([float(U[nt, probe_idx]) for nt in range(problem.Nt + 1)])

    # Strictly decreasing backward (allowing tiny rounding).
    diffs = np.diff(profile)
    assert np.all(diffs >= -1e-6), (
        f"Temporal-plateau or non-monotone profile detected. diffs = {diffs}; profile = {profile}"
    )
    # And total cost-to-go shrinkage is non-trivial (NOT a plateau).
    assert profile[-1] - profile[0] > 0.1 * profile[-1], (
        f"Temporal plateau: U(0) ≈ U(T) ({profile[0]:.4f} vs {profile[-1]:.4f}). "
        f"This is the Issue #1118 symptom Howard must fix."
    )


# ---------------------------------------------------------------------------
# 4. Each discretisation option runs to completion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("discretisation", ["upwind_projection", "upwind_per_axis", "central"])
def test_each_discretisation_completes(discretisation):
    """All three A_adv assembly options produce finite U on a small 2D cloud.

    `central` is included for comparison only and may not converge on
    advection-dominant regimes, but should run without crashing.
    """
    pts, bdry, geom = _make_2d_cloud(nx=5, ny=5)
    problem = _MockProblem(geom, sigma=0.3, T=1.0, Nt=5, dimension=2)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem)

    x_c = np.array([2.0, 2.0])
    U_T = 0.5 * np.sum((pts - x_c[None, :]) ** 2, axis=1)

    howard = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,
        discretisation=discretisation,
        max_iter=15,
    )
    U = howard.solve_hjb_system(M_density=None, U_terminal=U_T)
    assert np.all(np.isfinite(U)), f"discretisation={discretisation} produced NaN/Inf"
    assert U.shape == (problem.Nt + 1, len(pts))
    # Terminal preserved bit-for-bit.
    assert np.allclose(U[problem.Nt], U_T)


# ---------------------------------------------------------------------------
# 5. 2D smoke + running_cost callable
# ---------------------------------------------------------------------------


def test_2d_smoke_with_running_cost_callable():
    """Running cost callable correctly enters the RHS. Use a constant
    running cost: U is shifted by `T · const` relative to the no-cost case.
    Just check that supplying running_cost produces a different result.
    """
    pts, bdry, geom = _make_2d_cloud(nx=4, ny=4)
    problem = _MockProblem(geom, sigma=0.2, T=1.0, Nt=5, dimension=2)
    gfdm = _make_gfdm_solver(pts, bdry, geom, problem)

    x_c = np.array([2.0, 2.0])
    U_T = 0.5 * np.sum((pts - x_c[None, :]) ** 2, axis=1)

    base = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,
    ).solve_hjb_system(M_density=None, U_terminal=U_T)

    n = len(pts)
    rc_const = 0.5
    with_rc = HJBHowardSolver(
        problem,
        stencil_provider=gfdm,
        alpha_star=lambda x, p, m, t: -p,
        running_cost=lambda t_idx: rc_const * np.ones(n),
    ).solve_hjb_system(M_density=None, U_terminal=U_T)

    # Adding a positive running cost shifts U(t<T, x) upward.
    interior_idx = np.array([i for i in range(n) if np.linalg.norm(pts[i] - x_c) < 0.7])
    if len(interior_idx) > 0:
        diff_at_t0 = float(np.mean(with_rc[0, interior_idx]) - np.mean(base[0, interior_idx]))
        assert diff_at_t0 > 0, f"Constant running cost did not increase U(0) at center; diff={diff_at_t0:.4f}"
