# Changelog

All notable changes to MFGArchon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **HJB-FP volatility consistency check in `FixedPointIterator`** (Issue #1082).
  Warns when `volatility_field=X` is passed AND `problem.sigma=Y` with
  `X != Y` (scalar case). HJB sees Y, FP sees X — Picard fixed point not
  a coherent MFG. Same trap pattern as #811. Silent for callable / matched.

- **Empirical per-stencil M-matrix verification tests for joint_socp**
  (Issue #1074, partial). New `tests/unit/test_alg/test_socp_m_matrix_property.py`
  verifies the 4 stencil-level invariants the paper claim depends on, across
  σ ∈ {0.5, 1.0, 1.5}: Laplacian consistency `sum(L)=0`, off-diagonal
  non-negative `L[off] ≥ 0`, center non-positive `L[center] ≤ 0`, and the
  per-edge cone bound `‖D[:,j]‖ ≤ (C/h_i) · L[j]` (the non-trivial constraint
  that closes the discrete comparison principle proof). Full assembled-matrix
  M-matrix verification deferred (depends on dt + advection regime).

### Changed

- **CFL diagnostic logging now emits at INFO once per solver instance**, then
  DEBUG on subsequent calls (Issue #1052). Previously every Picard iteration
  emitted the same "CFL diagnostic" line at INFO, spamming user logs and
  causing researchers to blanket-suppress warnings (which masked unrelated
  DeprecationWarnings — the Tier-C silent-semantic-shift bugs in #1043 went
  unnoticed for weeks partially for this reason). Applies to `HJBFDMSolver`
  and `FPFDMSolver`.

- **`SeparableHamiltonian.potential` docstring** now explicitly documents the
  "potential as reward" sign convention (Issue #1057, gotcha G-001). For an
  attractive potential at `x_c`, write `V(x, t) = -0.5*C*(x-x_c)**2`
  (inverted parabola, peak at `x_c`); for repulsive, write `+0.5*C*(x-x_c)**2`
  (bowl). This is opposite to standard MFG literature where V is "cost to
  avoid"; mfgarchon's convention is reward, agents concentrate at V_max.

### Changed

- **`JAXBackend` JIT cache uses explicit None-init pattern** (Issue #1068, partial).
  Replaced 4 `hasattr(self, "_jit_*")` duck-typing checks with explicit
  `is None` initialization in `__init__`. Per CLAUDE.md "Object Shape
  Stability". Other #1068 hasattr clusters (core/mfg_components, types/protocols)
  deferred — those need Protocol/ABC design.

### Fixed

- **`HJBGFDMSolver` diffusion-term arithmetic in `scheme="none"` path**
  (Issue #1073). Four sites in `hjb_gfdm.py` (residual_vectorized at L1840,
  residual_hamiltonian at L1889, jacobian_hamiltonian at L1926,
  jacobian_vectorized at L2056) sourced σ via the chain
  `getattr(self.problem, "diffusion", 0.0) or getattr(self.problem, "sigma", 0.0)`.
  Because `problem.diffusion` returns `σ²/2` (the PDE coefficient `D`) and
  is truthy whenever σ > 0, this resolved σ to `D`, then computed
  `0.5 · D² · Δu = (σ⁴/8) · Δu` instead of `D · Δu = (σ²/2) · Δu`.

  Ratio of buggy/correct = `σ²/4`:

  | σ | ratio | severity |
  |---|---|---|
  | 0.3 | 0.022 | 44× too small (paper Stage 3 high-Pe regime) |
  | 0.5 | 0.063 | 16× too small |
  | 1.0 | 0.250 | 4× too small |
  | 1.414 | 0.500 | 2× too small |
  | 2.0 | 1.000 | accidentally correct |

  Fix: replace all 4 sites with `self._get_sigma_value(None)` (same pattern
  already used correctly by `_compute_hjb_residual_with_cache` at L2024).

  **Active path**: only when `monotonicity_scheme="none"` (the default).
  QP/SOCP modes (`joint_socp`, `qp_m_matrix`) take a different code path
  via `_compute_hjb_residual_with_cache` and were always correct. So:
  - Tutorial / default-scheme users at σ ≠ 2 were getting wrong diffusion
  - Production paper experiments using `joint_socp` were unaffected
  - σ=2 is the only value where the bug coincidentally cancels

  Same trap pattern as Issue #811 (`MFGProblem(diffusion=...)` vs
  `sigma=`); cross-references same docstring at `core/mfg_problem.py:1306-1317`.
- **Picard NaN/Inf diagnostic now identifies HJB vs FP source** (Issue #1078).
  Previously `fixed_point_iterator.py:804` (Issue #688 fix) emitted a generic
  "NaN/Inf detected" warning when terminating early on non-finite iterates,
  with no indication of which side blew up. Now examines `U_new` / `M_new`
  (still in scope from earlier in the loop iteration) and labels the source
  as `HJB (Newton divergence)`, `FP (density blow-up)`, `both`, or
  `post-damping (likely Anderson acceleration)`. Five-line change, no new
  control flow.
- **`FPParticleSolver` meshfree KDE now uses reflection ghosts on reflecting
  BC axes** (Issue #1083). Previously `fp_particle.py:2026-2029` constructed
  `ParticleDensityQuery` without `reflect_bounds`, so boundary cells were
  underestimated by ~50% (per `particle_density_query.py:558` known limit).
  For Towel-on-Beach Gaussian with stall near the wall, this biased the next
  Picard iteration's drift, producing a wrong fixed point.

  New helper `_infer_reflect_bounds()` examines `self.boundary_conditions`
  and returns the bounds list when at least one segment is `NO_FLUX` /
  `REFLECTING` / `NEUMANN`. Per-axis disambiguation is deferred until BC
  framework exposes segment→axis mapping.- **`enforce_obstacle_boundary` no longer captures particles past the outer
- **`np.linalg.inv()` → `np.linalg.solve()` in 2 hot paths** (Issue #1066,
  partial — neighborhood_builder cache deferred). `joint_socp.py:193`
  computed `ATA_inv` then matmul'd with `e_grad[d]` in a Python loop;
  now uses a single `solve(ATA, [e_lap|e_grad].T)` (kills Python loop +
  squares fewer condition numbers). `sampling.py:736` Mahalanobis used
  `inv(cov)` (silently wrong for cond > 1e10); now uses `solve(cov, ...)`
  + `einsum`. Third site `neighborhood_builder.py:744` deferred
  (long-lived cache needs `lu_factor`/`lu_solve`).- **`enforce_obstacle_boundary` no longer captures particles past the outer  bounding box** (Issue #1064). When `FPParticleSolver` is configured with
- **`TensorProductGrid` validates `Nx_points >= 1` + finite/ordered bounds**
  (Issue #1077, partial). `Nx_points=[10, 0, 5]` and `bounds=[[1, 0]]` (lo > hi)
  now raise `ValueError`. N=1 (single-point grid, zero spacing) preserved.
  Other input-validation cases in #1077 deferred.- **`enforce_obstacle_boundary` no longer captures particles past the outer  bounding box** (Issue #1064). When `FPParticleSolver` is configured with  both `implicit_domain` (for obstacle reflection) and a `BoundaryConditions`
  containing a Dirichlet (absorbing) segment on the outer boundary,
  `enforce_obstacle_boundary` was projecting **all** particles outside the
  navigable region back inside — including those that had crossed the
  Dirichlet exit segment. The segment-aware BC then never saw them, so
  `total_absorbed` stayed at 0 and absorbing exits were silently disabled.

  The fix discriminates by bounding-box membership: only particles **inside
  the outer bbox** but in an obstacle interior get re-projected. Particles
  **past the outer bbox** are an outer-boundary concern and are left for
  the caller's segment-aware BC (which handles reflect / absorb / wrap per
  segment). Composes correctly with #1042 (callable-drift segment-aware
  routing).

- **`HJBSemiLagrangianSolver._stochastic_sl_step_nd` companion fixes**
  (Issue #1054): apply the analogous trio of correctness fixes to the nD
  stochastic SL path:
  1. **Monotone interpolation**: when `interpolation_method ∈ {"cubic",
     "quintic"}`, route through `RegularGridInterpolator(method="pchip")`
     (tensor-product monotone Hermite, scipy ≥ 1.10) instead of the
     non-monotone tensor-product cubic. Mirrors 1D Issue #1033.
  2. **Per-axis BC handling on Brownian feet**: apply iterated mirror
     reflection (`reflect`) or modular wrap (`wrap`) per axis to
     `y_plus`/`y_minus` before interpolation. Previously the nD path
     silently extrapolated via `bounds_error=False, fill_value=None`,
     producing values dependent on the nearest interior cell rather than
     respecting the SDE's reflection/periodicity. Mirrors 1D Issue #1048.
  3. **Vectorized batch interpolation**: replace per-(node, axis)
     `_interpolate_value` calls (which rebuilt the interpolator each call)
     with a single `RegularGridInterpolator` built once and queried on the
     full `(2*d*N_total, d)` departure batch. Linear interpolation
     continues to work alongside stochastic dispatch (Issue #1049 carries
     through).

- **`HJBSemiLagrangianSolver._stochastic_sl_step_1d` trio of fixes** (Issues
  #1033, #1048, #1049):
  1. **#1033**: replace `scipy.interpolate.CubicSpline` (non-monotone, blew up
     on stiff problems with `max|∇u|` exponential growth 6 → 100 → 10⁶ → NaN
     on 1D Towel-on-Beach in 17 Picard iters) with `PchipInterpolator`
     (monotone Hermite). Linear interpolation now uses `np.interp` directly
     when `interpolation_method="linear"`.
  2. **#1048**: replace `np.clip(y, xmin, xmax)` boundary handling with
     iterated mirror reflection `xmin + |((y − xmin) mod 2L) − L|`. Clamping
     collapsed all out-of-bounds characteristic feet onto the boundary node,
     biasing toward wall values and breaking upwind property near reflective
     boundaries. Reflection matches the underlying SDE's behavior for Neumann.
  3. **#1049**: remove the validation that **rejected** `interpolation_method
     ="linear"` with `diffusion_method="stochastic"` — that combination IS the
     proven-stable Carlini-Silva 2014 canonical scheme; the previously-required
     `cubic` is non-monotone and outside the stability proof. Now `linear` is
     the unwarned default for stochastic; cubic/quintic emit a `UserWarning`
     pointing to the proof status.

  See `mfg-research/docs/mfgarchon_gotchas.md` G-008 / G-009 / G-010 for the
  research-side audit. The dim-agnostic refactor (unifying `_stochastic_sl_step_1d`
  and `_stochastic_sl_step_nd` per the project's "dimension as parameter, not
  constraint" principle) is tracked separately as Issue #1050; this PR fixes
  the 1D path only.
- **`FPParticleSolver._get_grid_params` now fails fast on geometries that
  expose neither `.bounds`, `.xmin`/`.xmax`, nor `.coordinates`** (Issue #1053).
  Previously fell through to a silent `[(0.0, 1.0)] * dimension` fallback
  (the unit hypercube), which corrupted FP particle simulation on any
  non-standard geometry without a clear error. Now raises `TypeError` with
  a diagnostic pointing at the missing API.
- **`ImplicitDomain.project_to_domain(method='simple')` now uses Newton-on-SDF
  as a fallback** when the original line-search-toward-bbox-center fails
  (Issue #1047). Previously the line-search exhaustion path silently teleported
  particles to the bounding-box center — geometrically incorrect (e.g. for a
  navigable region with an off-center obstacle, every failure-to-project
  collapsed particles to one point, producing KDE singular covariance
  downstream with no clear diagnostic). Now: line search first; on failure,
  Newton iteration `x ← x − φ(x)·∇φ(x)/|∇φ|²` (uses `sdf_gradient` from
  `sdf_utils`); if both fail (degenerate gradient or non-converging), raises
  `RuntimeError` with diagnostic instead of silent corruption. Fail-fast.- **`FPParticleSolver._solve_fp_system_callable_drift` now honors segment-aware
  Dirichlet absorbing boundary conditions** (Issue #1042). Previously the
  callable-drift path always routed through `_apply_boundary_conditions_nd`
  (uniform topology BC) and ignored `boundary_conditions=BoundaryConditions(segments=[...])`
  with Dirichlet exit segments. Particles approaching exits piled up indefinitely
  instead of being absorbed; the grid-based-drift path correctly routed through
  `_apply_boundary_conditions_segment_aware`, but the callable-drift path bypassed
  it. The fix mirrors the grid-drift segment-aware branching: per-step variable
  particle count via list storage, Dirichlet absorption applied via
  `_apply_boundary_conditions_segment_aware`, exit-flux tracking populated
  (`exit_flux_history`, `total_absorbed`). Verified by trajectory storage type
  (now `list` for segment-aware vs `ndarray` for uniform).

- **CSG composite domains (`UnionDomain`, `IntersectionDomain`, `DifferenceDomain`,
  `ComplementDomain`) now expose `.bounds`** (Issue #1041), mirroring the
  `Hyperrectangle.bounds` API. Previously they had only `get_bounding_box()`,
  causing `FPParticleSolver._get_grid_params` to silently fall back to the
  unit hypercube `[(0, 1)] * d` when reading `geom.bounds`. On non-unit
  domains (e.g., `[0, 18] × [0, 8]`) particles got reflected/clipped against
  the wrong domain after every FP step → KDE singular covariance downstream.
  After the fix, FPParticleSolver reads the actual domain bounds end-to-end.
  `ComplementDomain.bounds` is a property delegating to `get_bounding_box()`
  (raises if not manually set, since `ComplementDomain` is unbounded by
  default — fail-fast is correct here).

### Added

- **`HJBGFDMSolver` now emits a `UserWarning` when `monotonicity_scheme` is
  unspecified** (Issue #1034). The default resolves to `"none"` (no QP
  correction), producing bare Wendland-Taylor LSQ stencils whose M-matrix
  structure is not enforced. On long-time-horizon problems (e.g. 1D
  Towel-on-Beach at T=8) this destabilizes FP-Particle coupling and produces
  catastrophic boundary oscillation. The warning surfaces the trap and points
  users to `monotonicity_scheme='joint_socp'` (paper-canonical) or
  `'qp_m_matrix'` (cheaper). Users intentionally using the bare scheme can
  pass `monotonicity_scheme='none'` explicitly to suppress the warning.
  Validated in
  `mfg-research/.../exp08_towel_2d_validation/_preflight_1d/post_mortem_1d_tob_debug.md`.

### Changed

- **Documented `HJBGFDMSolver.obstacle_sdf` sign convention** (Issue #1038).
  Convention: ``obstacle_sdf(x) < 0`` means "x is INSIDE the obstacle (to be
  filtered)". This matches a single-obstacle ``Hypersphere``/``Hyperrectangle``
  ``.signed_distance`` natively but is **inverted** for a CSG composite like
  ``DifferenceDomain.signed_distance`` (which uses the standard navigable-region
  convention). Pass ``obstacle.signed_distance`` directly, not
  ``domain.signed_distance``. Docstring example added in both
  ``HJBGFDMSolver.__init__`` and ``NeighborhoodBuilder.__init__``.

### Fixed

- **`ImplicitDomain.num_spatial_points` now caches the result** (Issue #1037).
  Previously the property recomputed from an unseeded Monte-Carlo volume
  estimate on every call, returning slightly different values across calls
  within one process. Downstream callers like
  `MFGComponents._setup_custom_initial_density` that pre-allocate based on
  the value and then iterate the spatial grid would overrun and surface an
  unhelpful `IndexError: index N out of bounds for size N` (Issue #1036,
  obsoleted by this cache fix).

## [0.19.6] - 2026-05-06

### Fixed

- **`HJBGFDMSolver.approximate_derivatives` now consults precomputed
  monotonicity-corrected weights (J/r consistency)** when the slow path is
  taken. Before this fix, when `monotonicity_scheme="joint_socp"` (or legacy
  `qp_optimization_level="precompute"`), the per-point HJB Newton path used
  inconsistent stencil weights:
    - **Jacobian**: assembled from `_cached_derivative_weights[i]` — populated
      with SOCP / M-matrix-QP weights at __init__ (PR #1030 fix).
    - **Residual**: computed by `approximate_derivatives` slow path, which
      used the bare Wendland-Taylor LSQ (`taylor_data["AtWA_inv"]` etc.),
      *bypassing* any precomputed monotonicity correction.

  Newton then solved `J · δu = -r` with `J` and `r` derived from different
  stencil weights, converging to a stationary point of the mongrel system
  rather than the true discrete-HJB fixed point. Empirically: at the exp08
  step 4 2D Towel-on-Beach validation N=100, raw `‖U_HJB,centered‖₂` at
  iter 1 was 244 with the inconsistency vs ~130 after the fix (47%
  reduction). At N=75 the inconsistency was tolerable (28% of nodes had
  bare W-T in BOTH J and r since SOCP was infeasible there); at finer h
  with higher SOCP coverage, the inconsistency dominated.

  The fix overrides gradient and Laplacian-trace entries in the multi-index
  derivative dict with values computed from the precomputed weights, only
  for nodes that have a precomputed stencil. Behavior is unchanged for
  nodes without a precomputed stencil and for `monotonicity_scheme="none"`
  (which routes through the fast path of `approximate_derivatives`).

### Notes

- This is a correctness fix, orthogonal to the user-visible API. No
  deprecation, no parameter changes. The 16 equivalence tests for the
  v0.18.0 `qp_optimization_level` rename continue to pass with bit-identical
  weights.

## [0.19.5] - 2026-05-06

### Added

- **Two-axis monotonicity API on `HJBGFDMSolver`** (PR #1030):
  - `monotonicity_scheme: "none" | "qp_m_matrix" | "joint_socp"` — what kind of
    constraint to enforce on stencil weights.
  - `monotonicity_application: "adaptive" | "always" | "precompute" | None` —
    when/how it is enforced (per-point QP at runtime vs. precomputed at
    construction).
  - Replaces the legacy `qp_optimization_level=` bundled parameter; equivalence
    is bit-identical, covered by 12 tests in
    `tests/unit/test_alg/test_hjb_gfdm_monotonicity_scheme_rename.py`.
- **First-class `monotonicity_scheme="joint_socp"` option** — precomputes
  joint SOCP-constrained weights (M-matrix on $-\Delta_h$ + per-edge cone
  $\|D_j\|_2 \le C\,h_i\,L_j$) at construction. Includes a Wendland-LSQ
  fast-path (paper Theorem `thm:joint_socp_feasibility`) and a CLARABEL
  CVXPY fallback. Replaces the research-side `patch_operator` monkey-patch
  workflow used through v0.19.4.
- **New module `mfgarchon/alg/numerical/gfdm_components/joint_socp.py`** with
  `PrecomputedJointSocpStencils`, mirroring `PrecomputedMonotoneStencils`.

### Deprecated

- **`qp_optimization_level=`** parameter on `HJBGFDMSolver`. Still accepted
  via `@deprecated_parameter` alias (3 minor versions / 6 months removal
  timeline per `DEPRECATION_LIFECYCLE_POLICY.md`). Emits `DeprecationWarning`
  and translates to the new two-axis API internally with bit-identical
  results.

### Fixed

- **HJB Newton Jacobian now consults precomputed SOCP / M-matrix-QP weights.**
  The lazy fill of `_cached_derivative_weights` (around line 2006 of
  `hjb_gfdm.py`) previously read directly from `_gfdm_operator.get_derivative_weights`,
  bypassing precomputed-stencil overrides — so `_D_lap` / `_D_grad` (used by
  the batch Hamiltonian path) saw SOCP-corrected weights, but the per-point
  Newton Jacobian saw bare Wendland-Taylor. This caused a 12× `u_err` gap
  in the exp08 2D Towel-on-Beach validation between the research-side
  `patch_operator` workflow and the new first-class `joint_socp` scheme; the
  fix restores numerical equivalence (`u_err = 2.115` to 4 sig figs at iter 1
  in both paths).
- **`monotonicity_scheme="joint_socp"` now aliases internal
  `qp_optimization_level` to `"precompute"`** (previously `"none"`). The
  legacy value silently gates HJB Newton path selection: `"none"` selects
  the batch Hamiltonian path, anything else selects per-point. SOCP weights
  must be consumed by the per-point path to match the legacy patch_operator
  workflow.

## [0.19.4] - 2026-04-18

### Removed (BREAKING)

- **`mfgarchon.config.structured_schemas`** module deleted (Issue #1010 B4).
  It defined 13 OmegaConf dataclass schemas (`MFGSchema`, `BeachProblemSchema`,
  `NewtonSchema`, `HJBSchema`, etc.) that encoded a tree shape different from
  the canonical Pydantic `MFGSolverConfig` (nested `solver.hjb.method` vs
  flat `hjb.method`). These schemas were used only by test code and by a
  handful of loader methods on `OmegaConfManager`; no production code loaded
  them. Keeping them alongside the canonical Pydantic hierarchy was the
  dual-schema smell the v0.19.0–v0.19.3 renovation was eliminating elsewhere.
- **Dataclass-tied methods removed from `OmegaConfManager`** (Issue #1010 B3):
  `load_structured_config`, `load_mfg_config`, `load_beach_config_structured`,
  `create_default_mfg_config`, `validate_structured_config`. All returned
  `TypedMFGConfig` / `TypedBeachConfig` dataclass-shaped objects that are
  now gone. The `TypedMFGConfig` / `TypedBeachConfig` type aliases were
  removed along with them.
- **Module-level dataclass wrappers removed**: `load_structured_mfg_config`,
  `load_structured_beach_config`, `create_default_structured_config` (all in
  `mfgarchon.config.omegaconf_manager`).

### Kept (no change)

- **Generic OmegaConf functionality**: `OmegaConfManager.{load_config,
  compose_config, create_pydantic_config, save_config, create_parameter_sweep,
  validate_config, get_config_template}`. These operate on plain YAML /
  DictConfig and do not depend on dataclass schemas. Parameter sweeps and
  CLI overrides keep working via these methods.
- **`bridge_to_pydantic`**: the one-way gate between OmegaConf DictConfig and
  Pydantic `MFGSolverConfig` remains the canonical validation point. Users
  wanting validated configs should call `OmegaConf.load(...)` then pipe the
  result through `bridge_to_pydantic`.
- **YAML example files** in `configs/*.yaml`: kept as user-facing examples.
  These use the OmegaConf-style tree (`problem.T`, `solver.hjb.method`);
  users who want Pydantic validation should transform to the flat Pydantic
  shape first or use them as OmegaConf-only loads.

### Tests

- **Removed** `tests/unit/test_config/test_structured_configs.py` (247 lines)
  and `tests/unit/test_config/test_structured_schemas.py` (632 lines). These
  tested the dataclass tree that no longer exists. Pydantic-side coverage
  lives in `test_core.py`, `test_mfg_methods.py`, and `test_bridge.py` added
  in v0.19.3.

### Migration

User code that called the removed APIs (unlikely — internal audit found zero
production callers outside `mfgarchon/config/` itself) should migrate to the
OmegaConf + bridge pattern:

```python
# Old (removed):
from mfgarchon.config.omegaconf_manager import load_structured_mfg_config
config = load_structured_mfg_config("config.yaml")
# config was a DictConfig with MFGSchema tree shape

# New:
from omegaconf import OmegaConf
from mfgarchon.config import MFGSolverConfig
from mfgarchon.config.bridge import bridge_to_pydantic

raw = OmegaConf.load("config.yaml")
config = bridge_to_pydantic(raw, MFGSolverConfig)  # Pydantic validation at this point
```

Note that the YAML file's tree shape may need adjustment to match Pydantic's
flat `{hjb, fp, picard, backend, logging}` structure; the legacy YAMLs use
`{problem, solver, experiment}` nesting.

### Context

Closes the B3+B4 items of Issue #1010. With this release, the config system
has **one canonical schema authority** (Pydantic models in `core.py`,
`mfg_methods.py`, `array_validation.py`) and **one validation crossing**
(`bridge_to_pydantic`). OmegaConf handles YAML transport only — no schemas.
The North Star design from v0.19.0 is now fully realized.

## [0.19.3] - 2026-04-18

### Changed

- **Internal cleanup** (B1.5b follow-up): `create_network_mfg_solver` now
  forwards the canonical `relaxation` kwarg to `FixedPointIterator` internally
  instead of the legacy `damping_factor`. The legacy-forwarding was a
  deliberate temporary measure to keep the B1.5b.3 PR mergeable before B1.5b.1
  (FixedPointIterator rename) landed. With both now on main, the factory no
  longer emits a self-generated `DeprecationWarning` from its own codebase.

### Fixed

- **`ExperimentConfig` forward-ref crash under pydantic 2.12.5** (Issue #1010 B5):
  `NDArray` was imported under `TYPE_CHECKING` in `mfgarchon/config/array_validation.py`
  but used as an annotation on the `MFGArrays.U_solution` / `M_solution` fields.
  Pydantic 2.12.5+ resolves field annotations at model-build time and rejected the
  unresolved forward reference with `PydanticUserError: class-not-fully-defined`,
  breaking any instantiation of `ExperimentConfig`, `MFGArrays`, or
  `CollocationConfig`. Fixed by importing `NDArray` at runtime (not under
  `TYPE_CHECKING`), with a `noqa: TC002` on the import explaining why the
  runtime import is deliberate. The `MFGArrays.model_rebuild()` and
  `ExperimentConfig.model_rebuild()` workaround calls in `test_array_validation.py`
  are no longer needed and have been removed.

### Tests

- **Canonical config module coverage** (Issue #1010 B2):
  Added dedicated unit tests for `mfgarchon/config/core.py` and
  `mfgarchon/config/mfg_methods.py`, which previously had only indirect
  coverage via factory tests and integration tests. Two new files (58 tests):
  - `tests/unit/test_config/test_core.py` — 23 tests covering
    `LoggingConfig`, `BackendConfig`, canonical-path `PicardConfig`, and
    `MFGSolverConfig` (defaults, range validators, `@model_validator` hooks,
    `save_intermediate`-requires-`output_dir`, `numpy`-cannot-use-`gpu`,
    `anderson_memory <= max_iterations`, `model_dump` round-trip).
  - `tests/unit/test_config/test_mfg_methods.py` — 35 tests covering the 14
    method configs: default instantiation, Literal/enum rejection of
    invalid values, range-bound enforcement, `@model_validator` hooks
    (e.g., `wind_dependent_bc` requires `ghost_nodes`, FEM auto-quadrature).

## [0.19.2] - 2026-04-18

### Changed — B1.5b series (solver ctor kwargs damping_* → relaxation_*)

Incremental rename propagating the naming change landed in v0.19.1 (`PicardConfig`)
through the solver constructors. Four sub-PRs shipped together in this release,
each adding `@deprecated_parameter` decorators + silent `@property` aliases for
backward compatibility. Removal of legacy names scheduled for v0.25.0 per the
3-version deprecation window.

- **B1.5b.1** (PR #1012): `FixedPointIterator` — 7 ctor kwargs renamed
  (`damping_factor`, `damping_factor_M`, `adaptive_damping`, `adaptive_damping_decay`,
  `adaptive_damping_min`, `damping_schedule`, `damping_schedule_M`). Legacy kwargs
  accepted via `@deprecated_parameter` + body redirect. Silent `@property` aliases
  preserve `iter.damping_factor` attribute reads without warning-flooding Picard
  hot loops. 16 equivalence tests in
  `tests/unit/test_alg/test_fixed_point_iterator_relaxation_alias.py`.
- **B1.5b.2** (PR #1013): Block iterators — `BlockIterator` (base),
  `BlockJacobiIterator`, `BlockGaussSeidelIterator`. Renames `damping_factor` →
  `relaxation` and `damping_factor_M` → `relaxation_M` on all three. Legacy kwargs
  accepted via `@deprecated_parameter`. Silent `@property` aliases on base
  class. Plus `SolverResult.metadata` key `"damping_factor"` → `"relaxation"`
  (narrow break; no bridge available for dict-key reads; impact limited to code
  that inspects `result.metadata["damping_factor"]`). 18 equivalence tests in
  `tests/unit/test_alg/test_block_iterators_relaxation_alias.py`.
- **B1.5b.3** (PR #1014): `NetworkMFGSolver` factory functions + `MultiPopulationIterator`.
  `MultiPopulationIterator.__init__(damping_factor=)` → `relaxation=` with
  `@deprecated_parameter` + silent `@property` alias for attribute access.
  `create_network_mfg_solver(damping_factor=)` → `relaxation=`, and
  `create_simple_network_solver(damping=)` → `relaxation=` (same `@deprecated_parameter`
  pattern applied to factory functions, not just classes). Internal
  `self.damping_factor` attribute reads rewritten to `self.relaxation`;
  single-letter aliases (`omega = self.damping_factor`) removed in favor of direct
  `self.relaxation` use per internal style preference. 7 equivalence tests in
  `tests/unit/test_alg/test_network_multipop_relaxation_alias.py`.
- **B1.5b.4** (PR #1015): `HJBFDMSolver` + `FixedPointSolver` (utils/numerical).
  Same pattern: `damping_factor` → `relaxation` ctor kwarg, silent `@property`
  alias. HJBFDMSolver's internal construction of `FixedPointSolver` now forwards
  `relaxation=...` (canonical), and docstring "recommend 0.5-0.8" numerical
  guidance removed per the no-opinionated-numerical-recommendations style. 12
  equivalence tests in
  `tests/unit/test_alg/test_hjb_fdm_fp_solver_relaxation_alias.py`.

## [0.19.1] - 2026-04-17

### Changed

- **`PicardConfig` field rename** (naming abstraction): the five damping-related
  fields are renamed from `damping_*` to `relaxation_*`:

  | Legacy field (deprecated) | Canonical field |
  |---|---|
  | `damping_factor` | `relaxation` |
  | `damping_factor_M` | `relaxation_M` |
  | `damping_schedule` | `relaxation_schedule` |
  | `damping_schedule_M` | `relaxation_schedule_M` |
  | `adaptive_damping` | `adaptive_relaxation` |

  `relaxation` is the more abstract name — it extends cleanly to over-relaxation
  (omega > 1) if the range constraint is loosened in future work, whereas
  "damping" is conceptually under-relaxation only. `FixedPointIterator` reads
  of `config.picard.*` updated to canonical names.

### Deprecated

- Legacy `damping_*` kwargs on `PicardConfig(...)` still accepted via
  `@model_validator(mode="before")` translation, with `DeprecationWarning`
  emitted. Removal scheduled for **v0.25.0** per standard 3-version window.
  Passing both legacy and canonical names for the same concept raises
  `ValueError` immediately (e.g. `PicardConfig(damping_factor=0.5, relaxation=0.8)`).

### Tests

- New `tests/unit/test_config/test_picard_relaxation_alias.py` (15 tests)
  provides the mandatory equivalence tests per CLAUDE.md deprecation policy:
  each legacy kwarg produces an instance `==` to the canonical kwarg.

### Out of scope (future patches)

- Solver constructor kwargs (`FixedPointIterator(damping_factor=...)`,
  `HJBFDMSolver(damping_factor=...)`, etc.) are **not** renamed in this
  release. The runtime-layer rename via `@deprecated_parameter` is B1.5b,
  tracked in #1010.

## [0.19.0] - 2026-04-17

### Removed (BREAKING)

- **`mfgarchon.config.pydantic_config`** — the legacy parallel config hierarchy.
  All 7 exported classes (`NewtonConfig`, `PicardConfig`, `GFDMConfig`, `ParticleConfig`,
  `HJBConfig`, `FPConfig`, `MFGSolverConfig`) are now available exclusively from
  `mfgarchon.config`. See `docs/user/migration_v0.19.md` for import updates and
  field-by-field mapping (legacy defaults differed from canonical by up to 1000x in
  some fields, e.g. `PicardConfig.tolerance: 1e-3 -> 1e-6`).
- Phantom factory functions removed from user docs: `create_fast_config`,
  `create_accurate_config`, `create_research_config`, `create_enhanced_config`
  (never existed as public API; docs referenced them in error). Use
  `create_fast_solver` / `create_accurate_solver` / `create_research_solver` from
  `mfgarchon.factory` for preset patterns, or `MFGSolverConfig()` for direct config.
- **`hydra-core>=1.3`** dependency (declared but unused — zero `@hydra.main`,
  `from hydra`, or `HydraConfig` references in the codebase). Can be reintroduced
  deliberately if HPC sweep workflows or config-group based solver selection
  become priorities.

### Fixed

- **GraphMFGSolver source_term alignment** (Issue #1006): `_get_time_slice` in
  `graph_coupling.py` had a hardcoded `dt=0.05` default. Any problem with
  `dt != 0.05` silently indexed wrong time slices — invisible to tests that
  happened to use `dt=0.05`. `dt` is now threaded through
  `compute_hjb_source` / `compute_fp_source` and required at the indexing site.
- **GraphMFGSolver source composability**: per-node `problem.source_term_hjb`,
  `problem.source_term_fp`, and `problem.nonlocal_operator` were ignored when
  combined with graph coupling (only the graph source was injected into the
  HJB/FP solvers). New `_compose_hjb_source` / `_compose_fp_source` methods
  layer problem-level sources on top of the graph coupling source, matching
  the Layer 1 design's composability promise.
- `GraphMFGSolver.__init__` now validates that all nodes share the same `dt`
  (required for coupling to be well-defined); raises `ValueError` otherwise.

### Changed

- `pyproject.toml`: version bumped `0.18.19` -> `0.19.0`.
- `mfgarchon/config/omegaconf_manager.py`: `MFGSolverConfig` now imported from
  canonical `.core` module (was `.pydantic_config`).
- User docs (`plugin_development.md`, `migration.md`, `usage_patterns.md`):
  updated 6 import statements to canonical `from mfgarchon.config import ...`
  path.
- `GraphCouplingOperator.compute_hjb_source` / `compute_fp_source`: signature
  changed from `(..., t: float)` (vestigial, never used) to `(..., dt: float)`
  (load-bearing, used for time indexing). Callers of the protocol need to
  update their kwarg name.

### Audit context

Driven by a dual-config-system audit that revealed every legacy/canonical class
pair had diverged — both in schema (different fields) and in defaults (up to 1000x,
e.g. `PicardConfig.tolerance`). A simple deprecation-redirect was impossible since
the hierarchies were different APIs rather than versions of one API. v0.19.0 is a
hard break; subsequent v0.19.x patches will complete the internal consolidation
(canonical-module tests, YAML loader migration, removal of the remaining OmegaConf
dataclass mirrors, and `ExperimentConfig` NDArray forward-ref fix). Umbrella
tracking issue: #1010.

## [0.18.0] - 2026-03-29

### Added

- **Geometry trait compliance** (PR #872)
  - `ImplicitDomain`: `manifold_dimension`, `get_tangent_space_basis()`, `compute_christoffel_symbols()`, `validate_lipschitz_regularity()`
  - `GraphGeometry`: `mark_region()`, `get_region_mask()`, `get_region_names()`, `intersect_regions()`, `union_regions()`
  - Region predicate factories: `box_region()`, `sphere_region()`, `sdf_region()`, `halfspace_region()` in `geometry.predicates`
- **Periodic BC model compatibility** warning in user guide (quadratic potential on periodic domain pitfall)

### Changed

- `GhostCellConfig` relocated from `_compat.py` to `ghost_cells.py` (canonical location)
- Common noise MFG test updated for modern API (Issues #670, #673)

### Removed

- `HybridFPParticleHJBFDM` coupling solver (-757 lines) — deprecated since v0.9.0, 9 versions past policy

### Fixed

- Stale `See Also` references in boundary conditions user guide (docs migrated to mfg-research)

## [0.17.16] - 2026-03-28

### Added

- **BC resolution layer** — `MathBCType`, `BCResolver`, `HJBResolver`, `FPResolver` (PR #856, Issue #848)
- **Periodic BC for SL diffusion** — Sherman-Morrison circulant solver (PR #865, Issue #858)
- **Jupyter Book v2 (MyST)** configuration for docs (PR #864)
- Runtime `DeprecationWarning` for legacy `fdm_bc_1d.BoundaryConditions` (PR #869)

### Changed

- FP FDM boundary assembly decoupled — dict dispatch, Dirichlet nD, fail-fast (PR #868, Issue #859)
- Boundary module restructured — split monolith, slim exports (PR #849, Issue #848)
- BC design docs consolidated from 14 to 10 files (PR #850)
- ~111 raw deprecation warnings migrated to structured decorators (PR #847, Issue #841)
- All FP FDM tests migrated to modern BC API (PR #870)
- All remaining tests migrated from legacy fdm_bc_1d (PR #871, -706 lines)

### Removed

- Duplicate FEM BC system (PR #851, Issue #848)
- Theory docs moved to mfg-research (-7,567 lines) (PRs #853-855, Issue #852)
- Stale `docs/theory/` references (-429 lines) (PR #863)
- 28 RL placeholder test files (-9,051 lines) (PR #867, Issue #833)

## [0.17.13] - 2026-03-26

### Added

- **True adjoint mode** (Issue #707, PR #829)
  - `HJBFDMSolver.build_linearized_operator(U, M, time)` — builds linearized HJB Jacobian
  - `adjoint_mode="jacobian_transpose"` in `BlockIterator` for true adjoint FP coupling
  - `LinearizedOperatorCapable` protocol for type-safe solver integration
  - 8 unit tests + 4 integration tests validating convergence, mass conservation, analytical correctness

### Removed

- Dead deprecated shim modules: `grid_operators.py`, `tensor_operators.py`, `differential_utils.py` (-484 lines, PR #828)

### Changed

- `tensor_calculus.py` internalized — no longer re-exported from `utils.numerical`, no deprecation warning on import (PR #825)

## [0.17.12] - 2026-03-26

### Changed

- Complete `mfg_pde` → `mfgarchon` rename across archives and docs (follow-up to #821)
- README updated: fix API examples, remove stale version tag, fix tutorial links (PR #822)
- CITATION.cff: general description, no specific solver list (PR #822)
- Deprecation guide updated for v0.17.11 changes (PR #826)

### Fixed

- `SpatialCoordinates`/`TemporalCoordinates` deprecation warnings during test collection (PR #823)
- Stale `DOMAIN_2D`/`DOMAIN_3D` references in docs (PR #823)

### Removed

- Orphan `test_issue_557_fix.py` from project root
- 3 stale `.venv` editable install artifacts from old `mfg_pde` package

### Infrastructure

- Rename `_GmshMeshBase` → `_MeshGeneratorBase` for backend-agnostic naming (PR #824)
- Bump `actions/upload-artifact` 6→7, `actions/download-artifact` 7→8, `docker/setup-buildx-action` 3→4 (#815-817)

## [0.17.7] - 2026-02-06

### Fixed

- **Thread-safe global singletons** (Issue #759)
  - Added `threading.Lock()` with double-check locking to 4 global managers:
    - `plugin_system.get_plugin_manager()`
    - `workflow.get_workflow_manager()`
    - `network_backend.get_backend_manager()`
    - `general_mfg_factory.get_general_factory()`
  - Prevents race conditions in multi-threaded environments

- **Visualization type annotations** (Issue #758)
  - Removed 41 `type: ignore[assignment]` suppressions
  - Used proper `Any` typing for optional dependency fallbacks

- **Import patterns** (Issues #756, #757)
  - Replaced wildcard imports with explicit imports in `acceleration/`
  - Removed `sys.path` manipulation anti-pattern in solver modules

### Changed

- **Test suite cleanup** (Issue #761)
  - Reduced unconditional skips from 24 to 15 (37% reduction)
  - Fixed ghost buffer tests (incorrect assertions)
  - Deleted obsolete tests for deprecated patterns
  - Created tracking issues for remaining skips (#762, #763)

- **Deprecation timelines standardized** to v1.0.0 for all deprecated APIs

### Removed

- Deleted `tests/integration/test_coupled_hjb_fp_2d.py` (tested deprecated inheritance pattern)

## [0.17.6] - 2026-02-06

### Changed

- **Renamed `u_final` to `u_terminal`** (Issue #670, PR #755)
  - All APIs now use `u_terminal` for HJB terminal condition (MFG literature standard)
  - Deprecated `u_final` parameter in `MFGComponents`, redirects to `u_terminal`
  - Deprecated `get_u_final()`, `get_final_u()` methods, redirect to `get_u_terminal()`
  - Deprecated `validate_u_final()`, redirects to `validate_u_terminal()`
  - **Migration**: Replace `u_final=` with `u_terminal=` in all code

- **Unified `volatility_field` API** (Issue #717, PR #755)
  - Single `volatility_field` parameter handles all volatility specifications:
    - Scalar `σ` → isotropic diffusion `D = σ²/2`
    - Diagonal `[σ₀, σ₁, ...]` → anisotropic `D = diag(σᵢ²)/2`
    - Matrix `Σ (d×d)` → tensor diffusion `D = ΣΣᵀ/2`
    - Spatially varying `Σ(x)` → `D(x) = Σ(x)Σ(x)ᵀ/2`
    - Callable `σ(t,x,m)` or `Σ(t,x,m)` → state-dependent
  - Auto-detection by input shape (no separate parameters needed)

### Deprecated

- **`diffusion_field` parameter** → Use `volatility_field` instead
- **`tensor_diffusion_field` parameter** → Use `volatility_field` with `(d,d)` array
- **`volatility_matrix` parameter** → Use `volatility_field` with `(d,d)` array

### Documentation

- Updated `docs/NAMING_CONVENTIONS.md` with volatility vs diffusion terminology
- Added SDE-PDE relationship: `dX = μdt + σdW` → `∂ₜm = -∇·(μm) + DΔm` where `D = σ²/2`

## [0.17.5] - 2026-02-06

### Added

- **Adaptive Picard damping** (Issue #583, PR #745)
  - `adapt_damping()` function detects error oscillation and dynamically reduces damping
  - Opt-in via `FixedPointIterator(adaptive_damping=True)` (default off, backward compatible)
  - Independent U/M adaptation with cautious recovery toward initial damping
  - Damping history recorded in `SolverResult.metadata["adaptive_damping"]`
  - Gradient clipping warning now directs users to adaptive damping as primary fix

### Removed

- **`bc_mode` parameter from `HJBFDMSolver`** (Issue #703, #625)
  - Removed deprecated `bc_mode` parameter from `__init__` signature
  - Removed adjoint-consistent BC logic block from `solve_hjb_system`
  - **Migration**: Use `AdjointConsistentProvider` in `BCSegment.value` instead
  - Callers passing `bc_mode=` will now get `TypeError`

### Fixed

- **Mass Conservation in FP FDM Solver** (Issue #615)
  - Fixed catastrophic mass conservation failure (99.4% error → 2.3%)
  - Changed default advection scheme from `gradient_upwind` to `divergence_upwind`
  - Removed confusing `conservative: bool` parameter

## [0.17.4] - 2026-02-06

**Validation Initiative Release: Comprehensive Input Validation (Issue #685)**

### Added

- **Callable signature detection and adaptation** (Issue #684, PR #738)
  - New `adapt_ic_callable()` in `mfgarchon/utils/callable_adapter.py`
  - Auto-detects and wraps IC/BC callables: `f(x)` scalar, `f(x)` array, `f(x,t)`, `f(t,x)`, `f(x,y)`, `f(x,y,z)`
  - Zero-overhead passthrough for the common `f(x_scalar)` case
  - Detailed error messages listing all attempted calling conventions on failure
  - Expanded-coordinate signatures `f(x,y)` emit `DeprecationWarning`
- **Custom function validation** (Issue #686, PR #733)
  - `validate_hamiltonian()`, `validate_drift()`, `validate_running_cost()` in validation module
  - Probing-based signature detection for Hamiltonian, drift, running cost functions
  - Wired into `MFGProblem._initialize_functions()`
- **Array/field validation** (Issue #687, PR #735)
  - `validate_array_dtype()`, `validate_array_shape()`, `validate_field_dimension()`
  - Shape and dtype validation for solver arrays wired into MFGProblem
- **Runtime safety validation** (Issue #688, PR #736)
  - `check_finite()`, `check_bounds()`, `validate_solver_output()`
  - NaN/Inf detection wired into `FixedPointIterator`
- **IC/BC validation wiring** (Issue #681, PR #728)
  - `validate_components()` checks m_initial/u_final at problem construction
  - NDArray and callable IC/BC validated against geometry shape
- **Newton-to-Value-Iteration adaptive fallback** (Issue #669, PR #727)
  - HJB solver automatically falls back from Newton to value iteration on divergence

### Fixed

- **Backend device selection tests on Apple Silicon** (PR #737)
  - Fixed MPS backend detection tests that failed on Apple Silicon

### Changed

- **Validation module fully wired** into `MFGProblem._initialize_functions()` pipeline

## [0.17.2] - 2026-01-18

**Maintenance Release: Legacy Parameter Deprecation + Codebase Cleanup**

This release completes two important maintenance priorities:
1. **Legacy parameter deprecation** (Issue #544) - Deprecates old MFGProblem parameters, migrates all internal code to modern Geometry API
2. **Solver mixin cleanup** (Issue #545) - Removes dead code from completed refactoring

Both changes are 100% backward compatible with clear migration paths for users.

### Added

- **DeprecationWarning for Legacy Parameters** (Issue #544, Phase 1) 🎯
  - Warns users when using deprecated parameters: `Nx`, `xmin`, `xmax`, `Lx`, `spatial_bounds`, `spatial_discretization`
  - Clear migration instructions in warning message pointing to `docs/migration/LEGACY_PARAMETERS.md`
  - Respects `suppress_warnings=True` flag for gradual migration
  - **Timeline**: 6-12 month deprecation period before v1.0.0 removal

- **Comprehensive Migration Guide** (Issue #544):
  - `docs/migration/LEGACY_PARAMETERS.md` - 180-line guide with 5 common patterns
  - Before/after examples for each migration pattern
  - Nx → Nx_points conversion explained (Nx=100 intervals → Nx_points=[101] grid points)
  - Troubleshooting section for common issues

- **Documentation** (Issue #544, #545):
  - `docs/development/PRIORITY_8_PHASE_2_STATUS.md` - Complete deprecation plan (112 lines)
  - Updated `docs/development/PRIORITY_LIST_2026-01.md` - Priority 7 & 8 marked complete
  - Updated `docs/development/NEXT_STEPS_2026-01-18.md` - Next development priorities

### Changed

- **All Tests Migrated to Geometry API** (Issue #544, Phase 2) 🎯
  - Migrated 7 test files with 23 MFGProblem/StochasticMFGProblem calls
  - Integration tests: test_lq_common_noise_analytical.py, test_mass_conservation_1d*.py, test_particle_gpu_pipeline.py, etc.
  - Unit tests: test_common_noise_solver.py (12 calls)
  - Fixed SimpleMFGProblem1D mock for Geometry API compatibility
  - **Test results**: 79 + 23 + 12 passing, zero regressions

- **All Examples Verified** (Issue #544):
  - All files in `examples/` already use modern Geometry API
  - Zero migration needed (modern API adopted early)

### Deprecated

- **MFGProblem legacy parameters** (Issue #544) - **DEPRECATED, will be removed in v1.0.0**
  - `Nx`, `xmin`, `xmax`, `Lx` - Use `geometry=TensorProductGrid(...)` instead
  - `spatial_bounds`, `spatial_discretization` - Use `geometry=TensorProductGrid(...)` instead
  - DeprecationWarning provides migration guidance
  - See `docs/migration/LEGACY_PARAMETERS.md` for complete migration guide

### Removed

- **Dead Code Cleanup** (Issue #545) 🎯
  - Deleted `hjb_gfdm_monotonicity.py` (28KB) - MonotonicityMixin no longer used
  - Updated 5 outdated comments in hjb_gfdm.py referencing removed mixin
  - Verified: All 11 solvers use composition or simple inheritance (zero mixins)

### Fixed

- Documentation consistency in solver architecture references

## [0.17.1] - 2026-01-17

**Feature Release: Adjoint-Consistent Boundary Conditions + Three-Mode Solving API**

This release adds two major features:
1. **Adjoint-consistent boundary conditions** for HJB solver (Issue #574) - fixes equilibrium inconsistency at reflecting boundaries
2. **Three-mode solving API** (Issue #580) - prevents non-dual solver pairings

Both features include comprehensive documentation, validated testing, and are 100% backward compatible.

### Added

- **Three-Mode Solving API** (Issue #580, PR #585) 🎯
  - **Safe Mode**: `problem.solve(scheme=NumericalScheme.FDM_UPWIND)` - Guaranteed dual pairing
  - **Expert Mode**: `problem.solve(hjb_solver=hjb, fp_solver=fp)` - Manual control with validation
  - **Auto Mode**: `problem.solve()` - Intelligent defaults (backward compatible)
  - Prevents non-dual solver pairings that break Nash equilibrium convergence
  - Educational warnings guide users toward correct pairings
  - 121 tests validate correctness, 100% backward compatible

- **New Types** (Issue #580):
  - `NumericalScheme` enum: User-facing scheme selection (FDM_UPWIND, FDM_CENTERED, SL_LINEAR, SL_CUBIC, GFDM)
  - `SchemeFamily` enum: Internal classification (FDM, SL, FVM, GFDM, PINN, GENERIC)
  - `DualityStatus` enum: Validation status (DISCRETE_DUAL, CONTINUOUS_DUAL, NOT_DUAL, VALIDATION_SKIPPED)
  - `DualityValidationResult` dataclass: Rich validation result object

- **New Utilities** (Issue #580):
  - `check_solver_duality()`: Validates HJB-FP adjoint relationship
  - `create_paired_solvers()`: Factory for validated solver pairs with config threading
  - `get_recommended_scheme()`: Intelligent scheme selection (Phase 3 TODO - currently returns FDM_UPWIND)

- **New Examples** (Issue #580):
  - `examples/basic/three_mode_api_demo.py`: Comprehensive three-mode demonstration (246 lines)

- **New Documentation** (Issue #580):
  - `docs/development/issue_580_adjoint_pairing_implementation.md`: Technical guide (578 lines)
  - `docs/user/three_mode_api_migration_guide.md`: User migration guide (448 lines)

- **Adjoint-Consistent Boundary Conditions** (Issue #574, PR #588) 🎯
  - **`bc_mode` parameter** in `HJBFDMSolver`: `"standard"` | `"adjoint_consistent"`
  - Fixes equilibrium inconsistency at reflecting boundaries when stall points occur at domain boundaries
  - Mathematical formula: `∂U/∂n = -σ²/2 · ∂ln(m)/∂n` (Robin-type BC coupling HJB to FP density gradient)
  - **2.13x convergence improvement** validated in boundary stall configuration (703 → 330 max error)
  - Automatic BC computation from density gradient each Picard iteration
  - Negligible overhead (<0.1%), often reduces total iterations due to better consistency
  - 100% backward compatible (default `bc_mode="standard"` preserves classical Neumann BC)
  - 11 tests passing (smoke, integration, validation)

- **New Utilities** (Issue #574):
  - `compute_boundary_log_density_gradient()`: Computes ∂ln(m)/∂n at boundaries
  - `compute_coupled_hjb_bc_values()`: Converts to HJB BC values for adjoint-consistent mode

- **New Tutorial** (Issue #574):
  - `examples/tutorials/06_boundary_condition_coupling.py`: Comprehensive tutorial (266 lines)
  - Step-by-step comparison of standard vs adjoint-consistent BC modes
  - 4-panel visualization (density, value function, differences, convergence history)

- **New Documentation** (Issue #574):
  - `docs/development/issue_574_robin_bc_design.md`: Mathematical derivation and design (339 lines)
  - `docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md`: BC consistency solution section
  - `CLAUDE.md`: Boundary condition coupling patterns

### Changed

- **MFGProblem.solve()** (Issue #580):
  - Added `scheme` parameter for Safe Mode
  - Added `hjb_solver` and `fp_solver` parameters for Expert Mode
  - Mode detection and validation implemented
  - Fully backward compatible (existing code uses Auto Mode)

- **Solver Traits** (Issue #580):
  - All HJB and FP solvers now have `_scheme_family` class attribute
  - Used for refactoring-safe duality validation
  - Trait-based classification survives renames and inheritance changes

- **Renamed** `OneDimensionalAMRMesh` → `OneDimensionalAMRGrid` (Issue #466)
  - The class is a structured grid, not an unstructured mesh
  - Backward compatibility alias `OneDimensionalAMRMesh` remains (deprecated)
- **Renamed** `create_1d_amr_mesh()` → `create_1d_amr_grid()`
  - Backward compatibility alias remains (deprecated)

### Deprecated

- **`create_solver()`** (Issue #580) - Use three-mode API instead
  - Replacement: `problem.solve(scheme=...)` (Safe Mode) or `problem.solve(hjb_solver=..., fp_solver=...)` (Expert Mode)
  - Will be removed in v1.0.0
  - Deprecation warning guides migration with examples

- `OneDimensionalAMRMesh` - use `OneDimensionalAMRGrid` instead
- `create_1d_amr_mesh()` - use `create_1d_amr_grid()` instead

### Fixed

- **Critical BC Type Recognition Bug** (Issue #574, PR #588):
  - Fixed BC type recognition for `'no_flux'` string in HJB solver (`base_hjb.py`)
  - Previously, `neumann_bc()` objects were misinterpreted as periodic boundaries
  - **Impact**: Affects ALL Neumann BC usage throughout codebase (not limited to Issue #574)
  - Solver now correctly recognizes `'no_flux'`, `'neumann'`, `'dirichlet'`, `'periodic'`, and `'robin'` BC types

- **Scientific Correctness** (Issue #580):
  - Prevents accidental mixing of incompatible discretizations (e.g., FDM + GFDM)
  - Ensures L_FP = L_HJB^T relationship for Nash gap convergence
  - Type A (discrete dual) vs Type B (continuous dual) distinction enforced

- **HJB Boundary Equilibrium Consistency** (Issue #574):
  - Adjoint-consistent BC mode fixes 2.65x error increase at boundary stall configurations
  - Enables correct convergence to Boltzmann-Gibbs equilibrium

## [0.16.2] - 2025-12-12

**Patch Release: Grid Interpolator Batched Points Fix**

### Fixed

- **Grid-to-grid interpolation** now works correctly (Issue #444)
  - `TensorProductGrid.get_interpolator()` supports batched points (2D array of shape `(N, dim)`)
  - Single point evaluation remains backward compatible (returns `float`)
  - Projection between grids of different resolutions now works in 1D, 2D, and 3D

### Changed

- Fixed test assertions for 1D grids that used incorrect array shapes

## [0.16.1] - 2025-12-12

**Patch Release: Nx/Nx_points Naming Consistency**

This release introduces consistent naming for spatial and temporal discretization:
- `Nx` = number of intervals (consistent with `Nt`)
- `Nx_points` = number of grid points (`Nx + 1`)
- `Nt_points` = number of time points (`Nt + 1`)

### Added

- **`Nx` property** to `TensorProductGrid` - returns intervals per dimension
- **`Nx_points` property** to `TensorProductGrid` - returns grid points per dimension
- **`Nt_points` property** to `MFGProblem` - returns `Nt + 1`

### Changed

- `TensorProductGrid` constructor now accepts:
  - `Nx=` for intervals (like `Nt`)
  - `Nx_points=` for points
  - `num_points=` (deprecated, use `Nx_points`)
- Updated all codebase usages from `num_points=` to `Nx_points=`
- Updated `NAMING_CONVENTIONS.md` to document the new convention

### Deprecated

- **`num_points` parameter and property** in `TensorProductGrid`
  - Use `Nx_points` instead
  - Will be removed in v1.0.0

## [0.16.0] - 2025-12-11

**Feature Release: Geometry-First API Unification**

This release completes the geometry-first API unification for `MFGProblem`. The `geometry` attribute is now always non-None and serves as the single source of truth for all spatial information. Legacy attributes emit deprecation warnings but remain functional for backward compatibility.

### Changed

**Geometry-First API (Issue #435, PRs #436-#443)**

- **`MFGProblem.geometry` is now always non-None** after initialization
  - All four init paths (`_init_1d_legacy`, `_init_nd`, `_init_geometry`, `_init_network`) set geometry
  - Legacy parameters (`xmin`, `xmax`, `Nx`) automatically create `TensorProductGrid`
  - Network problems create appropriate `NetworkGeometry` subclass

- **Legacy attributes converted to computed properties**
  - `xmin`, `xmax`, `Lx`, `Nx`, `dx`, `xSpace`, `_grid` now derive from `self.geometry`
  - Properties emit `DeprecationWarning` when accessed
  - Setters allow backward-compatible assignment (stores to `_*_override`)
  - Internal code uses helper methods to avoid triggering warnings

- **Helper properties for geometry type dispatch**
  - `problem.is_cartesian` - True for `TensorProductGrid`
  - `problem.is_network` - True for `NetworkGeometry`
  - `problem.is_implicit` - True for implicit/SDF geometries

**OmegaConf Configuration (Issue #429, PRs #431-#432)**

- **Renamed OmegaConf classes to `*Schema` suffix** for clear naming convention
  - `MFGConfig` → `MFGSchema`
  - `SolverConfig` → `SolverSchema`
  - `HJBConfig` → `HJBSchema`
  - `FPConfig` → `FPSchema`
  - etc.

- **Added Pydantic-OmegaConf bridge utilities**
  - `bridge_to_pydantic()` - Generic adapter for OmegaConf → Pydantic conversion
  - `save_effective_config()` - Save resolved config for reproducibility
  - `load_effective_config()` - Load previously saved config

### Deprecated

- **Legacy attribute access** (`problem.xmin`, `problem.xmax`, `problem.Nx`, `problem.dx`, `problem.xSpace`)
  - Use `problem.geometry.get_bounds()`, `problem.geometry.num_spatial_points`, `problem.geometry.get_spatial_grid()` instead
  - Will be removed in v1.0.0

### Documentation

- Updated `GEOMETRY_FIRST_API_GUIDE.md` with migration table and v0.16.0 patterns
- Updated `DEPRECATION_MODERNIZATION_GUIDE.md` with Phase 7 completion status
- Updated `quickstart.md` to use geometry-first API in all examples
- Updated `migration.md` with v0.16.0 current API section

## [0.14.1] - 2025-12-06

### Changed

- **Rename `PerfectMazeGenerator` → `MazeGeometry`**: Better reflects role as geometry class for MFG problems

### Fixed

- **MazeGeometry now satisfies GeometryProtocol**: Can be used directly with `MFGProblem`
  - `generate()` returns `self` instead of `Grid`
  - `dimension=2` (spatially embedded) instead of `0` (abstract graph)
  - `geometry_type=MAZE` instead of `NETWORK`
- **MAZE/NETWORK geometry handlers in MFGProblem**: Properly extracts `spatial_bounds` from graph geometries

### Deprecated

- **`fdm_bc_1d` module**: Migrated examples to unified BC API (`periodic_bc()` from `mfgarchon.geometry.boundary`)

## [0.14.0] - 2025-12-01

### Removed - API Simplification (2025-11-23)

**BREAKING CHANGES**: Removed unnecessary API layers to enforce clean 2-level architecture (Factory vs Expert).

- **Removed `ExampleMFGProblem`** (deprecated since v0.12.0)
  - Migration: Use `MFGProblem` directly
  - Old: `problem = ExampleMFGProblem(dimension=2, X=X, t=t, g=g, H=H)`
  - New: `components = MFGComponents(hamiltonian_func=H, final_value_func=g); problem = MFGProblem(spatial_bounds=..., components=components)`

- **Removed `MFGProblemBuilder`**
  - Redundant builder pattern that added cognitive load without benefit
  - Migration: Use `MFGProblem` with `MFGComponents` directly
  - Old: `problem = MFGProblemBuilder().hamiltonian(H, dH).domain(0,10,100).build()`
  - New: `components = MFGComponents(hamiltonian_func=H, hamiltonian_dm_func=dH); problem = MFGProblem(xmin=0, xmax=10, Nx=100, components=components)`

- **Removed `create_mfg_problem()` convenience function**
  - Redundant wrapper around `MFGProblem` constructor
  - Migration: Use `MFGProblem` with `MFGComponents` directly
  - Old: `problem = create_mfg_problem(H, dH, xmin=0, xmax=10, Nx=100)`
  - New: `components = MFGComponents(hamiltonian_func=H, hamiltonian_dm_func=dH); problem = MFGProblem(xmin=0, xmax=10, Nx=100, components=components)`

**Rationale**: Enforces clear 2-level architecture:
- **Level 1 (Factory)**: Pre-configured problems via `create_*_problem()` functions
- **Level 2 (Expert)**: Direct `MFGProblem` + `MFGComponents` for full control

See `docs/development/API_SIMPLIFICATION_PROPOSAL.md` for details.

### Fixed - 2D/nD Support (2025-11-23)

- **Fixed Gap 1**: `H()` and `dH_dm()` now handle 2D/nD tuple indices `(i,j)` correctly
  - No longer crashes with `TypeError: 'NoneType' object is not subscriptable`
  - Proper multi-dimensional indexing via `np.ravel_multi_index()`

- **Fixed Gap 2**: `_setup_custom_final_value()` now works for nD problems
  - Uses `geometry.get_spatial_grid()` for nD instead of assuming 1D `xSpace`
  - Custom terminal conditions work in 2D and higher dimensions

## [0.12.1] - 2025-11-11

**Patch Release: API Consistency Improvements (Week 1)**

This patch release implements Week 1 quick wins from Issue #277 (API Consistency Audit), converting boolean pairs to enums and tuple returns to dataclasses for improved API clarity and type safety.

### Changed

**API Modernization (Issue #277 Phase 2 Week 1)**

- **HamiltonianJacobians dataclass** replaces tuple return in `MFGProblem.get_hjb_hamiltonian_jacobian_contrib()`
  - Self-documenting API: `jacobians.diagonal` instead of `result[0]`
  - Type-safe structured return with named fields
  - Updated HJB solver to use dataclass attributes

- **ProfilingMode enum** replaces `enable_profiling`/`verbose` boolean pair in `StrategySelector`
  - Three clear states: `DISABLED`, `SILENT`, `VERBOSE`
  - String support: `profiling_mode="verbose"`
  - Full backward compatibility with deprecation warnings

- **MeshVisualizationMode enum** replaces `show_edges`/`show_quality` boolean pair in `visualize_mesh()`
  - Four visualization modes: `SURFACE`, `WITH_EDGES`, `QUALITY`, `QUALITY_WITH_EDGES`
  - String shortcuts for quick usage
  - Applies to both `base_geometry.py` and `base.py`

### Fixed

- Docstring examples now use correct lowercase `.dx`/`.dt` convention (2 violations fixed in `problem_protocols.py`)

### Deprecated

- `StrategySelector(enable_profiling=..., verbose=...)` → Use `profiling_mode=ProfilingMode.SILENT` instead
- `visualize_mesh(show_edges=..., show_quality=...)` → Use `mode=MeshVisualizationMode.WITH_EDGES` instead
- Old APIs remain functional with deprecation warnings until v2.0.0

## [0.12.0] - 2025-11-11

**Feature Release: Advanced Projection Methods & API Modernization**

This release adds advanced particle-to-grid projection methods (GPU KDE, multigrid operators), completes the Dx/Dt→dx/dt migration, implements adaptive hybrid CPU/GPU strategies, and introduces enum-based configuration with full backward compatibility.

### Added

**Advanced Projection Operators (PRs #269, #270, Issue #265)**

- **Multi-dimensional GPU KDE** for particle-to-grid projection
  - GPU-accelerated kernel density estimation for 1D/2D/3D
  - Scott's rule and Silverman's rule for automatic bandwidth selection
  - Memory-efficient implementation for large particle systems
  - Fallback to CPU implementation when GPU unavailable
  - Significantly improves accuracy over histogram-based projection

- **Conservative restriction and prolongation operators** for multigrid methods
  - Conservative restriction: Fine → coarse grid with exact mass conservation
  - High-order prolongation: Bilinear/bicubic interpolation for coarse → fine
  - Supports 1D/2D/3D grids with arbitrary refinement ratios
  - Essential for multigrid acceleration of MFG solvers

**Adaptive Hybrid Strategies (PR #268, Issue #262)**

- **Intelligent CPU/GPU backend selection** for particle methods
  - Automatic threshold-based selection (10,000 particles)
  - Performance-optimized decision making based on problem size
  - Graceful handling of backend=None (automatic selection)
  - Reduces GPU overhead for small problems, leverages GPU for large ones

**GFDM Gradient Operators (PR #267, Issue #261)**

- **Full drift computation** in particle FP solver: `α = -∇U`
  - Implements GFDM-based gradient operator for arbitrary grids
  - Replaces zero-drift placeholder with proper physics
  - All 36 particle FP tests pass with realistic dynamics

**Modern Configuration with Enums (PR #283, Issue #277 Phase 2)**

- **AdaptiveTrainingMode** enum for PINN adaptive training strategies
  - Values: `BASIC`, `CURRICULUM`, `MULTISCALE`, `FULL_ADAPTIVE`
  - Replaces boolean triplet: `enable_curriculum`, `enable_multiscale`, `enable_refinement`
  - Backward compatible via `__post_init__` deprecation handling

- **NormalizationType** enum for PINN normalization methods
  - Values: `NONE`, `INPUT`, `LOSS`, `BOTH`
  - Replaces boolean pair: `normalize_input`, `normalize_loss`

- **VarianceReductionMethod** enum for DGM variance reduction
  - Values: `NONE`, `BASELINE`, `CONTROL_VARIATE`, `BOTH`
  - Replaces boolean pair: `use_baseline`, `use_control_variates`

**Enhanced Dependency Management (PR #279, Issue #278)**

- Improved error messages when optional dependencies missing
- Better diagnostics for installation issues
- User-friendly guidance for installing GPU backends

**Examples Reorganization (PR #275)**

- Elevated `tutorials/` to peer level with `basic/` and `advanced/`
- Hierarchical organization: `applications/`, `notebooks/`, `plugins/`
- Enhanced tutorial content (tutorials 04 and 05)
- Cleaner examples directory structure

### Deprecated

- **`Dt` attribute**: Use lowercase `dt` instead (Issue #245, PR #259, #274). Backward compatibility maintained via deprecated property that emits `DeprecationWarning`. Will be removed in v1.0.0.
- **`Dx` attribute**: Use lowercase `dx` instead (Issue #245, PR #259, #274). Backward compatibility maintained via deprecated property that emits `DeprecationWarning`. Will be removed in v1.0.0.
- **Boolean configuration parameters**: Replaced with enums (Issue #277, PR #283). Old parameters still work with deprecation warnings. Will be removed in v1.0.0.
- **GridBasedMFGProblem**: Removed. Use `MFGProblem` with `spatial_bounds` and `spatial_discretization` for nD problems.

### Changed

- **Primary time step attribute**: Changed from `Dt` to `dt` throughout codebase (46 files, ~102 references) following official naming conventions (`docs/NAMING_CONVENTIONS.md` lines 24, 262)
  - Core: `mfgarchon/core/mfg_problem.py`, `mfgarchon/types/problem_protocols.py`
  - Solvers: All HJB, FP, and coupling solvers updated
  - Utilities: `experiment_manager.py`, `hjb_policy_iteration.py`
  - Tests: 15 test files (59 references)
  - Examples: 5 example files (8 references)
  - Benchmarks: 3 benchmark files (4 references)

- **Primary spatial spacing attribute**: Changed from `Dx` to `dx` for 1D problems (same scope as above)

### Fixed

- Test collection errors (PR #266): Removed 2,280 lines of obsolete test code
- Flaky TD3 test (Issue #237): Made `test_soft_update_all_target_networks` deterministic
- Parameter migration system: Added missing `max_iterations → max_picard_iterations` mapping

### Documentation

- API violations audit (PR #282, Issue #277 Phase 1)
- Array-based notation standard (Issue #243 Phase 1)
- Comprehensive dual geometry example

### Issues Closed

9 issues closed: #278, #277 (Phase 2), #273, #265, #262, #261, #260, #259, #243 (Phase 1), #237

### Migration Guide

**For users**: Update your code to use lowercase attributes:
```python
# OLD (deprecated but works with warnings in v0.12.0)
dt = problem.Dt
dx = problem.Dx

# NEW (recommended)
dt = problem.dt
dx = problem.dx
```

**For enum configurations**:
```python
# OLD (deprecated but works with warnings)
config = AdaptiveTrainingConfig(
    enable_curriculum=True,
    enable_multiscale=True,
    enable_refinement=True
)

# NEW (recommended)
from mfgarchon.alg.neural.pinn_solvers.adaptive_training import AdaptiveTrainingMode
config = AdaptiveTrainingConfig(
    training_mode=AdaptiveTrainingMode.FULL_ADAPTIVE
)
```

**For developers**: The deprecated properties and parameters will be completely removed in v1.0.0.

## [0.11.0] - 2025-11-10

**Major Release: Dual Geometry Architecture**

This release introduces complete dual geometry support, enabling HJB and FP solvers to use different discretizations. This enables multi-resolution methods (4-15× speedup), FEM meshes with obstacles, hybrid particle-grid methods, and network-based agent models.

### Added

**Dual Geometry Infrastructure (PR #258, Issues #257 & #245 Phase 4)**

- **GeometryProjector** class (`mfgarchon/geometry/projection.py`, 706 lines)
  - Automatic projection method selection based on geometry types
  - `project_hjb_to_fp()`: Maps HJB solution values to FP geometry
  - `project_fp_to_hjb()`: Maps FP density values to HJB geometry
  - Supports grid-to-grid, grid-to-particles, particles-to-grid (KDE)
  - Vectorized implementations for 1D/2D/3D

- **ProjectionRegistry** pattern
  - Decorator-based registration: `@ProjectionRegistry.register(SourceType, TargetType, direction)`
  - Hierarchical fallback: exact type → category match → generic
  - O(N) custom projectors (not O(N²))
  - User-extensible for custom geometry types

- **MFGProblem Dual Geometry Integration** (`mfgarchon/core/mfg_problem.py`)
  - New parameters: `hjb_geometry` and `fp_geometry`
  - Automatic `GeometryProjector` creation when geometries differ
  - Unified attribute access: `problem.hjb_geometry`, `problem.fp_geometry`
  - Full backward compatibility with single `geometry` parameter

- **FEM Mesh Support**
  - Automatic Delaunay interpolation for `UnstructuredMesh` ↔ `CartesianGrid` (requires scipy)
  - Nearest neighbor fallback when scipy unavailable
  - Works with Mesh2D, Mesh3D, TriangularAMRMesh
  - Graceful extrapolation handling (fills NaN with nearest neighbor)

- **Vectorized Grid Interpolators**
  - `SimpleGrid1D.get_interpolator()`: Binary search-based 1D interpolation
  - `SimpleGrid2D/3D.get_interpolator()`: RegularGridInterpolator wrapper
  - Accepts array of query points for batch interpolation
  - Used by projection system for efficient grid-to-mesh operations

### Documentation

**Comprehensive Dual Geometry Documentation** (5,000+ lines)

- **Theory**: `docs/theory/geometry_projection_mathematical_formulation.md` (556 lines)
  - Mathematical formulation of all projection methods
  - Error analysis (interpolation, KDE, nearest neighbor)
  - Performance complexity analysis (O(N log N), O(N), etc.)
  - Pseudocode for all algorithms

- **Developer Guide**: `docs/development/GEOMETRY_PROJECTION_IMPLEMENTATION_GUIDE.md` (797 lines)
  - Adding new geometry types and projections
  - Registry pattern usage and best practices
  - Debugging tips and performance optimization
  - Complete code examples for custom projections

- **User Guide**: `docs/user_guide/dual_geometry_usage.md` (679 lines)
  - Complete workflow examples
  - Use cases: multi-resolution, hybrid methods, network agents
  - Performance tips and FAQ
  - Best practices for choosing projection methods

- **FEM Mesh Guide**: `docs/user_guide/fem_mesh_projection_guide.md` (352 lines)
  - FEM mesh support levels (basic + optimized)
  - Comparison of nearest neighbor vs Delaunay
  - Use cases: complex domains, obstacles, CAD import
  - Complete examples with performance tips

- **Migration Guide Update**: `docs/migration/unified_problem_migration.md`
  - Updated with dual geometry integration
  - Examples showing unified API + dual geometry together
  - Updated deprecation timeline with v0.11.0 milestone

- **Completion Summary**: `docs/development/ISSUE_257_COMPLETION_SUMMARY.md` (379 lines)
  - Complete implementation details for all 5 phases
  - Performance impact and testing results
  - Known limitations and future enhancements

### Examples

- **Multi-Resolution MFG**: `examples/basic/dual_geometry_multiresolution.py` (323 lines)
  - Fine HJB grid (100×100) + coarse FP grid (25×25)
  - Demonstrates 4× speedup with minimal accuracy loss
  - Complete visualization of projections
  - Performance comparison with unified geometry

- **FEM Mesh with Obstacles**: `examples/advanced/dual_geometry_fem_mesh.py` (330 lines)
  - Complex domain with circular obstacle using Gmsh
  - Automatic vs manual Delaunay registration
  - Accuracy comparison of projection methods
  - Working example with 495 vertices, 884 elements

### Testing

- **Projection Tests**: `tests/unit/geometry/test_geometry_projection.py` (439 lines)
  - 20 unit tests covering all projection methods
  - Shape verification, accuracy tests, conservation tests
  - Tests for 1D, 2D, 3D projections
  - Registry pattern tests

- **Integration Tests**: `tests/unit/test_core/test_mfg_problem.py` (+131 lines)
  - 7 new tests for dual geometry MFGProblem integration
  - Backward compatibility verification
  - Error handling and validation tests

### Use Cases Enabled

| Use Case | HJB Geometry | FP Geometry | Benefit |
|----------|--------------|-------------|---------|
| Multi-resolution | Fine grid | Coarse grid | 4-15× speedup, 46% memory savings |
| Complex domains | Regular grid | FEM mesh | Fast HJB, handles obstacles naturally |
| Hybrid methods | Grid | Particles | Grid-based value, particle density |
| Network agents | Grid | Network graph | Spatial value, network-constrained agents |

### Performance

- Multi-resolution: 4-15× speedup (depending on resolution ratio)
- Projection overhead: <1% of solve time
- Memory savings: Up to 46% for 4× resolution ratio
- Grid→Points: O(N) with RegularGridInterpolator
- Particles→Grid KDE: GPU-accelerated available (1D)

### Changed

- README updated with v0.11.0 features and dual geometry examples
- Citation updated to v0.11.0

### Backward Compatibility

- ✅ Fully backward compatible
- Existing code using single `geometry` parameter continues to work
- `hjb_geometry` and `fp_geometry` are optional
- No breaking changes

### Closes

- Issue #257: Dual geometry architecture (5 phases complete)
- Issue #245 Phase 4: Documentation for unified MFG problem

---

## [0.10.0] - 2025-11-05

**Major Release: Geometry-First API**

This release introduces the geometry-first API, a new recommended pattern for constructing MFG problems using geometry objects. This provides better type safety, clearer separation of concerns, and unified support for diverse geometry types.

### Added

**PR #244: Phase 2 Array Notation - Backward Compatible Implementation**
- Added `_normalize_to_array()` helper method in `MFGProblem` (`mfg_problem.py:79-122`)
  - Automatically converts scalar inputs to arrays
  - Emits `DeprecationWarning` for scalar usage
  - Points users to `MATHEMATICAL_NOTATION_STANDARD.md`
- Updated `MFGProblem.__init__` signature to accept both scalar and array inputs:
  - `Nx`: `int | list[int]` (deprecated scalar, standard array)
  - `xmin`, `xmax`: `float | list[float]` (deprecated scalar, standard array)
- Both scalar and array inputs produce identical results with 100% backward compatibility
- Migration path for Phase 3 (v1.0.0): Remove deprecated scalar API

**PR #247: GeometryProtocol Foundation**
- Created `GeometryProtocol` runtime-checkable Protocol (`mfgarchon/geometry/geometry_protocol.py`)
  - Minimal interface for all geometry objects
  - Four required properties: `dimension`, `geometry_type`, `num_spatial_points`, `get_spatial_grid()`
- Created `GeometryType` enum with 7 types:
  - `CARTESIAN_GRID`: Regular tensor product grids
  - `NETWORK`: Graph/network geometries
  - `MAZE`: Maze environments
  - `DOMAIN_2D`, `DOMAIN_3D`, `DOMAIN_1D`: Cartesian/unstructured meshes
  - `IMPLICIT`: Level sets and signed distance functions
  - `CUSTOM`: User-defined geometries
- Added helper functions:
  - `detect_geometry_type()`: Self-aware type detection via attribute inspection
  - `is_geometry_compatible()`: Compatibility checking
  - `validate_geometry()`: Validation with informative error messages
- Implemented GeometryProtocol for 6 core geometry classes:
  - `Domain1D`: 1D Cartesian grids with grid caching
  - `BaseGeometry`: Abstract base for Domain2D/Domain3D meshes
  - `TensorProductGrid`: Arbitrary-dimension structured grids
  - `NetworkGeometry`: Graph-based geometries (Grid/Random/ScaleFree networks)
  - `ImplicitDomain`: Meshfree domains via signed distance functions (`Hyperrectangle`, `Hypersphere`)
  - `Grid` (mazes): Maze-based geometries from PerfectMazeGenerator
- Comprehensive design documentation (`docs/development/UNIFIED_GEOMETRY_PARAMETER_DESIGN.md`, 844 lines)

**Geometry-First API Implementation**
- Updated `MFGProblem._init_geometry()` to accept any GeometryProtocol-compliant object (`mfg_problem.py:647-768`)
  - Automatic geometry type detection via `geometry.geometry_type` enum
  - Specialized handling for CARTESIAN_GRID, IMPLICIT, DOMAIN_2D/3D, MAZE, NETWORK types
  - Generic fallback for CUSTOM geometries
- Added deprecation warnings for manual grid construction (`mfg_problem.py:350-363, 430-450`)
  - Warns users to migrate to geometry-first API
  - Points to migration guide with code examples
  - 100% backward compatibility maintained
- Created `docs/migration/GEOMETRY_FIRST_API_GUIDE.md` (400+ lines)
  - Quick start examples for all geometry types
  - Migration strategy from old to new API
  - Performance considerations and FAQ
- Created `examples/basic/geometry_first_api_demo.py` (350+ lines)
  - Demonstrates 8 geometry patterns (TensorProductGrid, Domain1D, Hyperrectangle, Hypersphere, Maze, 4D, reuse, refinement)
  - All examples tested and working
- Fixed normalization bug for implicit geometries (`mfg_problem.py:1158-1172`)
  - Handles `None` spatial_bounds for SDF-based geometries
  - Uses uniform approximation when structured grid info unavailable

### Changed

**API Improvements**
- `MFGProblem` now accepts both scalar and array notation for spatial parameters
- `MFGProblem` now accepts geometry objects via `geometry=` parameter (NEW recommended API)
- Array notation is the standard for manual construction (following `MATHEMATICAL_NOTATION_STANDARD.md`)
- Scalar inputs and manual grid construction trigger deprecation warnings

**Code Quality**
- Unified geometry interface across all geometry types via GeometryProtocol
- Protocol-based design enables duck typing without explicit inheritance
- Self-aware geometry types for automatic type detection in MFGProblem
- Enhanced type safety and consistency across geometry module
- Separation of concerns: geometry construction vs. problem temporal/diffusion parameters

### Deprecated

**API Patterns** (will be restricted in v1.0.0, removed in v2.0.0)
- Manual grid construction in `MFGProblem` (passing `spatial_bounds`, `spatial_discretization`, `xmin`, `xmax`, `Nx`)
  - Use geometry-first API instead: create geometry object, pass to `MFGProblem(geometry=...)`
  - Deprecation warnings provide migration examples
  - See `docs/migration/GEOMETRY_FIRST_API_GUIDE.md` for complete guide
- Scalar `Nx`, `xmin`, `xmax` parameters (if still using manual construction)
  - Use arrays instead: `Nx=[100]`, `xmin=[-2.0]`, `xmax=[2.0]`
  - Warnings guide users to `MATHEMATICAL_NOTATION_STANDARD.md`

**Deprecation Timeline**:
- v0.10.x: Warnings emitted, old API fully functional
- v0.11.x - v0.99.x: Continued warnings
- v1.0.0: Manual construction requires explicit `allow_manual_construction=True` flag
- v2.0.0: Complete removal of manual construction

### Documentation

- Array-Based Notation Migration plan (`docs/development/ARRAY_BASED_NOTATION_MIGRATION.md`)
- Mathematical Notation Standard (`docs/development/MATHEMATICAL_NOTATION_STANDARD.md`)
- Unified Geometry Parameter Design (`docs/development/UNIFIED_GEOMETRY_PARAMETER_DESIGN.md`)
- Geometry-First API Guide (`docs/migration/GEOMETRY_FIRST_API_GUIDE.md`)

### Future Work (Planned for 0.10.x series)

**v0.10.1** (Planned):
- Add GeometryProtocol compliance to AMR classes (OneDimensionalAMRMesh, AdaptiveMesh, TriangularAMRMesh, TetrahedralAMRMesh)
- Enable AMR meshes to be used directly in `MFGProblem(geometry=amr_mesh)`

**v0.10.2** (Planned):
- Design and implement dimension-agnostic boundary condition system (`BoundaryConditionND`)
- Support for nD boundary conditions (d > 3) with per-axis BC specification

**v0.10.3** (Planned):
- Rename `BaseGeometry` → `MeshGeometry` for clarity (breaking change with deprecation)
- Update all documentation and examples to reflect renamed class

### Testing

- All 3300+ tests passing
- Array notation backward compatibility validated
- GeometryProtocol compliance verified for all implemented geometries

## [0.9.1] - 2025-11-04

### Added

**PR #242: GFDM Operators with Unified Smoothing Kernels**
- `mfgarchon/utils/numerical/smoothing_kernels.py` (807 lines)
  - Unified kernel implementations: Gaussian, Wendland, Cubic Spline, Quintic Spline, Cubic, Quartic
  - Parameterized Wendland kernels: `WendlandKernel(k=0,1,2,3)` for C^0, C^2, C^4, C^6 smoothness
  - Arbitrary dimension support with proper normalization
  - Factory pattern: `create_kernel(kernel_type, dimension)`
  - Derivative support for gradient-based methods
- `mfgarchon/utils/numerical/gfdm_operators.py` (1050 lines)
  - Weighted least squares gradient/Hessian reconstruction
  - Support for structured and unstructured grids
  - Boundary condition handling (Dirichlet, Neumann)
  - Anisotropic/directional derivative support
- Theory documentation with differential operators (gradient, divergence, Laplacian)
- Comprehensive test suite (502 lines, 54 tests)
- Advanced example demos for nD geometry and implicit geometry

**PR #239: Maze Refactoring**
- Moved maze generation from `alg/reinforcement/environments` to `geometry/mazes`
- Makes maze utilities accessible to all solver types (PDE, particle, neural, RL)
- Backward compatibility through re-exports
- 6 core files relocated: `maze_generator`, `hybrid_maze`, `voronoi_maze`, `maze_config`, `maze_utils`, `maze_postprocessing`

### Changed
- Updated solver integrations to use unified kernel API
- Consolidated 4 separate Wendland classes into single parameterized implementation
- Updated test imports to reference new maze location

### Documentation
- Added `docs/theory/smoothing_kernels_mathematical_formulation.md` with complete mathematical foundations
- Dimension-specific formulas for differential operators (1D, 2D, 3D)
- SPH and GFDM application notes
- Implementation details with code references

### Testing
- All 3300+ tests passing
- New GFDM operator tests validated against analytical solutions
- Kernel tests cover edge cases, normalization, and derivatives

## [0.9.0] - 2025-11-03

### Phase 3 Complete: Unified Architecture

Major architecture refactoring completing Phase 3.1 (MFGProblem), Phase 3.2 (SolverConfig), and Phase 3.3 (Factory Integration).

### Added

**Issue #216: Missing Utilities (Complete - All 4 Parts)**
- **Part 1: Particle Interpolation** (commit 84e6e6d)
  - `interpolate_grid_to_particles()` - Grid → Particles (1D/2D/3D)
  - `interpolate_particles_to_grid()` - Particles → Grid (RBF, KDE, nearest)
  - `estimate_kde_bandwidth()` - Automatic bandwidth selection
  - Saves ~220 lines per research project
- **Part 2: Signed Distance Functions** (commit 83f59f4)
  - Primitives: `sdf_sphere()`, `sdf_box()` for 1D/2D/3D/nD
  - CSG operations: `sdf_union()`, `sdf_intersection()`, `sdf_complement()`, `sdf_difference()`
  - Smooth blending: `sdf_smooth_union()`, `sdf_smooth_intersection()`
  - Gradient: `sdf_gradient()` using finite differences
  - Saves ~150 lines per research project
- **Part 3: QP Solver Caching** (already existed)
  - `QPCache` - Hash-based caching with LRU eviction
  - `QPSolver` - Unified solver with warm-starting
  - Multiple backends: OSQP, scipy SLSQP, scipy L-BFGS-B
  - Saves ~180 lines per project + 2-5× GFDM speedup
- **Part 4: Convergence Monitoring** (already existed)
  - `AdvancedConvergenceMonitor` - Plotting, stagnation detection
  - `AdaptiveConvergenceWrapper` - Adaptive convergence criteria
  - Saves ~60 lines per project
- **Total Impact**: ~610 lines saved per research project + performance improvements

**Phase 3.1: Unified Problem Class (PR #218)**
- Single `MFGProblem` class replacing 5+ specialized problem classes
- Flexible `MFGComponents` system for custom problem definitions
- Auto-detection of problem types (standard, network, variational, stochastic, highdim)
- `MFGProblemBuilder` for programmatic problem construction
- Full backward compatibility with deprecated specialized classes

**Phase 3.2: Unified Configuration System (PR #222)**
- New `SolverConfig` class unifying 3 competing config systems
- Three usage patterns:
  - YAML files for experiments and reproducibility
  - Builder API for programmatic configuration
  - Presets for common use cases
- Modular config components: `PicardConfig`, `HJBConfig`, `FPConfig`, `BackendConfig`, `LoggingConfig`
- Preset configurations: fast, accurate, research, production, domain-specific
- YAML I/O with validation
- Legacy config compatibility layer

**Phase 3.3: Factory Integration (PR #224)**
- Unified problem factories supporting all MFG types:
  - `create_mfg_problem()` - Main factory for any problem type
  - `create_standard_problem()` - Standard HJB-FP MFG
  - `create_network_problem()` - Network/Graph MFG
  - `create_variational_problem()` - Variational/Lagrangian MFG
  - `create_stochastic_problem()` - Stochastic MFG with common noise
  - `create_highdim_problem()` - High-dimensional MFG (d > 3)
  - `create_lq_problem()` - Linear-Quadratic MFG
  - `create_crowd_problem()` - Crowd dynamics MFG
- Updated `solve_mfg()` interface:
  - New `config` parameter accepting `SolverConfig` instances or preset names
  - Deprecated `method` parameter (still works with warning)
  - Automatic config resolution from strings
- Extended `MFGComponents` for all problem types (network, variational, stochastic, highdim)
- Dual-output factory support: unified MFGProblem (default) or legacy classes (deprecated)
- New examples: `factory_demo.py`, updated `solve_mfg_demo.py`
- Comprehensive documentation:
  - Phase 3.3 design documents (2,000+ lines)
  - Problem type taxonomy
  - Migration guides
  - Completion summary

### Changed

**API Improvements**
- Simplified problem creation with unified factories
- Consistent configuration across all solver types
- Three flexible configuration patterns (YAML, Builder, Presets)
- Clearer separation: problem (math) vs solver (algorithm)

**Code Quality**
- Reduced code duplication through unification
- Better type safety with modern Python typing (`@overload`)
- Improved documentation with comprehensive examples
- Cleaner package structure

### Deprecated

**Problem Classes** (to be removed in v2.0.0)
- `LQMFGProblem` → Use `create_lq_problem()` or `MFGProblem`
- `NetworkMFGProblem` → Use `create_network_problem()` or `MFGProblem`
- `VariationalMFGProblem` → Use `create_variational_problem()` or `MFGProblem`
- `StochasticMFGProblem` → Use `create_stochastic_problem()` or `MFGProblem`

**Config Functions** (to be removed in v2.0.0)
- `create_fast_config()` → Use `presets.fast_solver()`
- `create_accurate_config()` → Use `presets.accurate_solver()`
- `create_research_config()` → Use `presets.research_solver()`
- Old `MFGSolverConfig` → Use new `SolverConfig`

**API Parameters** (to be removed in v2.0.0)
- `solve_mfg(method=...)` → Use `solve_mfg(config=...)`

### Migration Guide

**Old API**:
```python
from mfgarchon.problems import LQMFGProblem
from mfgarchon.config import create_accurate_config
from mfgarchon import solve_mfg

problem = LQMFGProblem(...)
result = solve_mfg(problem, method="accurate")
```

**New API** (Recommended):
```python
from mfgarchon.factory import create_lq_problem
from mfgarchon import solve_mfg

problem = create_lq_problem(...)
result = solve_mfg(problem, config="accurate")
```

### Documentation

- Added comprehensive Phase 3 design documents
- Created migration guides for Phase 3.2 and 3.3
- Updated examples with new unified API
- Added problem type taxonomy
- Created Phase 3 completion summary
- **New User Guides**:
  - `docs/user_guides/particle_interpolation.md` - Complete particle interpolation reference
  - `docs/user_guides/sdf_utilities.md` - Complete SDF utilities reference
  - `docs/migration/PHASE_3_MIGRATION_GUIDE.md` - Phase 3 migration guide
  - `docs/tutorials/01_getting_started.md` - Beginner tutorial
  - `docs/tutorials/02_configuration_patterns.md` - Configuration patterns tutorial

### Technical Details

**Total Changes**:
- ~8,000 lines added/modified
- 21 files changed
- 3 major PRs (#218, #222, #224)
- Full backward compatibility maintained

**Key Benefits**:
- Simpler, more consistent API
- Three flexible configuration patterns
- Better documentation and examples
- Easier to maintain and extend
- Better type safety
- Single source of truth

---

## [0.8.1] - 2025-10-08

### Fixed
- Full nD FP Solver implementation
- Semi-Lagrangian 2D solver
- Bug #8 resolution

---

## Historical Versions

Previous versions (< 0.8.1) were tracked in git history but not formally documented in CHANGELOG.

For detailed historical changes, see:
- Git commit history
- Closed issues and PRs
- Development documentation in `docs/development/`

---

**Note**: Starting with v0.9.0, all changes are documented in this CHANGELOG following semantic versioning and Keep a Changelog standards.
