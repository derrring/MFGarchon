"""
Precomputed Monotone Stencils for GFDM Solvers.

This module provides precomputation of M-matrix compliant GFDM stencils
for boundary points. By precomputing monotone weights at initialization,
we eliminate the need for runtime QP optimization while guaranteeing
the M-matrix property for numerical stability.

Mathematical Background:
    The M-matrix property for Laplacian stencils requires:
    - w_ii <= 0 (center weight non-positive)
    - w_ij >= 0 for j != i (off-diagonal non-negative)
    - sum_j w_ij = 0 (consistency/conservation)

    Standard GFDM may violate these conditions at boundary points due to
    one-sided neighbor distributions. This module projects unconstrained
    weights onto the M-matrix feasible set via QP:

        min_w ||w - w_unconstrained||^2
        s.t.  w_j >= 0 for j != center
              sum_j w_j = 0

Design Reference:
    See docs/PRECOMPUTED_MONOTONE_STENCILS_DESIGN.md for full mathematical
    derivation and performance analysis.

Author: MFGarchon Development Team
Created: 2026-01-30
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

from mfgarchon.alg.numerical.gfdm_components.joint_socp import (
    build_taylor_matrix_1d,
    build_taylor_matrix_2d,
    wendland_stencil_weights,
)

# Optional: OSQP for fast QP solving
try:
    import osqp

    import scipy.sparse as sp

    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

# Fallback: scipy for QP
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class MonotoneStencilData:
    """Data for a single precomputed monotone stencil."""

    weights: np.ndarray  # Monotone Laplacian weights
    neighbor_indices: np.ndarray  # Indices of neighbors
    center_in_neighbors: int | None  # Index of center point in neighbor list
    was_monotonized: bool  # True if QP was needed


class PrecomputedMonotoneStencils:
    """
    Cache for precomputed M-matrix compliant GFDM stencils.

    Precomputes monotone Laplacian weights for boundary points at initialization.
    At runtime, these weights can be used directly without QP optimization.

    Parameters
    ----------
    is_boundary : np.ndarray
        Boolean array indicating boundary points
    neighborhoods : dict
        Post-filter stencil dict (typically ``HJBGFDMSolver.neighborhoods`` built
        by ``NeighborhoodBuilder``). Stencils are built on
        ``neighborhoods[i]["indices"]``. Required for correctness on irregular
        clouds where ``adaptive_neighborhoods=True`` enlarges boundary stencils
        (Issue #1102 dual-source bug class — legacy fallback removed in v0.25.0).
    points : np.ndarray
        Collocation points, shape (n_total, dimension).
    delta : float
        Wendland kernel support radius. Sets the LSQ weighting used to compute
        unconstrained Laplacian weights on the supplied stencil.
    tolerance : float
        Numerical tolerance for M-matrix check (default: 1e-6)

    Attributes
    ----------
    stencils : dict[int, MonotoneStencilData]
        Precomputed stencil data for each boundary point
    stats : dict
        Precomputation statistics

    Example
    -------
    >>> stencils = PrecomputedMonotoneStencils(is_boundary, neighborhoods, points, delta)
    >>> lap_weights, neighbors = stencils.get_laplacian_weights(boundary_point_idx)
    """

    def __init__(
        self,
        is_boundary: np.ndarray,
        neighborhoods: dict,
        points: np.ndarray,
        delta: float,
        tolerance: float = 1e-6,
    ):
        # Single source of truth: stencils are always computed against the
        # explicitly-supplied post-filter neighborhoods (after visibility
        # filter, ghost nodes, adaptive δ-enlargement). The legacy fallback
        # to `op.get_derivative_weights()` (pre-adaptive op.neighborhoods)
        # was removed in v0.25.0 — it silently produced wrong results when
        # adaptive_neighborhoods modified the runtime stencils (Issue #1102
        # dual-source bug class).
        self._is_boundary = np.asarray(is_boundary)
        self._tolerance = tolerance
        self._neighborhoods = neighborhoods
        self._points = np.asarray(points)
        self._delta = float(delta)
        self._dimension = self._points.shape[1] if self._points.ndim == 2 else 1
        if self._dimension not in (1, 2):
            raise ValueError(
                f"PrecomputedMonotoneStencils currently supports 1D or 2D, got dimension {self._dimension}"
            )

        self.stencils: dict[int, MonotoneStencilData] = {}
        self.stats = {
            "n_boundary": 0,
            "n_monotonized": 0,
            "n_already_monotone": 0,
            "time_ms": 0.0,
            "avg_qp_time_ms": 0.0,
            "max_qp_time_ms": 0.0,
        }

        self._precompute()

    def _precompute(self) -> None:
        """Precompute monotone stencils for all boundary points."""
        t0 = time.time()
        qp_times = []

        boundary_indices = np.where(self._is_boundary)[0]
        self.stats["n_boundary"] = len(boundary_indices)

        for i in boundary_indices:
            i = int(i)
            stencil = self._compute_unconstrained_from_neighborhoods(i)
            if stencil is None:
                continue
            lap_weights, neighbor_indices = stencil

            # Find center point in neighbor list
            center_in_neighbors = self._find_center_in_neighbors(i, neighbor_indices)

            # Check M-matrix violation
            if self._violates_m_matrix(lap_weights, center_in_neighbors):
                # Solve QP to get monotone weights
                t_qp = time.time()
                w_monotone = self._solve_monotone_qp(lap_weights, center_in_neighbors)
                qp_times.append((time.time() - t_qp) * 1000)

                self.stencils[i] = MonotoneStencilData(
                    weights=w_monotone,
                    neighbor_indices=neighbor_indices,
                    center_in_neighbors=center_in_neighbors,
                    was_monotonized=True,
                )
                self.stats["n_monotonized"] += 1
            else:
                # Already monotone, store as-is
                self.stencils[i] = MonotoneStencilData(
                    weights=lap_weights,
                    neighbor_indices=neighbor_indices,
                    center_in_neighbors=center_in_neighbors,
                    was_monotonized=False,
                )
                self.stats["n_already_monotone"] += 1

        # Update statistics
        self.stats["time_ms"] = (time.time() - t0) * 1000
        if qp_times:
            self.stats["avg_qp_time_ms"] = np.mean(qp_times)
            self.stats["max_qp_time_ms"] = np.max(qp_times)

    def _find_center_in_neighbors(self, point_idx: int, neighbor_indices: np.ndarray) -> int | None:
        """Find the index of center point within the neighbor list."""
        for j, idx in enumerate(neighbor_indices):
            if idx == point_idx:
                return j
        return None

    def _compute_unconstrained_from_neighborhoods(self, point_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute unconstrained Wendland-LSQ Laplacian weights on the
        post-adaptive stencil at ``point_idx``.

        Mirrors the Wendland-LSQ fast-path in
        :func:`mfgarchon.alg.numerical.gfdm_components.joint_socp.solve_joint_socp_at_stencil`
        (lap-only): builds Taylor matrix on offsets from the post-filter
        neighborhood, then ``L = WA @ solve(A^T W A, e_lap)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            ``(lap_weights, neighbor_indices)`` for the post-adaptive stencil,
            or None if the neighborhood is missing or the LSQ is singular.
        """
        # Caller invariants (set when neighborhoods= is provided to __init__).
        assert self._neighborhoods is not None
        assert self._points is not None
        assert self._delta is not None
        nh = self._neighborhoods.get(point_idx)
        if nh is None:
            return None
        nbr = np.asarray(nh["indices"])
        if len(nbr) < 3:  # need at least k=3 Taylor cols in 1D, 6 in 2D
            return None

        offsets = self._points[nbr] - self._points[point_idx]
        if self._dimension == 1:
            offsets_1d = offsets.reshape(-1)
            A, _ = build_taylor_matrix_1d(offsets_1d)
            w_neighbor = wendland_stencil_weights(offsets_1d, self._delta)
            e_lap = np.array([0.0, 0.0, 1.0])
        else:
            A, _ = build_taylor_matrix_2d(offsets)
            w_neighbor = wendland_stencil_weights(offsets, self._delta)
            e_lap = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

        W_diag = np.diag(w_neighbor)
        ATA = A.T @ W_diag @ A
        try:
            # Issue #1066: solve, not inv — squares condition number.
            coeffs = np.linalg.solve(ATA, e_lap)
        except np.linalg.LinAlgError:
            return None
        lap_weights = (W_diag @ A) @ coeffs  # shape (n,)
        return lap_weights, nbr

    def _violates_m_matrix(self, weights: np.ndarray, center_idx: int | None) -> bool:
        """Check if Laplacian weights violate M-matrix property."""
        if center_idx is not None:
            off_diagonal = np.delete(weights, center_idx)
        else:
            off_diagonal = weights

        # M-matrix: off-diagonal must be non-negative
        # (center can be negative, sum should be zero)
        return np.any(off_diagonal < -self._tolerance)

    def _solve_monotone_qp(self, w_unconstrained: np.ndarray, center_idx: int | None) -> np.ndarray:
        """
        Solve QP to project weights onto M-matrix feasible set.

        min_w ||w - w_unconstrained||^2
        s.t.  w_j >= 0 for j != center
              sum_j w_j = 0
        """
        if OSQP_AVAILABLE:
            return self._solve_qp_osqp(w_unconstrained, center_idx)
        elif SCIPY_AVAILABLE:
            return self._solve_qp_scipy(w_unconstrained, center_idx)
        else:
            warnings.warn(
                "Neither OSQP nor scipy available for QP solving. Returning unconstrained weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            return w_unconstrained

    def _solve_qp_osqp(self, w_unc: np.ndarray, center_idx: int | None) -> np.ndarray:
        """Solve monotone weight QP using OSQP (~1ms)."""
        n = len(w_unc)

        # QP: min (1/2) w'Pw + q'w where P=I, q=-w_unc
        P = sp.eye(n, format="csc")
        q = -w_unc

        # Build constraint matrix
        constraint_rows = []
        l_list = []
        u_list = []

        # Inequality: w_j >= 0 for j != center
        for j in range(n):
            if center_idx is not None and j == center_idx:
                continue  # Center can be negative
            row = np.zeros(n)
            row[j] = 1.0
            constraint_rows.append(row)
            l_list.append(0.0)
            u_list.append(np.inf)

        # Equality: sum(w) = 0
        constraint_rows.append(np.ones(n))
        l_list.append(0.0)
        u_list.append(0.0)

        A = sp.csc_matrix(np.vstack(constraint_rows))
        lower = np.array(l_list)
        upper = np.array(u_list)

        # Solve
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=lower, u=upper, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        result = prob.solve()

        if result.info.status == "solved":
            return result.x
        else:
            warnings.warn(
                f"OSQP failed: {result.info.status}. Using unconstrained weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            return w_unc

    def _solve_qp_scipy(self, w_unc: np.ndarray, center_idx: int | None) -> np.ndarray:
        """Solve monotone weight QP using scipy SLSQP (fallback)."""
        n = len(w_unc)

        def objective(w: np.ndarray) -> float:
            return 0.5 * np.sum((w - w_unc) ** 2)

        def gradient(w: np.ndarray) -> np.ndarray:
            return w - w_unc

        # Constraints
        constraints = []

        # Inequality: w_j >= 0 for j != center
        for j in range(n):
            if center_idx is not None and j == center_idx:
                continue
            constraints.append({"type": "ineq", "fun": lambda w, idx=j: w[idx]})

        # Equality: sum(w) = 0
        constraints.append({"type": "eq", "fun": lambda w: np.sum(w)})

        result = minimize(
            objective,
            w_unc,
            jac=gradient,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-10},
        )

        return result.x

    def get_laplacian_weights(self, point_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Get precomputed monotone Laplacian weights for a point.

        Parameters
        ----------
        point_idx : int
            Index of the point

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            (weights, neighbor_indices) if available, None otherwise
        """
        if point_idx not in self.stencils:
            return None

        stencil = self.stencils[point_idx]
        return stencil.weights, stencil.neighbor_indices

    def has_stencil(self, point_idx: int) -> bool:
        """Check if precomputed stencil exists for a point."""
        return point_idx in self.stencils

    def is_monotonized(self, point_idx: int) -> bool:
        """Check if stencil required QP monotonization."""
        if point_idx not in self.stencils:
            return False
        return self.stencils[point_idx].was_monotonized

    def print_statistics(self) -> None:
        """Print precomputation statistics."""
        print("\n" + "=" * 60)
        print("PRECOMPUTED MONOTONE STENCILS STATISTICS")
        print("=" * 60)
        print(f"  Boundary points:     {self.stats['n_boundary']}")
        print(f"  Already monotone:    {self.stats['n_already_monotone']}")
        print(f"  Monotonized via QP:  {self.stats['n_monotonized']}")
        print(f"  Total precompute:    {self.stats['time_ms']:.1f} ms")
        if self.stats["n_monotonized"] > 0:
            print(f"  Avg QP time:         {self.stats['avg_qp_time_ms']:.2f} ms")
            print(f"  Max QP time:         {self.stats['max_qp_time_ms']:.2f} ms")
        print("=" * 60 + "\n")


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "PrecomputedMonotoneStencils",
    "MonotoneStencilData",
]


if __name__ == "__main__":
    """Quick smoke test."""
    print("Testing PrecomputedMonotoneStencils...")

    # This is a simple test that doesn't require full GFDM setup
    # Real integration test requires TaylorOperator

    print("  Module loads correctly")
    print(f"  OSQP available: {OSQP_AVAILABLE}")
    print(f"  scipy available: {SCIPY_AVAILABLE}")

    # Test QP solver directly
    if OSQP_AVAILABLE or SCIPY_AVAILABLE:
        # Create test instance with mock data
        w_unc = np.array([-5.0, 2.0, 3.0, -1.0, 1.0])  # sum = 0, but has negative off-diag
        center_idx = 0  # First element is center

        stencils = PrecomputedMonotoneStencils.__new__(PrecomputedMonotoneStencils)
        stencils._tolerance = 1e-6

        w_mono = stencils._solve_monotone_qp(w_unc, center_idx)

        # Check M-matrix
        off_diag = np.delete(w_mono, center_idx)
        assert np.all(off_diag >= -1e-6), "Off-diagonal should be non-negative"
        assert abs(np.sum(w_mono)) < 1e-6, "Sum should be zero"

        print(f"  QP test: w_unc = {w_unc}")
        print(f"           w_mono = {w_mono}")
        print(f"           sum(w_mono) = {np.sum(w_mono):.2e}")
        print(f"           min(off_diag) = {off_diag.min():.2e}")

    print("Smoke tests passed!")
