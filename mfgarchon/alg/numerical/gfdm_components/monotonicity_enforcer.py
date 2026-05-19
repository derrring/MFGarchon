"""
Monotonicity Enforcement Component for GFDM Solvers.

This component provides quadratic programming and monotonicity constraint
functionality for GFDM solvers that require monotone finite difference schemes.

Extracted from MonotonicityMixin as part of Issue #545 (mixin refactoring).
Uses composition pattern instead of inheritance for better testability and reusability.

Mathematical Background:
    For a monotone finite difference scheme, the discretization matrix must
    have the M-matrix property:
    - Diagonal elements: a_ii <= 0
    - Off-diagonal elements: a_ij >= 0 for i != j

    This ensures convergence to the viscosity solution (Barles-Souganidis 1991).

Author: MFGarchon Development Team
Created: 2026-01-11
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class MonotonicityEnforcer:
    """
    Handles monotonicity constraint enforcement via quadratic programming for GFDM.

    This component enables GFDM solvers to enforce monotone finite difference schemes
    through constrained QP optimization. The enforcer can use either:
    - Indirect Taylor coefficient constraints (physics-based, nD-compatible)
    - Direct Hamiltonian gradient constraints (for standard MFG problems)

    Composition Pattern (Issue #545):
        This component is injected into HJBGFDMSolver instead of using mixin inheritance.
        Benefits: Testable independently, optional (can be None), clear dependencies.

    Parameters
    ----------
    qp_solver : QPSolver
        Quadratic programming solver instance
    qp_constraint_mode : str
        Constraint type: "taylor" (indirect) or "hamiltonian" (direct)
    collocation_points : np.ndarray
        Collocation points, shape (n_points, dimension)
    neighborhoods : dict
        Neighborhood structure for each point
    multi_indices : list[tuple[int, ...]]
        Multi-indices for Taylor expansion
    domain_bounds : list[tuple[float, float]]
        Domain bounds per dimension
    delta : float
        Neighborhood radius parameter
    sigma_function : Callable[[int], float]
        Function to compute sigma value at point index

    Attributes
    ----------
    _qp_solver : QPSolver
        QP solver instance
    qp_constraint_mode : str
        Constraint type
    stats : dict
        QP solve statistics (successes, failures, timing)
    collocation_points : np.ndarray
        Collocation points
    neighborhoods : dict
        Neighborhood structure
    multi_indices : list[tuple[int, ...]]
        Taylor expansion multi-indices
    domain_bounds : list[tuple[float, float]]
        Domain bounds
    delta : float
        Neighborhood radius
    _sigma_function : Callable
        Function to get sigma at point
    _current_density : np.ndarray | None
        Current density field (for MFG coupling)

    Examples
    --------
    >>> # Create enforcer for GFDM solver with QP constraints
    >>> from mfgarchon.utils.numerical.qp_solver import QPSolver
    >>> qp_solver = QPSolver(backend="osqp")
    >>> enforcer = MonotonicityEnforcer(
    ...     qp_solver=qp_solver,
    ...     qp_constraint_mode="taylor",
    ...     collocation_points=collocation_points,
    ...     neighborhoods=neighborhoods,
    ...     multi_indices=multi_indices,
    ...     domain_bounds=[(0, 1), (0, 1)],
    ...     delta=0.1,
    ...     sigma_function=lambda idx: 0.5,
    ... )
    >>>
    >>> # Solve constrained QP for monotone coefficients
    >>> taylor_data = {...}  # Precomputed Taylor matrices
    >>> b = np.random.randn(10)
    >>> coeffs = enforcer.solve_constrained_qp(taylor_data, b, point_idx=0)
    >>>
    >>> # Check M-matrix property
    >>> weights = np.array([...])
    >>> is_monotone, diagnostics = enforcer.check_m_matrix(weights, point_idx=0)
    """

    def __init__(
        self,
        qp_solver: Any,
        qp_constraint_mode: str,
        collocation_points: np.ndarray,
        neighborhoods: dict,
        multi_indices: list[tuple[int, ...]],
        domain_bounds: list[tuple[float, float]],
        delta: float,
        sigma_function: Callable[[int], float],
    ):
        """Initialize monotonicity enforcer."""
        self._qp_solver = qp_solver
        self.qp_constraint_mode = qp_constraint_mode
        self.collocation_points = np.asarray(collocation_points)
        self.neighborhoods = neighborhoods
        self.multi_indices = multi_indices
        self.domain_bounds = domain_bounds
        self.delta = delta
        self._sigma_function = sigma_function

        # Statistics tracking
        self.stats = {
            "total_qp_solves": 0,
            "qp_successes": 0,
            "qp_failures": 0,
            "qp_fallbacks": 0,
            "qp_times": [],
            "osqp_solves": 0,
            "slsqp_solves": 0,
            "lbfgsb_solves": 0,
            "points_checked": 0,
            "violations_detected": 0,
            "violation_point_indices": set(),
            # Per-criterion violation counters (for diagnostics)
            "violation_laplacian": 0,
            "violation_gradient": 0,
            "violation_higher_order": 0,
        }

        # Optional MFG coupling
        self._current_density: np.ndarray | None = None
        self._current_gamma: float = 0.0

    def set_density_for_coupling(self, density: np.ndarray | None, gamma: float = 0.0) -> None:
        """
        Set current density field for MFG coupling in Hamiltonian constraints.

        Parameters
        ----------
        density : np.ndarray | None
            Current density field m(x) at collocation points
        gamma : float
            Coupling strength parameter (default: 0.0)
        """
        self._current_density = density
        self._current_gamma = gamma

    def solve_constrained_qp(self, taylor_data: dict, b: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Solve constrained quadratic programming problem for monotone derivative approximation.

        Solves: min ||W^(1/2) A x - W^(1/2) b||^2
        subject to: monotonicity constraints on finite difference weights

        Parameters
        ----------
        taylor_data : dict
            Dictionary containing precomputed matrices (A, W, etc.)
        b : np.ndarray
            Right-hand side vector
        point_idx : int
            Index of collocation point (for diagnostics)

        Returns
        -------
        np.ndarray
            Derivative coefficients satisfying monotonicity constraints
        """
        t0 = time.time()

        A = taylor_data["A"]
        W = taylor_data["W"]

        # Get neighborhood structure
        center_point = self.collocation_points[point_idx]
        neighborhood = self.neighborhoods[point_idx]
        neighbor_points = neighborhood["points"]
        neighbor_indices = neighborhood["indices"]

        # Build constraints based on mode
        constraints = []

        if self.qp_constraint_mode == "hamiltonian":
            # Direct Hamiltonian gradient constraints
            m_density = 0.0
            if self._current_density is not None and point_idx < len(self._current_density):
                m_density = self._current_density[point_idx]

            hamiltonian_constraints = self._build_hamiltonian_gradient_constraints(
                A,
                neighbor_indices,
                neighbor_points,
                center_point,
                point_idx,
                u_values=None,
                m_density=m_density,
                gamma=self._current_gamma,
            )
            constraints.extend(hamiltonian_constraints)
        else:
            # Indirect Taylor coefficient constraints (default, nD-compatible)
            monotonicity_constraints = self._build_monotonicity_constraints(
                A,
                neighbor_indices,
                neighbor_points,
                center_point,
                point_idx,
            )
            constraints.extend(monotonicity_constraints)

        # Build bounds for optimization variables
        bounds = self._build_coefficient_bounds()

        # Add boundary stability constraints if needed
        boundary_constraints = self._build_boundary_stability_constraints(point_idx)
        constraints.extend(boundary_constraints)

        # Solve weighted least squares with constraints
        try:
            result_x = self._qp_solver.solve_weighted_least_squares(
                A=A,
                b=b,
                W=W,
                bounds=bounds,
                constraints=constraints,
                point_id=point_idx,
            )

            # Update statistics
            self.stats["total_qp_solves"] += 1
            elapsed = time.time() - t0
            self.stats["qp_times"].append(elapsed)

            # Sync backend stats from QPSolver
            qp_stats = self._qp_solver.stats
            self.stats["osqp_solves"] = max(self.stats["osqp_solves"], qp_stats["osqp_solves"])
            self.stats["slsqp_solves"] = max(self.stats["slsqp_solves"], qp_stats["slsqp_solves"])
            self.stats["lbfgsb_solves"] = max(self.stats["lbfgsb_solves"], qp_stats["lbfgsb_solves"])
            self.stats["qp_successes"] = max(self.stats["qp_successes"], qp_stats["successes"])
            self.stats["qp_failures"] = max(self.stats["qp_failures"], qp_stats["failures"])

            return result_x

        except Exception as e:
            # Fallback to unconstrained if error occurs
            warnings.warn(
                f"QP constrained optimization failed at point {point_idx}: {e}. Falling back to unconstrained.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.stats["qp_fallbacks"] += 1
            elapsed = time.time() - t0
            self.stats["qp_times"].append(elapsed)
            return self._solve_unconstrained_fallback(taylor_data, b)

    def _build_coefficient_bounds(self) -> list[tuple[float | None, float | None]]:
        """Build bounds for Taylor coefficient optimization variables."""
        bounds: list[tuple[float | None, float | None]] = []

        for beta in self.multi_indices:
            order = sum(beta)
            if order == 0:  # Constant term
                bounds.append((None, None))
            elif order == 1:  # First derivatives
                bounds.append((-20.0, 20.0))
            elif order == 2:  # Second derivatives
                is_diagonal = sum(1 for b in beta if b != 0) == 1 and max(beta) == 2
                if is_diagonal:
                    bounds.append((-100.0, 100.0))
                else:
                    bounds.append((-50.0, 50.0))
            else:  # Higher order
                bounds.append((-2.0, 2.0))

        return bounds

    def _build_boundary_stability_constraints(self, point_idx: int) -> list[dict]:
        """Build stability constraints for points near boundaries."""
        constraints = []

        # Check if near boundary (vectorized, nD-compatible)
        center_point = self.collocation_points[point_idx]
        bounds_array = np.array(self.domain_bounds)
        threshold = 0.1 * self.delta
        near_left = np.abs(center_point - bounds_array[:, 0]) < threshold
        near_right = np.abs(center_point - bounds_array[:, 1]) < threshold
        near_boundary = np.any(near_left | near_right)

        if near_boundary:
            # Find diagonal second derivative indices
            diag_second_deriv_indices = []
            for k, beta in enumerate(self.multi_indices):
                is_diagonal = sum(beta) == 2 and sum(1 for b in beta if b != 0) == 1
                if is_diagonal:
                    diag_second_deriv_indices.append(k)

            if diag_second_deriv_indices:

                def constraint_stability(x, indices=diag_second_deriv_indices):
                    """Mild stability constraint near boundaries (nD-compatible)."""
                    return min(50.0 - abs(x[k]) for k in indices)

                constraints.append({"type": "ineq", "fun": constraint_stability})

        return constraints

    def _solve_unconstrained_fallback(self, taylor_data: dict, b: np.ndarray) -> np.ndarray:
        """Fallback to unconstrained solution using SVD or normal equations."""
        if taylor_data.get("use_svd", False):
            sqrt_W = taylor_data["sqrt_W"]
            U = taylor_data["U"]
            S = taylor_data["S"]
            Vt = taylor_data["Vt"]

            Wb = sqrt_W @ b
            UT_Wb = U.T @ Wb
            S_inv_UT_Wb = UT_Wb / S
            return Vt.T @ S_inv_UT_Wb
        elif taylor_data.get("AtWA_inv") is not None:
            return taylor_data["AtWA_inv"] @ taylor_data["AtW"] @ b
        else:
            A = taylor_data["A"]
            from scipy.linalg import lstsq

            if A is not None and b is not None:
                lstsq_result = lstsq(A, b)
                coeffs = lstsq_result[0] if lstsq_result is not None else np.zeros(len(b))
            else:
                coeffs = np.zeros(len(b) if b is not None else 1)
            return coeffs

    def check_monotonicity_violation(
        self,
        D_coeffs: np.ndarray,
        point_idx: int = 0,
        use_adaptive: bool | None = None,
    ) -> bool:
        """
        Check if unconstrained Taylor coefficients violate monotonicity.

        Parameters
        ----------
        D_coeffs : np.ndarray
            Taylor derivative coefficients from unconstrained solve
        point_idx : int
            Collocation point index (for debugging)
        use_adaptive : bool | None
            Override adaptive mode (deprecated, always uses basic check)

        Returns
        -------
        bool
            True if QP constraints are needed

        Mathematical Criteria:
            1. Laplacian negativity: D_2 < 0 (diffusion dominance)
            2. Gradient boundedness: |D_1| <= C*sigma^2*|D_2|
            3. Higher-order control: sum|D_k| < |D_2| for order >= 3
        """
        # Find multi-index locations
        laplacian_idx = None
        gradient_idx = None

        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 2 and all(b <= 2 for b in beta):
                if laplacian_idx is None:
                    laplacian_idx = k
            elif sum(beta) == 1:
                if gradient_idx is None:
                    gradient_idx = k

        if laplacian_idx is None:
            return False  # Cannot check without Laplacian

        # Extract coefficients
        D_laplacian = D_coeffs[laplacian_idx]
        tolerance = 1e-12
        laplacian_mag = abs(D_laplacian) + 1e-10

        # Criterion 1: Laplacian Negativity
        violation_1 = D_laplacian >= -tolerance

        # Criterion 2: Gradient Boundedness
        violation_2 = False
        if gradient_idx is not None:
            D_gradient = D_coeffs[gradient_idx]
            sigma = self._sigma_function(point_idx)
            scale_factor = 10.0 * max(sigma**2, 0.1)
            gradient_mag = abs(D_gradient)
            violation_2 = gradient_mag > scale_factor * laplacian_mag

        # Criterion 3: Higher-Order Control
        higher_order_norm = sum(abs(D_coeffs[k]) for k in range(len(D_coeffs)) if sum(self.multi_indices[k]) >= 3)
        violation_3 = higher_order_norm > laplacian_mag

        # Basic violation check
        has_violation = violation_1 or violation_2 or violation_3

        # Track per-criterion stats
        if has_violation:
            if violation_1:
                self.stats["violation_laplacian"] += 1
            if violation_2:
                self.stats["violation_gradient"] += 1
            if violation_3:
                self.stats["violation_higher_order"] += 1

        if not use_adaptive:
            return has_violation

        # Adaptive mode: quantitative severity
        severity = 0.0

        if violation_1:
            severity = max(severity, D_laplacian + tolerance)

        if violation_2:
            D_gradient = D_coeffs[gradient_idx]
            sigma = self._sigma_function(point_idx)
            scale_factor = 10.0 * max(sigma**2, 0.1)
            gradient_mag = abs(D_gradient)
            excess_gradient = gradient_mag / laplacian_mag - scale_factor
            severity = max(severity, excess_gradient)

        if violation_3:
            excess_higher_order = higher_order_norm / laplacian_mag - 1.0
            severity = max(severity, excess_higher_order)

        return severity > 0.0

    def check_m_matrix(self, weights: np.ndarray, point_idx: int, tolerance: float = 1e-12) -> tuple[bool, dict]:
        """
        Verify M-matrix property for finite difference weights.

        For a monotone scheme, the Laplacian weights must satisfy:
        - Diagonal (center): w_center <= 0
        - Off-diagonal (neighbors): w_j >= -tolerance for j != center

        Parameters
        ----------
        weights : np.ndarray
            Finite difference weights [n_neighbors]
        point_idx : int
            Index of collocation point
        tolerance : float
            Small tolerance for numerical errors

        Returns
        -------
        is_monotone : bool
            True if M-matrix property satisfied
        diagnostics : dict
            Dictionary with detailed information
        """
        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]

        # Find center point index in neighborhood
        center_idx_in_neighbors = None
        center_point = self.collocation_points[point_idx]

        for j, idx in enumerate(neighbor_indices):
            if idx == -1:  # Ghost particle
                if np.allclose(neighborhood["points"][j], center_point):
                    center_idx_in_neighbors = j
                    break
            elif idx == point_idx or np.allclose(self.collocation_points[idx], center_point):
                center_idx_in_neighbors = j
                break

        if center_idx_in_neighbors is None:
            w_center = 0.0
            neighbor_weights = weights
        else:
            w_center = weights[center_idx_in_neighbors]
            neighbor_weights = np.delete(weights, center_idx_in_neighbors)

        # Check M-matrix conditions
        center_ok = w_center <= tolerance
        neighbors_ok = np.all(neighbor_weights >= -tolerance)

        is_monotone = center_ok and neighbors_ok

        diagnostics = {
            "is_monotone": is_monotone,
            "center_ok": center_ok,
            "neighbors_ok": neighbors_ok,
            "w_center": float(w_center),
            "min_neighbor_weight": float(np.min(neighbor_weights)) if len(neighbor_weights) > 0 else 0.0,
            "max_neighbor_weight": float(np.max(neighbor_weights)) if len(neighbor_weights) > 0 else 0.0,
            "num_violations": int(np.sum(neighbor_weights < -tolerance)),
            "num_neighbors": len(neighbor_weights),
            "violation_severity": float(abs(np.min(neighbor_weights)))
            if len(neighbor_weights) > 0 and np.min(neighbor_weights) < -tolerance
            else 0.0,
        }

        return is_monotone, diagnostics

    def _build_monotonicity_constraints(
        self,
        A: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_points: np.ndarray,
        center_point: np.ndarray,
        point_idx: int,
    ) -> list[dict]:
        """
        Build M-matrix monotonicity constraints for finite difference weights.

        Uses INDIRECT constraints on Taylor coefficients D for physics-based
        constraint enforcement.

        Constraint Categories:
            1. Diffusion dominance: d^2u/dx^2 coefficient should be negative
            2. Gradient boundedness: du/dx shouldn't overwhelm diffusion
            3. Truncation error control: Higher derivatives should be small

        Parameters
        ----------
        A : np.ndarray
            Taylor expansion matrix [n_neighbors, n_coeffs]
        neighbor_indices : np.ndarray
            Indices of neighbor points
        neighbor_points : np.ndarray
            Coordinates of neighbor points
        center_point : np.ndarray
            Coordinates of center point
        point_idx : int
            Index of center collocation point

        Returns
        -------
        list[dict]
            List of constraint dictionaries for scipy.optimize.minimize
        """
        constraints = []

        # Find indices for derivatives (nD-compatible)
        laplacian_indices = []
        first_deriv_indices = []

        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 2 and sum(1 for b in beta if b != 0) == 1:
                laplacian_indices.append(k)
            elif sum(beta) == 1:
                first_deriv_indices.append(k)

        if not laplacian_indices:
            return constraints

        # Constraint 1: Negative Laplacian (Diffusion Dominance)
        def constraint_laplacian_negative(x, indices=laplacian_indices):
            """Enforce Laplacian components are negative."""
            laplacian_sum = sum(x[idx] for idx in indices)
            return -laplacian_sum

        constraints.append({"type": "ineq", "fun": constraint_laplacian_negative})

        # Constraint 2: Gradient Boundedness
        if first_deriv_indices:
            sigma = self._sigma_function(point_idx)
            sigma_sq = sigma**2

            def constraint_gradient_bounded(
                x,
                grad_indices=first_deriv_indices,
                lap_indices=laplacian_indices,
                sig_sq=sigma_sq,
            ):
                """Ensure gradient norm doesn't dominate Laplacian norm."""
                gradient_norm_sq = sum(x[idx] ** 2 for idx in grad_indices)
                gradient_norm = np.sqrt(gradient_norm_sq + 1e-20)
                laplacian_mag = sum(abs(x[idx]) for idx in lap_indices) + 1e-10
                scale_factor = 10.0 * max(sig_sq, 0.1)
                return scale_factor * laplacian_mag - gradient_norm

            constraints.append({"type": "ineq", "fun": constraint_gradient_bounded})

        # Constraint 3: Higher-Order Term Control
        def constraint_higher_order_small(x, lap_indices=laplacian_indices):
            """Keep higher-order terms small (truncation error control)."""
            higher_order_norm = 0.0
            for k, beta in enumerate(self.multi_indices):
                if sum(beta) >= 3:
                    higher_order_norm += abs(x[k])
            laplacian_mag = sum(abs(x[idx]) for idx in lap_indices) + 1e-10
            return laplacian_mag - higher_order_norm

        constraints.append({"type": "ineq", "fun": constraint_higher_order_small})

        return constraints

    def _build_hamiltonian_gradient_constraints(
        self,
        A: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_points: np.ndarray,
        center_point: np.ndarray,
        point_idx: int,
        u_values: np.ndarray | None = None,
        m_density: float = 0.0,
        gamma: float = 0.0,
    ) -> list[dict]:
        """
        Build direct Hamiltonian gradient constraints for monotonicity.

        For a monotone scheme, we require:
            dH_h/du_j >= 0  for all neighbors j != j_0 (center)

        For the standard MFG Hamiltonian H = 1/2|grad(u)|^2 + gamma*m*|grad(u)|^2 + V(x):
            dH_h/du_j = (1 + 2*gamma*m) * (sum_l c_{j_0,l} * u_l) * c_{j_0,j}

        Parameters
        ----------
        A : np.ndarray
            Taylor expansion matrix [n_neighbors, n_coeffs]
        neighbor_indices : np.ndarray
            Indices of neighbor points
        neighbor_points : np.ndarray
            Coordinates of neighbor points [n_neighbors, d]
        center_point : np.ndarray
            Coordinates of center point [d]
        point_idx : int
            Index of center collocation point
        u_values : np.ndarray | None
            Current value function estimates (optional)
        m_density : float
            Local population density m(x) at center point
        gamma : float
            Coupling strength parameter gamma >= 0

        Returns
        -------
        list[dict]
            List of constraint dictionaries for scipy.optimize.minimize
        """
        constraints = []

        # Find gradient indices in multi_indices
        gradient_indices = []
        for k, beta in enumerate(self.multi_indices):
            if sum(beta) == 1:
                gradient_indices.append((k, beta))

        if not gradient_indices:
            return constraints

        # Compute coupling factor (1 + 2*gamma*m)
        coupling_factor = 1.0 + 2.0 * gamma * m_density

        # Build constraints for each neighbor
        n_neighbors = len(neighbor_indices)
        for j in range(n_neighbors):
            # Skip center point
            if neighbor_indices[j] == point_idx:
                continue

            # Direction from center to neighbor j
            direction = neighbor_points[j] - center_point
            dist = np.linalg.norm(direction)

            if dist < 1e-12:
                continue

            unit_direction = direction / dist

            def make_constraint(grad_idx_list, unit_dir, cf):
                """Factory function to create closure with correct values."""

                def constraint_func(x):
                    """Hamiltonian gradient constraint: dH/du_j >= 0."""
                    grad_dot_dir = 0.0
                    for k_idx, beta in grad_idx_list:
                        dim_idx = beta.index(1)
                        grad_dot_dir += x[k_idx] * unit_dir[dim_idx]
                    return cf * grad_dot_dir

                return constraint_func

            constraint_fn = make_constraint(gradient_indices, unit_direction, coupling_factor)
            constraints.append({"type": "ineq", "fun": constraint_fn})

        return constraints

    def compute_fd_weights_from_taylor(self, taylor_data: dict, derivative_idx: int) -> np.ndarray | None:
        """
        Compute finite difference weights for a specific derivative.

        For GFDM with weighted least squares, given:
        - A: Taylor expansion matrix [n_neighbors, n_derivs]
        - W: Weight matrix [n_neighbors, n_neighbors]
        - We solve: min ||sqrt(W) @ (A @ D - b)||^2 to get D from b

        To get weights w such that D^β = w @ b (where b = u_center - u_neighbors):
        We need the β-th row of the solution operator (A^T W A)^{-1} A^T W

        Parameters
        ----------
        taylor_data : dict
            Precomputed Taylor matrices
        derivative_idx : int
            Index of derivative in multi_indices

        Returns
        -------
        np.ndarray | None
            Array of finite difference weights [n_neighbors] or None if computation fails
        """
        if taylor_data.get("use_svd"):
            # Use SVD decomposition
            U = taylor_data["U"]
            S = taylor_data["S"]
            Vt = taylor_data["Vt"]
            sqrt_W = taylor_data["sqrt_W"]

            # Compute: weights_matrix = Vt.T @ diag(1/S) @ U.T @ sqrt_W
            # Shape: [n_derivs, n_neighbors]
            weights_matrix = Vt.T @ np.diag(1.0 / S) @ U.T @ sqrt_W

            # Extract β-th row
            weights = weights_matrix[derivative_idx, :]
            return weights

        elif taylor_data.get("use_qr"):
            # Use QR decomposition - fall back to normal equations
            A = taylor_data["A"]
            W = taylor_data["W"]
            try:
                # Issue #1066: solve(AtWA, AtW) directly instead of inv(AtWA) @ AtW.
                # Squares condition number; on marginal stencils the inv path can
                # silently produce weights that violate M-matrix off-diag positivity.
                AtW = A.T @ W
                weights_matrix = np.linalg.solve(A.T @ W @ A, AtW)
                weights = weights_matrix[derivative_idx, :]
                return weights
            except np.linalg.LinAlgError:
                return None

        elif taylor_data.get("AtWA_inv") is not None:
            # Direct normal equations
            AtWA_inv = taylor_data["AtWA_inv"]
            W = taylor_data["W"]
            A = taylor_data["A"]
            weights_matrix = AtWA_inv @ A.T @ W
            weights = weights_matrix[derivative_idx, :]
            return weights

        else:
            raise ValueError(
                "taylor_data must contain one of: 'use_svd', 'use_qr', or 'AtWA_inv'. "
                f"Got keys: {list(taylor_data.keys())}"
            )

    def print_diagnostics(self, hjb_method_name: str = "HJB-GFDM", qp_cache: Any = None) -> None:
        """
        Print comprehensive QP diagnostic statistics.

        Reports QP solve counts, timings, success rates, and solver usage.
        Also includes QPSolver caching and warm-start statistics.

        Parameters
        ----------
        hjb_method_name : str
            Name of HJB method for display
        qp_cache : QPCache | None
            Optional QP cache instance for cache statistics
        """
        if not self.stats.get("total_qp_solves", 0):
            print("\nQP Diagnostics: No QP solves recorded")
            return

        print("\n" + "=" * 80)
        print(f"QP DIAGNOSTICS - {hjb_method_name}")
        print("=" * 80)

        # Basic counts
        total_solves = self.stats["total_qp_solves"]
        print("\nGFDM QP Solve Summary:")
        print(f"  Total QP solves:        {total_solves}")
        print(
            f"  Successful solves:      {self.stats['qp_successes']} "
            f"({100 * self.stats['qp_successes'] / max(total_solves, 1):.1f}%)"
        )
        print(
            f"  Failed solves:          {self.stats['qp_failures']} "
            f"({100 * self.stats['qp_failures'] / max(total_solves, 1):.1f}%)"
        )
        print(f"  Fallbacks:              {self.stats['qp_fallbacks']}")

        # M-matrix checking (for "auto" level)
        if self.stats["points_checked"] > 0:
            print("\nM-Matrix Violation Detection ('auto' level):")
            print(f"  Points checked:         {self.stats['points_checked']}")
            print(
                f"  Violations detected:    {self.stats['violations_detected']} "
                f"({100 * self.stats['violations_detected'] / max(self.stats['points_checked'], 1):.1f}%)"
            )

        # Timing statistics
        if self.stats["qp_times"]:
            times = np.array(self.stats["qp_times"])
            print("\nGFDM QP Solve Timing:")
            print(f"  Total time:             {np.sum(times):.2f} s")
            print(f"  Mean time per solve:    {np.mean(times) * 1000:.2f} ms")
            print(f"  Median time per solve:  {np.median(times) * 1000:.2f} ms")
            print(f"  Min time per solve:     {np.min(times) * 1000:.2f} ms")
            print(f"  Max time per solve:     {np.max(times) * 1000:.2f} ms")
            print(f"  Std dev:                {np.std(times) * 1000:.2f} ms")

        # QPSolver backend statistics
        qp_stats = self._qp_solver.stats
        print("\nQPSolver Backend Statistics:")
        print(f"  OSQP:                   {qp_stats['osqp_solves']}")
        print(f"  scipy (SLSQP):          {qp_stats['slsqp_solves']}")
        print(f"  scipy (L-BFGS-B):       {qp_stats['lbfgsb_solves']}")

        # Warm-start stats
        ws_total = qp_stats["warm_starts"] + qp_stats["cold_starts"]
        if ws_total > 0:
            print("\nWarm-Start Statistics:")
            print(
                f"  Warm starts:            {qp_stats['warm_starts']} ({100 * qp_stats['warm_starts'] / ws_total:.1f}%)"
            )
            print(f"  Cold starts:            {qp_stats['cold_starts']}")

        # Cache stats
        if qp_cache is not None:
            cache_total = qp_stats["cache_hits"] + qp_stats["cache_misses"]
            if cache_total > 0:
                print("\nCache Statistics:")
                print(
                    f"  Cache hits:             {qp_stats['cache_hits']} "
                    f"({100 * qp_stats['cache_hits'] / cache_total:.1f}%)"
                )
                print(f"  Cache misses:           {qp_stats['cache_misses']}")
                print(f"  Cache size:             {qp_cache.size} / {qp_cache.max_size}")

        print("=" * 80 + "\n")
