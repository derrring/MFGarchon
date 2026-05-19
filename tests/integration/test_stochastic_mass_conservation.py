#!/usr/bin/env python3
"""
Mass Conservation Test with Probabilistic Convergence Interpretation.

For stochastic particle methods, convergence should be interpreted in a
probabilistic framework:
- Errors fluctuate due to particle noise (normal behavior)
- Convergence in measure/distribution, not pointwise
- Use statistical stopping criteria (running averages, quantiles)
"""

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


class ProbabilisticConvergenceMonitor:
    """Monitor convergence using statistical criteria for stochastic methods."""

    def __init__(self, window_size=10, quantile=0.9):
        self.window_size = window_size
        self.quantile = quantile
        self.errors_u = []
        self.errors_m = []

    def add_iteration(self, error_u, error_m):
        """Add iteration error."""
        self.errors_u.append(error_u)
        self.errors_m.append(error_m)

    def get_running_statistics(self):
        """Get running statistics over recent window."""
        if len(self.errors_u) < self.window_size:
            return None

        recent_u = self.errors_u[-self.window_size :]
        recent_m = self.errors_m[-self.window_size :]

        return {
            "mean_u": np.mean(recent_u),
            "mean_m": np.mean(recent_m),
            "std_u": np.std(recent_u),
            "std_m": np.std(recent_m),
            "median_u": np.median(recent_u),
            "median_m": np.median(recent_m),
            "quantile_u": np.quantile(recent_u, self.quantile),
            "quantile_m": np.quantile(recent_m, self.quantile),
            "max_u": np.max(recent_u),
            "max_m": np.max(recent_m),
        }

    def check_stochastic_convergence(self, tolerance=1e-4):
        """
        Check if converged in statistical sense.

        Uses median error over window (robust to outliers from particle noise).
        """
        stats = self.get_running_statistics()
        if stats is None:
            return False

        # Use median (robust to spikes) instead of mean
        return stats["median_u"] < tolerance and stats["median_m"] < tolerance


def solve_with_stochastic_monitoring(seed=42, max_iterations=100, tolerance=1e-4, verbose=True):
    """
    Solve MFG with probabilistic convergence monitoring.

    Returns:
        (converged, result, masses, problem, monitor)
    """
    np.random.seed(seed)

    # Create problem
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[52], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=51,
        sigma=1.0,
        coupling_coefficient=0.5,
        components=_default_components(),
    )

    bc = no_flux_bc(dimension=1)

    fp_solver = FPParticleSolver(
        problem,
        num_particles=1000,
        normalize_kde_output=True,
        boundary_conditions=bc,
    )

    hjb_solver = HJBFDMSolver(problem)

    _ = FixedPointIterator(
        problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        relaxation=0.5,
    )

    # Custom iteration loop with stochastic monitoring
    monitor = ProbabilisticConvergenceMonitor(window_size=10, quantile=0.9)

    # Initialize
    (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
    Nt_points = problem.Nt + 1  # Temporal grid points
    U = np.zeros((Nt_points, Nx_points))  # Terminal condition will be set by HJB solver
    M = problem.m_initial  # Issue #670: unified naming

    converged = False
    iteration = 0

    if verbose:
        print("\n" + "=" * 80)
        print("Solving with Probabilistic Convergence Monitoring")
        print("=" * 80)
        print(f"Window size: {monitor.window_size}, Quantile: {monitor.quantile}")
        print(f"Stopping criterion: median error < {tolerance:.2e}")
        print()

    for iteration in range(1, max_iterations + 1):
        # Store previous iteration
        U_prev = U.copy()
        M_prev = M.copy()

        # FP forward step
        M = fp_solver.solve_fp_system(M, U)

        # HJB backward step
        U = hjb_solver.solve_hjb_system(M)

        # Compute errors
        error_u = np.linalg.norm(U - U_prev) / (np.linalg.norm(U_prev) + 1e-12)
        error_m = np.linalg.norm(M - M_prev) / (np.linalg.norm(M_prev) + 1e-12)

        # Add to monitor
        monitor.add_iteration(error_u, error_m)

        # Get statistics
        stats = monitor.get_running_statistics()

        if verbose and iteration % 5 == 0:
            if stats:
                print(
                    f"Iter {iteration:3d}: "
                    f"Instant: U={error_u:.2e} M={error_m:.2e} | "
                    f"Median: U={stats['median_u']:.2e} M={stats['median_m']:.2e} | "
                    f"90%ile: U={stats['quantile_u']:.2e} M={stats['quantile_m']:.2e}"
                )
            else:
                print(f"Iter {iteration:3d}: Instant: U={error_u:.2e} M={error_m:.2e} (warming up...)")

        # Check stochastic convergence
        if iteration >= monitor.window_size:
            if monitor.check_stochastic_convergence(tolerance):
                converged = True
                if verbose:
                    print(f"\n✅ Stochastic convergence achieved at iteration {iteration}")
                    print(f"   Median error U: {stats['median_u']:.2e}, M: {stats['median_m']:.2e}")
                break

    # Compute masses
    dx = problem.geometry.get_grid_spacing()[0]
    Nt_points = problem.geometry.get_grid_shape()[0]
    masses = np.array([float(np.trapezoid(M[t, :], dx=dx)) for t in range(Nt_points)])

    # Create result object
    class Result:
        def __init__(self, u, m, converged, iterations):
            self.u = u
            self.m = m
            self.converged = converged
            self.iterations = iterations
            self.final_error = monitor.get_running_statistics()["median_u"] if converged else np.nan

    result = Result(U, M, converged, iteration)

    return converged, result, masses, problem, monitor


def visualize_stochastic_convergence(result, masses, problem, monitor):
    """Visualize solution with stochastic convergence analysis."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Extract arrays
    U = result.u
    M = result.m
    x = problem.xSpace
    t = problem.tSpace

    # 1. Value function
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(U, aspect="auto", origin="lower", extent=[x[0], x[-1], t[0], t[-1]], cmap="viridis")
    ax1.set_xlabel("Space x")
    ax1.set_ylabel("Time t")
    ax1.set_title("Value Function u(t,x)")
    plt.colorbar(im1, ax=ax1)

    # 2. Density
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(M, aspect="auto", origin="lower", extent=[x[0], x[-1], t[0], t[-1]], cmap="plasma")
    ax2.set_xlabel("Space x")
    ax2.set_ylabel("Time t")
    ax2.set_title("Density m(t,x)")
    plt.colorbar(im2, ax=ax2)

    # 3. Mass conservation
    ax3 = fig.add_subplot(gs[0, 2])
    time_steps = np.arange(len(masses)) * problem.dt
    ax3.plot(time_steps, masses, "b-", linewidth=2, label="Total mass")
    ax3.axhline(y=masses[0], color="r", linestyle="--", linewidth=1, label=f"Initial = {masses[0]:.6f}")
    ax3.set_xlabel("Time t")
    ax3.set_ylabel("Total Mass")
    ax3.set_title("Mass Conservation")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Mass statistics
    mass_errors = np.abs(masses - masses[0])
    max_error = np.max(mass_errors)
    mean_error = np.mean(mass_errors)
    rel_error_pct = (max_error / masses[0]) * 100

    ax3.text(
        0.02,
        0.98,
        f"Max: {max_error:.2e}\nMean: {mean_error:.2e}\nRel: {rel_error_pct:.2f}%",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        fontsize=8,
    )

    # 4. Stochastic error evolution (U)
    ax4 = fig.add_subplot(gs[1, :2])
    iterations = np.arange(1, len(monitor.errors_u) + 1)
    ax4.semilogy(iterations, monitor.errors_u, "b-", alpha=0.3, linewidth=0.5, label="Instantaneous error")

    # Running statistics
    window = monitor.window_size
    if len(monitor.errors_u) >= window:
        medians = [np.median(monitor.errors_u[max(0, i - window) : i + 1]) for i in range(len(monitor.errors_u))]
        means = [np.mean(monitor.errors_u[max(0, i - window) : i + 1]) for i in range(len(monitor.errors_u))]

        ax4.semilogy(iterations, medians, "r-", linewidth=2, label=f"Median (window={window})")
        ax4.semilogy(iterations, means, "g--", linewidth=1.5, label=f"Mean (window={window})")

    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Relative Error (U)")
    ax4.set_title("Stochastic Convergence: Value Function Error")
    ax4.grid(True, alpha=0.3, which="both")
    ax4.legend()

    # 5. Stochastic error evolution (M)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.semilogy(iterations, monitor.errors_m, "b-", alpha=0.3, linewidth=0.5, label="Instantaneous")

    if len(monitor.errors_m) >= window:
        medians_m = [np.median(monitor.errors_m[max(0, i - window) : i + 1]) for i in range(len(monitor.errors_m))]
        ax5.semilogy(iterations, medians_m, "r-", linewidth=2, label=f"Median (w={window})")

    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Relative Error (M)")
    ax5.set_title("Density Error")
    ax5.grid(True, alpha=0.3, which="both")
    ax5.legend()

    # 6. Error histogram
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(np.log10(monitor.errors_u), bins=30, alpha=0.7, color="blue", edgecolor="black")
    ax6.set_xlabel("log10(Error U)")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Error Distribution")
    ax6.grid(True, alpha=0.3)

    # 7. Error statistics
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis("off")

    stats_text = f"""
    STOCHASTIC CONVERGENCE ANALYSIS
    {"=" * 50}

    Iterations: {result.iterations}
    Converged: {result.converged}

    Error Statistics (last {window} iterations):
    ------------------------------------------------
    Value Function (U):
      Median:  {np.median(monitor.errors_u[-window:]):.2e}
      Mean:    {np.mean(monitor.errors_u[-window:]):.2e}
      Std:     {np.std(monitor.errors_u[-window:]):.2e}
      90%ile:  {np.quantile(monitor.errors_u[-window:], 0.9):.2e}
      Max:     {np.max(monitor.errors_u[-window:]):.2e}

    Density (M):
      Median:  {np.median(monitor.errors_m[-window:]):.2e}
      Mean:    {np.mean(monitor.errors_m[-window:]):.2e}
      Std:     {np.std(monitor.errors_m[-window:]):.2e}
      90%ile:  {np.quantile(monitor.errors_m[-window:], 0.9):.2e}
      Max:     {np.max(monitor.errors_m[-window:]):.2e}

    Mass Conservation:
      Initial mass:     {masses[0]:.8f}
      Final mass:       {masses[-1]:.8f}
      Max deviation:    {max_error:.2e}
      Relative error:   {rel_error_pct:.4f}%

    Interpretation:
    ------------------------------------------------
    - Instantaneous errors fluctuate due to particle noise
    - Median/Mean provide robust statistical convergence
    - Error spikes are NORMAL for stochastic methods
    - Convergence in measure, not pointwise deterministic
    """

    ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes, fontsize=9, verticalalignment="top", family="monospace")

    plt.suptitle(
        "Mass Conservation: Stochastic Particle Method (Probabilistic Framework)", fontsize=14, fontweight="bold"
    )

    return fig


def main():
    """Run stochastic mass conservation test."""
    print("\n" + "=" * 80)
    print("MASS CONSERVATION TEST: PROBABILISTIC CONVERGENCE FRAMEWORK")
    print("=" * 80)
    print("\nFor stochastic particle methods:")
    print("  - Error fluctuations are NORMAL (particle noise)")
    print("  - Convergence in measure/distribution, not pointwise")
    print("  - Use statistical criteria: median, quantiles over window")
    print("=" * 80)

    # Run with stochastic monitoring
    converged, result, masses, problem, monitor = solve_with_stochastic_monitoring(
        seed=42, max_iterations=100, tolerance=1e-4, verbose=True
    )

    if converged:
        print("\n" + "=" * 80)
        print("✅ STOCHASTIC CONVERGENCE ACHIEVED")
        print("=" * 80)

        # Visualize
        print("\nGenerating visualization...")
        fig = visualize_stochastic_convergence(result, masses, problem, monitor)

        output_file = "stochastic_mass_conservation.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")

        plt.show()

    else:
        print("\n" + "=" * 80)
        print("❌ Did not converge within iteration limit")
        print("=" * 80)
        print("Try: Increase max_iterations or relax tolerance")


if __name__ == "__main__":
    main()
