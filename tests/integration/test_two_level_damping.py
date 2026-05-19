#!/usr/bin/env python3
"""
Test Two-Level Damping: Picard Damping + Anderson Acceleration.

This combines standard Picard damping with Anderson acceleration
for improved stability on stochastic particle-based solvers.
"""

import matplotlib

matplotlib.use("Agg")

import time

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


def run_solver(name: str, use_anderson: bool, damping_factor: float, anderson_beta: float | None = None):
    """Run solver with specified damping configuration."""
    np.random.seed(42)

    # Problem setup
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[26], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=25,
        sigma=1.0,
        coupling_coefficient=0.5,
        components=_default_components(),
    )
    bc = no_flux_bc(dimension=1)

    # Solvers
    fp_solver = FPParticleSolver(
        problem,
        num_particles=500,
        normalize_kde_output=True,
        boundary_conditions=bc,
    )
    hjb_solver = HJBFDMSolver(problem)

    # MFG solver
    mfg_solver = FixedPointIterator(
        problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        relaxation=damping_factor,
        use_anderson=use_anderson,
        anderson_depth=5,
        anderson_beta=anderson_beta or 1.0,
    )

    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"  Picard damping (theta): {damping_factor}")
    if use_anderson:
        print(f"  Anderson beta: {anderson_beta}")
    print(f"{'=' * 80}\n")

    # Run solver
    start_time = time.time()
    try:
        result = mfg_solver.solve(max_iterations=30, tolerance=1e-3, verbose=False)
        _U, M = result[:2]
    except Exception as e:
        print(f"Exception: {str(e)[:80]}...")
        _ = mfg_solver.U if hasattr(mfg_solver, "U") else np.zeros((26, 26))
        M = mfg_solver.M if hasattr(mfg_solver, "M") else problem.m_initial

    elapsed_time = time.time() - start_time

    # Get convergence history
    err_u = mfg_solver.l2distu_rel if hasattr(mfg_solver, "l2distu_rel") else []
    err_m = mfg_solver.l2distm_rel if hasattr(mfg_solver, "l2distm_rel") else []
    iterations = mfg_solver.iterations_run if hasattr(mfg_solver, "iterations_run") else 0

    # Mass conservation
    dx = problem.geometry.get_grid_spacing()[1]  # Spatial spacing
    Nt_points = problem.geometry.get_grid_shape()[0]  # Temporal grid points
    masses = np.array([float(np.trapezoid(M[t, :], dx=dx)) for t in range(Nt_points)])

    mass_dev = np.max(np.abs(masses - masses[0]))

    print(f"Iterations: {iterations}")
    print(f"Time: {elapsed_time:.2f}s ({elapsed_time / iterations:.3f}s/iter)")
    print(f"Final error U: {err_u[-1]:.2e}" if len(err_u) > 0 else "N/A")
    print(f"Final error M: {err_m[-1]:.2e}" if len(err_m) > 0 else "N/A")
    print(f"Mass deviation: {mass_dev:.2e}")

    # Check stability
    is_stable = len(err_u) > 0 and err_u[-1] < 1.0 and len(err_m) > 0 and err_m[-1] < 1.0 and mass_dev < 0.1
    print(f"Stable: {'YES' if is_stable else 'NO'}")

    return {
        "name": name,
        "err_u": err_u,
        "err_m": err_m,
        "masses": masses,
        "iterations": iterations,
        "time": elapsed_time,
        "mass_dev": mass_dev,
        "stable": is_stable,
    }


def main():
    """Test different damping configurations."""
    print("\n" + "=" * 80)
    print("TWO-LEVEL DAMPING TEST")
    print("=" * 80)

    # Test configurations
    configs = [
        # Baseline
        ("Damped Only (θ=0.5)", False, 0.5, None),
        # Anderson without Picard damping
        ("Anderson Only (β=1.0)", True, 1.0, 1.0),
        # Two-level: Light Picard + Anderson
        ("Two-Level: θ=0.7 + β=0.8", True, 0.7, 0.8),
        # Two-level: Moderate Picard + Anderson
        ("Two-Level: θ=0.5 + β=0.8", True, 0.5, 0.8),
        # Two-level: Heavy Picard + Anderson
        ("Two-Level: θ=0.3 + β=0.5", True, 0.3, 0.5),
    ]

    results = []
    for name, use_anderson, theta, anderson_beta in configs:
        try:
            result = run_solver(name, use_anderson, theta, anderson_beta)
            results.append(result)
        except Exception as e:
            print(f"Failed: {e}")

    # Create comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Convergence errors
    ax = axes[0, 0]
    for res in results:
        if len(res["err_u"]) > 0:
            iters = range(1, len(res["err_u"]) + 1)
            linestyle = "-" if res["stable"] else "--"
            ax.semilogy(iters, res["err_u"], linestyle, label=f"{res['name']} (U)", alpha=0.7)
    ax.axhline(y=1e-3, color="r", linestyle=":", alpha=0.5, label="Tolerance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Error U")
    ax.set_title("Convergence: U Errors")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. M errors
    ax = axes[0, 1]
    for res in results:
        if len(res["err_m"]) > 0:
            iters = range(1, len(res["err_m"]) + 1)
            linestyle = "-" if res["stable"] else "--"
            ax.semilogy(iters, res["err_m"], linestyle, label=f"{res['name']}", alpha=0.7)
    ax.axhline(y=1e-3, color="r", linestyle=":", alpha=0.5, label="Tolerance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Error M")
    ax.set_title("Convergence: M Errors")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Mass conservation
    ax = axes[1, 0]
    for res in results:
        t_steps = np.arange(len(res["masses"])) * (1.0 / 25)
        linestyle = "-" if res["stable"] else "--"
        ax.plot(t_steps, res["masses"], linestyle, label=f"{res['name']}", alpha=0.7)
    ax.axhline(y=1.0, color="k", linestyle=":", alpha=0.5)
    ax.fill_between([0, 1], 0.98, 1.02, alpha=0.1, color="gray", label="±2%")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Total Mass")
    ax.set_title("Mass Conservation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.1])

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = "SUMMARY\n" + "=" * 50 + "\n\n"
    summary_text += f"{'Method':<35} {'Time':>7} {'Mass Dev':>10} {'Stable':>8}\n"
    summary_text += "-" * 50 + "\n"

    for res in results:
        name_short = res["name"][:34]
        time_str = f"{res['time']:.1f}s"
        mass_str = f"{res['mass_dev']:.2e}"
        stable_str = "YES" if res["stable"] else "NO"
        summary_text += f"{name_short:<35} {time_str:>7} {mass_str:>10} {stable_str:>8}\n"

    summary_text += "\n" + "=" * 50 + "\n"
    summary_text += "\nKEY FINDINGS:\n"
    summary_text += "-" * 50 + "\n"

    # Find best stable configuration
    stable_results = [r for r in results if r["stable"]]
    if stable_results:
        fastest = min(stable_results, key=lambda r: r["time"])
        summary_text += f"\nFastest STABLE: {fastest['name']}\n"
        summary_text += f"  Time: {fastest['time']:.2f}s\n"
        summary_text += f"  Mass dev: {fastest['mass_dev']:.2e}\n"

    # Compare to baseline
    baseline = results[0]
    summary_text += f"\nBaseline: {baseline['name']}\n"
    summary_text += f"  Time: {baseline['time']:.2f}s\n"
    summary_text += f"  Mass dev: {baseline['mass_dev']:.2e}\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8, verticalalignment="top", family="monospace")

    plt.suptitle("Two-Level Damping: Picard + Anderson Acceleration", fontsize=13, fontweight="bold")

    # Save
    output_file = "two_level_damping_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {output_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
