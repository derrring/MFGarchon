#!/usr/bin/env python3
"""
Test Anderson Acceleration for Mass Conservation.

Compare standard damped iteration vs Anderson acceleration.
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


def run_solver(use_anderson: bool = False, backend: str | None = None):
    """Run solver with or without Anderson acceleration."""
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

    # MFG solver with optional Anderson
    mfg_solver = FixedPointIterator(
        problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        relaxation=0.5,
        backend=backend,
        use_anderson=use_anderson,
        anderson_depth=5,
        anderson_beta=1.0,
    )

    method_name = "Anderson" if use_anderson else "Damped"
    backend_name = backend or "numpy"
    print(f"\n{'=' * 80}")
    print(f"Testing: {method_name} with {backend_name} backend")
    print(f"{'=' * 80}\n")

    # Run solver
    start_time = time.time()
    try:
        result = mfg_solver.solve(
            max_iterations=30,
            tolerance=1e-3,
            verbose=False,  # Suppress iteration output
        )
        U, M = result[:2]
        converged = True
    except Exception as e:
        print(f"Exception (expected for stochastic): {str(e)[:80]}...")
        U = mfg_solver.U if hasattr(mfg_solver, "U") else np.zeros((26, 26))
        M = mfg_solver.M if hasattr(mfg_solver, "M") else problem.m_initial
        converged = False

    elapsed_time = time.time() - start_time

    # Get convergence history
    err_u = mfg_solver.l2distu_rel if hasattr(mfg_solver, "l2distu_rel") else []
    err_m = mfg_solver.l2distm_rel if hasattr(mfg_solver, "l2distm_rel") else []
    iterations = mfg_solver.iterations_run if hasattr(mfg_solver, "iterations_run") else 0

    # Mass conservation
    dx = problem.geometry.get_grid_spacing()[1]  # Spatial spacing
    Nt_points = problem.geometry.get_grid_shape()[0]  # Temporal grid points
    masses = np.array([float(np.trapezoid(M[t, :], dx=dx)) for t in range(Nt_points)])

    print(f"Iterations: {iterations}")
    print(f"Time: {elapsed_time:.2f}s ({elapsed_time / iterations:.3f}s/iter)")
    print(f"Final error U: {err_u[-1]:.2e}" if len(err_u) > 0 else "N/A")
    print(f"Final error M: {err_m[-1]:.2e}" if len(err_m) > 0 else "N/A")
    print(f"Mass deviation: {np.max(np.abs(masses - masses[0])):.2e}")

    return {
        "method": method_name,
        "backend": backend_name,
        "U": U,
        "M": M,
        "err_u": err_u,
        "err_m": err_m,
        "masses": masses,
        "iterations": iterations,
        "time": elapsed_time,
        "converged": converged,
    }


def main():
    """Compare Anderson vs standard damping."""
    print("\n" + "=" * 80)
    print("ANDERSON ACCELERATION TEST FOR MASS CONSERVATION")
    print("=" * 80)

    # Test configurations
    configs = [
        {"use_anderson": False, "backend": None},  # Baseline: damped + numpy
        {"use_anderson": True, "backend": None},  # Anderson + numpy
        {"use_anderson": True, "backend": "numba"},  # Anderson + numba (if available)
    ]

    results = []
    for config in configs:
        try:
            result = run_solver(**config)
            results.append(result)
        except Exception as e:
            print(f"Skipped {config}: {e}")

    if len(results) < 2:
        print("\nNeed at least 2 results for comparison!")
        return

    # Create comparison visualization
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot convergence comparison
    ax_conv = fig.add_subplot(gs[0, :])
    for res in results:
        label = f"{res['method']} ({res['backend']})"
        iters = range(1, len(res["err_u"]) + 1)
        ax_conv.semilogy(iters, res["err_u"], "o-", label=f"{label} - U", markersize=3)
        ax_conv.semilogy(iters, res["err_m"], "s-", label=f"{label} - M", markersize=3)

    ax_conv.axhline(y=1e-3, color="r", linestyle="--", alpha=0.5, label="Tolerance")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Relative Error")
    ax_conv.set_title("Convergence Comparison: Anderson vs Damped")
    ax_conv.legend(fontsize=8, ncol=2)
    ax_conv.grid(True, alpha=0.3)

    # Plot mass conservation for each method
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        t_steps = np.arange(len(res["masses"])) * (1.0 / 25)
        ax.plot(t_steps, res["masses"], "b-", linewidth=2)
        ax.axhline(y=res["masses"][0], color="r", linestyle="--", alpha=0.7)
        ax.fill_between(t_steps, res["masses"][0] - 0.02, res["masses"][0] + 0.02, alpha=0.2, color="gray")
        ax.set_xlabel("Time t")
        ax.set_ylabel("Total Mass")
        ax.set_title(f"{res['method']} ({res['backend']})\nMass Conservation")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.95, 1.05])

    # Plot density for each method
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(res["M"], aspect="auto", origin="lower", extent=[0, 1, 0, 1], cmap="plasma")
        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        ax.set_title("Density m(t,x)")
        plt.colorbar(im, ax=ax)

    plt.suptitle("Anderson Acceleration vs Standard Damping", fontsize=14, fontweight="bold")

    # Save
    output_file = "anderson_acceleration_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for res in results:
        speedup = results[0]["time"] / res["time"] if res["time"] > 0 else 1.0
        print(f"\n{res['method']} ({res['backend']}):")
        print(f"  Iterations: {res['iterations']}")
        print(f"  Time: {res['time']:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Mass deviation: {np.max(np.abs(res['masses'] - res['masses'][0])):.2e}")
        print(f"  Converged: {'Yes' if res['converged'] else 'No'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
