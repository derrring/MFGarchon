"""
HJB Solvers for Numerical Methods.

This module contains Hamilton-Jacobi-Bellman equation solvers using classical
numerical analysis approaches:

- BaseHJBSolver: Abstract base class for all HJB solvers
- HJBFDMSolver: Finite difference method (all dimensions: 1D, 2D, 3D, nD)
- HJBGFDMSolver: Generalized finite difference method (meshfree, nD)
- HJBHowardSolver: Howard's policy iteration inner solver, peer to GFDM Newton
  inner (resolves Issue #1118 Newton stiffness; requires SOCP-precomputed
  stencils on a stencil_provider HJBGFDMSolver)
- HJBSemiLagrangianSolver: Semi-Lagrangian approach (characteristic-based, nD)
- HJBWenoSolver: WENO (Weighted Essentially Non-Oscillatory) method (1D/2D/3D)
- PenaltyHJBSolver: Variational inequality wrapper (obstacle/optimal stopping)

All solvers inherit from BaseNumericalSolver and follow the new paradigm structure.
"""

from .base_hjb import BaseHJBSolver
from .hjb_fdm import ConvergenceError, HJBFDMSolver
from .hjb_gfdm import HJBGFDMSolver
from .hjb_howard import HJBHowardSolver
from .hjb_penalty import PenaltyHJBSolver
from .hjb_semi_lagrangian import HJBSemiLagrangianSolver
from .hjb_weno import HJBWenoSolver

__all__ = [
    "BaseHJBSolver",
    "ConvergenceError",
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBHowardSolver",
    "HJBSemiLagrangianSolver",
    "HJBWenoSolver",
    "PenaltyHJBSolver",
]
