"""Removal-gate tests for FixedPointIterator damping_* deprecation (Issue #1070).

Pre-v0.25.0, FixedPointIterator accepted 7 legacy `damping_*` ctor kwargs and
exposed 7 read-only `damping_*` `@property` aliases on instances, all
redirecting silently to the canonical `relaxation_*` names. v0.25.0 removed
them per the 3-version deprecation window.

Removal is enforced via `mfgarchon.utils.deprecation.validate_kwargs` with a
class-level `_REMOVED_KWARGS` migration map (matching the established
`MFGProblem._validate_kwargs` pattern). Passing any removed kwarg raises
`ValueError` with a curated "Use 'X' instead" message; reading the legacy
attribute names raises `AttributeError`. Pre-v0.25.0 this file tested the
equivalence of the legacy redirects; it has been rewritten to lock in the
post-removal behaviour and prevent silent reintroduction.
"""

from __future__ import annotations

import pytest

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem


def _make_test_problem():
    """Minimal MFG problem for iterator construction tests."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(Nx=11, xmin=0.0, xmax=1.0, T=0.2, Nt=5, sigma=0.3, components=components)


@pytest.fixture
def solvers():
    """Build a (problem, hjb, fp) triple for iterator construction."""
    problem = _make_test_problem()
    return problem, HJBFDMSolver(problem), FPFDMSolver(problem)


# The 7 deprecated kwargs removed in v0.25.0 alongside their canonical names.
LEGACY_KWARGS_REMOVED = [
    ("damping_factor", "relaxation"),
    ("damping_factor_M", "relaxation_M"),
    ("adaptive_damping", "adaptive_relaxation"),
    ("adaptive_damping_decay", "adaptive_relaxation_decay"),
    ("adaptive_damping_min", "adaptive_relaxation_min"),
    ("damping_schedule", "relaxation_schedule"),
    ("damping_schedule_M", "relaxation_schedule_M"),
]


class TestDeprecatedKwargsRaiseValueError:
    """Locked-in v0.25.0 removal: legacy kwargs are rejected via validate_kwargs."""

    @pytest.mark.parametrize(("legacy_name", "canonical_name"), LEGACY_KWARGS_REMOVED)
    def test_legacy_kwarg_raises_value_error(self, solvers, legacy_name, canonical_name):
        """Each removed kwarg raises `ValueError` with a curated migration message."""
        problem, hjb, fp = solvers
        with pytest.raises(ValueError, match=r"Deprecated kwargs detected in FixedPointIterator"):
            FixedPointIterator(problem, hjb, fp, **{legacy_name: 0.5})

    @pytest.mark.parametrize(("legacy_name", "canonical_name"), LEGACY_KWARGS_REMOVED)
    def test_legacy_kwarg_error_names_canonical_replacement(self, solvers, legacy_name, canonical_name):
        """The error message tells the user which canonical name to use."""
        problem, hjb, fp = solvers
        with pytest.raises(ValueError, match=rf"'{legacy_name}'.*'{canonical_name}'"):
            FixedPointIterator(problem, hjb, fp, **{legacy_name: 0.5})

    def test_canonical_kwargs_still_accepted(self, solvers):
        """The canonical names (which replaced the removed legacy ones) still work."""
        problem, hjb, fp = solvers
        iter_obj = FixedPointIterator(
            problem,
            hjb,
            fp,
            relaxation=0.4,
            relaxation_M=0.3,
            adaptive_relaxation=True,
            adaptive_relaxation_decay=0.8,
            adaptive_relaxation_min=0.01,
            relaxation_schedule="harmonic",
            relaxation_schedule_M="sqrt",
        )
        assert iter_obj.relaxation == 0.4
        assert iter_obj.relaxation_M == 0.3
        assert iter_obj.adaptive_relaxation is True
        assert iter_obj.adaptive_relaxation_decay == 0.8
        assert iter_obj.adaptive_relaxation_min == 0.01
        assert iter_obj.relaxation_schedule == "harmonic"
        assert iter_obj.relaxation_schedule_M == "sqrt"


class TestDeprecatedAttributesRaiseAttributeError:
    """Locked-in v0.25.0 removal: legacy `iter.damping_*` properties no longer exist."""

    @pytest.mark.parametrize(("legacy_name", "_canonical_name"), LEGACY_KWARGS_REMOVED)
    def test_legacy_attribute_raises_attribute_error(self, solvers, legacy_name, _canonical_name):
        """Each removed @property alias raises `AttributeError` on read."""
        problem, hjb, fp = solvers
        iter_obj = FixedPointIterator(problem, hjb, fp, relaxation=0.5)
        with pytest.raises(AttributeError, match=rf"{legacy_name}"):
            getattr(iter_obj, legacy_name)
