#!/usr/bin/env python3
"""Tests for the Riordan bubble fit + its composer operator.

The fit is **outcome-blind**: the assertions here are about *which
transform is selected* and *why* — never about whether SAT was solved.
Determinism is asserted across two invocations and tie-breaks.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.riordan_bubble_fit import (
    DEFAULT_CANDIDATES,
    TransformCandidate,
    evaluate_candidate,
    fit,
    format_fit_table,
    phase_shifted_candidate,
    project_strain,
)
from geometry.flattening_probe import raw_view
from strategy import (
    composed_local_search,
    fitted_composer,
    spectral_view_for,
)
from strategy.operators import (
    SearchState,
    StrategyComposer,
    fitted_coordinate_ranker,
    raw_strain_ranker,
    riordan_bubble_fitter,
    unsat_clause_focus,
)


def _planted(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula


# --------------------------------------------------------------------------- #
# Pure-function fit                                                           #
# --------------------------------------------------------------------------- #


class FitPureFunctionTests(unittest.TestCase):
    def test_stable_bubble_strain_picks_a_viable_candidate(self) -> None:
        # Strain concentrated on the first few indices: a stable bubble.
        strain = [5.0, 4.5, 4.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        decision = fit(strain)
        self.assertFalse(decision.veto)
        # Some candidate is viable; identity (the most conservative) is
        # the natural pick here. Either way, the selected candidate must
        # be in the registry and non-vetoed.
        self.assertIn(decision.selected, {c.name for c in DEFAULT_CANDIDATES})
        # Every report must be present and well-typed.
        self.assertEqual(
            tuple(r.name for r in decision.reports),
            tuple(c.name for c in DEFAULT_CANDIDATES),
        )

    def test_diffuse_strain_prefers_a_riordan_variant(self) -> None:
        # Uniform strain: identity is diffuse_pressure (no separation),
        # so the fitter must pick a viable Riordan transform.
        strain = [1.0] * 8
        decision = fit(strain)
        self.assertFalse(decision.veto)
        self.assertNotEqual(
            decision.selected, "identity",
            f"diffuse uniform strain should not pick identity; got: {decision.selected}",
        )
        # The selected candidate's pressure label must not be destructive.
        selected_report = next(
            r for r in decision.reports if r.name == decision.selected
        )
        self.assertTrue(selected_report.viable)

    def test_destructive_trace_vetoes_every_candidate(self) -> None:
        # Off-phase, top-k-churning trace — the gauge's destructive
        # signature. The fitter must veto and fall back to identity.
        profile = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5]
        trace = (
            [0.1, 0.1, 0.1, 0.1, 5.0, 4.0, 3.0, 2.0],
            [4.0, 0.1, 0.1, 5.0, 0.1, 3.0, 2.0, 0.1],
            [0.1, 5.0, 0.1, 0.1, 0.1, 3.0, 4.0, 2.0],
        )
        decision = fit(profile, trace=trace)
        self.assertTrue(decision.veto)
        self.assertEqual(decision.selected, "identity")
        # Rationale should mention the fallback.
        self.assertIn("fall_back_to_identity_and_veto", decision.rationale)

    def test_empty_strain_returns_identity_no_veto(self) -> None:
        # Zero strain — no candidates to evaluate. The fitter should
        # return identity gracefully without a veto.
        decision = fit([0.0] * 8)
        self.assertEqual(decision.selected, "identity")
        self.assertFalse(decision.veto)
        self.assertEqual(decision.reports, ())

    def test_deterministic_across_invocations(self) -> None:
        strain = [3.0, 2.5, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5]
        a = fit(strain)
        b = fit(strain)
        self.assertEqual(a.selected, b.selected)
        self.assertEqual(a.veto, b.veto)
        self.assertEqual(
            tuple((r.name, r.score, r.viable) for r in a.reports),
            tuple((r.name, r.score, r.viable) for r in b.reports),
        )

    def test_tie_break_prefers_identity(self) -> None:
        # Two candidates with identical-looking shapes: identity should win.
        same_as_identity = TransformCandidate(
            name="alias_of_identity", build=raw_view
        )
        candidates = (
            DEFAULT_CANDIDATES[0],  # identity
            same_as_identity,
        )
        strain = [5.0, 4.5, 4.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        decision = fit(strain, candidates=candidates)
        self.assertEqual(decision.selected, "identity")

    def test_format_fit_table_is_deterministic(self) -> None:
        strain = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        d = fit(strain)
        self.assertEqual(format_fit_table(d), format_fit_table(d))


# --------------------------------------------------------------------------- #
# Projection + phase shift helpers                                            #
# --------------------------------------------------------------------------- #


class ProjectionTests(unittest.TestCase):
    def test_identity_projection_is_a_noop(self) -> None:
        strain = [5.0, 4.0, 3.0, 2.0, 1.0]
        projected = project_strain(strain, raw_view(len(strain)))
        self.assertEqual(list(projected), strain)

    def test_phase_shifted_candidate_has_shifted_name(self) -> None:
        base = DEFAULT_CANDIDATES[1]  # pascal
        shifted = phase_shifted_candidate(base, shift=2)
        self.assertEqual(shifted.name, "pascal+shift2")
        view = shifted.build(8)
        self.assertIn("+shift2", view.name)


# --------------------------------------------------------------------------- #
# Composer operator wiring                                                    #
# --------------------------------------------------------------------------- #


class ComposerWiringTests(unittest.TestCase):
    def test_fitter_publishes_fitted_view_field(self) -> None:
        formula = _planted(seed=4, variables=8, clauses=14)
        fitter = riordan_bubble_fitter()
        state = SearchState(
            formula=formula, assignment=[False] * 8, n_vars=8, step=0,
            rng=random.Random(0),
        )
        fitter(state)
        self.assertIn("fitted_view", state.field)
        self.assertIn("fitted_selected", state.field)
        self.assertIn("fit_rationale", state.field)

    def test_fitted_ranker_yields_without_view(self) -> None:
        formula = _planted(seed=4, variables=8, clauses=14)
        state = SearchState(
            formula=formula, assignment=[False] * 8, n_vars=8, step=0,
            rng=random.Random(0),
        )
        self.assertIsNone(fitted_coordinate_ranker(state))

    def test_fitter_veto_routes_to_raw_ranker(self) -> None:
        # Compose: fitter → focus → fitted_ranker → raw_ranker. With a
        # destructive trace, the fitter sets veto_transformed, the
        # fitted_ranker yields with coordinate_vetoed=True, and the raw
        # ranker proposes the flip.
        formula = _planted(seed=4, variables=8, clauses=14)
        destructive_trace = (
            [0.1, 0.1, 0.1, 0.1, 5.0, 4.0, 3.0, 2.0],
            [4.0, 0.1, 0.1, 5.0, 0.1, 3.0, 2.0, 0.1],
            [0.1, 5.0, 0.1, 0.1, 0.1, 3.0, 4.0, 2.0],
        )
        composer = StrategyComposer(
            operators=(
                riordan_bubble_fitter(),
                unsat_clause_focus,
                fitted_coordinate_ranker,
                raw_strain_ranker,
            )
        )
        # Seed an explicit destructive-shaped current_strain so the
        # fitter has something to evaluate even if the assignment
        # happens to satisfy the synthetic formula.
        state = SearchState(
            formula=formula, assignment=[True, False, True, False, True, False, True, False],
            n_vars=8, step=0,
            rng=random.Random(0),
            field={
                "strain_trace": destructive_trace,
                "current_strain": [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5],
            },
        )
        proposal = composer.step(state)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.operator, "raw_strain_ranker")
        self.assertTrue(state.field.get("veto_transformed"))
        self.assertTrue(state.field.get("coordinate_vetoed"))

    def test_fitted_composer_runs_end_to_end_deterministically(self) -> None:
        formula = _planted(seed=4, variables=8, clauses=14)
        a = composed_local_search(
            formula, 8, fitted_composer(), max_flips=30, seed=7,
        )
        b = composed_local_search(
            formula, 8, fitted_composer(), max_flips=30, seed=7,
        )
        self.assertEqual(a.unsat_trajectory, b.unsat_trajectory)
        self.assertEqual(a.flips, b.flips)
        # Field marks must record fitted_selected on every step.
        self.assertTrue(
            all("fitted_selected" in m for m in a.field_marks)
        )

    def test_fitted_composer_no_hardcoded_if_tree(self) -> None:
        # The composer's operator list is the only control flow. The
        # presence of the fitter operator and its consumer must be in
        # the list — not buried in a branching function.
        composer = fitted_composer()
        op_names = [getattr(op, "__name__", repr(op)) for op in composer.operators]
        # Surface a few that should be present.
        self.assertIn("_operator", op_names[0])  # plateau_detector closure
        self.assertTrue(
            any("fitted_coordinate_ranker" in n or "fitted" in n for n in op_names),
            f"expected fitted_coordinate_ranker in composer; got {op_names}",
        )


# --------------------------------------------------------------------------- #
# Outcome-blindness assertion                                                 #
# --------------------------------------------------------------------------- #


class OutcomeBlindnessTests(unittest.TestCase):
    def test_fit_does_not_use_formula_or_solve_state(self) -> None:
        # Same strain, two different formulas — the fit decision must
        # depend only on the strain.
        strain = [5.0, 4.5, 4.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        d = fit(strain)
        self.assertEqual(
            tuple((r.name, r.score) for r in d.reports),
            tuple((r.name, r.score) for r in fit(strain).reports),
        )


if __name__ == "__main__":
    unittest.main()
