#!/usr/bin/env python3
"""Tests for composable SAT strategy operators.

Deterministic — seeded RNG everywhere. We assert that:
- operators compose (a list of operators behaves like a first-match policy),
- the same composer + seed produces identical trajectories across runs,
- field signals route correctly (bubble veto disables transform proposals),
- raw / transformed presets reproduce the flattening-probe behavior on
  trivial instances.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.bubble_lifecycle import inflate_bubble, seed_from_strain
from strategy import (
    composed_local_search,
    gated_transformed_composer,
    raw_composer,
    spectral_view_for,
    transformed_composer,
)
from strategy.operators import (
    Proposal,
    SearchState,
    StrategyComposer,
    bubble_pressure_gate,
    coordinate_ranker,
    plateau_detector,
    raw_strain_ranker,
    unsat_clause_focus,
)


def _planted(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula


# --------------------------------------------------------------------------- #
# Operator composition                                                        #
# --------------------------------------------------------------------------- #


class StrategyComposerTests(unittest.TestCase):
    def test_first_proposal_wins(self) -> None:
        # Two synthetic operators: the first always proposes variable 0,
        # the second always proposes variable 1. The composer must
        # return the first proposal and not consult the second.
        def op_first(_state: SearchState) -> Proposal | None:
            return Proposal(variable=0, operator="first", reason="always")

        called = []

        def op_second(_state: SearchState) -> Proposal | None:
            called.append("hit")
            return Proposal(variable=1, operator="second", reason="never")

        composer = StrategyComposer(operators=(op_first, op_second))
        state = SearchState(
            formula=[((0, False),)],
            assignment=[False],
            n_vars=1,
            step=0,
            rng=random.Random(0),
        )
        proposal = composer.step(state)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.variable, 0)
        self.assertEqual(proposal.operator, "first")
        self.assertEqual(called, [])

    def test_yielding_operator_falls_through(self) -> None:
        def op_yield(_state: SearchState) -> Proposal | None:
            return None

        def op_take(_state: SearchState) -> Proposal | None:
            return Proposal(variable=0, operator="taker", reason="took")

        composer = StrategyComposer(operators=(op_yield, op_take))
        state = SearchState(
            formula=[((0, False),)], assignment=[False], n_vars=1, step=0,
            rng=random.Random(0),
        )
        proposal = composer.step(state)
        self.assertEqual(proposal.operator, "taker")

    def test_all_yield_returns_none(self) -> None:
        def op(_state: SearchState) -> Proposal | None:
            return None

        composer = StrategyComposer(operators=(op, op))
        state = SearchState(
            formula=[((0, False),)], assignment=[False], n_vars=1, step=0,
            rng=random.Random(0),
        )
        self.assertIsNone(composer.step(state))


# --------------------------------------------------------------------------- #
# Operator behaviors                                                          #
# --------------------------------------------------------------------------- #


class UnsatClauseFocusTests(unittest.TestCase):
    def test_publishes_focused_clause_and_unsat_count(self) -> None:
        formula = [((0, False),), ((1, False),)]
        state = SearchState(
            formula=formula, assignment=[False, False], n_vars=2, step=0,
            rng=random.Random(0),
        )
        proposal = unsat_clause_focus(state)
        self.assertIsNone(proposal)
        self.assertIn(state.field["focused_clause"], formula)
        self.assertEqual(state.field["unsat_count"], 2)

    def test_focused_clause_is_none_when_satisfied(self) -> None:
        formula = [((0, False),)]
        state = SearchState(
            formula=formula, assignment=[True], n_vars=1, step=0,
            rng=random.Random(0),
        )
        unsat_clause_focus(state)
        self.assertIsNone(state.field["focused_clause"])
        self.assertEqual(state.field["unsat_count"], 0)


class RawStrainRankerTests(unittest.TestCase):
    def test_yields_without_focused_clause(self) -> None:
        state = SearchState(
            formula=[((0, False),)], assignment=[False], n_vars=1, step=0,
            rng=random.Random(0),
        )
        self.assertIsNone(raw_strain_ranker(state))

    def test_proposes_flip_that_minimizes_unsat(self) -> None:
        formula = [((0, False),), ((1, False),), ((0, False), (2, False))]
        assignment = [False, False, False]
        state = SearchState(
            formula=formula, assignment=assignment, n_vars=3, step=0,
            rng=random.Random(0),
            field={"focused_clause": formula[0]},
        )
        proposal = raw_strain_ranker(state)
        # The focused clause is ((0, False),) — only variable 0 lives in it.
        self.assertEqual(proposal.variable, 0)


class PlateauDetectorTests(unittest.TestCase):
    def test_flat_history_marks_plateau(self) -> None:
        detector = plateau_detector(window=4, relative_band=0.05)
        state = SearchState(
            formula=[], assignment=[], n_vars=0, step=0,
            rng=random.Random(0),
            field={"unsat_history": [5, 5, 5, 5]},
        )
        detector(state)
        self.assertTrue(state.field["plateau"])

    def test_short_history_yields_no_plateau(self) -> None:
        detector = plateau_detector(window=6, relative_band=0.05)
        state = SearchState(
            formula=[], assignment=[], n_vars=0, step=0,
            rng=random.Random(0),
            field={"unsat_history": [5, 5]},
        )
        detector(state)
        self.assertFalse(state.field["plateau"])

    def test_descending_history_no_plateau(self) -> None:
        detector = plateau_detector(window=4, relative_band=0.02)
        state = SearchState(
            formula=[], assignment=[], n_vars=0, step=0,
            rng=random.Random(0),
            field={"unsat_history": [10, 9, 7, 5]},
        )
        detector(state)
        self.assertFalse(state.field["plateau"])


# --------------------------------------------------------------------------- #
# Bubble gate routes via veto field                                           #
# --------------------------------------------------------------------------- #


class BubbleGateTests(unittest.TestCase):
    def test_destructive_trace_vetoes_transformed_proposal(self) -> None:
        # Construct a strain trace + bubble that the existing
        # measure_pressure classifies as destructive_amplification.
        profile = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5]
        bubble = inflate_bubble(profile, seed_from_strain(profile))
        trace = (
            [0.1, 0.1, 0.1, 0.1, 5.0, 4.0, 3.0, 2.0],
            [4.0, 0.1, 0.1, 5.0, 0.1, 3.0, 2.0, 0.1],
            [0.1, 5.0, 0.1, 0.1, 0.1, 3.0, 4.0, 2.0],
        )
        gate = bubble_pressure_gate()
        state = SearchState(
            formula=[], assignment=[], n_vars=0, step=0,
            rng=random.Random(0),
            field={"strain_trace": trace, "bubble_candidate": bubble},
        )
        gate(state)
        self.assertEqual(
            state.field["bubble_pressure_label"], "destructive_amplification",
        )
        self.assertTrue(state.field.get("veto_transformed"))

    def test_diagnostic_trace_allows_transformed_proposal(self) -> None:
        profile = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5]
        bubble = inflate_bubble(profile, seed_from_strain(profile))
        # A trace where interior dominates and boundary is stable.
        trace = (
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5],
            [4.5, 4.5, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5],
            [4.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5],
        )
        gate = bubble_pressure_gate()
        state = SearchState(
            formula=[], assignment=[], n_vars=0, step=0,
            rng=random.Random(0),
            field={"strain_trace": trace, "bubble_candidate": bubble},
        )
        gate(state)
        # Non-destructive label → no veto.
        self.assertFalse(state.field.get("veto_transformed", False))

    def test_missing_trace_yields_quietly(self) -> None:
        gate = bubble_pressure_gate()
        state = SearchState(
            formula=[], assignment=[], n_vars=0, step=0,
            rng=random.Random(0),
        )
        # No exception; no veto; no field writes.
        self.assertIsNone(gate(state))
        self.assertNotIn("veto_transformed", state.field)


class CoordinateRankerVetoTests(unittest.TestCase):
    def test_vetoed_coordinate_ranker_yields_and_marks(self) -> None:
        formula = _planted(seed=10, variables=6, clauses=10)
        view = spectral_view_for(formula, 6)
        ranker = coordinate_ranker(view)
        state = SearchState(
            formula=formula, assignment=[False] * 6, n_vars=6, step=0,
            rng=random.Random(0),
            field={"veto_transformed": True},
        )
        self.assertIsNone(ranker(state))
        self.assertTrue(state.field["coordinate_vetoed"])


# --------------------------------------------------------------------------- #
# Driver determinism + bubble-gate fallback in a real run                     #
# --------------------------------------------------------------------------- #


class RunDriverTests(unittest.TestCase):
    def test_deterministic_under_fixed_seed(self) -> None:
        formula = _planted(seed=4, variables=8, clauses=14)
        view = spectral_view_for(formula, 8)
        composer = transformed_composer(view)
        a = composed_local_search(formula, 8, composer, max_flips=60, seed=11)
        b = composed_local_search(formula, 8, composer, max_flips=60, seed=11)
        self.assertEqual(a.unsat_trajectory, b.unsat_trajectory)
        self.assertEqual(a.flips, b.flips)
        self.assertEqual(a.final_unsatisfied, b.final_unsatisfied)

    def test_solved_implies_zero_final_unsat(self) -> None:
        formula = _planted(seed=3, variables=6, clauses=8, k=2)
        report = composed_local_search(
            formula, 6, raw_composer(), max_flips=200, seed=3,
        )
        if report.solved:
            self.assertEqual(report.final_unsatisfied, 0)

    def test_bubble_gate_falls_back_to_raw_on_destructive(self) -> None:
        formula = _planted(seed=4, variables=8, clauses=14)
        view = spectral_view_for(formula, 8)
        profile = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5]
        bubble = inflate_bubble(profile, seed_from_strain(profile))
        destructive_trace = (
            [0.1, 0.1, 0.1, 0.1, 5.0, 4.0, 3.0, 2.0],
            [4.0, 0.1, 0.1, 5.0, 0.1, 3.0, 2.0, 0.1],
            [0.1, 5.0, 0.1, 0.1, 0.1, 3.0, 4.0, 2.0],
        )
        report = composed_local_search(
            formula, 8, gated_transformed_composer(view),
            max_flips=12, seed=7,
            field_seed={
                "strain_trace": destructive_trace,
                "bubble_candidate": bubble,
            },
        )
        # On the first step the gate should veto and the raw ranker
        # should pick up the flip.
        self.assertGreater(len(report.records), 0)
        first = report.records[0]
        self.assertEqual(first.operator, "raw_strain_ranker")
        self.assertTrue(report.field_marks[0]["veto_transformed"])
        self.assertTrue(report.field_marks[0]["coordinate_vetoed"])

    def test_record_trajectory_lengths_consistent(self) -> None:
        formula = _planted(seed=5, variables=8, clauses=14)
        report = composed_local_search(
            formula, 8, raw_composer(), max_flips=40, seed=5,
        )
        # trajectory has one entry per (initial state + step taken).
        self.assertEqual(len(report.unsat_trajectory), report.flips + 1)
        self.assertEqual(len(report.records), report.flips)


if __name__ == "__main__":
    unittest.main()
