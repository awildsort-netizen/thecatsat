"""Tests for the sat_metabolism helper module."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_metabolism as sm


class HelperTests(unittest.TestCase):
    def test_spins_to_assignment(self) -> None:
        self.assertEqual(
            sm.spins_to_assignment([0.5, -0.1, 0.0, -2.0]),
            (True, False, True, False),
        )

    def test_hamming_matches(self) -> None:
        self.assertEqual(sm.hamming([True, False, True], [True, False, True]), 0)
        self.assertEqual(sm.hamming([True, False, True], [False, False, True]), 1)
        self.assertEqual(sm.hamming([True, True, True], [False, False, False]), 3)

    def test_hamming_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            sm.hamming([True], [True, False])

    def test_assignment_hamming_movement(self) -> None:
        seq = [
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, False),
        ]
        self.assertEqual(sm.assignment_hamming_movement(seq), [1, 1, 0])

    def test_assignment_hamming_movement_empty(self) -> None:
        self.assertEqual(sm.assignment_hamming_movement([]), [])
        self.assertEqual(sm.assignment_hamming_movement([(True, False)]), [])

    def test_distance_delta_per_step(self) -> None:
        # series goes 10 -> 8 -> 9 -> 6. deltas: +2, -1, +3 (positive = progress).
        self.assertEqual(sm.distance_delta_per_step([10, 8, 9, 6]), [2, -1, 3])

    def test_unsat_clause_revisit_count(self) -> None:
        # values: 5 5 4 5 4 3. counts: {5:3, 4:2, 3:1}. revisits: 2+1 = 3.
        self.assertEqual(sm.unsat_clause_revisit_count([5, 5, 4, 5, 4, 3]), 3)
        self.assertEqual(sm.unsat_clause_revisit_count([5, 4, 3, 2]), 0)
        self.assertEqual(sm.unsat_clause_revisit_count([]), 0)

    def test_operator_gene_entropy(self) -> None:
        self.assertEqual(sm.operator_gene_entropy([]), 0.0)
        self.assertEqual(sm.operator_gene_entropy(["a"] * 5), 0.0)
        # two equally-frequent symbols -> 1 bit.
        self.assertAlmostEqual(
            sm.operator_gene_entropy(["a", "b", "a", "b"]), 1.0
        )
        # four equally-frequent symbols -> 2 bits.
        self.assertAlmostEqual(
            sm.operator_gene_entropy(["a", "b", "c", "d"]), 2.0
        )

    def test_motif_reuse_count(self) -> None:
        # motif (a,b,c) appears at positions 0 and 3 -> 1 reuse.
        names = ["a", "b", "c", "a", "b", "c"]
        self.assertEqual(sm.motif_reuse_count(names, motif_size=3), 1)
        # singletons: "a" appears twice, "b" twice, "c" twice -> 3 reuses total.
        self.assertEqual(sm.motif_reuse_count(names, motif_size=1), 3)
        self.assertEqual(sm.motif_reuse_count(["a", "b"], motif_size=3), 0)
        self.assertEqual(sm.motif_reuse_count([], motif_size=3), 0)
        self.assertEqual(sm.motif_reuse_count(["a", "b", "c"], motif_size=0), 0)

    def test_shortest_observed_prefix_to_improvement(self) -> None:
        # first strict drop below 10 is at index 2.
        self.assertEqual(
            sm.shortest_observed_prefix_to_improvement([10, 10, 9, 8]), 2
        )
        # no improvement -> None.
        self.assertIsNone(
            sm.shortest_observed_prefix_to_improvement([10, 11, 12])
        )
        # plateau at start -> None.
        self.assertIsNone(
            sm.shortest_observed_prefix_to_improvement([5, 5, 5])
        )
        self.assertIsNone(sm.shortest_observed_prefix_to_improvement([]))

    def test_distance_paid(self) -> None:
        movements = [2, 1, 3]
        deltas = [1, 2, 0]  # net = 3 resolved, total movement = 6 -> 2.0 paid.
        self.assertEqual(
            sm.distance_paid_per_incompatibility_resolved(movements, deltas),
            2.0,
        )

    def test_distance_paid_no_net_progress(self) -> None:
        self.assertIsNone(
            sm.distance_paid_per_incompatibility_resolved([5, 5], [-1, 1])
        )
        self.assertIsNone(
            sm.distance_paid_per_incompatibility_resolved([], [])
        )

    def test_active_operators_at_step(self) -> None:
        class FakeTrace:
            def __init__(self, t: int, op: str, active: bool):
                self.t = t
                self.operator = op
                self.active = active

        traces = [
            FakeTrace(0, "alpha", True),
            FakeTrace(0, "beta", False),
            FakeTrace(0, "gamma", True),
            FakeTrace(1, "alpha", True),
        ]
        self.assertEqual(
            sm.active_operators_at_step(traces, 0), ["alpha", "gamma"]
        )
        self.assertEqual(
            sm.active_operators_at_step(traces, 1), ["alpha"]
        )


class ConcentrationClimateActivationTests(unittest.TestCase):
    """OperatorTrace.active should follow the concentration climate.

    Under the baseline policy, the excitable_* operators used to be
    flat-off because the policy name didn't license them. With the
    activation-factor change in sat_composer, an enriched concentration
    channel (any channel above the uniform-prior floor) decompresses
    those traces into the `active` state even on the baseline policy.
    """

    def test_baseline_policy_activates_under_concentration_climate(self) -> None:
        import random

        import sat_furnace

        rng = random.Random(11)
        formula, planted = sat_furnace.generate_formula(
            "sat",
            variables=8,
            clauses=24,
            clause_size=3,
            rng=rng,
        )

        result = sat_furnace.run_furnace(
            formula=formula,
            variables=8,
            steps=20,
            rng=rng,
            temperature=0.85,
            learning_rate=0.18,
            inertia=0.5,
            noise=0.05,
            planted_assignment=planted,
            adaptive=False,
            memory_decay=0.92,
            memory_drive=0.12,
            policy="baseline",
        )

        excitable_traces = [
            trace
            for trace in result.operator_traces
            if trace.operator == "excitable_concentration"
        ]
        self.assertGreater(len(excitable_traces), 0)
        # At least one step should have an enriched concentration
        # climate and therefore an active excitable_concentration
        # trace. (Step 0 starts uniform, so we accept any-step.)
        self.assertTrue(
            any(trace.active for trace in excitable_traces),
            "excitable_concentration should activate under enriched "
            "climate even on baseline policy",
        )


if __name__ == "__main__":
    unittest.main()
