#!/usr/bin/env python3
"""Tests for ``concentration``: biased sampling over eligible providers."""

from __future__ import annotations

import random
import unittest
from collections import Counter

from concentration import (
    concentration_from_gene_tokens,
    run_many,
    run_rounds,
    sample_path,
    sample_provider,
)


class TestSampleProvider(unittest.TestCase):
    def test_seeded_sampling_is_deterministic(self):
        eligible = ("a", "b", "c", "d")
        field = {"a": 1.0, "b": 4.0, "c": 1.0, "d": 1.0}
        rng_one = random.Random(42)
        rng_two = random.Random(42)
        picks_one = [sample_provider(eligible, field, rng_one) for _ in range(20)]
        picks_two = [sample_provider(eligible, field, rng_two) for _ in range(20)]
        self.assertEqual(picks_one, picks_two)

    def test_unbiased_field_is_uniform_ish(self):
        eligible = ("a", "b", "c", "d")
        rng = random.Random(0)
        counts = {name: 0 for name in eligible}
        for _ in range(4000):
            counts[sample_provider(eligible, {}, rng)] += 1
        for name in eligible:
            self.assertGreater(counts[name], 800)
            self.assertLess(counts[name], 1200)

    def test_concentration_does_not_override_eligibility(self):
        eligible = ("a", "b")
        field = {"ghost": 1000.0, "a": 1.0, "b": 1.0}
        rng = random.Random(1)
        chosen = {sample_provider(eligible, field, rng) for _ in range(500)}
        self.assertEqual(chosen, {"a", "b"})
        self.assertNotIn("ghost", chosen)

    def test_concentration_shifts_distribution_among_eligibles(self):
        eligible = ("a", "b")
        rng_flat = random.Random(7)
        rng_warm = random.Random(7)
        flat = [sample_provider(eligible, {}, rng_flat) for _ in range(2000)]
        warm = [sample_provider(eligible, {"a": 9.0, "b": 1.0}, rng_warm) for _ in range(2000)]
        flat_a = flat.count("a")
        warm_a = warm.count("a")
        # flat ~ 1000, warm ~ 1800. Loose bounds, far enough apart to be safe.
        self.assertLess(flat_a, 1200)
        self.assertGreater(warm_a, 1600)
        self.assertGreater(warm_a, flat_a + 400)

    def test_all_zero_weights_fall_back_to_uniform(self):
        eligible = ("a", "b", "c")
        field = {"a": 0.0, "b": 0.0, "c": 0.0}
        rng = random.Random(123)
        chosen = {sample_provider(eligible, field, rng) for _ in range(300)}
        self.assertEqual(chosen, {"a", "b", "c"})

    def test_empty_eligible_raises(self):
        rng = random.Random(0)
        with self.assertRaises(ValueError):
            sample_provider((), {"a": 1.0}, rng)

    def test_default_weight_lets_unlisted_providers_compete(self):
        eligible = ("a", "b", "c")
        rng = random.Random(2026)
        counts = {name: 0 for name in eligible}
        # Only "a" listed; b and c get default_weight=1.0
        for _ in range(3000):
            counts[sample_provider(eligible, {"a": 1.0}, rng)] += 1
        # All three should be roughly equal — listing one at the default
        # value should not turn the others off.
        for name in eligible:
            self.assertGreater(counts[name], 700)


class TestSamplePath(unittest.TestCase):
    def test_path_records_chosen_eligible_and_gene_token(self):
        rng = random.Random(0)
        steps = [("decision", ("a", "b")), ("score", ("x", "y"))]
        path = sample_path(steps, {"a": 5.0, "x": 5.0}, rng)
        self.assertEqual(len(path.steps), 2)
        self.assertEqual(path.steps[0].target, "decision")
        self.assertEqual(path.steps[0].eligible, ("a", "b"))
        self.assertIn(path.steps[0].chosen, ("a", "b"))
        self.assertEqual(path.steps[0].gene_token, f"L:{path.steps[0].chosen}")
        self.assertEqual(path.gene_tokens, tuple(s.gene_token for s in path.steps))
        self.assertEqual(path.signature, tuple(s.chosen for s in path.steps))


class TestRunMany(unittest.TestCase):
    def test_seeded_run_many_is_reproducible(self):
        steps = [("t", ("a", "b", "c"))]
        rng_a = random.Random(99)
        rng_b = random.Random(99)
        _, dist_a = run_many(500, steps, {"a": 3.0}, rng_a)
        _, dist_b = run_many(500, steps, {"a": 3.0}, rng_b)
        self.assertEqual(dist_a, dist_b)

    def test_eligibility_holds_under_many_trials(self):
        steps = [("t", ("a", "b"))]
        field = {"a": 1.0, "b": 1.0, "c": 10000.0}
        rng = random.Random(0)
        paths, dist = run_many(2000, steps, field, rng)
        chosen_names = {step.chosen for path in paths for step in path.steps}
        self.assertEqual(chosen_names, {"a", "b"})
        self.assertNotIn(("c",), dist)

    def test_different_fields_yield_different_distributions(self):
        steps = [("t", ("a", "b"))]
        rng_flat = random.Random(11)
        rng_warm = random.Random(11)
        _, flat = run_many(2000, steps, {}, rng_flat)
        _, warm = run_many(2000, steps, {"a": 9.0}, rng_warm)
        self.assertNotEqual(flat, warm)
        self.assertGreater(warm[("a",)], flat[("a",)])

    def test_trial_count_zero_is_empty(self):
        rng = random.Random(0)
        paths, dist = run_many(0, [("t", ("a",))], {}, rng)
        self.assertEqual(paths, ())
        self.assertEqual(dist.total(), 0)

    def test_negative_trials_raises(self):
        rng = random.Random(0)
        with self.assertRaises(ValueError):
            run_many(-1, [("t", ("a",))], {}, rng)


class TestConcentrationFromGeneTokens(unittest.TestCase):
    def test_literal_warms_corresponding_provider(self):
        field = concentration_from_gene_tokens(["L:foo", "L:bar", "E"])
        self.assertIn("foo", field)
        self.assertIn("bar", field)
        self.assertGreater(field["foo"], 0.0)
        self.assertGreater(field["bar"], 0.0)

    def test_repeated_literals_accumulate_evidence(self):
        field = concentration_from_gene_tokens(["L:foo", "L:foo", "L:foo", "L:bar", "E"])
        # foo seen 3x, bar seen 1x — foo must be strictly heavier.
        self.assertGreater(field["foo"], field["bar"])

    def test_non_literal_tokens_create_no_provider_weights(self):
        field = concentration_from_gene_tokens(
            ["A:carry", "W:hot", "R", "D:1:foo,bar", "E"]
        )
        # No L: tokens were emitted, no M: expanded. No providers.
        self.assertEqual(field, {})

    def test_end_token_does_not_create_provider(self):
        field = concentration_from_gene_tokens(["L:foo", "E"])
        self.assertEqual(set(field), {"foo"})
        self.assertNotIn("E", field)

    def test_motif_expansion_warms_through_backreference(self):
        field = concentration_from_gene_tokens(
            ["D:1:foo,bar", "M:1", "L:foo", "E"]
        )
        # M:1 expands to L:foo, L:bar. Then L:foo. So foo seen 2x, bar 1x.
        self.assertIn("foo", field)
        self.assertIn("bar", field)
        self.assertGreater(field["foo"], field["bar"])

    def test_window_scale_amplifies_literals_inside_window(self):
        tokens = ["L:outside", "W:hot", "L:inside", "R", "E"]
        flat = concentration_from_gene_tokens(tokens)
        scaled = concentration_from_gene_tokens(
            tokens, window_scale={"hot": 5.0}
        )
        # Outside the window the weight is unchanged.
        self.assertEqual(flat["outside"], scaled["outside"])
        # Inside the window the literal is amplified.
        self.assertGreater(scaled["inside"], flat["inside"])

    def test_window_scale_does_not_invent_providers(self):
        # A window with no literals inside it leaves no trace in the field.
        field = concentration_from_gene_tokens(
            ["W:hot", "R", "E"], window_scale={"hot": 100.0}
        )
        self.assertEqual(field, {})

    def test_base_and_bump_are_independent_knobs(self):
        # base=10 with bump=0 means seen-once-or-many literals get the same
        # weight: pure presence/absence flag.
        field = concentration_from_gene_tokens(
            ["L:foo", "L:foo", "L:bar", "E"], base=10.0, bump=0.0
        )
        self.assertEqual(field["foo"], field["bar"])
        self.assertEqual(field["foo"], 10.0)

    def test_negative_base_or_bump_raises(self):
        with self.assertRaises(ValueError):
            concentration_from_gene_tokens(["L:foo", "E"], base=-1.0)
        with self.assertRaises(ValueError):
            concentration_from_gene_tokens(["L:foo", "E"], bump=-1.0)

    def test_field_feeds_sample_provider_without_overriding_eligibility(self):
        # Build a field that strongly warms a name that is NOT in the
        # eligible set. The eligibility rule must still hold.
        field = concentration_from_gene_tokens(
            ["L:foo", "L:foo", "L:foo", "L:foo", "L:foo", "E"]
        )
        eligible = ("a", "b")
        rng = random.Random(0)
        chosen = {sample_provider(eligible, field, rng) for _ in range(500)}
        self.assertEqual(chosen, {"a", "b"})
        self.assertNotIn("foo", chosen)

    def test_field_shifts_run_many_distribution_among_eligibles(self):
        # A gene trace heavy on "a" should bias the sampler toward "a"
        # when both "a" and "b" are eligible.
        field = concentration_from_gene_tokens(
            ["L:a", "L:a", "L:a", "L:a", "L:a", "L:b", "E"]
        )
        steps = [("t", ("a", "b"))]
        rng_flat = random.Random(33)
        rng_warm = random.Random(33)
        _, flat = run_many(2000, steps, {}, rng_flat)
        _, warm = run_many(2000, steps, field, rng_warm)
        self.assertGreater(warm[("a",)], flat[("a",)])

    def test_windowed_trace_produces_different_field_than_unwindowed(self):
        unwindowed = ["L:a", "L:b", "E"]
        windowed = ["W:hot", "L:a", "R", "L:b", "E"]
        flat_field = concentration_from_gene_tokens(unwindowed)
        warm_field = concentration_from_gene_tokens(
            windowed, window_scale={"hot": 10.0}
        )
        # Unwindowed: a and b are equal. Windowed under hot=10x: a >> b.
        self.assertAlmostEqual(flat_field["a"], flat_field["b"])
        self.assertGreater(warm_field["a"], warm_field["b"])


class TestRunRounds(unittest.TestCase):
    def test_seeded_rounds_are_deterministic(self):
        steps = [("t", ("a", "b", "c"))]
        rng_a = random.Random(123)
        rng_b = random.Random(123)
        rounds_a = run_rounds(4, 200, steps, {"a": 1.1}, rng_a)
        rounds_b = run_rounds(4, 200, steps, {"a": 1.1}, rng_b)
        self.assertEqual(
            [r.distribution for r in rounds_a],
            [r.distribution for r in rounds_b],
        )
        self.assertEqual(
            [r.field_in for r in rounds_a],
            [r.field_in for r in rounds_b],
        )

    def test_trace_warms_next_field_with_chosen_names(self):
        steps = [("t", ("a", "b"))]
        rng = random.Random(0)
        rounds = run_rounds(2, 50, steps, {}, rng, bump=1.0)
        # The trace is exactly one L: token per (path, step). With 50 trials
        # and one step, the round-0 trace has 50 entries; the round-1 field
        # must contain only names that were actually chosen, and their
        # weights must equal the chosen-counts (since round-0 field was
        # empty, decay has nothing to carry forward).
        trace_counts = Counter(token.split(":", 1)[1] for token in rounds[0].trace)
        self.assertEqual(set(rounds[1].field_in), set(trace_counts))
        for name, count in trace_counts.items():
            self.assertAlmostEqual(rounds[1].field_in[name], float(count))

    def test_no_decay_accumulates_weights(self):
        steps = [("t", ("a", "b"))]
        rng = random.Random(7)
        rounds = run_rounds(3, 100, steps, {"a": 1.0, "b": 1.0}, rng, decay=1.0)
        # With decay=1.0 every round adds 100 new bumps total across {a,b}.
        # Sum of weights monotonically grows by ~100 per round.
        sums = [sum(r.field_in.values()) for r in rounds]
        self.assertLess(sums[0], sums[1])
        self.assertLess(sums[1], sums[2])

    def test_full_decay_keeps_only_latest_trace(self):
        steps = [("t", ("a", "b"))]
        rng = random.Random(99)
        rounds = run_rounds(
            3, 80, steps, {"a": 99.0, "b": 99.0}, rng, decay=0.0, bump=1.0
        )
        # decay=0 zeros the carried field each round; the new field is built
        # from this round's trace alone, so total weight equals trials.
        for r in rounds[1:]:
            self.assertAlmostEqual(sum(r.field_in.values()), 80.0)

    def test_climate_does_not_override_eligibility(self):
        eligible = ("a", "b")
        steps = [("t", eligible)]
        # The climate names a forbidden operator; if it bypassed
        # eligibility the chosen set would include "ghost".
        rng = random.Random(0)
        rounds = run_rounds(
            3,
            150,
            steps,
            {"ghost": 1e6},
            rng,
            climate_tokens=("W:dev", "L:ghost", "L:ghost", "R"),
            window_scale={"dev": 100.0},
        )
        chosen = {
            step.chosen for r in rounds for path in r.paths for step in path.steps
        }
        self.assertEqual(chosen, {"a", "b"})

    def test_climate_biases_next_field_toward_named_operator(self):
        eligible = ("solve", "score", "reflect_operator")
        steps = [("op", eligible)]
        initial = {"solve": 3.0, "score": 3.0, "reflect_operator": 1.0}
        rng_prod = random.Random(20260519)
        rng_dev = random.Random(20260519)
        prod = run_rounds(5, 400, steps, initial, rng_prod, decay=0.7)
        dev = run_rounds(
            5,
            400,
            steps,
            initial,
            rng_dev,
            decay=0.7,
            climate_tokens=(
                "W:dev",
                "L:reflect_operator",
                "L:reflect_operator",
                "L:reflect_operator",
                "L:reflect_operator",
                "L:reflect_operator",
                "R",
            ),
            window_scale={"dev": 8.0},
        )
        last_prod = prod[-1].distribution[("reflect_operator",)]
        last_dev = dev[-1].distribution[("reflect_operator",)]
        # The dev climate should clearly lift reflect_operator's share.
        self.assertGreater(last_dev, last_prod + 30)

    def test_zero_rounds_returns_empty_tuple(self):
        rng = random.Random(0)
        rounds = run_rounds(0, 100, [("t", ("a",))], {}, rng)
        self.assertEqual(rounds, ())

    def test_negative_rounds_raises(self):
        rng = random.Random(0)
        with self.assertRaises(ValueError):
            run_rounds(-1, 1, [("t", ("a",))], {}, rng)

    def test_decay_out_of_range_raises(self):
        rng = random.Random(0)
        with self.assertRaises(ValueError):
            run_rounds(1, 1, [("t", ("a",))], {}, rng, decay=1.5)
        with self.assertRaises(ValueError):
            run_rounds(1, 1, [("t", ("a",))], {}, rng, decay=-0.1)

    def test_initial_field_is_not_mutated(self):
        initial = {"a": 5.0, "b": 5.0}
        snapshot = dict(initial)
        rng = random.Random(0)
        run_rounds(3, 50, [("t", ("a", "b"))], initial, rng)
        self.assertEqual(initial, snapshot)


if __name__ == "__main__":
    unittest.main()
