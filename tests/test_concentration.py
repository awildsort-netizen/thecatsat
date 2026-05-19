#!/usr/bin/env python3
"""Tests for ``concentration``: biased sampling over eligible providers."""

from __future__ import annotations

import random
import unittest

from concentration import run_many, sample_path, sample_provider


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


if __name__ == "__main__":
    unittest.main()
