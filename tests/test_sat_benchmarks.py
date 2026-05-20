#!/usr/bin/env python3
"""Tests for the composer-native SAT benchmark harness."""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from sat_benchmarks import (
    SolveResult,
    brute_force_solve,
    build_sat_benchmark_composer,
    dpll_solve,
    furnace_solve_via_iterate,
)
from sat_furnace import clause_satisfied


def _planted(seed: int, variables: int, clauses: int):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("sat", variables, clauses, 3, rng)
    return formula


class BruteAndDPLLAgreementTests(unittest.TestCase):
    """On tiny formulas brute and DPLL should both solve and agree on solved=True."""

    def test_brute_and_dpll_agree_on_tiny_planted_instances(self) -> None:
        for seed in (3, 5, 7, 11, 13):
            formula = _planted(seed, 6, 18)
            brute = brute_force_solve(formula, 6)
            dpll = dpll_solve(formula, 6)
            with self.subTest(seed=seed):
                self.assertTrue(brute.solved)
                self.assertTrue(dpll.solved)
                self.assertEqual(brute.final_unsatisfied, 0)
                self.assertEqual(dpll.final_unsatisfied, 0)
                # Each solver returns a satisfying assignment.
                self.assertIsNotNone(brute.assignment)
                self.assertIsNotNone(dpll.assignment)
                self.assertTrue(
                    all(clause_satisfied(c, brute.assignment) for c in formula)
                )
                self.assertTrue(
                    all(clause_satisfied(c, dpll.assignment) for c in formula)
                )

    def test_solveresult_required_fields_present_for_baselines(self) -> None:
        formula = _planted(7, 5, 14)
        for result in (brute_force_solve(formula, 5), dpll_solve(formula, 5)):
            with self.subTest(solver=result.solver_name):
                self.assertIsInstance(result, SolveResult)
                self.assertIn(
                    result.work_metric, ("assignments_checked", "decisions"),
                )
                self.assertGreaterEqual(result.work_units, 0)
                self.assertGreaterEqual(result.wall_time_s, 0.0)


class BenchmarkComposerShapeTests(unittest.TestCase):
    def test_benchmark_composer_produces_all_three_result_types(self) -> None:
        formula = _planted(7, 6, 18)
        composer = build_sat_benchmark_composer()
        out = composer.run(
            ("brute_result", "dpll_result", "furnace_benchmark_result"),
            {
                "formula": formula,
                "variables": 6,
                "furnace_steps": 15,
                "furnace_seed": 1,
            },
        )
        self.assertEqual(set(out), {
            "brute_result", "dpll_result", "furnace_benchmark_result",
        })
        for key in out:
            self.assertIsInstance(out[key], SolveResult)
        self.assertEqual(out["brute_result"].solver_name, "brute_force")
        self.assertEqual(out["dpll_result"].solver_name, "dpll")
        self.assertEqual(out["furnace_benchmark_result"].solver_name, "furnace")

    def test_benchmark_composer_plan_is_inputs_only(self) -> None:
        """All three operators read formula+variables; no hidden dependencies."""
        composer = build_sat_benchmark_composer()
        plan = composer.plan(
            ("brute_result", "dpll_result", "furnace_benchmark_result"),
            available_keys=("formula", "variables"),
        )
        self.assertEqual(plan.missing, ())
        self.assertEqual(set(plan.order), {
            "brute_force_solve", "dpll_solve", "furnace_solve_via_iterate",
        })


class FurnaceMetabolismShapeTests(unittest.TestCase):
    """Furnace result carries metabolism fields regardless of solved status."""

    _METABOLISM_KEYS = {
        "total_hamming_movement",
        "net_unsat_resolved",
        "distance_paid_per_resolved",
        "shortest_prefix_to_improvement",
        "unsat_clause_revisit_count",
    }

    def test_furnace_result_has_metabolism_fields_when_solved(self) -> None:
        formula = _planted(7, 6, 18)
        result = furnace_solve_via_iterate(
            formula, 6, steps=20, seed=11,
        )
        self.assertEqual(result.solver_name, "furnace")
        self.assertEqual(result.work_metric, "iterate_steps")
        self.assertTrue(self._METABOLISM_KEYS <= set(result.metabolism))

    def test_furnace_result_has_metabolism_fields_even_when_not_solved(self) -> None:
        # A very short budget on a harder instance is likely to leave
        # final_unsatisfied > 0. We don't assert *which* path we hit;
        # we assert the shape is the same either way.
        formula = _planted(23, 12, 48)
        result = furnace_solve_via_iterate(
            formula, 12, steps=3, seed=2,
        )
        self.assertEqual(result.solver_name, "furnace")
        self.assertTrue(self._METABOLISM_KEYS <= set(result.metabolism))
        # final_unsatisfied is non-negative integer regardless of solved.
        self.assertGreaterEqual(result.final_unsatisfied, 0)
        self.assertIsInstance(result.final_unsatisfied, int)
        # solved flag is a bool either way.
        self.assertIsInstance(result.solved, bool)


if __name__ == "__main__":
    unittest.main()
