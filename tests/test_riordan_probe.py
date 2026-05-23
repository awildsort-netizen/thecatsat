#!/usr/bin/env python3
"""Tests for the Riordan probe.

Deterministic — seeded RNG everywhere. We don't assert which view
wins; we assert the Pascal/Riordan transforms have the algebraic
properties we expect, that the probe runs end-to-end, and that the
report is reproducible under a fixed seed.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.flattening_probe import ProbeResult
from geometry.riordan_probe import (
    RiordanProbe,
    head_to_head,
    pascal_matrix,
    pascal_view,
    sierpinski_matrix,
    sierpinski_view,
    signed_pascal_matrix,
    signed_pascal_view,
)


def _planted(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula


class PascalMatrixTests(unittest.TestCase):
    def test_pascal_first_column_is_ones(self) -> None:
        p = pascal_matrix(6)
        np.testing.assert_array_equal(p[:, 0], np.ones(6))

    def test_pascal_diagonal_is_ones(self) -> None:
        p = pascal_matrix(6)
        np.testing.assert_array_equal(np.diag(p), np.ones(6))

    def test_pascal_is_lower_triangular(self) -> None:
        p = pascal_matrix(7)
        upper = np.triu(p, k=1)
        np.testing.assert_array_equal(upper, np.zeros_like(upper))

    def test_pascal_recurrence_holds(self) -> None:
        p = pascal_matrix(6)
        for i in range(1, 6):
            for j in range(1, i + 1):
                self.assertEqual(p[i, j], p[i - 1, j - 1] + p[i - 1, j])

    def test_signed_pascal_is_inverse_of_pascal(self) -> None:
        n = 7
        p = pascal_matrix(n)
        s = signed_pascal_matrix(n)
        product = p @ s
        np.testing.assert_allclose(product, np.eye(n), atol=1e-9)

    def test_sierpinski_is_zero_one(self) -> None:
        m = sierpinski_matrix(8)
        unique = set(np.unique(m).tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}))

    def test_sierpinski_matches_pascal_mod_2(self) -> None:
        p = pascal_matrix(8)
        m = sierpinski_matrix(8)
        np.testing.assert_array_equal(m, np.mod(p, 2.0))


class RiordanViewTests(unittest.TestCase):
    def test_pascal_view_columns_are_scale_bounded(self) -> None:
        # Row-normalized => each row has unit L2. The largest column
        # entry then is at most 1. We assert this holds across the
        # n_vars we actually run the suite on, so the probe can't blow
        # up numerically as instance size grows.
        for n in (4, 8, 12, 16):
            view = pascal_view(n)
            self.assertLessEqual(float(np.max(np.abs(view.basis))), 1.0 + 1e-9)
            row_norms = np.linalg.norm(view.basis, axis=1)
            np.testing.assert_allclose(row_norms, np.ones(n), atol=1e-9)

    def test_signed_pascal_view_columns_are_scale_bounded(self) -> None:
        for n in (4, 8, 12):
            view = signed_pascal_view(n)
            self.assertLessEqual(float(np.max(np.abs(view.basis))), 1.0 + 1e-9)

    def test_sierpinski_view_columns_are_scale_bounded(self) -> None:
        for n in (4, 8, 12):
            view = sierpinski_view(n)
            self.assertLessEqual(float(np.max(np.abs(view.basis))), 1.0 + 1e-9)

    def test_view_names_are_stable(self) -> None:
        self.assertEqual(pascal_view(6).name, "pascal")
        self.assertEqual(signed_pascal_view(6).name, "signed_pascal")
        self.assertEqual(sierpinski_view(6).name, "sierpinski")


class RiordanProbeTests(unittest.TestCase):
    def test_probe_runs_all_five_views(self) -> None:
        formula = _planted(seed=2, variables=8, clauses=14, k=3)
        probe = RiordanProbe(max_flips=60, seed=2)
        result = probe.run(formula=formula, n_vars=8, instance_id="t",
                           planted_satisfiable=True)
        expected = {"raw", "pascal", "signed_pascal", "sierpinski"}
        self.assertTrue(expected.issubset(result.runs.keys()))
        # Plus a spectral run, with the k baked into its name.
        spectral = [n for n in result.runs if n.startswith("spectral")]
        self.assertEqual(len(spectral), 1)
        for run in result.runs.values():
            self.assertLessEqual(run.flips, 60)
            self.assertEqual(len(run.strain_trajectory), run.flips + 1)
            self.assertGreaterEqual(run.final_unsatisfied, 0)

    def test_deterministic_under_fixed_seed(self) -> None:
        formula = _planted(seed=4, variables=8, clauses=14, k=3)
        a = RiordanProbe(max_flips=40, seed=11).run(
            formula=formula, n_vars=8, instance_id="d", planted_satisfiable=True,
        )
        b = RiordanProbe(max_flips=40, seed=11).run(
            formula=formula, n_vars=8, instance_id="d", planted_satisfiable=True,
        )
        for name in a.runs:
            self.assertEqual(
                a.runs[name].strain_trajectory,
                b.runs[name].strain_trajectory,
            )
            self.assertEqual(a.runs[name].flips, b.runs[name].flips)

    def test_solved_implies_zero_final_unsat(self) -> None:
        formula = _planted(seed=3, variables=6, clauses=8, k=2)
        result = RiordanProbe(max_flips=200, seed=3).run(
            formula=formula, n_vars=6, instance_id="t2", planted_satisfiable=True,
        )
        for run in result.runs.values():
            if run.solved:
                self.assertEqual(run.final_unsatisfied, 0)

    def test_head_to_head_counts_match_runs(self) -> None:
        formula = _planted(seed=9, variables=8, clauses=14, k=3)
        result = RiordanProbe(max_flips=60, seed=9).run(
            formula=formula, n_vars=8, instance_id="h", planted_satisfiable=True,
        )
        summary = head_to_head([result], baseline="raw")
        # Every non-raw view contributes exactly one comparison.
        non_raw = [n for n in result.runs if n != "raw"]
        self.assertEqual(set(summary.keys()), set(non_raw))
        for row in summary.values():
            self.assertEqual(row["wins"] + row["ties"] + row["losses"], 1)


class ReportReproducibilityTests(unittest.TestCase):
    """The end-to-end report has to be reproducible — that is the contract
    the doc page makes when it quotes win/tie/loss numbers.
    """

    def _run_suite(self) -> list[ProbeResult]:
        results: list[ProbeResult] = []
        probe = RiordanProbe(max_flips=80, seed=7)
        for seed in range(2):
            formula = _planted(seed=100 + seed, variables=8, clauses=14, k=2)
            results.append(
                probe.run(
                    formula=formula, n_vars=8,
                    instance_id=f"r{seed}", planted_satisfiable=True,
                )
            )
        return results

    def test_two_runs_produce_identical_head_to_head(self) -> None:
        a = head_to_head(self._run_suite(), baseline="raw")
        b = head_to_head(self._run_suite(), baseline="raw")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
