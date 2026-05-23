#!/usr/bin/env python3
"""Tests for the flattening probe.

Deterministic — seeded RNG everywhere. We don't assert which view
wins; we assert the probe runs, produces well-shaped outputs, and
that obvious invariants (raw-view basis is identity, strain is
nonnegative, flips don't exceed budget) hold.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.flattening_probe import (
    CoordinateView,
    FlatteningProbe,
    raw_view,
    signed_incidence,
    spectral_view,
)


def _planted(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula


class CoordinateViewTests(unittest.TestCase):
    def test_raw_view_basis_is_identity(self) -> None:
        view = raw_view(5)
        np.testing.assert_array_equal(view.basis, np.eye(5))
        self.assertEqual(view.n_vars, 5)
        self.assertEqual(view.k, 5)

    def test_signed_incidence_shape_and_signs(self) -> None:
        formula = [((0, False), (1, True)), ((1, False), (2, False))]
        matrix = signed_incidence(formula, n_vars=3)
        self.assertEqual(matrix.shape, (3, 2))
        # variable 0 positive in clause 0
        self.assertEqual(matrix[0, 0], 1.0)
        # variable 1 negated in clause 0, positive in clause 1
        self.assertEqual(matrix[1, 0], -1.0)
        self.assertEqual(matrix[1, 1], 1.0)
        # variable 2 only in clause 1
        self.assertEqual(matrix[2, 1], 1.0)
        self.assertEqual(matrix[0, 1], 0.0)

    def test_spectral_view_is_orthonormal_on_used_columns(self) -> None:
        formula = _planted(seed=1, variables=8, clauses=14, k=3)
        view = spectral_view(formula, n_vars=8)
        used = view.basis[:, : view.k]
        gram = used.T @ used
        np.testing.assert_allclose(gram, np.eye(view.k), atol=1e-8)

    def test_strain_nonnegative_and_matches_unsat_participation(self) -> None:
        formula = [((0, False), (1, True)), ((1, False), (2, False))]
        view = raw_view(3)
        assignment = [False, True, False]  # clause 0 unsat (0=F, ¬1=F), clause 1 sat
        strain = view.strain(formula, assignment)
        self.assertEqual(strain.per_variable, (1.0, 1.0, 0.0))
        self.assertEqual(strain.total, 2.0)
        self.assertGreaterEqual(min(strain.per_direction), 0.0)


class FlatteningProbeTests(unittest.TestCase):
    def test_probe_runs_and_emits_runs_per_view(self) -> None:
        formula = _planted(seed=2, variables=8, clauses=14, k=3)
        probe = FlatteningProbe(max_flips=80, seed=2)
        result = probe.run(formula=formula, n_vars=8, instance_id="t",
                           planted_satisfiable=True)
        self.assertIn("raw", result.runs)
        spectral_runs = [name for name in result.runs if name != "raw"]
        self.assertEqual(len(spectral_runs), 1)
        for run in result.runs.values():
            self.assertLessEqual(run.flips, 80)
            # solved: trajectory has flips+1 entries (one per step incl. final).
            # not solved: ran the budget then appended a final strain reading,
            # so trajectory has flips+1 entries too.
            self.assertEqual(len(run.strain_trajectory), run.flips + 1)
            self.assertGreaterEqual(run.final_unsatisfied, 0)

    def test_solved_implies_zero_final_unsat(self) -> None:
        # Tiny planted instance — both views should usually solve it
        # within the budget. We only assert the invariant, not the win.
        formula = _planted(seed=3, variables=6, clauses=8, k=2)
        probe = FlatteningProbe(max_flips=200, seed=3)
        result = probe.run(formula=formula, n_vars=6, instance_id="t2",
                           planted_satisfiable=True)
        for run in result.runs.values():
            if run.solved:
                self.assertEqual(run.final_unsatisfied, 0)

    def test_deterministic_under_fixed_seed(self) -> None:
        formula = _planted(seed=4, variables=8, clauses=14, k=3)
        probe_a = FlatteningProbe(max_flips=60, seed=11)
        probe_b = FlatteningProbe(max_flips=60, seed=11)
        result_a = probe_a.run(formula=formula, n_vars=8, instance_id="d",
                               planted_satisfiable=True)
        result_b = probe_b.run(formula=formula, n_vars=8, instance_id="d",
                               planted_satisfiable=True)
        for name in result_a.runs:
            self.assertEqual(
                result_a.runs[name].strain_trajectory,
                result_b.runs[name].strain_trajectory,
            )
            self.assertEqual(
                result_a.runs[name].flips, result_b.runs[name].flips,
            )

    def test_custom_views_argument_is_honored(self) -> None:
        formula = _planted(seed=5, variables=6, clauses=10, k=2)
        only_raw = (raw_view(6),)
        probe = FlatteningProbe(max_flips=40, seed=5)
        result = probe.run(formula=formula, n_vars=6, instance_id="c",
                           planted_satisfiable=True, views=only_raw)
        self.assertEqual(list(result.runs.keys()), ["raw"])


if __name__ == "__main__":
    unittest.main()
