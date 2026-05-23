#!/usr/bin/env python3
"""Tests for the tangent lift probe.

Deterministic — every test pins the exact grid the probe uses, so
metrics are reproducible to floating-point precision. We do not assert
"lifted is universally better"; we assert structural facts:

- the grid is the grid we say it is;
- the lifted chart names the singularity (boundary count > 0 near an
  asymptote);
- off-boundary reconstruction tan = sin/cos is numerically exact;
- in a band around an asymptote, the raw chart's
  ``max_finite_difference`` is many orders of magnitude larger than
  the lifted chart's.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.tangent_lift_probe import (
    ChartMetrics,
    LiftedChart,
    RawChart,
    TangentLiftProbe,
    deterministic_grid,
    lifted_chart_metrics,
    raw_chart_metrics,
)


class DeterministicGridTests(unittest.TestCase):
    def test_grid_is_symmetric_and_evenly_spaced(self) -> None:
        x = deterministic_grid(n_samples=11, span=1.0)
        self.assertEqual(x.size, 11)
        self.assertAlmostEqual(float(x[0]), -1.0)
        self.assertAlmostEqual(float(x[-1]), 1.0)
        spacings = np.diff(x)
        self.assertTrue(np.allclose(spacings, spacings[0]))

    def test_grid_rejects_too_few_samples(self) -> None:
        with self.assertRaises(ValueError):
            deterministic_grid(n_samples=2, span=1.0)

    def test_grid_is_reproducible(self) -> None:
        a = deterministic_grid(n_samples=51, span=math.pi)
        b = deterministic_grid(n_samples=51, span=math.pi)
        np.testing.assert_array_equal(a, b)


class BoundaryDetectionTests(unittest.TestCase):
    def test_boundary_mask_is_named_near_asymptote(self) -> None:
        # A band centred on pi/2 with enough samples that at least one
        # lands inside |cos x| <= 1e-3.
        x = np.linspace(math.pi / 2 - 0.05, math.pi / 2 + 0.05, 201)
        lifted = LiftedChart(x=x, boundary_eps=1.0e-3)
        boundary = lifted.boundary_mask
        # At least one sample must be detected as boundary.
        self.assertGreaterEqual(int(np.sum(boundary)), 1)
        # And the boundary samples must actually have small |cos x|.
        self.assertTrue(np.all(np.abs(lifted.cos[boundary]) <= 1.0e-3))

    def test_no_boundary_far_from_asymptote(self) -> None:
        # An interval bounded away from any pi/2 + k*pi: pick
        # [0.1, 1.4], which sits inside (0, pi/2).
        x = np.linspace(0.1, 1.4, 101)
        lifted = LiftedChart(x=x, boundary_eps=1.0e-3)
        self.assertEqual(int(np.sum(lifted.boundary_mask)), 0)

    def test_explosion_mask_catches_asymptote_in_raw(self) -> None:
        # A grid that lands precisely on the asymptote and a couple of
        # neighbours.
        x = np.array([math.pi / 2 - 1e-9, math.pi / 2, math.pi / 2 + 1e-9])
        raw = RawChart(x=x, explosion_threshold=1.0e6)
        # All three samples should be flagged: tan grows like 1/(pi/2 - x).
        self.assertEqual(int(np.sum(raw.explosion_mask)), 3)


class ReconstructionTests(unittest.TestCase):
    def test_off_boundary_reconstruction_is_exact(self) -> None:
        x = np.linspace(0.1, 1.4, 101)
        lifted = LiftedChart(x=x, boundary_eps=1.0e-3)
        values, valid = lifted.reconstruct_tan()
        self.assertTrue(np.all(valid))
        # tan = sin/cos exactly off boundary, to floating point.
        np.testing.assert_allclose(values, np.tan(x), atol=1e-12)

    def test_boundary_samples_are_masked_not_invented(self) -> None:
        # A band centred on pi/2 — boundary samples should come back nan.
        x = np.linspace(math.pi / 2 - 0.05, math.pi / 2 + 0.05, 201)
        lifted = LiftedChart(x=x, boundary_eps=1.0e-3)
        values, valid = lifted.reconstruct_tan()
        boundary = ~valid
        self.assertGreaterEqual(int(np.sum(boundary)), 1)
        self.assertTrue(np.all(np.isnan(values[boundary])))


class ChartStrainComparisonTests(unittest.TestCase):
    def test_raw_strain_dominates_lifted_strain_near_asymptote(self) -> None:
        x = np.linspace(math.pi / 2 - 0.05, math.pi / 2 + 0.05, 201)
        raw = raw_chart_metrics(RawChart(x=x))
        lifted = lifted_chart_metrics(LiftedChart(x=x))
        # Raw finite-difference must be enormous; lifted bounded by 2
        # (sin, cos both bounded by 1, so consecutive diffs <= 2).
        self.assertGreater(raw.max_finite_difference, 1.0e6)
        self.assertLessEqual(lifted.max_finite_difference, 2.0)
        # And the gap must be many orders of magnitude.
        self.assertGreater(
            raw.max_finite_difference,
            1.0e6 * lifted.max_finite_difference,
        )

    def test_raw_has_no_boundary_concept_lifted_does(self) -> None:
        x = np.linspace(math.pi / 2 - 0.05, math.pi / 2 + 0.05, 201)
        raw = raw_chart_metrics(RawChart(x=x))
        lifted = lifted_chart_metrics(LiftedChart(x=x))
        self.assertEqual(raw.boundary_points, 0)
        self.assertEqual(raw.masked_points, 0)
        self.assertGreaterEqual(lifted.boundary_points, 1)
        self.assertEqual(lifted.boundary_points, lifted.masked_points)

    def test_lifted_off_boundary_reconstruction_error_is_tiny(self) -> None:
        x = np.linspace(math.pi / 2 - 0.05, math.pi / 2 + 0.05, 201)
        lifted = lifted_chart_metrics(LiftedChart(x=x))
        # We don't promise zero (floating point), but it should be way
        # below any meaningful error scale.
        self.assertLess(lifted.reconstruction_error_off_boundary, 1.0e-9)


class ProbeReproducibilityTests(unittest.TestCase):
    def test_probe_run_is_reproducible(self) -> None:
        probe = TangentLiftProbe(n_samples=201, span=2.0 * math.pi)
        a = probe.run()
        b = probe.run()
        for key in ("raw", "lifted"):
            self.assertEqual(a[key], b[key])

    def test_probe_run_returns_well_shaped_metrics(self) -> None:
        probe = TangentLiftProbe(n_samples=101, span=math.pi)
        result = probe.run()
        self.assertIn("raw", result)
        self.assertIn("lifted", result)
        for row in result.values():
            self.assertIsInstance(row, ChartMetrics)
            self.assertEqual(row.n_samples, 101)
            self.assertGreaterEqual(row.boundary_points, 0)
            self.assertGreaterEqual(row.masked_points, 0)
            self.assertGreaterEqual(row.explosion_count, 0)
            self.assertGreaterEqual(row.clipping_burden, 0.0)
            self.assertGreaterEqual(row.reconstruction_error_off_boundary, 0.0)


if __name__ == "__main__":
    unittest.main()
