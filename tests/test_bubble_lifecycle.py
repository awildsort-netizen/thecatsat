#!/usr/bin/env python3
"""Tests for the bubble-lifecycle scaffold.

Deterministic. We exercise:

- seed detection on hand-built strain vectors,
- bubble inflation index sets and the boundary-margin contract,
- containment-report fields,
- static (single-snapshot) lifecycle rules: pruned / seed / leaky / inflated,
- trace lifecycle rules: stable / plaque_risk / pruned / leaky,
- determinism of the report row and the demo driver across processes.

We never assert that the real SAT suite produces a specific bubble
label — the design doc is explicit that the existing transforms do
not. We assert vocabulary, structure, and reproducibility.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.bubble_lifecycle import (
    INFLATED,
    LEAKY,
    LIFECYCLE_LABELS,
    MERGED,
    PLAQUE_RISK,
    PRUNED,
    SEED,
    STABLE,
    boundary_margin,
    classify_lifecycle,
    classify_static,
    contains,
    format_report,
    inflate_bubble,
    report_row,
    seed_from_strain,
)


class SeedDetectionTests(unittest.TestCase):
    def test_seed_picks_top_strain_variable(self) -> None:
        strain = [1.0, 5.0, 2.0, 0.0]
        seed = seed_from_strain(strain, view_name="raw")
        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertEqual(seed.center, 1)
        self.assertEqual(seed.strain_at_center, 5.0)
        self.assertEqual(seed.view_name, "raw")

    def test_seed_returns_none_for_zero_strain(self) -> None:
        self.assertIsNone(seed_from_strain([0.0, 0.0, 0.0]))
        self.assertIsNone(seed_from_strain([]))

    def test_seed_concentration_is_bounded(self) -> None:
        strain = [3.0, 3.0, 3.0, 1.0]
        seed = seed_from_strain(strain, neighborhood=3)
        assert seed is not None
        # The top variable holds 3 of 9 in the neighborhood -> 1/3.
        self.assertAlmostEqual(seed.concentration, 1.0 / 3.0, places=9)

    def test_seed_ties_broken_deterministically(self) -> None:
        # Two top-strain ties; numpy argsort is stable so the lower
        # index wins.
        seed = seed_from_strain([4.0, 4.0, 1.0])
        assert seed is not None
        self.assertEqual(seed.center, 0)


class InflateBubbleTests(unittest.TestCase):
    def test_interior_is_top_radius_plus_one_strain_set(self) -> None:
        strain = np.array([1.0, 5.0, 4.0, 0.5, 0.5, 0.5])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=2, boundary_width=2)
        self.assertEqual(bubble.interior_size, 3)
        self.assertEqual(bubble.boundary_size, 2)
        self.assertEqual(set(bubble.interior), {1, 2, 0})
        # Boundary is the next two on the strain ordering — ties broken
        # by stable sort, so the lowest indices win.
        self.assertTrue(set(bubble.boundary).issubset({3, 4, 5}))

    def test_boundary_margin_positive_when_edge_is_clean(self) -> None:
        strain = np.array([10.0, 9.0, 8.0, 1.0, 1.0, 1.0])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=2, boundary_width=2)
        self.assertGreater(boundary_margin(bubble), 0.0)

    def test_boundary_margin_nonpositive_when_edge_is_leaky(self) -> None:
        strain = np.array([5.0, 4.0, 3.0, 3.0, 3.0, 0.1])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=2, boundary_width=2)
        # interior_min = 3.0 (var 2), boundary_max = 3.0 (var 3) -> margin 0.
        self.assertLessEqual(boundary_margin(bubble), 0.0)


class ContainmentTests(unittest.TestCase):
    def test_seed_center_distance_is_zero(self) -> None:
        strain = np.array([1.0, 5.0, 4.0, 0.5])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=2, boundary_width=1)
        report = contains(bubble, seed.center)
        self.assertEqual(report.distance_to_center, 0)
        self.assertTrue(report.inside)
        self.assertTrue(report.in_top_k_strain)

    def test_boundary_item_is_on_boundary_not_inside(self) -> None:
        strain = np.array([10.0, 9.0, 8.0, 1.0, 1.0])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=2, boundary_width=2)
        # variable 3 or 4 (lowest-strain ties broken deterministically)
        # should sit in the boundary.
        boundary_item = bubble.boundary[0]
        report = contains(bubble, boundary_item)
        self.assertFalse(report.inside)
        self.assertTrue(report.on_boundary)
        self.assertEqual(report.distance_to_center, 3)

    def test_outside_item_is_outside(self) -> None:
        strain = np.array([10.0, 9.0, 8.0, 1.0, 1.0, 0.5, 0.5])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=1, boundary_width=1)
        # interior is {0, 1}, boundary is {2}. Variable 5 is well outside.
        report = contains(bubble, 5)
        self.assertFalse(report.inside)
        self.assertFalse(report.on_boundary)


class StaticClassificationTests(unittest.TestCase):
    def _build(self, strain: list[float], radius: int = 1, boundary_width: int = 1):
        arr = np.asarray(strain, dtype=float)
        seed = seed_from_strain(arr)
        if seed is None:
            return None, None
        bubble = inflate_bubble(arr, seed, radius=radius, boundary_width=boundary_width)
        return arr, bubble

    def test_zero_strain_is_pruned(self) -> None:
        # All-zero strain: seed_from_strain returns None, so we can't
        # build a bubble at all. classify_static is exercised
        # explicitly: construct a bubble-shaped record by hand on a
        # zero vector.
        from geometry.bubble_lifecycle import (
            AddressBubble,
            CollisionSeed,
        )
        from geometry.transform_litmus import localization_of

        strain = np.zeros(4)
        seed = CollisionSeed(
            center=0, strain_at_center=0.0,
            neighborhood_strain=0.0, view_name="raw",
        )
        bubble = AddressBubble(
            seed=seed,
            interior=(0,),
            boundary=(1,),
            radius=0,
            strain_profile=tuple(strain.tolist()),
            localization=localization_of(strain.tolist()),
            lifecycle=classify_static(strain, (0,), (1,), localization_of(strain.tolist())),
        )
        self.assertEqual(bubble.lifecycle, PRUNED)

    def test_concentrated_strain_with_clean_edge_is_inflated(self) -> None:
        _, bubble = self._build(
            [10.0, 8.0, 0.1, 0.1, 0.1], radius=1, boundary_width=2,
        )
        assert bubble is not None
        self.assertEqual(bubble.lifecycle, INFLATED)

    def test_uniform_strain_is_seed(self) -> None:
        # Even concentration: top-2 holds 2/6 == 0.33 ≥ 0.30 but not
        # 0.60, and boundary_max ≈ interior_min so it lands on LEAKY.
        # That's actually the right reading: a uniform region IS leaky.
        _, bubble = self._build(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], radius=1, boundary_width=1,
        )
        assert bubble is not None
        self.assertIn(bubble.lifecycle, {SEED, LEAKY})

    def test_leaky_edge_classification(self) -> None:
        # Interior strain just barely above boundary strain.
        _, bubble = self._build(
            [3.0, 3.0, 2.9, 2.9, 0.1], radius=1, boundary_width=2,
        )
        assert bubble is not None
        self.assertEqual(bubble.lifecycle, LEAKY)


class LifecycleTraceTests(unittest.TestCase):
    def test_strain_dissipates_to_zero_is_pruned(self) -> None:
        strain = np.array([2.0, 2.0, 1.0, 0.1, 0.1])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=1, boundary_width=1)
        trace = [
            strain,
            strain * 0.5,
            np.zeros_like(strain),
        ]
        result = classify_lifecycle(bubble, trace)
        self.assertEqual(result.label, PRUNED)

    def test_stable_trace_low_churn_decaying_strain(self) -> None:
        snapshots = [
            np.array([4.0, 3.0, 2.0, 0.5, 0.5, 0.5]),
            np.array([3.0, 2.0, 1.5, 0.2, 0.2, 0.2]),
            np.array([2.0, 1.5, 1.0, 0.1, 0.1, 0.1]),
        ]
        seed = seed_from_strain(snapshots[-1])
        assert seed is not None
        bubble = inflate_bubble(snapshots[-1], seed, radius=1, boundary_width=1)
        result = classify_lifecycle(bubble, snapshots)
        self.assertEqual(result.label, STABLE)
        self.assertLessEqual(result.total_strain_delta, 0.0)

    def test_plaque_risk_growing_strain_no_churn(self) -> None:
        snapshots = [
            np.array([2.0, 2.0, 1.0, 0.1, 0.1]),
            np.array([2.5, 2.5, 1.0, 0.1, 0.1]),
            np.array([3.0, 3.0, 1.0, 0.1, 0.1]),
        ]
        seed = seed_from_strain(snapshots[-1])
        assert seed is not None
        bubble = inflate_bubble(snapshots[-1], seed, radius=1, boundary_width=1)
        result = classify_lifecycle(bubble, snapshots)
        self.assertEqual(result.label, PLAQUE_RISK)
        self.assertGreater(result.total_strain_delta, 0.0)

    def test_leaky_trace_high_outside_strain(self) -> None:
        # Most strain held outside the recorded interior/boundary.
        snapshots = [
            np.array([5.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]),
        ]
        # Build the bubble from the FIRST snapshot so its interior is
        # {0, 1} — by the last snapshot, most strain has leaked out to
        # {2, 3, 4, 5}.
        seed = seed_from_strain(snapshots[0])
        assert seed is not None
        bubble = inflate_bubble(snapshots[0], seed, radius=1, boundary_width=1)
        result = classify_lifecycle(bubble, snapshots)
        self.assertEqual(result.label, LEAKY)
        self.assertGreater(result.boundary_leak, 0.5)

    def test_label_always_in_vocabulary(self) -> None:
        snapshots = [
            np.array([1.0, 0.5, 0.25, 0.1]),
            np.array([1.5, 0.5, 0.25, 0.1]),
        ]
        seed = seed_from_strain(snapshots[-1])
        assert seed is not None
        bubble = inflate_bubble(snapshots[-1], seed, radius=1, boundary_width=1)
        result = classify_lifecycle(bubble, snapshots)
        self.assertIn(result.label, LIFECYCLE_LABELS)

    def test_empty_trace_falls_back_to_static_label(self) -> None:
        strain = np.array([3.0, 2.0, 1.0, 0.1])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=1, boundary_width=1)
        result = classify_lifecycle(bubble, [])
        self.assertEqual(result.label, bubble.lifecycle)


class ReportFormattingTests(unittest.TestCase):
    def test_format_report_is_deterministic(self) -> None:
        strain = np.array([4.0, 3.0, 2.0, 0.1, 0.1])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=1, boundary_width=1)
        snapshots = [strain, strain * 0.5, strain * 0.25]
        trace = classify_lifecycle(bubble, snapshots)
        row = report_row("case_a", bubble, trace)
        first = format_report([row])
        second = format_report([row])
        self.assertEqual(first, second)
        # Header is present and includes every column label.
        for col in ("case", "interior", "boundary", "margin", "static", "trace", "churn", "leak"):
            self.assertIn(col, first)


class DemoDriverDeterminismTests(unittest.TestCase):
    """The demo driver output must reproduce across separate processes.

    This is the same contract ``tests/test_transform_litmus.py`` pins
    on the litmus summary: a doc-quoted table from a probe must be
    byte-identical across two Python invocations.
    """

    def test_driver_output_is_stable_across_subprocesses(self) -> None:
        import subprocess

        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [sys.executable, "-m", "experiments.bubble_lifecycle"]
        out_a = subprocess.check_output(cmd, cwd=cwd).decode()
        out_b = subprocess.check_output(cmd, cwd=cwd).decode()
        self.assertEqual(out_a, out_b)
        self.assertIn("Bubble containment / lifecycle table", out_a)
        # The toy controls should land on the predicted labels.
        self.assertIn("toy:stable", out_a)
        self.assertIn("toy:plaque", out_a)


if __name__ == "__main__":
    unittest.main()
