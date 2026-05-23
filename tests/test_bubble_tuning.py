#!/usr/bin/env python3
"""Tests for the bubble-tuning layer stacked on bubble-lifecycle.

We exercise:

- pressure-gauge metrics on hand-built strain traces,
- pressure-label classification (diffuse / diagnostic / destructive),
- verdict rule-table matching (stabilize / split / merge / prune / hold),
- hierarchy decisions (sub-bubble indices on a split),
- phase readout across a family of view observations,
- determinism of the tuning report.

We never assert that the real SAT suite produces a specific verdict —
the design doc is explicit that the existing transforms do not yet
nucleate stable bubbles. We pin vocabulary, structure, and
reproducibility.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.bubble_lifecycle import (
    inflate_bubble,
    seed_from_strain,
)
from geometry.bubble_tuning import (
    ALIGNED,
    DESTRUCTIVE_AMPLIFICATION,
    DIAGNOSTIC_AMPLIFICATION,
    DIFFUSE_PRESSURE,
    HOLD,
    MERGE,
    NEEDS_ANOTHER_LAYER,
    OFF_PHASE,
    OVER_SMOOTHED,
    PHASE_LABELS,
    PRESSURE_LABELS,
    PRUNE,
    RULES,
    SPLIT,
    STABILIZE,
    STRAIN_AMPLIFIED,
    VERDICT_LABELS,
    BubblePressure,
    PhaseObservation,
    PhaseReadout,
    TuningLaw,
    TuningVerdict,
    format_phase_readout,
    format_tuning_report,
    hierarchy_for,
    measure_pressure,
    read_phase,
    tuning_row,
    verdict_for,
)


# --------------------------------------------------------------------------- #
# Pressure gauge                                                              #
# --------------------------------------------------------------------------- #


class PressureMetricTests(unittest.TestCase):
    def _bubble(self, strain):
        arr = np.asarray(strain, dtype=float)
        seed = seed_from_strain(arr)
        assert seed is not None
        return arr, inflate_bubble(arr, seed, radius=2, boundary_width=2)

    def test_pressure_label_in_vocabulary(self) -> None:
        snapshots = [
            np.array([4.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([3.0, 3.0, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ]
        _, bubble = self._bubble(snapshots[-1])
        pressure = measure_pressure(bubble, snapshots)
        self.assertIn(pressure.pressure_label, PRESSURE_LABELS)
        self.assertEqual(pressure.snapshots, 2)

    def test_diffuse_pressure_low_std_high_mean(self) -> None:
        # Uniform vector: std/mean ratio is ~0; label is diffuse.
        snapshots = [
            np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        ]
        _, bubble = self._bubble(snapshots[-1])
        pressure = measure_pressure(bubble, snapshots)
        self.assertEqual(pressure.pressure_label, DIFFUSE_PRESSURE)

    def test_diagnostic_amplification_concentrated_clean_edge(self) -> None:
        # Strain concentrated on a small set, stable boundary, low off-bubble.
        snapshots = [
            np.array([5.0, 4.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
            np.array([5.5, 4.5, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
            np.array([6.0, 5.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ]
        _, bubble = self._bubble(snapshots[-1])
        pressure = measure_pressure(bubble, snapshots)
        self.assertEqual(pressure.pressure_label, DIAGNOSTIC_AMPLIFICATION)
        self.assertGreaterEqual(pressure.boundary_stability, 0.5)
        self.assertLessEqual(pressure.off_bubble_strain, 0.40)

    def test_destructive_amplification_high_std_with_leak(self) -> None:
        # Strain churns to indices outside the recorded bubble.
        snapshots = [
            np.array([5.0, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            np.array([0.1, 0.1, 5.0, 5.0, 5.0, 5.0, 0.1, 0.1]),
            np.array([0.1, 0.1, 5.0, 5.0, 5.0, 5.0, 0.1, 0.1]),
        ]
        # Build bubble from first snapshot so by the end, strain has leaked.
        seed = seed_from_strain(snapshots[0])
        assert seed is not None
        bubble = inflate_bubble(snapshots[0], seed, radius=1, boundary_width=1)
        pressure = measure_pressure(bubble, snapshots)
        self.assertEqual(pressure.pressure_label, DESTRUCTIVE_AMPLIFICATION)

    def test_interior_std_reflects_within_interior_variance(self) -> None:
        # Build a case where the interior holds {0, 1, 2} but their
        # values differ widely; expect non-trivial interior_std.
        snapshots = [
            np.array([10.0, 5.0, 1.0, 0.1, 0.1, 0.1]),
            np.array([10.0, 5.0, 1.0, 0.1, 0.1, 0.1]),
        ]
        _, bubble = self._bubble(snapshots[-1])
        pressure = measure_pressure(bubble, snapshots)
        # Interior strains are [10, 5, 1]; pstdev > 3.5.
        self.assertGreater(pressure.interior_std, 3.0)


# --------------------------------------------------------------------------- #
# Verdict rule table                                                          #
# --------------------------------------------------------------------------- #


def _pressure(
    interior_mean: float = 1.0,
    interior_std: float = 0.0,
    off: float = 0.0,
    stab: float = 1.0,
    total_mean: float = 1.0,
    total_std: float = 0.0,
    label: str = STRAIN_AMPLIFIED,
) -> BubblePressure:
    """Tiny helper for synthesizing pressure readings directly."""
    return BubblePressure(
        interior_mean=interior_mean,
        interior_std=interior_std,
        off_bubble_strain=off,
        boundary_stability=stab,
        total_mean=total_mean,
        total_std=total_std,
        snapshots=1,
        pressure_label=label,
    )


class VerdictRuleTableTests(unittest.TestCase):
    def test_resolved_strain_becomes_prune(self) -> None:
        p = _pressure(interior_mean=0.0, total_mean=0.0)
        v = verdict_for(p)
        self.assertEqual(v.action, PRUNE)
        self.assertEqual(v.law, "strain_dissipated")

    def test_off_bubble_dominant_becomes_merge(self) -> None:
        p = _pressure(off=0.75)
        v = verdict_for(p)
        self.assertEqual(v.action, MERGE)
        self.assertEqual(v.law, "off_bubble_dominant")

    def test_diffuse_pressure_becomes_hold(self) -> None:
        p = _pressure(label=DIFFUSE_PRESSURE)
        v = verdict_for(p)
        self.assertEqual(v.action, HOLD)
        self.assertEqual(v.law, "diffuse_pressure")

    def test_high_interior_std_becomes_split(self) -> None:
        p = _pressure(
            interior_mean=4.0,
            interior_std=3.0,
            label=DIAGNOSTIC_AMPLIFICATION,
        )
        v = verdict_for(p)
        self.assertEqual(v.action, SPLIT)
        self.assertEqual(v.law, "high_interior_variance")

    def test_destructive_label_becomes_hold(self) -> None:
        p = _pressure(label=DESTRUCTIVE_AMPLIFICATION, total_std=1.0)
        v = verdict_for(p)
        self.assertEqual(v.action, HOLD)
        self.assertEqual(v.law, "destructive_amplification")

    def test_clean_diagnostic_becomes_stabilize(self) -> None:
        p = _pressure(
            interior_mean=4.0,
            interior_std=0.5,
            off=0.05,
            stab=0.95,
            total_mean=2.0,
            total_std=1.0,
            label=DIAGNOSTIC_AMPLIFICATION,
        )
        v = verdict_for(p)
        self.assertEqual(v.action, STABILIZE)
        self.assertEqual(v.law, "stable_diagnostic_bubble")

    def test_default_law_is_hold(self) -> None:
        # A pressure reading no specific law claims falls to the default.
        p = _pressure(
            interior_mean=2.0,
            interior_std=0.0,
            off=0.3,
            stab=0.4,
            total_mean=1.0,
            total_std=0.05,
            label=STRAIN_AMPLIFIED,
        )
        v = verdict_for(p)
        self.assertEqual(v.action, HOLD)
        self.assertIn(v.law, {law.name for law in RULES} | {"hold_no_law_matched"})

    def test_all_law_actions_in_vocabulary(self) -> None:
        # Every rule produces a verdict label that lives in the public
        # vocabulary list. This is a contract test for downstream code.
        self.assertTrue(all(law.action in VERDICT_LABELS for law in RULES))


# --------------------------------------------------------------------------- #
# Hierarchy decisions                                                         #
# --------------------------------------------------------------------------- #


class HierarchyTests(unittest.TestCase):
    def _bubble(self, strain):
        arr = np.asarray(strain, dtype=float)
        seed = seed_from_strain(arr)
        assert seed is not None
        return inflate_bubble(arr, seed, radius=2, boundary_width=2)

    def test_split_yields_top_half_subbubble(self) -> None:
        bubble = self._bubble([10.0, 5.0, 1.0, 0.1, 0.1, 0.1])
        # High within-interior std → split. interior is {0,1,2}; top-half
        # is {0} (cut = max(1, 3//2) = 1).
        p = _pressure(
            interior_mean=5.3,
            interior_std=3.7,
            label=DIAGNOSTIC_AMPLIFICATION,
        )
        decision = hierarchy_for(bubble, p)
        self.assertEqual(decision.action, SPLIT)
        self.assertEqual(decision.subbubble_indices, (0,))

    def test_non_split_actions_have_empty_subbubble(self) -> None:
        bubble = self._bubble([10.0, 5.0, 1.0, 0.1, 0.1, 0.1])
        p = _pressure(interior_mean=0.0, total_mean=0.0)  # PRUNE
        decision = hierarchy_for(bubble, p)
        self.assertEqual(decision.action, PRUNE)
        self.assertEqual(decision.subbubble_indices, ())


# --------------------------------------------------------------------------- #
# Phase readout                                                               #
# --------------------------------------------------------------------------- #


class PhaseReadoutTests(unittest.TestCase):
    def test_aligned_when_one_view_dominates(self) -> None:
        obs = [
            PhaseObservation("raw", interior_share=0.30, off_bubble_share=0.50),
            PhaseObservation("pascal", interior_share=0.70, off_bubble_share=0.10),
            PhaseObservation("signed", interior_share=0.40, off_bubble_share=0.30),
        ]
        readout = read_phase(obs)
        self.assertEqual(readout.label, ALIGNED)
        self.assertEqual(readout.best_view, "pascal")

    def test_off_phase_when_no_view_localizes(self) -> None:
        obs = [
            PhaseObservation("raw", interior_share=0.20, off_bubble_share=0.60),
            PhaseObservation("pascal", interior_share=0.30, off_bubble_share=0.55),
            PhaseObservation("signed", interior_share=0.25, off_bubble_share=0.65),
        ]
        readout = read_phase(obs)
        self.assertEqual(readout.label, OFF_PHASE)

    def test_over_smoothed_when_family_collapses(self) -> None:
        # All views agree on a middling interior share with tiny spread.
        obs = [
            PhaseObservation("raw", interior_share=0.40, off_bubble_share=0.45),
            PhaseObservation("pascal", interior_share=0.42, off_bubble_share=0.43),
            PhaseObservation("signed", interior_share=0.41, off_bubble_share=0.44),
        ]
        readout = read_phase(obs)
        self.assertEqual(readout.label, OVER_SMOOTHED)

    def test_empty_observations_needs_layer(self) -> None:
        readout = read_phase([])
        self.assertEqual(readout.label, NEEDS_ANOTHER_LAYER)
        self.assertEqual(readout.best_view, "")

    def test_label_always_in_vocabulary(self) -> None:
        obs = [PhaseObservation("raw", 0.5, 0.5)]
        readout = read_phase(obs)
        self.assertIn(readout.label, PHASE_LABELS)


# --------------------------------------------------------------------------- #
# Reporting determinism                                                       #
# --------------------------------------------------------------------------- #


class ReportingTests(unittest.TestCase):
    def test_tuning_report_is_deterministic(self) -> None:
        strain = np.array([5.0, 4.0, 3.0, 0.1, 0.1, 0.1])
        seed = seed_from_strain(strain)
        assert seed is not None
        bubble = inflate_bubble(strain, seed, radius=2, boundary_width=2)
        snapshots = [strain, strain * 0.9, strain * 0.8]
        p = measure_pressure(bubble, snapshots)
        v = verdict_for(p)
        row = tuning_row("case_a", p, v)
        first = format_tuning_report([row])
        second = format_tuning_report([row])
        self.assertEqual(first, second)
        for col in ("case", "pressure", "i_mean", "i_std", "off", "b_stab", "action", "law"):
            self.assertIn(col, first)

    def test_phase_readout_format_includes_label(self) -> None:
        readout = PhaseReadout(
            label=ALIGNED,
            best_view="pascal",
            best_interior_share=0.7,
            spread=0.15,
            explanation="example",
        )
        rendered = format_phase_readout(readout)
        self.assertIn("aligned", rendered)
        self.assertIn("pascal", rendered)


class DemoDriverDeterminismTests(unittest.TestCase):
    """Same contract bubble_lifecycle pins: the demo output must be stable
    byte-for-byte across two separate Python invocations."""

    def test_driver_output_is_stable_across_subprocesses(self) -> None:
        import subprocess

        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [sys.executable, "-m", "experiments.bubble_tuning"]
        out_a = subprocess.check_output(cmd, cwd=cwd).decode()
        out_b = subprocess.check_output(cmd, cwd=cwd).decode()
        self.assertEqual(out_a, out_b)
        self.assertIn("Bubble tuning table", out_a)
        # Synthetic controls land on their predicted verdicts.
        self.assertIn("toy:diffuse", out_a)
        self.assertIn("toy:diagnostic", out_a)
        self.assertIn("toy:destructive", out_a)


if __name__ == "__main__":
    unittest.main()
