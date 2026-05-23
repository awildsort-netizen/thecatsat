#!/usr/bin/env python3
"""Demo driver for the bubble-lifecycle scaffold.

Three deterministic cases:

1. ``real_suite_top`` — pick one instance/view from the same
   22-instance suite ``experiments/transform_litmus.py`` runs, derive a
   strain trace by replaying the probe's decisions, build a bubble
   around the top-strain variable, and classify it. This is honest
   about what the *real* suite shows: PR #9 reported the localized /
   stable verdicts are empty, so we expect the bubble to land on
   ``seed`` / ``leaky`` / ``plaque_risk``, not on ``stable``.

2. ``toy_stable`` — a transparent synthetic strain trace where the
   pressure visibly concentrates onto a small index set and dissipates
   over time. Clearly labelled as a toy: it is the positive control
   for the lifecycle classifier, not evidence the SAT solver produces
   this regime.

3. ``toy_plaque`` — synthetic trace where the interior strain stays
   parked at the same indices and grows over time (the failure mode
   the design doc named ``semantic plaque``). The classifier should
   pick this up as ``plaque_risk``.

Run with: ``python experiments/bubble_lifecycle.py``
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import sat_furnace
from geometry.bubble_lifecycle import (
    BubbleReportRow,
    classify_lifecycle,
    format_report,
    inflate_bubble,
    report_row,
    seed_from_strain,
)
from geometry.flattening_probe import _per_variable_strain
from geometry.riordan_probe import RiordanProbe


PROBE_SEED = 7
MAX_FLIPS = 200
TRACE_SNAPSHOTS = 6


def _planted_formula(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula


def _replay_strain_trace(
    formula,
    n_vars: int,
    start_assignment: list[bool],
    decisions,
    snapshots: int,
) -> list[np.ndarray]:
    """Replay decisions, sampling per-variable strain at evenly-spaced steps.

    Always includes the initial and final per-variable strain. If the
    decision sequence is shorter than ``snapshots-1``, returns whatever
    snapshots are available without padding.
    """
    assignment = list(start_assignment)
    trace: list[np.ndarray] = [_per_variable_strain(formula, assignment, n_vars)]
    n = len(decisions)
    if n == 0:
        return trace
    sample_steps = max(1, n // max(1, snapshots - 1))
    for i, record in enumerate(decisions, start=1):
        assignment[record.flipped_variable] = not assignment[record.flipped_variable]
        if i % sample_steps == 0 or i == n:
            trace.append(_per_variable_strain(formula, assignment, n_vars))
    return trace


def _real_suite_case() -> BubbleReportRow:
    """Pick the highest-residual-strain instance from the easy 2-SAT slice.

    We use a small deterministic instance from the existing suite so
    the demo runs in well under a second. The point is not to find a
    stable bubble — PR #9 already told us the real suite mostly
    doesn't produce one — but to show the classifier's reading on a
    real strain trace, in the honest framing.
    """
    formula = _planted_formula(seed=202, variables=12, clauses=42, k=3)
    n_vars = 12
    probe = RiordanProbe(max_flips=MAX_FLIPS, seed=PROBE_SEED)
    result = probe.run(
        formula=formula,
        n_vars=n_vars,
        instance_id="3sat_v12_c42_s2",
        planted_satisfiable=True,
    )

    # Pick the non-raw view with the most residual strain — that's
    # where a bubble would have something to wrap around.
    start_rng = random.Random(PROBE_SEED)
    start = [start_rng.choice([False, True]) for _ in range(n_vars)]
    best_view = None
    best_strain_sum = -1.0
    best_trace: list[np.ndarray] = []
    for view_name, run in result.runs.items():
        if view_name == "raw" or run.solved:
            continue
        trace = _replay_strain_trace(
            formula, n_vars, start, run.decisions, TRACE_SNAPSHOTS
        )
        final_sum = float(trace[-1].sum())
        if final_sum > best_strain_sum:
            best_strain_sum = final_sum
            best_view = view_name
            best_trace = trace

    if best_view is None:
        # Every view solved — fall back to raw's pre-solve trajectory.
        run = result.runs["raw"]
        best_trace = _replay_strain_trace(
            formula, n_vars, start, run.decisions, TRACE_SNAPSHOTS
        )
        best_view = "raw"

    final_strain = best_trace[-1]
    seed = seed_from_strain(final_strain, view_name=best_view)
    if seed is None:
        # Defensive: solver ate all the strain. Build from initial.
        seed = seed_from_strain(best_trace[0], view_name=best_view)
        final_strain = best_trace[0]
    assert seed is not None, "real-suite case has no strain to bubble around"

    bubble = inflate_bubble(final_strain, seed, radius=2, boundary_width=2)
    trace = classify_lifecycle(bubble, best_trace)
    return report_row(
        case_id=f"real:3sat_v12_c42_s2/{best_view}",
        bubble=bubble,
        trace=trace,
    )


def _toy_stable_case() -> BubbleReportRow:
    """Synthetic trace: strain concentrates on {0, 1, 2} and decays.

    Labelled in the report as ``toy:stable`` so no one confuses it with
    real-suite evidence.
    """
    n_vars = 8
    snapshots = [
        np.array([4.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([3.0, 3.0, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2]),
        np.array([1.5, 1.5, 1.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
        np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05]),
    ]
    final = snapshots[-1]
    seed = seed_from_strain(final, view_name="toy")
    assert seed is not None
    bubble = inflate_bubble(final, seed, radius=2, boundary_width=2)
    trace = classify_lifecycle(bubble, snapshots)
    return report_row(case_id="toy:stable", bubble=bubble, trace=trace)


def _toy_plaque_case() -> BubbleReportRow:
    """Synthetic trace: strain parked on the same indices, *growing*.

    The lifecycle classifier should call this ``plaque_risk``: low
    interior churn, non-decreasing strain, address space held without
    doing work.
    """
    n_vars = 8
    snapshots = [
        np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2]),
        np.array([2.5, 2.5, 2.5, 0.2, 0.2, 0.2, 0.2, 0.2]),
        np.array([3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2, 0.2]),
        np.array([3.5, 3.5, 3.5, 0.2, 0.2, 0.2, 0.2, 0.2]),
        np.array([4.0, 4.0, 4.0, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ]
    final = snapshots[-1]
    seed = seed_from_strain(final, view_name="toy")
    assert seed is not None
    bubble = inflate_bubble(final, seed, radius=2, boundary_width=2)
    trace = classify_lifecycle(bubble, snapshots)
    return report_row(case_id="toy:plaque", bubble=bubble, trace=trace)


def _print_interpretation(rows: list[BubbleReportRow]) -> None:
    print()
    print("Interpretation (read with caution; this is a first scaffold):")
    print(
        "  - The real-suite row is honest about PR #9: on the existing "
        "22-instance suite, the geometry transforms do not produce a "
        "clean ``stable`` interior. The trace label there is whatever "
        "the classifier actually saw, not a forced positive."
    )
    print(
        "  - The two ``toy:*`` rows are *synthetic* controls. They show "
        "the classifier picks ``stable`` when strain visibly concentrates "
        "and dissipates, and ``plaque_risk`` when address space is held "
        "without work. Do not read them as evidence about SAT itself."
    )
    print(
        "  - Containment is cheap by construction: the interior is the "
        "top-radius+1 strain set and the boundary is the next few; the "
        "``margin`` column is interior_min strain minus boundary_max "
        "strain, positive when the bubble has a clean edge."
    )


def main() -> None:
    rows = [
        _real_suite_case(),
        _toy_stable_case(),
        _toy_plaque_case(),
    ]
    print("Bubble containment / lifecycle table")
    print(format_report(rows))
    _print_interpretation(rows)


if __name__ == "__main__":
    main()
