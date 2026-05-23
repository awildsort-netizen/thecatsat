#!/usr/bin/env python3
"""Demo driver for the bubble-tuning layer.

Three synthetic controls + one honest real-suite case, structured the
same way ``experiments/bubble_lifecycle.py`` is — but one layer up. The
controls exercise the three pressure regimes the gauge is meant to
distinguish:

1. ``toy:diffuse`` — uniform strain that never separates. Verdict:
   ``hold``. The point is that the gauge does *not* recommend a
   stabilize/split/merge action when nothing is actually happening.
2. ``toy:diagnostic`` — strain concentrated on a small interior with a
   clean edge. Verdict: ``stabilize``. Positive control for diagnostic
   amplification.
3. ``toy:destructive`` — strain that churns out of the bubble into the
   surroundings. Verdict: ``hold`` (with reason
   ``destructive_amplification``). Negative control: we do not
   stabilize off-phase.
4. ``real:3sat_v12_c42_s2/{view}`` — same instance and view selection
   ``experiments/bubble_lifecycle.py`` uses, run through the tuning
   gauge. As PR #9/#11 already told us, the existing transforms do not
   produce a clean stable interior; the row is whatever the gauge
   actually reads, not a forced positive.

The driver also prints an exploratory ``PhaseReadout`` over a single
real instance's transform family. The output is deliberately framed as
exploratory — see ``docs/representational_bubbles.md`` and the
interpretation paragraph at the bottom.

Run with: ``python -m experiments.bubble_tuning``
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import sat_furnace
from experiments.bubble_lifecycle import (
    PROBE_SEED,
    TRACE_SNAPSHOTS,
    _planted_formula,
    _replay_strain_trace,
)
from geometry.bubble_lifecycle import (
    inflate_bubble,
    seed_from_strain,
)
from geometry.bubble_tuning import (
    PhaseObservation,
    TuningReportRow,
    format_phase_readout,
    format_tuning_report,
    measure_pressure,
    read_phase,
    tuning_row,
    verdict_for,
)
from geometry.flattening_probe import _per_variable_strain
from geometry.riordan_probe import RiordanProbe


# --------------------------------------------------------------------------- #
# Synthetic controls                                                          #
# --------------------------------------------------------------------------- #


def _make_row(case_id: str, snapshots: list[np.ndarray]) -> TuningReportRow:
    final = snapshots[-1]
    seed = seed_from_strain(final, view_name=case_id)
    # Fall back to the first snapshot if the trace ends at zero strain.
    seed = seed or seed_from_strain(snapshots[0], view_name=case_id)
    assert seed is not None, f"case {case_id}: no strain to bubble around"
    base = final if final.sum() > 0.0 else snapshots[0]
    bubble = inflate_bubble(base, seed, radius=2, boundary_width=2)
    pressure = measure_pressure(bubble, snapshots)
    verdict = verdict_for(pressure)
    return tuning_row(case_id, pressure, verdict)


def _toy_diffuse() -> TuningReportRow:
    # Uniform strain; low std/mean ratio.
    snapshots = [
        np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    ]
    return _make_row("toy:diffuse", snapshots)


def _toy_diagnostic() -> TuningReportRow:
    # Concentrated interior, stable edge, total mean rises.
    snapshots = [
        np.array([5.0, 4.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        np.array([5.5, 4.5, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        np.array([6.0, 5.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ]
    return _make_row("toy:diagnostic", snapshots)


def _toy_destructive() -> TuningReportRow:
    # Strain leaks out of the recorded bubble into the surroundings.
    snapshots = [
        np.array([5.0, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        np.array([0.1, 0.1, 5.0, 5.0, 5.0, 5.0, 0.1, 0.1]),
        np.array([0.1, 0.1, 5.0, 5.0, 5.0, 5.0, 0.1, 0.1]),
    ]
    # We deliberately build the bubble off the *first* snapshot so by the
    # end of the trace strain has leaked out — that is the regime the
    # destructive label is meant to capture.
    seed = seed_from_strain(snapshots[0], view_name="toy:destructive")
    assert seed is not None
    bubble = inflate_bubble(snapshots[0], seed, radius=1, boundary_width=1)
    pressure = measure_pressure(bubble, snapshots)
    verdict = verdict_for(pressure)
    return tuning_row("toy:destructive", pressure, verdict)


# --------------------------------------------------------------------------- #
# Real-suite case                                                             #
# --------------------------------------------------------------------------- #


def _real_suite_case() -> tuple[TuningReportRow, list[PhaseObservation]]:
    """Pick the non-raw view with the most residual strain.

    Returns both the tuning row for that view *and* a phase observation
    per view — the latter feeds ``read_phase``. The point of the phase
    readout here is exploratory, not load-bearing: we expect ``off_phase``
    or ``needs_another_layer`` to be common until a bubble-aware
    transform is added.
    """
    formula = _planted_formula(seed=202, variables=12, clauses=42, k=3)
    n_vars = 12
    probe = RiordanProbe(max_flips=200, seed=PROBE_SEED)
    result = probe.run(
        formula=formula,
        n_vars=n_vars,
        instance_id="3sat_v12_c42_s2",
        planted_satisfiable=True,
    )

    start_rng = random.Random(PROBE_SEED)
    start = [start_rng.choice([False, True]) for _ in range(n_vars)]

    # Build per-view traces + summary observations via comprehension.
    view_payloads = [
        (view_name, _replay_strain_trace(formula, n_vars, start, run.decisions, TRACE_SNAPSHOTS))
        for view_name, run in result.runs.items()
        if not run.solved
    ]

    if not view_payloads:
        # Fall back to raw view's pre-solve trajectory if everything solved.
        run = result.runs["raw"]
        view_payloads = [
            ("raw", _replay_strain_trace(formula, n_vars, start, run.decisions, TRACE_SNAPSHOTS)),
        ]

    # Pick the view with the most residual strain at the final snapshot.
    best_view, best_trace = max(
        view_payloads, key=lambda vt: float(vt[1][-1].sum()),
    )

    final_strain = best_trace[-1] if float(best_trace[-1].sum()) > 0.0 else best_trace[0]
    seed = seed_from_strain(final_strain, view_name=best_view)
    if seed is None:
        seed = seed_from_strain(best_trace[0], view_name=best_view)
        final_strain = best_trace[0]
    assert seed is not None

    bubble = inflate_bubble(final_strain, seed, radius=2, boundary_width=2)
    pressure = measure_pressure(bubble, best_trace)
    verdict = verdict_for(pressure)
    row = tuning_row(f"real:3sat_v12_c42_s2/{best_view}", pressure, verdict)

    # Build phase observations per view: interior_share = top-3 strain
    # share at the final snapshot; off_bubble_share = 1 - interior_share -
    # next-2 boundary share. Deliberately simple.
    observations = [
        _observe_phase(view_name, trace) for view_name, trace in view_payloads
    ]
    return row, observations


def _observe_phase(view_name: str, trace: list[np.ndarray]) -> PhaseObservation:
    """Summarize a single view's final strain into a phase observation."""
    final = trace[-1]
    total = float(final.sum())
    if total <= 0.0:
        return PhaseObservation(view_name=view_name, interior_share=0.0, off_bubble_share=0.0)
    ranked = sorted(range(final.size), key=lambda i: (-float(final[i]), i))
    interior = ranked[:3]
    boundary = ranked[3:5]
    interior_share = float(final[interior].sum()) / total
    boundary_share = float(final[boundary].sum()) / total
    off_share = max(0.0, 1.0 - interior_share - boundary_share)
    return PhaseObservation(
        view_name=view_name,
        interior_share=interior_share,
        off_bubble_share=off_share,
    )


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def _print_interpretation() -> None:
    print()
    print("Interpretation (exploratory; this is a thin tuning layer):")
    print(
        "  - The three ``toy:*`` rows are synthetic controls for the gauge: "
        "diffuse pressure should not trigger any action, diagnostic "
        "amplification with a clean edge should stabilize, and "
        "destructive amplification (strain leaking off-bubble) should "
        "hold rather than stabilize off-phase."
    )
    print(
        "  - The real-suite row is honest about PR #9 and PR #11: on the "
        "existing 22-instance suite the geometry transforms do not "
        "nucleate a clean diagnostic bubble. The action column is whatever "
        "the rule table actually selected, not a forced positive."
    )
    print(
        "  - The phase readout is *exploratory*. We do not claim Riordan "
        "family membership helps broadly; the readout is a hook for "
        "future transforms (the doc's pacing guardrail still applies)."
    )
    print(
        "  - The verdicts are advisory. This layer does not change the "
        "solver loop, does not add a new CoordinateView, and does not "
        "claim a positive result on SAT."
    )


def main() -> None:
    real_row, observations = _real_suite_case()
    rows = [
        real_row,
        _toy_diffuse(),
        _toy_diagnostic(),
        _toy_destructive(),
    ]
    print("Bubble tuning table")
    print(format_tuning_report(rows))
    print()
    print(format_phase_readout(read_phase(observations)))
    _print_interpretation()


if __name__ == "__main__":
    main()
