#!/usr/bin/env python3
"""Driver/report for the transform litmus.

Reuses the same 22-instance seeded SAT suite that
``experiments/riordan_probe.py`` runs (easy 2-SAT, mid-density 3-SAT,
near-threshold 3-SAT, structural-UNSAT guardrails). For each
non-raw view on each instance it computes a
:class:`geometry.transform_litmus.LitmusReading`: localization
statistics on the residual per-variable strain, plus a verdict in the
small label set:

    resolved_to_boundary, localized_but_unstable, moved_singularity,
    amplified_pathology, no_change

Then it prints the per-(instance, view) table, a per-view verdict
breakdown, and the *association* table: for each verdict, the fraction
of (instance, view) pairs at that verdict where the view actually
solved (or strictly reduced ``final_unsatisfied``). That is the SAT
version of the tangent test's question — does what the litmus calls
"localized" overlap with what the solver calls "more addressable".

Run with: ``python experiments/transform_litmus.py``
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.flattening_probe import ProbeResult
from geometry.riordan_probe import RiordanProbe
from geometry.transform_litmus import (
    LITMUS_VERDICTS,
    LitmusReading,
    litmus_for_result,
    summarize,
)


PROBE_SEED = 7
MAX_FLIPS = 200


def _planted(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, planted = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula, planted, True


def _structural_unsat(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("unsat", variables, clauses, k, rng)
    return formula, None, False


# Same parameter grid as experiments/riordan_probe.py.
_NEAR_THRESHOLD_GRID: tuple[tuple[int, float], ...] = (
    (10, 4.0),
    (10, 4.3),
    (10, 4.6),
    (12, 4.0),
    (12, 4.3),
    (12, 4.6),
    (14, 4.3),
)


def _suite() -> list[tuple[str, list, int, bool | None]]:
    instances: list[tuple[str, list, int, bool | None]] = []
    for seed in range(3):
        formula, _, sat = _planted(seed=100 + seed, variables=8, clauses=14, k=2)
        instances.append((f"2sat_easy_v8_c14_s{seed}", formula, 8, sat))
    for seed in range(3):
        formula, _, sat = _planted(seed=200 + seed, variables=12, clauses=42, k=3)
        instances.append((f"3sat_v12_c42_s{seed}", formula, 12, sat))
    base_seed = 400
    for n_vars, ratio in _NEAR_THRESHOLD_GRID:
        n_clauses = max(1, int(round(n_vars * ratio)))
        for offset in range(2):
            seed = base_seed
            base_seed += 1
            formula, _, sat = _planted(
                seed=seed, variables=n_vars, clauses=n_clauses, k=3,
            )
            label = f"3sat_threshold_v{n_vars}_r{ratio:.1f}_s{offset}"
            instances.append((label, formula, n_vars, sat))
    for seed in range(2):
        formula, _, sat = _structural_unsat(seed=300 + seed, variables=8, clauses=16, k=3)
        instances.append((f"unsat_struct_v8_c16_s{seed}", formula, 8, sat))
    return instances


def _run_suite() -> tuple[
    list[tuple[str, list, int, bool | None]],
    list[ProbeResult],
    list[LitmusReading],
]:
    instances = _suite()
    probe = RiordanProbe(max_flips=MAX_FLIPS, seed=PROBE_SEED)
    results: list[ProbeResult] = []
    readings: list[LitmusReading] = []
    for instance_id, formula, n_vars, planted in instances:
        result = probe.run(
            formula=formula,
            n_vars=n_vars,
            instance_id=instance_id,
            planted_satisfiable=planted,
        )
        results.append(result)
        readings.extend(
            litmus_for_result(
                formula=formula,
                n_vars=n_vars,
                seed=PROBE_SEED,
                result=result,
            )
        )
    return instances, results, readings


def _print_per_reading(readings: list[LitmusReading]) -> None:
    print("Per (instance, view) litmus readings:")
    header = (
        f"{'instance':<36} {'view':<16} {'base_solved':<11} {'view_solved':<11} "
        f"{'b_unsat':<8} {'v_unsat':<8} {'b_top3':<8} {'v_top3':<8} "
        f"{'b_supp':<7} {'v_supp':<7} verdict"
    )
    print(header)
    print("-" * len(header))
    for r in readings:
        print(
            f"{r.instance_id:<36} {r.view_name:<16} "
            f"{'yes' if r.baseline_solved else 'no':<11} "
            f"{'yes' if r.view_solved else 'no':<11} "
            f"{r.baseline_final_unsat:<8} {r.view_final_unsat:<8} "
            f"{r.baseline_localization.top_k_share:<8.2f} "
            f"{r.view_localization.top_k_share:<8.2f} "
            f"{r.baseline_localization.support:<7} "
            f"{r.view_localization.support:<7} "
            f"{r.verdict}"
        )


def _print_per_view_verdicts(readings: list[LitmusReading]) -> None:
    print("Per-view verdict counts:")
    header = f"{'view':<18}" + "".join(f"{v:<24}" for v in LITMUS_VERDICTS)
    print(header)
    print("-" * len(header))
    by_view: dict[str, dict[str, int]] = {}
    for r in readings:
        row = by_view.setdefault(r.view_name, {v: 0 for v in LITMUS_VERDICTS})
        row[r.verdict] += 1
    for view_name in sorted(by_view):
        row = by_view[view_name]
        cells = "".join(f"{row[v]:<24}" for v in LITMUS_VERDICTS)
        print(f"{view_name:<18}{cells}")


def _print_association(readings: list[LitmusReading]) -> None:
    summary = summarize(readings)
    print("Association: verdict ↔ solve outcome")
    print(
        f"  (over {summary.n} (instance, non-raw-view) pairs across the suite)"
    )
    header = f"{'verdict':<24} {'count':<7} {'solve_rate':<12} {'improve_rate':<12}"
    print(header)
    print("-" * len(header))
    for v in LITMUS_VERDICTS:
        c = summary.verdict_counts[v]
        sr = summary.verdict_to_solve_rate[v]
        ir = summary.verdict_to_improvement_rate[v]
        print(f"{v:<24} {c:<7} {sr:<12.2f} {ir:<12.2f}")


def _print_interpretation(readings: list[LitmusReading]) -> None:
    summary = summarize(readings)
    n = summary.n
    if n == 0:
        return
    resolved = summary.verdict_counts["resolved_to_boundary"]
    localized = summary.verdict_counts["localized_but_unstable"]
    moved = summary.verdict_counts["moved_singularity"]
    amplified = summary.verdict_counts["amplified_pathology"]
    print("Interpretation (read with caution; this is a probe, not a proof):")
    print(
        f"  - {resolved}/{n} pairs landed on 'resolved_to_boundary' — the SAT "
        "analogue of the tangent lift's clean win. Solve rate at this verdict: "
        f"{summary.verdict_to_solve_rate['resolved_to_boundary']:.2f}."
    )
    print(
        f"  - {localized}/{n} pairs landed on 'localized_but_unstable' — residual "
        "strain collapsed onto a smaller set but the solver still didn't "
        "finish. Solve rate at this verdict: "
        f"{summary.verdict_to_solve_rate['localized_but_unstable']:.2f}."
    )
    print(
        f"  - {moved}/{n} pairs moved the singularity sideways (similar "
        "residual, different variables carrying it). Solve rate: "
        f"{summary.verdict_to_solve_rate['moved_singularity']:.2f}."
    )
    print(
        f"  - {amplified}/{n} pairs amplified pathology — the transform made "
        "things worse, not more addressable."
    )
    print(
        "  - The SAT analogue of the tangent test is the spread between the "
        "solve rate at 'resolved_to_boundary'/'localized_but_unstable' vs "
        "the solve rate at 'amplified_pathology'. A real signal looks like "
        "the first two being meaningfully higher than the last."
    )


def _print_reminder() -> None:
    print(
        "Reminder: this litmus measures whether a transform turns nonlocal "
        "SAT pathology into localized typed strain, and whether that "
        "correlates with solving. It does not claim the transforms solve "
        "SAT. A negative correlation here is just as informative as a "
        "positive one — it tells us the localization signal alone does "
        "not predict solver improvement on this suite."
    )


def main() -> None:
    _instances, _results, readings = _run_suite()
    _print_per_reading(readings)
    print()
    _print_per_view_verdicts(readings)
    print()
    _print_association(readings)
    print()
    _print_interpretation(readings)
    print()
    _print_reminder()


if __name__ == "__main__":
    main()
