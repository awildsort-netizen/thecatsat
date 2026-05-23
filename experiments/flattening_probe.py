#!/usr/bin/env python3
"""Driver/report for the flattening probe.

Generates a small batch of toy SAT instances (planted-satisfiable
2-SAT/3-SAT plus a couple of structural-unsat formulas), runs the
probe in raw and spectral coordinate views, and prints a compact
table. This is an exploratory measurement, not a benchmark — see
``geometry/flattening_probe.py`` for the framing.

Run with: ``python experiments/flattening_probe.py``
"""

from __future__ import annotations

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.flattening_probe import FlatteningProbe, ProbeResult


def _planted(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, planted = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula, planted, True


def _structural_unsat(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("unsat", variables, clauses, k, rng)
    return formula, None, False


def _suite() -> list[tuple[str, list, int, bool | None]]:
    instances: list[tuple[str, list, int, bool | None]] = []
    # easy 2-SAT (planted)
    for seed in range(3):
        formula, _, sat = _planted(seed=100 + seed, variables=8, clauses=14, k=2)
        instances.append((f"2sat_easy_v8_c14_s{seed}", formula, 8, sat))
    # medium 3-SAT (planted, near sat threshold)
    for seed in range(3):
        formula, _, sat = _planted(seed=200 + seed, variables=12, clauses=42, k=3)
        instances.append((f"3sat_v12_c42_s{seed}", formula, 12, sat))
    # deliberately strained / unsat
    for seed in range(2):
        formula, _, sat = _structural_unsat(seed=300 + seed, variables=8, clauses=16, k=3)
        instances.append((f"unsat_struct_v8_c16_s{seed}", formula, 8, sat))
    return instances


def _summarize(results: list[ProbeResult]) -> None:
    header = f"{'instance':<28} {'planted':<8} {'view':<14} {'solved':<7} {'flips':<6} {'final_unsat':<12} {'strain[0→end]':<18}"
    print(header)
    print("-" * len(header))
    for result in results:
        for view_name, run in result.runs.items():
            traj = run.strain_trajectory
            strain_label = f"{traj[0]:.1f}→{traj[-1]:.1f}" if traj else "n/a"
            planted_label = (
                "sat" if result.planted_satisfiable is True
                else "unsat" if result.planted_satisfiable is False
                else "?"
            )
            print(
                f"{result.instance_id:<28} {planted_label:<8} {view_name:<14} "
                f"{'yes' if run.solved else 'no':<7} {run.flips:<6} "
                f"{run.final_unsatisfied:<12} {strain_label:<18}"
            )

    print()
    _aggregate_table(results)


def _aggregate_table(results: list[ProbeResult]) -> None:
    print("Aggregate (mean over instances; lower=better):")
    print(f"{'view':<20} {'mean flips':<12} {'solve rate':<12} {'mean final_unsat':<18}")
    print("-" * 64)
    by_view: dict[str, list] = {}
    for result in results:
        for view_name, run in result.runs.items():
            by_view.setdefault(view_name, []).append(run)
    for view_name, runs in by_view.items():
        flips = [r.flips for r in runs]
        solved = [1 if r.solved else 0 for r in runs]
        final_unsat = [r.final_unsatisfied for r in runs]
        print(
            f"{view_name:<20} {statistics.mean(flips):<12.1f} "
            f"{statistics.mean(solved):<12.2f} {statistics.mean(final_unsat):<18.2f}"
        )

    print()
    _head_to_head(results)


def _head_to_head(results: list[ProbeResult]) -> None:
    """Per-instance comparison: how often does the spectral view reach
    a lower final-unsat than the raw view? This is the closest thing
    to a directional signal on toy data.
    """
    wins_spectral = 0
    wins_raw = 0
    ties = 0
    for result in results:
        runs = result.runs
        if "raw" not in runs:
            continue
        raw = runs["raw"]
        spectral_runs = [r for name, r in runs.items() if name != "raw"]
        for spectral in spectral_runs:
            if spectral.final_unsatisfied < raw.final_unsatisfied:
                wins_spectral += 1
            elif spectral.final_unsatisfied > raw.final_unsatisfied:
                wins_raw += 1
            else:
                ties += 1
    print(
        f"Head-to-head on final_unsat: spectral_better={wins_spectral} "
        f"raw_better={wins_raw} ties={ties}"
    )
    print()
    print(
        "Reminder: this is a probe, not a proof. The two views differ only "
        "in *which variable they pick to flip*; the same clause-check budget "
        "is paid on every step. A positive head-to-head on tiny instances "
        "is suggestive, not conclusive."
    )


def main() -> None:
    instances = _suite()
    probe = FlatteningProbe(max_flips=200, seed=7)
    results: list[ProbeResult] = []
    for instance_id, formula, n_vars, planted in instances:
        results.append(
            probe.run(
                formula=formula,
                n_vars=n_vars,
                instance_id=instance_id,
                planted_satisfiable=planted,
            )
        )
    _summarize(results)


if __name__ == "__main__":
    main()
