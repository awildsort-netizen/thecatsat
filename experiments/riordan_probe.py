#!/usr/bin/env python3
"""Driver/report for the Riordan probe.

Runs the same seeded SAT suite as ``experiments/flattening_probe.py``
but compares five coordinate views: raw, spectral, Pascal, signed
Pascal, and Sierpinski (Pascal mod 2). Prints a per-instance table
and a head-to-head summary against the raw baseline.

Run with: ``python experiments/riordan_probe.py``
"""

from __future__ import annotations

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.riordan_probe import RiordanProbe, head_to_head
from geometry.flattening_probe import ProbeResult


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
    for seed in range(3):
        formula, _, sat = _planted(seed=100 + seed, variables=8, clauses=14, k=2)
        instances.append((f"2sat_easy_v8_c14_s{seed}", formula, 8, sat))
    for seed in range(3):
        formula, _, sat = _planted(seed=200 + seed, variables=12, clauses=42, k=3)
        instances.append((f"3sat_v12_c42_s{seed}", formula, 12, sat))
    for seed in range(2):
        formula, _, sat = _structural_unsat(seed=300 + seed, variables=8, clauses=16, k=3)
        instances.append((f"unsat_struct_v8_c16_s{seed}", formula, 8, sat))
    return instances


def _summarize(results: list[ProbeResult]) -> None:
    header = (
        f"{'instance':<28} {'planted':<6} {'view':<16} "
        f"{'solved':<7} {'flips':<6} {'final':<6} {'strain[0→end]':<16}"
    )
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
                f"{result.instance_id:<28} {planted_label:<6} {view_name:<16} "
                f"{'yes' if run.solved else 'no':<7} {run.flips:<6} "
                f"{run.final_unsatisfied:<6} {strain_label:<16}"
            )

    print()
    _aggregate_table(results)
    print()
    _head_to_head(results)
    print()
    print(
        "Reminder: this is a probe, not a proof. All views pay the same "
        "clause-check budget per flip; the only thing that changes is "
        "which variable each view picks to flip. A neutral or negative "
        "head-to-head is just as useful a result as a positive one."
    )


def _aggregate_table(results: list[ProbeResult]) -> None:
    print("Aggregate (mean over instances; lower=better):")
    print(f"{'view':<18} {'mean flips':<12} {'solve rate':<12} {'mean final_unsat':<18}")
    print("-" * 62)
    by_view: dict[str, list] = {}
    for result in results:
        for view_name, run in result.runs.items():
            by_view.setdefault(view_name, []).append(run)
    for view_name, runs in by_view.items():
        flips = [r.flips for r in runs]
        solved = [1 if r.solved else 0 for r in runs]
        final_unsat = [r.final_unsatisfied for r in runs]
        print(
            f"{view_name:<18} {statistics.mean(flips):<12.1f} "
            f"{statistics.mean(solved):<12.2f} {statistics.mean(final_unsat):<18.2f}"
        )


def _head_to_head(results: list[ProbeResult]) -> None:
    print("Head-to-head vs raw baseline (final_unsat; lower=better):")
    print(f"{'view':<18} {'wins':<6} {'ties':<6} {'losses':<6}")
    print("-" * 40)
    summary = head_to_head(results, baseline="raw")
    for view_name in sorted(summary):
        row = summary[view_name]
        print(f"{view_name:<18} {row['wins']:<6} {row['ties']:<6} {row['losses']:<6}")


def main() -> None:
    instances = _suite()
    probe = RiordanProbe(max_flips=200, seed=7)
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
