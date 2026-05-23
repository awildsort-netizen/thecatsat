#!/usr/bin/env python3
"""Driver/report for the Riordan probe.

Runs a seeded SAT suite — easy 2-SAT, mid-density 3-SAT, an expanded
near-threshold 3-SAT slice, and structural-UNSAT guardrails — and
compares five coordinate views: raw, spectral, Pascal, signed Pascal,
and Sierpinski (Pascal mod 2). Prints a per-instance table, an
aggregate, a head-to-head summary against the raw baseline, a
per-family breakdown, motion-type labels on the interesting cases,
and a compact deterministic trace of the first few flips per view.

Run with: ``python experiments/riordan_probe.py``
"""

from __future__ import annotations

import os
import random
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.flattening_probe import ProbeResult
from geometry.riordan_probe import (
    RiordanProbe,
    compact_trace,
    family_of,
    head_to_head,
    head_to_head_by_family,
    motion_label,
)


def _planted(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, planted = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula, planted, True


def _structural_unsat(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("unsat", variables, clauses, k, rng)
    return formula, None, False


# Near-threshold 3-SAT parameter grid. The random 3-SAT phase
# transition sits near a clause-to-variable ratio of ~4.26; we sweep a
# small deterministic grid around it. Kept small on purpose — this is
# a paced expansion, not a benchmark.
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

    # Easy 2-SAT — both views should solve immediately.
    for seed in range(3):
        formula, _, sat = _planted(seed=100 + seed, variables=8, clauses=14, k=2)
        instances.append((f"2sat_easy_v8_c14_s{seed}", formula, 8, sat))

    # Original mid-density 3-SAT cases (kept so prior numbers stay
    # comparable across the PR history).
    for seed in range(3):
        formula, _, sat = _planted(seed=200 + seed, variables=12, clauses=42, k=3)
        instances.append((f"3sat_v12_c42_s{seed}", formula, 12, sat))

    # Expanded near-threshold 3-SAT slice. Two seeds per (n, ratio)
    # cell across the small grid above => +14 instances. Deterministic
    # seeds; kept clamped to small n so total runtime stays ~1–2s.
    base_seed = 400
    for n_vars, ratio in _NEAR_THRESHOLD_GRID:
        n_clauses = max(1, int(round(n_vars * ratio)))
        for offset in range(2):
            seed = base_seed
            base_seed += 1
            formula, _, sat = _planted(
                seed=seed, variables=n_vars, clauses=n_clauses, k=3,
            )
            label = (
                f"3sat_threshold_v{n_vars}_r{ratio:.1f}_s{offset}"
            )
            instances.append((label, formula, n_vars, sat))

    # Structural-UNSAT guardrails.
    for seed in range(2):
        formula, _, sat = _structural_unsat(seed=300 + seed, variables=8, clauses=16, k=3)
        instances.append((f"unsat_struct_v8_c16_s{seed}", formula, 8, sat))

    return instances


def _summarize(results: list[ProbeResult]) -> None:
    header = (
        f"{'instance':<36} {'planted':<6} {'view':<16} "
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
                f"{result.instance_id:<36} {planted_label:<6} {view_name:<16} "
                f"{'yes' if run.solved else 'no':<7} {run.flips:<6} "
                f"{run.final_unsatisfied:<6} {strain_label:<16}"
            )

    print()
    _aggregate_table(results)
    print()
    _head_to_head(results)
    print()
    _per_family_breakdown(results)
    print()
    _motion_breakdown(results)
    print()
    _interesting_cases(results)
    print()
    print(
        "Reminder: this is a probe, not a proof. All views pay the same "
        "clause-check budget per flip; the only thing that changes is "
        "which variable each view picks to flip. A neutral or negative "
        "head-to-head is just as useful a result as a positive one. The "
        "instances here are small on purpose — we are reading signal "
        "shape, not benchmark numbers."
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


def _per_family_breakdown(results: list[ProbeResult]) -> None:
    print("Per-family head-to-head vs raw baseline:")
    by_family = head_to_head_by_family(results, baseline="raw")
    families = sorted(by_family)
    header = f"{'family':<18} {'view':<18} {'wins':<6} {'ties':<6} {'losses':<6}"
    print(header)
    print("-" * len(header))
    for family in families:
        view_summary = by_family[family]
        for view_name in sorted(view_summary):
            row = view_summary[view_name]
            print(
                f"{family:<18} {view_name:<18} "
                f"{row['wins']:<6} {row['ties']:<6} {row['losses']:<6}"
            )


def _motion_breakdown(results: list[ProbeResult]) -> None:
    print("Motion-label distribution per view (vs raw baseline):")
    header = f"{'view':<18} label counts"
    print(header)
    print("-" * 60)
    by_view: dict[str, Counter] = {}
    for result in results:
        if "raw" not in result.runs:
            continue
        base = result.runs["raw"]
        for view_name, run in result.runs.items():
            if view_name == "raw":
                continue
            by_view.setdefault(view_name, Counter())[motion_label(base, run)] += 1
    for view_name in sorted(by_view):
        counts = by_view[view_name]
        rendered = ", ".join(f"{label}={n}" for label, n in sorted(counts.items()))
        print(f"{view_name:<18} {rendered}")


def _interesting_cases(results: list[ProbeResult]) -> None:
    """Compact trace on cases that aren't a flat tie across views.

    A case is "interesting" if any non-raw view produced a different
    motion label than ``matches_raw``. We print one row per
    (instance, view) for those.
    """
    print("Interesting cases (any view differs from raw on final_unsat or speed):")
    header = (
        f"{'instance':<36} {'view':<16} {'label':<22} trace"
    )
    print(header)
    print("-" * len(header))
    any_printed = False
    for result in results:
        if "raw" not in result.runs:
            continue
        base = result.runs["raw"]
        rows: list[tuple[str, str, str]] = []
        for view_name, run in result.runs.items():
            if view_name == "raw":
                continue
            label = motion_label(base, run)
            if label == "matches_raw":
                continue
            rows.append((view_name, label, compact_trace(run)))
        if not rows:
            continue
        any_printed = True
        # Print one anchoring row for raw, then the diverging views.
        print(
            f"{result.instance_id:<36} {'raw':<16} "
            f"{'baseline':<22} {compact_trace(base)}"
        )
        for view_name, label, trace in rows:
            print(f"{'':<36} {view_name:<16} {label:<22} {trace}")
    if not any_printed:
        print("(none — all views matched raw on every instance)")


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
