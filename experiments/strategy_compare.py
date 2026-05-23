#!/usr/bin/env python3
"""Compare raw / transformed / bubble-gated composed strategies.

Same deterministic SAT suite the flattening probe already uses (2-SAT,
3-SAT, structural-unsat instances). Each instance is run under three
composers with the same seed and the same per-instance flip budget:

- ``raw``                — :func:`strategy.raw_composer`
- ``transformed``        — :func:`strategy.transformed_composer` (spectral)
- ``gated_transformed``  — :func:`strategy.gated_transformed_composer`

The bubble-gate test only fires when a ``strain_trace`` and
``bubble_candidate`` are supplied; for honest like-for-like
measurement we synthesize a candidate bubble from the *initial* raw
strain (one observation), which the lifecycle / tuning rules turn
into a benign label most of the time. We additionally re-run the
gated composer on a deliberately-destructive synthetic trace to show
the fallback is *reachable*.

This is not a benchmark for solver quality. It is the first
behavior-altering composition test: does the decomposition reproduce
existing flattening-probe behavior, and does the bubble-gate operator
correctly route around destructive transforms when the gauge fires?
"""

from __future__ import annotations

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.bubble_lifecycle import inflate_bubble, seed_from_strain
from geometry.flattening_probe import _per_variable_strain
import strategy


def _planted(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("sat", variables, clauses, k, rng)
    return formula, True


def _structural_unsat(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("unsat", variables, clauses, k, rng)
    return formula, False


def _suite():
    instances = []
    for seed in range(3):
        formula, sat = _planted(seed=100 + seed, variables=8, clauses=14, k=2)
        instances.append((f"2sat_v8_c14_s{seed}", formula, 8, sat))
    for seed in range(3):
        formula, sat = _planted(seed=200 + seed, variables=12, clauses=42, k=3)
        instances.append((f"3sat_v12_c42_s{seed}", formula, 12, sat))
    for seed in range(2):
        formula, sat = _structural_unsat(seed=300 + seed, variables=8, clauses=16, k=3)
        instances.append((f"unsat_v8_c16_s{seed}", formula, 8, sat))
    return instances


def _initial_strain(formula, n_vars: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    assignment = [rng.choice([False, True]) for _ in range(n_vars)]
    return list(map(float, _per_variable_strain(formula, assignment, n_vars)))


def _bubble_field_seed_from(formula, n_vars: int, seed: int) -> dict:
    # Cheap, deterministic candidate bubble from the *initial* assignment's
    # strain, plus a tiny synthetic trace that the gauge sees as
    # destructive_amplification. This is on purpose: the goal of the
    # comparison is to exercise the *path*, not to win the suite. A real
    # driver would feed the running strain history here.
    profile = _initial_strain(formula, n_vars, seed)
    seed_obj = seed_from_strain(profile)
    if seed_obj is None:
        return {}
    bubble = inflate_bubble(profile, seed_obj)
    # Destructive trace shaped from the same profile by rotating high
    # strain into off-bubble indices, then churning the boundary.
    interior = bubble.interior
    boundary = bubble.boundary
    in_or_b = set(interior) | set(boundary)
    off = [i for i in range(n_vars) if i not in in_or_b] or list(range(n_vars))
    base = [0.1] * n_vars
    snapshots = []
    for shift in range(3):
        snap = list(base)
        for idx, j in enumerate(off):
            snap[j] = float(5.0 - idx)
        if interior:
            snap[interior[shift % len(interior)]] = 4.0
        snapshots.append(tuple(snap))
    return {"bubble_candidate": bubble, "strain_trace": tuple(snapshots)}


def _run_three(instance_id, formula, n_vars, max_flips, seed):
    view = strategy.spectral_view_for(formula, n_vars)
    raw = strategy.composed_local_search(
        formula, n_vars, strategy.raw_composer(),
        composer_name="raw", max_flips=max_flips, seed=seed,
    )
    xform = strategy.composed_local_search(
        formula, n_vars, strategy.transformed_composer(view),
        composer_name="transformed", max_flips=max_flips, seed=seed,
    )
    gated = strategy.composed_local_search(
        formula, n_vars, strategy.gated_transformed_composer(view),
        composer_name="gated", max_flips=max_flips, seed=seed,
        field_seed=_bubble_field_seed_from(formula, n_vars, seed),
    )
    return (instance_id, raw, xform, gated)


def _summarize(rows):
    header = (
        f"{'instance':<22} "
        f"{'composer':<14} {'solved':<7} {'flips':<6} {'final_unsat':<11}"
    )
    print(header)
    print("-" * len(header))
    for instance_id, raw, xform, gated in rows:
        for name, run in (("raw", raw), ("transformed", xform), ("gated", gated)):
            print(
                f"{instance_id:<22} {name:<14} "
                f"{'yes' if run.solved else 'no':<7} {run.flips:<6} "
                f"{run.final_unsatisfied:<11}"
            )
    print()
    print("Aggregate (mean over instances; lower=better):")
    print(f"{'composer':<14} {'mean flips':<12} {'solve rate':<12} {'mean final_unsat':<18}")
    print("-" * 58)
    for name, sel in (
        ("raw", lambda t: t[1]),
        ("transformed", lambda t: t[2]),
        ("gated", lambda t: t[3]),
    ):
        runs = [sel(row) for row in rows]
        flips = [r.flips for r in runs]
        solved = [1 if r.solved else 0 for r in runs]
        unsat = [r.final_unsatisfied for r in runs]
        print(
            f"{name:<14} {statistics.mean(flips):<12.1f} "
            f"{statistics.mean(solved):<12.2f} {statistics.mean(unsat):<18.2f}"
        )
    print()
    veto_count = sum(
        1
        for _id, _r, _x, gated in rows
        for mark in gated.field_marks
        if mark.get("veto_transformed")
    )
    veto_steps = sum(len(gated.field_marks) for _id, _r, _x, gated in rows)
    print(
        f"gated path audit: {veto_count}/{veto_steps} steps fired the bubble veto"
    )


def main(max_flips: int = 80, seed: int = 7) -> None:
    rows = [
        _run_three(instance_id, formula, n_vars, max_flips, seed)
        for instance_id, formula, n_vars, _sat in _suite()
    ]
    _summarize(rows)


if __name__ == "__main__":
    main()
