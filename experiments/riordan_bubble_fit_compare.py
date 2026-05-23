#!/usr/bin/env python3
"""Compare raw / transformed / gated / fitted composed strategies.

This driver extends ``experiments/strategy_compare.py`` with one more
preset: :func:`strategy.fitted_composer`, which uses the Riordan bubble
fitter to pick a transform by *bubble stability* — never by SAT
outcome.

The comparison runs on three slices:

1. The same flattening-probe suite (2-SAT, 3-SAT, structural-unsat).
   We expect honest neutrality here — the suite was not designed to
   produce stable bubbles, so the fitter should mostly choose
   ``identity`` (raw) and the numbers should match raw.
2. A synthetic stable-bubble control where strain is genuinely
   concentrated. Here we expect the fitter to pick a Riordan variant
   when it scores higher than identity, and to *not* veto.
3. A synthetic destructive control (off-phase, churning trace). The
   fitter must veto and fall back to identity.

We print the per-preset table, the aggregate, and — most importantly —
the audit log: how often the fitter selected which candidate, and how
often it vetoed.
"""

from __future__ import annotations

import collections
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from geometry.bubble_lifecycle import inflate_bubble, seed_from_strain
from geometry.flattening_probe import _per_variable_strain
from geometry.riordan_bubble_fit import fit, format_fit_table
import strategy


# --------------------------------------------------------------------------- #
# Suite reuse                                                                 #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Field-seed helpers (same shape as strategy_compare)                         #
# --------------------------------------------------------------------------- #


def _initial_strain(formula, n_vars: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    assignment = [rng.choice([False, True]) for _ in range(n_vars)]
    return list(map(float, _per_variable_strain(formula, assignment, n_vars)))


def _bubble_field_seed_from(formula, n_vars: int, seed: int) -> dict:
    profile = _initial_strain(formula, n_vars, seed)
    seed_obj = seed_from_strain(profile)
    if seed_obj is None:
        return {}
    bubble = inflate_bubble(profile, seed_obj)
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


# --------------------------------------------------------------------------- #
# Four-way run                                                                #
# --------------------------------------------------------------------------- #


def _run_four(instance_id, formula, n_vars, max_flips, seed):
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
    fitted = strategy.composed_local_search(
        formula, n_vars, strategy.fitted_composer(),
        composer_name="fitted", max_flips=max_flips, seed=seed,
    )
    return (instance_id, raw, xform, gated, fitted)


def _summarize(rows):
    header = (
        f"{'instance':<22} "
        f"{'composer':<14} {'solved':<7} {'flips':<6} {'final_unsat':<11}"
    )
    print(header)
    print("-" * len(header))
    for instance_id, raw, xform, gated, fitted in rows:
        for name, run in (
            ("raw", raw),
            ("transformed", xform),
            ("gated", gated),
            ("fitted", fitted),
        ):
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
        ("fitted", lambda t: t[4]),
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
    fitter_selections: collections.Counter = collections.Counter()
    fitter_vetoes = 0
    fitter_steps = 0
    for _id, _r, _x, _g, fitted in rows:
        for mark in fitted.field_marks:
            fitter_steps += 1
            sel = mark.get("fitted_selected")
            if sel:
                fitter_selections[sel] += 1
            if mark.get("veto_transformed"):
                fitter_vetoes += 1
    print("fitted audit:")
    print(f"  steps={fitter_steps}  vetoes={fitter_vetoes}")
    for name, count in fitter_selections.most_common():
        print(f"  selected[{name}] = {count}")


# --------------------------------------------------------------------------- #
# Synthetic stable-bubble + destructive controls for the fitter alone         #
# --------------------------------------------------------------------------- #


def _print_stable_control() -> None:
    # Strain concentrated on the first three indices — a stable bubble.
    print("\n--- synthetic stable-bubble strain (control) ---")
    strain = [5.0, 4.5, 4.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    decision = fit(strain)
    print(format_fit_table(decision))
    print(f"interpretation: fitter selected '{decision.selected}', "
          f"veto={decision.veto}")


def _print_diffuse_control() -> None:
    print("\n--- synthetic diffuse strain (control) ---")
    strain = [1.0] * 8
    decision = fit(strain)
    print(format_fit_table(decision))
    print(f"interpretation: fitter selected '{decision.selected}', "
          f"veto={decision.veto}")


def _print_destructive_control() -> None:
    print("\n--- synthetic destructive strain trace (control) ---")
    profile = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5]
    trace = (
        [0.1, 0.1, 0.1, 0.1, 5.0, 4.0, 3.0, 2.0],
        [4.0, 0.1, 0.1, 5.0, 0.1, 3.0, 2.0, 0.1],
        [0.1, 5.0, 0.1, 0.1, 0.1, 3.0, 4.0, 2.0],
    )
    decision = fit(profile, trace=trace)
    print(format_fit_table(decision))
    print(f"interpretation: fitter selected '{decision.selected}', "
          f"veto={decision.veto}")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(max_flips: int = 80, seed: int = 7) -> None:
    rows = [
        _run_four(instance_id, formula, n_vars, max_flips, seed)
        for instance_id, formula, n_vars, _sat in _suite()
    ]
    _summarize(rows)
    _print_stable_control()
    _print_diffuse_control()
    _print_destructive_control()


if __name__ == "__main__":
    main()
