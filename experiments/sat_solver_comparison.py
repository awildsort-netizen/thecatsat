#!/usr/bin/env python3
"""Compare native composer/furnace solver against simple baselines.

The three solvers — brute-force, DPLL, and the furnace driven through
``Composer.iterate`` — are registered as operators on a shared SAT
benchmark composer (see :mod:`sat_benchmarks`). All three consume the
same ``(formula, variables)`` input and produce a common
``SolveResult``, so the comparison is composer-native: ask the DAG for
``brute_result``, ``dpll_result``, ``furnace_benchmark_result`` and
read off uniform fields plus an optional ``metabolism`` mapping for
the geodesic accounting that only the furnace fills in.

Run with: ``python experiments/sat_solver_comparison.py``
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_furnace
from sat_benchmarks import build_sat_benchmark_composer


def planted_instance(seed: int, variables: int, clauses: int, k: int = 3):
    rng = random.Random(seed)
    formula, planted = sat_furnace.generate_formula(
        "sat", variables, clauses, k, rng
    )
    return formula, planted


def _fmt_opt(value, fmt: str = "{:.3f}") -> str:
    if value is None:
        return "n/a"
    return fmt.format(value)


def run_one_instance(*, label: str, seed: int, variables: int, clauses: int,
                     furnace_steps: int, furnace_seed: int) -> None:
    print()
    print("=" * 78)
    print(f"INSTANCE: {label}  (variables={variables} clauses={clauses}  "
          f"planted-SAT seed={seed})")
    print("=" * 78)
    formula, _ = planted_instance(seed, variables, clauses)
    composer = build_sat_benchmark_composer()
    out = composer.run(
        ("brute_result", "dpll_result", "furnace_benchmark_result"),
        {
            "formula": formula,
            "variables": variables,
            "furnace_steps": furnace_steps,
            "furnace_seed": furnace_seed,
        },
    )
    brute = out["brute_result"]
    dpll = out["dpll_result"]
    furnace = out["furnace_benchmark_result"]

    print(f"  {'solver':<14} {'solved':>6} {'time_s':>9} "
          f"{'work':>9} {'final_unsat':>11}  work_metric")
    print(f"  {'-'*14} {'-'*6} {'-'*9} {'-'*9} {'-'*11}  {'-'*30}")
    for r in (brute, dpll, furnace):
        print(f"  {r.solver_name:<14} {str(r.solved):>6} "
              f"{r.wall_time_s:>9.4f} "
              f"{r.work_units:>9} "
              f"{r.final_unsatisfied:>11}  {r.work_metric}")

    if furnace.metabolism:
        print()
        print(f"  furnace geodesic accounting:")
        for key, value in furnace.metabolism.items():
            shown = _fmt_opt(value) if isinstance(value, float) or value is None else value
            print(f"    {key:<33}: {shown}")


def main() -> None:
    print("Native composer/furnace solver vs simple baselines")
    print("(brute-force is honest dependency-free; DPLL is recursive unit-prop)")
    print("All three solvers run as operators on the benchmark composer.")
    print()

    cases = [
        # (label, seed, variables, clauses, furnace_steps, furnace_seed)
        ("tiny",   7, 8,  24, 25, 11),
        ("small", 13, 10, 36, 30, 17),
        ("med",   23, 12, 48, 40, 19),
    ]
    for label, seed, variables, clauses, fs, fseed in cases:
        run_one_instance(
            label=label,
            seed=seed,
            variables=variables,
            clauses=clauses,
            furnace_steps=fs,
            furnace_seed=fseed,
        )

    print()
    print("Caveats")
    print("  * Different vocabularies: brute / DPLL count discrete work to a")
    print("    yes/no; the furnace pays continuous Hamming distance to drive")
    print("    incompatibility down. Direct time/work comparisons favor the")
    print("    baselines on tiny instances — they are the right tool for SAT-")
    print("    as-decision. The furnace's value here is the *trajectory*.")
    print("  * The furnace is given a fixed step budget and does NOT see the")
    print("    planted assignment (planted_assignment=None). On unsat-by-bad-")
    print("    luck the furnace can leave final_unsatisfied > 0 — that is the")
    print("    distance still owed, not a wrong answer to a yes/no.")
    print("  * No external SAT solver is invoked — we deliberately avoid heavy")
    print("    deps. A MiniSAT/CaDiCaL row would slot into the same composer")
    print("    as another operator with the same SolveResult shape.")
    print()
    print("done.")


if __name__ == "__main__":
    main()
