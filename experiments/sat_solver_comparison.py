#!/usr/bin/env python3
"""Compare the native composer/furnace solver against simple baselines.

The native solver does not return a yes/no — it pays *distance* through
incompatibility. That makes a direct A/B against a classic SAT solver
slightly off-key: brute-force and DPLL count decisions and bottom out at
"satisfying assignment found", while the furnace reports a trajectory in
H/F/I space. So we report both vocabularies side-by-side on the same
small planted instances:

  - solved? (boolean)
  - wall_time_s
  - work counter (assignments_checked for brute-force, decisions for DPLL,
    composer.iterate step count for the furnace)
  - final unsatisfied clauses (always 0 for brute / DPLL on SAT instances;
    interesting for the furnace under stall)
  - distance_paid_per_resolved (furnace only — the geodesic-cost reading)

The furnace is driven through ``composer.iterate`` rather than
``run_furnace`` so the experiment is composer-native: the cycle in the
per-epoch dependency graph (next_spins -> spins, samples -> prev_samples)
is the rename contract, and "run N epochs" is a first-class request.

Run with: ``python experiments/sat_solver_comparison.py``
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_composer
import sat_furnace
import sat_metabolism as sm
from sat_furnace import _EPOCH_TARGETS, _init_epoch_context, clause_satisfied


# ---------------------------------------------------------------------------
# Baseline 1 — brute-force exhaustive assignment search.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BruteResult:
    solved: bool
    assignments_checked: int
    wall_time_s: float
    final_unsatisfied: int


def _unsat_count(formula, assignment) -> int:
    return sum(0 if clause_satisfied(c, assignment) else 1 for c in formula)


def brute_force_solve(formula, variables: int) -> BruteResult:
    """Enumerate the 2**variables truth table. Honest, dependency-free."""
    start = time.perf_counter()
    best_unsat = len(formula)
    checked = 0
    for mask in range(1 << variables):
        assignment = [bool((mask >> i) & 1) for i in range(variables)]
        checked += 1
        unsat = _unsat_count(formula, assignment)
        if unsat < best_unsat:
            best_unsat = unsat
        if unsat == 0:
            return BruteResult(
                solved=True,
                assignments_checked=checked,
                wall_time_s=time.perf_counter() - start,
                final_unsatisfied=0,
            )
    return BruteResult(
        solved=False,
        assignments_checked=checked,
        wall_time_s=time.perf_counter() - start,
        final_unsatisfied=best_unsat,
    )


# ---------------------------------------------------------------------------
# Baseline 2 — simple DPLL (unit propagation + first-unassigned branch).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DPLLResult:
    solved: bool
    decisions: int
    unit_propagations: int
    wall_time_s: float
    final_unsatisfied: int


def _dpll(formula, assignment, stats) -> bool:
    """Recursive DPLL. ``assignment`` is a list of ``True``/``False``/``None``."""
    # unit propagation
    changed = True
    while changed:
        changed = False
        for clause in formula:
            unassigned = []
            satisfied = False
            for var, neg in clause:
                v = assignment[var]
                if v is None:
                    unassigned.append((var, neg))
                else:
                    lit_value = (not v) if neg else v
                    if lit_value:
                        satisfied = True
                        break
            if satisfied:
                continue
            if not unassigned:
                return False  # conflict
            if len(unassigned) == 1:
                var, neg = unassigned[0]
                assignment[var] = (not neg)
                stats["unit_propagations"] += 1
                changed = True
    # check completion
    pivot = -1
    for i, v in enumerate(assignment):
        if v is None:
            pivot = i
            break
    if pivot == -1:
        return all(clause_satisfied(c, assignment) for c in formula)
    stats["decisions"] += 1
    for trial in (True, False):
        snapshot = list(assignment)
        assignment[pivot] = trial
        if _dpll(formula, assignment, stats):
            return True
        for i, v in enumerate(snapshot):
            assignment[i] = v
    return False


def dpll_solve(formula, variables: int) -> DPLLResult:
    start = time.perf_counter()
    assignment: list[bool | None] = [None] * variables
    stats = {"decisions": 0, "unit_propagations": 0}
    solved = _dpll(formula, assignment, stats)
    if solved:
        final_assignment = [bool(v) if v is not None else False for v in assignment]
        final_unsat = _unsat_count(formula, final_assignment)
    else:
        final_unsat = len(formula)
    return DPLLResult(
        solved=solved,
        decisions=stats["decisions"],
        unit_propagations=stats["unit_propagations"],
        wall_time_s=time.perf_counter() - start,
        final_unsatisfied=final_unsat,
    )


# ---------------------------------------------------------------------------
# Native furnace driven through composer.iterate.
# ---------------------------------------------------------------------------
_SAT_RENAME_MAP = {
    "next_spins": "spins",
    "next_velocity": "velocity",
    "samples": "prev_samples",
    "spatial_samples": "prev_spatial_samples",
    "operator_traces": "prev_operator_traces",
    "best_spins": "prev_best_spins",
    "best_unsatisfied": "prev_best_unsatisfied",
    "concentrations": "prev_concentrations",
}
_SAT_PRESERVE = ("fiber_memory",)


def _sat_before_step(ctx, _index):
    prev_samples = ctx.get("prev_samples") or []
    if prev_samples:
        ctx["previous_unsatisfied"] = prev_samples[-1].unsatisfied_clauses
        ctx["previous_integration"] = prev_samples[-1].integration
    else:
        ctx["previous_unsatisfied"] = ctx.get("prev_best_unsatisfied", 0)
        ctx["previous_integration"] = 0.0
    return {}


@dataclass(frozen=True)
class FurnaceCompareResult:
    solved: bool
    steps: int
    wall_time_s: float
    final_unsatisfied: int
    total_hamming_movement: int
    net_resolved: int
    distance_paid_per_resolved: float | None
    shortest_prefix_to_improvement: int | None
    revisit_count: int


def furnace_solve_via_iterate(
    formula,
    variables: int,
    *,
    steps: int,
    seed: int,
) -> FurnaceCompareResult:
    composer = sat_composer.build_solver_composer()
    rng = random.Random(seed)
    ctx = _init_epoch_context(
        formula=formula,
        variables=variables,
        steps=steps,
        rng=rng,
        temperature=0.85,
        learning_rate=0.18,
        inertia=0.5,
        noise=0.05,
        planted_assignment=None,  # honest: solver doesn't see the planted
        adaptive=False,
        memory_decay=0.92,
        memory_drive=0.12,
        policy="baseline",
        spike_threshold=0.35,
        spike_slope=8.0,
    )
    initial_assignment = sm.spins_to_assignment(ctx["spins"])

    start = time.perf_counter()
    iteration = composer.iterate(
        _EPOCH_TARGETS,
        count=steps,
        initial_context=ctx,
        rename_map=_SAT_RENAME_MAP,
        preserve=_SAT_PRESERVE,
        step_key="t",
        before_step=_sat_before_step,
        collect=("next_spins",),
    )
    final_ctx = dict(iteration.context)
    for src, dst in _SAT_RENAME_MAP.items():
        if dst in final_ctx:
            final_ctx[src] = final_ctx[dst]
    final = composer.run(
        ("final_assignment", "solved", "furnace_result"), final_ctx,
    )
    elapsed = time.perf_counter() - start
    result = final["furnace_result"]

    per_step_assignments = [
        sm.spins_to_assignment(step.collected["next_spins"])
        for step in iteration.steps
        if "next_spins" in step.collected
    ]
    unsat_series = [s.unsatisfied_clauses for s in result.samples]
    assignments_for_movement = [initial_assignment, *per_step_assignments]
    hamming_movements = sm.assignment_hamming_movement(assignments_for_movement)
    distance_deltas = sm.distance_delta_per_step([len(formula), *unsat_series])

    return FurnaceCompareResult(
        solved=bool(result.solved),
        steps=len(unsat_series),
        wall_time_s=elapsed,
        final_unsatisfied=unsat_series[-1] if unsat_series else len(formula),
        total_hamming_movement=sum(hamming_movements),
        net_resolved=sum(distance_deltas),
        distance_paid_per_resolved=sm.distance_paid_per_incompatibility_resolved(
            hamming_movements, distance_deltas
        ),
        shortest_prefix_to_improvement=sm.shortest_observed_prefix_to_improvement(
            [len(formula), *unsat_series]
        ),
        revisit_count=sm.unsat_clause_revisit_count(
            [len(formula), *unsat_series]
        ),
    )


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
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
    brute = brute_force_solve(formula, variables)
    dpll = dpll_solve(formula, variables)
    furnace = furnace_solve_via_iterate(
        formula, variables, steps=furnace_steps, seed=furnace_seed,
    )

    print(f"  {'solver':<14} {'solved':>6} {'time_s':>9} "
          f"{'work':>9} {'final_unsat':>11}  notes")
    print(f"  {'-'*14} {'-'*6} {'-'*9} {'-'*9} {'-'*11}  {'-'*30}")
    print(f"  {'brute-force':<14} {str(brute.solved):>6} "
          f"{brute.wall_time_s:>9.4f} "
          f"{brute.assignments_checked:>9} "
          f"{brute.final_unsatisfied:>11}  assignments_checked")
    print(f"  {'dpll':<14} {str(dpll.solved):>6} "
          f"{dpll.wall_time_s:>9.4f} "
          f"{dpll.decisions:>9} "
          f"{dpll.final_unsatisfied:>11}  "
          f"decisions; unit_props={dpll.unit_propagations}")
    print(f"  {'furnace':<14} {str(furnace.solved):>6} "
          f"{furnace.wall_time_s:>9.4f} "
          f"{furnace.steps:>9} "
          f"{furnace.final_unsatisfied:>11}  "
          f"composer.iterate steps")
    print()
    print(f"  furnace geodesic accounting:")
    print(f"    total_hamming_movement           : "
          f"{furnace.total_hamming_movement}")
    print(f"    net_unsat_resolved               : "
          f"{furnace.net_resolved}")
    print(f"    distance_paid_per_resolved       : "
          f"{_fmt_opt(furnace.distance_paid_per_resolved)}")
    print(f"    shortest_prefix_to_improvement   : "
          f"{furnace.shortest_prefix_to_improvement}")
    print(f"    unsat_clause_revisit_count       : "
          f"{furnace.revisit_count}")


def main() -> None:
    print("Native composer/furnace solver vs simple baselines")
    print("(brute-force is honest dependency-free; DPLL is recursive unit-prop)")
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
    print("    deps. A MiniSAT/CaDiCaL row would be the natural next addition.")
    print()
    print("done.")


if __name__ == "__main__":
    main()
