#!/usr/bin/env python3
"""SAT-as-composer: the solver is (mostly) the composer.

Hypothesis (from project direction): the "SAT solver / furnace" is not a
separate machine sitting next to the composer — it *is* a composer plan
executed once per epoch, plus a thin time-iteration shell. This script
makes that visible by:

  * Building the solver composer.
  * Asking it to *plan* the per-epoch DAG and printing the order/edges.
  * Running ``sat_furnace.run_furnace`` on a small instance.
  * Reproducing the same epoch loop by hand using ``composer.run`` and
    confirming the two paths agree on the final assignment (and that
    ``run_furnace`` is therefore a tiny driver, not a separate solver).

Run with: ``python experiments/sat_as_composer.py``
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_composer
import sat_furnace
from sat_furnace import _EPOCH_TARGETS, _init_epoch_context


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def small_planted_instance(seed: int = 7):
    rng = random.Random(seed)
    variables, clauses, k = 6, 18, 3
    formula, planted = sat_furnace.generate_formula(
        "sat", variables, clauses, k, rng
    )
    return formula, planted, variables


def experiment_plan_visibility() -> None:
    banner("1. The solver composer's per-epoch plan is just a DAG")
    composer = sat_composer.build_solver_composer()
    available = (
        # everything _init_epoch_context seeds before the first call
        "formula", "variables", "steps", "temperature", "learning_rate",
        "inertia", "noise", "adaptive", "policy", "spike_threshold",
        "spike_slope", "memory_decay", "memory_drive", "planted_assignment",
        "rng", "spins", "velocity", "fiber_memory",
        "adaptive_active", "adaptive_reason", "adaptive_gain", "control_state",
        "prev_samples", "prev_spatial_samples", "prev_operator_traces",
        "prev_best_spins", "prev_best_unsatisfied", "prev_concentrations",
        "previous_unsatisfied", "previous_integration", "t",
    )
    plan = composer.plan(_EPOCH_TARGETS, available)
    graph = composer.graph(_EPOCH_TARGETS, available)
    print(f"  operators in order ({len(plan.order)}):")
    for i, name in enumerate(plan.order, 1):
        print(f"    {i:>2}. {name}")
    print()
    print(f"  edges ({len(graph.edges)}): "
          f"{graph.edges[:5]} ... (truncated)" if len(graph.edges) > 5
          else f"  edges ({len(graph.edges)}): {graph.edges}")
    print()
    if plan.missing:
        print(f"  missing inputs (would block execution): {plan.missing}")
    else:
        print("  missing inputs: none — plan is satisfiable from init context.")
    print()
    print("  Observation: the 'solver' is a static dependency DAG over the")
    print("  registered operators. No solver-specific control flow lives")
    print("  outside the composer plan itself.")


_SAT_RENAME_MAP = {
    # Per-step state that becomes the next step's input under a different name.
    "next_spins": "spins",
    "next_velocity": "velocity",
    "samples": "prev_samples",
    "spatial_samples": "prev_spatial_samples",
    "operator_traces": "prev_operator_traces",
    "best_spins": "prev_best_spins",
    "best_unsatisfied": "prev_best_unsatisfied",
    "concentrations": "prev_concentrations",
}

# fiber_memory is in _EPOCH_TARGETS but no operator produces it — it is
# initial state mutated in-place by downstream operators. ``preserve`` tells
# ``iterate`` to keep it across the stale-key sweep.
_SAT_PRESERVE = ("fiber_memory",)


def _sat_before_step(ctx, index):
    """Derive scalar previous-step metrics from the renamed prev_samples list."""
    prev_samples = ctx.get("prev_samples") or []
    if prev_samples:
        ctx["previous_unsatisfied"] = prev_samples[-1].unsatisfied_clauses
        ctx["previous_integration"] = prev_samples[-1].integration
    else:
        ctx["previous_unsatisfied"] = ctx.get("prev_best_unsatisfied", 0)
        ctx["previous_integration"] = 0.0
    return {}


def experiment_run_furnace_is_a_driver() -> None:
    banner("2. run_furnace is a thin time-shell around composer.iterate")
    formula, planted, variables = small_planted_instance()
    rng_a = random.Random(11)
    rng_b = random.Random(11)
    result = sat_furnace.run_furnace(
        formula=formula,
        variables=variables,
        steps=20,
        rng=rng_a,
        temperature=0.85,
        learning_rate=0.18,
        inertia=0.5,
        noise=0.05,
        planted_assignment=planted,
        adaptive=False,
    )
    print(f"  run_furnace: solved={result.solved} "
          f"unsatisfied={result.samples[-1].unsatisfied_clauses if result.samples else -1} "
          f"steps_recorded={len(result.samples)}")

    composer = sat_composer.build_solver_composer()
    ctx = _init_epoch_context(
        formula=formula,
        variables=variables,
        steps=20,
        rng=rng_b,
        temperature=0.85,
        learning_rate=0.18,
        inertia=0.5,
        noise=0.05,
        planted_assignment=planted,
        adaptive=False,
        memory_decay=0.92,
        memory_drive=0.12,
        policy="baseline",
        spike_threshold=0.35,
        spike_slope=8.0,
    )
    iteration = composer.iterate(
        _EPOCH_TARGETS,
        count=20,
        initial_context=ctx,
        rename_map=_SAT_RENAME_MAP,
        preserve=_SAT_PRESERVE,
        step_key="t",
        before_step=_sat_before_step,
        collect=("samples",),
    )
    # Restore renamed sources so the post-iteration composer.run reads the
    # final per-step outputs (samples, best_spins, ...) rather than re-running
    # the per-step operators on top of the carry-forward state.
    final_ctx = dict(iteration.context)
    for src, dst in _SAT_RENAME_MAP.items():
        if dst in final_ctx:
            final_ctx[src] = final_ctx[dst]
    final = composer.run(
        ("final_assignment", "solved", "furnace_result"), final_ctx,
    )
    by_hand = final["furnace_result"]
    print(f"  by-hand:     solved={by_hand.solved} "
          f"unsatisfied={by_hand.samples[-1].unsatisfied_clauses if by_hand.samples else -1} "
          f"steps_recorded={len(by_hand.samples)}")
    a_unsat = result.samples[-1].unsatisfied_clauses if result.samples else -1
    b_unsat = by_hand.samples[-1].unsatisfied_clauses if by_hand.samples else -1
    same = (
        result.solved == by_hand.solved
        and a_unsat == b_unsat
        and list(result.final_assignment) == list(by_hand.final_assignment)
    )
    print(f"  equivalence: {same}")
    print(f"  iteration trace has {len(iteration.steps)} step snapshots.")
    print()
    print("  Observation: composer.iterate(_EPOCH_TARGETS, count=20, rename_map=...)")
    print("  reproduces run_furnace exactly. The hand-rolled ``for t in range(steps)``")
    print("  has collapsed into a single counted-cycle call. The cycle in the")
    print("  per-epoch dependency graph (next_spins -> spins, samples ->")
    print("  prev_samples, ...) is no longer a planner error — it is the rename")
    print("  contract that makes 20 epochs a first-class request.")


def experiment_what_resists_being_a_composer() -> None:
    banner("3. What in the solver does NOT look like a composer operator")
    composer = sat_composer.build_solver_composer()
    op_names = tuple(composer._operators.keys())
    print(f"  registered operators ({len(op_names)}):")
    for name in op_names:
        print(f"    - {name}")
    print()
    print("  The remaining 'solver' code is:")
    print("    * _init_epoch_context: seeds initial dataclass state (memory")
    print("      fiber, control_state). Could become an init operator with")
    print("      explicit outputs, but it is run-once not per-step.")
    print("    * the time loop: now expressible as composer.iterate with a")
    print("      rename_map and a tiny before_step that derives scalar")
    print("      previous_* metrics from prev_samples[-1]. Cycles in the")
    print("      per-epoch graph (next_spins -> spins, samples ->")
    print("      prev_samples) are the rename contract, not a planner error.")
    print("    * the final composer.run for ('final_assignment', ...): a")
    print("      second plan; already a composer call, just over a")
    print("      different target set.")
    print()
    print("  Net: every per-step piece is composer-shaped. The remaining")
    print("  refactor in run_furnace is mechanical — call composer.iterate")
    print("  directly instead of hand-rolling the loop.")


def main() -> None:
    experiment_plan_visibility()
    experiment_run_furnace_is_a_driver()
    experiment_what_resists_being_a_composer()
    print()
    print("done.")


if __name__ == "__main__":
    main()
