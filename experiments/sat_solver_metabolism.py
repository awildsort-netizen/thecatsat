#!/usr/bin/env python3
"""SAT solver metabolism trace.

Reframe the SAT solver as a *distance calculation*: it does not just
return ``find assignment``, it reduces an incompatibility distance
(unsatisfied-clause count) until the formula admits a stable embedding.
The question this experiment asks is not "did it solve?" but "did this
gene/climate produce a shorter geodesic through incompatibility, and at
what cost?".

We run the existing furnace via ``composer.iterate`` so we can collect
per-step ``next_spins``, then thread the resulting per-step assignments
through the tiny helpers in ``sat_metabolism.py`` to emit:

  - distance_delta_per_step
  - assignment_hamming_movement
  - unsat_clause_revisit_count
  - operator_gene_entropy (over ``L:<op>`` streamable gene tokens)
  - motif_reuse_count (over ``L:<op>`` streamable gene tokens)
  - shortest_observed_prefix_to_improvement
  - distance_paid_per_incompatibility_resolved

The "gene" record is built by adapting climate-active ``OperatorTrace``
entries into streamable gene tokens via
``sat_metabolism.operator_trace_gene_tokens`` and decoding them through
``streamable_genes.stream``. Entropy and motif reuse then run over the
real ``L:<op>`` token stream, not over bare operator names.

Two climates are compared at the same seed/instance: ``baseline`` and
``excitable_fiber`` (the second is an already-implemented composer
policy with a different gene mix — no concentration knob is forced into
SAT here).

Run with: ``python experiments/sat_solver_metabolism.py``
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_composer
import sat_furnace
import sat_metabolism as sm
import streamable_genes
from sat_furnace import _EPOCH_TARGETS, _init_epoch_context


# Same rename contract used by experiments/sat_as_composer.py — keeps the
# per-epoch DAG cyclic-by-rename instead of cyclic-by-error.
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


def _sat_before_step(ctx, index):
    prev_samples = ctx.get("prev_samples") or []
    if prev_samples:
        ctx["previous_unsatisfied"] = prev_samples[-1].unsatisfied_clauses
        ctx["previous_integration"] = prev_samples[-1].integration
    else:
        ctx["previous_unsatisfied"] = ctx.get("prev_best_unsatisfied", 0)
        ctx["previous_integration"] = 0.0
    return {}


@dataclass(frozen=True)
class MetabolismTrace:
    climate: str
    solved: bool
    initial_assignment: tuple[bool, ...]
    per_step_assignments: list[tuple[bool, ...]]
    unsat_series: list[int]
    integration_series: list[float]
    hamming_movements: list[int]
    distance_deltas: list[int]
    active_operators_per_step: list[list[str]]
    gene_tokens: tuple[str, ...]
    gene_entropy: float
    motif_reuse: int
    revisit_count: int
    shortest_prefix_to_improvement: int | None
    distance_paid: float | None


def small_planted_instance(seed: int = 7):
    rng = random.Random(seed)
    variables, clauses, k = 8, 24, 3
    formula, planted = sat_furnace.generate_formula(
        "sat", variables, clauses, k, rng
    )
    return formula, planted, variables


def run_metabolism(
    *,
    climate: str,
    formula,
    variables: int,
    planted,
    steps: int,
    seed: int,
    policy: str,
) -> MetabolismTrace:
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
        planted_assignment=planted,
        adaptive=False,
        memory_decay=0.92,
        memory_drive=0.12,
        policy=policy,
        spike_threshold=0.35,
        spike_slope=8.0,
    )
    initial_assignment = sm.spins_to_assignment(ctx["spins"])

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
    result = final["furnace_result"]

    per_step_assignments = [
        sm.spins_to_assignment(step.collected["next_spins"])
        for step in iteration.steps
        if "next_spins" in step.collected
    ]
    unsat_series = [s.unsatisfied_clauses for s in result.samples]
    integration_series = [s.integration for s in result.samples]

    assignments_for_movement = [initial_assignment, *per_step_assignments]
    hamming_movements = sm.assignment_hamming_movement(assignments_for_movement)
    distance_deltas = sm.distance_delta_per_step(
        [len(formula), *unsat_series]
    )

    active_ops_per_step = [
        sm.active_operators_at_step(result.operator_traces, t)
        for t in range(steps)
    ]
    # Build a real streamable gene-token sequence from the active
    # OperatorTrace entries (climate already filtered inactive operators
    # in _trace_append_operator). The token stream is the metabolism's
    # gene record; metrics below run over the L: tokens, not over the
    # naked operator names.
    sorted_traces = sorted(
        result.operator_traces, key=lambda tr: (int(tr.t), tr.operator)
    )
    gene_tokens = sm.operator_trace_gene_tokens(sorted_traces)
    # Round-trip through the streamable_genes decoder so the experiment
    # consumes the same vocabulary downstream consumers do.
    gene_state = streamable_genes.stream(gene_tokens)
    decoded_names = tuple(token.name for token in gene_state.emitted)
    # Compute entropy/motif reuse over the L:<op> token forms so the
    # vocabulary is the gene-stream vocabulary, not bare operator names.
    literal_tokens = tuple(f"L:{name}" for name in decoded_names)

    return MetabolismTrace(
        climate=climate,
        solved=bool(result.solved),
        initial_assignment=initial_assignment,
        per_step_assignments=per_step_assignments,
        unsat_series=unsat_series,
        integration_series=integration_series,
        hamming_movements=hamming_movements,
        distance_deltas=distance_deltas,
        active_operators_per_step=active_ops_per_step,
        gene_tokens=gene_tokens,
        gene_entropy=sm.operator_gene_entropy(literal_tokens),
        motif_reuse=sm.motif_reuse_count(literal_tokens, motif_size=3),
        revisit_count=sm.unsat_clause_revisit_count(
            [len(formula), *unsat_series]
        ),
        shortest_prefix_to_improvement=sm.shortest_observed_prefix_to_improvement(
            [len(formula), *unsat_series]
        ),
        distance_paid=sm.distance_paid_per_incompatibility_resolved(
            hamming_movements, distance_deltas
        ),
    )


def _fmt_opt(value, fmt: str = "{:.3f}") -> str:
    if value is None:
        return "n/a"
    return fmt.format(value)


def print_step_table(trace: MetabolismTrace, total_clauses: int) -> None:
    print()
    print(f"  step | unsat | d_unsat | hamming | active operators (L: gene tokens)")
    print(f"  -----+-------+---------+---------+-----------------------------------")
    print(f"   init|{total_clauses:>6} | {'':>7} | {'':>7} | (initial assignment)")
    for i in range(len(trace.unsat_series)):
        ops = trace.active_operators_per_step[i]
        ops_str = ", ".join(ops) if ops else "—"
        if len(ops_str) > 50:
            ops_str = ops_str[:47] + "..."
        print(
            f"  {i:>4} |"
            f" {trace.unsat_series[i]:>5} |"
            f" {trace.distance_deltas[i]:>+7} |"
            f" {trace.hamming_movements[i]:>7} |"
            f" {ops_str}"
        )


def print_summary(trace: MetabolismTrace) -> None:
    print()
    print(f"  climate: {trace.climate!r}")
    print(f"    solved: {trace.solved}")
    print(f"    final unsat: {trace.unsat_series[-1] if trace.unsat_series else 'n/a'}")
    literal_count = sum(1 for tok in trace.gene_tokens if tok.startswith("L:"))
    print(f"    gene_tokens (streamable): {len(trace.gene_tokens)} "
          f"({literal_count} L: literals + terminator)")
    print(f"    gene_entropy (bits over L: gene-token stream): "
          f"{trace.gene_entropy:.3f}")
    print(f"    motif_reuse_count (len-3 L: motifs): "
          f"{trace.motif_reuse}")
    print(f"    unsat_clause_revisit_count: {trace.revisit_count}")
    print(f"    shortest_observed_prefix_to_improvement: "
          f"{trace.shortest_prefix_to_improvement}")
    print(f"    distance_paid_per_incompatibility_resolved: "
          f"{_fmt_opt(trace.distance_paid)}")
    print(f"    total_hamming_movement: {sum(trace.hamming_movements)}")
    print(f"    net_unsat_resolved: {sum(trace.distance_deltas)}")


def compare_traces(traces: list[MetabolismTrace]) -> None:
    print()
    print("=" * 72)
    print("CLIMATE COMPARISON")
    print("=" * 72)
    print(f"  {'metric':<46} | " + " | ".join(f"{t.climate:>14}" for t in traces))
    print(f"  {'-' * 46}-+-" + "-+-".join("-" * 14 for _ in traces))

    def row(label, values, fmt="{}"):
        cells = " | ".join(f"{fmt.format(v) if v is not None else 'n/a':>14}" for v in values)
        print(f"  {label:<46} | {cells}")

    row("solved", [t.solved for t in traces])
    row("final unsat", [t.unsat_series[-1] for t in traces])
    row("gene_entropy", [t.gene_entropy for t in traces], "{:.3f}")
    row("motif_reuse_count (3-gram)", [t.motif_reuse for t in traces])
    row("unsat_clause_revisit_count", [t.revisit_count for t in traces])
    row("shortest_prefix_to_improvement",
        [t.shortest_prefix_to_improvement for t in traces])
    row("distance_paid_per_resolved",
        [t.distance_paid for t in traces], "{:.3f}")
    row("total_hamming_movement",
        [sum(t.hamming_movements) for t in traces])
    row("net_unsat_resolved",
        [sum(t.distance_deltas) for t in traces])


def main() -> None:
    formula, planted, variables = small_planted_instance(seed=7)
    total_clauses = len(formula)
    steps = 25
    seed = 11
    print(f"Instance: variables={variables} clauses={total_clauses} "
          f"k=3 planted-SAT seed=7")
    print(f"Steps: {steps}  rng seed: {seed}")

    climates = [
        ("baseline", "baseline"),
        ("excitable_fiber", sat_composer.EXCITABLE_POLICY),
    ]
    traces: list[MetabolismTrace] = []
    for climate_name, policy in climates:
        print()
        print("=" * 72)
        print(f"CLIMATE: {climate_name}  (policy={policy!r})")
        print("=" * 72)
        trace = run_metabolism(
            climate=climate_name,
            formula=formula,
            variables=variables,
            planted=planted,
            steps=steps,
            seed=seed,
            policy=policy,
        )
        print_step_table(trace, total_clauses)
        print_summary(trace)
        traces.append(trace)

    compare_traces(traces)

    print()
    print("Notes")
    print("  * 'gene' here is a real streamable gene-token stream: each")
    print("    climate-active OperatorTrace is adapted into an L:<op>")
    print("    literal and round-tripped through streamable_genes.stream.")
    print("    Entropy and motif reuse run over the resulting L: tokens.")
    print("  * distance_paid_per_incompatibility_resolved = total Hamming")
    print("    movement / net unsat resolved. Lower = the climate paid less")
    print("    assignment-space distance per unit of incompatibility")
    print("    discharged. This is the geodesic-cost reading.")
    print("  * unsat_clause_revisit_count > 0 = the trajectory loops over")
    print("    the same incompatibility height — a sign the geodesic is")
    print("    being retraced, not new ground.")
    print()
    print("done.")


if __name__ == "__main__":
    main()
