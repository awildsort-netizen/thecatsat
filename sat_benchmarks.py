#!/usr/bin/env python3
"""Composer-native SAT benchmark harness.

The three solvers we care to compare — brute-force, DPLL, and the
furnace driven through ``Composer.iterate`` — all consume the same
``(formula, variables)`` input and produce a common ``SolveResult``.
That makes them interchangeable operators in a benchmark composer:
ask the DAG for ``brute_result`` and you get brute; ask for
``furnace_benchmark_result`` and you get the furnace's trajectory
summary. The benchmark/calorimeter layer never needs to know which
solver wrote a row.

The native solver does not return yes/no — it pays *distance* through
incompatibility. So ``SolveResult`` keeps a small required surface
(``solved``, ``final_unsatisfied``, ``wall_time_s``, ``work_units``,
``work_metric``) plus an optional ``metabolism`` mapping that carries
the geodesic accounting (``distance_paid_per_resolved`` and friends)
only when it makes sense. Brute / DPLL leave it empty.

Dependency-free by default. An optional external-solver row
(CaDiCaL/MiniSat/Kissat/Glucose/PicoSAT) can be added by passing
``include_external=True`` to :func:`build_sat_benchmark_composer`; if no
supported binary is on PATH the external operator still returns a
``SolveResult`` (with ``work_metric='unavailable'``) so the harness
stays runnable everywhere.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from composer import Composer, FieldOperator
import sat_composer
import sat_metabolism as sm
from sat_furnace import (
    _EPOCH_RENAME_MAP,
    _EPOCH_TARGETS,
    _init_epoch_context,
    clause_satisfied,
)


@dataclass(frozen=True)
class SolveResult:
    """Common result shape across solvers.

    ``work_metric`` names the solver's notion of work (e.g.
    ``"assignments_checked"``, ``"decisions"``, ``"iterate_steps"``) and
    ``work_units`` is its scalar count. ``metabolism`` carries optional
    geodesic-accounting fields; brute / DPLL leave it empty.
    """

    solver_name: str
    solved: bool
    final_unsatisfied: int
    wall_time_s: float
    work_metric: str
    work_units: int
    assignment: tuple[bool, ...] | None = None
    metabolism: Mapping[str, Any] = field(default_factory=dict)


def _unsat_count(formula, assignment) -> int:
    return sum(0 if clause_satisfied(c, assignment) else 1 for c in formula)


# ---------------------------------------------------------------------------
# Brute-force exhaustive assignment search.
# ---------------------------------------------------------------------------
def brute_force_solve(formula, variables: int) -> SolveResult:
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
            return SolveResult(
                solver_name="brute_force",
                solved=True,
                final_unsatisfied=0,
                wall_time_s=time.perf_counter() - start,
                work_metric="assignments_checked",
                work_units=checked,
                assignment=tuple(assignment),
            )
    return SolveResult(
        solver_name="brute_force",
        solved=False,
        final_unsatisfied=best_unsat,
        wall_time_s=time.perf_counter() - start,
        work_metric="assignments_checked",
        work_units=checked,
        assignment=None,
    )


# ---------------------------------------------------------------------------
# DPLL with unit propagation.
# ---------------------------------------------------------------------------
def _dpll(formula, assignment, stats) -> bool:
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
                return False
            if len(unassigned) == 1:
                var, neg = unassigned[0]
                assignment[var] = (not neg)
                stats["unit_propagations"] += 1
                changed = True
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


def dpll_solve(formula, variables: int) -> SolveResult:
    start = time.perf_counter()
    assignment: list[bool | None] = [None] * variables
    stats = {"decisions": 0, "unit_propagations": 0}
    solved = _dpll(formula, assignment, stats)
    if solved:
        final = [bool(v) if v is not None else False for v in assignment]
        return SolveResult(
            solver_name="dpll",
            solved=True,
            final_unsatisfied=_unsat_count(formula, final),
            wall_time_s=time.perf_counter() - start,
            work_metric="decisions",
            work_units=stats["decisions"],
            assignment=tuple(final),
            metabolism={"unit_propagations": stats["unit_propagations"]},
        )
    return SolveResult(
        solver_name="dpll",
        solved=False,
        final_unsatisfied=len(formula),
        wall_time_s=time.perf_counter() - start,
        work_metric="decisions",
        work_units=stats["decisions"],
        assignment=None,
        metabolism={"unit_propagations": stats["unit_propagations"]},
    )


# ---------------------------------------------------------------------------
# Furnace via Composer.iterate.
# ---------------------------------------------------------------------------


def _sat_before_step(ctx, _index):
    prev_samples = ctx.get("prev_samples") or []
    if prev_samples:
        ctx["previous_unsatisfied"] = prev_samples[-1].unsatisfied_clauses
        ctx["previous_integration"] = prev_samples[-1].integration
    else:
        ctx["previous_unsatisfied"] = ctx.get("prev_best_unsatisfied", 0)
        ctx["previous_integration"] = 0.0
    return {}


def furnace_solve_via_iterate(
    formula,
    variables: int,
    *,
    steps: int = 30,
    seed: int = 0,
) -> SolveResult:
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
        planted_assignment=None,
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
        rename_map=_EPOCH_RENAME_MAP,
        preserve=("fiber_memory",),
        step_key="t",
        before_step=_sat_before_step,
        collect=("next_spins",),
    )
    final_ctx = dict(iteration.context)
    for src, dst in _EPOCH_RENAME_MAP.items():
        if dst in final_ctx:
            final_ctx[src] = final_ctx[dst]
    final = composer.run(
        ("final_assignment", "solved", "furnace_result"), final_ctx,
    )
    elapsed = time.perf_counter() - start
    fr = final["furnace_result"]

    per_step_assignments = [
        sm.spins_to_assignment(step.collected["next_spins"])
        for step in iteration.steps
        if "next_spins" in step.collected
    ]
    unsat_series = [s.unsatisfied_clauses for s in fr.samples]
    assignments_for_movement = [initial_assignment, *per_step_assignments]
    hamming_movements = sm.assignment_hamming_movement(assignments_for_movement)
    distance_deltas = sm.distance_delta_per_step([len(formula), *unsat_series])

    metabolism = {
        "total_hamming_movement": sum(hamming_movements),
        "net_unsat_resolved": sum(distance_deltas),
        "distance_paid_per_resolved": sm.distance_paid_per_incompatibility_resolved(
            hamming_movements, distance_deltas
        ),
        "shortest_prefix_to_improvement": sm.shortest_observed_prefix_to_improvement(
            [len(formula), *unsat_series]
        ),
        "unsat_clause_revisit_count": sm.unsat_clause_revisit_count(
            [len(formula), *unsat_series]
        ),
    }
    final_assignment = tuple(bool(v) for v in fr.final_assignment)
    return SolveResult(
        solver_name="furnace",
        solved=bool(fr.solved),
        final_unsatisfied=unsat_series[-1] if unsat_series else len(formula),
        wall_time_s=elapsed,
        work_metric="iterate_steps",
        work_units=len(unsat_series),
        assignment=final_assignment if fr.solved else None,
        metabolism=metabolism,
    )


# ---------------------------------------------------------------------------
# Benchmark composer.
# ---------------------------------------------------------------------------
def _brute_op() -> FieldOperator:
    def _run(ctx):
        return {"brute_result": brute_force_solve(ctx["formula"], ctx["variables"])}

    return FieldOperator(
        name="brute_force_solve",
        inputs=("formula", "variables"),
        outputs=("brute_result",),
        run=_run,
    )


def _dpll_op() -> FieldOperator:
    def _run(ctx):
        return {"dpll_result": dpll_solve(ctx["formula"], ctx["variables"])}

    return FieldOperator(
        name="dpll_solve",
        inputs=("formula", "variables"),
        outputs=("dpll_result",),
        run=_run,
    )


def _furnace_op() -> FieldOperator:
    def _run(ctx):
        return {
            "furnace_benchmark_result": furnace_solve_via_iterate(
                ctx["formula"],
                ctx["variables"],
                steps=int(ctx.get("furnace_steps", 30)),
                seed=int(ctx.get("furnace_seed", 0)),
            )
        }

    return FieldOperator(
        name="furnace_solve_via_iterate",
        inputs=("formula", "variables"),
        outputs=("furnace_benchmark_result",),
        run=_run,
    )


def build_sat_benchmark_composer(*, include_external: bool = False,
                                 external_timeout_s: float = 30.0) -> Composer:
    """Composer that exposes brute / DPLL / furnace as shared operators.

    All operators read ``(formula, variables)`` from the context;
    the furnace also reads optional ``furnace_steps`` and ``furnace_seed``.
    Targets: ``brute_result``, ``dpll_result``, ``furnace_benchmark_result``.

    When ``include_external=True`` an additional ``external_solve``
    operator is registered, producing ``external_result``. The operator
    auto-discovers a supported solver binary on PATH; if none is found
    it returns a ``SolveResult`` with ``work_metric='unavailable'``, so
    enabling the row never makes the harness fail on a fresh machine.
    """
    ops = [_brute_op(), _dpll_op(), _furnace_op()]
    if include_external:
        # Imported lazily so sat_benchmarks stays usable without the
        # external_sat module being touched at import time.
        from external_sat import external_solver_op
        ops.append(external_solver_op(timeout_s=external_timeout_s))
    return Composer(ops)
