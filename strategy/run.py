"""Run a composed strategy as a local-search loop.

Thin driver around a :class:`strategy.operators.StrategyComposer`. The
driver itself does *not* encode strategy — it just turns proposals into
flips, maintains the per-step unsat history (so plateau detectors can
read it), and exposes a small typed :class:`RunReport`. The interesting
behavior lives entirely in the operator list.

This module reuses pieces from :mod:`geometry.flattening_probe` for
strain reads and shares ``ProbeRunResult``-shaped output via
:class:`RunReport` so the existing report tooling stays interchangeable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from geometry.flattening_probe import _unsat_count
from sat_furnace import CNF
from strategy.operators import (
    SearchState,
    StrategyComposer,
)


@dataclass(frozen=True)
class RunRecord:
    """Per-step record: which operator proposed which flip, and the cost."""

    step: int
    operator: str
    reason: str
    flipped_variable: int
    unsatisfied_before: int
    unsatisfied_after: int


@dataclass(frozen=True)
class RunReport:
    """Aggregate result of one composed-strategy run."""

    composer_name: str
    solved: bool
    flips: int
    initial_unsatisfied: int
    final_unsatisfied: int
    unsat_trajectory: tuple[int, ...]
    records: tuple[RunRecord, ...]
    field_marks: tuple[dict, ...] = field(default_factory=tuple)


def composed_local_search(
    formula: CNF,
    n_vars: int,
    composer: StrategyComposer,
    *,
    composer_name: str = "composed",
    max_flips: int = 200,
    seed: int = 0,
    field_seed: dict | None = None,
) -> RunReport:
    """Run the composed strategy for up to ``max_flips`` steps.

    Determinism: a single ``random.Random(seed)`` drives both the
    starting assignment and the per-step operator decisions, so two
    invocations with the same ``(formula, n_vars, composer, seed)``
    produce byte-identical trajectories.

    ``field_seed`` is merged into the initial ``state.field``; the
    driver itself only writes ``unsat_history`` (an append-only list)
    and clears its own scratch keys between steps.
    """
    rng = random.Random(seed)
    assignment = [rng.choice([False, True]) for _ in range(n_vars)]
    initial_unsat = _unsat_count(formula, assignment)

    trajectory: list[int] = [initial_unsat]
    records: list[RunRecord] = []
    marks: list[dict] = []

    state = SearchState(
        formula=formula,
        assignment=assignment,
        n_vars=n_vars,
        step=0,
        rng=rng,
        field=dict(field_seed or {}),
    )
    state.field.setdefault("unsat_history", [initial_unsat])

    for step in range(max_flips):
        before = _unsat_count(formula, assignment)
        if before == 0:
            return RunReport(
                composer_name=composer_name,
                solved=True,
                flips=step,
                initial_unsatisfied=initial_unsat,
                final_unsatisfied=0,
                unsat_trajectory=tuple(trajectory),
                records=tuple(records),
                field_marks=tuple(marks),
            )

        # The driver refreshes per-step scratch state but preserves
        # accumulating channels like ``unsat_history``.
        state.step = step
        for key in ("focused_clause", "veto_transformed", "coordinate_vetoed",
                    "coordinate_dominant", "bubble_pressure_label",
                    "bubble_pressure_reason"):
            state.field.pop(key, None)

        proposal = composer.step(state)
        if proposal is None:
            # No operator wanted to act — break, mirroring how WalkSAT
            # would just stop progressing.
            break

        assignment[proposal.variable] = not assignment[proposal.variable]
        after = _unsat_count(formula, assignment)
        records.append(
            RunRecord(
                step=step,
                operator=proposal.operator,
                reason=proposal.reason,
                flipped_variable=proposal.variable,
                unsatisfied_before=before,
                unsatisfied_after=after,
            )
        )
        trajectory.append(after)
        # Snapshot a small set of field signals for downstream auditing.
        marks.append(
            {
                "step": step,
                "plateau": bool(state.field.get("plateau", False)),
                "veto_transformed": bool(state.field.get("veto_transformed", False)),
                "bubble_pressure_label": state.field.get("bubble_pressure_label"),
                "coordinate_vetoed": bool(state.field.get("coordinate_vetoed", False)),
            }
        )
        history = state.field.get("unsat_history") or []
        if not isinstance(history, list):
            history = list(history)
        history.append(after)
        state.field["unsat_history"] = history

    final_unsat = _unsat_count(formula, assignment)
    return RunReport(
        composer_name=composer_name,
        solved=(final_unsat == 0),
        flips=len(records),
        initial_unsatisfied=initial_unsat,
        final_unsatisfied=final_unsat,
        unsat_trajectory=tuple(trajectory),
        records=tuple(records),
        field_marks=tuple(marks),
    )
