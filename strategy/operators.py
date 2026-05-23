"""Composable SAT-strategy operators.

Each operator consumes a typed :class:`SearchState` (an immutable read of
the current formula, assignment, RNG, and a small bag of derived field
signals) and either:

- publishes :class:`Field` writes (e.g. a focused unsat clause, a plateau
  pressure read, a bubble pressure verdict) onto the state's ``field``
  channel for downstream operators to consume, *and/or*
- returns a :class:`Proposal` naming a variable to flip with a short
  rationale.

A :class:`StrategyComposer` is a list of operators run in order on the
same state; the **first** operator that returns a non-``None`` proposal
wins. Operators that only publish field signals (gates / detectors)
return ``None`` and act as upstream context for ranker operators.

The decomposition mirrors how ``geometry.flattening_probe._choose_raw``
and ``_choose_spectral`` were already organized internally — pick an
unsat clause, sometimes random-walk, otherwise rank by some scalar —
but lifted into composable pieces so the *ordering* and *composition*
becomes a configuration of operators, not a function body.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from geometry.flattening_probe import CoordinateView, _per_variable_strain
from sat_furnace import CNF, Clause, clause_satisfied


# --------------------------------------------------------------------------- #
# Typed state + proposal                                                      #
# --------------------------------------------------------------------------- #


Assignment = list[bool]


@dataclass
class SearchState:
    """The minimal typed read each strategy operator sees.

    ``field`` is a small mutable dict that operators publish into. This
    is the only mutable handle; operators do not modify ``assignment``
    or ``formula``. The composer applies the chosen flip on the
    assignment after a proposal is returned.
    """

    formula: CNF
    assignment: Assignment
    n_vars: int
    step: int
    rng: random.Random
    field: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    """A single-flip recommendation with a short rationale."""

    variable: int
    operator: str
    reason: str


# Operator signature: state → Proposal | None. Side effects are limited to
# writes into ``state.field`` (small, named, JSON-ish values).
StrategyOperator = Callable[[SearchState], "Proposal | None"]


# --------------------------------------------------------------------------- #
# Helpers — shared across operators, no control flow of their own             #
# --------------------------------------------------------------------------- #


def _unsat_clauses(state: SearchState) -> list[Clause]:
    return [c for c in state.formula if not clause_satisfied(c, state.assignment)]


def _best_variable_in_clause(state: SearchState, clause: Clause) -> int:
    """Greedy: flip whichever variable in the clause minimizes unsat count."""
    formula = state.formula
    assignment = state.assignment
    base = sum(0 if clause_satisfied(c, assignment) else 1 for c in formula)
    best_var = clause[0][0]
    best_unsat = base + 1
    for variable, _ in clause:
        assignment[variable] = not assignment[variable]
        unsat = sum(0 if clause_satisfied(c, assignment) else 1 for c in formula)
        assignment[variable] = not assignment[variable]
        if unsat < best_unsat:
            best_unsat = unsat
            best_var = variable
    return best_var


# --------------------------------------------------------------------------- #
# Operators                                                                   #
# --------------------------------------------------------------------------- #


def unsat_clause_focus(state: SearchState) -> Proposal | None:
    """Publish a randomly-chosen unsat clause onto ``state.field``.

    A *selector* operator — it does not propose a flip itself. Down-
    stream rankers read ``state.field['focused_clause']``. Returning
    ``None`` yields to the next operator. If there are no unsat
    clauses, publishes ``None`` (the composer's caller treats that as
    "solved" before re-entering).
    """
    clauses = _unsat_clauses(state)
    state.field["focused_clause"] = (
        state.rng.choice(clauses) if clauses else None
    )
    state.field["unsat_count"] = len(clauses)
    return None


def random_walk_kick(probability: float = 0.10) -> StrategyOperator:
    """Plateau-escape kick: with small probability, pick a random variable
    in the focused unsat clause and propose its flip.

    Closure over ``probability`` keeps the operator self-contained. The
    operator reads ``state.field['focused_clause']`` published by
    :func:`unsat_clause_focus`; if the focus operator hasn't run yet,
    or there is no focused clause, the kick yields.
    """
    def _operator(state: SearchState) -> Proposal | None:
        clause = state.field.get("focused_clause")
        if clause is None:
            return None
        if state.rng.random() >= probability:
            return None
        variable = clause[state.rng.randrange(len(clause))][0]
        return Proposal(
            variable=int(variable),
            operator="random_walk_kick",
            reason=f"plateau_kick(p={probability:.2f})",
        )
    return _operator


def raw_strain_ranker(state: SearchState) -> Proposal | None:
    """Greedy WalkSAT-style flip in the focused unsat clause.

    Reads ``state.field['focused_clause']``. Yields if no clause is
    focused. This is the "raw view" component lifted out of
    ``_choose_raw`` in :mod:`geometry.flattening_probe`.
    """
    clause = state.field.get("focused_clause")
    if clause is None:
        return None
    variable = _best_variable_in_clause(state, clause)
    return Proposal(
        variable=int(variable),
        operator="raw_strain_ranker",
        reason="greedy_min_unsat_in_focused_clause",
    )


def coordinate_ranker(view: CoordinateView) -> StrategyOperator:
    """Project per-variable strain onto ``view`` and propose by loading.

    This is the ``_choose_spectral`` body, lifted: find the dominant
    direction in the projected strain, then pick the variable with the
    largest loading along that direction *that also appears in some
    unsat clause*. Yields if all projected strain is zero, so a
    downstream raw ranker can take over.

    Reads/writes ``state.field['veto_transformed']`` — a bubble gate
    upstream of this operator may set it to ``True`` to disable
    transform-driven proposals on the current step. When vetoed, the
    operator publishes a ``coordinate_vetoed`` marker and yields.
    """
    def _operator(state: SearchState) -> Proposal | None:
        if state.field.get("veto_transformed"):
            state.field["coordinate_vetoed"] = True
            return None
        per_var = _per_variable_strain(state.formula, state.assignment, state.n_vars)
        per_direction = view.basis.T @ per_var
        if per_direction.size == 0 or np.allclose(per_direction, 0.0):
            return None
        dominant = int(np.argmax(np.abs(per_direction)))
        loadings = np.abs(view.basis[:, dominant])
        clauses = _unsat_clauses(state)
        if not clauses:
            return None
        candidate_vars = {var for clause in clauses for var, _ in clause}
        best_var = max(candidate_vars, key=lambda v: loadings[v])
        state.field["coordinate_dominant"] = dominant
        return Proposal(
            variable=int(best_var),
            operator=f"coordinate_ranker({view.name})",
            reason=f"dominant_dir={dominant}",
        )
    return _operator


# --------------------------------------------------------------------------- #
# Plateau detector — gate operator, publishes a field signal                  #
# --------------------------------------------------------------------------- #


def plateau_detector(
    window: int = 6,
    relative_band: float = 0.02,
) -> StrategyOperator:
    """Read recent unsat-count history; classify the trajectory as a plateau.

    Publishes ``state.field['plateau']`` ∈ {``True``, ``False``}. A
    plateau is when the last ``window`` unsat counts vary by less than
    ``relative_band`` × ``max(history)``. The strain history is read
    from ``state.field['unsat_history']``, which the run driver appends
    to before each step. Yields (returns ``None``).
    """
    def _operator(state: SearchState) -> Proposal | None:
        history = state.field.get("unsat_history") or []
        if len(history) < window:
            state.field["plateau"] = False
            return None
        tail = list(history)[-window:]
        ceiling = max(tail) if tail else 0
        amplitude = max(tail) - min(tail)
        threshold = max(1.0, relative_band * max(1.0, ceiling))
        state.field["plateau"] = amplitude <= threshold
        state.field["plateau_amplitude"] = float(amplitude)
        return None
    return _operator


# --------------------------------------------------------------------------- #
# Bubble pressure gate — composable, NOT a whole solver                       #
# --------------------------------------------------------------------------- #


# Pressure-label → veto action. We reuse the *labels* the bubble tuning
# module already publishes (rule-table style: first-match wins) and
# treat ``destructive_amplification`` as the case where a transformed
# coordinate proposal should be vetoed in favor of the raw ranker.
#
# This is a small, declarative mapping — the gate is composable, not a
# monolithic policy. Adding a new pressure label just extends the table.
_BUBBLE_VETO_TABLE: tuple[tuple[str, bool, str], ...] = (
    ("destructive_amplification", True, "transform off-phase; fall back to raw"),
    ("diffuse_pressure", True, "nothing separating; transformed pick is noise"),
    ("diagnostic_amplification", False, "transform localizing; allow"),
    ("strain_amplified", False, "default; allow"),
)


def bubble_pressure_gate() -> StrategyOperator:
    """Gate operator: vetoes transformed-coordinate proposals when bubble
    pressure says the chart is off-phase.

    The driver is expected to supply a recent strain trace + an
    :class:`AddressBubble` candidate. If those are present, the gate
    measures pressure via :func:`geometry.bubble_tuning.measure_pressure`
    and consults a tiny rule table to decide whether to publish
    ``veto_transformed=True``. If they are absent the operator yields
    quietly: bubble gating is *optional* policy, not the loop.
    """
    # Lazy import so this module stays importable in tests that don't
    # exercise the geometry stack.
    from geometry.bubble_lifecycle import AddressBubble
    from geometry.bubble_tuning import measure_pressure

    def _operator(state: SearchState) -> Proposal | None:
        trace = state.field.get("strain_trace")
        bubble = state.field.get("bubble_candidate")
        if not trace or bubble is None or not isinstance(bubble, AddressBubble):
            return None
        pressure = measure_pressure(bubble, trace)
        veto, reason = next(
            (
                (veto, reason)
                for label, veto, reason in _BUBBLE_VETO_TABLE
                if label == pressure.pressure_label
            ),
            (False, "unknown_label"),
        )
        state.field["bubble_pressure_label"] = pressure.pressure_label
        state.field["bubble_pressure_reason"] = reason
        if veto:
            state.field["veto_transformed"] = True
        return None
    return _operator


# --------------------------------------------------------------------------- #
# Riordan bubble fitter — composable, selects a view by stability             #
# --------------------------------------------------------------------------- #


def riordan_bubble_fitter(
    radius: int = 2,
    boundary_width: int = 2,
) -> StrategyOperator:
    """Gate-style operator: pick the most bubble-stable view among candidates.

    Reads ``state.field['strain_trace']`` if present, otherwise a
    single-snapshot strain reconstructed from the current assignment.
    Hands the strain (and optional trace) to
    :func:`geometry.riordan_bubble_fit.fit`. Publishes the decision onto
    the field so the downstream :func:`fitted_coordinate_ranker` can
    consume it; if the fitter vetoes everything, also publishes
    ``veto_transformed=True`` so the existing transform fallbacks fire.

    The fitter is composable, declarative, and outcome-blind: scoring
    uses only bubble pressure / containment metrics — never SAT solve
    state.
    """
    from geometry.riordan_bubble_fit import fit

    def _operator(state: SearchState) -> Proposal | None:
        strain = state.field.get("current_strain")
        if strain is None:
            per_var = _per_variable_strain(
                state.formula, state.assignment, state.n_vars,
            )
            strain = [float(x) for x in per_var]
        trace = state.field.get("strain_trace")
        decision = fit(
            strain,
            trace=trace,
            radius=radius,
            boundary_width=boundary_width,
        )
        state.field["fit_decision"] = decision
        state.field["fitted_view"] = decision.view
        state.field["fitted_selected"] = decision.selected
        state.field["fit_rationale"] = decision.rationale
        if decision.veto:
            state.field["veto_transformed"] = True
        return None

    return _operator


def fitted_coordinate_ranker(state: SearchState) -> Proposal | None:
    """Coordinate ranker that consumes the view chosen by the bubble fitter.

    Reads ``state.field['fitted_view']`` (populated by
    :func:`riordan_bubble_fitter`). If absent, yields. If a veto is in
    effect, yields and marks ``coordinate_vetoed=True``. Otherwise
    delegates to a one-shot :func:`coordinate_ranker` for the chosen
    view.
    """
    if state.field.get("veto_transformed"):
        state.field["coordinate_vetoed"] = True
        return None
    view = state.field.get("fitted_view")
    if view is None or not isinstance(view, CoordinateView):
        return None
    # The chosen view's basis may be ``identity``; running the
    # coordinate ranker on identity is a valid no-op (it falls through
    # to the raw ranker because the per-direction projection equals
    # the per-variable strain and the routing still works).
    return coordinate_ranker(view)(state)


# --------------------------------------------------------------------------- #
# Composer                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyComposer:
    """Run a list of strategy operators on a state; first proposal wins.

    Operators that publish only field signals (gates, detectors) return
    ``None`` and act as upstream context for the rankers that follow.
    Ordering matters: earlier operators see fewer field signals; later
    operators may veto themselves based on upstream writes.

    Composition style mirrors the bubble_tuning ``RULES`` table — a
    declarative list traversed with ``next()`` over the first match.
    """

    operators: tuple[StrategyOperator, ...]

    def step(self, state: SearchState) -> Proposal | None:
        return next(
            (
                proposal
                for proposal in (op(state) for op in self.operators)
                if proposal is not None
            ),
            None,
        )

    def with_prepended(self, *ops: StrategyOperator) -> "StrategyComposer":
        return StrategyComposer(operators=tuple(ops) + self.operators)

    def with_appended(self, *ops: StrategyOperator) -> "StrategyComposer":
        return StrategyComposer(operators=self.operators + tuple(ops))
