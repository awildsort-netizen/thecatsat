"""Tiny helpers for SAT-as-distance / solver-metabolism experiments.

These are pure functions over already-emitted furnace artifacts
(samples, operator_traces, per-step assignments). They are intentionally
small so they can be tested in isolation and reused without dragging in
solver state.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def spins_to_assignment(spins: Sequence[float]) -> tuple[bool, ...]:
    """Threshold soft spins to a discrete boolean assignment."""
    return tuple(float(s) >= 0.0 for s in spins)


def hamming(a: Sequence[bool], b: Sequence[bool]) -> int:
    """Hamming distance between two equal-length boolean assignments."""
    if len(a) != len(b):
        raise ValueError(f"hamming length mismatch: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b) if bool(x) != bool(y))


def assignment_hamming_movement(
    assignments: Sequence[Sequence[bool]],
) -> list[int]:
    """Per-step Hamming distance between consecutive assignments.

    Returns one fewer entry than ``assignments``; the i-th entry is the
    movement between step i and step i+1.
    """
    return [hamming(assignments[i], assignments[i + 1]) for i in range(len(assignments) - 1)]


def distance_delta_per_step(unsat_series: Sequence[int]) -> list[int]:
    """Change in unsatisfied-clause count step-to-step.

    Positive value = incompatibility decreased (progress along the geodesic).
    """
    return [int(unsat_series[i]) - int(unsat_series[i + 1]) for i in range(len(unsat_series) - 1)]


def unsat_clause_revisit_count(unsat_series: Sequence[int]) -> int:
    """Number of times an unsatisfied-clause count is revisited.

    Counts revisits, not distinct values: if the trajectory hits the value
    7 four times, that contributes 3 revisits (the first visit is free).
    """
    counts = Counter(int(u) for u in unsat_series)
    return sum(c - 1 for c in counts.values() if c > 1)


def operator_gene_entropy(operator_names: Iterable[str]) -> float:
    """Shannon entropy (in bits) of the distribution over operator names.

    Empty input returns 0.0.
    """
    counts = Counter(operator_names)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def motif_reuse_count(operator_names: Sequence[str], motif_size: int = 3) -> int:
    """Count how many length-``motif_size`` sequential motifs repeat.

    A motif is a tuple of consecutive operator names. Returns the number of
    extra occurrences beyond the first for any motif (so a motif seen 3
    times contributes 2). motif_size <= 0 or sequences shorter than the
    motif return 0.
    """
    if motif_size <= 0 or len(operator_names) < motif_size:
        return 0
    motifs = [
        tuple(operator_names[i : i + motif_size])
        for i in range(len(operator_names) - motif_size + 1)
    ]
    counts = Counter(motifs)
    return sum(c - 1 for c in counts.values() if c > 1)


def shortest_observed_prefix_to_improvement(
    unsat_series: Sequence[int],
) -> int | None:
    """Length of the shortest prefix that improves on the initial unsat count.

    Returns the step index t (1-based count of steps taken) at which the
    series first strictly drops below ``unsat_series[0]``. Returns ``None``
    if no improvement is ever observed, or the series is empty.
    """
    if not unsat_series:
        return None
    start = int(unsat_series[0])
    for i in range(1, len(unsat_series)):
        if int(unsat_series[i]) < start:
            return i
    return None


def distance_paid_per_incompatibility_resolved(
    movements: Sequence[int],
    deltas: Sequence[int],
) -> float | None:
    """Average Hamming movement spent per unit of unsat reduction.

    Sums per-step Hamming movement and divides by the *net* reduction in
    unsatisfied clauses. Returns ``None`` if no net reduction occurred
    (the geodesic accountant can't divide by zero or a negative).
    """
    total_movement = sum(int(m) for m in movements)
    net_resolved = sum(int(d) for d in deltas)
    if net_resolved <= 0:
        return None
    return total_movement / net_resolved


def active_operators_at_step(
    traces: Iterable, t: int
) -> list[str]:
    """Names of operators that were ``active`` at step ``t``.

    ``traces`` is an iterable of objects with ``t``, ``operator``, ``active``
    attributes (typically ``sat_furnace.OperatorTrace``).
    """
    return [tr.operator for tr in traces if int(tr.t) == int(t) and bool(tr.active)]


def operator_trace_gene_tokens(
    traces: Iterable, *, end: bool = True
) -> tuple[str, ...]:
    """Convert an ``OperatorTrace`` stream into streamable gene tokens.

    Only ``active`` traces emit a ``L:<operator>`` literal; inactive traces
    are skipped so the token stream reflects the same operators the climate
    actually licensed. Traces are visited in their input order — callers
    typically pass them already sorted by ``t``. A terminating ``E`` is
    appended when ``end=True`` so the result is directly consumable by
    ``streamable_genes.stream``.
    """
    tokens: list[str] = [f"L:{tr.operator}" for tr in traces if bool(tr.active)]
    if end:
        tokens.append("E")
    return tuple(tokens)
