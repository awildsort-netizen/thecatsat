#!/usr/bin/env python3
"""Bubble tuning: variance-based pressure gauge stacked on the lifecycle scaffold.

PR #11 (`geometry/bubble_lifecycle.py`) gave us seeds, inflation, cheap
containment, and lifecycle labels. It still left two questions open:

1. *Is the strain rising because the bubble is doing diagnostic work —
   amplifying a real distinction into something the boundary can hold
   — or because the chart is off-phase and the amplification is
   destructive?*
2. *Is the bubble we already have the right grain, or does its
   interior want to be split into sub-bubbles?*

This module is the smallest layer that distinguishes those regimes. It
reads four cheap numbers off a bubble + a short strain trace:

- ``interior_mean`` — average strain on the bubble's interior,
- ``interior_std`` — strain variance on the interior (sub-bubble cue),
- ``off_bubble_strain`` — share of total strain outside interior ∪ boundary,
- ``boundary_stability`` — how much the boundary set turns over.

A short declarative rule table maps the gauge to a ``TuningVerdict`` —
``stabilize``, ``split``, ``merge``, ``prune``, or ``hold``. A separate
``PhaseReadout`` reads a *family* of transform observations (e.g. the
spectral / Pascal / signed / Sierpinski views ``riordan_probe`` already
runs) and classifies the family as ``aligned``, ``off_phase``,
``over_smoothed``, or ``needs_another_layer``.

The deliberate style choices:

- **No big ``if`` ladder for verdicts.** Verdicts live in a
  ``RULES`` table of ``TuningLaw`` objects, each with a predicate over a
  ``BubblePressure`` snapshot and a short rationale. The first law
  whose predicate matches wins; a default law catches the rest.
- **No manual ``for`` loops for reductions.** Means, stds, and shares
  are computed via comprehensions and ``statistics.mean`` /
  ``statistics.pstdev``.
- **Small, pure functions.** Each piece (gauge, verdict, hierarchy,
  phase) is a function from inputs to a dataclass; the rule table is
  the only place control flow lives.

The module deliberately does *not* alter the solver loop. It is a
read-only diagnostic stacked on PR #11 — exactly the pacing the
representational-bubbles doc set.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

from geometry.bubble_lifecycle import AddressBubble


# --------------------------------------------------------------------------- #
# Verdict + phase labels                                                      #
# --------------------------------------------------------------------------- #


STABILIZE = "stabilize"
SPLIT = "split"
MERGE = "merge"
PRUNE = "prune"
HOLD = "hold"

VERDICT_LABELS = (STABILIZE, SPLIT, MERGE, PRUNE, HOLD)


# Sublabels for "strain is rising" — factored away from the
# old judgmental `amplified_pathology` framing.
DIAGNOSTIC_AMPLIFICATION = "diagnostic_amplification"
DESTRUCTIVE_AMPLIFICATION = "destructive_amplification"
DIFFUSE_PRESSURE = "diffuse_pressure"
STRAIN_AMPLIFIED = "strain_amplified"

PRESSURE_LABELS = (
    DIAGNOSTIC_AMPLIFICATION,
    DESTRUCTIVE_AMPLIFICATION,
    DIFFUSE_PRESSURE,
    STRAIN_AMPLIFIED,
)


ALIGNED = "aligned"
OFF_PHASE = "off_phase"
OVER_SMOOTHED = "over_smoothed"
NEEDS_ANOTHER_LAYER = "needs_another_layer"

PHASE_LABELS = (ALIGNED, OFF_PHASE, OVER_SMOOTHED, NEEDS_ANOTHER_LAYER)


# --------------------------------------------------------------------------- #
# Pressure gauge                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BubblePressure:
    """Cheap variance-based pressure read on a bubble + trace.

    All fields are floats in well-defined ranges and computed by simple
    reductions over a strain trace (means, stds, shares). The pressure
    label is one of :data:`PRESSURE_LABELS`; it is a factored sublabel
    of the lifecycle label, not a replacement for it.
    """

    interior_mean: float
    interior_std: float
    off_bubble_strain: float
    boundary_stability: float
    total_mean: float
    total_std: float
    snapshots: int
    pressure_label: str


def _share_outside(snapshot: Sequence[float], inside_ids: frozenset[int]) -> float:
    """Share of strain in ``snapshot`` not held by indices in ``inside_ids``."""
    total = sum(snapshot)
    return 0.0 if total <= 0.0 else (
        sum(v for i, v in enumerate(snapshot) if i not in inside_ids) / total
    )


def _safe_stdev(values: Sequence[float]) -> float:
    """Population std that gracefully returns 0.0 for empty / single-element."""
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


# A pressure-label predicate runs against the four headline numbers and
# the lifecycle/static read. Order matters; first match wins.
_PressurePredicate = Callable[["BubblePressure"], bool]


def _is_diffuse_pressure(p: "BubblePressure") -> bool:
    # Low std, non-trivial mean — nothing is separating.
    return p.total_mean > 0.0 and p.total_std <= 0.20 * max(p.total_mean, 1e-9)


def _is_diagnostic(p: "BubblePressure") -> bool:
    # Std is up *and* the boundary is stable *and* off-bubble strain is low.
    return (
        p.total_std > 0.20 * max(p.total_mean, 1e-9)
        and p.boundary_stability >= 0.5
        and p.off_bubble_strain <= 0.40
    )


def _is_destructive(p: "BubblePressure") -> bool:
    # Std is up but the boundary churns OR strain has leaked out.
    return (
        p.total_std > 0.20 * max(p.total_mean, 1e-9)
        and (p.boundary_stability < 0.5 or p.off_bubble_strain > 0.40)
    )


_PRESSURE_RULES: tuple[tuple[str, _PressurePredicate], ...] = (
    (DIFFUSE_PRESSURE, _is_diffuse_pressure),
    (DIAGNOSTIC_AMPLIFICATION, _is_diagnostic),
    (DESTRUCTIVE_AMPLIFICATION, _is_destructive),
)


def _classify_pressure(p: "BubblePressure") -> str:
    """First-match pressure label; falls back to the bland ``strain_amplified``."""
    return next(
        (label for label, predicate in _PRESSURE_RULES if predicate(p)),
        STRAIN_AMPLIFIED,
    )


def measure_pressure(
    bubble: AddressBubble,
    trace: Sequence[Sequence[float]],
) -> BubblePressure:
    """Compute :class:`BubblePressure` from a bubble + per-variable strain trace.

    Pure function. No mutation. No manual indexing loops. The trace is
    expected to be the same per-variable strain history the lifecycle
    classifier already consumes.
    """
    interior_ids = frozenset(bubble.interior)
    boundary_ids = frozenset(bubble.boundary)
    in_or_boundary = interior_ids | boundary_ids

    # Interior mean / std are averaged across snapshots: at each snapshot
    # we compute mean and std on the interior indices, then average.
    snapshot_lists = [list(s) for s in trace]
    interior_means = [
        _safe_mean([s[i] for i in interior_ids if i < len(s)])
        for s in snapshot_lists
    ]
    interior_stds = [
        _safe_stdev([s[i] for i in interior_ids if i < len(s)])
        for s in snapshot_lists
    ]
    total_means = [_safe_mean(s) for s in snapshot_lists]
    total_stds = [_safe_stdev(s) for s in snapshot_lists]

    off_shares = [_share_outside(s, in_or_boundary) for s in snapshot_lists]

    # Boundary stability: 1 - turnover. We compare the top-(interior +
    # boundary) set per snapshot to the bubble's recorded interior ∪
    # boundary and report 1 minus the symmetric-difference rate.
    target = interior_ids | boundary_ids
    k = len(target)
    boundary_stabilities = [
        _boundary_stability_at(s, target, k) for s in snapshot_lists
    ]

    pressure = BubblePressure(
        interior_mean=_safe_mean(interior_means),
        interior_std=_safe_mean(interior_stds),
        off_bubble_strain=_safe_mean(off_shares),
        boundary_stability=_safe_mean(boundary_stabilities),
        total_mean=_safe_mean(total_means),
        total_std=_safe_mean(total_stds),
        snapshots=len(snapshot_lists),
        pressure_label=STRAIN_AMPLIFIED,  # provisional; rewritten below
    )
    return _with_label(pressure, _classify_pressure(pressure))


def _with_label(p: BubblePressure, label: str) -> BubblePressure:
    """Return a copy of ``p`` with ``pressure_label`` replaced (frozen dataclass)."""
    return BubblePressure(
        interior_mean=p.interior_mean,
        interior_std=p.interior_std,
        off_bubble_strain=p.off_bubble_strain,
        boundary_stability=p.boundary_stability,
        total_mean=p.total_mean,
        total_std=p.total_std,
        snapshots=p.snapshots,
        pressure_label=label,
    )


def _boundary_stability_at(
    snapshot: Sequence[float],
    target: frozenset[int],
    k: int,
) -> float:
    """1 - symmetric-difference rate between top-k and target. No manual loops."""
    if k == 0 or len(snapshot) == 0:
        return 1.0
    ranked = sorted(
        range(len(snapshot)), key=lambda i: (-snapshot[i], i),
    )
    top_k = frozenset(ranked[:k])
    symdiff = top_k ^ target
    return 1.0 - len(symdiff) / (2 * k)


# --------------------------------------------------------------------------- #
# Verdicts: declarative rule table                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TuningVerdict:
    """A tuning recommendation + its rationale.

    ``action`` is one of :data:`VERDICT_LABELS`. ``law`` is the name of
    the law that matched. ``explanation`` is a short human-readable
    rationale derived from the law's description. ``pressure`` is the
    gauge reading the verdict was made on.
    """

    action: str
    law: str
    explanation: str
    pressure: BubblePressure


@dataclass(frozen=True)
class TuningLaw:
    """One law in the verdict rule table.

    A law is (name, predicate, action, explanation). ``predicate``
    consumes a :class:`BubblePressure` and returns a bool. Order in the
    table matters: first-match wins.
    """

    name: str
    predicate: Callable[[BubblePressure], bool]
    action: str
    explanation: str


# --------------------------------------------------------------------------- #
# Law predicates                                                              #
# --------------------------------------------------------------------------- #


def _diffuse(p: BubblePressure) -> bool:
    return p.pressure_label == DIFFUSE_PRESSURE


def _destructive(p: BubblePressure) -> bool:
    return p.pressure_label == DESTRUCTIVE_AMPLIFICATION


def _high_within_bubble_std(p: BubblePressure) -> bool:
    # Interior std rivals interior mean — interior wants to split.
    return p.interior_mean > 0.0 and p.interior_std >= 0.50 * p.interior_mean


def _resolved(p: BubblePressure) -> bool:
    # Bubble has done its work; total strain has gone to (near) zero.
    return p.total_mean <= 1e-9


def _stable_clean(p: BubblePressure) -> bool:
    # Diagnostic amplification with a stable boundary and tight interior.
    return (
        p.pressure_label == DIAGNOSTIC_AMPLIFICATION
        and p.off_bubble_strain <= 0.20
        and p.interior_std < 0.50 * max(p.interior_mean, 1e-9)
    )


def _strain_outside_dominant(p: BubblePressure) -> bool:
    # More strain off-bubble than in-bubble: merge with a neighbor.
    return p.off_bubble_strain >= 0.60


# --------------------------------------------------------------------------- #
# The rule table                                                              #
# --------------------------------------------------------------------------- #


RULES: tuple[TuningLaw, ...] = (
    TuningLaw(
        name="strain_dissipated",
        predicate=_resolved,
        action=PRUNE,
        explanation="total strain has dissipated; the bubble has resolved",
    ),
    TuningLaw(
        name="off_bubble_dominant",
        predicate=_strain_outside_dominant,
        action=MERGE,
        explanation="most strain lives outside the bubble; merge with an adjacent region",
    ),
    TuningLaw(
        name="diffuse_pressure",
        predicate=_diffuse,
        action=HOLD,
        explanation="low variance + nontrivial mean; nothing is separating yet, do not act",
    ),
    TuningLaw(
        name="high_interior_variance",
        predicate=_high_within_bubble_std,
        action=SPLIT,
        explanation="high within-bubble std; the interior wants a sub-bubble",
    ),
    TuningLaw(
        name="destructive_amplification",
        predicate=_destructive,
        action=HOLD,
        explanation="std is up but the boundary churns or leaks; do not stabilize off-phase",
    ),
    TuningLaw(
        name="stable_diagnostic_bubble",
        predicate=_stable_clean,
        action=STABILIZE,
        explanation="diagnostic amplification with a clean edge; stabilize and route",
    ),
)


_DEFAULT_LAW = TuningLaw(
    name="hold_no_law_matched",
    predicate=lambda _p: True,
    action=HOLD,
    explanation="no specific law matched; hold and keep observing",
)


def verdict_for(pressure: BubblePressure) -> TuningVerdict:
    """Find the first matching law and return its verdict.

    Field-style: no if/elif ladder; a ``next()`` over the rule table.
    """
    law = next(
        (rule for rule in RULES if rule.predicate(pressure)),
        _DEFAULT_LAW,
    )
    return TuningVerdict(
        action=law.action,
        law=law.name,
        explanation=law.explanation,
        pressure=pressure,
    )


# --------------------------------------------------------------------------- #
# Hierarchy decision                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HierarchyDecision:
    """Bubble-hierarchy recommendation derived from the verdict.

    ``action`` mirrors the verdict's action but adds extra structure for
    the split case: ``subbubble_indices`` lists the top-half interior
    indices that should become a sub-bubble. For non-split actions the
    field is empty.
    """

    action: str
    subbubble_indices: tuple[int, ...]
    explanation: str


def hierarchy_for(
    bubble: AddressBubble,
    pressure: BubblePressure,
) -> HierarchyDecision:
    """Promote a verdict into a hierarchy decision.

    For ``split``: the sub-bubble is the top-half of the current
    interior on its own strain ordering. For everything else, the
    action is forwarded and ``subbubble_indices`` is empty.
    """
    v = verdict_for(pressure)
    indices = (
        _top_half(bubble) if v.action == SPLIT else ()
    )
    return HierarchyDecision(
        action=v.action,
        subbubble_indices=indices,
        explanation=v.explanation,
    )


def _top_half(bubble: AddressBubble) -> tuple[int, ...]:
    """The top-half-by-strain slice of the bubble's interior."""
    profile = bubble.strain_profile
    interior_sorted = sorted(
        bubble.interior, key=lambda i: (-profile[i], i),
    )
    cut = max(1, len(interior_sorted) // 2)
    return tuple(interior_sorted[:cut])


# --------------------------------------------------------------------------- #
# Phase readout (Riordan-family aware, exploratory)                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhaseObservation:
    """One transform-family observation: (view_name, final per-variable strain)."""

    view_name: str
    interior_share: float
    off_bubble_share: float


@dataclass(frozen=True)
class PhaseReadout:
    """Phase classification across a family of transform observations.

    ``label`` is one of :data:`PHASE_LABELS`. ``best_view`` is the view
    with the highest interior_share. The readout is *exploratory* —
    no claim that Riordan family membership helps broadly.
    """

    label: str
    best_view: str
    best_interior_share: float
    spread: float
    explanation: str


def _phase_law_aligned(
    interior_shares: list[float], off_shares: list[float]
) -> bool:
    # A view exists with >= 0.60 interior share and low off-share.
    return any(
        i >= 0.60 and o <= 0.30
        for i, o in zip(interior_shares, off_shares)
    )


def _phase_law_off_phase(
    interior_shares: list[float], off_shares: list[float]
) -> bool:
    # No view has dominant interior, off-shares are mostly high.
    return (
        max(interior_shares) < 0.45
        and _safe_mean(off_shares) >= 0.50
    )


def _phase_law_over_smoothed(
    interior_shares: list[float], off_shares: list[float]
) -> bool:
    # Spread is tiny and means are middling — everything looks the same.
    return (
        _safe_stdev(interior_shares) <= 0.05
        and 0.20 <= _safe_mean(interior_shares) <= 0.55
    )


_PHASE_RULES: tuple[
    tuple[str, Callable[[list[float], list[float]], bool], str], ...
] = (
    (
        ALIGNED,
        _phase_law_aligned,
        "at least one transform produces a dominant interior with a clean edge",
    ),
    (
        OFF_PHASE,
        _phase_law_off_phase,
        "no transform localizes; strain mostly leaks off-bubble",
    ),
    (
        OVER_SMOOTHED,
        _phase_law_over_smoothed,
        "all transforms agree on a middling interior share; the family has collapsed",
    ),
)


def read_phase(observations: Sequence[PhaseObservation]) -> PhaseReadout:
    """Classify a transform family's alignment with the bubble framing.

    Exploratory. The first phase law matching the family wins; the
    default is ``needs_another_layer``.
    """
    if not observations:
        return PhaseReadout(
            label=NEEDS_ANOTHER_LAYER,
            best_view="",
            best_interior_share=0.0,
            spread=0.0,
            explanation="no observations; cannot classify phase",
        )

    interior_shares = [o.interior_share for o in observations]
    off_shares = [o.off_bubble_share for o in observations]

    best = max(observations, key=lambda o: (o.interior_share, -o.off_bubble_share))
    spread = _safe_stdev(interior_shares)

    matched = next(
        (
            (label, explanation)
            for label, predicate, explanation in _PHASE_RULES
            if predicate(interior_shares, off_shares)
        ),
        (
            NEEDS_ANOTHER_LAYER,
            "no phase law matched; the family may need an additional transform layer",
        ),
    )
    return PhaseReadout(
        label=matched[0],
        best_view=best.view_name,
        best_interior_share=best.interior_share,
        spread=spread,
        explanation=matched[1],
    )


# --------------------------------------------------------------------------- #
# Compact reporting                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TuningReportRow:
    """One row of the bubble-tuning report."""

    case_id: str
    pressure_label: str
    interior_mean: float
    interior_std: float
    off_bubble_strain: float
    boundary_stability: float
    action: str
    law: str


def tuning_row(
    case_id: str,
    pressure: BubblePressure,
    verdict: TuningVerdict,
) -> TuningReportRow:
    return TuningReportRow(
        case_id=case_id,
        pressure_label=pressure.pressure_label,
        interior_mean=pressure.interior_mean,
        interior_std=pressure.interior_std,
        off_bubble_strain=pressure.off_bubble_strain,
        boundary_stability=pressure.boundary_stability,
        action=verdict.action,
        law=verdict.law,
    )


def format_tuning_report(rows: Sequence[TuningReportRow]) -> str:
    """Deterministic plain-text tuning table."""
    header = (
        f"{'case':<32} {'pressure':<28} "
        f"{'i_mean':<7} {'i_std':<7} {'off':<6} {'b_stab':<7} "
        f"{'action':<11} {'law':<28}"
    )
    body = (
        f"{r.case_id:<32} {r.pressure_label:<28} "
        f"{r.interior_mean:<7.2f} {r.interior_std:<7.2f} "
        f"{r.off_bubble_strain:<6.2f} {r.boundary_stability:<7.2f} "
        f"{r.action:<11} {r.law:<28}"
        for r in rows
    )
    return "\n".join([header, "-" * len(header), *body])


def format_phase_readout(readout: PhaseReadout) -> str:
    return (
        f"phase: {readout.label} "
        f"(best={readout.best_view or '-'} "
        f"share={readout.best_interior_share:.2f} "
        f"spread={readout.spread:.2f}) "
        f"— {readout.explanation}"
    )
