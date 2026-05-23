#!/usr/bin/env python3
"""Riordan bubble fit: pick the transform that gives the most stable bubble.

Stacked on top of :mod:`geometry.riordan_probe` (which supplies the
Pascal / signed-Pascal / Sierpinski Riordan-family bases),
:mod:`geometry.bubble_lifecycle` (seed + inflate), and
:mod:`geometry.bubble_tuning` (pressure gauge + verdict table).

The question the module answers is, by deliberate construction, **not**
"which transform solves SAT here?". It is:

    Given a per-variable strain vector, which candidate transform
    (raw / pascal / signed_pascal / sierpinski, optionally phase shifts)
    produces the cleanest *bubble interior* — high interior concentration,
    low off-bubble leak, low within-bubble variance — under the
    existing bubble pressure / lifecycle gauges?

That is, choose the transform by **pressure / containment** criteria,
never by solver outcome. A constructive transform is one that turns the
diffuse raw strain into something a bubble can hold; a destructive
transform smears it.

Design notes:

- **No giant if/elif tree.** Candidates are :class:`TransformCandidate`s
  in a small registry; the fit reduces over the registry with a
  composite score, and ties break deterministically by candidate name.
- **No outcome leakage.** Scoring sees only strain + bubble + pressure;
  it never sees ``solved`` or the SAT formula.
- **Fallback to identity.** If every candidate scores destructive /
  diffuse, the fitter returns the raw view and publishes a veto signal
  the existing :func:`strategy.operators.bubble_pressure_gate` can
  consume.

The module is pure. Its only side effect is the field-channel write the
companion operator in :mod:`strategy.operators` performs when it consumes
the :class:`FitDecision` this module returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from geometry.bubble_lifecycle import (
    AddressBubble,
    inflate_bubble,
    seed_from_strain,
)
from geometry.bubble_tuning import (
    DESTRUCTIVE_AMPLIFICATION,
    DIAGNOSTIC_AMPLIFICATION,
    DIFFUSE_PRESSURE,
    BubblePressure,
    measure_pressure,
)
from geometry.flattening_probe import CoordinateView, raw_view
from geometry.riordan_probe import (
    pascal_view,
    signed_pascal_view,
    sierpinski_view,
)


# --------------------------------------------------------------------------- #
# Candidate registry                                                          #
# --------------------------------------------------------------------------- #


# Each candidate is (name, view-builder). The view builder takes ``n_vars``
# and returns a :class:`CoordinateView`. The fitter consults the registry
# in order; ties on the composite score break by registry position, then
# alphabetically by name — both deterministic.
ViewBuilder = Callable[[int], CoordinateView]


@dataclass(frozen=True)
class TransformCandidate:
    """One row of the fit registry: a named transform and its view builder."""

    name: str
    build: ViewBuilder


def _identity_view(n: int) -> CoordinateView:
    return raw_view(n)


def _pascal(n: int) -> CoordinateView:
    return pascal_view(n)


def _signed_pascal(n: int) -> CoordinateView:
    return signed_pascal_view(n)


def _sierpinski(n: int) -> CoordinateView:
    return sierpinski_view(n)


# Default candidate set. ``identity`` (a.k.a. raw) is always first so a
# tied score keeps the most conservative choice. The Riordan family
# follows in canonical order — Pascal, then its signed inverse, then the
# mod-2 mask. Phase / reindex variants can be appended without changing
# the fitter.
DEFAULT_CANDIDATES: tuple[TransformCandidate, ...] = (
    TransformCandidate(name="identity", build=_identity_view),
    TransformCandidate(name="pascal", build=_pascal),
    TransformCandidate(name="signed_pascal", build=_signed_pascal),
    TransformCandidate(name="sierpinski", build=_sierpinski),
)


# --------------------------------------------------------------------------- #
# Phase / reindex helpers — small, optional, deterministic                    #
# --------------------------------------------------------------------------- #


def _shifted_basis(basis: np.ndarray, shift: int) -> np.ndarray:
    """Roll a basis's column order by ``shift`` positions.

    A cheap "phase shift" on the same Riordan structure: same recurrence,
    different column-to-coordinate alignment. The fitter can pick a
    shifted variant when the raw alignment is off-phase relative to the
    strain.
    """
    if basis.size == 0:
        return basis
    return np.roll(basis, shift=shift, axis=1)


def phase_shifted_candidate(
    base: TransformCandidate, shift: int
) -> TransformCandidate:
    """A new candidate whose basis is ``base``'s basis rolled by ``shift``."""

    def build(n: int) -> CoordinateView:
        view = base.build(n)
        return CoordinateView(
            name=f"{view.name}+shift{shift}", basis=_shifted_basis(view.basis, shift)
        )

    return TransformCandidate(name=f"{base.name}+shift{shift}", build=build)


# --------------------------------------------------------------------------- #
# Strain projection                                                           #
# --------------------------------------------------------------------------- #


def project_strain(strain: Sequence[float], view: CoordinateView) -> np.ndarray:
    """Project a per-variable strain vector through ``view`` and back.

    The strain is re-weighted by the view's basis: ``view.basis @ (view.basis.T @ s)``.
    Identity (raw) is the obvious fixed point; Pascal / signed-Pascal /
    Sierpinski rebalance the per-variable strain according to their
    recurrence weights. The result is non-negative (we take ``abs``)
    because bubble-lifecycle and pressure gauges only consume magnitudes.
    """
    arr = np.asarray(strain, dtype=float)
    if arr.size == 0 or view.basis.size == 0:
        return arr
    projected = view.basis @ (view.basis.T @ arr)
    return np.abs(projected)


# --------------------------------------------------------------------------- #
# Per-candidate fit report                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CandidateReport:
    """One row of the fit table: candidate + its bubble + its pressure + score.

    ``score`` is the composite stability score (higher = more stable).
    ``viable`` is True when the candidate's pressure label is not in the
    set of destructive labels — a downstream operator may choose to veto
    everything when *no* candidate is viable.
    """

    name: str
    view_name: str
    interior: tuple[int, ...]
    boundary: tuple[int, ...]
    pressure_label: str
    interior_share: float
    off_bubble_strain: float
    interior_std: float
    boundary_stability: float
    score: float
    viable: bool


# Pressure labels we consider *destructive* (a hint not to stabilize on).
_DESTRUCTIVE_LABELS: frozenset[str] = frozenset(
    {DESTRUCTIVE_AMPLIFICATION, DIFFUSE_PRESSURE}
)


def _interior_share(strain: np.ndarray, interior: Sequence[int]) -> float:
    total = float(strain.sum())
    if total <= 0.0 or not interior:
        return 0.0
    return float(strain[list(interior)].sum() / total)


def _stability_score(
    interior_share: float,
    off_bubble_strain: float,
    interior_std: float,
    interior_mean: float,
    boundary_stability: float,
) -> float:
    """Composite stability score in roughly [-1, 1]; higher is more stable.

    The score rewards a high interior share, a stable boundary, and
    diagnostic (not destructive) amplification. It penalizes leak and
    within-bubble variance. The weights are deliberately small integers
    so the score reads as a sum of cheap signals, not a tuned model.
    """
    within_var = (interior_std / interior_mean) if interior_mean > 0.0 else 0.0
    return (
        1.0 * interior_share
        + 0.5 * boundary_stability
        - 1.0 * off_bubble_strain
        - 0.5 * min(within_var, 1.0)
    )


def evaluate_candidate(
    candidate: TransformCandidate,
    strain: Sequence[float],
    *,
    trace: Sequence[Sequence[float]] | None = None,
    radius: int = 2,
    boundary_width: int = 2,
) -> CandidateReport | None:
    """Apply ``candidate`` to ``strain``, inflate a bubble, score the result.

    Returns ``None`` if the strain projects to a zero vector (no bubble
    can be inflated). The score and pressure label come from the
    existing bubble-lifecycle + tuning gauges; this function does *not*
    invent a new metric, it just routes through the published ones.

    If ``trace`` is supplied, each snapshot is projected through the
    candidate's basis before the gauge reads pressure; this lets a real
    time-varying strain history reveal destructive amplification (top-k
    churn, boundary turnover) that a static snapshot cannot.
    """
    arr = np.asarray(strain, dtype=float)
    if arr.size == 0:
        return None

    view = candidate.build(arr.size)
    projected = project_strain(arr, view)
    if not np.any(projected > 0.0):
        return None

    seed = seed_from_strain(projected.tolist(), view_name=candidate.name)
    if seed is None:
        return None

    bubble = inflate_bubble(
        projected.tolist(), seed, radius=radius, boundary_width=boundary_width
    )
    # If we got a real trace, project each snapshot through the same
    # basis so the pressure gauge reads the candidate's own time history.
    # Otherwise the cheapest honest trace is the projected strain
    # replayed across a few snapshots — a degenerate static read.
    if trace:
        gauge_trace = tuple(
            tuple(project_strain(snap, view).tolist()) for snap in trace
        )
    else:
        gauge_trace = tuple(tuple(projected.tolist()) for _ in range(3))
    pressure = measure_pressure(bubble, gauge_trace)

    share = _interior_share(projected, bubble.interior)
    score = _stability_score(
        interior_share=share,
        off_bubble_strain=pressure.off_bubble_strain,
        interior_std=pressure.interior_std,
        interior_mean=pressure.interior_mean,
        boundary_stability=pressure.boundary_stability,
    )

    return CandidateReport(
        name=candidate.name,
        view_name=view.name,
        interior=bubble.interior,
        boundary=bubble.boundary,
        pressure_label=pressure.pressure_label,
        interior_share=share,
        off_bubble_strain=pressure.off_bubble_strain,
        interior_std=pressure.interior_std,
        boundary_stability=pressure.boundary_stability,
        score=score,
        viable=pressure.pressure_label not in _DESTRUCTIVE_LABELS,
    )


# --------------------------------------------------------------------------- #
# Fit decision                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FitDecision:
    """The fitter's chosen candidate plus the full table.

    - ``selected`` names the chosen candidate.
    - ``view`` is the corresponding :class:`CoordinateView`.
    - ``bubble`` is the inflated bubble on the projected strain.
    - ``veto`` is True when *no* candidate was viable; downstream
      operators should fall back to raw / suppress the transformed
      ranker.
    - ``reports`` is the full per-candidate table (deterministic order).
    """

    selected: str
    view: CoordinateView
    bubble: AddressBubble | None
    veto: bool
    reports: tuple[CandidateReport, ...]
    rationale: str


_IDENTITY_NAME = "identity"


def _pick_best(reports: Sequence[CandidateReport]) -> CandidateReport | None:
    """Pick the highest-score viable report.

    Tie-break: registry order (preserved by stability of ``max``) then
    name. Returns ``None`` only if ``reports`` is empty.
    """
    viable = [r for r in reports if r.viable]
    pool = viable or list(reports)
    if not pool:
        return None
    # Score descending; on tie, prefer ``identity`` (the most
    # conservative), then alphabetic. Encode all of that in a sort key.
    def key(r: CandidateReport) -> tuple[float, int, str]:
        prefer_identity = 0 if r.name == _IDENTITY_NAME else 1
        return (-r.score, prefer_identity, r.name)

    pool.sort(key=key)
    return pool[0]


def fit(
    strain: Sequence[float],
    *,
    trace: Sequence[Sequence[float]] | None = None,
    candidates: Sequence[TransformCandidate] = DEFAULT_CANDIDATES,
    radius: int = 2,
    boundary_width: int = 2,
) -> FitDecision:
    """Run every candidate, score, choose the most stable.

    Pure function. Deterministic in ``strain`` and ``candidates``. If
    every candidate's pressure label is destructive/diffuse, the
    decision's ``veto`` flag is set, and the conservative ``identity``
    view is returned so a downstream raw ranker can take over.

    If a ``trace`` (per-variable strain history) is given, the gauge
    runs on the projected trace; otherwise the static strain is replayed
    across snapshots. The trace path is what surfaces destructive
    amplification — top-k churn and boundary turnover.
    """
    arr = np.asarray(strain, dtype=float)
    n = arr.size
    reports = tuple(
        report
        for report in (
            evaluate_candidate(
                c, strain, trace=trace,
                radius=radius, boundary_width=boundary_width,
            )
            for c in candidates
        )
        if report is not None
    )

    if not reports:
        # Nothing to inflate — empty / zero strain. Return identity with
        # veto False so the composer keeps yielding to the raw ranker.
        return FitDecision(
            selected=_IDENTITY_NAME,
            view=raw_view(n),
            bubble=None,
            veto=False,
            reports=(),
            rationale="empty_strain_no_candidates",
        )

    best = _pick_best(reports)
    assert best is not None  # reports is non-empty
    any_viable = any(r.viable for r in reports)
    veto = not any_viable

    if veto:
        # Override the picked view with identity — the conservative
        # fallback signals "no constructive transform; do not amplify".
        identity_report = next(
            (r for r in reports if r.name == _IDENTITY_NAME), None
        )
        chosen_name = _IDENTITY_NAME
        chosen_view = raw_view(n)
        chosen_bubble = None
        if identity_report is not None:
            chosen_bubble = inflate_bubble(
                project_strain(arr, raw_view(n)).tolist(),
                seed_from_strain(arr.tolist(), view_name=_IDENTITY_NAME)
                or seed_from_strain([1.0] + [0.0] * (n - 1)),
                radius=radius,
                boundary_width=boundary_width,
            )
        rationale = (
            "no_viable_candidate; "
            f"best_destructive={best.name}@{best.pressure_label}; "
            "fall_back_to_identity_and_veto"
        )
        return FitDecision(
            selected=chosen_name,
            view=chosen_view,
            bubble=chosen_bubble,
            veto=True,
            reports=reports,
            rationale=rationale,
        )

    # At least one viable candidate.
    chosen_view = _view_for(best.name, n, candidates)
    chosen_bubble = inflate_bubble(
        project_strain(arr, chosen_view).tolist(),
        seed_from_strain(
            project_strain(arr, chosen_view).tolist(), view_name=best.name
        )
        or seed_from_strain([1.0] + [0.0] * (n - 1)),
        radius=radius,
        boundary_width=boundary_width,
    )
    rationale = (
        f"selected={best.name}@{best.pressure_label} "
        f"score={best.score:.3f} interior_share={best.interior_share:.2f}"
    )
    return FitDecision(
        selected=best.name,
        view=chosen_view,
        bubble=chosen_bubble,
        veto=False,
        reports=reports,
        rationale=rationale,
    )


def _view_for(
    name: str, n: int, candidates: Sequence[TransformCandidate]
) -> CoordinateView:
    """Look up the view builder by name; default to raw if not found."""
    for c in candidates:
        if c.name == name:
            return c.build(n)
    return raw_view(n)


# --------------------------------------------------------------------------- #
# Compact report formatting                                                   #
# --------------------------------------------------------------------------- #


def format_fit_table(decision: FitDecision) -> str:
    """Deterministic plain-text fit table — one row per candidate."""
    header = (
        f"{'candidate':<22} {'pressure':<28} "
        f"{'i_share':<8} {'off':<6} {'i_std':<7} "
        f"{'b_stab':<7} {'score':<7} {'viable':<6}"
    )
    lines = [header, "-" * len(header)]
    for r in decision.reports:
        lines.append(
            f"{r.name:<22} {r.pressure_label:<28} "
            f"{r.interior_share:<8.2f} {r.off_bubble_strain:<6.2f} "
            f"{r.interior_std:<7.2f} {r.boundary_stability:<7.2f} "
            f"{r.score:<7.3f} {'yes' if r.viable else 'no':<6}"
        )
    lines.append("")
    lines.append(
        f"selected={decision.selected} "
        f"veto={'yes' if decision.veto else 'no'} "
        f"— {decision.rationale}"
    )
    return "\n".join(lines)
