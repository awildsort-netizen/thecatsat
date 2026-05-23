#!/usr/bin/env python3
"""Stable local interiors with cheap containment tests and lifecycle labels.

A small first scaffold for the representational-bubble framing in
``docs/representational_bubbles.md``. PR #10 deliberately ships only
the design doc; this module adds the smallest empirical scaffold that
sits *next to* the existing geometry probes (``flattening_probe``,
``riordan_probe``, ``transform_litmus``) without changing them.

What this module *does*:

- Represents a candidate ``AddressBubble`` as a small index set around
  a high-strain center, derived from a per-variable strain vector
  (e.g. the residual strain a ``LitmusReading`` already replays).
- Exposes three cheap containment tests:
    1. distance-to-center (rank distance on the strain ordering),
    2. top-k strain membership,
    3. boundary test based on a strain-share *margin* between the
       interior tail and the next-out variable.
- Classifies the bubble across a short trace of strain vectors using
  the lifecycle labels in the design doc:
    ``seed``, ``inflated``, ``stable``, ``leaky``, ``plaque_risk``,
    ``merged``, ``pruned``.

What it does **not** do (deliberately, per PR #10's pacing guardrail):

- No new ``CoordinateView``, no solver loop changes, no merge/prune
  *implementation* beyond classification of an already-merged or
  already-pruned trace.
- No claim that the existing SAT suite produces stable bubbles. The
  experiment driver (see ``experiments/bubble_lifecycle.py``) reports
  what the real suite shows *and* runs one transparent synthetic toy
  trace, clearly labelled.

The scaffold is intentionally cheap (numpy + dataclasses), deterministic
given its inputs, and reuses :class:`StrainLocalization` from
``transform_litmus`` for boundary-margin language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from geometry.transform_litmus import StrainLocalization, localization_of


# --------------------------------------------------------------------------- #
# Lifecycle labels                                                            #
# --------------------------------------------------------------------------- #


SEED = "seed"
INFLATED = "inflated"
STABLE = "stable"
LEAKY = "leaky"
PLAQUE_RISK = "plaque_risk"
MERGED = "merged"
PRUNED = "pruned"

LIFECYCLE_LABELS = (SEED, INFLATED, STABLE, LEAKY, PLAQUE_RISK, MERGED, PRUNED)


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CollisionSeed:
    """Evidence that a region wants its own address space.

    Operationally: a variable with high residual strain *and*
    behavioral coupling to nearby variables. The cheapest read on this
    available from the existing probes is ``per_variable`` strain from
    ``transform_litmus`` (residual unsat-clause participation).
    Behavioral coupling is approximated by the *concentration* of
    strain in the seed's local index neighborhood — a low-address-
    distance, high-behavior-distance region in the design doc's
    vocabulary.
    """

    center: int
    strain_at_center: float
    neighborhood_strain: float
    view_name: str

    @property
    def concentration(self) -> float:
        """Share of local-neighborhood strain held by the center."""
        if self.neighborhood_strain <= 0.0:
            return 0.0
        return float(self.strain_at_center / self.neighborhood_strain)


@dataclass(frozen=True)
class AddressBubble:
    """A local interior around a collision seed.

    The bubble is specified by an *interior* index set (the top-strain
    variables forming the inflated address space) plus a *boundary*
    set (the next-out variables that form its typed edge). The
    ``radius`` is the rank distance on the strain ordering used to
    pick the interior. ``strain_profile`` is the per-variable strain
    vector this bubble was derived from, so containment tests are
    reproducible.
    """

    seed: CollisionSeed
    interior: tuple[int, ...]
    boundary: tuple[int, ...]
    radius: int
    strain_profile: tuple[float, ...]
    localization: StrainLocalization
    lifecycle: str

    @property
    def interior_size(self) -> int:
        return len(self.interior)

    @property
    def boundary_size(self) -> int:
        return len(self.boundary)


@dataclass(frozen=True)
class ContainmentReport:
    """Output of ``contains(bubble, item)``.

    ``inside`` is the headline yes/no. The other fields record *why*:
    distance-to-center on the strain ranking, whether the item is in
    the top-k strain set, whether it sits on the boundary, and the
    boundary margin (interior min share minus next-out share — large
    margin means the bubble has a clean edge).
    """

    item: int
    inside: bool
    on_boundary: bool
    distance_to_center: int
    in_top_k_strain: bool
    boundary_margin: float


@dataclass(frozen=True)
class LifecycleTrace:
    """Classification of a bubble across a short strain trace.

    The trace is a list of per-variable strain vectors, ordered in
    time. The classification rules in :func:`classify_lifecycle` look
    at: whether total strain decays, whether the interior membership
    churns, and whether the strain leaks to variables outside the
    bubble. The trace must contain at least one snapshot.
    """

    label: str
    interior_churn: float
    total_strain_delta: float
    boundary_leak: float
    snapshots: int


# --------------------------------------------------------------------------- #
# Seed detection                                                              #
# --------------------------------------------------------------------------- #


def _strain_rank_order(strain: Sequence[float]) -> list[int]:
    """Variable indices sorted by strain descending (ties broken by index).

    Deterministic; pure function of ``strain``.
    """
    arr = np.asarray(strain, dtype=float)
    # numpy argsort is stable; negate for descending, then break ties by
    # smaller index naturally because the input order is preserved.
    return [int(i) for i in np.argsort(-arr, kind="stable")]


def seed_from_strain(
    per_variable: Sequence[float],
    *,
    view_name: str = "raw",
    neighborhood: int = 3,
) -> CollisionSeed | None:
    """Pick the top-strain variable as the bubble seed.

    Returns ``None`` if all strain is zero (nothing to inflate around).
    ``neighborhood`` is the rank-window size used to estimate local
    concentration — small by design.
    """
    arr = np.asarray(per_variable, dtype=float)
    if arr.size == 0 or arr.sum() <= 0.0:
        return None
    order = _strain_rank_order(arr)
    center = order[0]
    n_window = max(1, min(neighborhood, arr.size))
    neighborhood_strain = float(arr[order[:n_window]].sum())
    return CollisionSeed(
        center=center,
        strain_at_center=float(arr[center]),
        neighborhood_strain=neighborhood_strain,
        view_name=view_name,
    )


# --------------------------------------------------------------------------- #
# Bubble construction                                                         #
# --------------------------------------------------------------------------- #


def inflate_bubble(
    per_variable: Sequence[float],
    seed: CollisionSeed,
    *,
    radius: int = 2,
    boundary_width: int = 2,
) -> AddressBubble:
    """Build an :class:`AddressBubble` around ``seed`` from a strain vector.

    The interior is the top-``radius+1`` variables on the strain
    ordering (the seed plus ``radius`` neighbors). The boundary is the
    next ``boundary_width`` variables — the typed edge. Lifecycle is
    assigned via :func:`classify_static` (single-snapshot rules).
    """
    arr = np.asarray(per_variable, dtype=float)
    order = _strain_rank_order(arr)
    radius = max(0, radius)
    boundary_width = max(0, boundary_width)

    interior_size = min(radius + 1, arr.size)
    interior = tuple(order[:interior_size])
    boundary = tuple(order[interior_size:interior_size + boundary_width])

    loc = localization_of(arr.tolist())
    lifecycle = classify_static(arr, interior, boundary, loc)
    return AddressBubble(
        seed=seed,
        interior=interior,
        boundary=boundary,
        radius=radius,
        strain_profile=tuple(float(x) for x in arr),
        localization=loc,
        lifecycle=lifecycle,
    )


def classify_static(
    strain: np.ndarray,
    interior: tuple[int, ...],
    boundary: tuple[int, ...],
    loc: StrainLocalization,
) -> str:
    """Static (single-snapshot) lifecycle label.

    Rules, in order:

    1. Empty interior or all-zero strain → :data:`PRUNED`.
    2. Interior holds < 30% of total strain → :data:`SEED`
       (collision detected, no real interior yet).
    3. Interior holds >= 30% but the boundary margin is tiny
       (boundary strain >= 80% of interior min) → :data:`LEAKY`.
    4. Interior holds the dominant share (>= 60%) with a clean
       boundary margin → :data:`INFLATED`. Whether the bubble is
       actually *stable* is a multi-snapshot question;
       :data:`classify_lifecycle` answers that.
    5. Otherwise → :data:`SEED`.
    """
    total = float(strain.sum())
    if not interior or total <= 0.0:
        return PRUNED
    interior_strain = float(strain[list(interior)].sum())
    interior_share = interior_strain / total

    interior_min = float(strain[list(interior)].min()) if interior else 0.0
    boundary_max = float(strain[list(boundary)].max()) if boundary else 0.0

    if interior_share < 0.30:
        return SEED
    if interior_min > 0.0 and boundary_max >= 0.80 * interior_min:
        return LEAKY
    if interior_share >= 0.60:
        return INFLATED
    return SEED


# --------------------------------------------------------------------------- #
# Containment tests                                                           #
# --------------------------------------------------------------------------- #


def contains(bubble: AddressBubble, item: int) -> ContainmentReport:
    """Cheap containment test: distance + top-k + boundary margin.

    Distance-to-center is rank distance on the strain ordering: 0 for
    the seed center, 1 for the next-highest-strain variable, etc.
    ``in_top_k_strain`` is interior membership. ``on_boundary`` is
    boundary membership. ``boundary_margin`` is the gap between the
    interior's minimum strain and the boundary's maximum strain
    (positive = clean edge; non-positive = leaky).
    """
    strain = np.asarray(bubble.strain_profile, dtype=float)
    order = _strain_rank_order(strain)
    rank_index = {v: i for i, v in enumerate(order)}
    distance = rank_index.get(item, len(order))

    interior_set = set(bubble.interior)
    boundary_set = set(bubble.boundary)
    inside = item in interior_set
    on_boundary = item in boundary_set

    if bubble.interior and bubble.boundary:
        interior_min = float(strain[list(bubble.interior)].min())
        boundary_max = float(strain[list(bubble.boundary)].max())
        margin = interior_min - boundary_max
    else:
        margin = 0.0

    return ContainmentReport(
        item=item,
        inside=inside,
        on_boundary=on_boundary,
        distance_to_center=distance,
        in_top_k_strain=inside,
        boundary_margin=margin,
    )


def boundary_margin(bubble: AddressBubble) -> float:
    """Convenience: interior_min strain minus boundary_max strain."""
    if not bubble.interior or not bubble.boundary:
        return 0.0
    strain = np.asarray(bubble.strain_profile, dtype=float)
    interior_min = float(strain[list(bubble.interior)].min())
    boundary_max = float(strain[list(bubble.boundary)].max())
    return interior_min - boundary_max


# --------------------------------------------------------------------------- #
# Lifecycle across a trace                                                    #
# --------------------------------------------------------------------------- #


def classify_lifecycle(
    bubble: AddressBubble,
    trace: Sequence[Sequence[float]],
    *,
    churn_threshold: float = 0.5,
    leak_threshold: float = 0.5,
) -> LifecycleTrace:
    """Classify a bubble across a short trace of strain vectors.

    The trace is per-variable strain over time. We measure:

    - **interior_churn** — symmetric-difference rate between the
      bubble's recorded interior and the top-strain set of each
      snapshot. ``0`` = interior membership stays constant; ``1`` =
      total turnover at every snapshot.
    - **total_strain_delta** — final total strain minus initial total
      strain. Negative = strain dissipating; positive = strain growing.
    - **boundary_leak** — average share of total strain held by
      variables that are neither interior nor boundary across the
      trace. High leak = the bubble is not actually containing the
      pressure it was inflated around.

    Labels (final):

    - :data:`PRUNED` — strain ended at zero (the bubble's region
      resolved).
    - :data:`MERGED` — interior expanded into the boundary set
      consistently (boundary churn high but stays adjacent).
    - :data:`STABLE` — low churn, non-positive strain delta, low leak.
    - :data:`LEAKY` — significant leak share regardless of churn.
    - :data:`PLAQUE_RISK` — non-trivial interior strain, low churn,
      strain not decreasing — the bubble holds address space without
      doing work, the failure mode named in
      ``parser_evolver/docs/hallucination_geometry.md``.
    - :data:`INFLATED` — fallback when the trace is too short or the
      readings are mixed.
    """
    if not trace:
        return LifecycleTrace(
            label=bubble.lifecycle,
            interior_churn=0.0,
            total_strain_delta=0.0,
            boundary_leak=0.0,
            snapshots=0,
        )

    interior_set = set(bubble.interior)
    boundary_set = set(bubble.boundary)
    radius = len(bubble.interior)
    snapshots = [np.asarray(s, dtype=float) for s in trace]

    # Interior churn: per-snapshot symmetric difference between the
    # bubble's interior and that snapshot's top-radius variables.
    if radius == 0:
        churn = 0.0
    else:
        churns = []
        for s in snapshots:
            order = _strain_rank_order(s)
            top = set(order[:radius])
            symdiff = top ^ interior_set
            churns.append(len(symdiff) / (2 * radius))
        churn = float(np.mean(churns))

    total_first = float(snapshots[0].sum())
    total_last = float(snapshots[-1].sum())
    total_delta = total_last - total_first

    # Boundary leak: average share of strain held by variables not in
    # interior ∪ boundary.
    leak_shares: list[float] = []
    for s in snapshots:
        total = float(s.sum())
        if total <= 0.0:
            leak_shares.append(0.0)
            continue
        outside = [i for i in range(s.size) if i not in interior_set and i not in boundary_set]
        if not outside:
            leak_shares.append(0.0)
            continue
        leak_shares.append(float(s[outside].sum()) / total)
    leak = float(np.mean(leak_shares)) if leak_shares else 0.0

    # Merge detection: did the interior ranking expand into boundary?
    final_top_2r = set(_strain_rank_order(snapshots[-1])[:max(1, 2 * radius)])
    merged_into_boundary = (
        radius > 0
        and boundary_set
        and boundary_set.issubset(final_top_2r)
        and len(final_top_2r & interior_set) >= radius // 2
    )

    if total_last <= 0.0:
        label = PRUNED
    elif merged_into_boundary and churn >= 0.25:
        label = MERGED
    elif leak >= leak_threshold:
        label = LEAKY
    elif churn <= churn_threshold and total_delta <= 0.0:
        label = STABLE
    elif churn <= churn_threshold and total_delta > 0.0:
        label = PLAQUE_RISK
    else:
        label = INFLATED

    return LifecycleTrace(
        label=label,
        interior_churn=churn,
        total_strain_delta=total_delta,
        boundary_leak=leak,
        snapshots=len(snapshots),
    )


# --------------------------------------------------------------------------- #
# Compact reporting                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BubbleReportRow:
    """One row of the demo containment/lifecycle report."""

    case_id: str
    interior: tuple[int, ...]
    boundary: tuple[int, ...]
    boundary_margin: float
    static_label: str
    trace_label: str
    interior_churn: float
    boundary_leak: float
    total_strain_delta: float


def report_row(
    case_id: str,
    bubble: AddressBubble,
    trace: LifecycleTrace,
) -> BubbleReportRow:
    return BubbleReportRow(
        case_id=case_id,
        interior=bubble.interior,
        boundary=bubble.boundary,
        boundary_margin=boundary_margin(bubble),
        static_label=bubble.lifecycle,
        trace_label=trace.label,
        interior_churn=trace.interior_churn,
        boundary_leak=trace.boundary_leak,
        total_strain_delta=trace.total_strain_delta,
    )


def format_report(rows: Sequence[BubbleReportRow]) -> str:
    """Deterministic plain-text containment/lifecycle table."""
    header = (
        f"{'case':<32} {'interior':<14} {'boundary':<14} "
        f"{'margin':<8} {'static':<10} {'trace':<12} "
        f"{'churn':<7} {'leak':<7} {'Δstrain':<8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.case_id:<32} "
            f"{str(list(row.interior)):<14} "
            f"{str(list(row.boundary)):<14} "
            f"{row.boundary_margin:<8.2f} "
            f"{row.static_label:<10} "
            f"{row.trace_label:<12} "
            f"{row.interior_churn:<7.2f} "
            f"{row.boundary_leak:<7.2f} "
            f"{row.total_strain_delta:<+8.2f}"
        )
    return "\n".join(lines)
