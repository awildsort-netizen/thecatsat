#!/usr/bin/env python3
"""Riordan probe: do recurrence-preserving basis changes navigate strain better?

A companion to ``flattening_probe`` that adds a third family of
coordinate views built from the **Riordan / Pascal** family of
lower-triangular matrices. The motivating idea (informal):

    A Riordan array is a change-of-basis on generating functions that
    preserves combinatorial ancestry — it remaps coordinates without
    forgetting how each one was generated from the ones before it.
    Rotations (spectral views) preserve angles but discard recurrence
    structure. We ask whether a recurrence-preserving remap of the
    per-variable strain vector exposes flip directions that the raw
    and spectral views miss.

This is still a probe, not a proof. The transforms here pay the same
clause-check budget per flip as the raw/spectral views. The only
thing that changes is *which variable to flip next*.

Vocabulary
----------
- **Pascal view**: basis is a row-normalized lower-triangular Pascal
  matrix; column ``j`` holds the binomial weighting that maps a raw
  variable index ``i`` onto its ``j``-th "ancestor" coordinate.
- **Signed Pascal view**: the Pascal *inverse* — same shape, signs
  alternated by ``(-1)^(i-j)``. This is the Riordan pair partner of
  the plain Pascal matrix; together they invert each other.
- **Sierpinski view**: binomial coefficients mod 2 — the Pascal mask
  that draws Sierpinski's triangle. Bounded in ``{0, 1}``, useful as
  a "structure without magnitude" sanity check.

All bases are row-normalized to unit L2 length so that strain
projections stay scale-bounded across instance sizes.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from geometry.flattening_probe import (
    CoordinateView,
    FlatteningProbe,
    ProbeResult,
    _choose_raw,
    _choose_spectral,
    raw_view,
    spectral_view,
)
from sat_furnace import CNF, clause_satisfied


# --------------------------------------------------------------------------- #
# Riordan / Pascal basis constructors                                         #
# --------------------------------------------------------------------------- #


def pascal_matrix(n: int) -> np.ndarray:
    """Lower-triangular Pascal matrix: ``P[i, j] = C(i, j)`` for ``j <= i``.

    Built by the additive recurrence ``P[i, j] = P[i-1, j-1] + P[i-1, j]``
    rather than by computing binomial coefficients directly, both because
    it is cheaper for the sizes we care about and because the recurrence
    *is* the structure we are trying to preserve.
    """
    if n <= 0:
        return np.zeros((0, 0), dtype=float)
    matrix = np.zeros((n, n), dtype=float)
    matrix[0, 0] = 1.0
    for i in range(1, n):
        matrix[i, 0] = 1.0
        for j in range(1, i + 1):
            matrix[i, j] = matrix[i - 1, j - 1] + matrix[i - 1, j]
    return matrix


def signed_pascal_matrix(n: int) -> np.ndarray:
    """Signed Pascal matrix: ``S[i, j] = (-1)^(i-j) * C(i, j)``.

    This is the multiplicative inverse of ``pascal_matrix(n)``; the pair
    ``(P, S)`` is the canonical Riordan involution. It is the cleanest
    test of "recurrence-preserving coordinate flip": signs encode the
    inverse relation, magnitudes encode the ancestry weights.
    """
    base = pascal_matrix(n)
    if base.size == 0:
        return base
    signs = np.fromfunction(lambda i, j: (-1.0) ** (i - j), base.shape)
    return base * signs


def sierpinski_matrix(n: int) -> np.ndarray:
    """Pascal mod 2 — the Sierpinski mask. Entries are 0/1."""
    base = pascal_matrix(n)
    return np.mod(base, 2.0)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize to unit L2 length, leaving all-zero rows alone.

    We row-normalize (not column-normalize) because the basis matrix is
    applied as ``basis.T @ strain``: each *column* of ``basis`` is a
    direction in variable space, and we want each direction to be
    bounded. Row-normalizing the lower-triangular ``P`` and *then*
    transposing gives unit-length columns over the rows where the
    Pascal recurrence actually has support.
    """
    out = matrix.astype(float, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return out / safe


def pascal_view(n_vars: int) -> CoordinateView:
    """Pascal view: row-normalized lower-triangular Pascal basis."""
    basis = _row_normalize(pascal_matrix(n_vars))
    return CoordinateView(name="pascal", basis=basis)


def signed_pascal_view(n_vars: int) -> CoordinateView:
    """Signed Pascal view: row-normalized inverse-Pascal basis."""
    basis = _row_normalize(signed_pascal_matrix(n_vars))
    return CoordinateView(name="signed_pascal", basis=basis)


def sierpinski_view(n_vars: int) -> CoordinateView:
    """Sierpinski (Pascal mod 2) view: row-normalized 0/1 basis."""
    basis = _row_normalize(sierpinski_matrix(n_vars))
    return CoordinateView(name="sierpinski", basis=basis)


# --------------------------------------------------------------------------- #
# Chooser dispatch                                                            #
# --------------------------------------------------------------------------- #


_RIORDAN_VIEW_NAMES = frozenset({"pascal", "signed_pascal", "sierpinski"})


def _choose_riordan(formula: CNF, assignment, view: CoordinateView, rng: random.Random) -> int:
    """Pick the variable to flip via the Riordan-basis strain direction.

    Same shape as ``_choose_spectral`` from the flattening probe: project
    per-variable strain onto the view's basis, find the direction
    carrying the most strain, then flip the variable with the largest
    loading along that direction that also participates in some unsat
    clause. Reused intentionally — the question is "does the *basis*
    matter?", not "does the chooser?".
    """
    return _choose_spectral(formula, assignment, view, rng)


class RiordanProbe(FlatteningProbe):
    """Flattening probe that defaults to raw + spectral + Riordan views.

    Inherits the FlatteningProbe search loop unchanged so any difference
    between views is attributable to the basis, not the loop. The
    chooser for any Riordan-named view falls back to the spectral
    chooser (same projection logic; different basis).
    """

    def run(  # type: ignore[override]
        self,
        formula: CNF,
        n_vars: int,
        *,
        instance_id: str = "anon",
        planted_satisfiable: bool | None = None,
        views: Sequence[CoordinateView] | None = None,
    ) -> ProbeResult:
        if views is None:
            views = (
                raw_view(n_vars),
                spectral_view(formula, n_vars),
                pascal_view(n_vars),
                signed_pascal_view(n_vars),
                sierpinski_view(n_vars),
            )
        return super().run(
            formula=formula,
            n_vars=n_vars,
            instance_id=instance_id,
            planted_satisfiable=planted_satisfiable,
            views=views,
        )

    @staticmethod
    def _chooser_for(view: CoordinateView):  # type: ignore[override]
        if view.name == "raw":
            return _choose_raw
        if view.name in _RIORDAN_VIEW_NAMES:
            return _choose_riordan
        return _choose_spectral


# --------------------------------------------------------------------------- #
# Head-to-head summary helper                                                 #
# --------------------------------------------------------------------------- #


def head_to_head(results: list[ProbeResult], baseline: str = "raw") -> dict[str, dict[str, int]]:
    """For each non-baseline view, count win/tie/loss vs the baseline on
    final unsatisfied-clause count. Lower is better.
    """
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        if baseline not in result.runs:
            continue
        base_run = result.runs[baseline]
        for name, run in result.runs.items():
            if name == baseline:
                continue
            row = summary.setdefault(name, {"wins": 0, "ties": 0, "losses": 0})
            if run.final_unsatisfied < base_run.final_unsatisfied:
                row["wins"] += 1
            elif run.final_unsatisfied > base_run.final_unsatisfied:
                row["losses"] += 1
            else:
                row["ties"] += 1
    return summary


# --------------------------------------------------------------------------- #
# Per-family categorization + motion-type labels                              #
# --------------------------------------------------------------------------- #


def family_of(instance_id: str) -> str:
    """Coarse family label derived from the instance id prefix.

    The suite identifiers follow the pattern ``<family>_..._s<seed>``;
    we split on the first underscore-block that doesn't look like a
    parameter to derive a stable family label. Unknown ids fall through
    to ``other``.
    """
    if instance_id.startswith("2sat_easy"):
        return "2sat_easy"
    if instance_id.startswith("3sat_threshold"):
        return "3sat_threshold"
    if instance_id.startswith("3sat_v"):
        return "3sat_mid"
    if instance_id.startswith("unsat_struct"):
        return "unsat_struct"
    return "other"


def head_to_head_by_family(
    results: list[ProbeResult], baseline: str = "raw"
) -> dict[str, dict[str, dict[str, int]]]:
    """Same shape as ``head_to_head`` but bucketed by family.

    Returns ``{family: {view: {wins, ties, losses}}}``. A family with
    no comparisons (e.g. baseline missing from every instance) is
    omitted.
    """
    by_family: dict[str, list[ProbeResult]] = {}
    for result in results:
        by_family.setdefault(family_of(result.instance_id), []).append(result)
    return {fam: head_to_head(rs, baseline=baseline) for fam, rs in by_family.items()}


def plateau_length(trajectory: Sequence[float], tolerance: float = 1e-9) -> int:
    """Length of the longest stretch where total strain barely changed.

    A "plateau" here is a run of consecutive trajectory entries whose
    pairwise differences are within ``tolerance``. Returns the length
    of the longest such run (minimum 1 if the trajectory is non-empty).
    Used as one of the motion-type signals.
    """
    if not trajectory:
        return 0
    best = 1
    current = 1
    for prev, nxt in zip(trajectory, trajectory[1:]):
        if abs(nxt - prev) <= tolerance:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
    return best


def motion_label(
    baseline_run, view_run, *, plateau_threshold: int = 5
) -> str:
    """Compact label for how ``view_run`` differs from ``baseline_run``.

    Vocabulary, deliberately small:

    - ``unblocks_plateau`` — baseline plateaued (long flat strain
      stretch and didn't solve) and the view solved.
    - ``matches_raw`` — same final unsat as baseline.
    - ``destabilizes`` — view ended worse than baseline.
    - ``faster_same_outcome`` — same final unsat, but the view used
      meaningfully fewer flips (solved cases only).
    - ``slower_same_outcome`` — same final unsat, but the view used
      meaningfully more flips (solved cases only).

    This is not a typology; it is a small label set for orienting the
    eye over the case table.
    """
    if view_run.final_unsatisfied < baseline_run.final_unsatisfied:
        base_plateau = plateau_length(baseline_run.strain_trajectory)
        if not baseline_run.solved and base_plateau >= plateau_threshold and view_run.solved:
            return "unblocks_plateau"
        return "improves"
    if view_run.final_unsatisfied > baseline_run.final_unsatisfied:
        return "destabilizes"
    # Tie on final unsat.
    if view_run.solved and baseline_run.solved:
        if view_run.flips + 5 < baseline_run.flips:
            return "faster_same_outcome"
        if view_run.flips > baseline_run.flips + 5:
            return "slower_same_outcome"
    return "matches_raw"


def compact_trace(run, *, head: int = 4) -> str:
    """One-line deterministic trace of the first few flips + endpoints.

    Returns a string like ``"v3,v0,v5,v3 | plateau=12 | strain 12.0→3.0"``.
    The ``head`` controls how many variable picks we expose; default is
    small so the trace stays compact even in tables.
    """
    picks = ",".join(f"v{d.flipped_variable}" for d in run.decisions[:head])
    if not picks:
        picks = "-"
    plateau = plateau_length(run.strain_trajectory)
    if run.strain_trajectory:
        start = run.strain_trajectory[0]
        end = run.strain_trajectory[-1]
        strain_part = f"strain {start:.1f}→{end:.1f}"
    else:
        strain_part = "strain n/a"
    return f"{picks} | plateau={plateau} | {strain_part}"
