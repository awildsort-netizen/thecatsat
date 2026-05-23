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
