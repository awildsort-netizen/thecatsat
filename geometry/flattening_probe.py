#!/usr/bin/env python3
"""Flattening probe: does a cheap coordinate transform reduce constraint strain?

This is a *probe*, not a proof. The user's framing is that some SAT
hardness may be representation geometry — the solver searches for
coordinate systems where constraints stop fighting each other.
Concretely we ask whether picking variables to flip in a transformed
(spectral) basis of the signed variable-clause incidence matrix moves
through lower aggregate strain than picking them in the raw symbolic
basis.

Nothing here claims to dent NP-hardness. The transformed view does the
same number of clause checks per step; the question is only whether
its *flip choices* track contradiction better on tiny instances.

Vocabulary
----------
- ``CoordinateView``: a representation of variables/clauses/assignments.
  We expose two: ``raw_view`` (canonical variable indices) and
  ``spectral_view`` (top-k right-singular-vector basis of the signed
  incidence matrix).
- ``ConstraintStrain``: per-variable scalar that summarizes local
  contradiction pressure. We use the count of currently-unsatisfied
  clauses each variable participates in.
- ``FlatteningProbe``: runs the same WalkSAT-flavored search through
  two views and reports flips-to-solve, residual strain, and the
  trajectory of total strain.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from sat_furnace import CNF, Clause, clause_satisfied


def _stable_view_seed_offset(name: str) -> int:
    """Process-independent integer derived from a view name.

    The probe seeds each view's RNG from ``self.seed + offset(view)`` so
    different views explore differently from the same starting point.
    Using Python's built-in ``hash()`` here would make the seed
    process-dependent (PYTHONHASHSEED randomization), which breaks
    cross-run reproducibility of the report. ``hashlib.md5`` is
    deterministic across processes; we fold to a small int.
    """
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 10_000


Assignment = list[bool]


@dataclass(frozen=True)
class ConstraintStrain:
    """Local contradiction pressure decomposed onto a coordinate basis.

    ``per_variable`` is the raw per-variable strain (unsat-clause
    participation count). ``per_direction`` is the same vector
    projected onto whatever basis the view is using; for ``raw_view``
    the two coincide. ``total`` is just ``sum(per_variable)``.
    """

    per_variable: tuple[float, ...]
    per_direction: tuple[float, ...]
    total: float


@dataclass(frozen=True)
class CoordinateView:
    """A representation of variables/clauses/assignments.

    A view is fully specified by its ``basis`` (an n_vars × k matrix
    whose columns are coordinate directions; ``k == n_vars`` for an
    invertible view, ``k < n_vars`` is a partial flattening). For the
    raw view the basis is the identity; for the spectral view it is
    the top right-singular vectors of the signed incidence matrix.
    """

    name: str
    basis: np.ndarray  # shape (n_vars, k)

    @property
    def n_vars(self) -> int:
        return int(self.basis.shape[0])

    @property
    def k(self) -> int:
        return int(self.basis.shape[1])

    def project(self, per_variable: Sequence[float]) -> np.ndarray:
        vec = np.asarray(per_variable, dtype=float)
        return self.basis.T @ vec

    def strain(self, formula: CNF, assignment: Assignment) -> ConstraintStrain:
        per_var = _per_variable_strain(formula, assignment, self.n_vars)
        per_dir = self.project(per_var)
        return ConstraintStrain(
            per_variable=tuple(per_var),
            per_direction=tuple(float(x) for x in per_dir),
            total=float(per_var.sum()),
        )


def _per_variable_strain(formula: CNF, assignment: Assignment, n_vars: int) -> np.ndarray:
    strain = np.zeros(n_vars, dtype=float)
    for clause in formula:
        if clause_satisfied(clause, assignment):
            continue
        for variable, _ in clause:
            strain[variable] += 1.0
    return strain


def signed_incidence(formula: CNF, n_vars: int) -> np.ndarray:
    """Signed variable-clause incidence: ``+1`` if literal is positive,
    ``-1`` if negated, ``0`` otherwise. Shape ``(n_vars, n_clauses)``.
    """
    matrix = np.zeros((n_vars, len(formula)), dtype=float)
    for clause_index, clause in enumerate(formula):
        for variable, is_negated in clause:
            matrix[variable, clause_index] = -1.0 if is_negated else 1.0
    return matrix


def raw_view(n_vars: int) -> CoordinateView:
    return CoordinateView(name="raw", basis=np.eye(n_vars))


def spectral_view(formula: CNF, n_vars: int, k: int | None = None) -> CoordinateView:
    """Top-``k`` right-singular directions of the signed incidence matrix.

    These are the directions along which variables co-vary most
    strongly across clauses — the natural candidate for "coordinates
    where constraints stop fighting each other". When ``k`` is ``None``
    we keep the full rank, giving an orthogonal rotation of the raw
    basis; smaller ``k`` produces a deliberate partial flattening.
    """
    if len(formula) == 0:
        return raw_view(n_vars)
    incidence = signed_incidence(formula, n_vars)
    # Operate on variable-variable correlation: U Sigma V^T, take U.
    # ``full_matrices=False`` keeps it cheap on small instances.
    u, _sigma, _vt = np.linalg.svd(incidence, full_matrices=False)
    if k is None:
        k = u.shape[1]
    k = max(1, min(k, u.shape[1]))
    basis = u[:, :k]
    # Pad back to n_vars columns with zero-fill so projection is well-defined
    # but the view still has rank k.
    if k < n_vars:
        padded = np.zeros((n_vars, n_vars))
        padded[:, :k] = basis
        basis = padded
    return CoordinateView(name=f"spectral(k={k})", basis=basis)


# --------------------------------------------------------------------------- #
# Search loop                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class StepRecord:
    step: int
    flipped_variable: int
    unsatisfied: int
    total_strain: float
    dominant_direction: int


@dataclass
class ProbeRunResult:
    view_name: str
    solved: bool
    flips: int
    final_unsatisfied: int
    initial_unsatisfied: int
    strain_trajectory: tuple[float, ...]
    decisions: tuple[StepRecord, ...]


@dataclass
class ProbeResult:
    """Aggregate comparison across views on a single instance."""

    instance_id: str
    n_vars: int
    n_clauses: int
    planted_satisfiable: bool | None
    runs: dict[str, ProbeRunResult] = field(default_factory=dict)


VariableChooser = Callable[[CNF, Assignment, CoordinateView, random.Random], int]


def _choose_raw(formula: CNF, assignment: Assignment, view: CoordinateView, rng: random.Random) -> int:
    """WalkSAT-style: pick an unsat clause, then the variable in it
    whose flip would minimize the resulting unsat count (greedy, with
    rare random walk to escape plateaus).
    """
    unsat_clauses = [c for c in formula if not clause_satisfied(c, assignment)]
    clause = rng.choice(unsat_clauses)
    if rng.random() < 0.1:
        return clause[rng.randrange(len(clause))][0]
    return _best_variable_in_clause(formula, assignment, clause)


def _choose_spectral(formula: CNF, assignment: Assignment, view: CoordinateView, rng: random.Random) -> int:
    """Project per-variable strain onto the view's basis, find the
    direction carrying the most strain, then flip the variable with
    the largest loading along it that also appears in some unsat
    clause. This is "follow the strain in the rotated frame".
    """
    strain = view.strain(formula, assignment)
    per_direction = np.asarray(strain.per_direction)
    if per_direction.size == 0 or np.allclose(per_direction, 0.0):
        return _choose_raw(formula, assignment, view, rng)
    dominant = int(np.argmax(np.abs(per_direction)))
    loadings = np.abs(view.basis[:, dominant])
    unsat_clauses = [c for c in formula if not clause_satisfied(c, assignment)]
    if not unsat_clauses:
        return int(np.argmax(loadings))
    candidate_vars = {variable for clause in unsat_clauses for variable, _ in clause}
    if rng.random() < 0.1:
        clause = rng.choice(unsat_clauses)
        return clause[rng.randrange(len(clause))][0]
    best_var = max(candidate_vars, key=lambda v: loadings[v])
    return best_var


def _best_variable_in_clause(formula: CNF, assignment: Assignment, clause: Clause) -> int:
    best_var = clause[0][0]
    best_unsat = _unsat_count(formula, assignment) + 1
    for variable, _ in clause:
        assignment[variable] = not assignment[variable]
        unsat = _unsat_count(formula, assignment)
        assignment[variable] = not assignment[variable]
        if unsat < best_unsat:
            best_unsat = unsat
            best_var = variable
    return best_var


def _unsat_count(formula: CNF, assignment: Assignment) -> int:
    return sum(0 if clause_satisfied(c, assignment) else 1 for c in formula)


VIEW_CHOOSERS: dict[str, VariableChooser] = {
    "raw": _choose_raw,
}


@dataclass
class FlatteningProbe:
    """Compare local search in raw vs transformed coordinate views.

    Each view is given the *same* random seed and the same starting
    assignment, so any difference in flips-to-solve or strain decay is
    attributable to the choice of basis.
    """

    max_flips: int = 200
    seed: int = 0

    def run(
        self,
        formula: CNF,
        n_vars: int,
        *,
        instance_id: str = "anon",
        planted_satisfiable: bool | None = None,
        views: Sequence[CoordinateView] | None = None,
    ) -> ProbeResult:
        if views is None:
            views = (raw_view(n_vars), spectral_view(formula, n_vars))
        result = ProbeResult(
            instance_id=instance_id,
            n_vars=n_vars,
            n_clauses=len(formula),
            planted_satisfiable=planted_satisfiable,
        )
        # Shared starting assignment so views diverge only by their
        # flip choices, not by initial conditions.
        start_rng = random.Random(self.seed)
        start = [start_rng.choice([False, True]) for _ in range(n_vars)]

        for view in views:
            chooser = self._chooser_for(view)
            run_rng = random.Random(self.seed + _stable_view_seed_offset(view.name))
            result.runs[view.name] = _run_view(
                formula=formula,
                start=list(start),
                view=view,
                chooser=chooser,
                rng=run_rng,
                max_flips=self.max_flips,
            )
        return result

    @staticmethod
    def _chooser_for(view: CoordinateView) -> VariableChooser:
        if view.name == "raw":
            return _choose_raw
        return _choose_spectral


def _run_view(
    *,
    formula: CNF,
    start: Assignment,
    view: CoordinateView,
    chooser: VariableChooser,
    rng: random.Random,
    max_flips: int,
) -> ProbeRunResult:
    assignment = list(start)
    initial_unsat = _unsat_count(formula, assignment)
    trajectory: list[float] = []
    decisions: list[StepRecord] = []

    for step in range(max_flips):
        unsat = _unsat_count(formula, assignment)
        strain = view.strain(formula, assignment)
        trajectory.append(strain.total)
        if unsat == 0:
            return ProbeRunResult(
                view_name=view.name,
                solved=True,
                flips=step,
                final_unsatisfied=0,
                initial_unsatisfied=initial_unsat,
                strain_trajectory=tuple(trajectory),
                decisions=tuple(decisions),
            )
        variable = chooser(formula, assignment, view, rng)
        per_dir = np.asarray(strain.per_direction)
        dominant = int(np.argmax(np.abs(per_dir))) if per_dir.size else 0
        decisions.append(
            StepRecord(
                step=step,
                flipped_variable=variable,
                unsatisfied=unsat,
                total_strain=strain.total,
                dominant_direction=dominant,
            )
        )
        assignment[variable] = not assignment[variable]

    final_unsat = _unsat_count(formula, assignment)
    trajectory.append(float(_per_variable_strain(formula, assignment, view.n_vars).sum()))
    return ProbeRunResult(
        view_name=view.name,
        solved=final_unsat == 0,
        flips=max_flips,
        final_unsatisfied=final_unsat,
        initial_unsatisfied=initial_unsat,
        strain_trajectory=tuple(trajectory),
        decisions=tuple(decisions),
    )
