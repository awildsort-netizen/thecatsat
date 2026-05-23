#!/usr/bin/env python3
"""Transform litmus: does a SAT coordinate change localize conflict?

A companion diagnostic to :mod:`geometry.flattening_probe` and
:mod:`geometry.riordan_probe`. The motivation comes from the tangent
lift probe in :mod:`geometry.tangent_lift_probe`: there, a coordinate
change turns the global scalar blow-up of ``tan(x)`` into a typed
*local* boundary condition. The litmus asks the SAT analogue of that
question, on the runs the flattening/Riordan probes already produce:

    Does a transform turn nonlocal conflict — strain spread thinly
    across many variables, long flat plateaus where flips just shuffle
    — into *local* typed structure: strain concentrated on a small
    named set of variables or clauses?

It is the SAT version of the tangent test, and it is intentionally
narrow. It does **not** ask whether the transform solves SAT. It asks
whether the transform makes the residual pathology more *addressable*:
fewer variables carrying most of the unsat pressure, shorter plateaus,
and (separately) whether that correlates with the transform actually
helping the solver finish.

Vocabulary
----------
- ``StrainLocalization`` — top-k share, Herfindahl index, and Gini
  coefficient over the per-variable strain at run end. Higher top-k
  share / Herfindahl, and higher Gini, mean strain has collapsed onto
  a small set of variables. Lower means it is spread out.
- ``LitmusVerdict`` — small label set describing how the *transform's*
  final state compares to the baseline (raw): ``resolved_to_boundary``,
  ``localized_but_unstable``, ``moved_singularity``,
  ``amplified_pathology``, ``both_solved``, ``no_change``.
- ``LitmusReading`` — the full record per (instance, view) used by the
  driver to print the table and correlation summary.

The litmus is deterministic given the probe run inputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geometry.flattening_probe import ProbeResult, ProbeRunResult, _per_variable_strain
from sat_furnace import CNF


# --------------------------------------------------------------------------- #
# Localization metrics                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrainLocalization:
    """How concentrated final per-variable strain is across variables.

    All three statistics are computed on the *final* per-variable
    strain vector — the residual pressure that survived the run. They
    are zero when the run solved (no residual). Otherwise:

    - ``top_k_share`` — fraction of total strain carried by the top
      ``k`` variables (default ``k=3``). Closer to 1 means strain has
      collapsed onto a small named set.
    - ``herfindahl`` — sum of squared shares. Equals ``1`` when a
      single variable carries everything; equals ``1/n`` for a uniform
      spread. Higher = more concentrated.
    - ``gini`` — standard Gini coefficient on the strain shares. ``0``
      uniform, near ``1`` heavily concentrated.
    - ``support`` — number of variables with strictly positive strain.
    """

    top_k_share: float
    herfindahl: float
    gini: float
    support: int
    k: int


def _shares(per_variable: Sequence[float]) -> np.ndarray:
    arr = np.asarray(per_variable, dtype=float)
    total = float(arr.sum())
    if total <= 0.0:
        return np.zeros_like(arr)
    return arr / total


def top_k_share(per_variable: Sequence[float], k: int = 3) -> float:
    shares = _shares(per_variable)
    if shares.size == 0:
        return 0.0
    k = max(1, min(k, shares.size))
    return float(np.sort(shares)[-k:].sum())


def herfindahl(per_variable: Sequence[float]) -> float:
    shares = _shares(per_variable)
    return float(np.sum(shares * shares))


def gini(per_variable: Sequence[float]) -> float:
    arr = np.asarray(per_variable, dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0 or arr.size == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = arr.size
    # Mean absolute difference formulation.
    cum = np.cumsum(sorted_arr)
    # Gini = (2 * sum_i i * x_i) / (n * sum) - (n + 1) / n
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * sorted_arr)) / (n * cum[-1]) - (n + 1) / n)


def support(per_variable: Sequence[float]) -> int:
    arr = np.asarray(per_variable, dtype=float)
    return int(np.sum(arr > 0.0))


def localization_of(per_variable: Sequence[float], *, k: int = 3) -> StrainLocalization:
    return StrainLocalization(
        top_k_share=top_k_share(per_variable, k=k),
        herfindahl=herfindahl(per_variable),
        gini=gini(per_variable),
        support=support(per_variable),
        k=k,
    )


def _replay_per_variable_strain(
    formula: CNF, start: list[bool], run: ProbeRunResult, n_vars: int
) -> np.ndarray:
    """Apply ``run.decisions`` to ``start`` and return final per-variable strain."""
    assignment = list(start)
    for record in run.decisions:
        v = record.flipped_variable
        assignment[v] = not assignment[v]
    return _per_variable_strain(formula, assignment, n_vars)


# --------------------------------------------------------------------------- #
# Verdicts                                                                    #
# --------------------------------------------------------------------------- #


# Verdict labels, deliberately small and SAT-named (not the tangent
# probe's names, since the analogy isn't tight enough to pretend the
# vocabularies are interchangeable).
RESOLVED_TO_BOUNDARY = "resolved_to_boundary"
LOCALIZED_BUT_UNSTABLE = "localized_but_unstable"
MOVED_SINGULARITY = "moved_singularity"
AMPLIFIED_PATHOLOGY = "amplified_pathology"
BOTH_SOLVED = "both_solved"
NO_CHANGE = "no_change"

LITMUS_VERDICTS = (
    RESOLVED_TO_BOUNDARY,
    LOCALIZED_BUT_UNSTABLE,
    MOVED_SINGULARITY,
    AMPLIFIED_PATHOLOGY,
    BOTH_SOLVED,
    NO_CHANGE,
)


@dataclass(frozen=True)
class LitmusReading:
    """Per (instance, view) litmus output.

    Compares the transform's final state to the raw baseline. All
    fields are deterministic given the probe runs.
    """

    instance_id: str
    view_name: str
    baseline_solved: bool
    view_solved: bool
    baseline_final_unsat: int
    view_final_unsat: int
    baseline_localization: StrainLocalization
    view_localization: StrainLocalization
    verdict: str
    # Strain spread delta: positive => transform spread strain MORE thinly.
    # (Useful as a sanity check when interpreting "moved_singularity".)
    support_delta: int
    top_k_share_delta: float


def classify(
    baseline: ProbeRunResult,
    view: ProbeRunResult,
    baseline_loc: StrainLocalization,
    view_loc: StrainLocalization,
    *,
    localization_jump: float = 0.10,
    plateau_threshold: int = 5,
) -> str:
    """Return one of the verdicts above.

    Rules, in order:

    1. If the view solved and the baseline did not → ``resolved_to_boundary``
       *only if* the view's pre-solve strain trajectory shows the
       residual collapsed onto few variables (top-k share crossed the
       ``localization_jump`` threshold above the baseline's residual
       share, OR the baseline had a long plateau the view broke). When
       it's just "the view solved" without that structural signal we
       still call it ``resolved_to_boundary`` — solving *is* the
       cleanest possible localization. The threshold catches the
       case where the baseline already solved.

    2. If both solved → ``both_solved`` (no residual to localize on
       either side; speed differences are :mod:`riordan_probe`'s
       motion labels' job).

    3. If neither solved and the view ended with strictly more total
       residual unsat than the baseline → ``amplified_pathology``.

    4. If neither solved and the view's residual support is
       meaningfully smaller (fewer variables carry the strain) or its
       top-k share is meaningfully larger → ``localized_but_unstable``.

    5. If neither solved and the residual unsat is similar but the
       support has *shifted* (different variables hold the strain),
       call it ``moved_singularity``.

    6. Otherwise → ``no_change``.
    """
    if view.solved and not baseline.solved:
        return RESOLVED_TO_BOUNDARY
    if view.solved and baseline.solved:
        return BOTH_SOLVED
    # Neither solved (or only the baseline solved).
    if not view.solved and baseline.solved:
        # The transform regressed an already-solving baseline.
        return AMPLIFIED_PATHOLOGY
    # Both failed to solve.
    if view.final_unsatisfied > baseline.final_unsatisfied:
        return AMPLIFIED_PATHOLOGY
    # Same or fewer residual unsat clauses.
    support_shrunk = view_loc.support + 1 < baseline_loc.support
    top_k_grew = view_loc.top_k_share - baseline_loc.top_k_share >= localization_jump
    if support_shrunk or top_k_grew:
        return LOCALIZED_BUT_UNSTABLE
    if view.final_unsatisfied == baseline.final_unsatisfied:
        # Same residual magnitude — did the residual move?
        if view_loc.support != baseline_loc.support or abs(
            view_loc.top_k_share - baseline_loc.top_k_share
        ) >= localization_jump:
            return MOVED_SINGULARITY
    return NO_CHANGE


def litmus_for_view(
    formula: CNF,
    n_vars: int,
    seed: int,
    instance_id: str,
    baseline: ProbeRunResult,
    view: ProbeRunResult,
    *,
    k: int = 3,
) -> LitmusReading:
    """Compute the litmus reading for a single transform view.

    Uses the same starting-assignment derivation the probe uses (a
    ``random.Random(seed)`` consumed for ``n_vars`` booleans), so the
    per-variable strain we replay is exactly what the probe saw.
    """
    start_rng = random.Random(seed)
    start = [start_rng.choice([False, True]) for _ in range(n_vars)]

    if baseline.solved:
        baseline_pv = np.zeros(n_vars, dtype=float)
    else:
        baseline_pv = _replay_per_variable_strain(formula, list(start), baseline, n_vars)
    if view.solved:
        view_pv = np.zeros(n_vars, dtype=float)
    else:
        view_pv = _replay_per_variable_strain(formula, list(start), view, n_vars)

    baseline_loc = localization_of(baseline_pv, k=k)
    view_loc = localization_of(view_pv, k=k)
    verdict = classify(baseline, view, baseline_loc, view_loc)
    return LitmusReading(
        instance_id=instance_id,
        view_name=view.view_name,
        baseline_solved=baseline.solved,
        view_solved=view.solved,
        baseline_final_unsat=baseline.final_unsatisfied,
        view_final_unsat=view.final_unsatisfied,
        baseline_localization=baseline_loc,
        view_localization=view_loc,
        verdict=verdict,
        support_delta=view_loc.support - baseline_loc.support,
        top_k_share_delta=view_loc.top_k_share - baseline_loc.top_k_share,
    )


def litmus_for_result(
    formula: CNF,
    n_vars: int,
    seed: int,
    result: ProbeResult,
    *,
    baseline: str = "raw",
    k: int = 3,
) -> list[LitmusReading]:
    """Litmus readings for every non-baseline view in a probe result."""
    if baseline not in result.runs:
        return []
    base_run = result.runs[baseline]
    readings: list[LitmusReading] = []
    for view_name, run in result.runs.items():
        if view_name == baseline:
            continue
        readings.append(
            litmus_for_view(
                formula=formula,
                n_vars=n_vars,
                seed=seed,
                instance_id=result.instance_id,
                baseline=base_run,
                view=run,
                k=k,
            )
        )
    return readings


# --------------------------------------------------------------------------- #
# Correlation between verdict and solve improvement                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LitmusSummary:
    """Aggregate over many litmus readings.

    ``verdict_to_solve_rate`` answers: among (instance, view) pairs
    landing on a given verdict, in what fraction did the *view* solve?
    That is the SAT-equivalent of "did localization correlate with
    actually finishing".

    ``verdict_to_improvement_rate`` is the fraction of pairs at that
    verdict where the view strictly reduced ``final_unsatisfied`` vs.
    the baseline (the strict-improvement version of the same question).

    ``verdict_counts`` is the raw count per verdict.
    """

    verdict_counts: dict[str, int]
    verdict_to_solve_rate: dict[str, float]
    verdict_to_improvement_rate: dict[str, float]
    n: int


def summarize(readings: Sequence[LitmusReading]) -> LitmusSummary:
    counts: dict[str, int] = {v: 0 for v in LITMUS_VERDICTS}
    solved: dict[str, int] = {v: 0 for v in LITMUS_VERDICTS}
    improved: dict[str, int] = {v: 0 for v in LITMUS_VERDICTS}
    for r in readings:
        counts[r.verdict] += 1
        if r.view_solved:
            solved[r.verdict] += 1
        if r.view_final_unsat < r.baseline_final_unsat:
            improved[r.verdict] += 1
    solve_rate = {
        v: (solved[v] / counts[v]) if counts[v] else 0.0 for v in LITMUS_VERDICTS
    }
    improve_rate = {
        v: (improved[v] / counts[v]) if counts[v] else 0.0 for v in LITMUS_VERDICTS
    }
    return LitmusSummary(
        verdict_counts=counts,
        verdict_to_solve_rate=solve_rate,
        verdict_to_improvement_rate=improve_rate,
        n=len(readings),
    )
