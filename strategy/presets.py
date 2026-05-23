"""Preset strategy compositions used by tests and the comparison driver.

Each preset is a small operator list, in order. Reading these is the
recommended way to see what the decomposition *looks like* in
practice. New presets should be added here so the operator graph stays
visible alongside the operator definitions.
"""

from __future__ import annotations

from geometry.flattening_probe import CoordinateView, spectral_view
from sat_furnace import CNF
from strategy.operators import (
    StrategyComposer,
    bubble_pressure_gate,
    coordinate_ranker,
    fitted_coordinate_ranker,
    plateau_detector,
    random_walk_kick,
    raw_strain_ranker,
    riordan_bubble_fitter,
    unsat_clause_focus,
)


def raw_composer(walk_probability: float = 0.10) -> StrategyComposer:
    """Reproduces ``flattening_probe._choose_raw`` as an operator list.

    Order:
      1. focus an unsat clause,
      2. with small probability, random-walk-kick,
      3. otherwise greedy raw strain ranker.
    """
    return StrategyComposer(
        operators=(
            unsat_clause_focus,
            random_walk_kick(walk_probability),
            raw_strain_ranker,
        )
    )


def transformed_composer(
    view: CoordinateView,
    walk_probability: float = 0.10,
) -> StrategyComposer:
    """Reproduces ``flattening_probe._choose_spectral`` as an operator list.

    The coordinate ranker runs ahead of the raw ranker so transform-
    driven proposals win on steps where they are confident. The raw
    ranker is kept as a fallback for steps where the transform's
    projected strain is zero.
    """
    return StrategyComposer(
        operators=(
            unsat_clause_focus,
            random_walk_kick(walk_probability),
            coordinate_ranker(view),
            raw_strain_ranker,
        )
    )


def gated_transformed_composer(
    view: CoordinateView,
    walk_probability: float = 0.10,
) -> StrategyComposer:
    """The composed intervention: bubble-gated transform with raw fallback.

    Order:
      1. plateau detector (publishes ``plateau`` signal),
      2. bubble-pressure gate (may veto the transformed ranker),
      3. focus an unsat clause,
      4. random walk kick (probability),
      5. coordinate (transformed) ranker — yields on veto,
      6. raw strain ranker — picks up when the transform yielded.
    """
    return StrategyComposer(
        operators=(
            plateau_detector(),
            bubble_pressure_gate(),
            unsat_clause_focus,
            random_walk_kick(walk_probability),
            coordinate_ranker(view),
            raw_strain_ranker,
        )
    )


def fitted_composer(walk_probability: float = 0.10) -> StrategyComposer:
    """Riordan bubble-fit composer: choose a transform by bubble stability.

    Order:
      1. plateau detector — publishes ``plateau``,
      2. riordan bubble fitter — picks the most stable view among
         identity / pascal / signed_pascal / sierpinski candidates by
         bubble pressure / containment criteria (no SAT outcome leakage),
         publishes ``fitted_view`` and may set ``veto_transformed``,
      3. focus an unsat clause,
      4. random walk kick (probability),
      5. fitted coordinate ranker — runs on the chosen view, yields on
         veto,
      6. raw strain ranker — picks up when the fitted ranker yielded.

    There is no hardcoded if-tree over transform families: the routing
    is entirely a field-signal flow through the operator list.
    """
    return StrategyComposer(
        operators=(
            plateau_detector(),
            riordan_bubble_fitter(),
            unsat_clause_focus,
            random_walk_kick(walk_probability),
            fitted_coordinate_ranker,
            raw_strain_ranker,
        )
    )


def spectral_view_for(formula: CNF, n_vars: int) -> CoordinateView:
    """Convenience: the default spectral view used by the flattening probe."""
    return spectral_view(formula, n_vars)
