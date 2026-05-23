"""Composable SAT strategy operators.

Decomposes the local-search SAT strategies in :mod:`geometry.flattening_probe`
(``_choose_raw`` / ``_choose_spectral``) into small operators that read a
typed :class:`SearchState` and either produce a :class:`Proposal` or yield
to the next operator. The :class:`StrategyComposer` is a first-match list
over those operators; branching lives in the ordering and in field signals
each operator reads, not in a monolithic if/elif tree.

This package is the first *behavior-altering* step after the merged
geometry/bubble metrics stack: it does not replace the furnace, it does
not hardcode a bubble solver. Bubble pressure is exposed as one
composable gate operator alongside the rest of the ecology.
"""

from strategy.operators import (
    Proposal,
    SearchState,
    StrategyComposer,
    StrategyOperator,
    bubble_pressure_gate,
    coordinate_ranker,
    plateau_detector,
    random_walk_kick,
    raw_strain_ranker,
    unsat_clause_focus,
)
from strategy.presets import (
    gated_transformed_composer,
    raw_composer,
    spectral_view_for,
    transformed_composer,
)
from strategy.run import composed_local_search, RunRecord, RunReport

__all__ = (
    "gated_transformed_composer",
    "raw_composer",
    "spectral_view_for",
    "transformed_composer",
    "Proposal",
    "SearchState",
    "StrategyComposer",
    "StrategyOperator",
    "bubble_pressure_gate",
    "composed_local_search",
    "coordinate_ranker",
    "plateau_detector",
    "random_walk_kick",
    "raw_strain_ranker",
    "unsat_clause_focus",
    "RunRecord",
    "RunReport",
)
