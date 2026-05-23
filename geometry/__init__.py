"""Geometry experiments for SAT.

Small, deliberately-scoped probes that ask whether cheap coordinate
transforms produce measurable local flattening on toy SAT instances.

See :mod:`geometry.flattening_probe` for the main entry point and
:mod:`experiments.flattening_probe` for the driver/report.
"""

from geometry.flattening_probe import (
    ConstraintStrain,
    CoordinateView,
    FlatteningProbe,
    ProbeResult,
    raw_view,
    spectral_view,
)
from geometry.transform_litmus import (
    LITMUS_VERDICTS,
    LitmusReading,
    LitmusSummary,
    StrainLocalization,
    gini,
    herfindahl,
    litmus_for_result,
    litmus_for_view,
    localization_of,
    summarize,
    top_k_share,
)

__all__ = [
    "ConstraintStrain",
    "CoordinateView",
    "FlatteningProbe",
    "LITMUS_VERDICTS",
    "LitmusReading",
    "LitmusSummary",
    "ProbeResult",
    "StrainLocalization",
    "gini",
    "herfindahl",
    "litmus_for_result",
    "litmus_for_view",
    "localization_of",
    "raw_view",
    "spectral_view",
    "summarize",
    "top_k_share",
]
