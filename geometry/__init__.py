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
from geometry.bubble_lifecycle import (
    LIFECYCLE_LABELS,
    AddressBubble,
    BubbleReportRow,
    CollisionSeed,
    ContainmentReport,
    LifecycleTrace,
    boundary_margin,
    classify_lifecycle,
    classify_static,
    contains,
    format_report,
    inflate_bubble,
    report_row,
    seed_from_strain,
)

__all__ = [
    "AddressBubble",
    "BubbleReportRow",
    "CollisionSeed",
    "ConstraintStrain",
    "ContainmentReport",
    "CoordinateView",
    "FlatteningProbe",
    "LIFECYCLE_LABELS",
    "LITMUS_VERDICTS",
    "LifecycleTrace",
    "LitmusReading",
    "LitmusSummary",
    "ProbeResult",
    "StrainLocalization",
    "boundary_margin",
    "classify_lifecycle",
    "classify_static",
    "contains",
    "format_report",
    "gini",
    "herfindahl",
    "inflate_bubble",
    "litmus_for_result",
    "litmus_for_view",
    "localization_of",
    "raw_view",
    "report_row",
    "seed_from_strain",
    "spectral_view",
    "summarize",
    "top_k_share",
]
