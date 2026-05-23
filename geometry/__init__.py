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

__all__ = [
    "ConstraintStrain",
    "CoordinateView",
    "FlatteningProbe",
    "ProbeResult",
    "raw_view",
    "spectral_view",
]
