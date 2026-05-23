#!/usr/bin/env python3
"""Driver/report for the tangent lift probe.

Runs the analytic probe described in ``geometry/tangent_lift_probe.py``
on two grids — a wide, multi-asymptote grid and a narrow band around a
single asymptote — and prints a small comparison table. No grand
claims.

Run with: ``python experiments/tangent_lift_probe.py``
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.tangent_lift_probe import (
    ChartMetrics,
    LiftedChart,
    RawChart,
    deterministic_grid,
    lifted_chart_metrics,
    raw_chart_metrics,
)


def _wide_grid() -> dict[str, ChartMetrics]:
    x = deterministic_grid(n_samples=401, span=3.0 * math.pi)
    return {
        "raw": raw_chart_metrics(RawChart(x=x)),
        "lifted": lifted_chart_metrics(LiftedChart(x=x)),
    }


def _near_asymptote_grid() -> dict[str, ChartMetrics]:
    # A narrow band straddling x = pi/2. Same span on both charts so
    # the comparison is honest.
    import numpy as np

    centre = math.pi / 2.0
    half_width = 0.05
    x = np.linspace(centre - half_width, centre + half_width, 201)
    return {
        "raw": raw_chart_metrics(RawChart(x=x)),
        "lifted": lifted_chart_metrics(LiftedChart(x=x)),
    }


def _print_table(label: str, rows: dict[str, ChartMetrics]) -> None:
    print()
    print(label)
    print("-" * len(label))
    header = (
        f"{'chart':<16} {'n':<5} {'max_fd':>14} {'boundary':>9} "
        f"{'masked':>7} {'explode':>8} {'clip_mass':>12} "
        f"{'recon_err':>12} {'cond_strain':>13}"
    )
    print(header)
    print("-" * len(header))
    for _, m in rows.items():
        print(
            f"{m.chart:<16} {m.n_samples:<5d} "
            f"{m.max_finite_difference:>14.3e} "
            f"{m.boundary_points:>9d} {m.masked_points:>7d} "
            f"{m.explosion_count:>8d} {m.clipping_burden:>12.3e} "
            f"{m.reconstruction_error_off_boundary:>12.3e} "
            f"{m.condition_strain:>13.3e}"
        )


def _interpret(wide: dict[str, ChartMetrics]) -> None:
    raw = wide["raw"]
    lift = wide["lifted"]
    print()
    print("Interpretation")
    print("--------------")
    print(
        "- Raw chart: max finite difference between consecutive samples is "
        f"{raw.max_finite_difference:.3e}; "
        f"{raw.explosion_count} samples cross the explosion threshold."
    )
    print(
        "- Lifted chart: coordinates (sin, cos) are bounded by 1, so the same "
        f"finite-difference statistic is {lift.max_finite_difference:.3e}. "
        f"The singularity is now a typed boundary: {lift.boundary_points} of "
        f"{lift.n_samples} samples live on |cos x| <= eps and are *named* as "
        "boundary points rather than left to explode."
    )
    print(
        "- Off-boundary reconstruction tan = sin/cos is exact to "
        f"{lift.reconstruction_error_off_boundary:.3e} on the unmasked set."
    )
    print(
        "- The chart didn't make tan(x) easier to compute *at* the asymptote. "
        "It made the asymptote a locally-typed event rather than a globally-"
        "scary scalar value. Singularity-as-coordinate-degeneracy."
    )


def main() -> None:
    wide = _wide_grid()
    near = _near_asymptote_grid()
    _print_table("Wide grid: x in [-3pi, 3pi], 401 samples", wide)
    _print_table("Near-asymptote band: x in pi/2 +/- 0.05, 201 samples", near)
    _interpret(wide)


if __name__ == "__main__":
    main()
