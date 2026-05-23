#!/usr/bin/env python3
"""Tangent lift probe: is a singularity a coordinate artifact?

The user's framing: ``tan(x)`` is a notoriously awkward target if you
try to fit/evaluate it in the scalar coordinate ``x`` — it has a dense
set of vertical asymptotes at ``x = pi/2 + k*pi`` where any
finite-precision representation explodes. But the same map factors
cleanly as the ratio of two bounded coordinates ::

    tan(x) = sin(x) / cos(x)

In the ``(sin x, cos x)`` chart the function is *structurally simple*:
two smooth, bounded coordinates plus a single, locally-typed
degeneracy at ``cos x = 0``. The "singularity" becomes a boundary
condition — a place where one of the two coordinates is small —
rather than a value that explodes.

This probe is an analytic, deterministic, dependency-light measurement
of that claim on a fixed grid of ``x`` values. It is not a learning
experiment; there is no model being trained. The point is to put
numbers on what changes when you swap the chart, on the same samples.

What we measure
---------------
For a grid of ``x`` values straddling several asymptotes:

- **Raw chart**: represent the target as the scalar ``tan(x)`` and use
  the scalar ``x`` as the input coordinate. We measure
  ``max_finite_difference`` (a finite-difference proxy for gradient
  magnitude on the target), ``explosion_count`` (samples whose
  ``|tan x|`` exceeds a clip threshold), ``clipping_burden`` (how much
  total mass we'd lose by clipping), and ``condition_strain`` (mean
  ratio of consecutive raw outputs, capped). No samples are "masked";
  the raw chart has no notion of a boundary, only of large numbers.

- **Lifted chart**: represent the same samples as two coordinates
  ``(sin x, cos x)`` and reconstruct ``tan x = sin x / cos x`` only
  where ``|cos x| > eps``. The boundary becomes a *typed* event —
  ``boundary_points`` counts where ``|cos x| <= eps``,
  ``masked_points`` is exactly the count of samples we declined to
  reconstruct — and reconstruction is measured only on the unmasked
  set (``reconstruction_error_off_boundary``). We also report the
  lifted chart's own internal strain: ``max_finite_difference`` of
  ``(sin, cos)``, which is bounded by 1.

This is intentionally analytic. The headline claim is geometric, not
statistical: in the raw chart the singularity is *delocalized*
(neighbours of an asymptote explode), in the lifted chart it is
*localized* (a sample is either on the boundary set or smoothly
reconstructible).

No grand claims. This is one example. The framing it supports — "a
singularity is a request for a better chart" — is the thing we want a
small honest number for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def deterministic_grid(
    *,
    n_samples: int = 401,
    span: float = 3.0 * math.pi,
) -> np.ndarray:
    """Return a fixed, evenly-spaced grid on ``[-span, span]``.

    Determinism is the whole point — every metric we report here is a
    function of this exact grid. ``n_samples`` is odd by default so the
    grid is symmetric about 0.
    """

    if n_samples < 3:
        raise ValueError("need at least 3 samples for a finite difference")
    return np.linspace(-span, span, n_samples)


# ---------------------------------------------------------------------------
# Chart definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawChart:
    """The raw scalar chart: input ``x``, target ``tan(x)``.

    We do not silently clip. We *report* how many samples would have to
    be clipped at a given threshold, because that count is the chart's
    own admission that it has nowhere to put those samples.
    """

    x: np.ndarray
    explosion_threshold: float = 1.0e6

    @property
    def target(self) -> np.ndarray:
        return np.tan(self.x)

    @property
    def explosion_mask(self) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            t = np.abs(self.target)
        return ~np.isfinite(t) | (t > self.explosion_threshold)


@dataclass(frozen=True)
class LiftedChart:
    """The lifted chart: input ``x``, coordinates ``(sin x, cos x)``.

    Reconstruction of ``tan x`` is only attempted where
    ``|cos x| > boundary_eps``. Everywhere else, the chart declares the
    sample a boundary point and declines to invent a value. This is the
    structural difference we care about.
    """

    x: np.ndarray
    boundary_eps: float = 1.0e-3

    @property
    def sin(self) -> np.ndarray:
        return np.sin(self.x)

    @property
    def cos(self) -> np.ndarray:
        return np.cos(self.x)

    @property
    def boundary_mask(self) -> np.ndarray:
        return np.abs(self.cos) <= self.boundary_eps

    def reconstruct_tan(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(values, valid_mask)`` for the reconstruction.

        ``values`` is filled with ``nan`` on boundary samples; callers
        should restrict any error metric to the ``valid_mask`` set.
        """

        cos = self.cos
        valid = ~self.boundary_mask
        values = np.full_like(self.x, np.nan, dtype=float)
        values[valid] = self.sin[valid] / cos[valid]
        return values, valid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartMetrics:
    """One row of the comparison table. All scalars."""

    chart: str
    n_samples: int
    max_finite_difference: float
    boundary_points: int
    masked_points: int
    explosion_count: int
    clipping_burden: float
    reconstruction_error_off_boundary: float
    condition_strain: float


def _finite_diff_max(values: np.ndarray) -> float:
    """Max ``|f[i+1] - f[i]|`` over finite consecutive pairs.

    The raw chart will produce arbitrarily large finite differences as
    samples bracket an asymptote; the lifted chart's coordinates are
    bounded by 1, so this number is bounded by 2.
    """

    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("inf")
    return float(np.max(np.abs(diffs)))


def _condition_strain(values: np.ndarray, cap: float = 1.0e6) -> float:
    """Mean of capped consecutive ratios ``|f[i+1]| / max(|f[i]|, 1)``.

    A finite, deterministic proxy for "how much do neighbouring outputs
    differ in scale". Capped to keep the number reportable when the raw
    chart actually explodes.
    """

    finite = np.isfinite(values)
    if finite.sum() < 2:
        return float("inf")
    a = np.abs(values[finite])
    if a.size < 2:
        return float("inf")
    ratios = a[1:] / np.maximum(a[:-1], 1.0)
    ratios = np.minimum(ratios, cap)
    return float(np.mean(ratios))


def raw_chart_metrics(chart: RawChart) -> ChartMetrics:
    target = chart.target
    explosion = chart.explosion_mask
    finite_target = np.where(np.isfinite(target), target, 0.0)
    clipped_mass = float(
        np.sum(
            np.abs(np.where(explosion, finite_target, 0.0))
        )
    )
    return ChartMetrics(
        chart="raw_scalar_x",
        n_samples=int(chart.x.size),
        max_finite_difference=_finite_diff_max(target),
        boundary_points=0,
        masked_points=0,
        explosion_count=int(np.sum(explosion)),
        clipping_burden=clipped_mass,
        reconstruction_error_off_boundary=0.0,
        condition_strain=_condition_strain(target),
    )


def lifted_chart_metrics(
    lifted: LiftedChart,
    *,
    reference_x: np.ndarray | None = None,
) -> ChartMetrics:
    """Metrics for the lifted ``(sin, cos)`` chart on the same grid.

    Reconstruction error is computed against the ground-truth
    ``tan(reference_x)`` on the unmasked set; ``reference_x`` defaults
    to ``lifted.x`` (which is the honest comparison: same samples,
    different chart).
    """

    if reference_x is None:
        reference_x = lifted.x
    values, valid = lifted.reconstruct_tan()
    # Coordinate-side strain: the (sin, cos) curve is what we'd actually
    # learn/store, and it is bounded.
    sin_diff = _finite_diff_max(lifted.sin)
    cos_diff = _finite_diff_max(lifted.cos)
    coord_max_fd = max(sin_diff, cos_diff)
    boundary = lifted.boundary_mask
    if valid.any():
        ref = np.tan(reference_x[valid])
        # Reconstruction error is measured where *both* ref and value
        # are finite — far from the asymptote this is essentially all
        # of the unmasked set.
        finite_pair = np.isfinite(values[valid]) & np.isfinite(ref)
        if finite_pair.any():
            recon_err = float(
                np.max(np.abs(values[valid][finite_pair] - ref[finite_pair]))
            )
        else:
            recon_err = float("inf")
    else:
        recon_err = float("inf")
    return ChartMetrics(
        chart="lifted_sin_cos",
        n_samples=int(lifted.x.size),
        max_finite_difference=coord_max_fd,
        boundary_points=int(np.sum(boundary)),
        masked_points=int(np.sum(boundary)),
        explosion_count=0,
        clipping_burden=0.0,
        reconstruction_error_off_boundary=recon_err,
        condition_strain=_condition_strain(lifted.sin)
        + _condition_strain(lifted.cos),
    )


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TangentLiftProbe:
    """Run both charts on the same deterministic grid and return rows."""

    n_samples: int = 401
    span: float = 3.0 * math.pi
    boundary_eps: float = 1.0e-3
    explosion_threshold: float = 1.0e6

    def run(self) -> dict[str, ChartMetrics]:
        x = deterministic_grid(n_samples=self.n_samples, span=self.span)
        raw = RawChart(x=x, explosion_threshold=self.explosion_threshold)
        lifted = LiftedChart(x=x, boundary_eps=self.boundary_eps)
        return {
            "raw": raw_chart_metrics(raw),
            "lifted": lifted_chart_metrics(lifted),
        }


__all__ = [
    "ChartMetrics",
    "LiftedChart",
    "RawChart",
    "TangentLiftProbe",
    "deterministic_grid",
    "lifted_chart_metrics",
    "raw_chart_metrics",
]
