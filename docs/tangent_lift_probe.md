# Tangent lift probe — singularity as coordinate artifact

A small, analytic, deterministic probe that asks a single question:
when ``tan(x)`` is awkward to fit/evaluate in the scalar coordinate
``x``, *where does the awkwardness live*? Is it in the target itself,
or in our choice of chart?

## Framing

``tan(x)`` has a dense set of vertical asymptotes at
``x = pi/2 + k*pi``. In the scalar chart ``x -> tan(x)`` those
asymptotes are singularities: the target value runs off to infinity,
finite differences explode, any finite-precision representation has
to either clip or accept catastrophic gradients near the asymptote.

But ``tan(x)`` *factors*:

```
tan(x) = sin(x) / cos(x)
```

Both ``sin(x)`` and ``cos(x)`` are everywhere smooth and bounded by
1. The vertical asymptote isn't a feature of the function — it's a
feature of the *ratio*. Lift the same map to the chart
``x -> (sin(x), cos(x))`` and the singularity becomes a single,
locally-typed event: a sample is on the boundary set when
``|cos(x)|`` is small, and the chart names that condition rather than
hiding it inside a runaway scalar.

That's the framing this probe puts numbers on. **A singularity is
sometimes a request for a better chart.**

## What the probe measures

On a fixed, deterministic grid of ``x`` values, the probe computes
two charts on the same samples and reports one row per chart.

Columns:

- ``max_finite_difference`` — max ``|f[i+1] - f[i]|`` between
  consecutive samples. For the lifted chart this is computed on
  ``(sin, cos)`` and is bounded by 2; for the raw chart this is
  computed on ``tan`` and is bounded only by the floating-point
  range.
- ``boundary_points`` — number of samples where ``|cos x| <= eps``.
  Defined only on the lifted chart; the raw chart has no boundary
  concept.
- ``masked_points`` — number of samples for which the lifted chart
  declines to reconstruct ``tan``. Equal to ``boundary_points``: the
  point of the lift is that the boundary is *named*, not papered over.
- ``explosion_count`` — number of raw samples whose ``|tan x|``
  exceeds a clip threshold. Defined only on the raw chart.
- ``clipping_burden`` — total absolute mass the raw chart would lose
  if those samples were clipped. The lifted chart's burden is 0
  because boundary samples are masked, not invented.
- ``reconstruction_error_off_boundary`` — max
  ``|sin x / cos x - tan x|`` over unmasked lifted samples. Should be
  ~floating-point epsilon: off boundary the lifted chart is the raw
  target, exactly.
- ``condition_strain`` — a finite proxy for how much consecutive
  outputs differ in scale. Capped so the raw chart's number is
  reportable even when ``tan`` actually blows up.

## Early reading (shipped with this PR)

Run:

```
python3 experiments/tangent_lift_probe.py
```

Wide grid, ``x in [-3pi, 3pi]``, 401 samples:

| chart            | max_fd     | boundary | masked | explode | clip_mass  | recon_err  |
|------------------|-----------:|---------:|-------:|--------:|-----------:|-----------:|
| raw_scalar_x     | 5.4e+15    | 0        | 0      | 2       | 1.1e+16    | 0.0        |
| lifted_sin_cos   | 4.7e-02    | 2        | 2      | 0       | 0.0        | 7.1e-15    |

Near-asymptote band, ``x in pi/2 +/- 0.05``, 201 samples:

| chart            | max_fd     | boundary | masked | explode | clip_mass  | recon_err  |
|------------------|-----------:|---------:|-------:|--------:|-----------:|-----------:|
| raw_scalar_x     | 1.6e+16    | 0        | 0      | 1       | 1.6e+16    | 0.0        |
| lifted_sin_cos   | 5.0e-04    | 5        | 5      | 0       | 0.0        | 2.8e-14    |

What this shows, and what it doesn't:

- **It does show:** the same samples, under the lifted chart, have
  17+ orders of magnitude less consecutive-sample strain; the
  asymptote is a typed boundary of size 2 (wide grid) or 5 (band)
  rather than a sea of exploding scalars; off-boundary reconstruction
  is exact to floating point.
- **It does not show:** that the lifted chart "solves" the
  singularity. ``tan`` is still undefined at ``cos = 0``. The point is
  that the chart now *labels* that fact instead of being silently
  destroyed by it.

## What this is and is not

This is **one example**, not a theorem. It is the cleanest possible
instance of "singularity as coordinate artifact": a function whose
singularities are exactly the loci where one of its lifted
coordinates vanishes. Many real singularities are not of this form,
and many representation problems aren't well posed as coordinate
choice.

But it is a clean benchmark for the framing:

- **Memorization = many local tangent patches.** A model that
  approximates ``tan(x)`` by stitching together piecewise local
  approximations near non-asymptote regions has no story for the
  asymptote, and no way to recognize that the patches are all
  fragments of the same circular structure.
- **Understanding = discovering the circle.** Once you know ``tan``
  factors through ``(sin, cos)``, the asymptote stops being a
  failure mode and starts being a feature you can detect and route
  around — a place where the chart says "I do not extend here," not
  a place where it lies.
- **Hallucination conceals absence; regeneration remembers absence.**
  The raw chart, asked for a value at an asymptote, either clips to
  some large finite number (concealing the singularity) or returns
  infinity (concealing structure). The lifted chart, asked the same
  question, returns "this sample is on the boundary set." That is
  the difference between hallucinating a value and refusing to invent
  one.

## Honest caveats

- The metrics are analytic and deterministic; there is no model and
  no training. We are comparing two ways of *representing* the same
  ``x`` values, not two ways of learning ``tan``.
- The lifted chart's "win" on ``max_finite_difference`` is partly
  tautological — sin and cos are bounded by construction. The
  non-tautological parts are: ``boundary_points`` are named (the
  chart admits a typed degeneracy), and off-boundary reconstruction
  is exact (the chart has not lost information about the original
  target).
- This generalizes to other singularities only when there is an
  embedding into a higher-dimensional space where the singular set
  becomes a coordinate degeneracy. That's a substantial assumption
  about the problem, not a free pass.

## See also

- ``geometry/tangent_lift_probe.py`` — the module.
- ``experiments/tangent_lift_probe.py`` — the driver / report.
- ``tests/test_tangent_lift_probe.py`` — determinism, boundary
  detection, reconstruction, and strain-comparison tests.
- ``docs/flattening_probe.md`` — the sibling probe on SAT (different
  domain, same underlying question: when is geometry the cure?).
