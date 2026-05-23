# Flattening Probe

## Framing

The conjecture being tested (informally): some SAT hardness may be
*representation geometry*. A solver in the raw symbolic basis sees
constraints as fighting each other; in a better-chosen coordinate
basis they might co-vary and stop pulling in opposite directions.
Local search inside that better basis might descend strain faster
than the same search in the raw basis.

We are not claiming this dents NP-hardness. We are asking a much
smaller question: on toy instances, does picking which variable to
flip *in a transformed coordinate basis* reduce conflict strain or
flips-to-solve compared to picking in the raw basis, holding the
clause-check budget per step equal?

## What the probe does

- Generates small instances via `sat_furnace.generate_formula`:
  planted-satisfiable 2-SAT and 3-SAT, plus a couple of
  structural-unsat formulas as a sanity floor.
- Builds two `CoordinateView`s of the variables:
  - **raw**: identity basis. Each direction is a single variable.
  - **spectral(k)**: top-`k` left-singular vectors of the signed
    variable-clause incidence matrix. These are the directions along
    which variables co-vary most strongly across clauses — a cheap
    candidate for "coordinates where the constraints stop fighting
    each other".
- Runs the same WalkSAT-flavoured loop in each view from the *same*
  starting assignment. Both views use a 10% random-walk floor for
  plateau escape; the only thing they disagree on is *which variable
  to flip* on each greedy step.
  - Raw: pick an unsat clause, flip the variable that minimizes the
    resulting unsat count (textbook WalkSAT).
  - Spectral: project the per-variable strain vector onto the
    rotated basis, identify the direction carrying the most strain,
    flip the variable with the largest loading on that direction
    that participates in some currently-unsat clause.
- Reports flips-to-solve, residual unsat after a budget, and the
  total-strain trajectory per view.

## Why this is honest

- The spectral view does the same number of clause checks per flip.
  It is not a smarter algorithm — it is the same algorithm asking a
  different question about *which variable* to act on.
- The probe runs on toy sizes (≤12 variables). The interesting
  signal would be "the spectral view picks better flips when the
  constraint graph has structure" — which is exactly the regime
  where representation geometry could plausibly matter.
- The structural-unsat instances are a guardrail: a transform that
  starts "solving" UNSAT formulas is a transform that has broken
  something.

## Early reading

On the seeded suite shipped with this PR:

- Easy 2-SAT: both views solve immediately; raw is faster simply
  because the spectral basis is overkill on a near-trivial instance.
- Mid-density 3-SAT (12 vars, 42 clauses, near the SAT threshold):
  spectral solves all 3 seeds; raw plateaus on one of them. On the
  shared-seed comparisons spectral wins 1, ties 7, loses 0.
- Structural unsat: both views correctly fail to "solve" (and
  trajectories don't decay — the strain has nowhere to go).

That's a faint but non-empty directional signal on the regime where
constraints are actually fighting each other. It is suggestive, not
conclusive. The next experiments would be (a) larger instance
sweeps, (b) other transforms (Laplacian eigenmaps of the
factor-graph, random projections as a null), and (c) measuring
strain *decay rate* not just final value.

## Where things live

- `geometry/flattening_probe.py`: types and search loop.
- `experiments/flattening_probe.py`: driver + table.
- `tests/test_flattening_probe.py`: deterministic tests.
