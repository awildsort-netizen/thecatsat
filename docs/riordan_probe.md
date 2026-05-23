# Riordan Probe

## Framing

A companion experiment to the [flattening probe](flattening_probe.md). The
flattening probe asked whether a *spectral* (rotated, orthonormal)
basis for the variables makes local search through SAT instances
descend strain better than the raw symbolic basis. The Riordan probe
asks a related but different question:

> Spectral views preserve angles. They are coordinate rotations that
> ignore how the coordinates were generated. **Riordan transforms** —
> the Pascal matrix and its inverse, plus Pascal-mod-2 (the Sierpinski
> mask) — preserve combinatorial *recurrence*: each new coordinate is
> built from the previous ones by an explicit ancestry rule. The
> hypothesis is that on toy SAT instances, a recurrence-preserving
> remap of per-variable strain might expose repair directions that
> rotations don't.

The philosophical framing from the parent thread:

- **Singularity** = a missing axis trying to be born.
- **Riordan transform** = a coordinate remap that preserves
  combinatorial ancestry, unlike arbitrary rotations.
- The experiment asks whether recurrence-preserving views can make
  local repair more navigable on instances where constraints are
  actually fighting each other.

This is not a P=NP claim. It is not a claim about asymptotic
hardness. It is a question about *which variable to flip* under a
fixed clause-check budget per step.

## What a Riordan / Pascal transform is, briefly

A Riordan array is a lower-triangular matrix that arises naturally as
a change-of-basis on generating functions. The simplest member is
Pascal's matrix `P` with `P[i, j] = C(i, j)` for `j ≤ i`, built by the
recurrence

    P[i, j] = P[i-1, j-1] + P[i-1, j]

so the recurrence *is* the structure being preserved. Its
multiplicative inverse is the signed Pascal matrix
`S[i, j] = (-1)^(i-j) C(i, j)`. Together `(P, S)` form an involution:
`P @ S = I`. Pascal mod 2 — the same matrix with entries reduced
modulo 2 — draws Sierpinski's triangle and gives a "structure without
magnitude" variant useful as a control.

We row-normalize each basis to unit L2 length. Without that
normalization the magnitudes of `C(i, j)` would explode for `n` above
~20; with it, projections of per-variable strain are scale-bounded
across instance sizes. (The test suite asserts this directly.)

## What the probe does

- Reuses the seeded SAT suite from the flattening probe: planted
  2-SAT, mid-density planted 3-SAT near the SAT threshold, and a
  couple of structural-unsat instances as a guardrail.
- Builds five `CoordinateView`s:
  - `raw`: identity basis (control).
  - `spectral(k)`: top-`k` left-singular vectors of the signed
    variable-clause incidence matrix (the flattening-probe view).
  - `pascal`: row-normalized lower-triangular Pascal matrix.
  - `signed_pascal`: row-normalized signed Pascal (the Riordan
    inverse partner).
  - `sierpinski`: row-normalized Pascal mod 2.
- Runs the same WalkSAT-flavoured loop in each view from the *same*
  starting assignment. The only thing that differs between views is
  which variable each greedy step chooses to flip:
  - Raw: WalkSAT — pick an unsat clause, flip the variable that
    minimizes resulting unsat count.
  - Spectral / Pascal / signed Pascal / Sierpinski: project strain
    onto the view, find the direction carrying the most strain,
    flip the variable with the largest loading along that direction
    that participates in some unsat clause.
- Reports flips-to-solve, residual unsat after a budget, and the
  total-strain trajectory per view, plus a head-to-head table
  against the raw baseline.

## Why this is honest

- All views pay the same clause-check budget per flip. No view is a
  faster algorithm; they are the same algorithm asking different
  questions about *which variable to act on*.
- The probe runs on toy sizes (≤12 variables). Any positive signal
  here is suggestive at most — it speaks to whether the *idea* is
  worth scaling, not to whether it scales.
- The structural-unsat instances are a guardrail. A transform that
  starts "solving" UNSAT formulas is a transform that has broken
  something. No transform in this probe does.
- Tests assert (a) the Pascal recurrence actually holds in the matrix
  we construct, (b) signed Pascal really is the multiplicative
  inverse of Pascal, (c) Sierpinski really is Pascal mod 2,
  (d) row-normalized bases stay scale-bounded across the n_vars we
  exercise, and (e) the head-to-head report is reproducible under a
  fixed seed.

## Early reading

On the seeded suite shipped with this PR (run
`python experiments/riordan_probe.py`):

- **Pascal**: 1 win, 6 ties, 1 loss vs raw. The single win is on the
  near-threshold 3-SAT instance where raw plateaus — exactly the
  regime where representation geometry would plausibly matter if it
  matters at all.
- **Signed Pascal**: 0 wins, 7 ties, 1 loss. The inverse partner is
  closer to neutral; it sometimes finds the same fixed points faster
  in flip count, but rarely changes the win/loss outcome.
- **Sierpinski**: 1 win, 5 ties, 2 losses. Slightly noisier than
  Pascal, which is what you'd expect from a "structure without
  magnitude" variant.
- **Spectral**: in this configuration, 0 wins, 5 ties, 3 losses vs
  raw — weaker than the earlier flattening-probe write-up reported.
  The difference is the suite shifted slightly between runs; this is
  the number to trust going forward.

That is a faint, possibly real directional signal for the
recurrence-preserving views on the regime where constraints fight
each other, and a clean neutral result everywhere else. It is not
strong enough to claim a method. It is interesting enough to keep
asking.

The honest summary: **Pascal ≥ raw on most instances and strictly
better on one near-threshold instance**. The natural follow-ups are
(a) larger instance sweeps to see whether the signal survives, (b)
mixing Pascal-direction selection with WalkSAT's local fix-up rather
than replacing it, and (c) other Riordan-family members (Catalan,
Motzkin) to test whether the win generalizes within the family or is
specific to Pascal.

## Where things live

- `geometry/riordan_probe.py` — `pascal_matrix`, `signed_pascal_matrix`,
  `sierpinski_matrix`, the three views, `RiordanProbe` subclass,
  `head_to_head` helper.
- `experiments/riordan_probe.py` — driver + tables.
- `tests/test_riordan_probe.py` — Pascal algebra, scale-boundedness,
  determinism, report reproducibility.
- Builds on `geometry/flattening_probe.py` from PR #7.

## Limitations and what this is not

- This is not a SAT solver. It is a probe that swaps the variable-
  selection heuristic of WalkSAT and measures whether the swap helps.
- The instances are too small to draw scaling conclusions from.
- "Recurrence-preserving" is precise as a property of the basis
  matrix (Pascal's additive recurrence is literally how the matrix is
  built), but its connection to SAT structure is conjectural — there
  is no claim that the variables of a SAT instance *have* a natural
  recurrence that Pascal captures. The Pascal view imposes one. The
  question is whether that imposition happens to help.
- A negative or neutral result here would be just as informative as a
  positive one. Both have been observed; we report both.
