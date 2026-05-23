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

## Early reading (initial 8-instance suite)

On the original seeded suite (8 instances: easy 2-SAT, 3 mid-density
3-SAT, structural-UNSAT):

- **Pascal**: 1 win, 6 ties, 1 loss vs raw. The single win is on the
  near-threshold 3-SAT instance where raw plateaus — exactly the
  regime where representation geometry would plausibly matter if it
  matters at all.
- **Signed Pascal**: 0 wins, 7 ties, 1 loss. The inverse partner is
  closer to neutral.
- **Sierpinski**: 1 win, 5 ties, 2 losses.
- **Spectral**: 0 wins, 5 ties, 3 losses vs raw.

That looked like a faint directional signal for Pascal on the regime
where constraints fight each other. The follow-up below tested
whether that signal survives a slightly larger and more deliberate
near-threshold slice.

## Expanded read (paced suite, 22 instances)

Run `python experiments/riordan_probe.py`. The expanded suite adds
14 near-threshold 3-SAT cases on a small deterministic grid of
`(n_vars, clauses/variables)` pairs around the random-3-SAT phase
transition at ratio ≈ 4.26: cells `(10, 4.0)`, `(10, 4.3)`,
`(10, 4.6)`, `(12, 4.0)`, `(12, 4.3)`, `(12, 4.6)`, `(14, 4.3)`,
two seeds each. Runtime stays ~2.5s. The full head-to-head against
raw on final unsatisfied-clause count:

| view             | wins | ties | losses |
|------------------|-----:|-----:|-------:|
| pascal           | 0    | 12   | 10     |
| signed_pascal    | 0    | 19   | 3      |
| sierpinski       | 1    | 11   | 10     |
| spectral(k=8)    | 0    | 4    | 1      |
| spectral(k=10)   | 0    | 2    | 4      |
| spectral(k=12)   | 0    | 6    | 3      |
| spectral(k=14)   | 0    | 1    | 1      |

Per-family slice (just `3sat_threshold`, where the question is
actually contested):

| view             | wins | ties | losses |
|------------------|-----:|-----:|-------:|
| pascal           | 0    | 6    | 8      |
| signed_pascal    | 0    | 12   | 2      |
| sierpinski       | 0    | 6    | 8      |
| spectral(k=10)   | 0    | 2    | 4      |
| spectral(k=12)   | 0    | 5    | 1      |
| spectral(k=14)   | 0    | 1    | 1      |

This is honestly weaker than the initial read. **The Pascal win
observed on `3sat_v12_c42_s2` does not generalize.** Across 14
deterministic near-threshold cases:

- Pascal goes 0/6/8 — meaningfully worse than raw on this slice.
- Signed Pascal becomes the cleanest neutral (0/12/2): it almost
  never destabilizes, but it also doesn't outright unblock raw
  anywhere in the expanded slice.
- Sierpinski's one win across the full suite is an `unblocks_plateau`
  on `3sat_threshold_v10_r4.3_s1`, where raw plateaus at strain 21→3
  for 62 consecutive steps and three of the four non-raw views all
  reach 0. That single co-win is suggestive (it's the only place all
  three recurrence views agree that raw is stuck), not conclusive.
- Spectral views are noisy at small `k` and not noticeably better at
  larger `k` on this slice.
- Both structural-UNSAT instances still come back unsatisfiable
  across every view; the guardrail holds.

## Motion-type labels

To orient the eye over a longer case table, each non-raw view gets
one of a small fixed set of labels per instance, relative to the raw
baseline:

- `matches_raw` — same final unsat, comparable flip count.
- `improves` — strictly fewer final-unsat clauses.
- `unblocks_plateau` — raw plateaued (long flat-strain stretch and
  didn't solve) and the view solved.
- `destabilizes` — view ended worse than raw.
- `faster_same_outcome` / `slower_same_outcome` — same final unsat
  but meaningfully different flip counts (solved cases only).

These are deliberately not a typology. They are a small fixed
vocabulary so the case table is scannable. The full per-view
distribution lives in the driver output; the only one worth pulling
out is that across the expanded suite there is exactly one
`unblocks_plateau` label (Sierpinski on
`3sat_threshold_v10_r4.3_s1`) — the rest of the non-trivial
divergences from raw are `slower_same_outcome` or `destabilizes`.

## Honest summary

The signal seen in the original 8-instance suite **did not survive**
a moderate, deliberately-scoped expansion to 22 instances. On
near-threshold 3-SAT specifically, Pascal is roughly break-even and
sometimes worse than raw, Sierpinski is similar, and signed Pascal is
mostly neutral. The one clean co-win across the recurrence views is
on a single instance where raw is unambiguously plateaued, which is
interesting but is one instance.

The framing — "some hardness is representation geometry; rotating the
coordinate system might help" — still has a foothold on plateau
cases, but the strong reading ("Pascal helps on near-threshold 3-SAT
specifically") is not supported by this slice. That is itself useful
information.

## Paced next steps

Not in this PR. Listed in rough order of cost.

- **Sharpen the plateau case.** Build a tiny sub-suite of instances
  curated to make raw plateau (long flat strain, didn't solve in
  budget). Re-test whether recurrence-preserving views agree on that
  sub-population. This is the only place a directional signal still
  shows.
- **Mix, don't replace.** Keep WalkSAT's local fix-up and let the
  Pascal projection only break ties on which clause to repair. The
  current chooser fully replaces raw's heuristic; that's a strong
  intervention.
- **Strain decay rate, not just endpoint.** The current head-to-head
  only looks at final unsat. Two views with the same final value
  can have very different trajectories; the motion labels hint at
  this but don't quantify it.
- **One more Riordan member, not all of them.** Catalan or Motzkin
  would test whether the (small) plateau signal is Pascal-specific
  or family-wide. Resist running all of them at once.
- **A random-rotation null.** A view whose basis is a random
  orthogonal matrix is the right control for "did spectral help
  because it's spectral, or because it's not identity?" — but only
  worth running if the plateau sub-suite shows anything first.

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
