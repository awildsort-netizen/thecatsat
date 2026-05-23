# Transform litmus: does a SAT coordinate change *localize* conflict?

A small diagnostic that connects two earlier probes in this repository:

- **PR #7 / `geometry/flattening_probe.py`, `geometry/riordan_probe.py`** — runs
  WalkSAT-style local search through raw, spectral, and Riordan/Pascal
  coordinate views on a seeded SAT suite, reports flips, solve flag,
  residual unsat, motion labels, and compact traces. The expanded
  result was sobering: across 22 instances and 4 transforms only one
  case (`3sat_v12_c42_s2` × `signed_pascal`) cleanly unblocks a raw
  plateau.
- **PR #8 / `geometry/tangent_lift_probe.py`** — the user's *tangent
  test* in continuous form: the scalar `tan(x)` blows up globally;
  lifted into `(sin x, cos x)` the same singularity becomes a typed
  boundary condition. The probe puts numbers on the gap (~17 orders
  of magnitude in `max_finite_difference` near `π/2`).

The user's question is the natural follow-up: **does the same litmus
apply on SAT?** Concretely — when a coordinate change does *not* solve
an instance, does it at least turn nonlocal conflict (residual strain
spread thinly across many variables, plateaus where flips just
shuffle) into *localized* typed strain — residual pressure collapsed
onto a small named set? And does that localization correlate with the
transform actually helping the solver finish?

This page documents what we measure, what we found on the existing
22-instance suite, and — honestly — what we did **not** find.

## What's added

- `geometry/transform_litmus.py` — three localization statistics
  (`top_k_share`, `herfindahl`, `gini`), a `StrainLocalization` record
  collecting them plus residual `support`, a small classifier
  producing one of six verdicts, and a `summarize` helper that
  computes the verdict↔solve association across many readings.
- `experiments/transform_litmus.py` — driver that reuses the **exact
  same** 22-instance seeded suite as `experiments/riordan_probe.py`,
  computes a litmus reading per (instance, non-raw view), and prints
  the per-pair table, per-view verdict counts, and the association
  table.
- `tests/test_transform_litmus.py` — 21 deterministic tests: metric
  extremes, classifier rules on synthetic runs, end-to-end
  reproducibility on a probe result, and a **cross-process
  reproducibility test** that guards the determinism contract.
- A small fix to `geometry/flattening_probe.py`: it previously used
  Python's built-in `hash()` to derive per-view RNG offsets, which is
  randomized per Python process (PYTHONHASHSEED). Within a single
  process this was invisible — and the existing test suite passed —
  but across processes the report numbers wobbled. The litmus needs
  cross-process determinism, so we switched to `hashlib.md5`.

## The verdicts

For each (instance, non-raw view) pair the litmus emits one of:

| verdict                  | meaning                                                                                            |
|--------------------------|----------------------------------------------------------------------------------------------------|
| `resolved_to_boundary`   | The view solved an instance the raw baseline did not. The SAT analogue of the tangent lift's win. |
| `localized_but_unstable` | Same or fewer residual unsat clauses, but residual strain *concentrated* onto a smaller set.       |
| `moved_singularity`      | Same residual magnitude, but strain shifted to a different set of variables.                       |
| `amplified_pathology`    | Strictly more residual unsat than the baseline. The transform made things worse.                   |
| `both_solved`            | Both views solved. No residual to localize on either side.                                         |
| `no_change`              | Neither solved, residual magnitude and localization roughly equal.                                 |

Localization is the **top-k share** (default `k=3`) of the residual
per-variable strain, plus `support` (how many variables have positive
residual strain), plus Herfindahl and Gini as supplementary statistics.

## What we measured on the existing suite

`max_flips=200`, `seed=7`, same instance generator as `experiments/riordan_probe.py`.
88 (instance, non-raw view) pairs total (22 instances × 4 non-raw views
that always run, plus the spectral view whose `k` varies with `n_vars`).

```
Per-view verdict counts:
view              resolved   localized    moved     amplified    both_solved   no_change
                  _to_       _but_        _singu    _patho                     
                  boundary   unstable     larity    logy                       
------------------------------------------------------------------------------------------
pascal            1          0            0         6            13            2
sierpinski        0          0            0         14           7             1
signed_pascal     1          0            0         4            15            2
spectral(k=10)    0          0            0         2            4             0
spectral(k=12)    0          0            0         4            5             0
spectral(k=14)    0          0            0         1            1             0
spectral(k=8)     0          0            0         0            3             2

Association: verdict ↔ solve outcome
verdict                  count   solve_rate   improve_rate
----------------------------------------------------------
resolved_to_boundary     2       1.00         1.00
localized_but_unstable   0       0.00         0.00
moved_singularity        0       0.00         0.00
amplified_pathology      31      0.00         0.00
both_solved              48      1.00         0.00
no_change                7       0.00         0.00
```

## Honest reading

The litmus separates the runs cleanly, but the answer to the user's
question is **mostly no, with a small caveat**:

1. **The interesting middle verdicts are empty.** No pair landed on
   `localized_but_unstable` and none on `moved_singularity`. On this
   suite, when a transform fails to solve, it doesn't fail in a
   structurally useful way — the residual strain is not collapsing onto
   a smaller named set while the solver still runs out of flips. It
   either solves outright (`resolved_to_boundary` / `both_solved`),
   stays roughly the same (`no_change`), or makes things worse
   (`amplified_pathology`).

2. **`amplified_pathology` is the dominant non-trivial verdict.** 31 of
   88 pairs ended with strictly more residual unsat than raw. That
   includes both "raw solved, view didn't" (which is largely
   structural — the view's `_choose_spectral` chooser is making bad
   variable picks when raw's WalkSAT-greedy was already finding a
   path) and "neither solved, view ended worse".

3. **`resolved_to_boundary` is real but rare.** Exactly 2/88 pairs
   (e.g. `signed_pascal` on `3sat_v12_c42_s2`) unblock a raw failure.
   That matches PR #7's "one plateau-unblocking case in full suite"
   finding — the litmus just gives it a name and isolates which view
   does it.

4. **The SAT analogue of the tangent test does not light up.** The
   tangent lift's signal is the ~17-orders-of-magnitude gap between
   raw and lifted `max_finite_difference`. The litmus's equivalent
   would be: when a transform doesn't solve, does its residual strain
   nonetheless look more "boundary-like" (high top-k share, small
   support) than the raw baseline's? On this suite the answer is no
   — the `localized_but_unstable` bucket is empty.

What this tells us is the conservative reading. The SAT analogue of
"some hardness is representation geometry" is *not* falsified, but on
this suite it is also not visibly supported by a localization-without-
solving signal. The one positive finding — `resolved_to_boundary` at
solve rate 1.00 by construction — is too small a sample (2) to read as
anything other than the existing PR #7 result re-described.

## What the litmus does **not** claim

- It does not measure whether transforms reduce solver runtime in
  absolute terms (`flips` differences are PR #7's motion labels' job).
- It does not propose a coordinate change that *would* light up the
  middle verdicts — the litmus is diagnostic, not prescriptive.
- It does not claim the metric generalizes off this suite. The
  instances are small on purpose (≤14 variables).
- It does not equate the SAT and tangent-lift stories. The tangent
  lift had a *typed* boundary condition the lifted chart can name
  (`cos x = 0`). The SAT litmus's `localized_but_unstable` verdict is
  a much weaker claim — concentrated residual strain, not a typed
  event. The two probes share a question, not a mechanism.

## Reproducing

```bash
python3 -m unittest tests.test_flattening_probe \
                     tests.test_riordan_probe \
                     tests.test_transform_litmus
python3 experiments/transform_litmus.py
```

The cross-process determinism test in
`tests/test_transform_litmus.py::ReportStabilityTests` is the contract
that backs the numbers above.

## Future experiments (not in this PR)

- Add a transform family designed to *cause* `localized_but_unstable`
  (e.g. a basis that explicitly clusters variables by clause
  co-occurrence and then projects to top-1). That would test whether
  the verdict is achievable at all on this suite, or whether it is
  empty because the suite is too small.
- Per-clause version of the metric: which *clauses* carry residual
  unsat, not which variables. SAT solvers act on clauses; the
  variable-level localization may be the wrong unit of analysis.
- Larger near-threshold sweeps: the interesting case is `n=12..20`
  near the 4.26 ratio; the current grid stops at `n=14`.
