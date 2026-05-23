# Riordan bubble fit

Companion follow-up to PR #13. PR #13 introduced composable strategy
operators and a bubble-pressure gate — a *veto* signal that successfully
shuts the transformed coordinate ranker off when the chart is off-phase
(destructive amplification). This was useful as a guard but not yet
constructive: it never picked a transform, it only chose when *not* to
use one.

This module adds the smallest constructive layer that sits next to the
gate: a **Riordan bubble fitter** that *selects* a transform from a
small candidate set by bubble pressure / containment criteria — never by
SAT outcome.

## What the fit is, what it isn't

**What it is.** A pure function from a per-variable strain vector
(optionally plus a strain trace) to a typed `FitDecision`. The decision
names a selected `CoordinateView` chosen from a candidate registry
— by default identity, Pascal, signed Pascal, Sierpinski — plus its
inflated bubble and its rationale. Selection scores each candidate via:

- interior share (strain captured by the inflated bubble's interior)
- off-bubble strain (leak share outside interior ∪ boundary)
- within-bubble std relative to interior mean
- boundary stability (1 minus top-k turnover rate)

These are composed in a small linear combination, deliberately
unsubtle — one of these flips sign and the score moves the same way.
The pressure label comes from `geometry.bubble_tuning.measure_pressure`;
labels in `{destructive_amplification, diffuse_pressure}` mark a
candidate non-viable.

If *no* candidate is viable, the fitter falls back to identity and
emits a veto signal that the existing operator stack already knows how
to consume.

**What it isn't.** It is not a solver, and it is not allowed to look at
solver outcome. The same strain produces the same decision regardless
of which SAT formula generated it — outcome blindness is one of the
tests.

It is also deliberately small: the candidate set is a registry, the
score is four cheap signals, and tie-breaks are a deterministic sort.
No giant `if/elif` over transform families lives anywhere in this code.

## Composable operator

`strategy.operators.riordan_bubble_fitter` is the composable wrapper.
It reads `current_strain` (or reconstructs it from the assignment) and
optional `strain_trace` from the search state's field, runs `fit`,
and publishes:

- `fitted_view` — the selected `CoordinateView`
- `fitted_selected` — the candidate name
- `fit_rationale` — short string for audit logs
- `veto_transformed=True` only when no candidate was viable

A second operator, `fitted_coordinate_ranker`, consumes `fitted_view`
and runs the existing `coordinate_ranker` on it. On veto it yields
(marking `coordinate_vetoed=True`), and the raw ranker behind it takes
over. The routing is entirely field-signal flow — there is no branch
that reads "if pascal then …".

`strategy.fitted_composer()` wires it up:

1. `plateau_detector` — publishes `plateau`
2. `riordan_bubble_fitter` — picks the view, may veto
3. `unsat_clause_focus` — publishes the focused clause
4. `random_walk_kick`
5. `fitted_coordinate_ranker` — uses the picked view; yields on veto
6. `raw_strain_ranker` — fallback

## Honest results

Same suite as the flattening probe / strategy compare driver, same
seed, 80 flips per instance:

```
composer       mean flips   solve rate   mean final_unsat
----------------------------------------------------------
raw            37.1         0.62         0.38
transformed    55.1         0.38         1.00
gated          37.1         0.62         0.38
fitted         61.2         0.25         0.88
```

The fitted composer **underperforms raw on this suite**. The audit log
explains why: on 490 steps, the fitter selected `identity` 418 times
and `signed_pascal` 72 times, with zero vetoes. When the raw strain
genuinely concentrates (most steps), identity wins the score and the
behavior matches raw. The 72 `signed_pascal` selections are steps
where raw strain looked diffuse to the gauge but the transformed pick
turned out to be a *worse* search direction than the WalkSAT-style raw
greedy — the gauge said "transform localizes the strain", but on this
suite localized strain wasn't a better proxy for "good flip" than
greedy unsat-minimization.

The honest takeaway: **bubble pressure is a coarse proxy for "stable
representation", not for "good local-search direction" on small,
already-tractable SAT instances**. The suite has too few stable-bubble
candidates to give the fitter something constructive to do, and on the
remaining steps the gauge's bias toward Riordan-family picks is small
but real harm.

## Where the fitter behaves correctly

Three transparent synthetic controls live in
`experiments/riordan_bubble_fit_compare.py`:

1. **Stable bubble strain** `[5, 4.5, 4, 0.5, …, 0.5]`. The fitter
   picks `identity` — high concentration on the canonical axes, no
   need for a basis change.
2. **Diffuse strain** `[1, 1, 1, …, 1]`. Identity is `diffuse_pressure`
   (non-viable). The fitter picks `signed_pascal`, which has the
   highest interior share among the viable Riordan variants.
3. **Destructive trace** (off-phase, top-k churning). Every candidate
   reads `destructive_amplification`; the fitter vetoes and falls back
   to identity. This is the path PR #13's gate established, now
   reached without a hand-crafted bubble candidate.

The controls show the *fitter behavior* is correct on each regime; the
suite numbers show those regimes are not the regimes this suite was
generating.

## Where this lands

Constructively: the fitter is the smallest module that lets the
StrategyComposer pick a transform without a hardcoded if-tree, and the
field-signal flow extends cleanly when more candidates are added
(phase shifts, reindexings, other Riordan rows).

Honestly: it does not improve solve rate on the existing suite. The
useful question this PR raises is what a *stable-bubble-rich* SAT
suite would look like — instances where the raw strain is genuinely
diffuse but a recurrence-preserving rebasis concentrates it. The
existing suite does not generate those.

This is phase / bubble tuning, not solver magic.
