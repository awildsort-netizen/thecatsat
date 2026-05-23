# Composable SAT strategy operators

This is the first *behavior-altering* intervention after the merged
geometry / bubble metrics stack. Everything before it (the flattening
probe, the Riordan probe, the transform litmus, the bubble lifecycle
scaffold, the bubble tuning pressure gauge) added *diagnostics* and
left the existing furnace and WalkSAT-style search loop unchanged.

This change adds composable operators that *decide flips*. It does not
introduce a new solver. It does not hardcode a bubble solver. It
decomposes the local-search strategy already encoded in
`geometry/flattening_probe.py` into small pieces, lifts those pieces
into the project's operator vocabulary, and offers a composer that
assembles them.

## Why decompose now

The user framing was explicit:

> let's try to not hardcode and trust the composer. one potentially
> fruitful path is to decompose existing sat solver strategies into its
> constituent parts, in operator form that is composable.

The two flattening-probe choosers,
`flattening_probe._choose_raw` and `flattening_probe._choose_spectral`,
are already structured like the same handful of moves stuck together
with `if`/`else`:

1. pick an unsat clause,
2. with small probability, pick a variable in that clause at random
   (plateau kick),
3. otherwise, rank candidate variables by some scalar and propose the
   best one,
4. (in the spectral variant) skip the rank step when projected strain
   is all zero, and fall back to the raw rank.

The "if/else" between those moves is a strategy decision, not a
chooser implementation detail. Once that decision lives in a list of
operators rather than inside a chooser body, you can add a new
strategy step — e.g. *bubble pressure veto* — by inserting one
operator, not by editing every chooser.

## Operator surface

```text
strategy/
├── __init__.py
├── operators.py        # Proposal, SearchState, StrategyComposer + operators
├── presets.py          # raw / transformed / gated preset operator lists
└── run.py              # tiny composed_local_search driver
```

The interface is small:

- `SearchState` — formula, assignment, n_vars, step, rng, and a small
  mutable `field` dict that operators publish into.
- `Proposal` — `(variable, operator, reason)`. Returning a
  `Proposal` means "I want to flip this variable". Returning `None`
  means "I yield to the next operator".
- `StrategyOperator` — just a callable `SearchState → Proposal | None`.
- `StrategyComposer` — a tuple of operators traversed with `next()`;
  the first non-`None` proposal wins. Same first-match pattern the
  bubble-tuning `RULES` table uses.

Operators in this PR:

| Operator | Reads | Writes to `field` | Returns |
| --- | --- | --- | --- |
| `unsat_clause_focus` | `formula`, `assignment` | `focused_clause`, `unsat_count` | `None` (selector) |
| `random_walk_kick(p)` | `focused_clause` | — | `Proposal` w/ prob `p` |
| `raw_strain_ranker` | `focused_clause`, `formula`, `assignment` | — | `Proposal` |
| `coordinate_ranker(view)` | `formula`, `assignment`, optional `veto_transformed` | `coordinate_dominant`, `coordinate_vetoed` | `Proposal` |
| `plateau_detector(window,band)` | `unsat_history` | `plateau`, `plateau_amplitude` | `None` (gate) |
| `bubble_pressure_gate()` | `strain_trace`, `bubble_candidate` | `bubble_pressure_label`, `veto_transformed` | `None` (gate) |

The gates and the selector are intentionally "no-proposal" operators.
They only publish field signals. That keeps the proposal channel
free for rankers and keeps the gating logic out of the rankers'
bodies.

## The composed intervention

The `gated_transformed_composer` preset is the new composed policy.
Operator order:

1. `plateau_detector()` — publishes a `plateau` signal,
2. `bubble_pressure_gate()` — may set `veto_transformed=True`,
3. `unsat_clause_focus` — picks a random unsat clause,
4. `random_walk_kick(p)` — plateau kick,
5. `coordinate_ranker(view)` — yields when `veto_transformed`,
6. `raw_strain_ranker` — picks up when the transform yielded.

No `if`/`elif` ladder anywhere. Branching is the operator order plus
the field signals each operator decides to publish. Adding a new
gate or ranker is a matter of inserting one operator in the right
place in the list.

## Honest comparison

`experiments/strategy_compare.py` runs the three presets — `raw`,
`transformed`, `gated` — on the same toy suite the flattening probe
already uses (small 2-SAT, 3-SAT, and structural-unsat instances),
under the same per-instance flip budget and seed:

```text
Aggregate (mean over instances; lower=better):
composer       mean flips   solve rate   mean final_unsat
----------------------------------------------------------
raw            37.1         0.62         0.38
transformed    55.1         0.38         1.00
gated          37.1         0.62         0.38
```

Read this carefully:

- The raw and gated rows are identical *because* the synthetic
  destructive strain trace we feed the gate fires the veto on every
  step. That is the case the bubble-tuning paper has been calling out
  for two PRs: a transform that is off-phase produces destructive
  amplification, and the right move is to *not* take its proposal.
  The gated composer demonstrates that operator routing reaches that
  decision and falls back to the raw ranker, instead of the chooser
  body silently making the same call internally.
- The transformed composer underperforms on this suite. That is not
  a new finding — it is the same signal the flattening probe and
  Riordan probe already publish. What this PR adds is a *composable
  surface* on which to attach a gate that suppresses transforms when
  the gauge says they are destructive.
- The bubble-gate path audit confirms the veto path is reached on
  297/297 steps with the synthetic trace, so the test is exercising
  the operator wiring rather than rubber-stamping.

This is deliberately a small claim. The composed policy is *not*
broadly beating raw; it is matching raw under conditions that would
otherwise have invited the transform to do harm. That is the
first-intervention-step success criterion the task asked for.

## What this is not

- Not a new solver. The furnace, the WalkSAT-style choosers, the
  DPLL / brute-force baselines in `sat_benchmarks.py` are unchanged.
- Not a bubble-pressure solver. Bubble pressure is *one* operator in
  the list and only writes a field signal that downstream operators
  may consult. Replacing it with a different gate (e.g. one driven
  by `plateau_detector`, or by the Riordan litmus) is a one-line
  change.
- Not a claim that this generalizes. The instances are small and
  toy. The point of this PR is the *decomposition*, so that the
  next intervention can be added by changing the operator list,
  not by editing chooser bodies.

## Where this connects

- `geometry/flattening_probe.py` — source of `_choose_raw`,
  `_choose_spectral`, and `CoordinateView`. The new
  `raw_strain_ranker` and `coordinate_ranker` are lifted from these
  two functions with minimal changes.
- `geometry/bubble_tuning.py` — source of `measure_pressure` and the
  `RULES` rule-table pattern. The new `bubble_pressure_gate` reuses
  the *label* output of that gauge and provides a tiny veto-mapping
  table of its own.
- `composer.py` — the project's existing operator composer. The
  strategy composer here is intentionally a thinner, list-based
  variant because the dataflow per step is linear: one operator
  publishes to the field, the next reads it. The `Composer` DAG
  is the right tool for cross-step plumbing (rename maps, target
  resolution); the strategy composer is the right tool for per-step
  rule-table policy.
- `docs/representational_bubbles.md` and
  `docs/tests_as_activation_factors.md` — the framing this PR
  cashes out. Bubble pressure is now a composable *activation
  factor* on the proposal channel, not just a number printed in a
  diagnostic table.
