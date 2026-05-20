# Control-Flow Trim Map

Companion to `interpretation_sieve.md` and `conversation_metabolism.md`.

This pass inventories the loops and branches in the source and classifies
them by what they're *for*. The goal isn't to delete control flow — kernel
math/solver loops are doing real work — but to identify the *orchestration
weeds*: wrapper-level loops, mode if-chains, and manual provider selection
that the Composer can reconstruct through planning, iteration, eligibility,
and providers.

## Density snapshot (LOC vs. loops + ifs)

```
composer.py                 lines= 432  loops=19  ifs=30  density=11.3%
sprite_detector.py          lines= 481  loops=19  ifs=34  density=11.0%
spectral_calorimeter.py     lines= 455  loops=10  ifs=40  density=11.0%
external_sat.py             lines= 277  loops= 8  ifs=20  density=10.1%
attention_policies.py       lines= 145  loops= 2  ifs=12  density= 9.7%
streamable_genes.py         lines= 160  loops= 4  ifs=10  density= 8.8%
sat_furnace.py              lines= 905  loops=32  ifs=46  density= 8.6% *
bytecode_gene_summary.py    lines= 318  loops= 7  ifs=19  density= 8.2%
benchmark_calorimeter.py    lines=2596  loops=63  ifs=115 density= 6.9%
```

(* before this pass, ~1pt higher; the `run_furnace` orchestration loop
moved into `Composer.iterate`.)

## Classification

### Done in this pass

**`sat_furnace.run_furnace`** — manual epoch driver lifted into
`Composer.iterate`.

The original loop did six things by hand for each step:

1. Stamp `t` into context.
2. Carry six per-step outputs forward into `prev_*` accumulator inputs.
3. Derive `previous_unsatisfied` / `previous_integration` from
   `prev_samples[-1]` (with a scalar fallback).
4. Drop the `_STALE_EPOCH_KEYS` frozenset (33 keys) so the planner
   re-derives them.
5. Run `composer.run(_EPOCH_TARGETS, ctx)`.
6. Re-route `next_spins`/`next_velocity` into `spins`/`velocity`.

All six are bookkeeping the Composer already knows how to do:

- (1) → `step_key="t"`
- (2, 6) → `rename_map={"samples": "prev_samples", ...,
  "next_spins": "spins", ...}`
- (3) → `before_step=_carry_previous_scalars`
- (4) → automatic stale-cleanup of `target_set - rename_sources`
- (5) → driven by `composer.iterate(...)`

`_STALE_EPOCH_KEYS` was deleted: it was exactly `_EPOCH_TARGETS - {fiber_memory}`,
and `iterate` derives the equivalent set from the rename map. `fiber_memory`
(mutated in place across iterations) is the one key that needs explicit
`preserve=("fiber_memory",)`.

Net: -41 / +48 lines, control flow inside `run_furnace` collapses from
24 lines of manual epoch wiring to a single `composer.iterate(...)` call
plus a 4-line closing remap.

The closing remap (`for src, dst in _EPOCH_RENAME_MAP.items(): final_ctx[src] = final_ctx[dst]`)
is the one remaining manual step. It exists because the rename map drops
the unprefixed keys (`samples`, `operator_traces`, ...) once they've been
carried to `prev_*`, but the closing `solver.final_assignment` /
`solver.furnace_result` plan reads them by their unprefixed names. A
future improvement would be either a small `IterationResult.final_context`
view that exposes the rename sources, or letting `composer.iterate` take
a `final_targets` parameter that runs an unrenamed closing plan.

## Candidates (not implemented this pass)

### 1. `benchmark_calorimeter._mutation_controls` if-chain — done in `computer/mutation-control-trim`

**Kind (was):** mode if-chain (9 branches keyed on `candidate.mutation`).

**Shape:** each branch is a 2-4 line dict-like assignment of
`(adaptive, policy, threshold, slope, decay, drive, lr_scale, inertia_delta,
noise_delta)` — i.e. a small parameter delta per mutation tag.

**What was done:** intermediate step on the way to operator-per-tag.
Each tag is a tiny module-level handler `(_mutation_<tag>(state, strength))`
mutating a shared `state` dict, and the table `_MUTATION_HANDLERS` maps
tag → handler. The if-chain collapses to one dict lookup + handler call,
followed by the same clamp-and-pack into `MutationControls`. The handler
table is what a composer-native version would consume directly: each
handler is already side-effect-typed (`Callable[[dict, float], None]`),
so wrapping each as a `FieldOperator` with `enabled=ctx["mutation"]==TAG`
is now a near-mechanical lift if/when we promote the calorimeter to a
composer plan.

**Composer reconstruction (still open):** register one tiny
`FieldOperator` per mutation tag, each declaring
`enabled=lambda ctx: ctx["mutation"] == TAG`, all writing to the same set
of output keys. The composer's plan would then fire exactly the eligible
one. Mutation tags become *operators with eligibility*, not branches in a
control-flow tree. Worth doing alongside candidate #2 below.

**Behavior preservation:** new tests
`test_mutation_controls_table_covers_every_handler` and
`test_mutation_controls_specific_deltas_match_legacy_chain` pin every
registered tag, the default branch, and three specific numeric deltas
from the previous if-chain.

### 2. `benchmark_calorimeter` policy/mutation cascade in `run_trial` neighbourhood

**Kind:** mixed — orchestration ifs choosing policy + adaptive flags +
delta values.

**Composer reconstruction:** the calorimeter is currently the *consumer*
of `run_furnace`; the policy/adaptive choice could itself become a small
composer plan whose final operator outputs the `FurnaceResult`. This
would let the calorimeter be benchmarked the same way `sat_benchmarks`
was promoted (commit `0b066ad`).

**Risk:** high. The calorimeter has wide downstream contracts (CSV
columns, mutation telemetry). A composer-native rewrite is a larger
piece of work that wants its own branch.

### 3. `sat_furnace._spin_update_step` inner loop (lines 305–319)

**Kind:** kernel math loop — per-variable drive accumulation, velocity
update, tanh spin update.

**Decision:** keep. This is the inner mathematical kernel of the
solver; the `if adaptive_active and memory_scale > 0` and
`if mixed_drive is not None and mixed_scale > 0` branches gate
optional drives that the planner upstream has already decided to
provide (or set to zero/None). Lifting these into operator eligibility
would push per-step branching into the planner and likely cost more
than it saves.

### 4. `sat_furnace.generate_formula` mode dispatch (lines 88–96)

**Kind:** mode if-chain (4 kinds: sat, hard_sat, unsat, random).

**Composer reconstruction:** four eligibility-gated `FieldOperator`s,
all producing `(formula, planted)`, each enabled on `ctx["kind"]`.

**Risk:** low-value. This is a 4-line dispatch in a top-level entry
point; promoting it to a composer adds machinery without removing
business logic. Leave as-is unless we add a fifth kind that shares
infrastructure.

### 5. `external_sat` parser branches (lines 81–135)

**Kind:** parser branching (DIMACS / minisat / picosat output formats).

**Decision:** keep. This is interpretation logic for *external* tool
output and follows the structure of those tools' protocols. Not
orchestration; not a candidate.

### 6. `walksat_baseline_with_trace` flip loop (`benchmark_calorimeter.py:399`)

**Kind:** baseline solver kernel loop.

**Decision:** keep. This is one of two solver baselines we benchmark
the furnace against. Reshaping it into a composer plan would defeat
the purpose of having a *minimal* baseline implementation.

### 7. `sprite_detector` / `spectral_calorimeter` density

**Kind:** detection/feature-extraction loops over spatial/spectral
frames.

**Decision:** keep. The density is high because these files do nested
walks over per-frame data; the loops are tight and local. No
orchestration weeds.

## Heuristic for future passes

> **Kernel** loops compute a value from a value. **Orchestration**
> loops compose existing operators across a parameter (a step index,
> a mode tag, a candidate). Only orchestration loops are candidates
> for `Composer.iterate` / eligibility / provider selection.

The `run_furnace` lift demonstrates the tell: the original loop
contained no math — every line was either context plumbing (rename,
clear, stamp) or a single `composer.run(_EPOCH_TARGETS, ctx)` call.
When a loop's body is "wire context, then call composer.run", the
loop is orchestration weeds.

## Typed boundaries vs. defensive ifs

A related principle for future trims: many `if x is None`, `isinstance`,
or normalization branches inside function bodies aren't business logic —
they're a function re-checking what its annotations already claim. The
direction is to move those checks to the *boundary* (the
operator/composer entry) and let bodies trust their signatures.

The composer already has the canonical boundary tool: `require_keys`
(`composer.py:421`). Any operator whose inputs are validated by
`require_keys` can drop matching `if key not in ctx` / `if key is None`
checks from its body.

### What counts as a defensive weed

- `if x is None: return default` at the top of a function whose
  parameter is annotated non-`Optional` — the caller (or a missing
  provider) is the real source.
- `isinstance(value, str): value.lower() == "true"` style normalization
  scattered across consumers when the value should have been normalized
  once at the CSV/IO border.
- `if not isinstance(result, Mapping): raise` checks duplicated across
  several operators when the composer already enforces the shape via
  `require_keys` + the multi-output dispatch in `composer.py:394–398`.

### What is NOT a weed (keep)

- Kernel checks: DPLL's `if assignment[var] is None` is the algorithm,
  not a guard (`sat_benchmarks.py:112`).
- Border conversion: `metric_float` / `metric_bool` in
  `benchmark_calorimeter.py:1706–1717` — these *are* the boundary
  (CSV → typed); their `try/except` and `isinstance(value, str)` are the
  authoritative normalization, not a redundant check.
- Union-type dispatch on a legitimately polymorphic interface, e.g.
  `attention_policies.resolve` (`attention_policies.py:101–106`)
  normalizing `None | str | Iterable[str]`. This *could* be tightened by
  making every policy return a tuple, but that's a contract change to
  every registered policy and is a separate task.
- Multi-output result dispatch in `composer.py:394–398`
  (`dataclass` vs `Mapping` vs `raise`) — this is the boundary itself.

### Candidates worth a small boundary helper

Not implemented in this pass — recorded for the next trim:

1. **`mutation_controls_from_candidate` early return on
   `candidate.gene == "none" or candidate.score <= 0.0`** — this is
   eligibility, not a defensive check. In a composer-native version it
   would be a single `enabled=` predicate on the mutation operator, and
   the disabled-controls fallback would simply not run.
2. **Repeated `str(row.get(key, "none"))` / `float(row.get(key, 0.0))`
   coercion at `benchmark_calorimeter.py:271–276`** — this is a row →
   typed `GeneMutationCandidate` border. A small
   `GeneMutationCandidate.from_row(row)` classmethod or a
   `require_typed_keys` composer validator (built on `require_keys`)
   would move the coercion to one place and let downstream code trust
   the candidate's annotations.
3. **`if value is None` guards in `streamable_genes.py:119`,
   `sprite_detector.py:90`, `spectral_calorimeter.py:316`** — each is a
   single-use optional default. Most are fine where they are; only worth
   moving if the same key acquires a third or fourth consumer.

### Principle

> Annotations describe what a function *expects to receive*. If the
> composer (or its `require_keys` validator) enforces those expectations
> at the operator boundary, the function body doesn't need to re-check
> them. Trust the ecology: validate once at the seam, simplify the
> middle.

This is the typed-border counterpart to the orchestration heuristic
above: kernels compute, boundaries validate, and the body in between
should be free of either job.
