# Hallucination Geometry and Semantic Plaque

A design note for `parser_evolver`. Companion to the top-level
`docs/interpretation_sieve.md` and `docs/tests_as_activation_factors.md`.
This file is a conceptual map, not an implementation spec — it names the
geometry the existing typed-hallucination artifacts already hint at, and
sketches where the seed is allowed to grow.

## The claim

Hallucination is not "wrong output." Output is the last frame of a much
longer story. A hallucination is a **localized geometric strain**
between four things that should be in agreement:

- the **source manifold** (the evidence spans actually present in the
  input — what the page contains, where, in what order),
- the **operator pathway** (which operators fired, in what window, on
  what regions — the trajectory the solver actually took),
- the **attractor form** (the AF's required shape — columns, role
  vocabulary, row constraints, coverage expectations),
- the **emitted structure** (the candidate rows, traces, and
  hallucination artifacts the run produced).

When the four disagree locally — when the path the solver took does not
sit flush against the evidence under the attractor's required shape —
that disagreement *is* the hallucination. The misprint at the end is
just where the strain finally became legible.

This reframes the existing `Hallucination` sum type
(`unsupported_cell`, `field_role_confusion`, `validator_rejection`,
`missing_emitter`, `low_coverage_region`, `overfit_pattern`) as readings
of strain in different parts of that geometry, not as independent error
codes.

## Why structured loss must not collapse too early

A scalar loss is a single projection of a multi-dimensional strain
field. Collapse the field too early and you lose the ability to
distinguish "the emitter never fired" from "the emitter fired on the
wrong span" from "the row passed the schema but the columns came from
unrelated regions." All three give a similar headline number; only the
shape tells you what to repair.

Useful components, kept separate as long as possible:

- **unsupported_mass** — emitted cells with no source span behind them
  (the candidate is making things up out of nothing).
- **role_confusion** — a value that validates as field A but was emitted
  into B. The geometry says: the operator pathway crossed a role
  boundary it shouldn't have.
- **attractor_strain** — the schema-side cost. Required columns
  missing, row constraints rejecting, vocabulary drift.
- **low_coverage_entropy** — the source manifold is rich but the
  emitter pathway barely touched it; a large region of evidence is dark.
- **overfit_instability** — one field emitting far more rows than the
  candidate's row count; a local operator has grown too sharp and is
  paving over neighboring structure.

These are still numbers, but they are numbers with *location*. They
point back at trace regions and operator slots, which is what later
flow regression and `propose()`-style mutation will need.

## Hallucination regions behave mass-like

Once an unresolved constraint persists across turns, it stops behaving
like a momentary error and starts bending future computation:

- operators get pulled toward it (repair traffic concentrates),
- repairs accumulate at its edges (each patch adds a guard, a special
  case, a fallback),
- attention revisits the same region disproportionately,
- nearby interpretations curve around it (parses that would have used
  the region's spans now route around them),
- compute budget concentrates there (beam slots, embedding lookups,
  validator invocations).

That is the operational meaning of "mass-like" in this repo: a region
whose unresolved strain is large enough that future trajectories curve
in its presence. Not a metaphor about physics — a claim about where the
solver's budget actually goes.

## Stress-energy metaphor

A useful (and deliberately partial) analogy:

| Geometry | parser_evolver |
| --- | --- |
| Matter distribution | evidence spans on the source manifold |
| Transport dynamics | operator pathway (the run's gene-string + windows) |
| Material properties | gene bytecode — what this operator is *willing* to do |
| Boundary conditions | attractor form — the AF's required shape |
| Curvature / stress | hallucination regions; persistent strain |
| Local relaxation | repairs (guards, narrowings, span re-anchoring) |
| Geodesic | the solver's trajectory through interpretation space |

The point of the analogy is not the equations; it is that **geometry,
material, and trajectory are coupled**. You cannot fix a hallucination
by editing the emitted row alone, any more than you can flatten a
curved surface by repainting the line drawn on it.

## Operators have material profiles

Two operators with the same signature (same `needs`, same `provides`,
same embedding tokens) are not the same operator. Their *material
profile* differs:

- **brittleness** — how sharply they fail at the edge of their domain,
- **elasticity** — how much input variation they absorb before output
  shape changes,
- **dissipativity** — whether their failures are loud (typed
  hallucinations) or quiet (silently wrong cells),
- **overfit tendency** — how readily they grow guards that pattern-match
  one fixture,
- **stabilizing power** — whether their presence reduces strain in
  neighboring operators or merely shifts it,
- **repair affinity** — how often they end up as the patch site when
  something downstream breaks,
- **allocation / branching behavior** — beam slots consumed, extensions
  proposed, embedding lookups triggered,
- **bytecode topology** — flat vs motif-fused, header-heavy vs
  body-heavy,
- **historical repair traces** — the scar tissue this operator has
  accumulated across runs.

Conceptual sketch — not the current `ParseOperator` shape, just where
the seed could grow:

```ts
// Sketch only. Not required by current code.
type OperatorMaterial = {
  brittleness: number;          // [0,1], 1 = cliff at domain edge
  elasticity: number;           // [0,1], 1 = absorbs input variation
  dissipativity: number;        // [0,1], 1 = failures surface as typed halls
  overfitTendency: number;      // [0,1], 1 = grows guards readily
  stabilizingPower: number;     // signed; +ve reduces neighbor strain
  repairAffinity: number;       // [0,1], how often patches land here
  branchingCost: number;        // beam slots / extensions per call
  bytecodeTopology: "flat" | "motif-fused" | "header-heavy";
  repairTraces: TraceRegionId[]; // historical scar locations
};
```

## Semantic plaque

**Semantic plaque is repair history that has become material.**

The cycle:

```
repeated ambiguity
  -> repair traffic concentrates
    -> operator guards / special cases accumulate
      -> local constraint thickens
        -> nearby flow distorts to route around it
          -> new ambiguity arises at the rerouted edges
            -> repeated ambiguity (back to the top)
```

Useful local scar tissue (a guard that disambiguates dates inside URL
slugs, say) can, repeated enough, become **global brittleness**: the
operator graph grows dense around the scar, the path length of a
typical solve increases, and the system spends more of every run
maintaining the patches than doing the parse.

This is the cost the repo's anti-if-statement theme is reacting to.
Each new `if` is a thin layer of plaque. Sometimes necessary, often
load-bearing, but never free.

### Distinguishing healthy repair from plaque

Both reduce *local* error. Only one is good for the field.

| Healthy repair | Semantic plaque |
| --- | --- |
| relaxes strain | reduces local error while adding global strain |
| restores flow through the region | reroutes flow around the region |
| shortens future solver paths | lengthens future solver paths |
| reduces operator density at the site | increases operator density |
| reduces future repair pressure | increases future repair pressure |
| dissipates as the input distribution evolves | persists and ossifies |

A guard that survives because it keeps suppressing the same
hallucination on every run, while operator density and path length
around it climb, is plaque. A guard that fires once, fixes the strain,
and then stops being needed as upstream embeddings sharpen is healthy
repair.

## Maintenance operators

If repair history is material, then maintenance is an operator class,
not a one-shot cleanup. Names worth reserving:

- **`prune.dead_guards`** — remove guards whose firings have dropped to
  zero across the fixture set.
- **`compress.repair_chains`** — fuse a sequence of small narrowing
  repairs into a single sharper operator (the motif-fusion path, applied
  to scar tissue).
- **`dissolve.local_overfit`** — back off an operator that has grown
  fixture-specific, restoring a broader form.
- **`reroute.operator_flow`** — when plaque is bending future paths,
  open an alternative pathway and let beam search choose.
- **`merge.redundant_emitters`** — two emitters with overlapping
  `provides` and similar embeddings collapse into one.
- **`revascularize.trace_region`** — a region that has gone dark
  (low coverage, repeated `low_coverage_region` hallucinations) gets a
  proposed emitter from `FailurePressure.propose()`.

These are maintenance operators *because* the loss has locality and
memory. Without trace regions, none of them have a place to land.

## Anti-if-statement, restated as field dynamics

A large imperative branch hardcodes curvature. It says: at this point
in the pathway, regardless of the rest of the field, fork. That is a
fixed boundary baked into the operator pathway, and it cannot relax as
the surrounding evidence changes.

The healthier shape is for branching to **emerge** from local
interactions:

- operator **signatures** (`needs`/`provides`) gate by structure,
- operator **embeddings** gate by similarity to remaining AF needs,
- attractor **pressure** (which columns are still missing, which
  validators still failing) shapes which extensions survive,
- **traceable repair history** lets persistent strain propose new
  operators rather than new `if`s.

The `if` is a dam; signature + embedding + AF pressure is a riverbed.
Both route flow, but only one re-cuts itself as the water changes.

## Why fixture snapshots and trace regions matter

Once loss has **locality** and **memory**, history accumulates
spatially.

- A `TraceRegion` is the smallest unit of "this happened here, by this
  operator, on this span." Every emitter writes one per cell; every
  cell points back via `FieldHypothesis.traceRegionId`.
- Bounded fixture snapshots (`fixtures/`, `prepass/`) freeze the source
  manifold so that strain measured today is comparable to strain
  measured next week. Without a fixed manifold, every "improvement" is
  also a change of test, and regressions hide in the drift.

Together they give the system a body to inspect: scar locations,
repair-traffic heatmaps, regions that have gone dark, operators that
keep ending up as the patch site. Flow regression — Riordan-style or
otherwise — can grow on top of this without touching existing
operators, exactly because the geometry is already being recorded.

## What this changes about the seed (and what it doesn't)

Nothing in this document requires code changes today. The seed already
ships the load-bearing primitives:

- `Hallucination` as a sum type with `weight` (structured, not
  collapsed),
- `TraceRegion` (locality + memory),
- `FailurePressure.propose()` (the hook for plaque-driven mutation),
- bounded fixtures + monitor pre-pass (a stable body to inspect over
  time).

What the document does is name the shape these primitives are
participating in, so future work — material profiles, maintenance
operators, plaque metrics, flow regression — has somewhere to attach.
