# Signatures first — design note

> "I'm hoping we are still on point for not doing needs and requires
> explicitly: that should be provided by reflection already present in
> TypeScript and Python."
>
> "Why do we need `needs` / `reads`? Actively seek alignment with the
> host language. TypeScript already has `?` for optional properties;
> syntax sugar is not just sugar — it collapses entire classes of
> redundant manual ontology."
>
> "What is with the urge to keep 'legacy' shit for this — why not just
> prune? You're accumulating bullshit during prototyping."

This note records the load-bearing principles behind
[`operator_reflection.ts`](../operator_reflection.ts), the new shape of
`ParseOperator.io`, and the migration of every primitive operator to
`defineOperator`.

## Two principles, one direction

**Host-language alignment.** TypeScript already has a vocabulary for
"required vs. optional property" — the `?` modifier on object types.
It has a vocabulary for "what does this function consume and produce"
— its parameter and return types. When we invent a parallel ontology
(`needs`, `requires`, `reads`, `provides` as separate hand-authored
string arrays alongside an implementation that already encodes the
same information), we duplicate what the type system already knows.
Any duplication is a place for drift. Syntax sugar is not "just" sugar:
the `?` on `{ traceRegions?: TraceRegion[] }` isn't a shorthand for some
other declaration; it *is* the declaration. By treating that single
declaration as authoritative, an entire layer of bookkeeping collapses
into nothing.

**Prototype pruning discipline.** While the system is young, redundant
scaffolds should be deleted aggressively. Compatibility adapters are
semantic plaque unless they protect a real external boundary (a
shipped public API, an over-the-wire protocol, an on-disk format with
existing consumers). None of those apply here: `parser_evolver` is
prototype clay and every "legacy" path we keep is a place future-us
has to defend twice. Prune.

## How the language reflects the principle

A single `inputs` channel spec, declared once. Each channel's value is
a typed marker — either `required<T>()` (must be in scope before the
operator runs, gates solver eligibility) or `optional<T>()` (a
contextual read that may or may not be in scope). The same marker
drives two things at once:

1. The **type** of the run body's input parameter: a mapped type
   collapses required entries into normal properties and optional
   entries into `?`-properties. The `?` you see on
   `input["trace.regions"]` is the same `?` TypeScript uses for any
   optional property.
2. The **runtime `io` record**: `Object.keys(inputs)` partitioned by a
   tag-check yields `requiredInputs` and `optionalInputs`;
   `Object.keys(outputs)` yields `outputs`. The solver reads these
   directly.

There is no second slot for either. The `?` is the ontology.

```ts
defineOperator({
  id: "regex.emit.date",
  cost: 2,
  tokens: ["regex", "extract", "date", ...],

  inputs: {
    "text.normalized": required<string>(),
    "spans.url":       required<readonly FieldHypothesis[]>(),
    // Optional read: the `?` on the typed input bag IS the projection.
    "trace.regions":   optional<readonly TraceRegion[]>(),
  },
  outputs: {
    "spans.dated":   required<readonly FieldHypothesis[]>(),
    "trace.regions": required<readonly TraceRegion[]>(),
  },

  // input["text.normalized"] : string
  // input["spans.url"]       : readonly FieldHypothesis[]
  // input["trace.regions"]   : readonly TraceRegion[] | undefined
  run: (_ctx, input) => { ... },
});
```

A drift between the run body and the declared inputs is now a type
error. The signature can't get out of sync with the implementation
because they are the same declaration projected two ways.

## What was pruned

- **`OperatorSignature` is gone.** `ParseOperator.io: OperatorIO` is
  the single shape every consumer reads:
  `{ requiredInputs, optionalInputs, outputs, tokens }`. The solver
  and the embedding layer were updated to read this directly. No
  adapter, no `toLegacySignature` function, no `signature.needs` field.
- **All five primitives migrated** to `defineOperator`. The previous
  PR migrated only `regex.emit.date`; the rest were left as a
  "side-by-side comparison" — which is exactly the kind of bullshit
  prototype clay shouldn't accumulate. The migration surfaced the
  same `trace.regions` drift on `regex.emit.url` and `regex.emit.title`
  (their hand-authored `needs` claimed `["text.normalized"]` and
  `["text.normalized", "spans.url"]` but their run bodies also read
  `bag["trace.regions"]`); honestly declared as `optional<...>` now.
- **`ProposedStaticOperator.needs` / `.provides` are gone.** The
  proposal carries only authored material (`evidenceFields`,
  `requestTemplate`, `materialHints`, `tokens`, `cost`). The lifter
  reflects the operator's `io` from the run body's typed channel spec
  via `defineOperator`.

## What this means for the solver / embedding / disassembler

They read `op.io.requiredInputs` for eligibility, `op.io.outputs` for
channel availability, `op.io.tokens` for embedding. That's all. The
indirection through a separate `signature` object — useful only when
the same info was authored elsewhere and might disagree — went away
because the disagreement isn't possible anymore.

## Where this goes next

- Apply the same principle on the Python side using `inspect.signature`,
  `typing.get_type_hints`, and dataclass fields. Python's runtime
  reflection is stronger and can probably eliminate even the
  `required<T>()` / `optional<T>()` helpers (they're TypeScript's
  price for not having parameter introspection at runtime).
- Reduce `materialHints: readonly string[]` on browser-oracle
  proposals to a derived projection over the request and trace, so
  embedding tokens too grow out of the implementation rather than
  being hand-listed.
- If a downstream consumer ever needs the old `{needs, provides}`
  shape (a CLI dump, a bytecode disassembler), build it at the
  boundary — not on `ParseOperator` itself.
