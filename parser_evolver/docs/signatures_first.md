# Signatures first — design note

> "I'm hoping we are still on point for not doing needs and requires
> explicitly: that should be provided by reflection already present in
> TypeScript and Python."

This note records the load-bearing principle behind
[`operator_reflection.ts`](../operator_reflection.ts), the changes to
[`operators.ts`](../operators.ts), and the new shape of
[`browser_oracle/types.ts`](../browser_oracle/types.ts).

## The deeper principle: host-language alignment

Actively seek alignment with what the host language already encodes.
TypeScript has a vocabulary for "required vs. optional property" — the
`?` modifier on object types. It has a vocabulary for "what does this
function consume and produce" — its parameter and return types. When
we invent a parallel ontology (`needs`, `requires`, `reads`, `provides`
as separate hand-authored string arrays alongside an implementation
that already encodes the same information), we duplicate what the type
system already knows. **Any duplication is a place for drift.**

Syntax sugar is not "just" sugar. The `?` on `{ traceRegions?: TraceRegion[] }`
isn't a shorthand for some other declaration; it *is* the declaration
that this property may be present or absent. By treating that single
declaration as authoritative, an entire layer of bookkeeping (a parallel
"reads vs. needs" split, a string array per operator, the rules for
keeping them in sync) collapses into nothing. The class of bug
"signature says needs=X but run body reads Y" stops being possible
because there is no second slot to disagree.

Python has stronger reflection (`inspect.signature`,
`typing.get_type_hints`, dataclass fields). The same principle applies:
when both languages already encode the answer, the answer should be
*projected* from the language, not duplicated alongside it.

## How TypeScript reflects this

A single `inputs` channel spec, declared once. Each channel's value is
a typed marker — either `required<T>()` (must be in scope before the
operator runs, gates solver eligibility) or `optional<T>()` (a
contextual read that may or may not be in scope). The same marker
drives two things at once:

1. The **type** of the run body's input parameter: a mapped type
   collapses required entries into normal properties and optional
   entries into `?`-properties. The `?` you see on
   `input["trace.regions"]` is the same `?` TypeScript uses for any
   optional property — it isn't a parser_evolver concept.
2. The **runtime projection**: `Object.keys(inputs)` partitioned by a
   tag-check is the eligibility set the solver uses to schedule the
   operator.

There is no second slot for either of those. The `?` is the ontology.

```ts
defineOperator({
  id: "regex.emit.date",
  cost: 2,
  tokens: ["regex", "extract", "date", ...],

  // A single declaration. Required vs. optional is the property
  // modifier — exactly the way TypeScript already says it.
  inputs: {
    "text.normalized": required<string>(),
    "spans.url":       required<readonly FieldHypothesis[]>(),
    // Read-through accumulator: may or may not be in scope. The `?`
    // on the typed input bag is derived from `optional<T>()`.
    "trace.regions":   optional<readonly TraceRegion[]>(),
  },
  outputs: {
    "spans.dated":   required<readonly FieldHypothesis[]>(),
    "trace.regions": required<readonly TraceRegion[]>(),
  },

  // input["text.normalized"] is `string`
  // input["spans.url"]       is `readonly FieldHypothesis[]`
  // input["trace.regions"]   is `readonly TraceRegion[] | undefined`
  run: (_ctx, input) => { ... },
});
```

A drift between the run body and the declared inputs is now a type
error: read a key that isn't declared and the index access fails;
declare a key and don't use it and the unused-symbol checker complains.
The signature can't get out of sync with the implementation because
they are the same declaration projected two ways.

## Legacy signature as an adapter, not first-class vocabulary

The existing solver, embedding layer, and bytecode disassembler all
consume the flat `OperatorSignature = { needs, provides, tokens }`
shape on `ParseOperator`. That shape is preserved for backward
compatibility, but it is now an **adapter output** rather than first-
class vocabulary:

- `toLegacySignature(requiredInputs, outputs, tokens)` is the one
  place that maps the reflected shape to the legacy shape.
- `defineOperator` calls it at construction. Downstream code reads
  `signature.needs` and `signature.provides` exactly as before; the
  values now come from a projection rather than from authored strings.
- The reflected partition (`requiredInputs`, `optionalInputs`,
  `outputs`) is also exposed on the returned operator as `reflected`
  so other code can query "what does this operator read optionally?"
  without re-parsing the `inputs` spec.

The legacy shape stays because the solver/embedding/disassembler use
it as a small flat description of channels at a glance — a useful
projection. It is no longer the source of truth.

## What this changes about `ProposedStaticOperator`

The browser-oracle distiller used to emit a proposal carrying explicit
`needs` and `provides` fields alongside `evidenceFields`. That was the
same duplication problem at a different layer:

- `provides` was always equal to `evidenceFields`. Keeping both
  invited drift.
- `needs: ["url"]` was a hand-wired constant that did not match what
  the lifted run body actually consumed from the channel bag.

The proposal now carries only the **authored material** the distiller
actually has — `evidenceFields`, `requestTemplate`, `materialHints`,
etc. — and `liftProposalToOperator` derives the legacy signature from
the lifted run body's typed `inputs`/`outputs` via `defineOperator`.
A proposal cannot disagree with the lifted operator's IO, because the
lifter is the only thing that decides what the IO is.

## Why this is the smallest coherent version of the principle

- One operator (`regex.emit.date`) is migrated to `defineOperator` as
  a working example. The other primitives keep their hand-authored
  signatures so the comparison stays visible side-by-side and the
  migration can proceed incrementally without a big-bang refactor.
- `defineOperator` lives in its own file and adds nothing to the
  existing `OperatorSignature` shape — the solver, embedding layer,
  and bytecode-genes module continue to read `signature.needs` /
  `signature.provides` exactly as they did. They are consumers of
  the legacy adapter; they do not need to know how it is produced.
- No new vocabulary leaks into public-facing types. The proposal type
  loses two fields. The reflected partition is available but optional;
  callers that don't care can keep treating an operator as a
  `ParseOperator`.

## Where this goes next

- Migrate the remaining primitives in `operators.ts`. Each one is a
  small mechanical edit and is likely to expose the same kind of
  silent drift that turned up on `regex.emit.date` (its hand-authored
  `needs` did not list `trace.regions`, even though the run body read
  it). The migration honestly declares those reads as optional and
  the inaccuracy goes away.
- Apply the same principle on the Python side. Python's runtime
  reflection is stronger — `inspect.signature`,
  `typing.get_type_hints`, and dataclass fields can probably eliminate
  even the `required<T>()` / `optional<T>()` helpers and read directly
  from function annotations.
- Reduce `materialHints: readonly string[]` on browser-oracle
  proposals to a derived projection over the request and trace, so
  the embedding tokens too grow out of the implementation rather
  than being hand-listed.
