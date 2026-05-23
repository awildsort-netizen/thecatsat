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
> prune?"
>
> "Why didn't we use the type system for required and optional? And
> what is `outputs`? These could be replaced by an operator signature
> type, no? Ideally one that TypeScript already does?"

The four user corrections above land at the same place from different
angles. This note describes where the design ended up and where the
type system can't go further.

## The principle

**The operator's run function is the operator's signature.**

TypeScript already encodes everything we need at compile time:

- **Required vs. optional inputs** — the `?` property modifier on the
  function's input parameter type.
- **Outputs** — the keys of the function's return type.
- **Channel value shapes** — the property types at those keys.

A `ParseOperator<I, O>` is essentially a typed function plus a tiny
runtime witness. There is no `OperatorSignature` type with `needs`,
`provides`, `tokens` arrays. There is no `OperatorIO` storage of those
arrays. There is no `defineOperator` factory. The value is its own
declaration:

```ts
export const regexEmitDate: ParseOperator<
  {
    "text.normalized": string;
    "spans.url":       readonly FieldHypothesis[];
    "trace.regions"?: readonly TraceRegion[];   // <-- TypeScript's `?`
  },
  {
    "spans.dated":   readonly FieldHypothesis[];
    "trace.regions": readonly TraceRegion[];
  }
> = {
  id: "regex.emit.date", cost: 2, tokens: [...],
  run: (_ctx, input) => { /* input typed by I, return by O */ },
  channels: {
    requiredInputs: ["text.normalized", "spans.url"],
    optionalInputs: ["trace.regions"],
    outputs:        ["spans.dated", "trace.regions"],
  },
};
```

TypeScript's contextual typing flows from the `ParseOperator<I, O>`
annotation into the run body's input parameter, into the return type,
and into the `channels` arrays (whose elements are typed by
`ChannelsOf<I, O>` — drawn from `RequiredKeys<I>`, `OptionalKeys<I>`,
and `keyof O`). The type system rejects any channel name that isn't
present in `I` or `O`. We verified this with `@ts-expect-error`
markers on a smoke test: foreign channel names in `channels.*` fail
to typecheck.

## The TypeScript-erasure boundary (the honest part)

TypeScript types are fully erased at runtime. The beam-search solver
needs to ask at runtime, "given the current channel-bag, which
operators are eligible to run next?" That requires `Object.keys`-style
access to the channel names — which means *some* value-level residue
of the type-level declaration has to live on the operator object.

The minimum residue is:

- `channels.requiredInputs: readonly string[]` — what gates eligibility.
- `channels.optionalInputs: readonly string[]` — typed for the run
  body, not eligibility (could be inferred at runtime as
  `output - required`, but listing it makes the documentation visible
  and is one line either way).
- `channels.outputs: readonly string[]` — what becomes available
  downstream.
- `tokens: readonly string[]` — the embedding signal; no type-level
  analogue.

These four arrays are the entire runtime residue. They are typed by
`ChannelsOf<I, O>` so the compiler enforces "every channel name here
must be a key of I or O" — the runtime witness cannot drift away from
the function type by accident.

What the type system **cannot** enforce compactly:

- **Exhaustiveness**: `channels.requiredInputs` could be `[]` even if
  `I` has required keys. Encoding "this readonly tuple must list every
  key of this union, with no duplicates" in TypeScript requires either
  recursive union-to-tuple tricks (unsound or fragile) or codegen. We
  do not attempt this. In practice the run body's typed access to
  `input["spans.url"]: readonly FieldHypothesis[]` makes an unlisted
  required key visible quickly: the solver will schedule the operator
  before `spans.url` is in scope, the run body will receive
  `undefined`, and existing tests will fail.

That is the documented boundary. We chose to live with it rather than
ship codegen.

## What this revision pruned vs. the previous one

The previous design carried a stored `op.io: OperatorIO` field with
arrays the solver read directly. Even that turned out to be
unnecessary: `op.channels` is the same data already typed via
`ChannelsOf<I, O>`, and the solver can call `signatureOf(op)` (a tiny
derivation function) when it wants the unified view. There is no
duplicate `op.io`.

Also pruned:

- `OperatorSignature` type (gone).
- `OperatorIO` type (gone).
- `toLegacySignature` adapter (gone).
- `ReflectedOperator` wrapper type (gone).
- `defineOperator` factory function (gone — declaring the value with
  a `ParseOperator<I, O>` annotation does all the work).
- `required<T>()` / `optional<T>()` value-marker helpers (gone — the
  `?` modifier on `I` does the same job, more idiomatically).

## What the solver / embedding read

```ts
import { signatureOf } from "./operator_reflection.js";

// In solver.ts:
signatureOf(op).requiredInputs   // for eligibility
signatureOf(op).outputs          // for availableChannels and saturation
op.tokens                        // for embedding (no view needed)
```

`signatureOf` is the only derivation. No stored `signature`/`io` field
to keep in sync with anything.

## Where this goes next

- **Python**. `inspect.signature`, `typing.get_type_hints`, and
  dataclass field introspection let Python do at runtime what
  TypeScript can only do at compile time. A Python `ParseOperator`
  can probably drop the `channels` witness entirely and derive it
  from the run function's annotations on demand. That is the obvious
  follow-up.
- **`materialHints` on `ProposedStaticOperator`** is still
  hand-authored. It's a content concept (embedding tokens harvested
  from the trace), not an IO concept; deriving it from a projection
  over the request and trace is a noted next step.
- **Bytecode disassembler views**. If something downstream wants a
  flat `{needs, provides}`-style summary for display, build it at the
  boundary via `signatureOf(op)` — not by storing it on the operator.
