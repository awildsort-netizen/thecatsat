# Signatures first — design note

> "I'm hoping we are still on point for not doing needs and requires
> explicitly: that should be provided by reflection already present in
> TypeScript and Python."

This note records the load-bearing principle behind
[`operator_reflection.ts`](../operator_reflection.ts), the changes to
[`operators.ts`](../operators.ts), and the new shape of
[`browser_oracle/types.ts`](../browser_oracle/types.ts).

## The principle

A `ParseOperator`'s `signature` (its `needs` and `provides`) is **not a
separate manual ontology**. It is a *projection* of the operator's IO
shape — what its `run` body reads from the channel bag and what it
writes back. The string keys at those reads and writes **are** the
channel names. The signature exists because the beam solver, the
embedding layer, and a future bytecode disassembler all want a small,
flat description of channels at a glance — but that description should
be derived from the implementation, not maintained alongside it.

When the signature is hand-authored:

- A `run` body that reads a new bag key (e.g. an accumulator channel)
  passes typecheck but is silently inaccurate in `signature.needs`.
- A `run` body that writes a new output drifts away from `provides`,
  so the solver underestimates the operator's reach and refuses to
  schedule downstream ops that depend on it.
- Two parts of the codebase (`operators.ts` and any proposal-style
  builder like `browser_oracle/distiller.ts`) keep their own copy of
  the rule for what counts as `needs` / `provides`, which is exactly
  how ontologies fork.

## How TypeScript reflects this

TypeScript has no runtime type information, but it has *exact* compile-
time information about the keys of an object literal. `defineOperator`
takes that compile-time guarantee and projects it down to runtime via
`Object.keys`:

```ts
defineOperator({
  id: "regex.emit.date",
  cost: 2,
  tokens: ["regex", "extract", "date", ...],

  // The keys here ARE the signature.needs.
  needs: {
    "text.normalized": CHANNEL as string,
    "spans.url":       CHANNEL as readonly FieldHypothesis[],
  },
  // The keys here ARE the signature.provides.
  outputs: {
    "spans.dated":   CHANNEL as readonly FieldHypothesis[],
    "trace.regions": CHANNEL as readonly TraceRegion[],
  },

  // The run body is typed by the same channel specs:
  //   input["text.normalized"]    is string | undefined
  //   input["spans.url"]          is readonly FieldHypothesis[] | undefined
  //   the returned object MUST be keyed exactly by "spans.dated" and "trace.regions"
  run: (_ctx, input) => { ... },
});
```

The signature is computed once at construction and frozen onto the
operator. A drift between the declared IO and the implementation now
fails typecheck rather than fails silently.

For genuine read-through channels (e.g. an accumulator the operator
also writes back), `reads` is a separate slot — it types the run body
without becoming a solver-eligibility need. Splitting `needs` and
`reads` was the small concession that made the principle tractable
without lying: every channel the run body touches is declared
somewhere, but only the ones that gate scheduling end up in
`signature.needs`.

## What this changes about `ProposedStaticOperator`

The browser-oracle distiller used to emit a proposal carrying explicit
`needs` and `provides` fields alongside `evidenceFields`. With the
principle stated above, that duplication is exactly what we wanted to
avoid:

- `provides` was always equal to `evidenceFields`. Keeping both
  invited drift.
- `needs: ["url"]` was a hand-wired constant that did not match what
  the lifted run-body actually consumes from the channel bag.

Now `ProposedStaticOperator` carries only the **authored material** the
distiller actually has: `evidenceFields`, `requestTemplate`,
`materialHints`, etc. The signature is derived at
`liftProposalToOperator` time via `defineOperator`, so a proposal
cannot disagree with the lifted operator's IO. The proposal stays a
useful artifact for bytecode-style search (it carries enough material
hints and request shape for downstream lookup), but it is no longer a
parallel ontology.

## Why this is "smallest coherent"

- One operator (`regex.emit.date`) is migrated to `defineOperator` as a
  proof of life and a working example. The other primitives keep their
  hand-authored signatures for now — they remain useful as a side-by-
  side comparison and can be migrated incrementally without forcing a
  big-bang refactor.
- `defineOperator` lives in its own small file
  ([`operator_reflection.ts`](../operator_reflection.ts)) and changes
  nothing about the existing `OperatorSignature` shape — the solver,
  the embedding layer, and the bytecode-genes module continue to read
  `signature.needs` / `signature.provides` exactly as they did.
- The `ProposedStaticOperator` change removes two fields (`needs`,
  `provides`) and leaves every consumer that read them either updated
  (the test asserts `evidenceFields` and the lifted signature
  separately) or unchanged (the bridge code only forwards proposals).

## Where this goes next

- Migrate the remaining primitives in `operators.ts` to
  `defineOperator`. Each one is a small mechanical edit; doing them
  one-at-a-time lets typecheck catch every signature drift that has
  accumulated.
- Mirror the same principle on the Python side
  (`composer.py`, `disassembly_matcher.py`, etc.) using Python's
  richer runtime reflection (`inspect.signature`, dataclass fields,
  `typing.get_type_hints`).
- Replace the `materialHints: readonly string[]` field on proposals
  with a derived projection over the request and trace, so even the
  embedding tokens grow out of the implementation rather than being
  hand-listed.
