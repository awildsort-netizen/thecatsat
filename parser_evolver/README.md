# parser_evolver — thecatsat as parser-evolver (TypeScript seed)

CSV is an attractor basin, not an output format. `parser_evolver` is a tiny
TypeScript prototype that evolves *parser-creatures* — short gene-strings of
operators — and asks the `CsvAF` climate which creatures fold a page into a
stable table without lying about the source.

This is a seed, deliberately small. See `docs/interpretation_sieve.md` and
`docs/tests_as_activation_factors.md` for the H7/thecatsat language used in
the comments here.

## Files

- `types.ts` — `ParseOperator`, `Gene`, `GeneString`, `CsvAF`, etc. Operators
  carry an `OperatorSignature` of needs/provides/tokens; the solver composes
  by signature, not by hand-wired conditionals.
- `embedding.ts` — symbolic operator embedding (token-bag cosine). Free
  polymorphism: relatives of needed work rank above strangers.
- `operators.ts` — four primitives: `normalize.whitespace`, `regex.emit`,
  `row.assemble.proximity`, `row.enforce.schema`. Every emitted cell carries
  its source `Span`.
- `af.ts` — `companyUpdatesAF`: required columns (date, title, url),
  cell-sourced constraint, run-level diversity bonus, heavy hallucination
  penalty.
- `solver.ts` — `makeBeamSolver({beam, maxLen})`. Beam search over
  gene-strings of length ≤ `maxLen`, extensions discovered by
  signature-driven eligibility.
- `demo.ts` — runs the solver over Blockchain.com-style sample text.
- `test.ts` — tiny assertion runner that the AF actually punishes lying
  creatures.

## Run

```
npx tsx parser_evolver/demo.ts
npx tsx parser_evolver/test.ts
```

## What this seed is *not* yet

- No motif fusion, no streamable-gene decoder, no concentration field. The
  bytecode here is flat. Distributions over genes are a natural next step.
- No GPU/shader pipeline. `RowKernel` in `types.ts` is the design hook for
  later lowering a heap of rows to a `kernel(row, col) -> cell` pass.
- No mutation/crossover yet — only enumerative/beam search. A mutation
  operator that swaps a gene for an embedding-near relative is the obvious
  first evolutionary step.
