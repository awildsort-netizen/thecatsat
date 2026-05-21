# parser_evolver — thecatsat as parser-evolver (TypeScript seed)

CSV is an attractor basin, not an output format. `parser_evolver` is a tiny
TypeScript prototype that evolves *parser-creatures* — short gene-strings of
operators — and asks the `CsvAF` climate which creatures fold a page into a
stable table without lying about the source.

This is a seed, deliberately small. See `docs/interpretation_sieve.md` and
`docs/tests_as_activation_factors.md` for the H7/thecatsat language used in
the comments here.

## Files

- `types.ts` — `ParseOperator` (needs/provides/embedding-tokens),
  `Gene`/`GeneString` (typed bytecode), `CsvAF`, and the first-class typed
  hallucination artifacts: `Hallucination`, `HallucinationKind`,
  `TraceRegion`, `FailurePressure`, plus a `RowKernel` shader design hook.
- `embedding.ts` — symbolic operator embedding (token-bag cosine). Used by
  the solver to prune extensions by similarity to *remaining* AF needs.
- `operators.ts` — five primitives plus an AF-bound enforcer:
  `normalize.whitespace`, `regex.emit.url`, `regex.emit.date` (excludes
  dates inside URL slugs), `regex.emit.title` (excludes date-like and
  url-like lines), `row.assemble.proximity` (per-date forward window),
  and `makeEnforceSchema(af)` which filters rows by the AF's
  `rowConstraints` and emits typed `validator_rejection` hallucinations.
  Regex spans come from the `d` (indices) flag, not `indexOf`.
- `af.ts` — `companyUpdatesAF`: required `date`/`title`, optional `url`,
  typed `hallucinations(rows)` that detect `unsupported_cell`,
  `validator_rejection`, `field_role_confusion`, `missing_emitter`,
  `low_coverage_region`, `overfit_pattern`.
- `solver.ts` — `makeBeamSolver({beam, maxLen, extensionTopK})`. Beam
  search over gene-strings; extensions are pruned by cosine to
  *remaining* AF needs (operators whose provides are saturated drop to
  near-zero); ties on total score break by last-gene similarity;
  candidates are deduped by gene-string identity.
- `demo.ts` — runs the solver over Blockchain.com-style sample text.
- `test.ts` — 14 assertions covering slice equality, validator round-trip,
  role-misassignment guards, URL cross-bleed, fabricated row penalty,
  typed-hallucination kinds, embedding-driven (topK=1) discovery, and
  TraceRegion consistency.

## Run

```
cd parser_evolver
npm run check              # tsc --noEmit
npm run test               # tsx test.ts
npm run demo               # tsx demo.ts
npm run validate-fixtures  # tsx fixtures/validate.ts
npm run all                # all of the above
```

## Training data

`fixtures/` holds a bounded snapshot set (real HTML from a small list of
public Blockchain.com pages plus one PRNewswire URL) and a hand-labeled
`training.csv` that maps each row to an `evidence_quote` present verbatim in
its snapshot. The validator (`npm run validate-fixtures`) re-checks columns,
vocabulary, snapshot existence, and evidence-quote presence so the CSV
cannot silently drift from the snapshots. See `fixtures/README.md` for the
refresh procedure and the note that snapshots are fixtures, not crawler
output.

## Typed hallucinations

`Hallucination` is a sum type. Each kind has a `weight` that the AF folds
into `scoreRun`. Kinds today:

| Kind | When it fires |
| --- | --- |
| `unsupported_cell` | cell carries no usable source span |
| `misassigned_span` | (reserved) span belongs to another field's region |
| `field_role_confusion` | value validates as field A but was emitted as B |
| `missing_emitter` | required column has no emitter contributing |
| `validator_rejection` | value present but failed column validator |
| `low_coverage_region` | no rows assembled |
| `overfit_pattern` | a field emitted >3× rows count |

`FailurePressure` is the type-level hook for "persistent hallucinations
propose new operators" — `propose()` is intentionally undefined here so a
future mutation operator can fill it in.

`TraceRegion` is the minimum data hook for later flow-regression passes:
every emitter writes one per cell, the cell points back via
`FieldHypothesis.traceRegionId`, and the candidate carries the full
`traces` list. Riordan-style flow regression can grow on top of this
without touching existing operators.

## What this seed is *not* yet

- No motif fusion / streamable gene decoder; bytecode is flat.
- No mutation/crossover. The obvious first move is an embedding-near
  gene swap, then a hallucination-driven `propose()` that asks for new
  emitters when a kind persists.
- No GPU/shader pipeline. `RowKernel` is only the design hook.
- `misassigned_span` is reserved but not yet detected.
