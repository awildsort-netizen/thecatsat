# parser_evolver/prepass — structured pre-pass for the Blockchain.com monitor

A monitor pre-pass that takes the bounded fixture snapshots (see
`../fixtures/`) and runs them through `parser_evolver`'s CsvAF + beam
solver to produce a small, structured candidate digest. The digest is
what a *future* recurring Blockchain.com monitor could read instead of
re-extracting from raw HTML on every tick. **This module is not wired
into any live scheduled task and does no network I/O.**

## Why a pre-pass

The recurring monitor today reasons over fetched pages directly. That is
expensive when most ticks see the same surface: a status page that says
"all systems operational", a legal page whose `Last Updated:` date has
not changed, a blog index that is still a hydrated SPA shell. A pre-pass
lets `parser_evolver` do the cheap structural part once per snapshot,
emit:

  - the controlled-vocabulary cells the AF actually backs up;
  - a `confidence` derived from solver score + evidence presence +
    reachability (not a free-text hint);
  - a typed `hallucination_kinds` summary so the monitor sees *why* the
    pre-pass is unsure;
  - a `trace_region_count` — the dataflow hook for later flow-regression;
  - and an `expected_action` that has been honestly *escalated* when the
    snapshot was a SPA shell or a 404.

The monitor can then skip deeper reasoning on `monitor` / `archive` /
`ingest-as-fixture-only` rows and focus its cycles on the
`flag-for-review` / `needs-rendered-fetch` ones — the cases where the
parser-creature did not see a stable table.

## The contract

A pre-pass digest row is structurally:

```
source_id, source_type, url, observed_date, title,
category, materiality_hint, field_effect_hint,
expected_action, confidence, hallucination_kinds,
trace_region_count, evidence_quote, snapshot_path
```

The validator (`validate-digest.ts`) enforces:

  1. controlled vocabulary on `source_type`, `category`,
     `materiality_hint`, `field_effect_hint`, `expected_action`;
  2. `confidence` in `[0,1]`, `trace_region_count` a non-negative
     integer;
  3. `hallucination_kinds` is a JSON object whose keys are known
     `HallucinationKind`s and whose values are non-negative integers;
  4. `evidence_quote` (when non-empty) appears verbatim in the snapshot
     — either in the same HTML-stripped view that the pre-pass uses or
     in the raw bytes (legal pages tend to inline the `Last Updated:`
     date in markup that survives strip);
  5. **high-materiality rows require either an official source
     (`legal` / `status-page` / `press-release`) or an escalation
     action (`flag-for-review` / `needs-rendered-fetch`).** This is
     the rule that prevents the pre-pass from silently treating a
     marketing page as material.

## Degraded-source semantics

Two negative examples are first-class in the fixture set:

  - `blog-ipo-announce` — the IPO post URL returns the *same SPA shell*
    as the blog index. The manifest records this as
    `reachable: "shell_only"`. The pre-pass overrides the training
    `expected_action` to `needs-rendered-fetch`, lowers confidence,
    and refuses to claim extraction succeeded on the post body.
  - `prnewswire-ipo` — the URL returned 404. Manifest records
    `reachable: false`. Pre-pass overrides to `flag-for-review`.

This is the *spirit* of the prototype: the CSV is the attractor basin,
parser-creatures are selected for *seeing the page as a stable table
without lying*, and the pre-pass exposes the difference between "I see
this" and "I can't see this" rather than papering over it.

## Run

```bash
cd parser_evolver
npm run prepass             # stdout digest (CSV)
npm run prepass -- --write  # also write fixtures/digest.csv + digest.json
npm run validate-digest     # validate the digest CSV against snapshots
npm run prepass-test        # 19 assertions over the bounded fixture
npm run all                 # everything (check, test, demo,
                            #            validate-fixtures,
                            #            prepass-test, validate-digest)
```

## What a future monitor would do

In rough terms:

```
on tick:
  for each source in monitored_set:
    snapshot = fetch(source)           # bounded, polite, like fixtures/
    update manifest entry              # http_status, sha256, reachable
  digest = runPrepass()                # this module, deterministic
  for row in digest:
    if row.expected_action in { monitor, archive, ingest-as-fixture-only, no-op }:
      continue                         # cheap path — pre-pass is enough
    if row.expected_action == needs-rendered-fetch:
      rendered = headless_render(row.url)
      maybe_alert(rendered)
    if row.expected_action == flag-for-review:
      enqueue_for_human(row)
```

The deeper LLM reasoning the monitor uses today only runs on the small
set of rows the pre-pass could not extract confidently. That's the
saving: the pre-pass is the *cheap structural pre-filter* before deeper
search or reasoning.

## Files

- `types.ts` — `DigestRow` plus the controlled vocabularies (mirrors
  `fixtures/validate.ts` and adds `needs-rendered-fetch` for honest
  escalation of SPA-shell sources).
- `prepass.ts` — reads `fixtures/manifest.json` + `fixtures/training.csv`,
  strips each snapshot to visible text, runs the parser_evolver beam
  solver against `companyUpdatesAF`, then folds solver score, evidence
  presence, and manifest reachability into the digest row.
- `run.ts` — CLI; writes to stdout and (with `--write`) to
  `fixtures/digest.csv` + `fixtures/digest.json`.
- `validate-digest.ts` — validator + library. Importable
  (`check`, `toRows`) and runnable as a CLI.
- `test.ts` — 19 assertions including: vocabulary coverage; the IPO
  shell-only and 404 escalation rules; the high-materiality contract;
  evidence backing; that emitted CSV round-trips through the validator.

## What this is *not*

- Not a crawler. No fetching, no scheduling, no follow-link, no
  pagination. Reads only the bounded fixture set.
- Not wired into the scheduled Perplexity monitor. The monitor is
  untouched; this is the shape of what *could* feed it later.
- Not a finished extractor. `parser_evolver` only assembles a
  date/title/url row for pages that *contain* those inline; for SPA
  shells it correctly reports `low_coverage_region` and the pre-pass
  escalates rather than pretends.
