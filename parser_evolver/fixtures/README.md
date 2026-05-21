# parser_evolver/fixtures — bounded snapshots + AF training CSV

These files are training fixtures for the `CsvAF` (company-updates) attractor
basin in `parser_evolver`. They are **not** a crawler and not a live data
source. The point is to give the AF a small, reviewable, provenance-rich set
of real bytes so future training/regression work has ground truth to fold
against.

## Contents

- `snapshots/` — raw HTML captured from a small set of public Blockchain.com
  pages plus one PRNewswire URL. Files are kept as fetched (no rewriting).
- `manifest.json` — per-snapshot provenance: source URL, fetch timestamp,
  HTTP status, content-type, byte size, sha256, and a `notes` field that
  records reachability quirks (SPA shell only, 404, etc.).
- `training.csv` — hand-labeled rows over those snapshots. Each row carries
  an `evidence_quote` that must appear verbatim in its `snapshot_path`, plus
  vocabulary-controlled `category` / `materiality_hint` / `field_effect_hint`
  / `expected_action` columns the AF can train against.
- `validate.ts` — small TypeScript validator (no external CSV dep) that
  checks columns, vocabulary, snapshot existence, evidence-quote presence,
  and `(url,title,observed_date)` uniqueness.

## Run the validator

```
cd parser_evolver
npm run validate-fixtures
# or, directly:
npx tsx fixtures/validate.ts
```

Exits non-zero on any issue.

## Refresh the snapshots

Snapshots are intentionally checked in — they are the unit of reproducibility.
To refresh, fetch each URL listed in `manifest.json` with a polite,
bounded-rate fetcher (the reference command used to seed this directory is
recorded in the PR description). After refetching:

1. Update `snapshot_path` files in place.
2. Update `fetch_ts`, `http_status`, `content_type`, `byte_size`, and
   `sha256` in `manifest.json`.
3. Re-check that every `evidence_quote` in `training.csv` still appears in
   its snapshot. The validator does this for you.

If a page becomes unreachable, **do not silently drop it** — keep the row in
the manifest with `reachable: false` (or `"shell_only"`) and a note. A
disappeared page is a real signal for AF training.

## What this is *not*

- Not an automated crawler. There is no scheduler, no follow-link, no
  pagination. Eight URLs were fetched once.
- Not wired into alerts. `expected_action` is a label for AF training, not
  a runtime hook.
- Not exhaustive. The CSV is small on purpose; broaden it through review,
  not bulk import.
