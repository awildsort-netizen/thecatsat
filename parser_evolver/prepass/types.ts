// Digest row produced by the monitor pre-pass.
//
// One row per (snapshot, training-row) pair — the pre-pass folds the
// hand-labeled training CSV against the snapshot text using parser_evolver
// operators and emits a row that downstream monitor code could consume
// without re-doing the extraction. Vocabulary mirrors training.csv (the
// CSV is the attractor basin), with two additions surfaced by the
// pre-pass itself:
//
//   - `confidence` — derived from solver score + evidence presence, not
//     a free-text hint;
//   - `hallucination_kinds` — typed pressure summary from CsvAF, so a
//     downstream consumer sees *why* the pre-pass is unsure;
//   - `trace_region_count` — how many TraceRegions accumulated during
//     decompression, the data hook for later flow-regression.
//
// A pre-pass row is not an alert. It is a structured candidate for a
// future monitor to read instead of re-parsing the HTML.

export const SOURCE_TYPE = [
  "status-page",
  "blog-index",
  "blog-post",
  "legal",
  "press-release",
  "marketing",
] as const;
export type SourceType = (typeof SOURCE_TYPE)[number];

export const CATEGORY = [
  "incident",
  "maintenance",
  "policy-update",
  "corporate-announcement",
  "product-update",
  "marketing",
  "not-found",
] as const;
export type Category = (typeof CATEGORY)[number];

export const MATERIALITY = ["none", "low", "medium", "high"] as const;
export type Materiality = (typeof MATERIALITY)[number];

export const FIELD_EFFECT = [
  "none",
  "availability",
  "liquidity",
  "regulatory",
  "legal-terms",
] as const;
export type FieldEffect = (typeof FIELD_EFFECT)[number];

// The pre-pass *adds* two non-training actions on top of the training
// vocabulary so it can honestly say "I couldn't extract; let a renderer
// or human take it from here" instead of fabricating extraction.
export const EXPECTED_ACTION = [
  "monitor",
  "archive",
  "flag-for-review",
  "ingest-as-fixture-only",
  "no-op",
  "needs-rendered-fetch",
] as const;
export type ExpectedAction = (typeof EXPECTED_ACTION)[number];

export type HallucinationKindCount = Readonly<Record<string, number>>;

export type DigestRow = {
  readonly source_id: string;
  readonly source_type: SourceType;
  readonly url: string;
  readonly observed_date: string;
  readonly title: string;
  readonly category: Category;
  readonly materiality_hint: Materiality;
  readonly field_effect_hint: FieldEffect;
  readonly expected_action: ExpectedAction;
  readonly confidence: number;
  readonly hallucination_kinds: HallucinationKindCount;
  readonly trace_region_count: number;
  readonly evidence_quote: string;
  readonly snapshot_path: string;
};

export const DIGEST_COLUMNS = [
  "source_id",
  "source_type",
  "url",
  "observed_date",
  "title",
  "category",
  "materiality_hint",
  "field_effect_hint",
  "expected_action",
  "confidence",
  "hallucination_kinds",
  "trace_region_count",
  "evidence_quote",
  "snapshot_path",
] as const;
export type DigestColumn = (typeof DIGEST_COLUMNS)[number];
