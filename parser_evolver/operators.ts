// Primitive parse operators.
//
// Each operator is a tendency: it declares what channels it needs and what
// channels it provides, and exposes embedding tokens so the solver can
// substitute relatives without conditional branching.
//
// Channels (provides/needs):
//   "text.normalized"       — whitespace/markup-flattened body
//   "spans.url"             — URL spans (kept first; date detector excludes them)
//   "spans.dated"           — date spans NOT contained in any URL span
//   "spans.titled"          — title-like lines that are not dates and not urls
//   "rows.assembled"        — RowHypotheses built by proximity assembly
//   "rows.validated"        — RowHypotheses that satisfied all AF rowConstraints
//   "trace.regions"         — TraceRegion artifacts accumulated across operators
//
// Every emitter records a TraceRegion alongside the cell. Validators or
// downstream operators can read those regions; a validator that rejects a
// cell can also write a typed `Hallucination` (see types.ts) into the bag
// under "hallucinations.collected", which the AF reads at scoreRun.

import type {
  CsvAF,
  FieldHypothesis,
  Hallucination,
  ParseContext,
  ParseOperator,
  RowHypothesis,
  Span,
  TraceRegion,
} from "./types.js";
import { CHANNEL, defineOperator } from "./operator_reflection.js";

// ---------------------------------------------------------------------------
// Small helpers — pure, no for-loops.
// ---------------------------------------------------------------------------

const collapseWhitespace = (s: string): string =>
  s.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();

const spanContains = (outer: Span, inner: Span): boolean =>
  inner[0] >= outer[0] && inner[1] <= outer[1];

const containedInAny = (s: Span, outers: readonly Span[]): boolean =>
  outers.some((o) => spanContains(o, s));

// Region IDs are stable per (operator, span, channel) so traces are
// idempotent and cells can reference them by id.
const traceId = (operator: string, span: Span, channel: string): string =>
  `${operator}@${channel}#${span[0]}:${span[1]}`;

const region = (operator: string, channel: string, span: Span, label: string): TraceRegion => ({
  id: traceId(operator, span, channel),
  label,
  span,
  channel,
  operator,
});

// Regex helpers — uses the `d` flag so we get capture-group indices instead
// of fragile indexOf math.
type EmitParams = {
  readonly field: string;
  readonly pattern: string;
  readonly group: number; // which capture group is the cell value (0 for whole match)
  readonly confidence: number;
  readonly channel: string;
  readonly operatorId: string;
};

type MatchWithIndices = RegExpMatchArray & { indices?: ReadonlyArray<readonly [number, number] | undefined> };

const matchSpan = (m: MatchWithIndices, group: number): Span | undefined => {
  const idx = m.indices?.[group];
  return idx === undefined ? undefined : ([idx[0], idx[1]] as const);
};

const emitFrom = (text: string, p: EmitParams): readonly FieldHypothesis[] => {
  const re = new RegExp(p.pattern, "gd");
  const matches = Array.from(text.matchAll(re) as Iterable<MatchWithIndices>);
  const lifted: readonly (FieldHypothesis | undefined)[] = matches.map((m) => {
    const span = matchSpan(m, p.group);
    const value = m[p.group];
    if (span === undefined || value === undefined) return undefined;
    const fh: FieldHypothesis = {
      field: p.field,
      value: value.trim(),
      span,
      operator: p.operatorId,
      confidence: p.confidence,
      evidence: `regex:${p.pattern.slice(0, 28)}`,
      traceRegionId: traceId(p.operatorId, span, p.channel),
    };
    return fh;
  });
  return lifted.filter((x): x is FieldHypothesis => x !== undefined);
};

const tracesFor = (hits: readonly FieldHypothesis[], channel: string, operatorId: string, label: string): readonly TraceRegion[] =>
  hits.map((h) => region(operatorId, channel, h.span, `${label}:${h.value.slice(0, 20)}`));

// ---------------------------------------------------------------------------
// Operator 1 — whitespace normalization.
// ---------------------------------------------------------------------------

export const normalizeWhitespace: ParseOperator = {
  id: "normalize.whitespace",
  cost: 1,
  signature: {
    needs: [],
    provides: ["text.normalized"],
    tokens: ["normalize", "whitespace", "text", "flatten", "clean", "prep"],
  },
  run: (ctx) => ({ "text.normalized": collapseWhitespace(ctx.normalizedText) }),
};

// ---------------------------------------------------------------------------
// Operator 2 — URL emitter. Goes first so date emitter can exclude URL ranges.
// ---------------------------------------------------------------------------

const URL_PARAMS: EmitParams = {
  field: "url",
  pattern: "https?:\\/\\/[^\\s)]+",
  group: 0,
  confidence: 0.95,
  channel: "spans.url",
  operatorId: "regex.emit.url",
};

export const regexEmitUrl: ParseOperator = {
  id: "regex.emit.url",
  cost: 2,
  signature: {
    needs: ["text.normalized"],
    provides: ["spans.url", "trace.regions"],
    tokens: ["regex", "extract", "url", "link", "href", "address"],
  },
  run: (_ctx, input) => {
    const bag = input as Record<string, unknown>;
    const text = (bag["text.normalized"] as string | undefined) ?? "";
    const hits = emitFrom(text, URL_PARAMS);
    return {
      "spans.url": hits,
      "trace.regions": [
        ...((bag["trace.regions"] as readonly TraceRegion[] | undefined) ?? []),
        ...tracesFor(hits, URL_PARAMS.channel, URL_PARAMS.operatorId, "url"),
      ],
    };
  },
};

// ---------------------------------------------------------------------------
// Operator 3 — date emitter. Excludes dates that fall inside any URL span;
// those are slug fragments, not date fields.
// ---------------------------------------------------------------------------

const DATE_PARAMS: EmitParams = {
  field: "date",
  pattern: "(?:\\d{4}-\\d{2}-\\d{2})|(?:[A-Z][a-z]+\\s+\\d{1,2},\\s+\\d{4})",
  group: 0,
  confidence: 0.9,
  channel: "spans.dated",
  operatorId: "regex.emit.date",
};

// regexEmitDate is built via `defineOperator` so that `signature.needs`
// and `signature.provides` are derived from the typed `needs`/`outputs`
// channel specs below — the implementation cannot drift from the
// declared signature, because the same keys flow into both. Other
// primitives keep their hand-authored signatures for now; they remain
// useful as a comparison and can be migrated incrementally.
export const regexEmitDate: ParseOperator = defineOperator({
  id: "regex.emit.date",
  cost: 2,
  tokens: ["regex", "extract", "date", "iso", "calendar", "month"],
  needs: {
    "text.normalized": CHANNEL as string,
    "spans.url": CHANNEL as readonly FieldHypothesis[],
  },
  reads: {
    // Read-through accumulator: this operator extends prior regions
    // rather than replacing them. Listed as `reads` so it is typed for
    // the run body without becoming a solver-eligibility need.
    "trace.regions": CHANNEL as readonly TraceRegion[],
  },
  outputs: {
    "spans.dated": CHANNEL as readonly FieldHypothesis[],
    "trace.regions": CHANNEL as readonly TraceRegion[],
  },
  run: (_ctx, input) => {
    const text = input["text.normalized"] ?? "";
    const urlSpans = (input["spans.url"] ?? []).map((u) => u.span);
    const raw = emitFrom(text, DATE_PARAMS);
    const hits = raw.filter((h) => !containedInAny(h.span, urlSpans));
    return {
      "spans.dated": hits,
      "trace.regions": [
        ...(input["trace.regions"] ?? []),
        ...tracesFor(hits, DATE_PARAMS.channel, DATE_PARAMS.operatorId, "date"),
      ],
    };
  },
});

// ---------------------------------------------------------------------------
// Operator 4 — title emitter. Excludes lines that look like a date or a URL,
// or that sit inside a URL span. Anchored on a leading capital letter.
// ---------------------------------------------------------------------------

// Pattern picks up the line body in capture group 1 so /d gives us the
// title's own span, not the leading newline.
const TITLE_PARAMS: EmitParams = {
  field: "title",
  pattern: "(?:^|\\n)([A-Z][^\\n]{9,139})(?=\\n|$)",
  group: 1,
  confidence: 0.6,
  channel: "spans.titled",
  operatorId: "regex.emit.title",
};

const looksLikeDate = (v: string): boolean =>
  /^\d{4}-\d{2}-\d{2}$/.test(v) || /^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$/.test(v);
const looksLikeUrl = (v: string): boolean => /^https?:\/\//.test(v);

export const regexEmitTitle: ParseOperator = {
  id: "regex.emit.title",
  cost: 2,
  signature: {
    needs: ["text.normalized", "spans.url"],
    provides: ["spans.titled", "trace.regions"],
    tokens: ["regex", "extract", "title", "headline", "heading", "name"],
  },
  run: (_ctx, input) => {
    const bag = input as Record<string, unknown>;
    const text = (bag["text.normalized"] as string | undefined) ?? "";
    const urlSpans = ((bag["spans.url"] as readonly FieldHypothesis[] | undefined) ?? []).map((u) => u.span);
    const raw = emitFrom(text, TITLE_PARAMS);
    const hits = raw.filter(
      (h) => !looksLikeDate(h.value) && !looksLikeUrl(h.value) && !containedInAny(h.span, urlSpans),
    );
    return {
      "spans.titled": hits,
      "trace.regions": [
        ...((bag["trace.regions"] as readonly TraceRegion[] | undefined) ?? []),
        ...tracesFor(hits, TITLE_PARAMS.channel, TITLE_PARAMS.operatorId, "title"),
      ],
    };
  },
};

// ---------------------------------------------------------------------------
// Operator 5 — proximity assembly.
//
// A row exists where a date sits near a title. The URL is picked from URLs
// that come *after* the date and before the next date — a forward-window
// pick avoids the cross-bleed where the previous incident's URL bleeds into
// the next row.
// ---------------------------------------------------------------------------

const distance = (a: Span, b: Span): number =>
  Math.min(Math.abs(a[0] - b[1]), Math.abs(b[0] - a[1]));

const nearest = <T extends { readonly span: Span }>(anchor: Span, items: readonly T[]): T | undefined =>
  items
    .map((it) => ({ it, d: distance(anchor, it.span) }))
    .reduce<{ it?: T; d: number }>((best, cur) => (cur.d < best.d ? cur : best), { d: Infinity }).it;

const buildRow = (
  date: FieldHypothesis,
  title: FieldHypothesis | undefined,
  url: FieldHypothesis | undefined,
): RowHypothesis => {
  const fields: Record<string, FieldHypothesis> = { date };
  title && (fields.title = title);
  url && (fields.url = url);
  const present = Object.values(fields);
  const avgConf = present.reduce((s, f) => s + f.confidence, 0) / present.length;
  return { fields, score: avgConf };
};

export const proximityAssemble: ParseOperator = {
  id: "row.assemble.proximity",
  cost: 3,
  signature: {
    needs: ["spans.dated", "spans.titled"],
    provides: ["rows.assembled"],
    tokens: ["assemble", "row", "proximity", "segment", "group", "near", "anchor", "window"],
  },
  run: (_ctx, input) => {
    const bag = input as Record<string, unknown>;
    const dates = [...((bag["spans.dated"] as readonly FieldHypothesis[] | undefined) ?? [])].sort(
      (a, b) => a.span[0] - b.span[0],
    );
    const titles = (bag["spans.titled"] as readonly FieldHypothesis[] | undefined) ?? [];
    const urls = (bag["spans.url"] as readonly FieldHypothesis[] | undefined) ?? [];

    const rows = dates.map((d, i) => {
      const next = dates[i + 1]?.span[0] ?? Number.MAX_SAFE_INTEGER;
      const windowUrls = urls.filter((u) => u.span[0] >= d.span[1] && u.span[1] <= next);
      const windowTitles = titles.filter((t) => t.span[0] >= d.span[1] && t.span[1] <= next);
      return buildRow(d, nearest(d.span, windowTitles) ?? nearest(d.span, titles), windowUrls[0]);
    });
    return { "rows.assembled": rows };
  },
};

// ---------------------------------------------------------------------------
// Operator 6 — schema enforcement / validator pass.
//
// Drops rows that fail any AF rowConstraint. The AF reference flows in via
// the params bag so the operator stays a pure ParseOperator. Rejected rows
// are converted to typed `Hallucination`s under "hallucinations.collected".
// ---------------------------------------------------------------------------

export const makeEnforceSchema = (af: CsvAF): ParseOperator => ({
  id: "row.enforce.schema",
  cost: 1,
  signature: {
    needs: ["rows.assembled"],
    provides: ["rows.validated", "hallucinations.collected"],
    tokens: ["enforce", "schema", "validate", "filter", "required", "columns", "reject"],
  },
  run: (_ctx, input) => {
    const bag = input as Record<string, unknown>;
    const rows = (bag["rows.assembled"] as readonly RowHypothesis[] | undefined) ?? [];
    const kept = rows.filter((r) => af.rowConstraints.every((c) => c(r)));
    const rejections: readonly Hallucination[] = rows
      .filter((r) => !af.rowConstraints.every((c) => c(r)))
      .map<Hallucination>((r) => ({
        kind: "validator_rejection",
        operator: "row.enforce.schema",
        weight: 1,
        note: `row failed rowConstraints; fields=${Object.keys(r.fields).join(",")}`,
      }));
    const prior = (bag["hallucinations.collected"] as readonly Hallucination[] | undefined) ?? [];
    return {
      "rows.validated": kept,
      "hallucinations.collected": [...prior, ...rejections],
    };
  },
});

// Default ecology — does not include enforceSchema, which is AF-specific
// and constructed by the consumer.
export const PRIMITIVES: readonly ParseOperator[] = [
  normalizeWhitespace,
  regexEmitUrl,
  regexEmitDate,
  regexEmitTitle,
  proximityAssemble,
];
