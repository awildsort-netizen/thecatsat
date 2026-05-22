// Primitive parse operators.
//
// Each operator declares a single `inputs`/`outputs` channel spec
// through `defineOperator`. Required inputs gate solver eligibility;
// optional inputs (typed via `optional<T>()`) are contextual reads
// that show up as `?`-properties on the run body's input bag. The
// operator's `io` record is derived from these declarations — no
// hand-authored second copy.
//
// Channels:
//   "text.normalized"           — whitespace/markup-flattened body
//   "spans.url"                 — URL spans (kept first; date detector excludes them)
//   "spans.dated"               — date spans NOT contained in any URL span
//   "spans.titled"              — title-like lines that are not dates and not urls
//   "rows.assembled"            — RowHypotheses built by proximity assembly
//   "rows.validated"            — RowHypotheses that satisfied all AF rowConstraints
//   "trace.regions"             — TraceRegion artifacts accumulated across operators
//   "hallucinations.collected"  — typed Hallucination artifacts from validators
//
// Every emitter records a TraceRegion alongside the cell. A validator
// that rejects a cell writes a typed `Hallucination` (see types.ts)
// under "hallucinations.collected", which the AF reads at scoreRun.

import type {
  CsvAF,
  FieldHypothesis,
  Hallucination,
  ParseOperator,
  RowHypothesis,
  Span,
  TraceRegion,
} from "./types.js";
import { defineOperator, optional, required } from "./operator_reflection.js";

// ---------------------------------------------------------------------------
// Small helpers — pure, no for-loops.
// ---------------------------------------------------------------------------

const collapseWhitespace = (s: string): string =>
  s.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();

const spanContains = (outer: Span, inner: Span): boolean =>
  inner[0] >= outer[0] && inner[1] <= outer[1];

const containedInAny = (s: Span, outers: readonly Span[]): boolean =>
  outers.some((o) => spanContains(o, s));

const traceId = (operator: string, span: Span, channel: string): string =>
  `${operator}@${channel}#${span[0]}:${span[1]}`;

const region = (operator: string, channel: string, span: Span, label: string): TraceRegion => ({
  id: traceId(operator, span, channel),
  label,
  span,
  channel,
  operator,
});

type EmitParams = {
  readonly field: string;
  readonly pattern: string;
  readonly group: number;
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

export const normalizeWhitespace = defineOperator({
  id: "normalize.whitespace",
  cost: 1,
  tokens: ["normalize", "whitespace", "text", "flatten", "clean", "prep"],
  inputs: {},
  outputs: {
    "text.normalized": required<string>(),
  },
  run: (ctx) => ({ "text.normalized": collapseWhitespace(ctx.normalizedText) }),
});

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

export const regexEmitUrl = defineOperator({
  id: "regex.emit.url",
  cost: 2,
  tokens: ["regex", "extract", "url", "link", "href", "address"],
  inputs: {
    "text.normalized": required<string>(),
    "trace.regions": optional<readonly TraceRegion[]>(),
  },
  outputs: {
    "spans.url": required<readonly FieldHypothesis[]>(),
    "trace.regions": required<readonly TraceRegion[]>(),
  },
  run: (_ctx, input) => {
    const hits = emitFrom(input["text.normalized"], URL_PARAMS);
    return {
      "spans.url": hits,
      "trace.regions": [
        ...(input["trace.regions"] ?? []),
        ...tracesFor(hits, URL_PARAMS.channel, URL_PARAMS.operatorId, "url"),
      ],
    };
  },
});

// ---------------------------------------------------------------------------
// Operator 3 — date emitter. Excludes dates that fall inside any URL span.
// ---------------------------------------------------------------------------

const DATE_PARAMS: EmitParams = {
  field: "date",
  pattern: "(?:\\d{4}-\\d{2}-\\d{2})|(?:[A-Z][a-z]+\\s+\\d{1,2},\\s+\\d{4})",
  group: 0,
  confidence: 0.9,
  channel: "spans.dated",
  operatorId: "regex.emit.date",
};

export const regexEmitDate = defineOperator({
  id: "regex.emit.date",
  cost: 2,
  tokens: ["regex", "extract", "date", "iso", "calendar", "month"],
  inputs: {
    "text.normalized": required<string>(),
    "spans.url": required<readonly FieldHypothesis[]>(),
    "trace.regions": optional<readonly TraceRegion[]>(),
  },
  outputs: {
    "spans.dated": required<readonly FieldHypothesis[]>(),
    "trace.regions": required<readonly TraceRegion[]>(),
  },
  run: (_ctx, input) => {
    const urlSpans = input["spans.url"].map((u) => u.span);
    const raw = emitFrom(input["text.normalized"], DATE_PARAMS);
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
// or that sit inside a URL span.
// ---------------------------------------------------------------------------

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

export const regexEmitTitle = defineOperator({
  id: "regex.emit.title",
  cost: 2,
  tokens: ["regex", "extract", "title", "headline", "heading", "name"],
  inputs: {
    "text.normalized": required<string>(),
    "spans.url": required<readonly FieldHypothesis[]>(),
    "trace.regions": optional<readonly TraceRegion[]>(),
  },
  outputs: {
    "spans.titled": required<readonly FieldHypothesis[]>(),
    "trace.regions": required<readonly TraceRegion[]>(),
  },
  run: (_ctx, input) => {
    const urlSpans = input["spans.url"].map((u) => u.span);
    const raw = emitFrom(input["text.normalized"], TITLE_PARAMS);
    const hits = raw.filter(
      (h) => !looksLikeDate(h.value) && !looksLikeUrl(h.value) && !containedInAny(h.span, urlSpans),
    );
    return {
      "spans.titled": hits,
      "trace.regions": [
        ...(input["trace.regions"] ?? []),
        ...tracesFor(hits, TITLE_PARAMS.channel, TITLE_PARAMS.operatorId, "title"),
      ],
    };
  },
});

// ---------------------------------------------------------------------------
// Operator 5 — proximity assembly.
//
// A row exists where a date sits near a title. The URL is picked from
// URLs that come *after* the date and before the next date — a
// forward-window pick avoids cross-bleed across rows.
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

export const proximityAssemble = defineOperator({
  id: "row.assemble.proximity",
  cost: 3,
  tokens: ["assemble", "row", "proximity", "segment", "group", "near", "anchor", "window"],
  inputs: {
    "spans.dated": required<readonly FieldHypothesis[]>(),
    "spans.titled": required<readonly FieldHypothesis[]>(),
    // URLs are optional: assembly can produce a date+title row without
    // a URL, so the URL emitter doesn't gate scheduling here.
    "spans.url": optional<readonly FieldHypothesis[]>(),
  },
  outputs: {
    "rows.assembled": required<readonly RowHypothesis[]>(),
  },
  run: (_ctx, input) => {
    const dates = [...input["spans.dated"]].sort((a, b) => a.span[0] - b.span[0]);
    const titles = input["spans.titled"];
    const urls = input["spans.url"] ?? [];

    const rows = dates.map((d, i) => {
      const next = dates[i + 1]?.span[0] ?? Number.MAX_SAFE_INTEGER;
      const windowUrls = urls.filter((u) => u.span[0] >= d.span[1] && u.span[1] <= next);
      const windowTitles = titles.filter((t) => t.span[0] >= d.span[1] && t.span[1] <= next);
      return buildRow(d, nearest(d.span, windowTitles) ?? nearest(d.span, titles), windowUrls[0]);
    });
    return { "rows.assembled": rows };
  },
});

// ---------------------------------------------------------------------------
// Operator 6 — schema enforcement / validator pass.
//
// Drops rows that fail any AF rowConstraint. The AF reference is captured
// in the closure rather than flowing through the channel bag. Rejected
// rows become typed `Hallucination`s under "hallucinations.collected".
// ---------------------------------------------------------------------------

export const makeEnforceSchema = (af: CsvAF): ParseOperator =>
  defineOperator({
    id: "row.enforce.schema",
    cost: 1,
    tokens: ["enforce", "schema", "validate", "filter", "required", "columns", "reject"],
    inputs: {
      "rows.assembled": required<readonly RowHypothesis[]>(),
      "hallucinations.collected": optional<readonly Hallucination[]>(),
    },
    outputs: {
      "rows.validated": required<readonly RowHypothesis[]>(),
      "hallucinations.collected": required<readonly Hallucination[]>(),
    },
    run: (_ctx, input) => {
      const rows = input["rows.assembled"];
      const kept = rows.filter((r) => af.rowConstraints.every((c) => c(r)));
      const rejections: readonly Hallucination[] = rows
        .filter((r) => !af.rowConstraints.every((c) => c(r)))
        .map<Hallucination>((r) => ({
          kind: "validator_rejection",
          operator: "row.enforce.schema",
          weight: 1,
          note: `row failed rowConstraints; fields=${Object.keys(r.fields).join(",")}`,
        }));
      return {
        "rows.validated": kept,
        "hallucinations.collected": [...(input["hallucinations.collected"] ?? []), ...rejections],
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
