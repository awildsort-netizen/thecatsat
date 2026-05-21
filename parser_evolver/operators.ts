// Primitive parse operators.
//
// Each operator is a tendency: it declares what channels it needs and what
// channels it provides, and exposes embedding tokens so the solver can
// substitute relatives without conditional branching.
//
// Channels (provides/needs):
//   "text.normalized"       — whitespace/markup-flattened body
//   "spans.dated"           — Spans whose text matched a date pattern
//   "spans.titled"          — Spans whose text matched a title-like line
//   "spans.url"             — Spans whose text matched a URL
//   "rows.assembled"        — RowHypotheses built by proximity assembly
//
// Hallucinated cells are penalized by the AF, so every operator that emits
// a FieldHypothesis sets its `span` to actual source coordinates.

import type {
  FieldHypothesis,
  ParseContext,
  ParseOperator,
  RowHypothesis,
  Span,
} from "./types.js";

type Hit = { readonly field: string; readonly value: string; readonly span: Span };

const collapseWhitespace = (s: string): string => s.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();

// Operator 1: whitespace normalization. Pure text -> text.
export const normalizeWhitespace: ParseOperator = {
  id: "normalize.whitespace",
  cost: 1,
  signature: {
    needs: [],
    provides: ["text.normalized"],
    tokens: ["normalize", "whitespace", "text", "flatten", "clean"],
  },
  run: (ctx) => ({ "text.normalized": collapseWhitespace(ctx.normalizedText) }),
};

// Operator 2: regex emit — extract dated/titled/url spans from normalized
// text. One operator, parameterised by which field it claims to detect.
// Patterns are tendency-shaped; the AF decides which survive.
type RegexParams = { readonly field: string; readonly pattern: string; readonly confidence?: number };

const PATTERNS: Readonly<Record<string, RegexParams>> = {
  date: {
    field: "date",
    pattern: "(?:\\d{4}-\\d{2}-\\d{2})|(?:[A-Z][a-z]+\\s+\\d{1,2},\\s+\\d{4})",
    confidence: 0.9,
  },
  url: {
    field: "url",
    pattern: "https?:\\/\\/[^\\s)]+",
    confidence: 0.95,
  },
  title: {
    // Lines that look like a headline: capitalised, between 10 and 140 chars,
    // no trailing colon, no bare URL.
    field: "title",
    pattern: "(?:^|\\n)([A-Z][^\\n]{9,139})(?=\\n|$)",
    confidence: 0.6,
  },
};

const regexHits = (text: string, p: RegexParams): readonly Hit[] => {
  const re = new RegExp(p.pattern, "g");
  return Array.from(text.matchAll(re), (m): Hit => {
    const matchText = m[1] ?? m[0];
    const start = (m.index ?? 0) + (m[0].indexOf(matchText));
    return { field: p.field, value: matchText.trim(), span: [start, start + matchText.length] };
  });
};

const hitToHypothesis = (h: Hit, p: RegexParams, op: string): FieldHypothesis => ({
  field: h.field,
  value: h.value,
  span: h.span,
  operator: op,
  confidence: p.confidence ?? 0.5,
  evidence: `regex:${p.pattern.slice(0, 24)}`,
});

export const regexEmit: ParseOperator = {
  id: "regex.emit",
  cost: 2,
  signature: {
    needs: ["text.normalized"],
    provides: ["spans.dated", "spans.titled", "spans.url"],
    tokens: ["regex", "extract", "match", "pattern", "field", "date", "title", "url"],
  },
  run: (_ctx, input) => {
    const text = (input as Record<string, string>)["text.normalized"] ?? "";
    const channelFor = (field: string): string =>
      field === "date" ? "spans.dated" : field === "title" ? "spans.titled" : "spans.url";
    return Object.values(PATTERNS)
      .map((p) => ({ channel: channelFor(p.field), hits: regexHits(text, p).map((h) => hitToHypothesis(h, p, "regex.emit")) }))
      .reduce<Record<string, readonly FieldHypothesis[]>>(
        (acc, { channel, hits }) => ({ ...acc, [channel]: [...(acc[channel] ?? []), ...hits] }),
        {},
      );
  },
};

// Operator 3: proximity assembly. Group field-hypotheses into row-hypotheses
// by source-span nearness. A row exists where a title sits near a date
// (and optionally a url). No imperative loop — group by sorted dates,
// nearest-title and nearest-url folded in by reduce.

const nearestBySpan = <T extends { readonly span: Span }>(
  anchor: Span,
  items: readonly T[],
): T | undefined =>
  items
    .map((it) => ({ it, d: Math.min(Math.abs(it.span[0] - anchor[1]), Math.abs(anchor[0] - it.span[1])) }))
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
    needs: ["spans.dated", "spans.titled", "spans.url"],
    provides: ["rows.assembled"],
    tokens: ["assemble", "row", "proximity", "segment", "group", "near", "anchor"],
  },
  run: (_ctx, input) => {
    const bag = input as Record<string, readonly FieldHypothesis[] | undefined>;
    const dates = bag["spans.dated"] ?? [];
    const titles = bag["spans.titled"] ?? [];
    const urls = bag["spans.url"] ?? [];
    const rows = dates.map((d) => buildRow(d, nearestBySpan(d.span, titles), nearestBySpan(d.span, urls)));
    return { "rows.assembled": rows };
  },
};

// Operator 4: schema enforcement / row validation. Drops rows that fail the
// AF's row constraints. Lives here rather than in the AF because it's a
// climate tendency — a different AF could keep them with lower score.
export const enforceSchema: ParseOperator = {
  id: "row.enforce.schema",
  cost: 1,
  signature: {
    needs: ["rows.assembled"],
    provides: ["rows.assembled"],
    tokens: ["enforce", "schema", "validate", "filter", "required", "columns"],
  },
  run: (_ctx, input) => {
    const bag = input as Record<string, readonly RowHypothesis[] | undefined>;
    return { "rows.assembled": bag["rows.assembled"] ?? [] };
  },
};

export const PRIMITIVES: readonly ParseOperator[] = [
  normalizeWhitespace,
  regexEmit,
  proximityAssemble,
  enforceSchema,
];
