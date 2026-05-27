// Motif-invariance brittleness metric.
//
// Background: thecatsat treats a parser-creature as a *compression* of the
// source manifold into a stable table. A compression that survives
// structurally-equivalent recompositions of its input is defending pattern;
// one that fragments is defending boundary (an outline that has been
// confused for the essence). Hallucination strain catches outright lies,
// but it does not catch *brittleness* — a creature can be low-strain on
// the training manifold and still collapse the moment the manifold is
// nudged. This module makes that latent failure first-class.
//
// Vocabulary (kept tight on purpose):
//
//   Motif      — the structural shape of what a creature emits, abstracted
//                from textual specifics. A multiset of (field, value-class,
//                role-position) tuples plus row-arity stats. The motif is
//                what should be invariant; the literal cells are not.
//
//   Recomposer — a semantics-preserving perturbation of ParseContext.
//                Examples: reorder blocks, swap equivalent date formats,
//                permute internal block lines, expand/contract whitespace.
//                Each recomposer commits to *not* changing what a stable
//                pattern would emit; only what an outline-defending parser
//                would emit.
//
//   Brittleness = 1 − mean Jaccard(motif(original), motif(perturbed))
//                 across a probe set. Bounded in [0, 1]. Zero = pure
//                 pattern; one = pure outline.
//
// The metric is intentionally cheap: it re-runs the creature's gene-string
// on perturbed contexts, computes motifs, and Jaccards them. No network,
// no learning, no extra operators. It is meant to be folded into solver
// selection as an additive penalty whose weight is a single knob.

import type {
  CsvAF,
  FieldHypothesis,
  Gene,
  GeneString,
  ParseCandidate,
  ParseContext,
  ParseOperator,
  RowHypothesis,
} from "./types.js";

// ---------------------------------------------------------------------------
// Motif extraction
// ---------------------------------------------------------------------------

// Value class is a *coarse* type tag: it abstracts "2026-04-12" and
// "April 1, 2026" to the same class ("date.iso" vs "date.long"), which is
// what we want — a compression that depends on date *format* is defending
// outline, not pattern. We keep the iso/long distinction visible so we can
// see brittleness against date-format swaps specifically.
export type ValueClass =
  | "date.iso"
  | "date.long"
  | "url.https"
  | "url.other"
  | "title.short"
  | "title.long"
  | "empty"
  | "other";

const classifyValue = (field: string, value: string): ValueClass => {
  if (!value) return "empty";
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return "date.iso";
  if (/^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$/.test(value)) return "date.long";
  if (/^https:\/\/\S+$/.test(value)) return "url.https";
  if (/^https?:\/\/\S+$/.test(value)) return "url.other";
  if (field === "title") return value.length > 60 ? "title.long" : "title.short";
  return "other";
};

// A motif feature is a (field, value-class) tag with no positional or
// textual identity. Multisets of these are what we compare across
// recompositions. Per-row arity (how many fields per row) is folded in
// separately so we can tell "every row dropped a field" apart from "a
// few cells changed class".
export type MotifFeature = `${string}::${ValueClass}`;

const featureOf = (fh: FieldHypothesis): MotifFeature =>
  `${fh.field}::${classifyValue(fh.field, fh.value)}` as MotifFeature;

export type Motif = {
  readonly features: ReadonlyMap<MotifFeature, number>; // multiset
  readonly rowCount: number;
  readonly arityHistogram: ReadonlyMap<number, number>; // rows-with-k-fields -> count
};

const tally = <K>(items: readonly K[]): ReadonlyMap<K, number> =>
  items.reduce<Map<K, number>>((m, k) => m.set(k, (m.get(k) ?? 0) + 1), new Map<K, number>());

export const motifOf = (rows: readonly RowHypothesis[]): Motif => {
  const cells = rows.flatMap((r) => Object.values(r.fields));
  const features = tally(cells.map(featureOf));
  const arities = tally(rows.map((r) => Object.keys(r.fields).length));
  return { features, rowCount: rows.length, arityHistogram: arities };
};

// ---------------------------------------------------------------------------
// Jaccard over multisets (1 − distance) and arity coherence
// ---------------------------------------------------------------------------

const multisetJaccard = <K>(a: ReadonlyMap<K, number>, b: ReadonlyMap<K, number>): number => {
  if (a.size === 0 && b.size === 0) return 1;
  const keys = new Set<K>([...a.keys(), ...b.keys()]);
  let inter = 0;
  let union = 0;
  keys.forEach((k) => {
    const av = a.get(k) ?? 0;
    const bv = b.get(k) ?? 0;
    inter += Math.min(av, bv);
    union += Math.max(av, bv);
  });
  return union === 0 ? 1 : inter / union;
};

// Motif similarity weights feature-set Jaccard against arity-distribution
// Jaccard. Equal weights keep "what kind of cells" and "how many fields
// per row" symmetric. Both terms are bounded [0,1], so the combined
// similarity is too.
export const motifSimilarity = (a: Motif, b: Motif): number => {
  const featSim = multisetJaccard(a.features, b.features);
  const arSim = multisetJaccard(a.arityHistogram, b.arityHistogram);
  return 0.5 * featSim + 0.5 * arSim;
};

// ---------------------------------------------------------------------------
// Recomposers — semantics-preserving perturbations of ParseContext
// ---------------------------------------------------------------------------
//
// Each recomposer is documented with the invariant it claims to preserve.
// If a parser-creature changes its motif in response to one of these
// recompositions, it is defending boundary against a change the source
// manifold considers irrelevant.

export type Recomposer = {
  readonly name: string;
  readonly apply: (ctx: ParseContext) => ParseContext;
  // What does this recomposer claim is invariant? Used in benchmark
  // reporting so a brittleness number always traces back to a stated
  // claim about the source manifold.
  readonly invariant: string;
};

// Split text into blocks separated by blank lines. Robust to trailing
// whitespace. Returns the array of blocks; reassemble with join("\n\n").
const splitBlocks = (text: string): string[] =>
  text.split(/\n\s*\n/).map((b) => b.trim()).filter((b) => b.length > 0);

// Deterministic shuffle using a Linear Congruential Generator seeded by
// the recomposer name + input length. We want *reproducible* perturbation
// so brittleness scores don't drift between runs.
const lcg = (seed: number): (() => number) => {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
};

const stableHash = (s: string): number => {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) h = ((h ^ s.charCodeAt(i)) * 16777619) >>> 0;
  return h;
};

const shuffled = <T>(xs: readonly T[], seed: number): T[] => {
  const rng = lcg(seed);
  const out = [...xs];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
};

const withText = (ctx: ParseContext, text: string): ParseContext => ({
  ...ctx,
  rawText: text,
  normalizedText: text,
});

// Block reorder: the order of independent records is not part of the
// pattern. A parser that emits different cells just because blocks
// appear in a different order is defending position.
export const reorderBlocks: Recomposer = {
  name: "reorder_blocks",
  invariant: "block (record) order does not change the set of rows emitted",
  apply: (ctx) => {
    const blocks = splitBlocks(ctx.normalizedText);
    if (blocks.length <= 1) return ctx;
    const seed = stableHash("reorder_blocks::" + ctx.normalizedText);
    return withText(ctx, shuffled(blocks, seed).join("\n\n"));
  },
};

// Whitespace dilation: extra blank lines / indentation should not change
// any cell. A parser that depends on tight spacing is brittle to format
// drift.
export const dilateWhitespace: Recomposer = {
  name: "dilate_whitespace",
  invariant: "extra blank lines and trailing spaces do not change cell content",
  apply: (ctx) => {
    const lines = ctx.normalizedText.split("\n");
    const dilated = lines.flatMap((l, i) => (i % 3 === 0 ? [l, ""] : [l + " "]));
    return withText(ctx, dilated.join("\n"));
  },
};

// Date-format swap: "2026-04-12" ↔ "April 12, 2026" is the same date. A
// parser that *can* emit dates in either format on its own runs but
// changes motif when the input format flips is brittle to surface form.
const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

const isoToLong = (iso: string): string => {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return iso;
  const month = MONTH_NAMES[parseInt(m[2]!, 10) - 1] ?? m[2];
  return `${month} ${parseInt(m[3]!, 10)}, ${m[1]}`;
};

const longToIso = (long: string): string => {
  const m = /^([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})$/.exec(long);
  if (!m) return long;
  const idx = MONTH_NAMES.indexOf(m[1]!);
  if (idx < 0) return long;
  return `${m[3]}-${String(idx + 1).padStart(2, "0")}-${String(parseInt(m[2]!, 10)).padStart(2, "0")}`;
};

export const swapDateFormat: Recomposer = {
  name: "swap_date_format",
  invariant: "iso and long-form date strings denote the same date",
  apply: (ctx) => {
    // Only rewrite dates that appear as their own line — leaves URL
    // slugs containing dates alone (those are not, semantically, dates
    // the parser should consume as cells).
    const rewritten = ctx.normalizedText
      .split("\n")
      .map((line) => {
        const trimmed = line.trim();
        if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return line.replace(trimmed, isoToLong(trimmed));
        if (/^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$/.test(trimmed)) return line.replace(trimmed, longToIso(trimmed));
        return line;
      })
      .join("\n");
    return withText(ctx, rewritten);
  },
};

// Default probe set: small, fast, covers the three claims above.
export const DEFAULT_RECOMPOSERS: readonly Recomposer[] = [
  reorderBlocks,
  dilateWhitespace,
  swapDateFormat,
];

// ---------------------------------------------------------------------------
// Brittleness scoring
// ---------------------------------------------------------------------------

// Re-run a gene-string against a context. This is a thin shadow of the
// solver's decompress/evaluate path so brittleness can stand alone — it
// would be cleaner to import from solver.ts, but we want this module to
// be self-contained for tests and future callers.
const runGenes = (
  ctx: ParseContext,
  genes: GeneString,
  ops: ReadonlyMap<string, ParseOperator>,
): readonly RowHypothesis[] => {
  type Bag = Readonly<Record<string, unknown>>;
  const bag: Bag = genes.reduce<Bag>(
    (acc, g) => {
      const op = ops.get(g.operatorId);
      if (op === undefined) return acc;
      const out = op.run(ctx, acc) as Bag;
      return { ...acc, ...out };
    },
    { "text.normalized": ctx.normalizedText },
  );
  return (
    (bag["rows.validated"] as readonly RowHypothesis[] | undefined) ??
    (bag["rows.assembled"] as readonly RowHypothesis[] | undefined) ??
    []
  );
};

export type BrittlenessReport = {
  readonly score: number; // in [0, 1]; lower = more pattern, less outline
  readonly perRecomposer: ReadonlyMap<string, number>; // similarity per recomposer
  readonly motifs: ReadonlyMap<string, Motif>; // for inspection
  readonly originalMotif: Motif;
};

export type BrittlenessConfig = {
  readonly recomposers?: readonly Recomposer[];
  // Minimum row count on the original to bother computing brittleness.
  // An empty creature has a trivial motif; we keep its brittleness at 0
  // (uninformative, not lying — same posture as ParseDiagnostics).
  readonly minRowsForMeasurement?: number;
};

const DEFAULT_CONFIG: Required<BrittlenessConfig> = {
  recomposers: DEFAULT_RECOMPOSERS,
  minRowsForMeasurement: 1,
};

export const brittlenessOf = (
  candidate: ParseCandidate,
  ctx: ParseContext,
  ops: readonly ParseOperator[],
  cfg: BrittlenessConfig = {},
): BrittlenessReport => {
  const C = { ...DEFAULT_CONFIG, ...cfg };
  const opIndex = new Map(ops.map((o) => [o.id, o] as const));
  const originalMotif = motifOf(candidate.rows);

  if (candidate.rows.length < C.minRowsForMeasurement) {
    return {
      score: 0,
      perRecomposer: new Map(),
      motifs: new Map(),
      originalMotif,
    };
  }

  const perRecomposer = new Map<string, number>();
  const motifs = new Map<string, Motif>();

  C.recomposers.forEach((r) => {
    const perturbedCtx = r.apply(ctx);
    const perturbedRows = runGenes(perturbedCtx, candidate.genes, opIndex);
    const perturbedMotif = motifOf(perturbedRows);
    motifs.set(r.name, perturbedMotif);
    perRecomposer.set(r.name, motifSimilarity(originalMotif, perturbedMotif));
  });

  const meanSim =
    Array.from(perRecomposer.values()).reduce((s, x) => s + x, 0) / perRecomposer.size;
  const score = 1 - meanSim;
  return { score, perRecomposer, motifs, originalMotif };
};

// ---------------------------------------------------------------------------
// Solver-side penalty
// ---------------------------------------------------------------------------
//
// Folding brittleness into the solver is intentionally additive: a
// caller passes a `BrittlenessPenalty` to the evaluator (see solver.ts
// patch) and the weight is the single dial. Weight 0 = original solver
// behavior, by construction. Higher weights pressure the beam to select
// pattern-defending creatures even at small cost to immediate strain.

export type BrittlenessPenalty = {
  readonly weight: number;
  readonly af: CsvAF;
  readonly ctx: ParseContext;
  readonly ops: readonly ParseOperator[];
  readonly config?: BrittlenessConfig;
};

export const brittlenessPenalty = (
  candidate: ParseCandidate,
  pen: BrittlenessPenalty,
): number => {
  if (pen.weight <= 0) return 0;
  const report = brittlenessOf(candidate, pen.ctx, pen.ops, pen.config);
  return pen.weight * report.score;
};

// Re-export the Gene type so callers writing recomposed seeds don't need
// a second import from types.
export type { Gene };
