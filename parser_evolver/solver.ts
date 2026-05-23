// Beam-search solver over gene-strings.
//
// A gene-string is a typed bytecode of operator instructions. Decompression
// runs the genes in order, threading the produced channels forward. Each
// operator's `io` record says which channels it requires and which it
// outputs, so the solver can keep only extensions whose requiredInputs
// are met by the channels currently in scope without ever testing
// `if op.id === ...`.
//
// Embedding similarity is load-bearing here: at each frontier, eligible
// extensions are pruned to top-k by cosine to the AF's column-token bag,
// so relatives of needed work prune the search before scoring. Ties on
// total score break by the last gene's similarity to the AF tokens.

import { fit, similarity, embed } from "./embedding.js";
import { signatureOf } from "./operator_reflection.js";
import type {
  CsvAF,
  Gene,
  GeneString,
  Hallucination,
  HallucinationKind,
  ParseCandidate,
  ParseContext,
  ParseOperator,
  RowHypothesis,
  Solver,
  TraceRegion,
} from "./types.js";

type ChannelBag = Readonly<Record<string, unknown>>;

type DecompressionTrace = {
  readonly genes: GeneString;
  readonly bag: ChannelBag;
  readonly cost: number;
};

const merge = <T extends ChannelBag>(a: T, b: T): T => ({ ...a, ...b } as T);

const runGene = (op: ParseOperator, ctx: ParseContext, bag: ChannelBag): ChannelBag =>
  merge(bag, op.run(ctx, bag) as ChannelBag);

const decompress = (
  ctx: ParseContext,
  genes: GeneString,
  ops: ReadonlyMap<string, ParseOperator>,
): DecompressionTrace => {
  const seed: DecompressionTrace = {
    genes: [],
    bag: { "text.normalized": ctx.normalizedText },
    cost: 0,
  };
  return genes.reduce<DecompressionTrace>((acc, g) => {
    const op = ops.get(g.operatorId);
    return op === undefined
      ? acc
      : { genes: [...acc.genes, g], bag: runGene(op, ctx, acc.bag), cost: acc.cost + op.cost };
  }, seed);
};

// "rows.validated" wins over "rows.assembled" — a validator pass replaces
// the upstream row list when present.
const extractRows = (bag: ChannelBag): readonly RowHypothesis[] =>
  (bag["rows.validated"] as readonly RowHypothesis[] | undefined) ??
  (bag["rows.assembled"] as readonly RowHypothesis[] | undefined) ??
  [];

const extractTraces = (bag: ChannelBag): readonly TraceRegion[] =>
  (bag["trace.regions"] as readonly TraceRegion[] | undefined) ?? [];

const extractHallucinations = (bag: ChannelBag): readonly Hallucination[] =>
  (bag["hallucinations.collected"] as readonly Hallucination[] | undefined) ?? [];

const ZERO_KINDS: Record<HallucinationKind, number> = {
  unsupported_cell: 0,
  misassigned_span: 0,
  field_role_confusion: 0,
  missing_emitter: 0,
  validator_rejection: 0,
  low_coverage_region: 0,
  overfit_pattern: 0,
};

const summarise = (items: readonly Hallucination[]): Record<HallucinationKind, number> =>
  items.reduce<Record<HallucinationKind, number>>(
    (acc, h) => ({ ...acc, [h.kind]: acc[h.kind] + 1 }),
    { ...ZERO_KINDS },
  );

const diagnostics = (
  trace: DecompressionTrace,
  rows: readonly RowHypothesis[],
  af: CsvAF,
  extraHallucinations: readonly Hallucination[],
) => {
  const cells = rows.flatMap((r) => Object.values(r.fields));
  const unsourced = cells.filter((c) => c.span[1] <= c.span[0]).length;
  const allH = [...af.hallucinations(rows), ...extraHallucinations];
  // hallucinationRisk: only "real" risk when cells exist. An empty creature
  // is uninformative, not actively lying — keep its risk at 0 so the beam
  // can climb past it as soon as one valid cell appears.
  return {
    coverage: rows.length === 0 ? 0 : cells.length / (rows.length * af.columns.length),
    complexity: trace.cost,
    hallucinationRisk: cells.length === 0 ? 0 : unsourced / cells.length,
    stability: rows.length === 0 ? 0 : rows.filter((r) => r.score > 0.5).length / rows.length,
    pressure: { summary: summarise(allH) },
  };
};

// Progress: how many distinct AF-relevant channels the bag has filled so
// far. A small reward for progress lets the beam climb out of the empty
// basin without overpowering the AF's hallucination penalties.
const PROGRESS_CHANNELS = ["spans.url", "spans.dated", "spans.titled", "rows.assembled", "rows.validated"] as const;
const progressBonus = (bag: ChannelBag): number =>
  0.2 * PROGRESS_CHANNELS.filter((ch) => Array.isArray(bag[ch]) && (bag[ch] as unknown[]).length > 0).length;

const afTokens = (af: CsvAF): readonly string[] =>
  af.columns.flatMap((c) => [c.name, "row", "table", "csv", "field"]);

const evaluate = (
  ctx: ParseContext,
  af: CsvAF,
  genes: GeneString,
  ops: ReadonlyMap<string, ParseOperator>,
): ParseCandidate => {
  const trace = decompress(ctx, genes, ops);
  const rows = extractRows(trace.bag);
  const extraH = extractHallucinations(trace.bag);
  const diag = diagnostics(trace, rows, af, extraH);
  const base = af.scoreRun(rows);
  const score = base - 0.1 * diag.complexity - 5 * diag.hallucinationRisk + progressBonus(trace.bag);
  return { genes, rows, score, diagnostics: diag, traces: extractTraces(trace.bag) };
};

const availableChannels = (
  genes: GeneString,
  ops: ReadonlyMap<string, ParseOperator>,
): ReadonlySet<string> =>
  genes.reduce<Set<string>>((set, g) => {
    const op = ops.get(g.operatorId);
    if (op) signatureOf(op).outputs.forEach((p) => set.add(p));
    return set;
  }, new Set<string>(["text.normalized"]));

// Eligibility: requiredInputs ⊆ available. Ranking + pruning is by
// embedding cosine to the *remaining* AF tokens — column tokens whose
// corresponding span channel is not yet filled. Operators whose outputs
// are saturated drop to near-zero so the solver doesn't waste its
// top-K budget re-running the same emitter. `topK` is the load-bearing
// knob — without it the embedding never gates the search.
const COLUMN_TO_CHANNEL: Readonly<Record<string, string>> = {
  date: "spans.dated",
  title: "spans.titled",
  url: "spans.url",
};

const remainingTokens = (af: CsvAF, channels: ReadonlySet<string>): readonly string[] => {
  const open = af.columns.filter((c) => !channels.has(COLUMN_TO_CHANNEL[c.name] ?? ""));
  return open.length === 0
    ? ["row", "table", "csv", "assemble", "validate"]
    : open.flatMap((c) => [c.name, "field", "extract", "emit"]);
};

const eligibleExtensions = (
  af: CsvAF,
  ops: readonly ParseOperator[],
  channels: ReadonlySet<string>,
  topK: number,
): readonly { op: ParseOperator; sim: number }[] => {
  const tokens = remainingTokens(af, channels);
  const saturation = (op: ParseOperator): number => {
    const outs = signatureOf(op).outputs;
    return outs.length === 0
      ? 0
      : outs.every((p) => channels.has(p))
        ? 0.01
        : 1;
  };
  return ops
    .filter((op) => signatureOf(op).requiredInputs.every((n) => channels.has(n)))
    .map((op) => ({ op, sim: fit(op, tokens) * saturation(op) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, topK);
};

export type BeamConfig = {
  readonly beam: number;
  readonly maxLen: number;
  // Top-k operators (by embedding similarity to AF tokens) kept at each
  // expansion step. Lower values let the embedding dominate; higher values
  // restore exhaustive enumeration.
  readonly extensionTopK: number;
};

const DEFAULTS: BeamConfig = { beam: 8, maxLen: 6, extensionTopK: 4 };

// Tie-break: candidates with equal total score are reordered by the
// similarity of their last gene to the AF tokens. This is the second place
// the embedding steers search.
const candidateTieKey = (
  c: ParseCandidate,
  ops: ReadonlyMap<string, ParseOperator>,
  tokens: readonly string[],
): number => {
  const last = c.genes[c.genes.length - 1];
  const op = last && ops.get(last.operatorId);
  // Embed the operator against `tokens` by constructing an alternate
  // view where `tokens` replaces the operator's own — a one-off
  // probe used only for tie-breaking.
  return op ? similarity(embed(op), embed({ ...op, tokens })) : 0;
};

export const makeBeamSolver = (cfg: Partial<BeamConfig> = {}): Solver => {
  const C: BeamConfig = { ...DEFAULTS, ...cfg };
  return {
    search: (ctx, af, ops, seedGenes) => {
      const opIndex = new Map(ops.map((o) => [o.id, o] as const));
      const tokens = afTokens(af);
      const seeds: readonly GeneString[] = seedGenes && seedGenes.length > 0 ? seedGenes : [[]];

      const extend = (cand: ParseCandidate): readonly ParseCandidate[] =>
        eligibleExtensions(af, ops, availableChannels(cand.genes, opIndex), C.extensionTopK).map(
          ({ op }) =>
            evaluate(ctx, af, [...cand.genes, { operatorId: op.id } satisfies Gene], opIndex),
        );

      const dedupe = (xs: readonly ParseCandidate[]): readonly ParseCandidate[] => {
        const seen = new Map<string, ParseCandidate>();
        xs.forEach((c) => {
          const k = c.genes.map((g) => g.operatorId).join(">>");
          const prior = seen.get(k);
          if (prior === undefined || c.score > prior.score) seen.set(k, c);
        });
        return Array.from(seen.values());
      };

      const rank = (xs: readonly ParseCandidate[]): readonly ParseCandidate[] =>
        [...dedupe(xs)]
          .sort((a, b) => {
            const ds = b.score - a.score;
            return ds !== 0 ? ds : candidateTieKey(b, opIndex, tokens) - candidateTieKey(a, opIndex, tokens);
          })
          .slice(0, C.beam);

      const advance = (cands: readonly ParseCandidate[]): readonly ParseCandidate[] =>
        rank([...cands, ...cands.flatMap(extend)]);

      const initial = seeds.map((g) => evaluate(ctx, af, g, opIndex));
      return Array.from({ length: C.maxLen }).reduce<readonly ParseCandidate[]>(
        (acc) => advance(acc),
        initial,
      );
    },
  };
};
