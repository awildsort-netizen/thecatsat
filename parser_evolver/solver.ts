// Beam-search solver over gene-strings.
//
// A gene-string is a typed bytecode of operator instructions. Decompression
// = run the genes in order, threading the provided channels forward. Each
// operator's signature says what channels it needs and provides, so the
// solver can keep only the extensions that actually satisfy unmet needs
// without ever testing `if op.id === ...`.
//
// The search is dataflow over a frontier of candidates; each round extends
// every live candidate with every operator whose needs are met by the
// channels currently in scope, scored by the AF. This is small on purpose.

import { embed, similarity } from "./embedding.js";
import type {
  CsvAF,
  Gene,
  GeneString,
  ParseCandidate,
  ParseContext,
  ParseOperator,
  RowHypothesis,
  Solver,
} from "./types.js";

type ChannelBag = Readonly<Record<string, unknown>>;

type DecompressionTrace = {
  readonly genes: GeneString;
  readonly bag: ChannelBag;
  readonly cost: number;
};

const merge = <T extends ChannelBag>(a: T, b: T): T => ({ ...a, ...b } as T);

const runGene = (op: ParseOperator, ctx: ParseContext, bag: ChannelBag): ChannelBag => {
  const out = op.run(ctx, bag) as ChannelBag;
  return merge(bag, out);
};

const decompress = (ctx: ParseContext, genes: GeneString, ops: ReadonlyMap<string, ParseOperator>): DecompressionTrace => {
  const seed: DecompressionTrace = { genes: [], bag: { "text.normalized": ctx.normalizedText }, cost: 0 };
  return genes.reduce<DecompressionTrace>((acc, g) => {
    const op = ops.get(g.operatorId);
    return op === undefined
      ? acc
      : { genes: [...acc.genes, g], bag: runGene(op, ctx, acc.bag), cost: acc.cost + op.cost };
  }, seed);
};

const extractRows = (bag: ChannelBag): readonly RowHypothesis[] =>
  (bag["rows.assembled"] as readonly RowHypothesis[] | undefined) ?? [];

const diagnostics = (trace: DecompressionTrace, rows: readonly RowHypothesis[]) => {
  const cells = rows.flatMap((r) => Object.values(r.fields));
  const unsourced = cells.filter((c) => c.span[1] <= c.span[0]).length;
  return {
    coverage: rows.length === 0 ? 0 : cells.length / (rows.length * 3),
    complexity: trace.cost,
    hallucinationRisk: cells.length === 0 ? 1 : unsourced / cells.length,
    stability: rows.length === 0 ? 0 : rows.filter((r) => r.score > 0.5).length / rows.length,
  };
};

const evaluate = (ctx: ParseContext, af: CsvAF, genes: GeneString, ops: ReadonlyMap<string, ParseOperator>): ParseCandidate => {
  const trace = decompress(ctx, genes, ops);
  const rows = extractRows(trace.bag);
  const diag = diagnostics(trace, rows);
  const base = af.scoreRun(rows);
  // Penalise complexity and hallucination at the candidate level; the AF
  // handles the row-level penalty. Bytecode-like complexity is a tax.
  const score = base - 0.1 * diag.complexity - 5 * diag.hallucinationRisk;
  return { genes, rows, score, diagnostics: diag };
};

const availableChannels = (genes: GeneString, ops: ReadonlyMap<string, ParseOperator>): ReadonlySet<string> =>
  genes.reduce<Set<string>>((set, g) => {
    const op = ops.get(g.operatorId);
    op?.signature.provides.forEach((p) => set.add(p));
    return set;
  }, new Set<string>(["text.normalized"]));

// Signature-driven eligibility: an operator may extend a candidate iff its
// needs are already provided. Ties broken by embedding similarity to the
// AF's column tokens, so relatives of "wanted work" rank above strangers.
const af_tokens = (af: CsvAF): readonly string[] => af.columns.flatMap((c) => [c.name, "row", "table", "csv"]);

const eligibleExtensions = (
  af: CsvAF,
  ops: readonly ParseOperator[],
  channels: ReadonlySet<string>,
): readonly ParseOperator[] => {
  const wanted = af_tokens(af);
  return ops
    .filter((op) => op.signature.needs.every((n) => channels.has(n)))
    .map((op) => ({ op, w: similarity(embed(op), embed({ ...op, signature: { ...op.signature, tokens: wanted } })) }))
    .sort((a, b) => b.w - a.w)
    .map(({ op }) => op);
};

export type BeamConfig = { readonly beam: number; readonly maxLen: number };

const DEFAULTS: BeamConfig = { beam: 6, maxLen: 5 };

export const makeBeamSolver = (cfg: BeamConfig = DEFAULTS): Solver => ({
  search: (ctx, af, ops, seedGenes) => {
    const opIndex = new Map(ops.map((o) => [o.id, o] as const));
    const seeds: readonly GeneString[] = seedGenes && seedGenes.length > 0 ? seedGenes : [[]];

    type Step = { readonly cands: readonly ParseCandidate[] };

    const extend = (cand: ParseCandidate): readonly ParseCandidate[] => {
      const channels = availableChannels(cand.genes, opIndex);
      return eligibleExtensions(af, ops, channels).map((op) =>
        evaluate(ctx, af, [...cand.genes, { operatorId: op.id } satisfies Gene], opIndex),
      );
    };

    const advance = (cands: readonly ParseCandidate[]): readonly ParseCandidate[] =>
      [...cands, ...cands.flatMap(extend)].sort((a, b) => b.score - a.score).slice(0, cfg.beam);

    const initial: Step = { cands: seeds.map((g) => evaluate(ctx, af, g, opIndex)) };
    const final = Array.from({ length: cfg.maxLen }).reduce<Step>(
      (step) => ({ cands: advance(step.cands) }),
      initial,
    );
    return final.cands;
  },
});
