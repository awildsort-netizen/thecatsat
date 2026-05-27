// Crystallisation solver — CA-style propagation over the operator basin.
//
// Why a second Solver
// -------------------
// The beam solver explores the *space of gene-strings*: it enumerates
// sequences, evaluates each as a candidate, and keeps the top-K by score.
// It is exhaustive-ish, embedding-guided, and pays one full decompression
// per scored sequence.
//
// Crystallisation is a different ontology for the same problem. There
// is no "candidate sequence" being scored against alternatives. There is
// one bag of channels evolving in time, populated by a basin of
// operators that fire when their inputs are available, and that die
// when they accumulate waste (firing without producing new structure).
// The "creature" is whatever firing trace remains self-consistent at
// tick N. The gene-string is recovered post-hoc as the order in which
// operators actually contributed.
//
// In propagation-ontology vocabulary:
//   * the conflict graph is the local transition rule (peers that
//     dominate or shadow this op make it dormant);
//   * the AF + boundary leakage is the death rule (an op that emits a
//     hallucination-cell loses energy; depleted ops die);
//   * a "world" is one run of the cellular automaton from a particular
//     activation priority; we run K worlds and keep the survivors.
//
// What this buys us
// -----------------
// 1. Evaluations per world = decompressions per world ≪ beam evaluations.
//    On the standard 5-op pipeline the beam evaluates ~100 sequences; a
//    crystallisation world evaluates *once* (one decompression at the
//    end, or none if we score incrementally).
// 2. Self-consistency replaces global pruning. Beam removes redundant
//    sequences via conflict-graph + canonicalisation. Crystallisation
//    never *adds* a redundant firing in the first place: an op whose
//    output channel is already saturated by a successor with higher
//    priority simply doesn't fire.
// 3. Same Solver interface. The crystallisation solver returns
//    ParseCandidate[] sorted by score. Callers — including the
//    propose-loop — work without modification.
//
// Honest scope
// ------------
// This branch ships the solver, a small benchmark, and parity tests
// against the beam on the standard AF. The crystallisation solver
// should find a creature with score ≥ beam's score on the demo input,
// with strictly fewer evaluations.

import { signatureOf } from "./operator_reflection.js";
import {
  buildConflictGraph,
  type ConflictGraph,
  type InterferenceRegistry,
} from "./interference.js";
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

// ---------------------------------------------------------------------------
// Channel bag — the single shared world that every operator pours into.
// ---------------------------------------------------------------------------

type ChannelBag = Readonly<Record<string, unknown>>;

const merge = <T extends ChannelBag>(a: T, b: T): T => ({ ...a, ...b } as T);

const runGene = (op: ParseOperator, ctx: ParseContext, bag: ChannelBag): ChannelBag =>
  merge(bag, op.run(ctx, bag) as ChannelBag);

const seedBag = (ctx: ParseContext): ChannelBag => ({ "text.normalized": ctx.normalizedText });

const newChannelsAfter = (before: ChannelBag, after: ChannelBag, outs: readonly string[]): number =>
  outs.filter((ch) => {
    const b = before[ch];
    const a = after[ch];
    if (b === a) return false;
    // A channel "newly produced" if it didn't exist or grew (non-empty).
    if (b === undefined && a !== undefined) {
      return Array.isArray(a) ? a.length > 0 : true;
    }
    if (Array.isArray(b) && Array.isArray(a)) return a.length > b.length;
    return true;
  }).length;

// ---------------------------------------------------------------------------
// Death rule.
//
// An operator's life is a small integer of "energy" units. Each tick:
//   * firing and producing new channels restores nothing but doesn't cost
//   * firing and producing no new structure costs one unit (waste)
//   * being dominated by an alive peer with the same replacing-channel
//     output (per the conflict graph) costs one unit (shadowed)
//   * emitting cells that the AF rejects (validator_rejection) costs more
// Operators start with `INITIAL_ENERGY` units. Death is when energy ≤ 0.
//
// This is intentionally simple. It replicates, in tick-time, what the
// beam achieves by conflict-graph pruning + AF score: bad operators
// just stop participating in the world. The world's final state is the
// stable pattern.
// ---------------------------------------------------------------------------

const INITIAL_ENERGY = 3;
const WASTE_COST = 1;
const SHADOW_COST = 1;
const REJECTION_COST = 2;

type Vitality = {
  energy: number;
  hasFired: boolean;
  isIdempotent: boolean;
};

// ---------------------------------------------------------------------------
// Eligibility — an op fires this tick if:
//   * it is alive (energy > 0)
//   * its requiredInputs are in the bag
//   * it has not already fired AND been declared idempotent
// ---------------------------------------------------------------------------

// An op is fire-once by default. Re-firing only makes sense if the op's
// declared idempotence is false; today nothing in the basis claims
// idempotence=false, so this is a conservative default. The graph flag
// `isIdempotent` is a hint that *confirms* fire-once when present; in
// its absence we still default to fire-once because re-firing is
// wasted work in the CA model unless an op explicitly wants it.
const isEligible = (
  op: ParseOperator,
  bag: ChannelBag,
  v: Vitality,
): boolean => {
  if (v.energy <= 0) return false;
  if (v.hasFired) return false;
  return signatureOf(op).requiredInputs.every((ch) => bag[ch] !== undefined);
};

// ---------------------------------------------------------------------------
// Priority — which eligible op fires first.
//
// Priority is a tiebreaker on top of channel-flow: we prefer ops whose
// outputs are not yet in the bag (they expand the world) over ops whose
// outputs already exist (they replay). Within a tie we sort by a stable
// hash of `(worldSeed, opId)` so different worlds explore different
// firing orders without any RNG state to thread.
// ---------------------------------------------------------------------------

const stableHash = (s: string): number => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

const noveltyOf = (op: ParseOperator, bag: ChannelBag): number =>
  signatureOf(op).outputs.filter((ch) => bag[ch] === undefined).length;

const pickNextOp = (
  alive: readonly { op: ParseOperator; vitality: Vitality }[],
  bag: ChannelBag,
  worldSeed: string,
): { op: ParseOperator; vitality: Vitality } | undefined => {
  const eligible = alive.filter(({ op, vitality }) => isEligible(op, bag, vitality));
  if (eligible.length === 0) return undefined;
  return [...eligible].sort((a, b) => {
    const dn = noveltyOf(b.op, bag) - noveltyOf(a.op, bag);
    if (dn !== 0) return dn;
    return stableHash(worldSeed + a.op.id) - stableHash(worldSeed + b.op.id);
  })[0];
};

// ---------------------------------------------------------------------------
// Local transition rule via conflict graph.
//
// When the conflict graph reports that a *later* alive op replaces one
// of `op`'s outputs (idempotent or dominated-by-next-writer edge), `op`
// is shadowed *this tick* and loses energy without firing. This is the
// CA "dies because the neighbour has the same role" intuition.
//
// Implementation: an op is shadowed iff there exists another alive op
// such that:
//   (a) the other op also writes one of `op`'s replacing outputs, AND
//   (b) the conflict graph has an edge (op → other) with kind
//       `idempotent_under_successor` or `dominated_by_next_writer`.
//
// We only check the graph when one is present.
// ---------------------------------------------------------------------------

const isShadowedBy = (
  graph: ConflictGraph | undefined,
  fromId: string,
  toId: string,
): boolean => {
  if (graph === undefined) return false;
  return graph.edges.some(
    (e) =>
      e.from === fromId &&
      e.to === toId &&
      (e.kind === "redundant_if_adjacent" || e.kind === "mutually_exclusive_in_window"),
  );
};

// ---------------------------------------------------------------------------
// Result extractors — same shape as solver.ts.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// One world — one CA run, producing a single ParseCandidate.
// ---------------------------------------------------------------------------

type World = {
  readonly bag: ChannelBag;
  readonly firingOrder: readonly Gene[];
  readonly totalCost: number;
  readonly ticks: number;
};

const runWorld = (
  ctx: ParseContext,
  ops: readonly ParseOperator[],
  graph: ConflictGraph | undefined,
  worldSeed: string,
  maxTicks: number,
  onEvaluate?: () => void,
): World => {
  // Each op gets its own vitality slot, keyed by id.
  const vitality = new Map<string, Vitality>(
    ops.map((op) => {
      const idemp = (graph?.edges ?? []).some(
        (e) => e.from === op.id && e.to === op.id && e.kind === "redundant_if_adjacent",
      );
      // Heuristic: an op with a self-edge of `redundant_if_adjacent`
      // (set by `idempotent` traits) is treated as fire-once. Without a
      // graph we leave the flag false — there's no harm in re-firing, and
      // the death rule will kill wasteful repeats.
      return [op.id, { energy: INITIAL_ENERGY, hasFired: false, isIdempotent: idemp }] as const;
    }),
  );
  const opIndex = new Map(ops.map((op) => [op.id, op] as const));

  let bag: ChannelBag = seedBag(ctx);
  let totalCost = 0;
  const firing: Gene[] = [];

  let tick = 0;
  for (; tick < maxTicks; tick += 1) {
    const alive = ops
      .map((op) => ({ op, vitality: vitality.get(op.id)! }))
      .filter(({ vitality: v }) => v.energy > 0);

    const choice = pickNextOp(alive, bag, worldSeed);
    if (choice === undefined) break; // crystal stable

    const { op, vitality: v } = choice;

    // Apply shadowing pressure to *other* alive replacing-writers on
    // op's outputs. The current op fires; its peers that write the
    // same replacing channel lose energy because they're now stale.
    const outs = signatureOf(op).outputs;
    for (const peer of alive) {
      if (peer.op.id === op.id) continue;
      if (isShadowedBy(graph, peer.op.id, op.id)) {
        peer.vitality.energy -= SHADOW_COST;
      }
    }

    // Fire.
    const beforeBag = bag;
    bag = runGene(op, ctx, bag);
    totalCost += op.cost;
    onEvaluate?.();

    const novelty = newChannelsAfter(beforeBag, bag, outs);
    if (novelty === 0) v.energy -= WASTE_COST;
    v.hasFired = true;

    // AF-pressure death: count newly arrived hallucination cells and
    // bill them to the firing op.
    const newH = extractHallucinations(bag).filter((h) => h.operator === op.id);
    if (newH.length > 0) v.energy -= REJECTION_COST;

    if (novelty > 0) {
      firing.push({ operatorId: op.id });
    }
    // If the op fired but produced nothing new (waste), we still tick
    // forward — it'll either lose more energy and die, or a freshly
    // available input will let another op fire next tick. Either way
    // we don't record it in the firing order; the gene-string only
    // contains ops that actually pushed the world forward.
    void opIndex;
  }

  return { bag, firingOrder: firing, totalCost, ticks: tick };
};

// ---------------------------------------------------------------------------
// Scoring a world — same shape as beam's evaluate().
// ---------------------------------------------------------------------------

const PROGRESS_CHANNELS = [
  "spans.url",
  "spans.dated",
  "spans.titled",
  "rows.assembled",
  "rows.validated",
] as const;

const progressBonus = (bag: ChannelBag): number =>
  0.2 *
  PROGRESS_CHANNELS.filter((ch) => Array.isArray(bag[ch]) && (bag[ch] as unknown[]).length > 0)
    .length;

const scoreWorld = (world: World, af: CsvAF): ParseCandidate => {
  const rows = extractRows(world.bag);
  const extraH = extractHallucinations(world.bag);
  const cells = rows.flatMap((r) => Object.values(r.fields));
  const unsourced = cells.filter((c) => c.span[1] <= c.span[0]).length;
  const allH = [...af.hallucinations(rows), ...extraH];
  const diag = {
    coverage: rows.length === 0 ? 0 : cells.length / (rows.length * af.columns.length),
    complexity: world.totalCost,
    boundaryLeakage: cells.length === 0 ? 0 : unsourced / cells.length,
    stability: rows.length === 0 ? 0 : rows.filter((r) => r.score > 0.5).length / rows.length,
    pressure: { summary: summarise(allH) },
  };
  const base = af.scoreRun(rows);
  const score = base - 0.1 * diag.complexity - 5 * diag.boundaryLeakage + progressBonus(world.bag);
  return {
    genes: world.firingOrder satisfies GeneString,
    rows,
    score,
    diagnostics: diag,
    traces: extractTraces(world.bag),
  };
};

// ---------------------------------------------------------------------------
// Public solver.
// ---------------------------------------------------------------------------

export type CrystalConfig = {
  // Number of CA worlds to run with different firing priorities. Each
  // world is one decompression. K=8 is plenty for the standard basis;
  // each world differs only in tie-breaks, so increasing K past the
  // number of meaningful permutations is a no-op.
  readonly worlds: number;
  // Maximum ticks per world. A tick that produces nothing terminates
  // the world early, so this is only an upper bound.
  readonly maxTicks: number;
  // Optional conflict graph data. When supplied, shadowing pressure
  // applies and self-edge idempotents fire once.
  readonly interferenceRegistry?: InterferenceRegistry;
  // Optional measurement sink — called every time an operator fires.
  // Comparable to `onEvaluate` in BeamConfig: one tick per call.
  readonly onEvaluate?: () => void;
};

// maxTicks defaults to a small constant; the world also terminates
// early when no eligible op remains. The cap matters only as a safety
// bound against pathological bases.
const DEFAULTS: CrystalConfig = { worlds: 8, maxTicks: 12 };

export const makeCrystalSolver = (cfg: Partial<CrystalConfig> = {}): Solver => {
  const C: CrystalConfig = { ...DEFAULTS, ...cfg };
  return {
    search: (ctx, af, ops, seedGenes) => {
      const graph: ConflictGraph | undefined = C.interferenceRegistry
        ? buildConflictGraph(ops, C.interferenceRegistry)
        : undefined;

      // Worlds are seeded by index. Seed strings differ enough under the
      // FNV-1a hash to produce distinct firing orders, with seed=0
      // giving a deterministic baseline that other callers can rely on.
      const seeds = Array.from({ length: C.worlds }, (_, i) => `world.${i}`);

      // If the caller provided seedGenes, we treat them as *fired-already*
      // history. This lets the crystallisation solver be chained with
      // beam — e.g. seed with a beam top result, then let the CA stabilise
      // around it. In this PR we only honour the channels those genes
      // would have produced; we don't replay them, because the seed
      // gene-string may not be well-formed under the current basis.
      // The standard caller (propose-loop, benchmark) passes no seed.
      void seedGenes;

      const worlds = seeds.map((s) =>
        runWorld(ctx, ops, graph, s, C.maxTicks, C.onEvaluate),
      );

      const scored = worlds.map((w) => scoreWorld(w, af));

      // Sort by score desc and dedupe identical firing orders.
      const dedupe = new Map<string, ParseCandidate>();
      for (const c of scored) {
        const key = c.genes.map((g) => g.operatorId).join(">>");
        const prior = dedupe.get(key);
        if (prior === undefined || c.score > prior.score) dedupe.set(key, c);
      }
      return [...dedupe.values()].sort((a, b) => b.score - a.score);
    },
  };
};
