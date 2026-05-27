// Operator interference and conflict graph.
//
// Motivation
// ----------
// Today the solver reasons about operators in two ways: (1) the implicit
// data-flow graph encoded by `channels.requiredInputs/outputs`, and
// (2) an embedding-driven similarity to AF tokens. Both are necessary
// and neither is sufficient. The data-flow graph tells us when an
// operator *can* run, not whether running it is *useful*; the embedding
// tells us which operators look relevant to an AF, but it can't see
// that two operators write the same channel with different semantics,
// or that running an idempotent operator twice is wasted work.
//
// `InterferenceSpec` lifts the algebra of operators from "implicit and
// runtime-only" to "declared and inspectable". It is *additive* — every
// existing operator keeps working without an interference spec; the
// module supplies a conservative default ("assume worst") for operators
// that haven't been characterised.
//
// Design philosophy: shape the space, don't micromanage the code
// --------------------------------------------------------------
// A traits system invites the bad shape: tagged unions with a `kind`
// discriminator and a central `switch` that maps tag → behaviour.
// That recreates the parallel ontology the rest of this codebase
// refuses (see `operator_reflection.ts`: the operator's run function
// type IS its signature).
//
// Instead, every trait value here *is its own contribution to the
// analyses*. The composer (= `buildConflictGraph`, `basisCoherence`,
// `redundanciesIn`) declares a small fixed set of folds — sinks for
// replacing channels, accumulator channels, self-edges, commutation
// offers — and iterates traits, calling each one's contribution
// methods. The trait values pour into the sinks. There is no
// `switch (kind)`, no central dispatch table, no `assertNever`.
// Adding a new trait variant is a one-line constructor that defines
// what it pours; existing composers are untouched.
//
// The trade-off: this style is less grep-able. You can't ask "what
// are the Application variants?" by grepping for `kind:`. The
// vocabulary lives in the constructor names (`idempotent`,
// `purelyAdditive`, `commutesWithPeers`) and in the contribution
// method signatures. In return, every contribution lives next to the
// constructor that produces it — value and behaviour are co-located.
//
// Channel semantics
// -----------------
// Two write-disciplines, both expressed as composable values:
//   * accumulator — writes append (or monoidally merge). Two
//                    accumulators to the same channel do not conflict.
//   * replacing   — writes overwrite. Two replacing-writers to the
//                    same channel adjacently make the first dominated.

import type { ParseOperator } from "./operator_reflection.js";
import type { CsvAF } from "./types.js";

// ---------------------------------------------------------------------------
// Composer sinks — the small fixed surface every contribution pours into.
//
// `ConflictSinks` is mutable-by-design: traits pour into it during a
// single composer pass. The composer wraps it, drains it, and returns
// an immutable ConflictGraph. No consumer of the public API ever sees
// a sink.
// ---------------------------------------------------------------------------

type ConflictSinks = {
  // Channels written by this operator under each discipline. Filled by
  // WriteDiscipline contributions.
  readonly replacingChannels: Set<string>;
  readonly accumulatorChannels: Set<string>;
  // Edges this operator's traits want to register *on its own behalf*
  // (e.g. an idempotence self-loop). Filled by Application
  // contributions.
  readonly selfEdges: ConflictEdge[];
  // Does this operator offer the commutation half-permission? Filled
  // by Commutation contributions.
  readonly offersCommutation: { value: boolean };
};

const emptySinks = (): ConflictSinks => ({
  replacingChannels: new Set<string>(),
  accumulatorChannels: new Set<string>(),
  selfEdges: [],
  offersCommutation: { value: false },
});

// ---------------------------------------------------------------------------
// WriteDiscipline — value-is-contribution.
//
// A WriteDiscipline knows three things about itself:
//   * which channel it concerns (for inspection / debugging / metrics);
//   * how to contribute its replacing-write semantics;
//   * how to contribute its accumulator-write semantics.
//
// The constructors `replacing()` and `accumulator()` produce values
// whose contributions are precisely the semantics named. The composer
// holds a list of these and calls every contribution; each value
// either contributes meaningfully or contributes a no-op. There is no
// branching on what kind of discipline it is.
// ---------------------------------------------------------------------------

export type WriteDiscipline = {
  readonly channel: string;
  readonly contributeReplacing: (sink: Set<string>) => void;
  readonly contributeAccumulator: (sink: Set<string>) => void;
};

export const replacing = (channel: string): WriteDiscipline => ({
  channel,
  contributeReplacing: (sink) => {
    sink.add(channel);
  },
  contributeAccumulator: () => {},
});

export const accumulator = (channel: string): WriteDiscipline => ({
  channel,
  contributeReplacing: () => {},
  contributeAccumulator: (sink) => {
    sink.add(channel);
  },
});

// ---------------------------------------------------------------------------
// Application — value-is-contribution.
//
// An Application's contribution is the set of self-edges it implies
// on its operator. `idempotent` and `purelyAdditive` both contribute
// the same redundant-if-adjacent self-loop today; they remain
// *distinct values* because future composers may treat them
// differently (e.g. a propose() pass that tries to upgrade idempotent
// emitters to purely-additive ones can read the trait's identity
// without needing a separate label).
//
// `nonIdempotent` is the conservative default: a value that
// contributes nothing.
// ---------------------------------------------------------------------------

export type Application = {
  readonly contributeSelfEdges: (opId: string, sink: ConflictEdge[]) => void;
};

export const idempotent: Application = {
  contributeSelfEdges: (opId, sink) => {
    sink.push({ from: opId, to: opId, kind: "redundant_if_adjacent" });
  },
};

export const purelyAdditive: Application = {
  contributeSelfEdges: (opId, sink) => {
    sink.push({ from: opId, to: opId, kind: "redundant_if_adjacent" });
  },
};

export const nonIdempotent: Application = {
  contributeSelfEdges: () => {},
};

// ---------------------------------------------------------------------------
// Commutation — value-is-contribution.
//
// A Commutation contributes a single boolean to its operator's sink:
// "do I offer the half-permission to commute with a peer?". The
// composer pairs up two half-permissions when minting `commutes`
// edges, after filtering out pairs that share a must_precede
// relationship.
//
// We keep this as composable values rather than a single boolean
// field on the spec so that, in the next pass, finer commutation
// claims (e.g. "I commute only with operators in set S") can be
// expressed without changing the consumer surface.
// ---------------------------------------------------------------------------

export type Commutation = {
  readonly contributeOffer: (sink: { value: boolean }) => void;
};

export const commutesWithPeers: Commutation = {
  contributeOffer: (sink) => {
    sink.value = true;
  },
};

export const nonCommuting: Commutation = {
  contributeOffer: () => {},
};

// ---------------------------------------------------------------------------
// InterferenceSpec — a bag of contributions.
//
// The spec has no `kind`-shaped fields. It is a list of write-
// disciplines plus a single Application and a single Commutation
// (since application and commutation are operator-wide properties).
// The composer iterates `outputs` for write-discipline contributions
// and calls `application` and `commutation` once each.
//
// `DEFAULT_INTERFERENCE` is the conservative value: a spec whose
// every contribution is a no-op. The composer treats it identically
// to a spec that wasn't supplied.
// ---------------------------------------------------------------------------

export type InterferenceSpec = {
  readonly outputs: readonly WriteDiscipline[];
  readonly application: Application;
  readonly commutation: Commutation;
};

export const DEFAULT_INTERFERENCE: InterferenceSpec = {
  outputs: [],
  application: nonIdempotent,
  commutation: nonCommuting,
};

// ---------------------------------------------------------------------------
// Side-table registry.
//
// Transitional: until operators self-declare their interference
// directly on the ParseOperator value, the registry holds the bag of
// specs by operator id. The shape above is already prepared for the
// migration — an operator could carry `interference: InterferenceSpec`
// and the registry becomes a thin compatibility shim.
// ---------------------------------------------------------------------------

export type InterferenceRegistry = ReadonlyMap<string, InterferenceSpec>;

export const lookupInterference = (
  registry: InterferenceRegistry,
  opId: string,
): InterferenceSpec => registry.get(opId) ?? DEFAULT_INTERFERENCE;

export const makeInterferenceRegistry = (
  entries: readonly (readonly [string, InterferenceSpec])[],
): InterferenceRegistry => new Map(entries);

// Pour an InterferenceSpec into a fresh sinks bag. This is the *only*
// place in the module where a spec is unpacked into its primitive
// contributions; everything downstream reads from the sinks.
const pourSpec = (opId: string, spec: InterferenceSpec): ConflictSinks => {
  const sinks = emptySinks();
  spec.outputs.forEach((o) => {
    o.contributeReplacing(sinks.replacingChannels);
    o.contributeAccumulator(sinks.accumulatorChannels);
  });
  spec.application.contributeSelfEdges(opId, sinks.selfEdges);
  spec.commutation.contributeOffer(sinks.offersCommutation);
  return sinks;
};

// Helpers used by the basis-coherence metric. These return the
// channel sets *after* pouring; they exist so callers don't have to
// build sinks themselves.
export const replacingChannelsOf = (spec: InterferenceSpec): readonly string[] => {
  const sink = new Set<string>();
  spec.outputs.forEach((o) => o.contributeReplacing(sink));
  return Array.from(sink);
};

export const accumulatorChannelsOf = (spec: InterferenceSpec): readonly string[] => {
  const sink = new Set<string>();
  spec.outputs.forEach((o) => o.contributeAccumulator(sink));
  return Array.from(sink);
};

// ---------------------------------------------------------------------------
// ConflictGraph
// ---------------------------------------------------------------------------

export type ConflictEdgeKind =
  | "must_precede"
  // A.must_precede(B) when B reads a channel A writes. Encodes the
  // data-flow ordering already implicit in channels; making it explicit
  // lets the solver detect impossible orderings before evaluation.
  | "redundant_if_adjacent"
  // op X op when op contributes a redundant-if-adjacent self-edge.
  | "mutually_exclusive_in_window"
  // A and B both write the same replacing channel; whichever runs
  // second strictly dominates the first within the same beam path.
  | "commutes";
  // Adjacent pair (A, B) may be reordered without changing the bag.

export type ConflictEdge = {
  readonly from: string;
  readonly to: string;
  readonly kind: ConflictEdgeKind;
  readonly channel?: string;
};

export type ConflictGraph = {
  readonly operators: readonly string[];
  readonly edges: readonly ConflictEdge[];
  readonly outgoing: ReadonlyMap<string, readonly ConflictEdge[]>;
  readonly incoming: ReadonlyMap<string, readonly ConflictEdge[]>;
};

const groupBy = <K, V>(items: readonly V[], key: (v: V) => K): ReadonlyMap<K, readonly V[]> => {
  const out = new Map<K, V[]>();
  items.forEach((v) => {
    const k = key(v);
    const list = out.get(k);
    if (list === undefined) out.set(k, [v]);
    else list.push(v);
  });
  return out;
};

// The composer. Note the absence of any `switch` or `kind` inspection:
// every per-trait decision is made by the trait value itself pouring
// into a sink.
export const buildConflictGraph = (
  ops: readonly ParseOperator[],
  registry: InterferenceRegistry,
): ConflictGraph => {
  const edges: ConflictEdge[] = [];

  // Pour every operator's spec into a per-operator sink. We hold the
  // sinks keyed by opId so subsequent passes (mutually-exclusive,
  // commutation) can read the contributions back out.
  const sinksByOp = new Map<string, ConflictSinks>(
    ops.map((op) => [op.id, pourSpec(op.id, lookupInterference(registry, op.id))] as const),
  );

  // Data-flow ordering: A.must_precede(B) when B requires a channel A
  // outputs. Read directly from ParseOperator.channels — the
  // signatures-first source of truth.
  ops.forEach((producer) => {
    const outputs = new Set(producer.channels.outputs);
    ops.forEach((consumer) => {
      if (consumer.id === producer.id) return;
      consumer.channels.requiredInputs.forEach((chan) => {
        if (outputs.has(chan)) {
          edges.push({ from: producer.id, to: consumer.id, kind: "must_precede", channel: chan });
        }
      });
    });
  });

  // Self-edges: each operator's Application has already poured its
  // self-edges into the per-op sink; flush them into the main edge
  // list.
  sinksByOp.forEach((sink) => {
    sink.selfEdges.forEach((e) => edges.push(e));
  });

  // Mutually-exclusive-in-window: bucket operators by the replacing
  // channels they declared (also already poured by WriteDiscipline
  // contributions).
  const replacingWriters = new Map<string, string[]>();
  sinksByOp.forEach((sink, opId) => {
    sink.replacingChannels.forEach((chan) => {
      const list = replacingWriters.get(chan) ?? [];
      list.push(opId);
      replacingWriters.set(chan, list);
    });
  });
  replacingWriters.forEach((writers, chan) => {
    writers.forEach((a) =>
      writers.forEach((b) => {
        if (a !== b) edges.push({ from: a, to: b, kind: "mutually_exclusive_in_window", channel: chan });
      }),
    );
  });

  // Commutation: pairs where both ops poured an offer AND no
  // must_precede edge connects them in either direction.
  const mustPrecedePairs = new Set<string>(
    edges.filter((e) => e.kind === "must_precede").map((e) => `${e.from}->${e.to}`),
  );
  const offers = (opId: string): boolean =>
    sinksByOp.get(opId)?.offersCommutation.value === true;

  ops.forEach((a) =>
    ops.forEach((b) => {
      if (a.id >= b.id) return;
      if (!offers(a.id) || !offers(b.id)) return;
      const feedsAtoB = mustPrecedePairs.has(`${a.id}->${b.id}`);
      const feedsBtoA = mustPrecedePairs.has(`${b.id}->${a.id}`);
      if (feedsAtoB || feedsBtoA) return;
      edges.push({ from: a.id, to: b.id, kind: "commutes" });
      edges.push({ from: b.id, to: a.id, kind: "commutes" });
    }),
  );

  return {
    operators: ops.map((o) => o.id),
    edges,
    outgoing: groupBy(edges, (e) => e.from),
    incoming: groupBy(edges, (e) => e.to),
  };
};

// ---------------------------------------------------------------------------
// Gene-string analysis using the conflict graph
// ---------------------------------------------------------------------------

export type Redundancy = {
  readonly position: number;
  readonly opId: string;
  readonly reason: "idempotent_repeat" | "dominated_by_next_writer";
  readonly channel?: string;
};

export const redundanciesIn = (
  geneIds: readonly string[],
  graph: ConflictGraph,
): readonly Redundancy[] => {
  const edgeBetween = (from: string, to: string, kind: ConflictEdgeKind): ConflictEdge | undefined =>
    graph.outgoing.get(from)?.find((e) => e.to === to && e.kind === kind);

  const out: Redundancy[] = [];
  for (let i = 0; i < geneIds.length - 1; i++) {
    const a = geneIds[i]!;
    const b = geneIds[i + 1]!;

    if (a === b && edgeBetween(a, a, "redundant_if_adjacent")) {
      out.push({ position: i + 1, opId: b, reason: "idempotent_repeat" });
      continue;
    }
    const muex = edgeBetween(a, b, "mutually_exclusive_in_window");
    if (muex !== undefined) {
      out.push({
        position: i,
        opId: a,
        reason: "dominated_by_next_writer",
        channel: muex.channel,
      });
    }
  }
  return out;
};

// Canonicalise a gene-string under known commutations. Adjacent
// commuting operators are sorted by id so that gene-strings differing
// only by commuting reorderings deduplicate to the same key.
export const canonicaliseUnderCommutation = (
  geneIds: readonly string[],
  graph: ConflictGraph,
): readonly string[] => {
  const commutes = (a: string, b: string): boolean =>
    (graph.outgoing.get(a) ?? []).some((e) => e.to === b && e.kind === "commutes");

  const out = [...geneIds];
  for (let i = 0; i < out.length - 1; i++) {
    if (commutes(out[i]!, out[i + 1]!) && out[i]! > out[i + 1]!) {
      [out[i], out[i + 1]] = [out[i + 1]!, out[i]!];
    }
  }
  return out;
};

// ---------------------------------------------------------------------------
// Basis-coherence metric
// ---------------------------------------------------------------------------
//
// A second-order signal: scores the operator set + AF *as a whole*,
// independent of any particular creature. This is the metric the
// propose() loop would optimise against when synthesising new
// operators — closing a coverage gap raises coherence, producing dead
// channels lowers it.

export type BasisCoherence = {
  readonly coverage: number;
  readonly waste: number;
  readonly conflictDensity: number;
  readonly coherence: number;
  readonly uncoveredColumns: readonly string[];
  readonly orphanChannels: readonly string[];
};

// "Channel emits column C": the channel name contains the column name.
// Deliberately weak — relies on the existing channel-naming convention.
// A channel `a.b.c` is treated as a dotted path of tokens. A column is
// "likely emitted" by that channel iff one of the path tokens *starts*
// with the column name (e.g. `dated` covers `date`, `titled` covers
// `title`, `url` covers `url`). This is intentionally weak but avoids
// the false positive of `rows.validated` matching `date` via substring.
const channelLikelyEmitsColumn = (channel: string, column: string): boolean => {
  const col = column.toLowerCase();
  return channel
    .toLowerCase()
    .split(".")
    .some((tok) => tok.startsWith(col));
};

export const basisCoherence = (
  ops: readonly ParseOperator[],
  af: CsvAF,
  registry: InterferenceRegistry,
): BasisCoherence => {
  const graph = buildConflictGraph(ops, registry);

  const allInputs = new Set<string>();
  const allOutputs = new Set<string>();
  ops.forEach((op) => {
    op.channels.requiredInputs.forEach((c) => allInputs.add(c));
    op.channels.optionalInputs.forEach((c) => allInputs.add(c));
    op.channels.outputs.forEach((c) => allOutputs.add(c));
  });

  // Coverage.
  const hasRowProducer = ops.some(
    (op) => op.channels.outputs.includes("rows.assembled") || op.channels.outputs.includes("rows.validated"),
  );
  const uncovered: string[] = [];
  const requiredColumns = af.columns.filter((c) => c.required ?? true);
  const columnsCovered = requiredColumns.filter((col) => {
    const emitterExists = ops.some((op) =>
      op.channels.outputs.some((chan) => channelLikelyEmitsColumn(chan, col.name)),
    );
    if (!emitterExists) uncovered.push(col.name);
    return emitterExists;
  });
  const coverage =
    requiredColumns.length === 0
      ? 1
      : hasRowProducer
        ? columnsCovered.length / requiredColumns.length
        : 0;

  // Waste: channels output but never read. Accumulator outputs are
  // explicitly side-channels meant for the *caller*, not for other
  // operators downstream in the gene-string, so they don't count as
  // waste. The accumulator set is derived from the same value-pouring
  // mechanism the composer uses.
  const TERMINAL_CHANNELS = new Set(["rows.validated", "rows.assembled", "hallucinations.collected"]);
  const accumulatorOutputs = new Set<string>();
  ops.forEach((op) =>
    accumulatorChannelsOf(lookupInterference(registry, op.id)).forEach((c) => accumulatorOutputs.add(c)),
  );
  const orphans: string[] = [];
  allOutputs.forEach((chan) => {
    if (allInputs.has(chan)) return;
    if (TERMINAL_CHANNELS.has(chan)) return;
    if (accumulatorOutputs.has(chan)) return;
    orphans.push(chan);
  });
  const waste = allOutputs.size === 0 ? 0 : orphans.length / allOutputs.size;

  // Conflict density.
  const muexCount = graph.edges.filter((e) => e.kind === "mutually_exclusive_in_window").length;
  const conflictDensity = ops.length === 0 ? 0 : muexCount / ops.length;

  const coherence = coverage * (1 - waste) * Math.exp(-conflictDensity);

  return {
    coverage,
    waste,
    conflictDensity,
    coherence,
    uncoveredColumns: uncovered,
    orphanChannels: orphans,
  };
};

export const summariseCoherence = (b: BasisCoherence): string =>
  `coherence=${b.coherence.toFixed(3)} (coverage=${b.coverage.toFixed(2)}, waste=${b.waste.toFixed(2)}, conflict=${b.conflictDensity.toFixed(2)})`;
