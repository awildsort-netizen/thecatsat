// Boundary repair via the propose() hook — one HallucinationKind end-to-end.
//
// What this module is for
// ------------------------
// The framework already emits typed hallucinations: a hallucination is
// not a scalar penalty, it is a typed boundary failure with a `kind`
// and (optionally) a `field`, `span`, and `note`. Each kind names a
// specific way the propagation can fail:
//
//   missing_emitter      — no propagator reaches a required column
//   field_role_confusion — two boundaries collided on the same span
//   overfit_pattern      — a propagator eats more than the table absorbs
//   validator_rejection  — a value crossed its boundary but the AF
//                          refused it
//   ...etc (seven kinds in types.ts).
//
// The propagation ontology says: leakage *classifies*, classification
// *proposes*. For each kind of boundary failure, there is a typed
// repair morphology. This module is the first end-to-end demonstration
// of that loop, for a single kind: `missing_emitter`.
//
// The repair morphology for `missing_emitter`
// -------------------------------------------
// Pressure: the AF says "no operator contributed to required column X".
// Diagnosis: the basis has no propagator that emits the channel
// associated with X (e.g. `spans.dated` for X="date").
// Repair: synthesise a `regex.emit.<X>` operator and add it to the
// basis. The conflict graph will pick up the new emitter
// automatically; the next solver run sees a propagator where there
// was previously a gap.
//
// We are *not* doing pattern synthesis from data in this PR. The
// regex library is keyed by column name. A real synthesis pass —
// generating the pattern from the AF's validator and from observed
// rejected cells — is the obvious next step and slots into the same
// `Proposer` interface without further plumbing.
//
// Scope of this branch
// --------------------
// One kind, one repair, end-to-end. The branch ships:
//   * A `Proposer` interface: (Hallucination, AF, ops) → operator?
//   * A `missingEmitterProposer` implementing it for `missing_emitter`
//   * A `runProposeLoop` driver that runs the solver, collects
//     hallucinations from the top creature, asks each proposer to
//     suggest repairs, adds the repairs to the basis, and re-runs.
//     Termination: when a round produces zero new operators.
//
// The loop terminates because each round either closes a boundary
// failure (reducing the set of failures the next round will see) or
// produces no repair (terminating).

import type {
  CsvAF,
  FieldHypothesis,
  Hallucination,
  ParseOperator,
  Span,
  TraceRegion,
} from "./types.js";

// ---------------------------------------------------------------------------
// Proposer interface.
//
// A proposer is a value: given a typed hallucination and the current
// basis, it returns either a new operator (a repair) or undefined
// (no repair available for this hallucination from this proposer).
//
// Proposers don't dispatch on `kind` themselves — the caller dispatches
// on `kind` by maintaining a per-kind registry. Each proposer is the
// repair morphology for exactly one kind. This is the same
// value-is-contribution discipline the rest of the codebase uses:
// the value's identity says what it repairs, not a tag inside it.
// ---------------------------------------------------------------------------

export type Proposer = {
  readonly handles: Hallucination["kind"];
  readonly propose: (h: Hallucination, af: CsvAF, ops: readonly ParseOperator[]) =>
    | ParseOperator
    | undefined;
};

// ---------------------------------------------------------------------------
// Default patterns keyed by column name.
//
// In a real synthesis pass these would come from the AF's validators
// (a column whose validator is `looksLikeDate` already encodes the
// regex shape) or from inverse-induction on rejected cells. Today they
// are just a small library that demonstrates the loop works
// end-to-end. The shape — `column → {pattern, group, confidence}` —
// is what a synthesiser would emit; we hand-author a few entries so
// the loop has something to demonstrate.
// ---------------------------------------------------------------------------

type EmitterRecipe = {
  readonly pattern: string;
  readonly group: number;
  readonly confidence: number;
};

const DEFAULT_RECIPES: Readonly<Record<string, EmitterRecipe>> = {
  date: {
    pattern: "(?:\\d{4}-\\d{2}-\\d{2})|(?:[A-Z][a-z]+\\s+\\d{1,2},\\s+\\d{4})",
    group: 0,
    confidence: 0.9,
  },
  url: {
    pattern: "https?:\\/\\/[^\\s)]+",
    group: 0,
    confidence: 0.95,
  },
  title: {
    pattern: "(?:^|\\n)([A-Z][^\\n]{9,139})(?=\\n|$)",
    group: 1,
    confidence: 0.6,
  },
};

// The conventional channel name an emitter for column X writes to. The
// naming is the same convention the existing operators follow:
// `spans.url`, `spans.dated`, `spans.titled`. We use the column name
// for the obvious ones; future proposers can pass a different
// convention.
const channelForField = (field: string): string => {
  if (field === "date") return "spans.dated";
  if (field === "title") return "spans.titled";
  if (field === "url") return "spans.url";
  return `spans.${field}`;
};

// ---------------------------------------------------------------------------
// Emitter synthesis.
//
// We build a fresh ParseOperator whose run function performs the
// same regex-extraction the hand-authored emitters do. The operator's
// channels are declared so the existing solver, conflict-graph, and
// derivation modules pick it up without modification — a synthesised
// emitter is just another operator with the right shape.
// ---------------------------------------------------------------------------

type MatchWithIndices = RegExpMatchArray & {
  indices?: ReadonlyArray<readonly [number, number] | undefined>;
};

const matchSpan = (m: MatchWithIndices, group: number): Span | undefined => {
  const idx = m.indices?.[group];
  return idx === undefined ? undefined : ([idx[0], idx[1]] as const);
};

const synthesiseEmitter = (field: string, recipe: EmitterRecipe): ParseOperator => {
  const channel = channelForField(field);
  const operatorId = `propose.regex.emit.${field}`;
  const run = (
    _ctx: unknown,
    input: { "text.normalized": string; "trace.regions"?: readonly TraceRegion[] },
  ): Record<string, unknown> => {
    const re = new RegExp(recipe.pattern, "gd");
    const matches = Array.from(input["text.normalized"].matchAll(re) as Iterable<MatchWithIndices>);
    const hits: FieldHypothesis[] = [];
    for (const m of matches) {
      const span = matchSpan(m, recipe.group);
      const value = m[recipe.group];
      if (span === undefined || value === undefined) continue;
      hits.push({
        field,
        value: value.trim(),
        span,
        operator: operatorId,
        confidence: recipe.confidence,
        evidence: `propose:regex:${recipe.pattern.slice(0, 24)}`,
        traceRegionId: `${operatorId}@${channel}#${span[0]}:${span[1]}`,
      });
    }
    const newTraces: readonly TraceRegion[] = hits.map((h) => ({
      id: h.traceRegionId!,
      label: `propose:${field}:${h.value.slice(0, 20)}`,
      span: h.span,
      channel,
      operator: operatorId,
    }));
    return {
      [channel]: hits,
      "trace.regions": [...(input["trace.regions"] ?? []), ...newTraces],
    };
  };
  // Cast through `unknown` because we are constructing a concrete
  // operator with computed channel keys; the caller will see it as
  // ParseOperator<any, any>, which is the documented escape hatch in
  // operator_reflection.ts for runtime-constructed operators.
  return {
    id: operatorId,
    cost: 2,
    tokens: ["regex", "extract", field, "propose", "synthesise"],
    run: run as never,
    channels: {
      requiredInputs: ["text.normalized"],
      optionalInputs: ["trace.regions"],
      outputs: [channel, "trace.regions"],
    },
  } as unknown as ParseOperator;
};

// ---------------------------------------------------------------------------
// missingEmitterProposer.
//
// Repair morphology for `missing_emitter`: look up the column's recipe
// and synthesise a `propose.regex.emit.<field>` operator. Returns
// undefined when:
//   * the hallucination has no `field` (mis-typed entry)
//   * no recipe is available for that field name
//   * the basis already contains an emitter for the channel we'd
//     produce (avoid spamming duplicates across rounds)
// ---------------------------------------------------------------------------

const channelEmittedBy = (op: ParseOperator): readonly string[] =>
  op.channels.outputs.filter((c) => c.startsWith("spans."));

const basisAlreadyEmits = (channel: string, ops: readonly ParseOperator[]): boolean =>
  ops.some((op) => channelEmittedBy(op).includes(channel));

export const missingEmitterProposer: Proposer = {
  handles: "missing_emitter",
  propose: (h, _af, ops) => {
    if (h.field === undefined) return undefined;
    const recipe = DEFAULT_RECIPES[h.field];
    if (recipe === undefined) return undefined;
    const channel = channelForField(h.field);
    if (basisAlreadyEmits(channel, ops)) return undefined;
    return synthesiseEmitter(h.field, recipe);
  },
};

// ---------------------------------------------------------------------------
// runProposeLoop — the driver.
//
// Given a solver, a context, an AF, a starting operator set, and a
// list of proposers, run the solver, collect hallucinations from the
// top creature, ask each proposer to repair each hallucination it
// handles, add accepted repairs to the basis, and re-run.
//
// Termination: a round that adds zero operators ends the loop. This
// is guaranteed in practice because:
//   * each proposer is responsible for refusing duplicates (see
//     `basisAlreadyEmits` above);
//   * the only kinds with proposers in this PR can each be repaired
//     at most once per (kind, field) pair.
// A safety bound `maxRounds` guards against bugs in either invariant.
// ---------------------------------------------------------------------------

export type ProposeLoopResult = {
  readonly rounds: number;
  readonly addedOperators: readonly string[]; // ids, in the order added
  readonly finalOps: readonly ParseOperator[];
};

import type {
  ParseCandidate,
  ParseContext,
  Solver,
} from "./types.js";

// ---------------------------------------------------------------------------
// Structural hallucinations.
//
// The AF surfaces `missing_emitter` only when *some row exists* and a required
// column is absent from it. That works as long as the broken basis still
// assembles rows — but when the missing column is the assembler's anchor
// (e.g. `date` for proximity assembly), the run produces zero rows and the
// row-evidence diagnostic vanishes.
//
// The propagation ontology disagrees: the boundary failure is geometric,
// not statistical. "No propagator reaches required column X" is visible in
// the operator graph (no op outputs `spans.X`) regardless of what the run
// happened to produce. So we read the graph directly and synthesise the
// missing_emitter hallucinations the AF would surface if it could.
//
// This is the *repair morphology's* job, not the AF's. The AF describes
// pressure observed in the run; the proposer is allowed to read structural
// pressure too. We merge both sources before dispatching, deduping by
// (kind, field).
// ---------------------------------------------------------------------------

const structuralMissingEmitters = (
  af: CsvAF,
  ops: readonly ParseOperator[],
): readonly Hallucination[] => {
  const required = af.columns.filter((c) => c.required).map((c) => c.name);
  return required
    .filter((field) => !basisAlreadyEmits(channelForField(field), ops))
    .map<Hallucination>((field) => ({
      kind: "missing_emitter",
      field,
      weight: 2,
      note: `structural: no operator emits channel ${channelForField(field)} for required column ${field}`,
    }));
};

const dedupeHallucinations = (
  hs: readonly Hallucination[],
): readonly Hallucination[] => {
  const seen = new Set<string>();
  const out: Hallucination[] = [];
  for (const h of hs) {
    const key = `${h.kind}::${h.field ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(h);
  }
  return out;
};

export const runProposeLoop = (
  solver: Solver,
  ctx: ParseContext,
  af: CsvAF,
  initialOps: readonly ParseOperator[],
  proposers: readonly Proposer[],
  maxRounds = 4,
): ProposeLoopResult => {
  let ops: readonly ParseOperator[] = initialOps;
  const added: string[] = [];

  for (let round = 0; round < maxRounds; round += 1) {
    const candidates = solver.search(ctx, af, ops);
    const top: ParseCandidate | undefined = candidates[0];
    const rows = top?.rows ?? [];
    const hallucinations = dedupeHallucinations([
      ...af.hallucinations(rows),
      ...structuralMissingEmitters(af, ops),
    ]);

    // Find new operators this round, deduping by id.
    const newOpsThisRound: ParseOperator[] = [];
    const seenIds = new Set(ops.map((o) => o.id));
    for (const h of hallucinations) {
      for (const p of proposers) {
        if (p.handles !== h.kind) continue;
        const candidate = p.propose(h, af, ops);
        if (candidate === undefined) continue;
        if (seenIds.has(candidate.id)) continue;
        seenIds.add(candidate.id);
        newOpsThisRound.push(candidate);
      }
    }

    if (newOpsThisRound.length === 0) {
      return { rounds: round, addedOperators: added, finalOps: ops };
    }
    newOpsThisRound.forEach((op) => added.push(op.id));
    ops = [...ops, ...newOpsThisRound];
  }
  return { rounds: maxRounds, addedOperators: added, finalOps: ops };
};
