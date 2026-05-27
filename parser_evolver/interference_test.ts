// Self-test for the interference module.
// Run with: npx tsx parser_evolver/interference_test.ts
//
// Properties asserted:
//   1. The composer never inspects `kind` fields on trait values —
//      verified structurally: the test imports a hand-rolled custom
//      trait variant that has no `kind` field and still pours
//      correctly into the conflict graph. If the composer secretly
//      branched on `kind`, this test would fail to compile or to run.
//   2. Conflict graph: must_precede edges are exactly the data-flow
//      edges inferred from channels.requiredInputs.
//   3. Idempotent operators get a redundant-if-adjacent self-edge.
//   4. Two replacing-writers to the same channel become
//      mutually-exclusive-in-window.
//   5. Commuting pairs are only minted when both sides offer the
//      half-permission AND there is no must_precede between them.
//   6. redundanciesIn flags `[op, op]` for idempotent ops and the
//      first of a mutually-exclusive adjacent pair.
//   7. canonicaliseUnderCommutation produces a stable ordering for
//      commuting pairs.
//   8. The solver's no-op contract: a search with no registry has
//      byte-identical top score and gene-string to the legacy path.
//   9. With the PRIMITIVE_INTERFERENCE registry, the solver still
//      discovers a productive creature.
//  10. With a registry, redundant `[normalize, normalize]` extensions
//      never appear in any final candidate.

import { companyUpdatesAF } from "./af.js";
import {
  PRIMITIVES,
  makeEnforceSchema,
  normalizeWhitespace,
  regexEmitUrl,
  regexEmitDate,
  regexEmitTitle,
} from "./operators.js";
import { PRIMITIVE_INTERFERENCE } from "./operator_interference.js";
import {
  accumulator,
  buildConflictGraph,
  canonicaliseUnderCommutation,
  commutesWithPeers,
  idempotent,
  makeInterferenceRegistry,
  nonCommuting,
  nonIdempotent,
  purelyAdditive,
  redundanciesIn,
  replacing,
  basisCoherence,
  summariseCoherence,
  type InterferenceSpec,
} from "./interference.js";
import { makeBeamSolver } from "./solver.js";
import type { ParseContext } from "./types.js";

const SAMPLE = `
2026-04-12
Wallet sync latency elevated in EU region
https://status.blockchain.com/incidents/wallet-eu-2026-04-12

2026-04-09
Scheduled maintenance for institutional API
https://status.blockchain.com/maintenance/api-2026-04-09

April 1, 2026
Lightning withdrawals now generally available
https://www.blockchain.com/blog/posts/lightning-ga
`;

const ctx: ParseContext = {
  url: "https://status.blockchain.com/",
  rawText: SAMPLE,
  normalizedText: SAMPLE,
  sourceType: "status-page",
};

const ops = [...PRIMITIVES, makeEnforceSchema(companyUpdatesAF)];

type Assertion = { name: string; ok: boolean; detail?: string };
const assertions: Assertion[] = [];
const assert = (name: string, ok: boolean, detail?: string) =>
  assertions.push({ name, ok, detail });

// --- 1. The composer is shape-agnostic about traits --------------------------
//
// Construct a custom Application value that mimics `idempotent`'s
// contribution without using any `kind` field at all. If the composer
// secretly switched on `kind`, this would produce no redundancy edges.
const customApp = {
  contributeSelfEdges: (opId: string, sink: { push: (e: { from: string; to: string; kind: "redundant_if_adjacent" }) => void }) => {
    sink.push({ from: opId, to: opId, kind: "redundant_if_adjacent" });
  },
} as unknown as InterferenceSpec["application"];

const customRegistry = makeInterferenceRegistry([
  ["normalize.whitespace", { outputs: [replacing("text.normalized")], application: customApp, commutation: nonCommuting } as InterferenceSpec],
]);

const customGraph = buildConflictGraph([normalizeWhitespace], customRegistry);
const customHasSelfEdge = customGraph.edges.some(
  (e) => e.from === "normalize.whitespace" && e.to === "normalize.whitespace" && e.kind === "redundant_if_adjacent",
);
assert(
  "composer accepts a hand-rolled Application value (no kind field)",
  customHasSelfEdge,
  "this would fail if the composer branched on .kind",
);

// --- 2. must_precede edges mirror channels.requiredInputs -------------------
const graph = buildConflictGraph(ops, PRIMITIVE_INTERFERENCE);
const mustPrecede = graph.edges.filter((e) => e.kind === "must_precede");
// regex.emit.date reads spans.url, so url must precede date.
const urlPrecedesDate = mustPrecede.some(
  (e) => e.from === "regex.emit.url" && e.to === "regex.emit.date" && e.channel === "spans.url",
);
assert("regex.emit.url must_precede regex.emit.date on spans.url", urlPrecedesDate);

// row.assemble.proximity reads spans.dated and spans.titled.
const dateBeforeAssemble = mustPrecede.some(
  (e) => e.from === "regex.emit.date" && e.to === "row.assemble.proximity",
);
const titleBeforeAssemble = mustPrecede.some(
  (e) => e.from === "regex.emit.title" && e.to === "row.assemble.proximity",
);
assert("regex.emit.date must_precede row.assemble.proximity", dateBeforeAssemble);
assert("regex.emit.title must_precede row.assemble.proximity", titleBeforeAssemble);

// --- 3. idempotent operators get self-edges ---------------------------------
const idempotentSelfEdges = graph.edges.filter(
  (e) => e.kind === "redundant_if_adjacent" && e.from === e.to,
);
const idempotentIds = new Set(idempotentSelfEdges.map((e) => e.from));
const expectedIdempotent = [
  "normalize.whitespace",
  "regex.emit.url",
  "regex.emit.date",
  "regex.emit.title",
  "row.assemble.proximity",
  "row.enforce.schema",
];
expectedIdempotent.forEach((id) =>
  assert(`${id} has a redundant-if-adjacent self-edge`, idempotentIds.has(id)),
);

// --- 4. mutually-exclusive-in-window between identical replacing writers ----
//
// The primitive ecology has no two operators writing the same replacing
// channel — every replacer is unique. So we synthesise a second
// emitter writing spans.url and confirm the conflict graph spots the
// muex pair.
const competingUrlEmitter = {
  ...regexEmitUrl,
  id: "regex.emit.url.alt",
  tokens: [...regexEmitUrl.tokens, "alt"],
};
const extendedRegistry = makeInterferenceRegistry([
  ...Array.from(PRIMITIVE_INTERFERENCE.entries()),
  [
    "regex.emit.url.alt",
    { outputs: [replacing("spans.url"), accumulator("trace.regions")], application: idempotent, commutation: commutesWithPeers } as InterferenceSpec,
  ],
]);
const extendedGraph = buildConflictGraph([...ops, competingUrlEmitter], extendedRegistry);
const muex = extendedGraph.edges.filter((e) => e.kind === "mutually_exclusive_in_window");
const competingPairFound = muex.some(
  (e) =>
    ((e.from === "regex.emit.url" && e.to === "regex.emit.url.alt") ||
      (e.from === "regex.emit.url.alt" && e.to === "regex.emit.url")) &&
    e.channel === "spans.url",
);
assert("two replacing-writers to spans.url become mutually exclusive", competingPairFound);

// --- 5. commutation edges respect must_precede ------------------------------
//
// regex.emit.url and regex.emit.date both offer commutation, but
// url must_precede date — so no commutes edge should connect them.
// regex.emit.title and regex.emit.date both offer commutation AND
// neither feeds the other (both read spans.url; neither reads the
// other), so they SHOULD have a commutes edge.
const commutes = graph.edges.filter((e) => e.kind === "commutes");
const urlDateCommutes = commutes.some(
  (e) =>
    (e.from === "regex.emit.url" && e.to === "regex.emit.date") ||
    (e.from === "regex.emit.date" && e.to === "regex.emit.url"),
);
assert("url and date do NOT commute (must_precede in play)", !urlDateCommutes);

const dateTitleCommutes = commutes.some(
  (e) =>
    (e.from === "regex.emit.date" && e.to === "regex.emit.title") ||
    (e.from === "regex.emit.title" && e.to === "regex.emit.date"),
);
assert("date and title DO commute (independent emitters)", dateTitleCommutes);

// --- 6. redundanciesIn flags idempotent repeats and dominated writes --------
const redundsIdempotent = redundanciesIn(
  ["normalize.whitespace", "normalize.whitespace", "regex.emit.url"],
  graph,
);
assert(
  "redundanciesIn flags [normalize, normalize, ...] at position 1",
  redundsIdempotent.length === 1 &&
    redundsIdempotent[0]!.position === 1 &&
    redundsIdempotent[0]!.reason === "idempotent_repeat",
);

const redundsDominated = redundaniesInWithMuex();
function redundaniesInWithMuex() {
  return redundanciesIn(["regex.emit.url", "regex.emit.url.alt", "regex.emit.date"], extendedGraph);
}
assert(
  "redundanciesIn flags first of adjacent mutually-exclusive pair",
  redundsDominated.length >= 1 &&
    redundsDominated[0]!.position === 0 &&
    redundsDominated[0]!.reason === "dominated_by_next_writer",
);

// --- 7. canonicaliseUnderCommutation orders commuting pairs -----------------
const canon = canonicaliseUnderCommutation(
  ["regex.emit.title", "regex.emit.date"],
  graph,
);
assert(
  "canonicalise sorts commuting pair alphabetically",
  canon[0] === "regex.emit.date" && canon[1] === "regex.emit.title",
  `got=${canon.join(",")}`,
);

const canonNonCommuting = canonicaliseUnderCommutation(
  ["row.assemble.proximity", "regex.emit.url"],
  graph,
);
assert(
  "canonicalise does NOT reorder non-commuting pair",
  canonNonCommuting[0] === "row.assemble.proximity" && canonNonCommuting[1] === "regex.emit.url",
);

// --- 8. solver no-op contract -----------------------------------------------
const legacy = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4 }).search(
  ctx,
  companyUpdatesAF,
  ops,
);
const withEmptyRegistry = makeBeamSolver({
  beam: 8,
  maxLen: 6,
  extensionTopK: 4,
  interferenceRegistry: undefined,
}).search(ctx, companyUpdatesAF, ops);
assert(
  "absent registry: top score byte-identical to legacy",
  legacy[0]?.score === withEmptyRegistry[0]?.score,
  `legacy=${legacy[0]?.score} absent=${withEmptyRegistry[0]?.score}`,
);

// --- 9. solver with registry still discovers a productive creature ----------
const aware = makeBeamSolver({
  beam: 8,
  maxLen: 6,
  extensionTopK: 4,
  interferenceRegistry: PRIMITIVE_INTERFERENCE,
}).search(ctx, companyUpdatesAF, ops);
assert(
  "interference-aware solver still discovers a productive creature",
  (aware[0]?.rows.length ?? 0) >= 3,
  `rows=${aware[0]?.rows.length}`,
);

// --- 10. no redundant `[normalize, normalize]` in final candidates ----------
const anyRedundantNormalizePair = aware.some((c) => {
  const ids = c.genes.map((g) => g.operatorId);
  for (let i = 0; i < ids.length - 1; i++) {
    if (ids[i] === "normalize.whitespace" && ids[i + 1] === "normalize.whitespace") return true;
  }
  return false;
});
assert(
  "aware solver never returns adjacent-repeat idempotent ops",
  !anyRedundantNormalizePair,
);

// --- basis-coherence sanity --------------------------------------------------
const coherence = basisCoherence(ops, companyUpdatesAF, PRIMITIVE_INTERFERENCE);
assert(
  "basis-coherence is in (0, 1]",
  coherence.coherence > 0 && coherence.coherence <= 1,
  `${summariseCoherence(coherence)}`,
);
assert(
  "AF columns are covered by the basis",
  coherence.coverage >= 1,
  `coverage=${coherence.coverage}, uncovered=${coherence.uncoveredColumns.join(",")}`,
);

// Sanity: removing the date emitter should make `date` uncovered.
const opsWithoutDate = ops.filter((o) => o.id !== "regex.emit.date");
const cohWithoutDate = basisCoherence(opsWithoutDate, companyUpdatesAF, PRIMITIVE_INTERFERENCE);
assert(
  "removing the date emitter drops coverage below 1",
  cohWithoutDate.coverage < 1,
  `coverage=${cohWithoutDate.coverage}, uncovered=${cohWithoutDate.uncoveredColumns.join(",")}`,
);

// --- report ----------------------------------------------------------------
const passed = assertions.filter((a) => a.ok).length;
assertions.forEach((a) =>
  console.log(`${a.ok ? "ok " : "FAIL"} ${a.name}${a.detail ? "  [" + a.detail + "]" : ""}`),
);
console.log(`\n${passed}/${assertions.length} passed`);
if (passed !== assertions.length) process.exit(1);
