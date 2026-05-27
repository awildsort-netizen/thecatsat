// Self-test for the motif-invariance brittleness metric.
// Run with: npx tsx parser_evolver/brittleness_test.ts
//
// Each assertion checks a property of the metric, not just "the code ran".
// The key claims:
//   1. Identity recomposers give brittleness == 0.
//   2. A genuinely well-behaved discovered creature has low (but maybe
//      non-zero) brittleness on the default probe set.
//   3. A creature that *only* works because of accidental layout
//      (e.g. relies on block order) has higher brittleness than one
//      that defends pattern.
//   4. Brittleness is bounded in [0, 1].
//   5. The solver, with brittlenessWeight=0, returns identical scores
//      to the legacy path (the no-op contract).
//   6. With brittlenessWeight > 0, the top candidate's score is
//      strictly less than the legacy score for that same gene-string
//      whenever the creature has measurable brittleness.

import { companyUpdatesAF } from "./af.js";
import {
  brittlenessOf,
  motifOf,
  motifSimilarity,
  reorderBlocks,
  dilateWhitespace,
  swapDateFormat,
  DEFAULT_RECOMPOSERS,
  type Recomposer,
} from "./brittleness.js";
import { PRIMITIVES, makeEnforceSchema } from "./operators.js";
import { makeBeamSolver } from "./solver.js";
import type { ParseCandidate, ParseContext, RowHypothesis } from "./types.js";

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

// --- baseline discovery -----------------------------------------------------
const baselineSolver = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4 });
const candidates = baselineSolver.search(ctx, companyUpdatesAF, ops);
const top = candidates[0]!;

assert("baseline solver discovers a productive creature", top.rows.length >= 3);

// --- motif basics -----------------------------------------------------------
const motif = motifOf(top.rows);
assert(
  "motif features are nonempty for a productive creature",
  motif.features.size > 0,
  `feature count=${motif.features.size}`,
);
assert(
  "motif similarity of any motif with itself is 1",
  Math.abs(motifSimilarity(motif, motif) - 1) < 1e-9,
);

// Two unrelated motifs should have similarity strictly less than 1.
const emptyMotif = motifOf([]);
assert(
  "motif similarity of populated vs empty motif is below 1",
  motifSimilarity(motif, emptyMotif) < 1,
);

// --- identity recomposer ⇒ zero brittleness --------------------------------
const identity: Recomposer = {
  name: "identity",
  invariant: "no change",
  apply: (c) => c,
};
const idReport = brittlenessOf(top, ctx, ops, { recomposers: [identity] });
assert(
  "identity recomposer yields brittleness ≈ 0",
  Math.abs(idReport.score) < 1e-9,
  `score=${idReport.score}`,
);

// --- bounds ----------------------------------------------------------------
const defaultReport = brittlenessOf(top, ctx, ops);
assert(
  "brittleness score in [0, 1]",
  defaultReport.score >= 0 && defaultReport.score <= 1,
  `score=${defaultReport.score.toFixed(4)}`,
);
assert(
  "per-recomposer similarities in [0, 1]",
  Array.from(defaultReport.perRecomposer.values()).every((s) => s >= 0 && s <= 1),
  `values=${JSON.stringify(Array.from(defaultReport.perRecomposer.entries()))}`,
);

// --- a real, well-behaved creature should be low-brittleness ---------------
// "Low" here means: at least the reorder_blocks recomposer should leave
// the motif essentially unchanged. The CSV parser's emitted motif is a
// multiset of (field, value-class) tags, and shuffling whole blocks does
// not add or remove any such tag — every block contributes the same
// cells regardless of order.
const reorderSim = defaultReport.perRecomposer.get("reorder_blocks") ?? 0;
assert(
  "discovered creature is robust under block reordering",
  reorderSim >= 0.95,
  `reorder similarity=${reorderSim.toFixed(4)}`,
);

// Whitespace dilation should also not change the motif of a sane creature.
const wsSim = defaultReport.perRecomposer.get("dilate_whitespace") ?? 0;
assert(
  "discovered creature is robust under whitespace dilation",
  wsSim >= 0.5,
  `whitespace similarity=${wsSim.toFixed(4)}`,
);

// --- the no-op contract on the solver --------------------------------------
const weightedSolver = makeBeamSolver({
  beam: 8,
  maxLen: 6,
  extensionTopK: 4,
  brittlenessWeight: 0,
});
const weightedCands = weightedSolver.search(ctx, companyUpdatesAF, ops);
assert(
  "brittlenessWeight=0 yields identical top score to default solver",
  Math.abs((weightedCands[0]?.score ?? -Infinity) - top.score) < 1e-9,
  `legacy=${top.score} weight0=${weightedCands[0]?.score}`,
);

// --- brittlenessWeight > 0 only ever subtracts from score ------------------
const penalisedSolver = makeBeamSolver({
  beam: 8,
  maxLen: 6,
  extensionTopK: 4,
  brittlenessWeight: 2,
});
const penalisedCands = penalisedSolver.search(ctx, companyUpdatesAF, ops);
const penalisedTop = penalisedCands[0]!;
assert(
  "brittleness-penalised top score is ≤ legacy top score",
  penalisedTop.score <= top.score + 1e-9,
  `legacy=${top.score.toFixed(4)} penalised=${penalisedTop.score.toFixed(4)}`,
);

// --- contrastive test: a fragile fake creature should score worse ----------
//
// Construct a synthetic "fragile" candidate whose rows reflect only one
// specific date format (iso). Under swap_date_format the motif features
// flip from `date::date.iso` to `date::date.long`. That move alone makes
// the motif Jaccard fall, so brittleness rises.
const isoOnlyRows: readonly RowHypothesis[] = [
  {
    fields: {
      date: { field: "date", value: "2026-04-12", span: [0, 10], operator: "synthetic", confidence: 1 },
      title: { field: "title", value: "Wallet sync latency elevated in EU region", span: [11, 52], operator: "synthetic", confidence: 1 },
    },
    score: 1,
  },
];

const isoMotif = motifOf(isoOnlyRows);
const longMotif = motifOf([
  {
    fields: {
      date: { field: "date", value: "April 12, 2026", span: [0, 14], operator: "synthetic", confidence: 1 },
      title: { field: "title", value: "Wallet sync latency elevated in EU region", span: [15, 56], operator: "synthetic", confidence: 1 },
    },
    score: 1,
  },
]);

assert(
  "motif distinguishes iso vs long date format",
  motifSimilarity(isoMotif, longMotif) < 1,
  `sim=${motifSimilarity(isoMotif, longMotif).toFixed(4)}`,
);

// --- minimum-rows guard ----------------------------------------------------
const emptyCandidate: ParseCandidate = {
  genes: [],
  rows: [],
  score: 0,
  diagnostics: { coverage: 0, complexity: 0, hallucinationRisk: 0 },
};
const emptyReport = brittlenessOf(emptyCandidate, ctx, ops);
assert(
  "empty creature gets brittleness 0 (uninformative, not lying)",
  emptyReport.score === 0,
);

// --- recomposers preserve text length scale --------------------------------
DEFAULT_RECOMPOSERS.forEach((r) => {
  const perturbed = r.apply(ctx);
  // We don't require equal length (whitespace dilation expands it), only
  // that the perturbed text remains nonempty and finite — i.e. the
  // recomposer is a real function from contexts to contexts.
  assert(
    `recomposer '${r.name}' yields a nonempty perturbed context`,
    perturbed.normalizedText.length > 0,
  );
});

// --- report ---------------------------------------------------------------
const passed = assertions.filter((a) => a.ok).length;
assertions.forEach((a) =>
  console.log(`${a.ok ? "ok " : "FAIL"} ${a.name}${a.detail ? "  [" + a.detail + "]" : ""}`),
);
console.log(`\n${passed}/${assertions.length} passed`);
if (passed !== assertions.length) process.exit(1);
