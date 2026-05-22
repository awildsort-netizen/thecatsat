// Strengthened self-test for the parser-evolver.
// Run with: npx tsx parser_evolver/test.ts
//
// Each assertion checks something the climate actually claims, not just
// "the pipeline ran".

import { companyUpdatesAF, summarisePressure } from "./af.js";
import { PRIMITIVES, makeEnforceSchema, regexEmitDate } from "./operators.js";
import type { ReflectedOperator } from "./operator_reflection.js";
import { makeBeamSolver } from "./solver.js";
import type { FieldHypothesis, ParseContext, RowHypothesis } from "./types.js";

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

const assertions: { name: string; ok: boolean; detail?: string }[] = [];
const assert = (name: string, ok: boolean, detail?: string) =>
  assertions.push({ name, ok, detail });

const ops = [...PRIMITIVES, makeEnforceSchema(companyUpdatesAF)];
const solver = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4 });
const candidates = solver.search(ctx, companyUpdatesAF, ops);
const top = candidates[0];

assert("solver returns at least one candidate", candidates.length > 0);
assert("top candidate emits at least one row", (top?.rows.length ?? 0) > 0);

// --- Hard cell-level guarantees ---------------------------------------------
const cells: readonly FieldHypothesis[] = top?.rows.flatMap((r) => Object.values(r.fields)) ?? [];

const sliceMatches = cells.every(
  (c) => SAMPLE.slice(c.span[0], c.span[1]) === c.value || SAMPLE.slice(c.span[0], c.span[1]).trim() === c.value,
);
assert("every cell's source slice equals its value", sliceMatches, `cells=${cells.length}`);

const validators: Record<string, (v: string) => boolean> = {
  date: (v) => /^\d{4}-\d{2}-\d{2}$/.test(v) || /^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$/.test(v),
  title: (v) => v.length >= 8 && v.length <= 160 && !/^https?:\/\//.test(v) && !validators.date!(v),
  url: (v) => /^https?:\/\/\S+$/.test(v),
};

const everyCellPassesItsValidator = cells.every((c) => validators[c.field]?.(c.value) ?? true);
assert("every kept cell passes its column validator", everyCellPassesItsValidator);

// --- Misassignment guard ----------------------------------------------------
const noDateInTitle = cells
  .filter((c) => c.field === "title")
  .every((c) => !validators.date!(c.value));
assert("no title cell is actually a date string", noDateInTitle);

const noTitleInDate = cells
  .filter((c) => c.field === "date")
  .every((c) => validators.date!(c.value));
assert("no date cell is actually a non-date string", noTitleInDate);

// --- URL non-bleeding (each URL belongs to the row whose date precedes it) --
const rows = top?.rows ?? [];
const urlsInOrder = rows.map((r) => r.fields.url?.span);
const sortedAscending = urlsInOrder.every((s, i) => i === 0 || s === undefined || urlsInOrder[i - 1] === undefined || s[0] >= urlsInOrder[i - 1]![1]);
assert("URLs flow forward across rows (no cross-bleed)", sortedAscending);

// --- Positive hallucination control: fabricate a row with a bad span -------
//
// Build a creature-shaped output by hand and let the AF score it. It should
// score strictly worse than the discovered creature, demonstrating that the
// climate actually punishes lying.
const fabricated: RowHypothesis = {
  fields: {
    date: { field: "date", value: "2099-99-99", span: [0, 0], operator: "test.fake", confidence: 0.5 },
    title: { field: "title", value: "Made up announcement", span: [0, 0], operator: "test.fake", confidence: 0.5 },
    url: { field: "url", value: "https://example.com/nope", span: [0, 0], operator: "test.fake", confidence: 0.5 },
  },
  score: 0.5,
};
const fabricatedScore = companyUpdatesAF.scoreRun([fabricated]);
const realScore = top?.score ?? -Infinity;
assert(
  "fabricated row with empty spans scores strictly worse than discovered creature",
  fabricatedScore < realScore,
  `fabricated=${fabricatedScore.toFixed(3)} real=${realScore.toFixed(3)}`,
);

// --- Typed hallucinations exist for the fabricated row ---------------------
const fabPressure = summarisePressure([fabricated]);
assert(
  "fabricated row produces unsupported_cell hallucinations",
  fabPressure.unsupported_cell >= 3,
  `kinds=${JSON.stringify(fabPressure)}`,
);

// --- Role confusion detector fires for a date emitted as a title -----------
const roleConfused: RowHypothesis = {
  fields: {
    date: { field: "date", value: "2026-04-12", span: [1, 11], operator: "real", confidence: 0.9 },
    title: { field: "title", value: "2026-04-12", span: [1, 11], operator: "test.confused", confidence: 0.5 },
  },
  score: 0.5,
};
const rcPressure = summarisePressure([roleConfused]);
assert(
  "field_role_confusion fires when a date string is emitted as a title",
  rcPressure.field_role_confusion >= 1,
  `kinds=${JSON.stringify(rcPressure)}`,
);

// --- Lying-by-omission creature: no emitters at all -----------------------
const lying = solver.search(ctx, companyUpdatesAF, [PRIMITIVES[0]!, PRIMITIVES[4]!]);
const lyingScore = lying[0]?.score ?? -Infinity;
assert(
  "creature with no emitters does not outscore the discovered creature",
  lyingScore <= realScore,
);

// --- Embedding ranking is actually doing work ------------------------------
// If we set extensionTopK = 1, the solver is forced to follow embedding
// ranking step-by-step. It should still find a creature with non-zero score,
// proving that the embedding can steer the search to a useful pathway.
const guidedSolver = makeBeamSolver({ beam: 4, maxLen: 6, extensionTopK: 1 });
const guidedTop = guidedSolver.search(ctx, companyUpdatesAF, ops)[0];
assert(
  "embedding-only ranking (topK=1) still discovers a productive creature",
  (guidedTop?.score ?? -Infinity) > 0,
  `guided=${guidedTop?.score?.toFixed(3)}`,
);

// --- TraceRegions are first-class -----------------------------------------
const traces = top?.traces ?? [];
assert("top candidate accumulates at least one TraceRegion", traces.length > 0);
assert(
  "every kept cell that has a traceRegionId points to a real trace region",
  cells.every((c) => c.traceRegionId === undefined || traces.some((t) => t.id === c.traceRegionId)),
);

// --- defineOperator: signature reflected from typed IO, no parallel ontology
// `regexEmitDate` is built via `defineOperator`. The `inputs` spec is a
// single declaration whose property-modifier-style helpers (`required<T>()`
// / `optional<T>()`) drive both the run-body's input type (`?`-property
// for optional channels) and the legacy `signature.needs` / `signature.provides`
// projection. The assertions below witness that:
//
//   (a) only required-input keys reach `signature.needs` — so the
//       optional read-through channel `trace.regions` is NOT in needs;
//   (b) the reflected view exposes the optional partition explicitly;
//   (c) the operator participates in the solver's top creature unchanged.
const reflectedDate: ReflectedOperator = regexEmitDate;
assert(
  "regexEmitDate.signature.needs reflects only required inputs (trace.regions is optional and excluded)",
  ["text.normalized", "spans.url"].every((k) => regexEmitDate.signature.needs.includes(k)) &&
    !regexEmitDate.signature.needs.includes("trace.regions") &&
    regexEmitDate.signature.needs.length === 2,
  `needs=${regexEmitDate.signature.needs.join(",")}`,
);
assert(
  "regexEmitDate.signature.provides reflects all declared output channels",
  ["spans.dated", "trace.regions"].every((k) => regexEmitDate.signature.provides.includes(k)) &&
    regexEmitDate.signature.provides.length === 2,
  `provides=${regexEmitDate.signature.provides.join(",")}`,
);
assert(
  "reflected view surfaces the optional-input partition (trace.regions is optional)",
  reflectedDate.reflected?.optionalInputs.includes("trace.regions") === true &&
    reflectedDate.reflected?.requiredInputs.includes("trace.regions") === false,
  `optional=${reflectedDate.reflected?.optionalInputs.join(",")}; required=${reflectedDate.reflected?.requiredInputs.join(",")}`,
);
const dateChainTop = top?.genes.some((g) => g.operatorId === "regex.emit.date");
assert(
  "the signature-derived operator participates in the top creature",
  dateChainTop === true,
  `genes=${top?.genes.map((g) => g.operatorId).join(">>")}`,
);

const failed = assertions.filter((a) => !a.ok);
assertions.forEach((a) =>
  console.log(`${a.ok ? "ok" : "FAIL"}  ${a.name}${a.detail ? `  [${a.detail}]` : ""}`),
);
console.log(`\n${assertions.length - failed.length}/${assertions.length} passed`);
if (failed.length > 0) (globalThis as { process?: { exit: (n: number) => void } }).process?.exit(1);
