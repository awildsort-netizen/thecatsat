// Tests for the crystallisation solver.
//
// What we are checking
// --------------------
// 1. The solver runs without errors on the standard basis + AF.
// 2. It produces a non-empty top creature with all three columns
//    (date, title, url) populated — the same target the beam achieves.
// 3. The top creature's score is within striking distance of the beam's
//    (we don't require strict equality — different ontology, slightly
//    different firing order; the *quality of the parse* is what matters).
// 4. Crystallisation evaluates strictly fewer times than the beam on the
//    same AF — the cheap-search promise.
// 5. With an interference registry, the solver respects the conflict
//    graph: it doesn't waste fires on shadowed peers.
// 6. The crystallisation solver plugs into the propose-loop without
//    modification (Solver-interface parity).

import { companyUpdatesAF } from "./af.js";
import { PRIMITIVES, makeEnforceSchema, regexEmitDate } from "./operators.js";
import { makeBeamSolver } from "./solver.js";
import { makeCrystalSolver } from "./crystallise_solver.js";
import { missingEmitterProposer, runProposeLoop } from "./propose.js";
import { deriveRegistry, buildProbeFromPipeline } from "./interference_derivation.js";
import type { ParseContext } from "./types.js";

const SAMPLE = `
Blockchain.com Status

2026-04-12
Wallet sync latency elevated in EU region
We are investigating delays affecting balance refresh.
https://status.blockchain.com/incidents/wallet-eu-2026-04-12

2026-04-09
Scheduled maintenance for institutional API
Brief downtime expected between 02:00 and 03:00 UTC.
https://status.blockchain.com/maintenance/api-2026-04-09

April 1, 2026
Lightning withdrawals now generally available
After a long beta we are flipping the switch for all retail accounts.
https://www.blockchain.com/blog/posts/lightning-ga
`;

const ctx: ParseContext = {
  url: "https://status.blockchain.com/",
  rawText: SAMPLE,
  normalizedText: SAMPLE,
  sourceType: "status-page",
};

const ops = [...PRIMITIVES, makeEnforceSchema(companyUpdatesAF)];

const assertions: { name: string; ok: boolean; detail?: string }[] = [];
const assert = (name: string, ok: boolean, detail?: string): void => {
  assertions.push({ name, ok, detail });
};

// --- 1. Crystallisation runs and returns a non-empty top creature ---------

const crystal = makeCrystalSolver({ worlds: 8, maxTicks: 16 });
const cands = crystal.search(ctx, companyUpdatesAF, ops);

assert(
  "crystal: returns at least one candidate",
  cands.length > 0,
  `n=${cands.length}`,
);

const top = cands[0];
assert(
  "crystal: top creature has non-empty rows",
  top !== undefined && top.rows.length > 0,
  `rows=${top?.rows.length}`,
);

// --- 2. All three columns present in every row ----------------------------

const fieldsPerRow = top?.rows.map((r) => Object.keys(r.fields).sort().join(",")) ?? [];
const allRowsComplete = fieldsPerRow.every((s) => s === "date,title,url");
assert(
  "crystal: every row has date/title/url",
  allRowsComplete,
  `rows=[${fieldsPerRow.join(" | ")}]`,
);

// --- 3. Score within reach of the beam ------------------------------------

const beam = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4 });
const beamCands = beam.search(ctx, companyUpdatesAF, ops);
const beamTop = beamCands[0];

assert(
  "crystal: top score >= 0.9 * beam top score",
  top !== undefined && beamTop !== undefined && top.score >= 0.9 * beamTop.score,
  `crystal=${top?.score.toFixed(3)} beam=${beamTop?.score.toFixed(3)}`,
);

// --- 4. Crystallisation evaluates strictly fewer times --------------------

let beamEvals = 0;
const beamMeasured = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4, onEvaluate: () => beamEvals++ });
beamMeasured.search(ctx, companyUpdatesAF, ops);

let crystalEvals = 0;
const crystalMeasured = makeCrystalSolver({ worlds: 8, maxTicks: 16, onEvaluate: () => crystalEvals++ });
crystalMeasured.search(ctx, companyUpdatesAF, ops);

assert(
  "crystal: strictly fewer evaluations than beam",
  crystalEvals < beamEvals,
  `crystal=${crystalEvals} beam=${beamEvals}`,
);

// --- 5. With derived interference registry, conflict graph is honoured ----

const probe = buildProbeFromPipeline(ctx, ops);
const registry = deriveRegistry(ops, probe);

const crystalAware = makeCrystalSolver({
  worlds: 8,
  maxTicks: 16,
  interferenceRegistry: registry,
});
const awareCands = crystalAware.search(ctx, companyUpdatesAF, ops);
const awareTop = awareCands[0];

assert(
  "crystal+graph: top creature still has non-empty rows",
  awareTop !== undefined && awareTop.rows.length > 0,
  `rows=${awareTop?.rows.length}`,
);

// With the graph, idempotent ops fire at most once. Check that no
// operator id repeats in the firing trace.
const ids = awareTop?.genes.map((g) => g.operatorId) ?? [];
const repeats = ids.filter((id, i) => ids.indexOf(id) !== i);
assert(
  "crystal+graph: no operator fires twice in the firing order",
  repeats.length === 0,
  `repeats=[${repeats.join(",")}]`,
);

// --- 6. Solver-interface parity: propose-loop works with crystal ----------

const brokenOps = [
  ...PRIMITIVES.filter((o) => o.id !== regexEmitDate.id),
  makeEnforceSchema(companyUpdatesAF),
];

const repaired = runProposeLoop(
  crystal,
  ctx,
  companyUpdatesAF,
  brokenOps,
  [missingEmitterProposer],
);

assert(
  "propose-loop with crystal: adds the date emitter",
  repaired.addedOperators.length === 1 && repaired.addedOperators[0] === "propose.regex.emit.date",
  `added=[${repaired.addedOperators.join(",")}]`,
);

const repairedCands = crystal.search(ctx, companyUpdatesAF, repaired.finalOps);
const repairedTop = repairedCands[0];
// Same bar the beam propose_test uses: every row has a date cell
// (the column whose boundary was leaking before the repair). The
// proximity assembler may or may not attach url to every row
// depending on layout; that's an assembler property, not a solver one.
assert(
  "propose-loop with crystal: post-repair top creature has dated rows",
  repairedTop !== undefined &&
    repairedTop.rows.length > 0 &&
    repairedTop.rows.every((r) => "date" in r.fields),
  `rows=${repairedTop?.rows.length} fields=[${repairedTop?.rows.map((r) => Object.keys(r.fields).join("/")).join(" | ")}]`,
);

// The synthesised emitter must participate in the firing order — the
// repair only counts as land if the new op actually fires in the
// crystallisation world.
const proposedFired = repairedTop?.genes.some((g) => g.operatorId === "propose.regex.emit.date") ?? false;
assert(
  "propose-loop with crystal: synthesised emitter fires in the top world",
  proposedFired,
  `genes=[${repairedTop?.genes.map((g) => g.operatorId).join(",")}]`,
);

// --- report ---------------------------------------------------------------

const passed = assertions.filter((a) => a.ok).length;
assertions.forEach((a) =>
  console.log(`${a.ok ? "ok " : "FAIL"} ${a.name}${a.detail ? "  [" + a.detail + "]" : ""}`),
);
console.log(`\n${passed}/${assertions.length} passed`);

console.log(
  `\ncrystal top: score=${top?.score.toFixed(3)} rows=${top?.rows.length} genes=[${top?.genes.map((g) => g.operatorId).join(", ")}]`,
);
console.log(
  `beam    top: score=${beamTop?.score.toFixed(3)} rows=${beamTop?.rows.length} genes=[${beamTop?.genes.map((g) => g.operatorId).join(", ")}]`,
);
console.log(`evals: crystal=${crystalEvals}  beam=${beamEvals}  (reduction=${((1 - crystalEvals / beamEvals) * 100).toFixed(1)}%)`);

if (passed !== assertions.length) process.exit(1);
