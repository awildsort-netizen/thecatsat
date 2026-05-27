// Test: propose-loop closes a missing_emitter boundary failure.
//
// Setup: take the standard primitives but REMOVE regex.emit.date. The
// AF should report `missing_emitter` for the `date` column (because
// `date` is required and no operator contributes it). The propose
// loop sees that hallucination and synthesises a replacement emitter.
//
// Expected:
//   * round 1 produces one new operator (`propose.regex.emit.date`)
//   * round 2 produces zero new operators (no more missing_emitter
//     because the synthesised emitter covers the channel)
//   * the final solver run finds a creature with all three columns
//     populated again
//   * propagation-coherence rises from <1 (uncovered=date) to 1.0

import { companyUpdatesAF } from "./af.js";
import {
  PRIMITIVES,
  makeEnforceSchema,
  regexEmitDate,
} from "./operators.js";
import { makeBeamSolver } from "./solver.js";
import { missingEmitterProposer, runProposeLoop } from "./propose.js";
import { propagationCoherence, summariseCoherence } from "./interference.js";
import {
  buildProbeFromPipeline,
  deriveRegistry,
} from "./interference_derivation.js";
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

const assertions: { name: string; ok: boolean; detail?: string }[] = [];
const assert = (name: string, ok: boolean, detail?: string): void => {
  assertions.push({ name, ok, detail });
};

// --- 1. Construct a deliberately broken basis (no date emitter) -----------
const brokenOps = [
  ...PRIMITIVES.filter((o) => o.id !== regexEmitDate.id),
  makeEnforceSchema(companyUpdatesAF),
];

// Sanity check: the broken basis lacks date coverage.
const cohBefore = propagationCoherence(brokenOps, companyUpdatesAF, new Map());
assert(
  "broken basis: coverage < 1 (date column uncovered)",
  cohBefore.coverage < 1,
  `coverage=${cohBefore.coverage} uncovered=[${cohBefore.uncoveredColumns.join(",")}]`,
);

// --- 2. Run the propose loop ----------------------------------------------
const solver = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4 });
const result = runProposeLoop(solver, ctx, companyUpdatesAF, brokenOps, [missingEmitterProposer]);

assert(
  "propose loop terminates",
  result.rounds <= 4,
  `rounds=${result.rounds}`,
);

assert(
  "propose loop adds exactly one operator (date emitter)",
  result.addedOperators.length === 1 && result.addedOperators[0] === "propose.regex.emit.date",
  `added=[${result.addedOperators.join(",")}]`,
);

// --- 3. The repaired basis covers the date column -------------------------
const cohAfter = propagationCoherence(result.finalOps, companyUpdatesAF, new Map());
assert(
  "after repair: coverage = 1.0",
  cohAfter.coverage >= 1,
  `coverage=${cohAfter.coverage} uncovered=[${cohAfter.uncoveredColumns.join(",")}]`,
);

// --- 4. The repaired basis produces a valid creature ----------------------
const finalCandidates = solver.search(ctx, companyUpdatesAF, result.finalOps);
const finalTop = finalCandidates[0];
const finalRows = finalTop?.rows ?? [];

assert(
  "after repair: top creature produces non-empty rows",
  finalRows.length > 0,
  `rows=${finalRows.length}`,
);

// Every row must have a date cell — that's the column whose boundary
// was leaking before the repair.
const allRowsHaveDate = finalRows.every((r) => "date" in r.fields);
assert(
  "after repair: every row has a date cell",
  allRowsHaveDate,
  `rows=[${finalRows.map((r) => Object.keys(r.fields).join("/")).join(" | ")}]`,
);

// The synthesised emitter participates in the top creature.
const synthesisedInTop = finalTop?.genes.some((g) => g.operatorId === "propose.regex.emit.date") ?? false;
assert(
  "after repair: synthesised emitter participates in top creature",
  synthesisedInTop,
  `genes=[${finalTop?.genes.map((g) => g.operatorId).join(",")}]`,
);

// --- 5. The repaired registry derives cleanly (propagation invariant) -----
//
// The derived InterferenceSpec for the synthesised emitter should match
// the spec the existing emitters have: replacing on `spans.dated`,
// accumulator on `trace.regions`, idempotent.
const probe = buildProbeFromPipeline(ctx, result.finalOps);
const registry = deriveRegistry(result.finalOps, probe);
const cohDerived = propagationCoherence(result.finalOps, companyUpdatesAF, registry);
assert(
  "after repair: derived registry preserves coherence=1.0",
  cohDerived.coherence >= 0.999,
  `${summariseCoherence(cohDerived)}`,
);

// --- 6. Idempotence: a second propose loop is a no-op ---------------------
const second = runProposeLoop(solver, ctx, companyUpdatesAF, result.finalOps, [missingEmitterProposer]);
assert(
  "second propose loop is a no-op (no new operators)",
  second.addedOperators.length === 0,
  `added=[${second.addedOperators.join(",")}]`,
);

// --- report ----------------------------------------------------------------
const passed = assertions.filter((a) => a.ok).length;
assertions.forEach((a) =>
  console.log(`${a.ok ? "ok " : "FAIL"} ${a.name}${a.detail ? "  [" + a.detail + "]" : ""}`),
);
console.log(`\n${passed}/${assertions.length} passed`);

console.log(`\nbefore repair: ${summariseCoherence(cohBefore)}`);
console.log(`after  repair: ${summariseCoherence(cohAfter)}`);
console.log(`rounds: ${result.rounds}  added: [${result.addedOperators.join(", ")}]`);
console.log(
  `top creature: score=${finalTop?.score.toFixed(3)} rows=${finalRows.length} genes=[${finalTop?.genes.map((g) => g.operatorId).join(", ")}]`,
);

if (passed !== assertions.length) process.exit(1);
