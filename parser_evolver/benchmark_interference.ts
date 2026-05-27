// Benchmark: interference-aware solver vs. legacy solver.
//
// Run with: npx tsx parser_evolver/benchmark_interference.ts
//
// We run the beam search twice on the same context, AF, and operator
// set. The legacy run uses no interference registry — exactly the
// behaviour we shipped before this branch. The aware run hands the
// solver the primitive interference registry and lets it prune
// redundancies and dedupe under commutation before evaluation.
//
// Three numbers tell us whether the branch is doing real work:
//
//   * candidates_evaluated — number of gene-strings the solver actually
//     scored. The aware run should evaluate strictly fewer.
//   * top creature — the gene-string and total score of the winner.
//     The aware run must not lose to the legacy run; ideally the two
//     are identical (we are pruning provably-dominated candidates,
//     not productive ones).
//   * basis_coherence — a single scalar over the operator set as a
//     whole. The conflict graph reads channels and writes — a basis
//     where every required column has an emitter and no operator
//     orphans an output channel scores 1.0.

import { companyUpdatesAF, summarisePressure } from "./af.js";
import { PRIMITIVES, makeEnforceSchema } from "./operators.js";
import { makeBeamSolver } from "./solver.js";
import { PRIMITIVE_INTERFERENCE } from "./operator_interference.js";
import { basisCoherence, summariseCoherence } from "./interference.js";
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

This page does not list a particular incident.
Subscribe to updates if you want the firehose.
`;

const ctx: ParseContext = {
  url: "https://status.blockchain.com/",
  rawText: SAMPLE,
  normalizedText: SAMPLE,
  sourceType: "status-page",
};

const ops = [...PRIMITIVES, makeEnforceSchema(companyUpdatesAF)];

type Run = {
  label: string;
  evaluations: number;
  topGene: string;
  topScore: number;
  rows: number;
  pressure: string;
};

const runOnce = (label: string, useRegistry: boolean): Run => {
  let evaluations = 0;
  const solver = makeBeamSolver({
    beam: 8,
    maxLen: 6,
    extensionTopK: 4,
    interferenceRegistry: useRegistry ? PRIMITIVE_INTERFERENCE : undefined,
    onEvaluate: () => {
      evaluations += 1;
    },
  });
  const candidates = solver.search(ctx, companyUpdatesAF, ops);
  const top = candidates[0];
  return {
    label,
    evaluations,
    topGene: top?.genes.map((g) => g.operatorId).join(" >> ") ?? "(empty)",
    topScore: top?.score ?? 0,
    rows: top?.rows.length ?? 0,
    pressure: top ? JSON.stringify(summarisePressure(top.rows ?? [])) : "(no candidate)",
  };
};

const legacy = runOnce("legacy   ", false);
const aware = runOnce("aware    ", true);

// Basis-coherence report on the operator set as a whole.
const coherence = basisCoherence(ops, companyUpdatesAF, PRIMITIVE_INTERFERENCE);

// --- output ---------------------------------------------------------------
const fmt = (r: Run) =>
  [
    `[${r.label}] evaluations=${String(r.evaluations).padStart(4)}`,
    `score=${r.topScore.toFixed(3)}`,
    `gene-string=${r.topGene}`,
  ].join("  ");

console.log("=== interference benchmark ===");
console.log(fmt(legacy));
console.log(fmt(aware));

const reduction =
  legacy.evaluations === 0
    ? 0
    : (1 - aware.evaluations / legacy.evaluations) * 100;
console.log(
  `\nsearch-cost reduction: ${reduction.toFixed(1)}%  ` +
    `(legacy=${legacy.evaluations} \u2192 aware=${aware.evaluations})`,
);

const sameTop = legacy.topGene === aware.topGene;
const scoreDelta = aware.topScore - legacy.topScore;
console.log(
  `top-creature equivalence: ${sameTop ? "identical" : "diverged"}` +
    `   score \u0394=${scoreDelta >= 0 ? "+" : ""}${scoreDelta.toFixed(3)}`,
);

console.log(`\nbasis-coherence (over current operator set):`);
console.log(`  ${summariseCoherence(coherence)}`);
if (coherence.uncoveredColumns.length > 0) {
  console.log(`  uncovered AF columns: ${coherence.uncoveredColumns.join(", ")}`);
}
if (coherence.orphanChannels.length > 0) {
  console.log(`  orphan channels:       ${coherence.orphanChannels.join(", ")}`);
}

console.log(`\nlegacy   rows=${legacy.rows}  pressure: ${legacy.pressure}`);
console.log(`aware    rows=${aware.rows}   pressure: ${aware.pressure}`);

// Exit non-zero if either invariant is violated. The branch's contract
// is: the aware solver must not lose to legacy on top-score, and must
// not evaluate more candidates.
const ok = aware.topScore + 1e-9 >= legacy.topScore && aware.evaluations <= legacy.evaluations;
if (!ok) {
  console.error("\nINVARIANT FAILED: aware solver regressed against legacy.");
  process.exit(1);
}
