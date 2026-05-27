// Head-to-head: crystallisation vs beam on the standard AF.
//
// Both solvers are given the same operator basis and AF. We measure:
//   * evaluations actually performed (onEvaluate hook count)
//   * top score of the returned candidate list
//   * gene-string of the top creature
//
// The crystallisation solver should land a comparable creature with
// dramatically fewer evaluations — that is the headline of this branch.

import { companyUpdatesAF } from "./af.js";
import { PRIMITIVES, makeEnforceSchema } from "./operators.js";
import { makeBeamSolver } from "./solver.js";
import { makeCrystalSolver } from "./crystallise_solver.js";
import { buildProbeFromPipeline, deriveRegistry } from "./interference_derivation.js";
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

const measure = (label: string, run: (onEval: () => void) => { score: number; genes: string }): void => {
  let n = 0;
  const t0 = Date.now();
  const { score, genes } = run(() => n++);
  const ms = Date.now() - t0;
  console.log(`${label.padEnd(28)}  evals=${String(n).padStart(4)}  score=${score.toFixed(3).padStart(7)}  ${ms}ms`);
  console.log(`  genes: ${genes}`);
};

const probe = buildProbeFromPipeline(ctx, ops);
const registry = deriveRegistry(ops, probe);

console.log("Standard basis + companyUpdatesAF\n");

measure("beam (legacy, no graph)", (onEval) => {
  const s = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4, onEvaluate: onEval });
  const c = s.search(ctx, companyUpdatesAF, ops)[0];
  return { score: c.score, genes: c.genes.map((g) => g.operatorId).join(" >> ") };
});

measure("beam (interference graph)", (onEval) => {
  const s = makeBeamSolver({
    beam: 8,
    maxLen: 6,
    extensionTopK: 4,
    interferenceRegistry: registry,
    onEvaluate: onEval,
  });
  const c = s.search(ctx, companyUpdatesAF, ops)[0];
  return { score: c.score, genes: c.genes.map((g) => g.operatorId).join(" >> ") };
});

measure("crystal (no graph)", (onEval) => {
  const s = makeCrystalSolver({ worlds: 8, maxTicks: 12, onEvaluate: onEval });
  const c = s.search(ctx, companyUpdatesAF, ops)[0];
  return { score: c.score, genes: c.genes.map((g) => g.operatorId).join(" >> ") };
});

measure("crystal (graph)", (onEval) => {
  const s = makeCrystalSolver({
    worlds: 8,
    maxTicks: 12,
    interferenceRegistry: registry,
    onEvaluate: onEval,
  });
  const c = s.search(ctx, companyUpdatesAF, ops)[0];
  return { score: c.score, genes: c.genes.map((g) => g.operatorId).join(" >> ") };
});

measure("crystal (1 world)", (onEval) => {
  const s = makeCrystalSolver({ worlds: 1, maxTicks: 12, onEvaluate: onEval });
  const c = s.search(ctx, companyUpdatesAF, ops)[0];
  return { score: c.score, genes: c.genes.map((g) => g.operatorId).join(" >> ") };
});
