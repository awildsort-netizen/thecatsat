// Benchmark: derived registry vs. hand-authored vs. legacy.
//
// The claim of this branch is that the InterferenceSpec can be derived
// from the operator's lambda by running it on a recorded probe. If the
// claim is right, we expect:
//
//   * derived ≅ authored on all three numbers (evals, score, gene)
//   * derived ≅ authored on propagation-coherence
//   * both registries beat legacy on evals
//
// If derived diverges from authored, that's the signal: either the
// authored spec was wrong (the lambda disagrees with the claim) or the
// derivation method is too aggressive (claiming idempotence where
// behaviour says otherwise).

import { companyUpdatesAF, summarisePressure } from "./af.js";
import { PRIMITIVES, makeEnforceSchema } from "./operators.js";
import { makeBeamSolver } from "./solver.js";
import { PRIMITIVE_INTERFERENCE } from "./operator_interference.js";
import { propagationCoherence, summariseCoherence, type InterferenceRegistry } from "./interference.js";
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
const probe = buildProbeFromPipeline(ctx, ops);
const derivedRegistry = deriveRegistry(ops, probe);

type Run = {
  label: string;
  evaluations: number;
  topGene: string;
  topScore: number;
  rows: number;
  pressure: string;
};

const runOnce = (label: string, registry: InterferenceRegistry | undefined): Run => {
  let evaluations = 0;
  const solver = makeBeamSolver({
    beam: 8,
    maxLen: 6,
    extensionTopK: 4,
    interferenceRegistry: registry,
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

const legacy = runOnce("legacy   ", undefined);
const authored = runOnce("authored ", PRIMITIVE_INTERFERENCE);
const derived = runOnce("derived  ", derivedRegistry);

const fmt = (r: Run) =>
  [
    `[${r.label}] evaluations=${String(r.evaluations).padStart(4)}`,
    `score=${r.topScore.toFixed(3)}`,
    `gene-string=${r.topGene}`,
  ].join("  ");

console.log("=== derived-interference benchmark ===");
console.log(fmt(legacy));
console.log(fmt(authored));
console.log(fmt(derived));

const reductionVsLegacy = (r: Run): string =>
  legacy.evaluations === 0
    ? "n/a"
    : `${((1 - r.evaluations / legacy.evaluations) * 100).toFixed(1)}%`;

console.log(`\nsearch-cost reduction vs. legacy:`);
console.log(`  authored: ${reductionVsLegacy(authored)} (${legacy.evaluations} → ${authored.evaluations})`);
console.log(`  derived : ${reductionVsLegacy(derived)} (${legacy.evaluations} → ${derived.evaluations})`);

const sameAsAuthored =
  derived.topGene === authored.topGene &&
  Math.abs(derived.topScore - authored.topScore) < 1e-9 &&
  derived.evaluations === authored.evaluations;
console.log(
  `\nderived ≅ authored: ${sameAsAuthored ? "yes — identical evals, score, gene" : "NO"}`,
);

const cohA = propagationCoherence(ops, companyUpdatesAF, PRIMITIVE_INTERFERENCE);
const cohD = propagationCoherence(ops, companyUpdatesAF, derivedRegistry);
console.log(`\npropagation-coherence:`);
console.log(`  authored: ${summariseCoherence(cohA)}`);
console.log(`  derived : ${summariseCoherence(cohD)}`);

// Contract: derived must not regress against authored.
const ok =
  derived.topScore + 1e-9 >= authored.topScore &&
  derived.evaluations <= authored.evaluations &&
  Math.abs(cohD.coherence - cohA.coherence) < 1e-9;
if (!ok) {
  console.error("\nINVARIANT FAILED: derived registry diverged from authored.");
  process.exit(1);
}
