// Demo: evolve parser-creatures over a Blockchain.com-style updates page.
//
// Run with: npx tsx parser_evolver/demo.ts
//
// The sample text mimics the shape of a status/blog index — date, title,
// trailing url — interleaved with chatter the AF should reject. URLs end
// in date-shaped slugs so the date-vs-URL exclusion gets exercised.

import { companyUpdatesAF, summarisePressure } from "./af.js";
import { PRIMITIVES, makeEnforceSchema } from "./operators.js";
import { makeBeamSolver } from "./solver.js";
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
const solver = makeBeamSolver({ beam: 8, maxLen: 6, extensionTopK: 4 });
const candidates = solver.search(ctx, companyUpdatesAF, ops);

const top = candidates[0];
console.log("=== top parser-creature ===");
console.log("score:", top?.score.toFixed(3), "diag:", top?.diagnostics);
console.log("gene-string:", top?.genes.map((g) => g.operatorId).join(" >> "));
console.log("\nemitted rows (each cell shows the source span it points to):");
top?.rows.forEach((row, i) => {
  console.log(`row ${i}:`);
  Object.entries(row.fields).forEach(([k, f]) => {
    console.log(
      `  ${k.padEnd(6)} span=${JSON.stringify(f.span)} value=${JSON.stringify(f.value)} trace=${f.traceRegionId ?? "-"}`,
    );
  });
});

console.log("\n=== typed hallucination pressure ===");
console.log(summarisePressure(top?.rows ?? []));

console.log("\n=== runner-up creatures (top 5) ===");
candidates.slice(0, 5).forEach((c, i) => {
  console.log(`#${i} score=${c.score.toFixed(3)} genes=[${c.genes.map((g) => g.operatorId).join(", ")}]`);
});
