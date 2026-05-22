// Demo: walk the full browser-as-fallback-repair loop without running a
// real browser. Inputs are checked-in fixtures only.
//
//   static prepass on the bounded snapshot set
//     -> the IPO post escalates to needs-rendered-fetch (shell-only)
//   load a fixture browser-oracle trace (representing what an external
//     headless run would have produced for that URL)
//     -> distill it against the AF's evidence targets
//     -> propose a static fetch operator (next.route_payload.fetch)
//     -> bridge into a prepass hint that says 'use-static-operator'
//        and 'browser retired for this source'
//   load the *absence* fixture trace (same shape, but does not carry
//     the target evidence) and show that the distiller records
//     remembered absence and refuses to retire the browser.

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { distillTrace, loadTrace, liftProposalToOperator } from "./distiller.js";
import { hintFromDistillation } from "./prepass_bridge.js";
import { runPrepass } from "../prepass/prepass.js";
import type { EvidenceTarget } from "./types.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = (n: string) => resolve(HERE, "fixtures", n);

const banner = (s: string) => console.log(`\n=== ${s} ===`);

banner("Step 1: static prepass over bounded fixtures");
const prepass = runPrepass();
const ipoRow = prepass.rows.find((r) => r.source_id === "blog-ipo-announce");
if (!ipoRow) throw new Error("expected an IPO row from prepass");
console.log(`prepass row: ${ipoRow.source_id}`);
console.log(`  expected_action: ${ipoRow.expected_action}   (escalated because reachable=shell_only)`);
console.log(`  confidence:      ${ipoRow.confidence}`);
console.log("=> static parse cannot produce the IPO post body; browser fallback is armed.");

banner("Step 2: ingest fixture browser-oracle trace (NO live browser)");
const ipoTrace = loadTrace(FIX("blog-ipo-announce.trace.json"));
console.log(`pageUrl:           ${ipoTrace.pageUrl}`);
console.log(`requests captured: ${ipoTrace.requests.length}`);
for (const r of ipoTrace.requests) {
  console.log(`  [${r.requestOrder}] ${r.method} ${r.resourceType.padEnd(8)} ${r.status} ${r.url.slice(0, 90)}${r.url.length > 90 ? "…" : ""}`);
}

banner("Step 3: distill the trace against the AF's evidence targets");
const targets: readonly EvidenceTarget[] = [
  {
    field: "title",
    markers: [
      "Blockchain.com Announces Confidential Submission of Draft Registration Statement",
    ],
    weight: 2,
  },
  {
    field: "body",
    markers: ["Form F-1", "draft registration statement"],
    weight: 2,
  },
  { field: "date", markers: ["2026-05-20"], weight: 1 },
];
const dist = distillTrace(ipoTrace, targets, { source_id: "blog-ipo-announce" });
console.log(`evidenceProduced:        ${dist.evidenceProduced.join(", ")}`);
console.log(`minimalRequests (${dist.minimalRequests.length}):`);
for (const r of dist.minimalRequests) {
  console.log(`  - ${r.url.slice(0, 100)}…`);
}
console.log(`canRetireBrowser:        ${dist.canRetireBrowser}`);
console.log(`distillation confidence: ${dist.confidence}`);
console.log(`urlConstructionFragments:`);
for (const f of dist.urlConstructionFragments) {
  console.log(`  ${f.kind.padEnd(14)} ${f.value}${f.note ? `   // ${f.note}` : ""}`);
}

banner("Step 4: proposed static operator (authored material only)");
for (const p of dist.proposedOperators) {
  console.log(`id:             ${p.id}`);
  console.log(`evidenceFields: ${p.evidenceFields.join(", ")}    // the lifted operator's io is reflected from the run body, not from this`);
  console.log(`tokens:         ${p.tokens.join(", ")}`);
  console.log(`pattern:        ${p.urlPattern ?? "(literal)"}`);
  console.log(`requestTemplate.url: ${p.requestTemplate.url.slice(0, 100)}…`);
  console.log(`confidence:     ${p.confidence}`);
}

banner("Step 5: lift proposal to a parser_evolver ParseOperator (io derived)");
const lifted = liftProposalToOperator(dist.proposedOperators[0]!);
console.log(`lifted operator id: ${lifted.id}`);
console.log(`lifted io:          ${JSON.stringify(lifted.io)}    // reflected from the run-body's typed channel spec, not authored fields`);
const liftedOut = lifted.run(
  { url: ipoTrace.pageUrl, rawText: "", normalizedText: "" },
  { from: "demo" },
) as Record<string, unknown>;
const liftedPayload = liftedOut["browser_oracle.proposal"] as Record<string, unknown>;
console.log(`lifted.run output:  ${JSON.stringify({ proposalId: liftedPayload.proposalId, evidenceFields: liftedPayload.evidenceFields })}`);

banner("Step 6: bridge into a prepass hint");
const hint = hintFromDistillation(ipoRow, dist);
console.log(`hint.source_id:           ${hint.source_id}`);
console.log(`hint.priorExpectedAction: ${hint.priorExpectedAction}`);
console.log(`hint.action:              ${hint.action}`);
console.log(`hint.canRetireBrowser:    ${hint.canRetireBrowser}`);
console.log(`hint.note:                ${hint.note}`);

banner("Step 7: absence case — a trace that does NOT carry the evidence");
const absenceTrace = loadTrace(FIX("blog-index-absence.trace.json"));
const absenceDist = distillTrace(absenceTrace, targets, { source_id: "blog-ipo-announce" });
console.log(`evidenceProduced:     ${absenceDist.evidenceProduced.join(", ") || "(none)"}`);
console.log(`proposedOperators:    ${absenceDist.proposedOperators.length}`);
console.log(`canRetireBrowser:     ${absenceDist.canRetireBrowser}`);
console.log(`rememberedAbsences:`);
for (const a of absenceDist.rememberedAbsences) {
  console.log(`  - field=${a.field}  reason=${a.reason}`);
}
const absenceHint = hintFromDistillation(ipoRow, absenceDist);
console.log(`absence hint.action: ${absenceHint.action}`);
console.log(`absence hint.note:   ${absenceHint.note}`);

console.log("\nDone.");
