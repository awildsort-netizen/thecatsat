// Tests for the browser-oracle / trace-distillation prototype.
//
// These assertions encode the developmental-trace contract:
//
//   1. The static prepass row for the IPO post is shell-only and
//      escalated to needs-rendered-fetch (sanity link to prepass).
//   2. The fixture browser trace is consumed without launching a browser
//      and identifies the Next.js route-data fetch as the minimal
//      data-bearing request.
//   3. The distiller proposes a static operator with the expected id
//      (next.route_payload.fetch) that provides the required fields.
//   4. When evidence is absent, the distiller does NOT propose a
//      confident operator; it records remembered absence and the bridge
//      keeps `needs-rendered-fetch`.
//   5. Browser retirement is only true when the distillation covered
//      every requested field.

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { distillTrace, liftProposalToOperator, loadTrace } from "./distiller.js";
import { hintFromDistillation } from "./prepass_bridge.js";
import { runPrepass } from "../prepass/prepass.js";
import type { EvidenceTarget } from "./types.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = (n: string) => resolve(HERE, "fixtures", n);

const assertions: { name: string; ok: boolean; detail?: string }[] = [];
const assert = (name: string, ok: boolean, detail?: string) =>
  assertions.push({ name, ok, detail });

// ---------------------------------------------------------------------------
// Step 1: prepass shows the IPO post as shell-only -> needs-rendered-fetch.
// This is the "static failure" trigger that should kick off the
// developmental-trace fallback.
// ---------------------------------------------------------------------------

const prepass = runPrepass();
const ipoRow = prepass.rows.find((r) => r.source_id === "blog-ipo-announce");

assert("prepass produced an IPO row", ipoRow !== undefined);
assert(
  "prepass escalates IPO row to needs-rendered-fetch (static failure)",
  ipoRow?.expected_action === "needs-rendered-fetch",
  `got=${ipoRow?.expected_action}`,
);

// ---------------------------------------------------------------------------
// Step 2: a fixture browser trace is consumed and the route-payload
// fetch is identified as the minimal data-bearing request.
// ---------------------------------------------------------------------------

const ipoTrace = loadTrace(FIX("blog-ipo-announce.trace.json"));

const ipoTargets: readonly EvidenceTarget[] = [
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

const ipoDist = distillTrace(ipoTrace, ipoTargets, { source_id: "blog-ipo-announce" });

assert("distiller picked at least one minimal request", ipoDist.minimalRequests.length >= 1);
assert(
  "minimal request set includes the Next.js route-data fetch",
  ipoDist.minimalRequests.some((r) => r.url.includes("/_next/data/") && r.url.includes(".json")),
  `picked=${ipoDist.minimalRequests.map((r) => r.url).join(", ")}`,
);
assert(
  "minimal request set is small (<= 2 carriers)",
  ipoDist.minimalRequests.length <= 2,
  `picked=${ipoDist.minimalRequests.length}`,
);
assert(
  "evidenceProduced covers all required fields",
  ipoDist.evidenceProduced.length === ipoTargets.length,
  `produced=${ipoDist.evidenceProduced.join(",")}`,
);
assert(
  "the analytics POST is not in minimal requests",
  !ipoDist.minimalRequests.some((r) => r.url.includes("google-analytics")),
);
assert(
  "the og-image is not in minimal requests",
  !ipoDist.minimalRequests.some((r) => r.url.includes("og-image")),
);

// ---------------------------------------------------------------------------
// Step 3: the proposed static operator has the expected id and shape.
// ---------------------------------------------------------------------------

assert(
  "distiller proposed at least one static operator",
  ipoDist.proposedOperators.length >= 1,
);
const proposal = ipoDist.proposedOperators[0];
assert(
  "proposed operator id is next.route_payload.fetch",
  proposal?.id === "next.route_payload.fetch",
  `id=${proposal?.id}`,
);
assert(
  "proposal urlPattern is the Next.js route-data shape",
  proposal?.urlPattern === "/_next/data/{buildId}/{route}.json",
  `urlPattern=${proposal?.urlPattern}`,
);
assert(
  "proposal evidenceFields cover title and body and date (provides is derived from this)",
  ["title", "body", "date"].every((f) => (proposal?.evidenceFields ?? []).includes(f)),
  `evidenceFields=${proposal?.evidenceFields.join(",")}`,
);
assert(
  "proposal has positive confidence",
  (proposal?.confidence ?? 0) > 0.4,
  `conf=${proposal?.confidence}`,
);
assert(
  "proposal material hints include next-data + json",
  ["next-data", "json"].every((t) => (proposal?.materialHints ?? []).includes(t)),
  `tokens=${proposal?.materialHints.join(",")}`,
);

// Lift to a real ParseOperator — `io` is derived from the lifted run
// body's typed channel spec via defineOperator, not copied from
// explicit fields on the proposal.
const lifted = liftProposalToOperator(proposal!);
assert(
  "lifted operator has a derived io with requiredInputs/outputs/tokens",
  Array.isArray(lifted.io.requiredInputs) &&
    Array.isArray(lifted.io.outputs) &&
    Array.isArray(lifted.io.tokens),
);
assert(
  "lifted operator's requiredInputs is empty (source operator; reads nothing upstream)",
  lifted.io.requiredInputs.length === 0,
  `requiredInputs=${lifted.io.requiredInputs.join(",")}`,
);
assert(
  "lifted operator outputs the proposal channel (derived from IO)",
  lifted.io.outputs.includes("browser_oracle.proposal"),
  `outputs=${lifted.io.outputs.join(",")}`,
);
assert(
  "lifted operator's run returns a structured proposal in the typed output channel",
  (() => {
    const out = lifted.run(
      { url: ipoTrace.pageUrl, rawText: "", normalizedText: "" },
      { from: "test" },
    ) as Record<string, unknown>;
    const payload = out["browser_oracle.proposal"] as Record<string, unknown> | undefined;
    return payload?.proposalId === proposal!.id && typeof payload?.note === "string";
  })(),
);

// ---------------------------------------------------------------------------
// Step 4: browser retirement is true for the full-evidence IPO case.
// ---------------------------------------------------------------------------

assert(
  "canRetireBrowser is true when every required field is covered",
  ipoDist.canRetireBrowser === true,
  `retire=${ipoDist.canRetireBrowser}`,
);

// ---------------------------------------------------------------------------
// Step 5: absence case — the blog-index trace does NOT carry the IPO
// body. Distiller should not propose a confident operator and should
// emit a remembered absence; bridge should keep needs-rendered-fetch.
// ---------------------------------------------------------------------------

const absenceTrace = loadTrace(FIX("blog-index-absence.trace.json"));
const absenceDist = distillTrace(absenceTrace, ipoTargets, { source_id: "blog-ipo-announce" });

assert(
  "absence trace produces zero proposed operators (no markers matched)",
  absenceDist.proposedOperators.length === 0,
  `proposals=${absenceDist.proposedOperators.length}`,
);
assert(
  "absence trace records remembered absences for every requested field",
  absenceDist.rememberedAbsences.length === ipoTargets.length,
  `absences=${absenceDist.rememberedAbsences.length}`,
);
assert(
  "absence trace canRetireBrowser is false",
  absenceDist.canRetireBrowser === false,
);
assert(
  "absence trace confidence is 0",
  absenceDist.confidence === 0,
);
assert(
  "absence trace emits low_coverage_trace hallucination note",
  absenceDist.hallucinationNotes.some((h) => h.kind === "low_coverage_trace"),
);

// ---------------------------------------------------------------------------
// Step 6: prepass bridge surfaces the right action in each case.
// ---------------------------------------------------------------------------

const ipoHint = hintFromDistillation(ipoRow!, ipoDist);
assert(
  "bridge hint for IPO is 'use-static-operator' (browser can retire)",
  ipoHint.action === "use-static-operator",
  `action=${ipoHint.action}`,
);
assert(
  "bridge hint carries the proposed operators",
  ipoHint.proposedOperators.length === ipoDist.proposedOperators.length,
);

const absenceHint = hintFromDistillation(ipoRow!, absenceDist);
assert(
  "bridge hint for absence case is 'remembered-absence' (no proposals)",
  absenceHint.action === "remembered-absence",
  `action=${absenceHint.action}`,
);
assert(
  "bridge hint surfaces remembered absences",
  absenceHint.rememberedAbsences.length === ipoTargets.length,
);

// ---------------------------------------------------------------------------
// Step 7: ranking robustness — high-marker noise on a low-utility
// resource type (e.g. an image with a coincidental matching string)
// should not outrank a real JSON fetch. We construct an in-memory trace
// to assert this directly.
// ---------------------------------------------------------------------------

const noisyTrace = {
  pageUrl: "https://example.com/x",
  requests: [
    {
      id: "noise-1",
      url: "https://example.com/og.png",
      method: "GET",
      status: 200,
      resourceType: "image" as const,
      contentType: "image/png",
      requestOrder: 0,
      // Even if the snippet accidentally matched, the image filter
      // would drop it. To make the test sharper we put it in an
      // allowed resource type below.
      responseSnippet: "Form F-1",
    },
    {
      id: "real-1",
      url: "https://example.com/_next/data/build/blog/posts/p.json",
      method: "GET",
      status: 200,
      resourceType: "fetch" as const,
      contentType: "application/json",
      requestOrder: 1,
      responseSnippet: '{"post":{"body":"Form F-1 draft registration statement"}}',
    },
  ],
};
const noisyDist = distillTrace(noisyTrace, [
  { field: "body", markers: ["Form F-1"] },
]);
assert(
  "image resources are filtered out by default allowed-resource-types",
  !noisyDist.minimalRequests.some((r) => r.id === "noise-1"),
);
assert(
  "real JSON fetch is selected even when an image had a matching snippet",
  noisyDist.minimalRequests.some((r) => r.id === "real-1"),
);

// ---------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------

let passed = 0;
for (const a of assertions) {
  if (a.ok) {
    passed += 1;
    console.log(`ok  ${a.name}` + (a.detail ? `  [${a.detail}]` : ""));
  } else {
    console.log(`FAIL  ${a.name}` + (a.detail ? `  [${a.detail}]` : ""));
  }
}
console.log(`\n${passed}/${assertions.length} passed`);
if (passed !== assertions.length) process.exit(1);
