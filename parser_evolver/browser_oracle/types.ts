// Browser-oracle / developmental-trace types.
//
// The crawler stops asking "what's on the page?" and starts asking
// "what minimal causal structure produced the evidence I needed?". A
// rendered browser session is treated as a *developmental trace* — the
// embryology of the page — not as ground truth. The mature organism
// (the rendered DOM) is interesting; the path that built it
// (network requests, hydration payloads, script-driven URLs) is what
// tells us how to fetch the same evidence statically next time.
//
// A browser fallback in this system is a plasticity event:
//
//   static parse fails to produce required evidence
//     -> a (separately captured) browser oracle trace is consulted
//     -> the trace is distilled into a minimal set of data-bearing requests
//     -> a static fetch operator is proposed (needs / provides shaped)
//     -> on the next tick, the static operator runs first
//        and the browser is *retired* for this source.
//
// This module defines the JSON shape that an external browser session
// (Puppeteer, Playwright, a manual HAR export, etc.) would dump for us.
// No browser is launched inside this repo; we ingest fixture JSON only.
//
// Types here intentionally mirror the parser_evolver style:
//   - needs/provides on proposed operators (the same currency the AF
//     and beam solver already speak);
//   - typed absence ("remembered absence") instead of silent failure;
//   - small, type-driven, no broad imperative branching.

import type { ParseOperator } from "../types.js";

// ---------------------------------------------------------------------------
// Raw browser trace as captured by an external oracle.
//
// Shape is intentionally narrow: a HAR file has dozens of fields per
// entry, but a developmental trace really only needs to know
// "which requests carried which bytes, and what was the page's URL?".
// ---------------------------------------------------------------------------

export type ResourceType =
  | "document"
  | "xhr"
  | "fetch"
  | "script"
  | "stylesheet"
  | "image"
  | "font"
  | "media"
  | "other";

export type RequestTiming = {
  readonly startedAtMs: number;
  readonly durationMs?: number;
};

export type NetworkRequestTrace = {
  readonly id: string;
  readonly url: string;
  readonly method: string;
  readonly status: number;
  readonly resourceType: ResourceType;
  readonly contentType?: string;
  readonly initiator?: string;
  readonly requestOrder: number;
  readonly bytes?: number;
  readonly timing?: RequestTiming;
  // A short slice of the response — enough to detect target evidence
  // tokens. Full bodies live at `responseBodyPath` if the oracle wrote
  // them to disk; the snippet alone is sufficient for distillation in
  // this prototype.
  readonly responseSnippet?: string;
  readonly responseBodyPath?: string;
};

export type ScriptTrace = {
  readonly url: string;
  readonly bytes?: number;
  readonly note?: string;
};

// A "hydration payload" is a structured data blob that the browser
// applied to the DOM after first paint — Next.js' `__NEXT_DATA__`,
// Nuxt's `__NUXT__`, Apollo's `__APOLLO_STATE__`, etc. We capture them
// separately from network requests because they are usually inlined in
// the document HTML, not fetched as JSON.
export type HydrationPayload = {
  readonly framework: "next" | "nuxt" | "apollo" | "redux" | "other";
  readonly key: string;
  readonly textSnippet: string;
  readonly bytes?: number;
};

// A candidate region inside the rendered DOM where target evidence was
// observed. The oracle marks these; we don't try to re-extract from
// rendered HTML — we use them as ground truth for what evidence the
// distiller is looking for.
export type CandidateEvidenceRegion = {
  readonly label: string;
  readonly text: string;
  readonly cssPath?: string;
  readonly fields?: readonly string[];
};

export type BrowserOracleTrace = {
  readonly pageUrl: string;
  readonly capturedAt?: string;
  readonly userAgent?: string;
  readonly requests: readonly NetworkRequestTrace[];
  readonly scripts?: readonly ScriptTrace[];
  readonly hydrationPayloads?: readonly HydrationPayload[];
  readonly candidateEvidenceRegions?: readonly CandidateEvidenceRegion[];
  // Optional notes about the capture (fixture vs live, what driver, etc).
  readonly captureNotes?: string;
};

// ---------------------------------------------------------------------------
// What the distiller produces.
//
// A `ProposedStaticOperator` is shaped so it can later be promoted into a
// real ParseOperator (needs/provides/tokens) without changing the shape
// of operators.ts. It carries enough material to re-fetch the same
// bytes without a browser:
//
//   - `requestTemplate` or `urlPattern`: the literal or templated URL,
//     plus optional method/headers/body. Fragments captured from the
//     trace let a downstream synthesiser parameterise it.
//   - `evidenceFields`: which AF columns this operator would provide.
//   - `materialHints`: small tokens used by the embedding/solver layer
//     (resource type, content type, "json", "next-data", "blog-post").
//   - `sourceRequestIds`: which raw requests in the trace contributed.
// ---------------------------------------------------------------------------

export type UrlConstructionFragment = {
  readonly kind: "literal" | "path-segment" | "query-param" | "hash" | "templated";
  readonly value: string;
  readonly fromRequestId?: string;
  readonly note?: string;
};

export type RequestTemplate = {
  readonly url: string;
  readonly method: string;
  readonly contentType?: string;
  readonly accept?: string;
  // Open hook for future operator synthesis; intentionally not used by
  // the prototype solver yet.
  readonly templatedFragments?: readonly UrlConstructionFragment[];
};

export type ProposedStaticOperator = {
  readonly id: string;
  readonly needs: readonly string[];
  readonly provides: readonly string[];
  readonly tokens: readonly string[];
  readonly requestTemplate: RequestTemplate;
  readonly urlPattern?: string;
  readonly evidenceFields: readonly string[];
  readonly materialHints: readonly string[];
  readonly sourceRequestIds: readonly string[];
  // Notional cost — mirrors `ParseOperator.cost` so a future synthesis
  // step can drop the proposal straight into a solver.
  readonly cost: number;
  // How confident the distiller is that this operator will actually
  // re-produce the evidence statically. Distinct from solver score.
  readonly confidence: number;
};

// "Remembered absence" is the developmental-trace analogue of
// `low_coverage_region`: we *looked* for evidence and recorded that the
// trace doesn't carry it, so future ticks should not re-summon a
// browser hoping for a different result without new input.
export type RememberedAbsence = {
  readonly field: string;
  readonly reason:
    | "no_request_contained_evidence"
    | "evidence_only_in_dom_not_in_network"
    | "evidence_only_in_hydration_payload"
    | "trace_did_not_include_field";
  readonly note?: string;
};

export type HallucinationNote = {
  readonly kind: "low_coverage_trace" | "ambiguous_request_attribution" | "missing_hydration_payload";
  readonly note: string;
  readonly weight: number;
};

export type TraceDistillation = {
  readonly pageUrl: string;
  // Did the trace produce *any* of the requested evidence?
  readonly evidenceProduced: readonly string[];
  // Minimal data-bearing requests, ranked highest-overlap first.
  readonly minimalRequests: readonly NetworkRequestTrace[];
  // Optional structural hints harvested from URLs in the trace.
  readonly urlConstructionFragments: readonly UrlConstructionFragment[];
  readonly proposedOperators: readonly ProposedStaticOperator[];
  readonly rememberedAbsences: readonly RememberedAbsence[];
  readonly hallucinationNotes: readonly HallucinationNote[];
  // Browser retirement: if true, the next tick can run static operators
  // alone for this source. False keeps the browser fallback armed.
  readonly canRetireBrowser: boolean;
  // Aggregate confidence the distiller places on its proposal set.
  readonly confidence: number;
};

// ---------------------------------------------------------------------------
// Input describing what evidence we wanted in the first place. The
// distiller is *driven by need* — it doesn't try to summarise the
// trace, it tries to answer "did this trace produce the fields the AF
// said we still needed?".
// ---------------------------------------------------------------------------

export type EvidenceTarget = {
  // The AF field name (date, title, url, body, ...). Mirrors the
  // CsvAF column vocabulary the parser_evolver already speaks.
  readonly field: string;
  // Exact substrings whose presence in a response counts as evidence
  // for `field`. Multiple strings are OR-ed.
  readonly markers: readonly string[];
  // Optional weight — lets the caller say "title is required, url is
  // nice to have".
  readonly weight?: number;
};

// Helper for callers that want to promote a proposal to a real
// ParseOperator. Kept as a free function in distiller.ts; this is just
// the contract. A proposal lifts into an operator with the same
// needs/provides/tokens; the run-body becomes "fetch this URL and
// return its bytes as candidate input for the AF".
export type ProposalLifter = (proposal: ProposedStaticOperator) => ParseOperator;
