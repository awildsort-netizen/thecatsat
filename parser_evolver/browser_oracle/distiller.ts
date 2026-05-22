// Trace distillation: developmental trace -> minimal static fetch plan.
//
// A browser oracle trace is treated as the *developmental history* of a
// page: the mature DOM is the organism, the network requests are the
// pathways that built it. The distiller's job is to look at the trace
// driven by the AF's needs (which fields are still missing?) and ask
//
//   "What is the smallest set of requests that, taken together,
//    carry the evidence I was after — and what URL pattern would
//    let me re-summon those bytes statically next tick?"
//
// What we explicitly do NOT do:
//   - re-execute the trace,
//   - call out to the network,
//   - parse the rendered DOM and pretend that is the static fetch,
//   - claim success when the trace did not carry the evidence
//     (we record remembered absence instead).
//
// The output is a TraceDistillation: minimal data-bearing requests,
// proposed static operators shaped exactly like ParseOperator
// (needs/provides/tokens), and explicit absences. canRetireBrowser is
// only true when at least one proposal can produce every required
// evidence field from the minimal request set.

import { readFileSync } from "node:fs";
import type {
  BrowserOracleTrace,
  EvidenceTarget,
  HallucinationNote,
  NetworkRequestTrace,
  ProposedStaticOperator,
  RememberedAbsence,
  RequestTemplate,
  TraceDistillation,
  UrlConstructionFragment,
  ResourceType,
} from "./types.js";
import type { ParseOperator } from "../types.js";
import { defineOperator, required } from "../operator_reflection.js";

export function loadTrace(path: string): BrowserOracleTrace {
  return JSON.parse(readFileSync(path, "utf8")) as BrowserOracleTrace;
}

// ---------------------------------------------------------------------------
// Content-type usefulness. JSON and HTML/text carry evidence; binaries
// don't. We rank by usefulness *before* checking evidence overlap so
// that an analytics POST cannot accidentally outscore a route-data
// fetch even if a string happened to match.
// ---------------------------------------------------------------------------

const USEFULNESS: Readonly<Record<ResourceType, number>> = {
  document: 0.7,
  fetch: 1.0,
  xhr: 0.9,
  script: 0.2,
  stylesheet: 0,
  image: 0,
  font: 0,
  media: 0,
  other: 0.1,
};

const contentTypeBoost = (ct?: string): number => {
  if (!ct) return 0;
  const c = ct.toLowerCase();
  if (c.includes("application/json")) return 0.4;
  if (c.includes("text/html")) return 0.2;
  if (c.includes("text/plain")) return 0.1;
  return 0;
};

// ---------------------------------------------------------------------------
// Evidence overlap: which target fields does this request carry?
// ---------------------------------------------------------------------------

type RequestEvidence = {
  readonly request: NetworkRequestTrace;
  readonly fields: ReadonlySet<string>;
  readonly score: number;
};

const fieldsCarriedBy = (req: NetworkRequestTrace, targets: readonly EvidenceTarget[]): ReadonlySet<string> => {
  const snippet = req.responseSnippet ?? "";
  if (snippet === "") return new Set<string>();
  const found = new Set<string>();
  for (const t of targets) {
    if (t.markers.some((m) => m !== "" && snippet.includes(m))) found.add(t.field);
  }
  return found;
};

const scoreRequest = (req: NetworkRequestTrace, targets: readonly EvidenceTarget[]): RequestEvidence => {
  const fields = fieldsCarriedBy(req, targets);
  const useful = USEFULNESS[req.resourceType] + contentTypeBoost(req.contentType);
  // Field weight folds in target.weight (default 1).
  const weights = new Map(targets.map((t) => [t.field, t.weight ?? 1]));
  let fieldWeight = 0;
  for (const f of fields) fieldWeight += weights.get(f) ?? 1;
  // Status: 200-class is what we can re-fetch statically.
  const statusOk = req.status >= 200 && req.status < 300 ? 1 : 0;
  const score = statusOk * (useful + fieldWeight);
  return { request: req, fields, score };
};

// ---------------------------------------------------------------------------
// Minimal-cover: greedy set-cover over the target fields.
//
// We pick the highest-scoring request that carries at least one not-yet
// covered field. Repeat until either every field is covered or no
// remaining request adds anything. This is the "minimal causal
// structure that produced the evidence" — short, transparent, and easy
// to reason about.
// ---------------------------------------------------------------------------

const minimalCover = (
  ranked: readonly RequestEvidence[],
  targets: readonly EvidenceTarget[],
): readonly RequestEvidence[] => {
  const need = new Set(targets.map((t) => t.field));
  const picked: RequestEvidence[] = [];
  const remaining = [...ranked];
  while (need.size > 0) {
    // Choose the candidate that newly covers the most weight; tie-break
    // by score, then by requestOrder (earlier wins).
    let best: { ev: RequestEvidence; gain: number } | undefined;
    for (const ev of remaining) {
      const newFields = [...ev.fields].filter((f) => need.has(f));
      if (newFields.length === 0) continue;
      const weights = new Map(targets.map((t) => [t.field, t.weight ?? 1]));
      const gain = newFields.reduce((s, f) => s + (weights.get(f) ?? 1), 0);
      if (
        !best ||
        gain > best.gain ||
        (gain === best.gain && ev.score > best.ev.score) ||
        (gain === best.gain && ev.score === best.ev.score && ev.request.requestOrder < best.ev.request.requestOrder)
      ) {
        best = { ev, gain };
      }
    }
    if (!best) break;
    picked.push(best.ev);
    for (const f of best.ev.fields) need.delete(f);
    const idx = remaining.indexOf(best.ev);
    if (idx >= 0) remaining.splice(idx, 1);
  }
  return picked;
};

// ---------------------------------------------------------------------------
// URL-construction fragments — how would a future operator build this
// URL from scratch? For a Next.js route-data fetch we can usually point
// at the `buildId` path segment and the `slug` query param.
// ---------------------------------------------------------------------------

const NEXT_DATA_RE = /\/_next\/data\/([^/]+)\/(.+)\.json(?:\?(.*))?$/;

const constructionFragments = (
  picked: readonly RequestEvidence[],
  trace: BrowserOracleTrace,
): readonly UrlConstructionFragment[] => {
  const frags: UrlConstructionFragment[] = [];
  for (const { request } of picked) {
    const m = NEXT_DATA_RE.exec(request.url);
    if (m) {
      frags.push({ kind: "templated", value: "next.route_payload", fromRequestId: request.id, note: "Next.js route-data fetch" });
      frags.push({ kind: "path-segment", value: m[1] ?? "", fromRequestId: request.id, note: "buildId (rotates per deploy)" });
      frags.push({ kind: "path-segment", value: m[2] ?? "", fromRequestId: request.id, note: "page route + slug" });
      if (m[3]) {
        for (const part of m[3].split("&")) {
          const [k, v] = part.split("=");
          if (k) frags.push({ kind: "query-param", value: `${k}=${v ?? ""}`, fromRequestId: request.id });
        }
      }
    }
  }
  // Page slug as a literal fragment if it appears in the trace's pageUrl.
  const slugMatch = /\/posts\/([^/?#]+)/.exec(trace.pageUrl);
  if (slugMatch && slugMatch[1]) {
    frags.push({ kind: "literal", value: slugMatch[1], note: "post slug from pageUrl" });
  }
  return frags;
};

// ---------------------------------------------------------------------------
// Proposing a static operator. We synthesise one proposal per picked
// request (since each request is a different static fetch). For Next.js
// route-data we generate the well-known operator id; otherwise we fall
// back to a generic `static.fetch.<host>.<basename>`.
// ---------------------------------------------------------------------------

const safeId = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

const operatorIdFor = (req: NetworkRequestTrace, source_id?: string): string => {
  const m = NEXT_DATA_RE.exec(req.url);
  if (m) return "next.route_payload.fetch";
  if (source_id) return `static.fetch.${safeId(source_id)}_payload`;
  try {
    const u = new URL(req.url);
    const base = u.pathname.split("/").filter(Boolean).pop() ?? "payload";
    return `static.fetch.${safeId(u.hostname)}_${safeId(base)}`;
  } catch {
    return `static.fetch.${safeId(req.url)}`;
  }
};

const materialHintsFor = (req: NetworkRequestTrace): readonly string[] => {
  const tokens = new Set<string>(["network", "fetch", req.resourceType]);
  if (req.contentType?.includes("json")) tokens.add("json");
  if (req.contentType?.includes("text/html")) tokens.add("html");
  if (NEXT_DATA_RE.test(req.url)) {
    tokens.add("next-data");
    tokens.add("route-payload");
  }
  if (/\/blog\//.test(req.url)) tokens.add("blog-post");
  return [...tokens];
};

const proposeOperator = (
  ev: RequestEvidence,
  targets: readonly EvidenceTarget[],
  source_id?: string,
): ProposedStaticOperator => {
  const evidenceFields = [...ev.fields];
  // Confidence: starts at request usefulness, lifted by 200-status and
  // how many target fields it covers. Capped at 0.95 — a distiller can
  // never be certain a route will stay static next tick.
  const useful = USEFULNESS[ev.request.resourceType] + contentTypeBoost(ev.request.contentType);
  const ok = ev.request.status >= 200 && ev.request.status < 300 ? 1 : 0;
  const coverage = evidenceFields.length / Math.max(1, targets.length);
  const confidence = Math.min(0.95, Math.max(0, ok * (0.3 + 0.4 * useful + 0.4 * coverage)));
  // Note: `needs`/`provides` are deliberately not set on the proposal.
  // The lifter derives them from the implementation IO so the proposal
  // carries only authored material (evidenceFields, requestTemplate,
  // materialHints) and signature is a downstream projection.
  return {
    id: operatorIdFor(ev.request, source_id),
    tokens: materialHintsFor(ev.request),
    requestTemplate: {
      url: ev.request.url,
      method: ev.request.method,
      contentType: ev.request.contentType,
      accept: ev.request.contentType,
    },
    urlPattern: NEXT_DATA_RE.test(ev.request.url) ? "/_next/data/{buildId}/{route}.json" : undefined,
    evidenceFields,
    materialHints: materialHintsFor(ev.request),
    sourceRequestIds: [ev.request.id],
    cost: 1.2,
    confidence: Math.round(confidence * 1000) / 1000,
  };
};

// ---------------------------------------------------------------------------
// Distillation entry point.
// ---------------------------------------------------------------------------

export type DistillOptions = {
  readonly source_id?: string;
  // If set, the distiller will only consider these resource types as
  // candidate evidence carriers. Default: everything except images/fonts.
  readonly allowedResourceTypes?: readonly ResourceType[];
};

const DEFAULT_ALLOWED: readonly ResourceType[] = [
  "document",
  "fetch",
  "xhr",
  "script",
  "other",
];

export function distillTrace(
  trace: BrowserOracleTrace,
  targets: readonly EvidenceTarget[],
  opts?: DistillOptions,
): TraceDistillation {
  const allowed = new Set(opts?.allowedResourceTypes ?? DEFAULT_ALLOWED);
  const requests = trace.requests.filter((r) => allowed.has(r.resourceType));

  const ranked = requests
    .map((r) => scoreRequest(r, targets))
    .sort((a, b) => b.score - a.score || a.request.requestOrder - b.request.requestOrder);

  // A request only enters minimal-cover if it carries at least one
  // target field. Score-only-positive requests are kept for diagnostics
  // but don't reach the proposed-operator stage.
  const carriers = ranked.filter((ev) => ev.fields.size > 0);
  const picked = minimalCover(carriers, targets);

  const evidenceProduced = [...new Set(picked.flatMap((ev) => [...ev.fields]))];
  const requiredFields = targets.map((t) => t.field);
  const missingFields = requiredFields.filter((f) => !evidenceProduced.includes(f));

  const proposedOperators = picked.map((ev) => proposeOperator(ev, targets, opts?.source_id));

  // Remembered absences: explicit "this trace did not carry field X".
  const rememberedAbsences: RememberedAbsence[] = missingFields.map((field) => {
    // If a hydration payload carries the marker but no request did,
    // record that distinction — it tells a future synth step it cannot
    // statically refetch the field without re-rendering.
    const hydrationCarried = (trace.hydrationPayloads ?? []).some((h) =>
      targets
        .find((t) => t.field === field)
        ?.markers.some((m) => h.textSnippet.includes(m)) ?? false,
    );
    const domCarried = (trace.candidateEvidenceRegions ?? []).some((r) =>
      (r.fields ?? []).includes(field),
    );
    if (hydrationCarried) {
      return {
        field,
        reason: "evidence_only_in_hydration_payload",
        note: "Field appeared in an inlined hydration payload but no separate request carried it.",
      };
    }
    if (domCarried) {
      return {
        field,
        reason: "evidence_only_in_dom_not_in_network",
        note: "Field was observed in the rendered DOM but not in any captured network response.",
      };
    }
    return {
      field,
      reason: "no_request_contained_evidence",
      note: "No captured request body contained any marker for this field.",
    };
  });

  const hallucinationNotes: HallucinationNote[] = [];
  if (picked.length === 0 && requests.length > 0) {
    hallucinationNotes.push({
      kind: "low_coverage_trace",
      note: "Trace had requests but none carried target evidence; do not propose operators from this trace.",
      weight: 1,
    });
  }
  if (picked.length > 0 && evidenceProduced.length < requiredFields.length) {
    hallucinationNotes.push({
      kind: "low_coverage_trace",
      note: `Trace covered ${evidenceProduced.length}/${requiredFields.length} required fields.`,
      weight: 0.5,
    });
  }

  const urlConstructionFragments = constructionFragments(picked, trace);

  // canRetireBrowser: trace produced *every* required field from
  // minimal requests, and at least one operator is high-confidence.
  const canRetireBrowser =
    requiredFields.length > 0 &&
    missingFields.length === 0 &&
    proposedOperators.some((p) => p.confidence >= 0.6);

  const confidence =
    proposedOperators.length === 0
      ? 0
      : Math.round(
          (proposedOperators.reduce((s, p) => s + p.confidence, 0) / proposedOperators.length) * 1000,
        ) / 1000;

  return {
    pageUrl: trace.pageUrl,
    evidenceProduced,
    minimalRequests: picked.map((ev) => ev.request),
    urlConstructionFragments,
    proposedOperators,
    rememberedAbsences,
    hallucinationNotes,
    canRetireBrowser,
    confidence,
  };
}

// ---------------------------------------------------------------------------
// Lifting a proposal into a real ParseOperator. The run-body is a stub
// — it announces what it would fetch and returns a structured input
// shape that a downstream emitter can hand to the AF. We keep the
// actual fetch out of this prototype on purpose; the point is that the
// *operator shape* fits the parser_evolver vocabulary unchanged.
//
// The signature is *derived* from the IO declaration below via
// `defineOperator`. `needs` is empty (the lifted operator reads nothing
// from upstream — it is a source-style operator that announces a fetch
// plan) and `provides` is a single proposal-output channel keyed by
// the proposal id. Per-evidence-field provides are intentionally not
// claimed here, because this stub does not actually carry the bytes —
// the surrounding harness must execute the fetch and feed the result
// back. Lying about provides would defeat the whole point of letting
// signatures reflect implementation.
// ---------------------------------------------------------------------------

export type ProposalRunOutput = {
  readonly proposalId: string;
  readonly requestTemplate: RequestTemplate;
  readonly evidenceFields: readonly string[];
  readonly upstream: unknown;
  readonly note: string;
};

export const liftProposalToOperator = (proposal: ProposedStaticOperator): ParseOperator =>
  defineOperator({
    id: proposal.id,
    cost: proposal.cost,
    tokens: proposal.tokens,
    // No required inputs: this is a source-style operator that reads
    // nothing from upstream and announces a fetch plan. An empty
    // `inputs` object reflects exactly that — and the legacy
    // `signature.needs` projection comes out as the empty array
    // because there is nothing required to project.
    inputs: {},
    outputs: {
      "browser_oracle.proposal": required<ProposalRunOutput>(),
    },
    run: (_ctx, _input) => ({
      "browser_oracle.proposal": {
        proposalId: proposal.id,
        requestTemplate: proposal.requestTemplate,
        evidenceFields: proposal.evidenceFields,
        upstream: _input,
        note: "Lifted from a browser-oracle TraceDistillation. Static fetch is not executed inside parser_evolver; the surrounding harness must perform the request and feed bytes back as ParseContext.rawText.",
      },
    }),
  });
