# parser_evolver/browser_oracle — browser as fallback oracle, not as truth

A tiny prototype of the developmental-trace model for the crawler. The
architecture stops being scraping and becomes epistemology: the crawler
no longer asks "what's on the page?" but "**what minimal causal
structure produced the evidence I needed?**"

A rendered browser session is treated as a *developmental trace* — the
embryology of the page. The mature organism (the DOM) is interesting,
but the path that *built* it (network requests, hydration payloads,
script-driven URLs) is what tells us how to fetch the same evidence
statically next time.

A browser fallback is a **plasticity event**:

```
static parse fails to produce required evidence
  -> browser oracle trace is captured (externally, not in this repo)
  -> trace distillation: rank requests by overlap with target evidence
  -> propose a static fetch operator (needs / provides / tokens)
  -> next tick runs the static operator; browser is retired for this source
```

This is how the system can *learn how to find data without running a
browser*.

This module is fixture-driven on purpose. It does **not** launch a
browser, does **not** hit the network, and is **not** wired into the
live Perplexity scheduled monitor. The browser oracle is an *input*
to the system — JSON the surrounding harness (Playwright, Puppeteer, a
HAR export, a manual capture) provides; the parser_evolver side of the
loop is the distillation, operator synthesis, and remembered absence.

## The loop, slowly

1. **Static failure.** `parser_evolver/prepass` already escalates
   SPA-shell sources to `needs-rendered-fetch` (see
   [../prepass/README.md](../prepass/README.md)). For the Blockchain.com
   IPO post, the static snapshot is the same shell as the blog index —
   the post body is not in the HTML. The prepass row flips
   `expected_action: needs-rendered-fetch` and drops confidence.
2. **Browser oracle trace.** An external headless run produces a
   `BrowserOracleTrace` JSON capturing the network requests, scripts,
   hydration payloads, and candidate evidence regions for that URL.
   For this prototype we ship two fixture traces in
   [`fixtures/`](fixtures/) — see the `_fixture_note` field on each.
3. **Trace distillation.** Given the trace and a set of
   `EvidenceTarget`s (which AF fields are we missing, and what markers
   identify them?), the distiller:
     - ranks each request by content-type usefulness, status, and
       overlap with the target markers;
     - performs a greedy set-cover to pick the **smallest set of
       requests** that together carry every target field;
     - emits **URL-construction fragments** describing how the URL was
       built (Next.js `buildId`, `slug`, query params);
     - emits a **`ProposedStaticOperator`** per minimal request,
       carrying only authored material (`evidenceFields`,
       `requestTemplate`, `materialHints`, `tokens`, `cost`). The
       `needs` / `provides` signature is **derived** at
       `liftProposalToOperator` time from the lifted run-body's IO
       shape via [`defineOperator`](../operator_reflection.ts) — it is
       not hand-authored on the proposal, because that would duplicate
       (and inevitably drift from) the implementation. See
       [`../docs/signatures_first.md`](../docs/signatures_first.md).
       For Next.js route-data fetches the operator id is the
       well-known `next.route_payload.fetch`; otherwise we synthesise
       `static.fetch.<source_id>_payload`;
     - **records remembered absence** for any required field the trace
       did not carry, distinguishing
       `no_request_contained_evidence`,
       `evidence_only_in_dom_not_in_network`, and
       `evidence_only_in_hydration_payload`.
4. **Browser retirement.** `canRetireBrowser` is `true` *only* when
   every required field is covered by the minimal request set and at
   least one operator passes a confidence floor. Otherwise the browser
   fallback stays armed.
5. **Prepass bridge.** `prepass_bridge.ts` turns a distillation plus the
   existing prepass `DigestRow` into a `PrepassHint` with one of three
   actions: `use-static-operator` (browser retired), `keep-needs-rendered-fetch`
   (partial evidence), or `remembered-absence` (no proposals; do not
   re-summon a browser hoping for a different result).

## What's here

- [`types.ts`](types.ts) — `BrowserOracleTrace`, `NetworkRequestTrace`,
  `ScriptTrace`, `HydrationPayload`, `CandidateEvidenceRegion`,
  `EvidenceTarget`, `ProposedStaticOperator`, `TraceDistillation`,
  `RememberedAbsence`, `HallucinationNote`, `UrlConstructionFragment`.
- [`distiller.ts`](distiller.ts) — pure distiller, plus
  `liftProposalToOperator` which lifts a proposal into a real
  `ParseOperator` with the parser_evolver signature shape.
- [`prepass_bridge.ts`](prepass_bridge.ts) — `hintFromDistillation`.
- [`fixtures/`](fixtures/) — two synthetic traces. `blog-ipo-announce.trace.json`
  is the positive case (Next.js route-data carries the IPO body);
  `blog-index-absence.trace.json` is the negative case (a full session
  on the blog index, but the IPO body is not in any captured request).
  Both fixtures are clearly marked `synthetic` — they are **not**
  captured live.
- [`test.ts`](test.ts) — 28 assertions covering: prepass escalation,
  minimal-cover correctness, operator id / pattern / provides /
  tokens / confidence shape, lifting into `ParseOperator`, absence
  semantics, low-utility-resource filtering, and the bridge actions.
- [`demo.ts`](demo.ts) — narrates the seven-step loop end-to-end.

## Run

```bash
cd parser_evolver
npm install                       # tsc + tsx
npm run browser-oracle-test       # 28 assertions
npm run browser-oracle-demo       # end-to-end developmental-trace narration
npm run all                       # everything (check + all tests + all demos)
```

## What this is *not*

- Not a browser driver. Nothing here calls Playwright/Puppeteer/HAR.
  Traces are JSON we ingest.
- Not wired to the live scheduled monitor.
- Not an exhaustive operator synthesiser. We emit a `ProposedStaticOperator`
  shape; turning that into a fully run-able fetcher inside `parser_evolver`
  is a later step.
- Not a fully-general distiller. Set-cover is greedy and the
  evidence-marker matcher is substring-based. Both can sharpen later;
  the point of this seed is that the loop *closes* and the absence case
  doesn't lie.

## See also

- [../prepass/README.md](../prepass/README.md) — the static prepass
  whose escalation triggers the browser fallback.
- [../docs/hallucination_geometry.md](../docs/hallucination_geometry.md)
  — the typed-pressure / semantic-plaque language that the
  `RememberedAbsence` and `HallucinationNote` types extend into
  developmental space.
