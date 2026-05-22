# Browser as developmental trace — design note

Companion to [hallucination_geometry.md](hallucination_geometry.md).
This note explains the model that drives the
[`browser_oracle/`](../browser_oracle/) prototype.

## What changes

The crawler architecture stops being scraping and becomes
**epistemology**.

The old question — "what's on the page?" — assumes the page is the
truth. It treats the rendered DOM as ground and tries to extract from
it directly, then falls back to a browser whenever static fetches don't
return enough HTML. That makes the browser an emergency tool. It is
expensive, brittle, and easy to *mis-cache* (a one-off rendered page
gets cached as if it were the answer, instead of as a lesson about how
to find the answer next time).

The new question is:

> What minimal causal structure produced the evidence I needed?

The rendered DOM is **the mature organism**. The network requests,
hydration payloads, and script chains that built it are **the
developmental pathway**. We are interested in the pathway. The
organism is interesting because it tells us *which fragments of the
pathway carried the evidence we were after*, but the pathway itself is
what we will replay statically next tick.

## The plasticity event

A browser fallback in this model is a plasticity event: the system's
extraction layer is changing shape because the static path failed.
That is precisely when we should *learn*.

```
static parse fails to produce required evidence
        │
        ▼
external browser oracle runs (Playwright/Puppeteer/HAR/etc.)
        │   (out of this repo; just JSON to us)
        ▼
BrowserOracleTrace ingested as fixture / hint
        │
        ▼
TraceDistillation
   - rank requests by content-type usefulness × evidence overlap
   - greedy set-cover -> minimal data-bearing requests
   - propose static fetch operator (needs / provides / tokens)
   - record remembered absence for any uncovered field
        │
        ▼
PrepassHint
   - use-static-operator         (browser retired for this source)
   - keep-needs-rendered-fetch   (partial evidence)
   - remembered-absence          (no proposals; don't re-summon a browser)
```

The output is operator-shaped. The proposal is not a one-off
extraction — it is a `ProposedStaticOperator` whose `needs`/`provides`/
`tokens` plug straight into the parser_evolver beam solver. The
browser is retired not because we stopped caring about that source but
because we have a static operator that can produce the same evidence
without it.

## Why "remembered absence" matters

A naive learning loop would treat a browser session that *did not*
yield the target evidence as a no-op: try again later. The
developmental-trace model treats it as a positive signal — a recorded
*absence*. The trace failed to carry the IPO body? Then the IPO body
is not in the network layer of *this* URL, and a future tick should
not waste another browser run on the same URL with the same evidence
target. The system can still escalate to a human, or change the URL
target, or change the markers; what it cannot do is loop on a renderer
hoping for a different result.

This is exactly the typed-pressure stance from
`hallucination_geometry.md` extended into developmental space:
`low_coverage_region` says "this region of source produced no kept
cells"; `RememberedAbsence(reason=no_request_contained_evidence)` says
"this region of *development* produced no kept request".

## Why URL-construction fragments

The proposal cares about the URL **shape**, not the URL **string**. A
Next.js route-data URL looks like
`/_next/data/{buildId}/{route}.json?slug={slug}`. The `buildId`
rotates on every deploy. A fragment-aware proposal can survive a
rebuild by re-templating; a URL-string-only proposal cannot. The
prototype captures fragments but doesn't yet re-template at fetch
time — that is the obvious next step.

## What we explicitly avoid

- Re-executing the trace inside parser_evolver. No JS engine, no
  fetcher, no scheduler. The trace is JSON we read.
- Treating the rendered DOM as the static fetch. The DOM tells us
  what evidence was visible; the *requests* tell us how to get the
  same bytes back without a renderer.
- Claiming success on partial coverage. `canRetireBrowser` only flips
  true when the minimal request set covers every required field *and*
  at least one proposed operator clears a confidence floor.
- Talking to the live monitor. The prototype is fixture-driven and the
  scheduled monitor is untouched.

## Where this goes next

- A real synthesiser turning `UrlConstructionFragment`s into a fetch
  function (still pluggable, still parser_evolver-shaped).
- A small operator-promotion step: lift a `ProposedStaticOperator`
  into the operator set after N successful re-fetches.
- Replace the substring marker matcher with the embedding layer the
  beam solver already uses, so target fields can be described by
  tokens rather than literal strings.
- Pair this with `prepass`: rows whose
  `expected_action === "needs-rendered-fetch"` are exactly the rows
  the developmental-trace loop should run on.
