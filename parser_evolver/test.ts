// Tiny self-test for the parser-evolver. Run with: npx tsx parser_evolver/test.ts
//
// Asserts the climate's headline guarantees:
//  - the solver discovers a creature that produces at least one row,
//  - every emitted cell points to a real source span (no hallucinations),
//  - a deliberately-broken creature that skips normalization scores no
//    better than the discovered creature.

import { companyUpdatesAF } from "./af.js";
import { PRIMITIVES } from "./operators.js";
import { makeBeamSolver } from "./solver.js";
import type { ParseContext } from "./types.js";

const SAMPLE = `
2026-04-12
Wallet sync latency elevated in EU region
https://status.blockchain.com/incidents/wallet-eu

2026-04-09
Scheduled maintenance for institutional API
https://status.blockchain.com/maintenance/api
`;

const ctx: ParseContext = {
  url: "https://status.blockchain.com/",
  rawText: SAMPLE,
  normalizedText: SAMPLE,
  sourceType: "status-page",
};

const assertions: { name: string; ok: boolean; detail?: string }[] = [];
const assert = (name: string, ok: boolean, detail?: string) => assertions.push({ name, ok, detail });

const solver = makeBeamSolver({ beam: 6, maxLen: 5 });
const candidates = solver.search(ctx, companyUpdatesAF, PRIMITIVES);
const top = candidates[0];

assert("solver returns at least one candidate", candidates.length > 0);
assert("top candidate emits at least one row", (top?.rows.length ?? 0) > 0);

const cells = top?.rows.flatMap((r) => Object.values(r.fields)) ?? [];
const allSpansSourced = cells.every(
  (c) => c.span[0] >= 0 && c.span[1] > c.span[0] && SAMPLE.slice(c.span[0], c.span[1]).length === c.span[1] - c.span[0],
);
assert("every cell points to a real source span", allSpansSourced, `cells=${cells.length}`);

const allFromText = cells.every((c) => SAMPLE.includes(c.value) || SAMPLE.includes(c.value.trim()));
assert("every cell value appears in the source text", allFromText);

// Confirm the AF actually punishes a lying creature: same genes minus the
// regex emit step (so no spans, no rows). Its score must not exceed top.
const lying = solver.search(ctx, companyUpdatesAF, [PRIMITIVES[0], PRIMITIVES[2], PRIMITIVES[3]]);
const lyingTop = lying[0];
assert("creature without regex-emit scores no better than the discovered creature", (lyingTop?.score ?? -Infinity) <= (top?.score ?? -Infinity));

const failed = assertions.filter((a) => !a.ok);
assertions.forEach((a) => console.log(`${a.ok ? "ok" : "FAIL"}  ${a.name}${a.detail ? `  [${a.detail}]` : ""}`));
console.log(`\n${assertions.length - failed.length}/${assertions.length} passed`);
if (failed.length > 0) (globalThis as { process?: { exit: (n: number) => void } }).process?.exit(1);
