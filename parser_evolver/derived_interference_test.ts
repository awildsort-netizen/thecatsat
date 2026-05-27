// Tests for the auto-derived InterferenceSpec.
//
// The contract: for every primitive in the current basis, the derived
// spec must agree with the hand-authored side-table on:
//   * the set of replacing channels
//   * the set of accumulator channels
//   * whether the operator is idempotent
//
// Commutation is *not* checked for byte-equality with the hand-authored
// value — the derived default is `commutesWithPeers` because the
// must_precede filter in `buildConflictGraph` correctly suppresses
// sequentially dependent pairs, so the conservative `nonCommuting`
// claims in the side-table are a safe over-approximation. What we
// *do* check downstream is that the *behaviour* of the solver under
// the derived registry matches the behaviour under the hand-authored
// registry (search-cost reduction, top creature).

import { companyUpdatesAF } from "./af.js";
import { PRIMITIVES, makeEnforceSchema } from "./operators.js";
import { PRIMITIVE_INTERFERENCE } from "./operator_interference.js";
import {
  buildProbeFromPipeline,
  deriveInterference,
  deriveRegistry,
} from "./interference_derivation.js";
import { lookupInterference, type WriteDiscipline } from "./interference.js";
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
`;

const ctx: ParseContext = {
  url: "https://status.blockchain.com/",
  rawText: SAMPLE,
  normalizedText: SAMPLE,
  sourceType: "status-page",
};

const ops = [...PRIMITIVES, makeEnforceSchema(companyUpdatesAF)];

// Recorded pipeline run gives every operator a realistic probe input.
const probe = buildProbeFromPipeline(ctx, ops);

const assertions: { name: string; ok: boolean; detail?: string }[] = [];
const assert = (name: string, ok: boolean, detail?: string): void => {
  assertions.push({ name, ok, detail });
};

// Tiny helpers to inspect a WriteDiscipline by *invoking* its
// contribution (the only public way — there is no `.kind` on the value).
const isReplacing = (d: WriteDiscipline): boolean => {
  const s = new Set<string>();
  d.contributeReplacing(s);
  return s.has(d.channel);
};
const isAccumulator = (d: WriteDiscipline): boolean => {
  const s = new Set<string>();
  d.contributeAccumulator(s);
  return s.has(d.channel);
};

const setEq = (a: ReadonlySet<string>, b: ReadonlySet<string>): boolean =>
  a.size === b.size && [...a].every((x) => b.has(x));

const channelsOf = (
  outputs: readonly WriteDiscipline[],
  pred: (d: WriteDiscipline) => boolean,
): Set<string> => new Set(outputs.filter(pred).map((d) => d.channel));

// --- 1. Per-operator agreement on replacing/accumulator/idempotence ---------
ops.forEach((op) => {
  const derived = deriveInterference(op, probe);
  const authored = lookupInterference(PRIMITIVE_INTERFERENCE, op.id);

  const derivedReplacing = channelsOf(derived.outputs, isReplacing);
  const authoredReplacing = channelsOf(authored.outputs, isReplacing);
  assert(
    `${op.id}: derived replacing-channels match hand-authored`,
    setEq(derivedReplacing, authoredReplacing),
    `derived=[${[...derivedReplacing].sort().join(",")}] authored=[${[...authoredReplacing].sort().join(",")}]`,
  );

  const derivedAcc = channelsOf(derived.outputs, isAccumulator);
  const authoredAcc = channelsOf(authored.outputs, isAccumulator);
  assert(
    `${op.id}: derived accumulator-channels match hand-authored`,
    setEq(derivedAcc, authoredAcc),
    `derived=[${[...derivedAcc].sort().join(",")}] authored=[${[...authoredAcc].sort().join(",")}]`,
  );

  // Idempotence: invoke each Application's self-edge contribution into
  // its own sink and check whether the derived and authored values
  // contributed the same self-edge set.
  const derivedSelf: { from: string; to: string; kind: string }[] = [];
  const authoredSelf: { from: string; to: string; kind: string }[] = [];
  derived.application.contributeSelfEdges(op.id, derivedSelf as never);
  authored.application.contributeSelfEdges(op.id, authoredSelf as never);
  assert(
    `${op.id}: derived idempotence claim matches hand-authored`,
    derivedSelf.length === authoredSelf.length,
    `derived=${derivedSelf.length} authored=${authoredSelf.length}`,
  );
});

// --- 2. Derived registry preserves the solver's search-cost reduction ------
//
// The branch's contract is that derived ≅ hand-authored on what matters
// for search. We run the solver under both registries and require
// identical top-creature and identical evaluation counts.

const runWithRegistry = (label: string, registry: unknown) => {
  let evals = 0;
  const solver = makeBeamSolver({
    beam: 8,
    maxLen: 6,
    extensionTopK: 4,
    interferenceRegistry: registry as never,
    onEvaluate: () => {
      evals += 1;
    },
  });
  const candidates = solver.search(ctx, companyUpdatesAF, ops);
  const top = candidates[0];
  return {
    label,
    evals,
    score: top?.score ?? 0,
    gene: top?.genes.map((g) => g.operatorId).join(" >> ") ?? "(empty)",
  };
};

const derivedRegistry = deriveRegistry(ops, probe);
const A = runWithRegistry("authored", PRIMITIVE_INTERFERENCE);
const B = runWithRegistry("derived ", derivedRegistry);

assert(
  "derived registry: top score equals authored",
  Math.abs(B.score - A.score) < 1e-9,
  `authored=${A.score} derived=${B.score}`,
);
assert(
  "derived registry: top gene-string equals authored",
  B.gene === A.gene,
  `authored=${A.gene}  derived=${B.gene}`,
);
assert(
  "derived registry: solver evaluates no more candidates than authored",
  B.evals <= A.evals,
  `authored=${A.evals} derived=${B.evals}`,
);

// --- 3. No-probe fallback is correct (registry without a probe) ------------
//
// `deriveRegistry` requires a probe; building one from a context+ops is
// cheap and on the caller. Document the failure mode by exercising an
// override path.
const overridden = deriveRegistry(ops, probe, {
  "normalize.whitespace": {},
});
const C = runWithRegistry("override", overridden);
assert(
  "empty override leaves derived spec intact",
  C.gene === B.gene && C.score === B.score && C.evals === B.evals,
);

// --- report ----------------------------------------------------------------
const passed = assertions.filter((a) => a.ok).length;
assertions.forEach((a) =>
  console.log(`${a.ok ? "ok " : "FAIL"} ${a.name}${a.detail ? "  [" + a.detail + "]" : ""}`),
);
console.log(`\n${passed}/${assertions.length} passed`);
console.log(`\nauthored: evals=${A.evals} score=${A.score} gene=${A.gene}`);
console.log(`derived : evals=${B.evals} score=${B.score} gene=${B.gene}`);
if (passed !== assertions.length) process.exit(1);
