// Interference specs for the existing operators.
//
// Each entry below is a *claim* about the operator's algebra: how its
// writes compose, whether it's idempotent, which of its outputs are
// accumulators. Specs are expressed via the discriminated-union
// constructors in `interference.ts` so consumers dispatch on
// `kind` instead of branching on booleans. Adding a new operator
// without a spec falls back to DEFAULT_INTERFERENCE (assume worst),
// so the system stays correct under absence-of-information.
//
// Why two files? `interference.ts` is the *type and algorithm*: the
// shape of InterferenceSpec, the conflict graph, the propagation-coherence
// metric. This file is the *content*: which spec applies to which
// operator. Splitting them keeps the algorithmic surface clean and
// makes it easy to swap in a different operator basis without
// touching the algorithm.

import {
  accumulator,
  commutesWithPeers,
  idempotent,
  makeInterferenceRegistry,
  nonCommuting,
  nonIdempotent,
  purelyAdditive,
  replacing,
  type InterferenceRegistry,
  type InterferenceSpec,
} from "./interference.js";

// normalize.whitespace
// --------------------
// Pure function of ctx.normalizedText -> a single replacing write to
// text.normalized. Idempotent: collapsing whitespace twice is the same
// as collapsing once. No accumulator outputs. Does not commute with
// any peer (no other operator writes text.normalized in this basis,
// so the question is vacuous; the conservative claim is non_commuting).
const NORMALIZE_WHITESPACE: InterferenceSpec = {
  outputs: [replacing("text.normalized")],
  application: idempotent,
  commutation: nonCommuting,
};

// regex.emit.url
// --------------
// Replacing on spans.url, accumulator on trace.regions. The replacing
// channel stabilises after one run; the accumulator channel keeps
// growing on repeat application, so the operator is `idempotent` but
// not `purely_additive`. Commutes with peer emitters: regex.emit.url
// reads only text.normalized, regex.emit.title reads only
// text.normalized; reordering them does not change either's output.
// regex.emit.date is a must_precede dependency (it reads spans.url)
// and the conflict-graph builder filters that case out automatically,
// so we can safely advertise commutes_with_peers here.
const REGEX_EMIT_URL: InterferenceSpec = {
  outputs: [replacing("spans.url"), accumulator("trace.regions")],
  application: idempotent,
  commutation: commutesWithPeers,
};

// regex.emit.date
// ---------------
// Replacing on spans.dated, accumulator on trace.regions. Reads
// spans.url, so has a hard must_precede edge from regex.emit.url
// that the conflict graph picks up from channels.requiredInputs. We
// still advertise commutes_with_peers — the must_precede filter in
// buildConflictGraph will correctly suppress the (url, date) pair
// while admitting the (date, title) pair.
const REGEX_EMIT_DATE: InterferenceSpec = {
  outputs: [replacing("spans.dated"), accumulator("trace.regions")],
  application: idempotent,
  commutation: commutesWithPeers,
};

// regex.emit.title
// ----------------
// Replacing on spans.titled, accumulator on trace.regions. Reads
// text.normalized and spans.url (to exclude URL spans). Commutes
// with regex.emit.date (neither reads the other's primary output);
// the must_precede filter handles the (url, title) ordering.
const REGEX_EMIT_TITLE: InterferenceSpec = {
  outputs: [replacing("spans.titled"), accumulator("trace.regions")],
  application: idempotent,
  commutation: commutesWithPeers,
};

// row.assemble.proximity
// ----------------------
// Replacing on rows.assembled. Reads spans.dated and spans.titled
// (with optional spans.url); has must_precede edges from all three
// emitters. Idempotent on a fixed input. Does not commute with any
// peer: it's the only writer to rows.assembled.
const ROW_ASSEMBLE_PROXIMITY: InterferenceSpec = {
  outputs: [replacing("rows.assembled")],
  application: idempotent,
  commutation: nonCommuting,
};

// row.enforce.schema
// ------------------
// Replacing on rows.validated, accumulator on
// hallucinations.collected. Reads rows.assembled (must_precede from
// proximity). Idempotent on a fixed input.
const ROW_ENFORCE_SCHEMA: InterferenceSpec = {
  outputs: [replacing("rows.validated"), accumulator("hallucinations.collected")],
  application: idempotent,
  commutation: nonCommuting,
};

// The default registry for the primitive ecology. Consumers who add
// their own operators should call makeInterferenceRegistry with their
// extensions concatenated onto these entries.
export const PRIMITIVE_INTERFERENCE: InterferenceRegistry = makeInterferenceRegistry([
  ["normalize.whitespace", NORMALIZE_WHITESPACE],
  ["regex.emit.url", REGEX_EMIT_URL],
  ["regex.emit.date", REGEX_EMIT_DATE],
  ["regex.emit.title", REGEX_EMIT_TITLE],
  ["row.assemble.proximity", ROW_ASSEMBLE_PROXIMITY],
  ["row.enforce.schema", ROW_ENFORCE_SCHEMA],
]);

// Re-export the building blocks so a consumer building their own
// registry doesn't need a second import.
export {
  accumulator,
  commutesWithPeers,
  idempotent,
  nonCommuting,
  nonIdempotent,
  purelyAdditive,
  replacing,
};
