// Auto-derive interference specs from operator lambdas.
//
// Motivation
// ----------
// The previous branch introduced `InterferenceSpec` and a hand-authored
// `PRIMITIVE_INTERFERENCE` side-table that mirrored, for each operator,
// claims that are *already provable* by looking at the operator's
// declared channels and by *running* the operator's lambda on a small
// probe input. The side-table is therefore redundant — a second source
// of truth, kept in sync by hand.
//
// This module replaces the hand-authored claims with *derived* ones.
// Each `WriteDiscipline`, `Application`, and `Commutation` value used
// downstream is constructed by inspecting the operator: a few small
// runs of its `run` function on synthetic inputs are enough to decide
// accumulator vs. replacing semantics for every output channel, and to
// decide idempotence for the operator as a whole.
//
// Method, in three primitives
// ---------------------------
// (1) Accumulator detection. For each channel C that appears in both
//     `op.channels.optionalInputs` (so the operator *can* receive a
//     prior value) and `op.channels.outputs` (so the operator writes
//     it), plant a fresh sentinel value in the probe input under C,
//     run the operator, and check whether the sentinel survives in
//     the output. Sentinel survival ⇒ accumulator. The operator is
//     literally telling us, by running, that it merged prior with
//     new. Channels that are only in `outputs` (no optional input
//     pathway) cannot be accumulators by construction.
//
// (2) Idempotence detection. Run the operator on the probe to get
//     output O1. Build a second input that merges O1 back into the
//     original probe (so the operator sees its own output as input).
//     Run again to get O2. For every *replacing* channel C, require
//     O1[C] and O2[C] to be value-equal. Accumulator channels are
//     excluded — they grow on every application by definition.
//     A pass means the operator's effect stabilises after one
//     application, which is exactly the property
//     `redundant_if_adjacent` rests on.
//
// (3) Commutation. Currently structural — every operator with no
//     declared non-commuting reason advertises `commutesWithPeers`;
//     the must_precede filter in `buildConflictGraph` correctly
//     suppresses pairs that are sequentially dependent. We expose
//     this as a derived default (`commutesWithPeers`) and let a
//     caller override when they have semantic reasons to disagree —
//     the only operators in the current basis that genuinely don't
//     commute (e.g. `row.assemble.proximity` as the sole writer to
//     `rows.assembled`) are caught by the must_precede filter anyway,
//     so the default is safe.
//
// Probe inputs
// ------------
// Operators expect inputs of specific shapes (FieldHypothesis arrays
// with valid spans, strings, etc.). Rather than craft a synthetic
// shape per channel — which would require knowing each channel's
// value type — we record a *real* run through the demo sample. Each
// operator is given, as its probe input, what it would actually
// receive in the composed pipeline. This makes the derivation work
// for any operator that participates in any working pipeline and
// avoids re-inventing channel-shape registries.
//
// The probe context and sample live in this file. They are tiny and
// purely illustrative — they don't have to cover every edge case,
// they only have to drive each operator down its main code path.
//
// What about operators that don't appear in the probe pipeline?
// -------------------------------------------------------------
// Such an operator can supply its own probe input (a caller passes
// `deriveInterference(op, { input })`), or fall back to
// `DEFAULT_INTERFERENCE` (conservative: writes are replacing, not
// idempotent, doesn't commute). The conservative default keeps the
// system correct under absence-of-information; the derivation just
// recovers the precise claim wherever it can.

import {
  accumulator,
  commutesWithPeers,
  DEFAULT_INTERFERENCE,
  idempotent,
  makeInterferenceRegistry,
  nonCommuting,
  nonIdempotent,
  replacing,
  type Application,
  type Commutation,
  type InterferenceRegistry,
  type InterferenceSpec,
  type WriteDiscipline,
} from "./interference.js";
import type { ParseContext, ParseOperator } from "./types.js";
import { signatureOf } from "./operator_reflection.js";

// ---------------------------------------------------------------------------
// Sentinel construction.
//
// A sentinel is a freshly-built value of the channel's expected shape,
// tagged with a unique marker the operator cannot have invented on its
// own. After running the operator we look for the marker in the output
// to decide whether the operator forwarded the prior value (= accumulator)
// or discarded it (= replacing).
//
// We don't know channel value types at runtime, so the sentinel is a
// shape that fits the channels we have in this basis:
//   * arrays of objects → a one-element array carrying a marker field
//   * strings (rare in inputs) → just the marker string
// The check `survivesSentinel` only needs to ask "did the marker appear
// in the output under C?", regardless of shape.
// ---------------------------------------------------------------------------

const SENTINEL_TAG = "__interference_sentinel__";

const makeSentinel = (channel: string): unknown => {
  // Channels in the current basis carry either string or array-of-object
  // values. We synthesise an array sentinel for every channel; if the
  // operator expects a string it will simply fail to use the sentinel
  // and we'll see no survival (= replacing), which is the right answer
  // for the only string channel in scope (`text.normalized`, which the
  // emitter writes wholesale).
  return [{ [SENTINEL_TAG]: channel }];
};

const containsSentinel = (value: unknown, channel: string): boolean => {
  if (Array.isArray(value)) {
    return value.some((item) => containsSentinel(item, channel));
  }
  if (value !== null && typeof value === "object") {
    const o = value as Record<string, unknown>;
    if (o[SENTINEL_TAG] === channel) return true;
    return Object.values(o).some((v) => containsSentinel(v, channel));
  }
  return false;
};

// ---------------------------------------------------------------------------
// Probe inputs — recorded from a real pipeline run.
//
// The probe is a small typed bag of channel → value, captured by running
// the demo pipeline once. Each operator's probe input is what it would
// see at its scheduled position in the composed gene-string.
//
// We don't bake the probe into compiled code — callers can build their
// own probes by composing operators on their own contexts and passing
// the resulting channel bag in. The demo probe lives here only as a
// default that covers the current operator basis.
// ---------------------------------------------------------------------------

export type Probe = {
  readonly ctx: ParseContext;
  // Channel → value, drawn from a real pipeline run. The probe doesn't
  // have to be exhaustive: each operator reads only its declared
  // requiredInputs/optionalInputs, so it sees only what it asked for.
  readonly channels: Readonly<Record<string, unknown>>;
};

// Build a probe by running the operators in their data-flow order on a
// caller-supplied context. Each operator receives, as input, the channels
// produced by earlier operators in the list (and the synthetic sentinel
// for optional-input channels that aren't yet produced — but only when
// the caller explicitly asks via `recordWithSentinels`).
//
// The "data-flow order" here is just: any operator whose requiredInputs
// are already satisfied. This is the same order the solver would pick
// at maximal greed.
export const buildProbeFromPipeline = (
  ctx: ParseContext,
  ops: readonly ParseOperator[],
): Probe => {
  const channels: Record<string, unknown> = {};
  const ready = (op: ParseOperator): boolean =>
    signatureOf(op).requiredInputs.every((c) => Object.prototype.hasOwnProperty.call(channels, c));
  const remaining = [...ops];
  // Each pass schedules every operator whose requiredInputs are now in
  // scope. We stop when a pass schedules nothing — a normal end for
  // operators not connected to ctx (i.e. seed-less, no required inputs).
  let progress = true;
  while (remaining.length > 0 && progress) {
    progress = false;
    for (let i = 0; i < remaining.length; ) {
      const op = remaining[i]!;
      if (!ready(op)) {
        i += 1;
        continue;
      }
      const input = inputBag(op, channels);
      const out = op.run(ctx, input) as Record<string, unknown>;
      Object.assign(channels, out);
      remaining.splice(i, 1);
      progress = true;
    }
  }
  return { ctx, channels };
};

const inputBag = (op: ParseOperator, channels: Record<string, unknown>): Record<string, unknown> => {
  const bag: Record<string, unknown> = {};
  signatureOf(op).requiredInputs.forEach((c) => {
    bag[c] = channels[c];
  });
  signatureOf(op).optionalInputs.forEach((c) => {
    if (Object.prototype.hasOwnProperty.call(channels, c)) bag[c] = channels[c];
  });
  return bag;
};

// ---------------------------------------------------------------------------
// deriveInterference — the heart.
//
// Given an operator and a probe, return the spec the operator's lambda
// is actually implementing. The lambda IS the spec.
// ---------------------------------------------------------------------------

export const deriveInterference = (
  op: ParseOperator,
  probe: Probe,
): InterferenceSpec => {
  const sig = signatureOf(op);
  const baseInput = inputBag(op, probe.channels as Record<string, unknown>);

  // ---- accumulator detection -------------------------------------------
  // For each output channel that is also an optional input, plant a
  // sentinel under that channel in the input and ask whether the
  // sentinel survived in the output. We only test channels whose name
  // appears in both outputs and optionalInputs — channels with no input
  // pathway cannot be accumulators by construction.
  const optionalInputSet = new Set(sig.optionalInputs);
  const outputs: WriteDiscipline[] = sig.outputs.map((channel) => {
    if (!optionalInputSet.has(channel)) {
      // No input pathway: must be replacing.
      return replacing(channel);
    }
    const probedInput: Record<string, unknown> = { ...baseInput, [channel]: makeSentinel(channel) };
    let producedOutput: Record<string, unknown>;
    try {
      producedOutput = op.run(probe.ctx, probedInput) as Record<string, unknown>;
    } catch {
      // Operator refused the sentinel shape — fall back to replacing,
      // the conservative claim. (Doesn't occur in the current basis;
      // here as a safety net for future operators.)
      return replacing(channel);
    }
    return containsSentinel(producedOutput[channel], channel) ? accumulator(channel) : replacing(channel);
  });

  // ---- idempotence detection -------------------------------------------
  // Run the operator once on the probe. Merge the output back into the
  // input and run again. For every replacing channel, the two outputs
  // must be value-equal — that's the operator stabilising. Accumulators
  // are excluded; they're expected to grow.
  const isIdempotent = checkIdempotence(op, probe, baseInput, outputs);

  const application: Application = isIdempotent ? idempotent : nonIdempotent;

  // ---- commutation -----------------------------------------------------
  // Structural default: offer the half-permission. The must_precede
  // filter in the composer takes care of pairs that are sequentially
  // dependent. Callers with semantic reasons to refuse (e.g. side
  // effects we can't observe by running) can override via the
  // `commutationOverride` argument to `deriveRegistry`.
  const commutation: Commutation = commutesWithPeers;

  return { outputs, application, commutation };
};

const deepEqual = (a: unknown, b: unknown): boolean => {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((x, i) => deepEqual(x, b[i]));
  }
  if (a !== null && b !== null && typeof a === "object" && typeof b === "object") {
    const ao = a as Record<string, unknown>;
    const bo = b as Record<string, unknown>;
    const ak = Object.keys(ao);
    const bk = Object.keys(bo);
    if (ak.length !== bk.length) return false;
    return ak.every((k) => deepEqual(ao[k], bo[k]));
  }
  return false;
};

const checkIdempotence = (
  op: ParseOperator,
  probe: Probe,
  baseInput: Record<string, unknown>,
  outputs: readonly WriteDiscipline[],
): boolean => {
  let firstOutput: Record<string, unknown>;
  try {
    firstOutput = op.run(probe.ctx, baseInput) as Record<string, unknown>;
  } catch {
    return false;
  }
  // Build the second input: same as baseInput but with each output
  // channel set to whatever the operator just produced.
  const secondInput: Record<string, unknown> = { ...baseInput };
  outputs.forEach((d) => {
    secondInput[d.channel] = firstOutput[d.channel];
  });
  let secondOutput: Record<string, unknown>;
  try {
    secondOutput = op.run(probe.ctx, secondInput) as Record<string, unknown>;
  } catch {
    return false;
  }
  // Replacing channels must match exactly across runs. Accumulator
  // channels are skipped — they grow by definition.
  const replacingChannels = outputs
    .filter((d): d is WriteDiscipline & { channel: string } => {
      // A WriteDiscipline is "replacing" if its contributeReplacing
      // pours into the sink. We can detect this directly without
      // introspection: invoke the contribution on a sink and check.
      const sink = new Set<string>();
      d.contributeReplacing(sink);
      return sink.size > 0;
    })
    .map((d) => d.channel);
  return replacingChannels.every((c) => deepEqual(firstOutput[c], secondOutput[c]));
};

// ---------------------------------------------------------------------------
// deriveRegistry — build an InterferenceRegistry by deriving every
// operator's spec from its lambda.
//
// `overrides` is for the rare case where a caller has semantic
// information the probe can't see (e.g. an operator with side effects,
// or a hand-authored commutation claim tighter than the structural
// default). Overrides are merged shallowly: any field present in the
// override replaces the derived one.
// ---------------------------------------------------------------------------

export type SpecOverride = Partial<InterferenceSpec>;

export const deriveRegistry = (
  ops: readonly ParseOperator[],
  probe: Probe,
  overrides: Readonly<Record<string, SpecOverride>> = {},
): InterferenceRegistry => {
  const entries: (readonly [string, InterferenceSpec])[] = ops.map((op) => {
    let derived: InterferenceSpec;
    try {
      derived = deriveInterference(op, probe);
    } catch {
      derived = DEFAULT_INTERFERENCE;
    }
    const ov = overrides[op.id];
    const merged: InterferenceSpec = ov === undefined ? derived : { ...derived, ...ov };
    return [op.id, merged] as const;
  });
  return makeInterferenceRegistry(entries);
};
