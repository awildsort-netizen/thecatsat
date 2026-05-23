// Operator declarations driven by the run function's type.
//
// Principle, in three sentences:
//
//   1. The operator's run function — its parameter type and its return
//      type — IS the operator's signature. TypeScript already encodes
//      "required vs. optional input" via the `?` property modifier and
//      "output channels" via the return type. There should be no
//      parallel hand-authored ontology.
//
//   2. The solver runs at runtime, and TypeScript types are erased at
//      runtime. So beam-search composition cannot consult the run
//      function's *type* directly to ask "what channels does this
//      operator require?" — that type is gone. Some value-level
//      residue is unavoidable.
//
//   3. Minimise that residue, bind it to the function type so it can't
//      diverge (the type system rejects channel names that aren't
//      drawn from the function's signature), and derive the solver's
//      view from it on demand via `signatureOf(op)` rather than
//      storing a separate authoritative `op.io` field.
//
// Caller pattern (most TypeScript-native form):
//
//   export const regexEmitDate: ParseOperator<{
//     "text.normalized": string;
//     "spans.url":       readonly FieldHypothesis[];
//     "trace.regions"?: readonly TraceRegion[];
//   }, {
//     "spans.dated":   readonly FieldHypothesis[];
//     "trace.regions": readonly TraceRegion[];
//   }> = {
//     id: "regex.emit.date", cost: 2, tokens: [...],
//     run: (_ctx, input) => { ...input is typed by I, return by O... },
//     channels: {
//       requiredInputs: ["text.normalized", "spans.url"],
//       optionalInputs: ["trace.regions"],
//       outputs:        ["spans.dated", "trace.regions"],
//     },
//   };
//
// `I` and `O` are written once, at the value's type annotation. The
// `run` body's input parameter and return type flow from `I` and `O`
// by TypeScript's normal contextual typing. The `channels` arrays are
// typed by `ChannelsOf<I, O>`, so the type system rejects any channel
// name not present in `I` or `O`. There is no separate `defineOperator`
// generic to infer through — the value's own type annotation does it.

import type { ParseContext } from "./types.js";

// ---------------------------------------------------------------------------
// Type-level utilities.
// ---------------------------------------------------------------------------

export type RequiredKeys<T> = { [K in keyof T]-?: undefined extends T[K] ? never : K }[keyof T];
export type OptionalKeys<T> = { [K in keyof T]-?: undefined extends T[K] ? K : never }[keyof T];

// An operator's run function.
export type OperatorRun<I, O> = (ctx: ParseContext, input: I) => O;

// `ChannelsOf<I, O>` is the projection of `I` and `O` into channel-name
// arrays. The type system enforces that every name in these arrays is
// a key of `I` (for requiredInputs/optionalInputs) or `O` (for
// outputs). Foreign channel names are a type error.
export type ChannelsOf<I, O> = {
  readonly requiredInputs: readonly Extract<RequiredKeys<I>, string>[];
  readonly optionalInputs: readonly Extract<OptionalKeys<I>, string>[];
  readonly outputs:        readonly Extract<keyof O, string>[];
};

// ---------------------------------------------------------------------------
// ParseOperator.
// ---------------------------------------------------------------------------

// The defaults `any, any` make `ParseOperator` (with no type args)
// accept operators with any specific IO. The generic parameters are
// invariant via `channels`'s mapped type, so `unknown` defaults would
// collapse `RequiredKeys<unknown>` to `never` and reject specific
// operators. `any` is the right TypeScript-ism here.
export type ParseOperator<I = any, O = any> = {
  readonly id: string;
  readonly cost: number;
  readonly tokens: readonly string[];
  readonly channels: ChannelsOf<I, O>;
  readonly run: OperatorRun<I, O>;
};

// `signatureOf(op)` is the solver's view — a derivation, not stored
// state. The solver calls it whenever it needs to ask about an
// operator's IO; there is no `op.io` field to keep in sync.
export type OperatorSignatureView = {
  readonly requiredInputs: readonly string[];
  readonly optionalInputs: readonly string[];
  readonly outputs: readonly string[];
  readonly tokens: readonly string[];
};

export const signatureOf = (op: ParseOperator): OperatorSignatureView => ({
  requiredInputs: op.channels.requiredInputs,
  optionalInputs: op.channels.optionalInputs,
  outputs: op.channels.outputs,
  tokens: op.tokens,
});

// ---------------------------------------------------------------------------
// What about `defineOperator` / runtime checks?
//
// In this design the caller annotates the value with `ParseOperator<I, O>`
// and TypeScript propagates contextual types into the run body and the
// channels arrays. A `defineOperator` factory function would only
// re-package what is already a fully-typed object literal, so it's
// pure overhead. We skip it.
//
// Exhaustiveness of `channels.requiredInputs` (must list every required
// key of `I`) is not compactly expressible in TypeScript without
// codegen. In practice, the run body's typed access to `input[K]`
// makes a missing key noticeable at the call site: if the caller
// forgets to list "spans.url" as required, the solver may try to
// schedule the operator before spans.url is in scope, and the run
// body's typed `input["spans.url"]: readonly FieldHypothesis[]` would
// receive undefined at runtime — a developer error that tests catch.
// This is the documented TS-erasure boundary; see signatures_first.md.
