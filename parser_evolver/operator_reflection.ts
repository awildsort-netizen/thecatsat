// Signature-derived operator construction.
//
// Design principle: actively seek alignment with the host language.
// TypeScript already has a vocabulary for "required vs. optional
// property" — the `?` modifier — and a vocabulary for "what channels
// does this function consume / produce" — the function's parameter and
// return types. Inventing a parallel ontology (`needs` / `reads` /
// `provides` as separate hand-authored string arrays) duplicates what
// the type system already encodes, and any duplication is a place for
// drift. Syntax sugar is not just sugar: it collapses entire classes of
// redundant manual ontology.
//
// So `defineOperator` takes a single `inputs` channel spec and a single
// `outputs` channel spec. Every key in `inputs` is a channel the run
// body may read; every key in `outputs` is a channel the run body must
// produce. Within `inputs`, channels marked with the `optional<T>()`
// helper become `?`-properties on the typed input bag — they are
// readable but not required, and they are NOT projected into the
// eligibility set the solver uses to schedule the operator. Channels
// marked with `required<T>()` are required reads and DO gate
// eligibility. The `?` on the run body's input parameter and the
// signature projection are the same `?` — there is no second ontology.
//
// The `signature: { needs, provides, tokens }` object on `ParseOperator`
// still exists, but it is a *legacy projection* of this reflected
// shape, computed by `toLegacySignature()` for backward compatibility
// with the existing solver, embedding layer, and bytecode disassembler.
// It is no longer the source of truth; the typed channel specs are.

import type { OperatorSignature, ParseContext, ParseOperator } from "./types.js";

// ---------------------------------------------------------------------------
// Channel-spec primitives.
//
// A channel spec is a record whose KEYS are channel names. The VALUE at
// each key is a typed marker: either `required<T>()` (must be in scope
// before the operator runs) or `optional<T>()` (may be in scope; the
// run body reads it as `T | undefined`). The marker's runtime form is
// what lets `defineOperator` partition required vs. optional keys
// without a second hand-authored array.
// ---------------------------------------------------------------------------

declare const OPTIONAL_TAG: unique symbol;

// Brand carried only at the type level. The runtime form is a plain
// object with an OPTIONAL_TAG property; nothing else reads the brand.
export type Optional<T> = { readonly [OPTIONAL_TAG]: true; readonly __t?: T };

// Helpers the caller uses to declare each channel. `required<T>()`
// returns a value typed as `T`; `optional<T>()` returns a value typed as
// `Optional<T>`. Both are write-only sentinels — the runtime values are
// only used so `Object.keys` and a tag check can partition them.
export const required = <T>(): T => CHANNEL as unknown as T;
export const optional = <T>(): Optional<T> => OPTIONAL_SENTINEL as unknown as Optional<T>;

// Sentinel value attached to a channel-spec key. The spec object itself
// is never read at runtime — only `Object.keys`. `CHANNEL` documents
// that a caller shouldn't pass a meaningful value.
export const CHANNEL = Object.freeze({}) as never;
const OPTIONAL_MARK: unique symbol = Symbol.for("parser_evolver.optional");
const OPTIONAL_SENTINEL = Object.freeze({ [OPTIONAL_MARK]: true });

const isOptionalMarker = (v: unknown): boolean =>
  typeof v === "object" && v !== null && (v as Record<symbol, unknown>)[OPTIONAL_MARK] === true;

// ---------------------------------------------------------------------------
// Type-level projection: turn an inputs spec into the typed bag the run
// body receives. Keys whose value type extends `Optional<U>` become
// `?`-properties of type `U`; all other keys are required properties.
//
// This is the load-bearing alignment: the `?` the caller sees on
// `input.someChannel` is the same `?` TypeScript uses for any optional
// property, because that's exactly what it is.
// ---------------------------------------------------------------------------

export type ChannelSpec = Readonly<Record<string, unknown>>;

type RequiredKeys<I> = { [K in keyof I]: I[K] extends Optional<unknown> ? never : K }[keyof I];
type OptionalKeys<I> = { [K in keyof I]: I[K] extends Optional<unknown> ? K : never }[keyof I];

export type InputBag<I extends ChannelSpec> = Readonly<
  { [K in RequiredKeys<I>]: I[K] } & {
    [K in OptionalKeys<I>]?: I[K] extends Optional<infer U> ? U : never;
  }
>;

export type OutputBag<O extends ChannelSpec> = Readonly<{ [K in keyof O]: O[K] }>;

// ---------------------------------------------------------------------------
// defineOperator: a single inputs spec, a single outputs spec. The
// caller declares optionality via the property modifier-style helper
// `optional<T>()` rather than via a second slot.
// ---------------------------------------------------------------------------

export type DefineOperatorArgs<I extends ChannelSpec, O extends ChannelSpec> = {
  readonly id: string;
  readonly cost: number;
  // Tokens still live with the operator — they are the symbolic
  // embedding signal and have no implementation analogue to reflect
  // off. Co-locating them here keeps `defineOperator` a single source
  // of truth for everything that lands in the signature.
  readonly tokens: readonly string[];
  readonly inputs: I;
  readonly outputs: O;
  readonly run: (ctx: ParseContext, input: InputBag<I>) => OutputBag<O>;
};

// Partition the inputs spec at runtime. Required keys become the
// eligibility set; the union (required ∪ optional) is the full read
// set, used only when callers ask for it explicitly via `inputKeys()`.
const partitionInputs = (inputs: ChannelSpec): { requiredInputs: readonly string[]; optionalInputs: readonly string[] } => {
  const required: string[] = [];
  const optional: string[] = [];
  for (const [k, v] of Object.entries(inputs)) {
    (isOptionalMarker(v) ? optional : required).push(k);
  }
  return { requiredInputs: Object.freeze(required), optionalInputs: Object.freeze(optional) };
};

// ---------------------------------------------------------------------------
// Reflected views.
//
// The reflected operator carries the partition explicitly so other
// modules can query "what does this operator require?" / "what does it
// read optionally?" / "what does it produce?" without re-parsing the
// inputs spec. These are projections of the spec, not authored fields.
// ---------------------------------------------------------------------------

export type ReflectedOperator = ParseOperator & {
  readonly reflected: {
    readonly requiredInputs: readonly string[];
    readonly optionalInputs: readonly string[];
    readonly outputs: readonly string[];
  };
};

// Legacy adapter: emit the `{needs, provides, tokens}` shape the
// existing solver / embedding layer / bytecode disassembler consume.
// `needs` is the required-inputs set (only required reads gate
// scheduling); `provides` is the outputs set. This is a *projection*
// for backward compatibility — not first-class vocabulary.
export const toLegacySignature = (
  requiredInputs: readonly string[],
  outputs: readonly string[],
  tokens: readonly string[],
): OperatorSignature => ({
  needs: requiredInputs,
  provides: outputs,
  tokens,
});

// Build a ParseOperator whose `signature` is *derived* from its IO
// declaration. The signature is computed once at construction and frozen
// onto the operator; it cannot drift because the run body's TypeScript
// type forces the same keys.
export const defineOperator = <I extends ChannelSpec, O extends ChannelSpec>(
  args: DefineOperatorArgs<I, O>,
): ReflectedOperator => {
  const { requiredInputs, optionalInputs } = partitionInputs(args.inputs);
  const outputs = Object.freeze(Object.keys(args.outputs));
  const signature = toLegacySignature(requiredInputs, outputs, args.tokens);
  // The bag the solver passes in is untyped (`unknown`); we narrow it
  // to the declared shape at the boundary. The narrowing is a single
  // cast at the edge — internal code stays typed.
  const run: ParseOperator["run"] = (ctx, input) =>
    args.run(ctx, input as InputBag<I>) as unknown;
  return {
    id: args.id,
    cost: args.cost,
    signature,
    run,
    reflected: { requiredInputs, optionalInputs, outputs },
  };
};
