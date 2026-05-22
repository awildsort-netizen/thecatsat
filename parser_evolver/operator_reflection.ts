// Signature-derived operator construction.
//
// One vocabulary, no parallel ontology. An operator declares a single
// `inputs` channel spec and a single `outputs` channel spec; each
// `inputs` entry is marked `required<T>()` or `optional<T>()`. The
// `?` modifier on the typed input bag is the projection — exactly the
// way TypeScript already says "this property may be present or absent",
// not a parser_evolver concept layered on top.
//
// The same declaration drives two things:
//   - the type of the run body's input parameter (mapped type:
//     `required<T>()` -> required property of type T; `optional<T>()`
//     -> `?`-property of type T),
//   - the `OperatorIO` record on the returned operator
//     (`Object.keys(inputs)` partitioned by a tag check yields
//     `requiredInputs` and `optionalInputs`; `Object.keys(outputs)` is
//     `outputs`).
//
// Drift between the declared inputs and the run body's reads is a type
// error. There is no second slot to keep in sync.

import type { OperatorIO, ParseContext, ParseOperator } from "./types.js";

// ---------------------------------------------------------------------------
// Channel-spec primitives.
// ---------------------------------------------------------------------------

declare const OPTIONAL_TAG: unique symbol;

// Brand carried only at the type level. The runtime form is a tagged
// object — the tag is what `defineOperator` reads to partition.
export type Optional<T> = { readonly [OPTIONAL_TAG]: true; readonly __t?: T };

const OPTIONAL_MARK: unique symbol = Symbol.for("parser_evolver.optional");
const REQUIRED_SENTINEL = Object.freeze({});
const OPTIONAL_SENTINEL = Object.freeze({ [OPTIONAL_MARK]: true });
const isOptionalMarker = (v: unknown): boolean =>
  typeof v === "object" && v !== null && (v as Record<symbol, unknown>)[OPTIONAL_MARK] === true;

export const required = <T>(): T => REQUIRED_SENTINEL as unknown as T;
export const optional = <T>(): Optional<T> => OPTIONAL_SENTINEL as unknown as Optional<T>;

// ---------------------------------------------------------------------------
// Type-level projection: turn an inputs spec into the typed bag the run
// body receives. Keys whose value type extends `Optional<U>` become
// `?`-properties of type `U`; all other keys are required properties.
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
// defineOperator: a single inputs spec, a single outputs spec.
// ---------------------------------------------------------------------------

export type DefineOperatorArgs<I extends ChannelSpec, O extends ChannelSpec> = {
  readonly id: string;
  readonly cost: number;
  // Tokens have no implementation analogue to reflect off — they're
  // the symbolic embedding signal — so they're co-located here so the
  // operator has a single source of truth for everything in `io`.
  readonly tokens: readonly string[];
  readonly inputs: I;
  readonly outputs: O;
  readonly run: (ctx: ParseContext, input: InputBag<I>) => OutputBag<O>;
};

const partitionInputs = (inputs: ChannelSpec): { requiredInputs: readonly string[]; optionalInputs: readonly string[] } => {
  const requiredInputs: string[] = [];
  const optionalInputs: string[] = [];
  for (const [k, v] of Object.entries(inputs)) {
    (isOptionalMarker(v) ? optionalInputs : requiredInputs).push(k);
  }
  return { requiredInputs: Object.freeze(requiredInputs), optionalInputs: Object.freeze(optionalInputs) };
};

export const defineOperator = <I extends ChannelSpec, O extends ChannelSpec>(
  args: DefineOperatorArgs<I, O>,
): ParseOperator => {
  const { requiredInputs, optionalInputs } = partitionInputs(args.inputs);
  const outputs = Object.freeze(Object.keys(args.outputs));
  const io: OperatorIO = { requiredInputs, optionalInputs, outputs, tokens: args.tokens };
  // The bag the solver passes in is untyped (`unknown`); we narrow it
  // to the declared shape at the boundary. The narrowing is a single
  // cast at the edge — internal code stays typed.
  const run: ParseOperator["run"] = (ctx, input) =>
    args.run(ctx, input as InputBag<I>) as unknown;
  return { id: args.id, cost: args.cost, io, run };
};
