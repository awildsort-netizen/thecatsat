// Signature-derived operator construction.
//
// Design principle: function signatures are the source of truth for what
// an operator needs and provides. The string arrays in `OperatorSignature`
// are a *projection* of that signature — useful for the beam solver,
// for embedding search, and as bytecode-readable metadata — but they
// should not be hand-authored alongside the implementation. When they
// are, they drift: a `run` function starts reading a new bag key, or
// writing one, and the declared `needs`/`provides` lags behind.
//
// `defineOperator` is the small helper that closes that loop. The caller
// declares the operator's IO as typed channel maps — `needs` (required
// upstream channels), `reads` (optional read-through channels), and
// `outputs` — and supplies a `run` whose body operates on those typed
// channels. The signature's `needs` and `provides` are derived from
// `Object.keys(needs)` / `Object.keys(outputs)` at construction time
// (cached on the returned operator), so they cannot diverge from the
// declared IO of the run body.
//
// TypeScript's runtime reflection is limited (no decorators-as-data, no
// `typeof T` at runtime), so we use a pragmatic shape: the *keys* of the
// IO specs are the channel names and the *types* at those keys are the
// channel value shapes. The compiler enforces that the run body's input
// argument is keyed exactly by `keyof inputs`, and that its return value
// is keyed exactly by `keyof outputs`. A drift between implementation
// and declared signature now fails typecheck rather than fails silently.

import type { OperatorSignature, ParseContext, ParseOperator } from "./types.js";

// A channel spec is a record whose KEYS are the channel names and whose
// values are the (compile-time only) shape of bytes flowing through that
// channel. The runtime values are unused — we only use the keys to
// derive the signature.
export type ChannelSpec = Readonly<Record<string, unknown>>;

// Typed bag shapes derived from input specs.
//
// `needs` are channels the solver must see produced before this operator
// is eligible — they map to `OperatorSignature.needs`.
//
// `reads` are channels the run body may *read* but which the operator
// does NOT depend on for eligibility (e.g. a read-through accumulator
// channel an operator also writes back). These are optional in the
// type bag and are intentionally *not* projected into `signature.needs`.
//
// Splitting these two prevents a subtle drift: a hand-authored signature
// that omits a read-through accumulator passes typecheck but is silently
// inaccurate. Forcing the call site to list the channel here keeps the
// implementation honest while still letting the solver gate eligibility
// on real needs.
export type InputBag<N extends ChannelSpec, R extends ChannelSpec> = Readonly<
  { [K in keyof N]: N[K] } & Partial<{ [K in keyof R]: R[K] }>
>;
export type OutputBag<O extends ChannelSpec> = Readonly<{ [K in keyof O]: O[K] }>;

export type DefineOperatorArgs<
  N extends ChannelSpec,
  R extends ChannelSpec,
  O extends ChannelSpec,
> = {
  readonly id: string;
  readonly cost: number;
  // Tokens still live with the operator — they are the symbolic
  // embedding signal and have no implementation analogue to reflect
  // off. Co-locating them here keeps `defineOperator` a single
  // source of truth for everything that lands in the signature.
  readonly tokens: readonly string[];
  // Required upstream channels: gate solver eligibility and are
  // projected into `signature.needs`.
  readonly needs: N;
  // Optional read-through channels: typed for the run body but not
  // projected into `signature.needs`. Defaults to {} when omitted.
  readonly reads?: R;
  readonly outputs: O;
  readonly run: (ctx: ParseContext, input: InputBag<N, R>) => OutputBag<O>;
};

// Sentinel value attached to a channel-spec key. The spec object itself
// is never read at runtime — only `Object.keys`. We document that with
// `CHANNEL` so a caller doesn't accidentally pass a meaningful value.
export const CHANNEL = Object.freeze({}) as never;

// Derive a signature from input/output channel keys. Exposed so other
// modules (the browser_oracle proposal lifter, for example) can produce
// the same projection without re-implementing the rule.
export const deriveSignature = (
  needs: ChannelSpec,
  outputs: ChannelSpec,
  tokens: readonly string[],
): OperatorSignature => ({
  needs: Object.freeze(Object.keys(needs)),
  provides: Object.freeze(Object.keys(outputs)),
  tokens,
});

// Build a ParseOperator whose `signature` is *derived* from its IO
// declaration. The signature is computed once at construction and frozen
// onto the operator; it cannot drift because the run body's TypeScript
// type forces the same keys.
export const defineOperator = <
  N extends ChannelSpec,
  R extends ChannelSpec,
  O extends ChannelSpec,
>(
  args: DefineOperatorArgs<N, R, O>,
): ParseOperator => {
  const signature = deriveSignature(args.needs, args.outputs, args.tokens);
  // The bag the solver passes in is untyped (`unknown`); we narrow it
  // to the declared shape at the boundary. The narrowing is a single
  // cast at the edge — internal code stays typed.
  const run: ParseOperator["run"] = (ctx, input) =>
    args.run(ctx, input as InputBag<N, R>) as unknown;
  return { id: args.id, cost: args.cost, signature, run };
};
