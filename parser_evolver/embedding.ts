// Operator embedding space.
//
// Operators advertise tokens drawn from their name, comments, file path, and
// stated purpose. Two operators that share many tokens are doing similar
// kinds of work — that's the cheap polymorphism the solver can lean on
// instead of `if op.id === "x"` branches.
//
// This is symbolic, not learned: cosine-shaped Jaccard over token bags. A
// vector backend can drop in later behind the same `similarity` signature.

import type { ParseOperator } from "./types.js";

export type Embedding = ReadonlyMap<string, number>;

const bag = (tokens: readonly string[]): Embedding =>
  tokens.reduce<Map<string, number>>((acc, t) => {
    acc.set(t, (acc.get(t) ?? 0) + 1);
    return acc;
  }, new Map());

export const embed = (op: ParseOperator): Embedding => bag(op.tokens);

const dot = (a: Embedding, b: Embedding): number =>
  Array.from(a).reduce((s, [k, v]) => s + v * (b.get(k) ?? 0), 0);

const norm = (a: Embedding): number =>
  Math.sqrt(Array.from(a.values()).reduce((s, v) => s + v * v, 0));

// Cosine over token bags. Returns 0 when either side is empty.
export const similarity = (a: Embedding, b: Embedding): number => {
  const na = norm(a);
  const nb = norm(b);
  return na && nb ? dot(a, b) / (na * nb) : 0;
};

// Score how well an operator "fits" a need-token by token overlap. Used by
// the solver to pick relatives of a needed operator when the exact match
// isn't in the gene's eligibility set.
export const fit = (op: ParseOperator, queryTokens: readonly string[]): number =>
  similarity(embed(op), bag(queryTokens));
