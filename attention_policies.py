#!/usr/bin/env python3
"""Attention resolution as local operators.

The streamable-gene decoder currently treats a second ``A:`` before any
literal as "latest wins": the new hint silently overrides the pending one.
That is one resolution policy among several. Rather than freeze it as
hidden decoder behavior, this module exposes attention resolution as a
small family of pure functions that operate on the ordered run of
``A:``-tokens that precedes a literal.

Each policy takes the queued hints (in arrival order) and returns the
resolved attention payload to attach to the next emitted literal. The
payload can be a single string (compatible with today's GeneToken), a
tuple of strings (stack), or None.

This keeps the decoder thin and lets planners pick a policy locally,
per-window, or per-experiment without baking one rule into the spine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

AttentionPayload = str | tuple[str, ...] | None
AttentionPolicy = Callable[[Sequence[str]], AttentionPayload]


@dataclass(frozen=True)
class AttentionResolution:
    """Diagnostic record of how a policy handled a queue of hints."""

    policy: str
    queued: tuple[str, ...]
    resolved: AttentionPayload
    dropped: tuple[str, ...]


def latest_wins(queued: Sequence[str]) -> AttentionPayload:
    """Current decoder behavior: keep the last hint, drop the rest silently."""

    if not queued:
        return None
    return queued[-1]


def first_wins(queued: Sequence[str]) -> AttentionPayload:
    """Keep the first hint; later hints are dropped."""

    if not queued:
        return None
    return queued[0]


def accumulate(queued: Sequence[str]) -> AttentionPayload:
    """Carry every hint forward as an ordered tuple."""

    if not queued:
        return None
    return tuple(queued)


def stack_unique(queued: Sequence[str]) -> AttentionPayload:
    """Carry hints forward, dedup'd, order-preserving."""

    if not queued:
        return None
    seen: dict[str, None] = {}
    for hint in queued:
        seen.setdefault(hint, None)
    return tuple(seen)


def strict(queued: Sequence[str]) -> AttentionPayload:
    """Refuse to silently drop hints: raise if more than one is queued."""

    if not queued:
        return None
    if len(queued) > 1:
        raise ValueError(
            f"strict attention policy refuses to drop hints: queued={list(queued)}"
        )
    return queued[0]


POLICIES: dict[str, AttentionPolicy] = {
    "latest_wins": latest_wins,
    "first_wins": first_wins,
    "accumulate": accumulate,
    "stack_unique": stack_unique,
    "strict": strict,
}


def resolve(policy_name: str, queued: Sequence[str]) -> AttentionResolution:
    """Apply a named policy and return a diagnostic record."""

    policy = POLICIES[policy_name]
    queued_tuple = tuple(queued)
    resolved = policy(queued_tuple)
    if resolved is None:
        kept: tuple[str, ...] = ()
    elif isinstance(resolved, str):
        kept = (resolved,)
    else:
        kept = tuple(resolved)
    dropped = tuple(hint for hint in queued_tuple if hint not in kept)
    return AttentionResolution(
        policy=policy_name,
        queued=queued_tuple,
        resolved=resolved,
        dropped=dropped,
    )


def replay_attention_queues(tokens: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Reconstruct, per emitted literal, the run of ``A:`` hints that preceded it.

    Walks a raw token stream once and records, for each literal that gets
    emitted, the ordered list of ``A:`` hints that arrived since the
    previous literal (or since the start of the stream).

    Does not consult the decoder; this is the upstream view that
    attention policies operate on.
    """

    queues: list[tuple[str, ...]] = []
    pending: list[str] = []
    for token in tokens:
        head, _, rest = token.partition(":")
        if head == "A":
            pending.append(rest)
        elif head == "L":
            queues.append(tuple(pending))
            pending.clear()
        elif head == "M":
            # Motif expansion emits one or more literals; in the current
            # decoder, only the first inherits the pending hint, the rest
            # see it cleared. We mirror that for the first literal only.
            queues.append(tuple(pending))
            pending.clear()
        elif head == "E":
            break
        # W, R, D do not consume attention and do not emit literals.
    return tuple(queues)
