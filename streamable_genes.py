#!/usr/bin/env python3
"""Streamable gene-string spine.

A gene stream is a prefix-meaningful sequence of tokens that unfolds an
ecological event. Decoding is incremental: every consumed token advances a
small decoder state, and the state can be inspected mid-stream to expose
currently composable operator names, the active type window, and the motif
dictionary built so far.

Token grammar (one token per stream entry, ``:`` separated):

    L:<name>            literal operator/target token
    M:<i>               backreference to motif dictionary slot i
    D:<i>:<a>,<b>,...   define motif i as the listed sub-tokens
    W:<label>           open a type window (local climate)
    R                   reset / close the top type window
    A:<name>            attention-inheritance hint carried to next literal
    E                   end-of-stream

Motif expansion happens at decode time. ``D`` only registers; ``M`` unfolds.
A motif body may itself contain motif refs as long as they were defined
earlier in the stream (no forward refs, keeps it streamable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass(frozen=True)
class GeneToken:
    """A single decoded gene token after motif expansion.

    ``window`` is the label of the type window active when this token was
    emitted (``None`` if no window is open). ``attention`` carries an
    inheritance hint from the most recent ``A:`` token, consumed on use.
    """

    name: str
    window: str | None = None
    attention: str | None = None


@dataclass
class StreamState:
    """Mutable decoder state. Inspectable between any two ``feed`` calls."""

    emitted: list[GeneToken] = field(default_factory=list)
    motifs: dict[int, tuple[str, ...]] = field(default_factory=dict)
    window_stack: list[str] = field(default_factory=list)
    pending_attention: str | None = None
    ended: bool = False

    @property
    def window(self) -> str | None:
        return self.window_stack[-1] if self.window_stack else None

    def composable_now(self) -> tuple[str, ...]:
        """Operator names visible to a composer right now.

        Order is preserved, duplicates removed, ``E`` does not appear. This
        is the partial-decode hint the planner can read before EOF.
        """

        seen: dict[str, None] = {}
        for token in self.emitted:
            seen.setdefault(token.name, None)
        return tuple(seen)


class StreamableGenome:
    """Incremental decoder for a gene-token stream.

    Feed tokens one at a time (or in batches). After any prefix, inspect
    ``state.emitted``, ``state.window``, ``state.motifs``, and
    ``composable_now()``.
    """

    def __init__(self) -> None:
        self.state = StreamState()

    def feed(self, token: str) -> None:
        if self.state.ended:
            raise ValueError("stream already ended")
        head, _, rest = token.partition(":")
        if head == "L":
            self._emit(rest)
        elif head == "M":
            self._expand_motif(int(rest))
        elif head == "D":
            slot_str, _, body = rest.partition(":")
            self._define_motif(int(slot_str), body)
        elif head == "W":
            self.state.window_stack.append(rest)
        elif head == "R":
            if self.state.window_stack:
                self.state.window_stack.pop()
        elif head == "A":
            self.state.pending_attention = rest
        elif head == "E":
            self.state.ended = True
        else:
            raise ValueError(f"unknown gene token: {token!r}")

    def feed_all(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.feed(token)

    def _emit(self, name: str) -> None:
        attention = self.state.pending_attention
        self.state.pending_attention = None
        self.state.emitted.append(
            GeneToken(name=name, window=self.state.window, attention=attention)
        )

    def _expand_motif(self, slot: int) -> None:
        body = self.state.motifs.get(slot)
        if body is None:
            raise KeyError(f"motif {slot} not defined yet")
        for name in body:
            self._emit(name)

    def _define_motif(self, slot: int, body: str) -> None:
        names = tuple(part for part in body.split(",") if part)
        self.state.motifs[slot] = names


def stream(tokens: Iterable[str]) -> StreamState:
    """Decode an entire token iterable and return the final state."""

    genome = StreamableGenome()
    genome.feed_all(tokens)
    return genome.state


def iter_partial_states(tokens: Iterable[str]) -> Iterator[StreamState]:
    """Yield the decoder state after each consumed token.

    Useful for tests and for planners that want to observe how composable
    pathways grow as the stream unfolds.
    """

    genome = StreamableGenome()
    for token in tokens:
        genome.feed(token)
        yield genome.state


def pathway_hint(state: StreamState, known_operators: Iterable[str]) -> tuple[str, ...]:
    """Project visible names onto a composer's operator vocabulary.

    Returns the subset of currently composable names that the composer
    already knows about, preserving stream order. This is the smallest
    bridge between gene streams and the existing composer without forcing
    a global migration.
    """

    known = set(known_operators)
    return tuple(name for name in state.composable_now() if name in known)
