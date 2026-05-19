#!/usr/bin/env python3
"""Summarize raw bytecode token streams into readable gene-string forms.

The ``bytecode_genes`` lens emits a fine-grained stream:

    B:<OPNAME>@<offset>
    CALL:<qualname>
    LINE:<n>
    RET:<qualname>

That stream is body-level metabolism — useful for tracing, but offset-
specific and noisy. The streamable gene grammar (``streamable_genes``)
already speaks in:

    L:<name>          literal composition event
    D:<id>:<body>     motif definition
    M:<id>            motif reference

This module is the bridge. It exposes three readable summaries that drop
offsets in favour of structure:

  * ``boundary_summary``: collapse a token stream into one ``L:<qualname>``
    per code-boundary contiguous run. The result is composer-grade — what
    streamable_genes consumes natively.

  * ``motif_dictionary``: find repeated length-N opname n-grams and emit
    a streamable ``D:<id>:body`` / ``M:<id>`` rewrite. The compressed
    stream is shorter and readable; the dictionary tells you what each
    slot means.

  * ``opname_distribution`` / ``call_distribution`` / ``motif_distribution``:
    plain Counters over the readable units. Distributions over motifs and
    call targets are stable across CPython opcode renames — they ride on
    structure, not raw opcode strings.

The functions are pure over token sequences. They are tested with
structural assertions (counts, set relationships, prefix order) rather
than exact opcode strings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Token parsing helpers.
# ---------------------------------------------------------------------------
def opname_of(token: str) -> str | None:
    """Return the opname portion of a ``B:<OPNAME>@<offset>`` token, or None."""
    if not token.startswith("B:"):
        return None
    body = token[2:]
    head, sep, _ = body.rpartition("@")
    if not sep:
        return None
    return head or None


def call_target_of(token: str) -> str | None:
    """Return the qualname portion of a ``CALL:<qualname>`` token, or None."""
    if not token.startswith("CALL:"):
        return None
    return token[5:] or None


# ---------------------------------------------------------------------------
# Boundary summary: group tokens by their owning qualname.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundaryRun:
    """A contiguous run of tokens attributed to one code boundary.

    ``qualname`` is the function that owned the run (taken from the trace's
    qualname stamps, not from raw offsets). ``opname_counts`` is the
    distribution over distinct opnames seen in the run, *without* offsets —
    that is the readable summary. ``calls`` lists call targets dispatched
    inside the run, in order.
    """

    qualname: str
    token_count: int
    opname_counts: Counter
    calls: tuple[str, ...]
    line_set: tuple[int, ...]


def boundary_runs_from_records(records: Iterable) -> tuple[BoundaryRun, ...]:
    """Walk a ``TraceResult.records`` stream into contiguous boundary runs.

    ``records`` is an iterable of objects with ``kind``, ``qualname``,
    ``opname``, ``lineno`` attributes (typically ``bytecode_genes.TraceRecord``).
    A new run starts whenever ``qualname`` changes — this respects nested
    calls because the trace itself emits a new qualname for every entered
    code object.
    """

    runs: list[BoundaryRun] = []
    cur_qn: str | None = None
    op_counter: Counter = Counter()
    calls: list[str] = []
    lines: list[int] = []
    token_count = 0

    def flush() -> None:
        nonlocal op_counter, calls, lines, token_count, cur_qn
        if cur_qn is None:
            return
        runs.append(
            BoundaryRun(
                qualname=cur_qn,
                token_count=token_count,
                opname_counts=op_counter,
                calls=tuple(calls),
                line_set=tuple(sorted(set(lines))),
            )
        )
        op_counter = Counter()
        calls = []
        lines = []
        token_count = 0

    for record in records:
        qn = getattr(record, "qualname", None)
        if qn is None:
            continue
        if cur_qn is None or qn != cur_qn:
            flush()
            cur_qn = qn
        token_count += 1
        kind = getattr(record, "kind", None)
        if kind == "opcode":
            opname = getattr(record, "opname", None)
            if opname:
                op_counter[opname] += 1
        elif kind == "call":
            calls.append(qn)
        elif kind == "line":
            lineno = getattr(record, "lineno", None)
            if lineno is not None:
                lines.append(int(lineno))
    flush()
    return tuple(runs)


def boundary_summary_tokens(records: Iterable) -> tuple[str, ...]:
    """Compress trace records into one ``L:<qualname>`` token per run.

    This is the readable, composer-grade form: every contiguous run of
    activity inside one code boundary becomes a single literal token.
    The result is directly consumable by ``streamable_genes.stream``.
    """

    runs = boundary_runs_from_records(records)
    tokens = [f"L:{run.qualname}" for run in runs]
    tokens.append("E")
    return tuple(tokens)


# ---------------------------------------------------------------------------
# Motif dictionary: find repeated opname n-grams.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MotifDictionary:
    """Result of motif-extraction from an opname sequence.

    ``slots`` maps motif id -> tuple of opnames that comprise the motif.
    ``compressed_tokens`` is a streamable gene sequence where each repeated
    motif appears once as ``D:<id>:body`` and is then referenced by
    ``M:<id>``. ``raw_length`` is the unpacked literal count; the
    compression ratio is ``len(compressed_tokens) / raw_length`` when
    nonzero.
    """

    slots: dict[int, tuple[str, ...]]
    compressed_tokens: tuple[str, ...]
    raw_length: int

    @property
    def slot_count(self) -> int:
        return len(self.slots)


def opname_sequence(tokens: Iterable[str]) -> tuple[str, ...]:
    """Extract just the opnames from a ``B:<OPNAME>@<offset>`` stream."""

    seq: list[str] = []
    for tok in tokens:
        op = opname_of(tok)
        if op:
            seq.append(op)
    return tuple(seq)


def motif_dictionary(
    opnames: Sequence[str],
    *,
    motif_size: int = 3,
    min_repeats: int = 2,
) -> MotifDictionary:
    """Replace repeated length-``motif_size`` opname n-grams with motif refs.

    Greedy non-overlapping replacement: walk the sequence left-to-right,
    and whenever the next ``motif_size`` slots match an n-gram that
    appears at least ``min_repeats`` times in the original, emit ``M:<id>``
    and skip past it. Anything else emits ``L:<opname>``.

    Motifs are declared with ``D:<id>:body`` *before* their first use, so
    the output is decode-streamable. Definition order = first-use order.
    """

    if motif_size <= 0 or len(opnames) < motif_size:
        return MotifDictionary(
            slots={},
            compressed_tokens=tuple(f"L:{op}" for op in opnames) + ("E",),
            raw_length=len(opnames),
        )

    # Count every n-gram occurrence (overlapping count is fine for
    # picking candidates).
    ngrams: list[tuple[str, ...]] = [
        tuple(opnames[i : i + motif_size])
        for i in range(len(opnames) - motif_size + 1)
    ]
    counts = Counter(ngrams)
    candidates = {gram for gram, c in counts.items() if c >= min_repeats}

    slots: dict[int, tuple[str, ...]] = {}
    gram_to_slot: dict[tuple[str, ...], int] = {}
    next_id = 0

    out: list[str] = []
    i = 0
    while i < len(opnames):
        if i + motif_size <= len(opnames):
            gram = tuple(opnames[i : i + motif_size])
            if gram in candidates:
                if gram not in gram_to_slot:
                    slot = next_id
                    next_id += 1
                    gram_to_slot[gram] = slot
                    slots[slot] = gram
                    out.append("D:" + str(slot) + ":" + ",".join(gram))
                slot = gram_to_slot[gram]
                out.append(f"M:{slot}")
                i += motif_size
                continue
        out.append(f"L:{opnames[i]}")
        i += 1
    out.append("E")
    return MotifDictionary(
        slots=slots,
        compressed_tokens=tuple(out),
        raw_length=len(opnames),
    )


# ---------------------------------------------------------------------------
# Distributions over readable units.
# ---------------------------------------------------------------------------
def opname_distribution(tokens: Iterable[str]) -> Counter:
    """Counter over opnames from B:<OPNAME>@<offset> tokens.

    Offsets are dropped, so the distribution is stable across re-runs and
    independent of code layout.
    """

    counter: Counter = Counter()
    for tok in tokens:
        op = opname_of(tok)
        if op:
            counter[op] += 1
    return counter


def call_distribution(tokens: Iterable[str]) -> Counter:
    """Counter over call targets from CALL:<qualname> tokens."""

    counter: Counter = Counter()
    for tok in tokens:
        target = call_target_of(tok)
        if target:
            counter[target] += 1
    return counter


def motif_distribution(
    opnames: Sequence[str], *, motif_size: int = 3
) -> Counter:
    """Counter over length-``motif_size`` opname n-grams.

    Unlike ``motif_dictionary`` this does not rewrite — it just gives the
    distribution. Useful for spotting candidate motifs before compressing.
    """

    if motif_size <= 0 or len(opnames) < motif_size:
        return Counter()
    return Counter(
        tuple(opnames[i : i + motif_size])
        for i in range(len(opnames) - motif_size + 1)
    )


__all__ = [
    "BoundaryRun",
    "MotifDictionary",
    "boundary_runs_from_records",
    "boundary_summary_tokens",
    "call_distribution",
    "call_target_of",
    "motif_dictionary",
    "motif_distribution",
    "opname_distribution",
    "opname_of",
    "opname_sequence",
]
