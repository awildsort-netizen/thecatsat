#!/usr/bin/env python3
"""Disassembly matcher: a bidirectional naming/cue layer.

This module is NOT a bytecode-to-Python decompiler. It is a microscope-grade
naming-cue experiment that sits between three views of the same code:

* Python source                  (operator definitions)
* CPython bytecode               (what the interpreter executes)
* Gene-summary motifs            (readable opname n-grams + call targets)

Goal: when two functions share motif distributions and call dependencies,
they are *probably* doing the same kind of work — and the fields the
interpreter already gave us (``co_names``, ``co_varnames``, ``co_consts``,
``co_freevars``) carry enough hints to suggest names for operators and
variables we are about to lift back into Python.

The helpers are intentionally tiny:

* ``motif_similarity(a, b)``       — Jaccard over motif distributions.
* ``shared_motifs(a, b)``          — motifs present in both, by total count.
* ``naming_cues(func)``            — readable boundary-field cues.
* ``operator_name_candidates(func)`` — ordered name stems for an operator.

Both directions are supported by the same fields:

  Python -> bytecode -> gene summary -> naming cues
       (forward: what is this function doing?)

  gene summary + cues -> suggested operator/variable names
       (reverse: if I had to reconstruct Python from these motifs, what
       would I call the operator and its locals?)

True bytecode->Python reconstruction would need control-flow recovery,
stack-effect modelling, and SSA renaming. None of that lives here. What
lives here is the cue layer that would feed those steps.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from bytecode_genes import (
    CodeBoundary,
    static_bytecode_tokens,
    static_call_targets,
)
from bytecode_gene_summary import (
    motif_distribution,
    opname_sequence,
)


# ---------------------------------------------------------------------------
# Similarity over motif distributions.
# ---------------------------------------------------------------------------
def motif_similarity(
    a: Mapping[tuple[str, ...], int],
    b: Mapping[tuple[str, ...], int],
) -> float:
    """Jaccard similarity between two motif distributions.

    Both ``a`` and ``b`` are mappings from motif (opname tuple) to count;
    the count itself is ignored — only key sets matter. Returns 0.0 when
    both inputs are empty (no shared structure to compare).
    """

    keys_a = set(a)
    keys_b = set(b)
    if not keys_a and not keys_b:
        return 0.0
    inter = keys_a & keys_b
    union = keys_a | keys_b
    return len(inter) / len(union)


def shared_motifs(
    a: Mapping[tuple[str, ...], int],
    b: Mapping[tuple[str, ...], int],
) -> list[tuple[tuple[str, ...], int]]:
    """Motifs in both distributions, scored by ``min(count_a, count_b)``.

    Sorted by score descending then by motif lexicographically — stable
    output for tests and side-by-side prints.
    """

    common = set(a) & set(b)
    scored = [(motif, min(a[motif], b[motif])) for motif in common]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


# ---------------------------------------------------------------------------
# Naming cues from CodeBoundary fields.
# ---------------------------------------------------------------------------
_SPLIT_NAME = re.compile(r"[_\W]+")


def _stems(name: str) -> tuple[str, ...]:
    return tuple(part for part in _SPLIT_NAME.split(name) if part)


@dataclass(frozen=True)
class NamingCues:
    """Readable cues mined from a function's CodeBoundary.

    These are exactly the hints a renamer / decompiler would want before
    fabricating a Python identifier. Nothing here is normative — each
    field is a suggestion, ranked by how often it appears.
    """

    qualname: str
    arg_names: tuple[str, ...]
    local_names: tuple[str, ...]
    referenced_names: tuple[str, ...]   # co_names: globals/attrs
    freevars: tuple[str, ...]
    const_kinds: tuple[str, ...]        # type-name fingerprint of co_consts
    call_targets: tuple[str, ...]
    name_stems: tuple[str, ...]         # top stems mined from all of the above


def naming_cues(func: Callable[..., Any]) -> NamingCues:
    """Mine readable naming cues from the function's code boundary."""

    boundary = CodeBoundary.of(func)
    arg_names = boundary.varnames[: boundary.argcount]
    local_names = boundary.varnames[boundary.argcount :]
    call_targets = static_call_targets(func)

    const_kinds = tuple(
        type(c).__name__ for c in boundary.consts if c is not None
    )

    pool: Counter = Counter()
    for source in (
        boundary.names,
        local_names,
        arg_names,
        boundary.freevars,
        call_targets,
    ):
        for name in source:
            for stem in _stems(name):
                stem = stem.lower()
                if len(stem) < 2:
                    continue
                pool[stem] += 1

    stems = tuple(stem for stem, _ in pool.most_common(8))
    return NamingCues(
        qualname=boundary.qualname,
        arg_names=arg_names,
        local_names=local_names,
        referenced_names=boundary.names,
        freevars=boundary.freevars,
        const_kinds=const_kinds,
        call_targets=call_targets,
        name_stems=stems,
    )


# ---------------------------------------------------------------------------
# Operator name candidates.
# ---------------------------------------------------------------------------
def operator_name_candidates(func: Callable[..., Any], limit: int = 5) -> tuple[str, ...]:
    """Ordered name-stem candidates for an operator.

    Heuristics, all derived from data the interpreter already gave us:

    * Stems that already appear in the function's own qualname win — that
      is the author's chosen vocabulary, preserved verbatim.
    * Stems that appear in resolved call targets are next — what the
      function dispatches into is what it is "about".
    * Stems mined from referenced names (``co_names``) come last — they
      include attribute reads (``edges``, ``append``, ``items``) that are
      good operator suffixes.

    Duplicates are dropped while preserving first-seen order.
    """

    cues = naming_cues(func)
    qn_stems = [s.lower() for s in _stems(cues.qualname)]
    call_stems: list[str] = []
    for target in cues.call_targets:
        for stem in _stems(target):
            call_stems.append(stem.lower())
    referenced_stems: list[str] = []
    for name in cues.referenced_names:
        for stem in _stems(name):
            referenced_stems.append(stem.lower())

    ordered: list[str] = []
    for source in (qn_stems, call_stems, referenced_stems, list(cues.name_stems)):
        for stem in source:
            if len(stem) < 2:
                continue
            if stem not in ordered:
                ordered.append(stem)
    return tuple(ordered[:limit])


# ---------------------------------------------------------------------------
# Function-level comparison report.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DisassemblyMatch:
    """Side-by-side comparison of two functions' bytecode-derived motifs."""

    left_qualname: str
    right_qualname: str
    jaccard: float
    shared: tuple[tuple[tuple[str, ...], int], ...]
    shared_calls: tuple[str, ...]
    left_only_calls: tuple[str, ...]
    right_only_calls: tuple[str, ...]


def disassembly_match(
    left: Callable[..., Any],
    right: Callable[..., Any],
    *,
    motif_size: int = 3,
) -> DisassemblyMatch:
    """Compare two functions via their static motif and call distributions."""

    left_seq = opname_sequence(static_bytecode_tokens(left))
    right_seq = opname_sequence(static_bytecode_tokens(right))
    left_motifs = motif_distribution(left_seq, motif_size=motif_size)
    right_motifs = motif_distribution(right_seq, motif_size=motif_size)
    left_calls = set(static_call_targets(left))
    right_calls = set(static_call_targets(right))

    shared = tuple(shared_motifs(left_motifs, right_motifs))
    return DisassemblyMatch(
        left_qualname=getattr(left, "__qualname__", left.__name__),
        right_qualname=getattr(right, "__qualname__", right.__name__),
        jaccard=motif_similarity(left_motifs, right_motifs),
        shared=shared,
        shared_calls=tuple(sorted(left_calls & right_calls)),
        left_only_calls=tuple(sorted(left_calls - right_calls)),
        right_only_calls=tuple(sorted(right_calls - left_calls)),
    )


__all__ = [
    "DisassemblyMatch",
    "NamingCues",
    "disassembly_match",
    "motif_similarity",
    "naming_cues",
    "operator_name_candidates",
    "shared_motifs",
]
