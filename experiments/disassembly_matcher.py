#!/usr/bin/env python3
"""Disassembly-matching experiment over SAT operators.

NOT a decompiler. This is the bidirectional naming/cue layer that sits
between three views of the same code:

    Python source <--> bytecode motifs <--> readable gene summaries

It demonstrates two things on real SAT functions:

1.  *Forward* — when two operators share motif distributions and call
    targets, they are probably doing the same kind of work.

2.  *Reverse* — the CodeBoundary fields (co_names, co_varnames, co_consts,
    co_freevars, plus the static call-target heuristic) already carry
    enough cues to propose names for an operator and its locals. That is
    the *cue layer* that any future bytecode->Python pass would consume.

Run with: ``python experiments/disassembly_matcher.py``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disassembly_matcher import (
    disassembly_match,
    naming_cues,
    operator_name_candidates,
)
from sat_field import formula_graph, formula_graph_to_adjacency
from sat_furnace import _clause_pressures
from sat_metabolism import (
    assignment_hamming_movement,
    motif_reuse_count,
    operator_gene_entropy,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_motif(motif: tuple[str, ...]) -> str:
    return "·".join(motif)


# ---------------------------------------------------------------------------
# 1. Side-by-side disassembly matching.
# ---------------------------------------------------------------------------
def experiment_side_by_side() -> None:
    banner("1. Side-by-side disassembly matching across SAT operators")
    pairs = [
        (formula_graph, formula_graph_to_adjacency),
        (formula_graph, _clause_pressures),
        (_clause_pressures, assignment_hamming_movement),
        (motif_reuse_count, operator_gene_entropy),
    ]
    print(f"  {'LEFT':<38} {'RIGHT':<38} {'JACC':>5}  shared")
    for left, right in pairs:
        match = disassembly_match(left, right, motif_size=3)
        head = ", ".join(fmt_motif(m) for m, _ in match.shared[:2])
        print(
            f"  {match.left_qualname[:38]:<38} "
            f"{match.right_qualname[:38]:<38} "
            f"{match.jaccard:>5.2f}  {head}"
        )
    print()
    print("  Observation: pairs that operate on the same data shape share")
    print("  more motifs. Pairs from different layers (graph construction")
    print("  vs hamming distance) share less.")


# ---------------------------------------------------------------------------
# 2. Detail: what motifs and calls are shared.
# ---------------------------------------------------------------------------
def experiment_shared_detail() -> None:
    banner("2. Shared motifs + call dependencies — formula_graph vs clauses")
    match = disassembly_match(formula_graph, _clause_pressures, motif_size=3)
    print(f"  jaccard           : {match.jaccard:.3f}")
    print(f"  shared calls      : {match.shared_calls}")
    print(f"  left-only calls   : {match.left_only_calls[:6]}")
    print(f"  right-only calls  : {match.right_only_calls[:6]}")
    print(f"  top shared motifs :")
    for motif, score in match.shared[:6]:
        print(f"    {score:>3}×  {fmt_motif(motif)}")
    print()
    print("  Observation: the shared motifs are the iteration scaffolding")
    print("  (FOR_ITER / UNPACK / LOAD_FAST patterns) — that is what makes")
    print("  *both* of these functions clause-walkers, structurally.")


# ---------------------------------------------------------------------------
# 3. Naming cues per function.
# ---------------------------------------------------------------------------
def experiment_naming_cues() -> None:
    banner("3. Naming cues — what the CodeBoundary already tells us")
    for func in (formula_graph, formula_graph_to_adjacency, _clause_pressures):
        cues = naming_cues(func)
        print(f"  {cues.qualname}")
        print(f"    args         : {cues.arg_names}")
        print(f"    locals       : {cues.local_names[:6]}"
              f"{'...' if len(cues.local_names) > 6 else ''}")
        print(f"    co_names     : {cues.referenced_names[:6]}"
              f"{'...' if len(cues.referenced_names) > 6 else ''}")
        print(f"    freevars     : {cues.freevars}")
        print(f"    const kinds  : {cues.const_kinds[:6]}")
        print(f"    call targets : {cues.call_targets[:6]}")
        print(f"    top stems    : {cues.name_stems}")
        print()


# ---------------------------------------------------------------------------
# 4. Operator-name candidates (the reverse direction).
# ---------------------------------------------------------------------------
def experiment_name_candidates() -> None:
    banner("4. Operator-name candidates — if we had to reconstruct Python")
    targets = [
        formula_graph,
        formula_graph_to_adjacency,
        _clause_pressures,
        motif_reuse_count,
        operator_gene_entropy,
        assignment_hamming_movement,
    ]
    print(f"  {'FUNCTION':<48} candidates")
    for func in targets:
        candidates = operator_name_candidates(func, limit=5)
        qn = getattr(func, "__qualname__", func.__name__)
        print(f"  {qn[:48]:<48} {list(candidates)}")
    print()
    print("  Observation: the top candidate is almost always a stem the")
    print("  author already used (qualname > call-target > co_names).")
    print("  A reconstructor would seed identifier choices from this list")
    print("  *before* falling back to anonymous names like op_0, var_0.")


# ---------------------------------------------------------------------------
# 5. Closing note — what is missing for true reconstruction.
# ---------------------------------------------------------------------------
def experiment_closing_note() -> None:
    banner("5. What this is NOT — and what real reconstruction would need")
    print("  This experiment is a microscope and a cue layer. It is NOT a")
    print("  bytecode-to-Python decompiler. To actually round-trip a")
    print("  function from bytecode back to source we would need:")
    print()
    print("   * control-flow recovery (block graph, loop and if detection)")
    print("   * stack-effect modelling to reassemble expressions")
    print("   * SSA renaming using these naming cues as identifier hints")
    print("   * exception-table interpretation for try/except/finally")
    print("   * argument-default + closure-cell binding reconstruction")
    print()
    print("  What we have here is the readable midlayer: motif distributions")
    print("  + naming cues. A future pass can consume both directions of")
    print("  this layer — Python -> motifs (for naming experiments) and")
    print("  motifs -> Python (for reconstruction).")


def main() -> None:
    experiment_side_by_side()
    experiment_shared_detail()
    experiment_naming_cues()
    experiment_name_candidates()
    experiment_closing_note()


if __name__ == "__main__":
    main()
