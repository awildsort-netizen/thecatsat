#!/usr/bin/env python3
"""Use real Python bytecode as the first ecological substrate.

Hypothesis: instead of inventing a gene VM, observe operators *through* the
bytecode the CPython interpreter already executes for them. Python gives
free of charge: function objects, code objects, instruction offsets,
LOAD/CALL/STORE, locals/globals/freevars, line numbers, exception tables.

Static bytecode = the possible pathways an operator's body affords.
Runtime trace events = the pathway actually taken under one input.
Tests / activation factors decide which static motifs become events.

This script is a microscope slide, not a portable VM. CPython 3.12.8 is the
substrate; opcode names and offsets are stable within this minor version
only. Tests assert structural properties (token shapes, set relationships),
not exact opcode strings.

Run with: ``python experiments/python_bytecode_genes.py``
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytecode_genes import (
    CodeBoundary,
    pathway_diff,
    static_bytecode_tokens,
    static_call_targets,
    static_opnames,
    trace_call,
)
from sat_field import (
    formula_graph,
    formula_graph_to_adjacency,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_tokens(tokens, limit: int = 8) -> str:
    head = list(tokens)[:limit]
    suffix = f"  …(+{len(tokens) - limit})" if len(tokens) > limit else ""
    return "[" + ", ".join(head) + "]" + suffix


# ---------------------------------------------------------------------------
# 1. Static lens: what the operator's body affords.
# ---------------------------------------------------------------------------
def experiment_static_extraction() -> None:
    banner("1. Static bytecode extraction — what formula_graph affords")
    boundary = CodeBoundary.of(formula_graph)
    print(f"  qualname     : {boundary.qualname}")
    print(f"  argcount     : {boundary.argcount}   nlocals={boundary.nlocals}")
    print(f"  co_names     : {boundary.names}")
    print(f"  co_varnames  : {boundary.varnames}")
    print(f"  co_freevars  : {boundary.freevars}   (closed-over cells)")
    print(f"  co_consts    : {boundary.consts!r:.96}")

    static_tokens = static_bytecode_tokens(formula_graph)
    opnames = static_opnames(formula_graph)
    targets = static_call_targets(formula_graph)

    print(f"\n  static instructions       : {len(static_tokens)}")
    print(f"  distinct opnames          : {len(opnames)}")
    print(f"  opnames                   : {fmt_tokens(opnames, 12)}")
    print(f"  best-effort call targets  : {targets}")
    print(f"  first 8 instruction tokens: {fmt_tokens(static_tokens, 8)}")
    print()
    print("  Observation: co_names + heuristic call targets already surface")
    print("  the dispatched motifs (enumerate, append, FormulaGraph). This")
    print("  is the availability map — the substrate the operator *can* use.")


# ---------------------------------------------------------------------------
# 2. Runtime lens: actual pathway under a specific activation factor.
# ---------------------------------------------------------------------------
def experiment_runtime_trace() -> None:
    banner("2. Runtime-aware trace — actual instructions executed")
    formula = [((1, False), (2, True)), ((2, False), (3, True))]
    trace = trace_call(formula_graph, formula)

    op_tokens = trace.opcode_tokens()
    line_count = sum(1 for r in trace.records if r.kind == "line")
    call_count = sum(1 for r in trace.records if r.kind == "call")

    print(f"  input formula           : {formula}")
    print(f"  total trace records     : {len(trace.records)}")
    print(f"  opcode events           : {len(op_tokens)}")
    print(f"  line events             : {line_count}")
    print(f"  resolved CALL events    : {call_count}")
    print(f"  first 10 opcode tokens  : {fmt_tokens(op_tokens, 10)}")

    call_names = Counter(r.qualname for r in trace.records if r.kind == "call")
    print(f"  call multiset           : {dict(call_names)}")
    print()
    print("  Observation: opcode events come from sys.monitoring (PEP 669) —")
    print("  the modern 3.12+ API. Line/call events are always informative;")
    print("  opcode events would be portability-fragile in production code,")
    print("  but invaluable as a microscope here.")


# ---------------------------------------------------------------------------
# 3. Static-vs-activated: the latent pathway gap.
# ---------------------------------------------------------------------------
def experiment_static_vs_activated() -> None:
    banner("3. Static vs activated — availability minus event = latent path")
    cases = [
        ("empty",  []),
        ("unit",   [((1, False),)]),
        ("binary", [((1, False), (2, True))]),
        ("big",    [((i, bool(i % 2)), (i + 1, False)) for i in range(1, 6)]),
    ]
    print(f"  {'AF':>8} {'opcodes':>8} {'shared':>7} {'static':>7} "
          f"{'ratio':>6}  static_only_opnames")
    seen_static_only: dict[str, int] = {}
    for label, formula in cases:
        trace = trace_call(formula_graph, formula)
        diff = pathway_diff(formula_graph, trace)
        static_only_ops = Counter(
            tok.split("@")[0].split(":", 1)[1] for tok in diff.static_only
        )
        for op in static_only_ops:
            seen_static_only[op] = seen_static_only.get(op, 0) + 1
        compact = ",".join(f"{k}×{v}" for k, v in static_only_ops.most_common(4))
        print(
            f"  {label:>8} {len(trace.opcode_tokens()):>8} "
            f"{len(diff.shared):>7} {diff.static_total:>7} "
            f"{diff.activation_ratio:>6.2f}  {compact}"
        )
    print()
    print("  Observation: the empty formula touches ~1/3 of static tokens —")
    print("  the rest are the FOR_ITER body of the clause-walk. Adding even a")
    print("  unit clause activates the inner UNPACK_SEQUENCE/LOAD_ATTR/CALL")
    print("  motif. The 'static_only' set is the operator's *latent* pathway")
    print("  under this AF — a natural concentration-field input: motifs that")
    print("  exist as potential but not as event.")


# ---------------------------------------------------------------------------
# 4. Gene-token stream from bytecode observations.
# ---------------------------------------------------------------------------
def experiment_bytecode_gene_stream() -> None:
    banner("4. Bytecode gene-token stream — emit B:/CALL:/LINE: tokens")
    formula = [((1, False), (2, True)), ((2, False),)]
    trace = trace_call(formula_graph, formula)
    flat_stream = trace.tokens()
    print(f"  stream length             : {len(flat_stream)}")
    print(f"  first 16 tokens           : {fmt_tokens(flat_stream, 16)}")

    print()
    print("  Compared with L:<operator> tokens from streamable_genes.py, this")
    print("  stream carries strictly more structure:")
    print("   - L:<op>   says \"operator named X was composed\".")
    print("   - B:<OPNAME>@<offset> says \"this byte of X's body fired\".")
    print("   - CALL:<qualname> says \"X dispatched to Y\".")
    print("   - LINE:<n> says \"source line n was the cursor here\".")
    print()
    print("  The L: stream is the composer's plan; the B:/CALL:/LINE: stream")
    print("  is the executed body. The two together let us distinguish")
    print("  *availability* (which operators were composable) from")
    print("  *activation* (which bytes were actually executed under the AF).")


# ---------------------------------------------------------------------------
# 5. Cross-operator: chain formula_graph → formula_graph_to_adjacency.
# ---------------------------------------------------------------------------
def experiment_chained_operators() -> None:
    banner("5. Chained operators — two code-object compartments at once")
    formula = [((1, False), (2, True)), ((2, False), (3, True))]
    graph = formula_graph(formula)
    trace = trace_call(
        formula_graph_to_adjacency,
        graph,
        extra_codes=(),
    )
    own_tokens = trace.opcode_tokens()
    print(f"  formula_graph_to_adjacency: {len(own_tokens)} opcode events")
    print(f"  first 8                   : {fmt_tokens(own_tokens, 8)}")

    # Now follow the call into to_adjacency on the FormulaGraph dataclass.
    trace2 = trace_call(
        formula_graph_to_adjacency,
        graph,
        extra_codes=(graph.to_adjacency.__func__.__code__,),
    )
    qualnames = Counter(
        r.qualname for r in trace2.records if r.kind == "opcode"
    )
    print(f"  with extra_codes=to_adjacency")
    print(f"    opcode events per compartment: {dict(qualnames)}")
    print()
    print("  Observation: by attaching extra code objects we can watch a")
    print("  call traverse compartment boundaries — exactly the substrate a")
    print("  motif compressor needs (\"this opname run repeats across")
    print("  function calls, lift it into a motif\").")


def main() -> None:
    experiment_static_extraction()
    experiment_runtime_trace()
    experiment_static_vs_activated()
    experiment_bytecode_gene_stream()
    experiment_chained_operators()


if __name__ == "__main__":
    main()
