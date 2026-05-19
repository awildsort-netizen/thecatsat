#!/usr/bin/env python3
"""Bridge raw bytecode token streams to readable gene-string summaries.

The B:<OPNAME>@<offset> / CALL:<qualname> / LINE:<n> stream from
``bytecode_genes`` is fine-grained: it tells you which byte fired. That
is body-level metabolism — exactly the right resolution for tracing,
but offset-noisy and unreadable as a composition record.

Readable forms in the streamable gene grammar already exist:

  * L:<name>        a composition event named ``name``
  * D:<i>:body      define motif slot ``i`` as ``body``
  * M:<i>           use motif slot ``i``

This experiment runs the same SAT functions used elsewhere in the
project through both lenses side-by-side. For each function we show:

  1. RAW BYTECODE DISTRIBUTION — opname distribution (offsets dropped)
     and the call-target distribution.
  2. BOUNDARY SUMMARY — collapse the trace records into one
     ``L:<qualname>`` token per code-boundary run (composer-grade form).
  3. MOTIF DICTIONARY — find repeated length-3 opname n-grams and rewrite
     the stream into ``D:<id>:body`` / ``M:<id>`` form. Decode-streamable;
     round-trips through ``streamable_genes.stream``.

The point is not "look how much compression" — the point is to
demonstrate the *interpretation sieve*: raw → distribution → motif
gene string, three increasingly readable summaries of the same activity.

Run with: ``python experiments/bytecode_to_gene_distribution.py``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytecode_gene_summary import (
    boundary_runs_from_records,
    boundary_summary_tokens,
    call_distribution,
    motif_dictionary,
    motif_distribution,
    opname_distribution,
    opname_sequence,
)
from bytecode_genes import (
    static_bytecode_tokens,
    static_call_targets,
    trace_call,
)
from sat_field import formula_graph, formula_graph_to_adjacency
from sat_furnace import _clause_pressures
from streamable_genes import stream


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt_counter_top(counter, n: int = 8) -> str:
    items = counter.most_common(n)
    suffix = f"  …(+{len(counter) - n})" if len(counter) > n else ""
    return "{" + ", ".join(f"{k}×{v}" for k, v in items) + "}" + suffix


def fmt_token_head(tokens, n: int = 10) -> str:
    head = list(tokens)[:n]
    suffix = f"  …(+{len(tokens) - n})" if len(tokens) > n else ""
    return "[" + ", ".join(head) + "]" + suffix


# ---------------------------------------------------------------------------
# Demo cases. Activation factors chosen so each function does real work.
# ---------------------------------------------------------------------------
DEMO_FORMULA = [
    ((0, False), (1, True), (2, False)),
    ((1, False), (2, True), (3, False)),
    ((0, True),  (2, False), (3, True)),
    ((1, True),  (3, False)),
]


def case_formula_graph() -> None:
    banner("formula_graph — readable summaries side-by-side")
    func = formula_graph
    trace = trace_call(func, DEMO_FORMULA)

    static_tokens = static_bytecode_tokens(func)
    op_tokens = trace.opcode_tokens()
    call_tokens = trace.call_tokens()

    print(f"  AF: a 4-clause CNF over 4 variables (k=3)")
    print(f"  raw static B: tokens : {len(static_tokens)}")
    print(f"  raw traced B: tokens : {len(op_tokens)}")
    print(f"  CALL: tokens         : {len(call_tokens)}")
    print()

    print("  (1) RAW BYTECODE DISTRIBUTION — offsets dropped")
    op_dist = opname_distribution(op_tokens)
    call_dist = call_distribution(call_tokens)
    print(f"    opname distribution (top 8): {fmt_counter_top(op_dist, 8)}")
    print(f"    call   distribution        : "
          f"{fmt_counter_top(call_dist, 8)}")
    static_calls = static_call_targets(func)
    print(f"    (static call-target heuristic: {tuple(static_calls)})")

    print()
    print("  (2) BOUNDARY SUMMARY — one L:<qualname> per contiguous run")
    runs = boundary_runs_from_records(trace.records)
    summary_tokens = boundary_summary_tokens(trace.records)
    print(f"    runs: {len(runs)}")
    for run in runs[:6]:
        top = run.opname_counts.most_common(3)
        top_str = ", ".join(f"{k}×{v}" for k, v in top) or "(no opcodes)"
        print(f"      L:{run.qualname:<32} "
              f"tokens={run.token_count:>3}  top opnames: {top_str}")
    if len(runs) > 6:
        print(f"      …(+{len(runs) - 6} more runs)")
    print(f"    streamable tokens: {fmt_token_head(summary_tokens, 10)}")
    # Round-trip through the streamable decoder.
    state = stream(summary_tokens)
    composable = state.composable_now()
    print(f"    decoder composable_now({len(composable)}): "
          f"{composable[:6]}{'…' if len(composable) > 6 else ''}")

    print()
    print("  (3) MOTIF DICTIONARY — repeated length-3 opname n-grams")
    op_seq = opname_sequence(op_tokens)
    raw_motif_dist = motif_distribution(op_seq, motif_size=3)
    top_motifs = raw_motif_dist.most_common(4)
    if top_motifs:
        for motif, count in top_motifs:
            print(f"    raw n-gram count: {'-'.join(motif):<40} ×{count}")
    motif_dict = motif_dictionary(op_seq, motif_size=3, min_repeats=2)
    print(f"    motif slots: {motif_dict.slot_count}")
    for slot, body in list(motif_dict.slots.items())[:4]:
        print(f"      D:{slot}: {'-'.join(body)}")
    print(f"    compressed length: {len(motif_dict.compressed_tokens)} "
          f"(vs {len(op_seq) + 1} raw L:<op>+E)")
    print(f"    head: {fmt_token_head(motif_dict.compressed_tokens, 10)}")
    # Round-trip: the compressed stream decodes back to the opname list.
    round_trip = stream(motif_dict.compressed_tokens)
    decoded = tuple(t.name for t in round_trip.emitted)
    print(f"    round-trip decoder yields {len(decoded)} literals "
          f"(matches raw opname count = {len(op_seq)}): "
          f"{decoded == tuple(op_seq)}")


def case_formula_graph_to_adjacency() -> None:
    banner("formula_graph_to_adjacency — repeat the lens on a chained operator")
    func = formula_graph_to_adjacency
    graph = formula_graph(DEMO_FORMULA)
    trace = trace_call(
        func, graph,
        extra_codes=(graph.to_adjacency.__func__.__code__,),
    )
    op_tokens = trace.opcode_tokens()
    runs = boundary_runs_from_records(trace.records)
    op_seq = opname_sequence(op_tokens)
    motif_dict = motif_dictionary(op_seq, motif_size=3, min_repeats=2)
    print(f"  trace records           : {len(trace.records)}")
    print(f"  opcode tokens           : {len(op_tokens)}")
    print(f"  boundary runs           : {len(runs)}")
    distinct_qns = []
    seen: set[str] = set()
    for r in runs:
        if r.qualname not in seen:
            seen.add(r.qualname)
            distinct_qns.append(r.qualname)
    print(f"  distinct qualnames ({len(distinct_qns)}): {distinct_qns}")
    print(f"  motif slots discovered  : {motif_dict.slot_count}")
    print(f"  compressed token count  : {len(motif_dict.compressed_tokens)}")
    summary_tokens = boundary_summary_tokens(trace.records)
    print(f"  boundary summary head   : {fmt_token_head(summary_tokens, 12)}")


def case_clause_pressures() -> None:
    banner("_clause_pressures — a heavier SAT operator, motifs should emerge")
    spins = [0.1, -0.4, 0.2, -0.1]
    trace = trace_call(_clause_pressures, DEMO_FORMULA, spins, 0.35)
    op_tokens = trace.opcode_tokens()
    op_seq = opname_sequence(op_tokens)
    op_dist = opname_distribution(op_tokens)
    motif_dist = motif_distribution(op_seq, motif_size=3)
    motif_dict = motif_dictionary(op_seq, motif_size=3, min_repeats=2)

    print(f"  AF: spins={spins}  temperature=0.35")
    print(f"  raw opcode tokens     : {len(op_tokens)}")
    print(f"  distinct opnames      : {len(op_dist)}")
    print(f"  raw 3-gram distribution top 5:")
    for gram, count in motif_dist.most_common(5):
        print(f"    {'-'.join(gram):<48} ×{count}")
    print(f"  motif slots discovered: {motif_dict.slot_count}")
    for slot, body in list(motif_dict.slots.items())[:5]:
        print(f"    D:{slot}: {'-'.join(body)}")
    print(f"  raw->compressed: {len(op_seq) + 1} -> "
          f"{len(motif_dict.compressed_tokens)} tokens")
    # Stable property: decoder always recovers the original opnames.
    decoded = tuple(t.name for t in stream(motif_dict.compressed_tokens).emitted)
    print(f"  round-trip matches raw opnames: {decoded == tuple(op_seq)}")


def main() -> None:
    print("Bytecode token streams → readable gene-string summaries")
    print("Three increasingly readable lenses on the same activity:")
    print("  (1) opname/call distribution (offsets dropped)")
    print("  (2) boundary summary L:<qualname>")
    print("  (3) motif dictionary D:<id>:body / M:<id>")
    case_formula_graph()
    case_formula_graph_to_adjacency()
    case_clause_pressures()
    print()
    print("Notes")
    print("  * The boundary summary is composer-grade: it is exactly the")
    print("    grammar streamable_genes already consumes (L:<name>). It")
    print("    drops byte-level resolution in favour of compartment-level")
    print("    composition events.")
    print("  * The motif dictionary is decode-streamable: D:<id>:body must")
    print("    appear before its first M:<id> use, and the decoder always")
    print("    recovers the original opname sequence.")
    print("  * Distributions over opnames/calls/motifs ride on structure,")
    print("    not on raw offsets, so they survive opcode renames as long")
    print("    as the opname stays.")
    print()
    print("done.")


if __name__ == "__main__":
    main()
