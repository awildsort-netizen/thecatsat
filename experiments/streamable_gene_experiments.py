#!/usr/bin/env python3
"""Compact diagnostics for the streamable gene-string spine.

Run with: ``python experiments/streamable_gene_experiments.py``

Each experiment prints a short readable section. Everything is a diagnostic;
nothing is a scalar to optimize. The point is to make ecological behavior
(prefix composability, motif reuse, window locality, attention single-use,
composer overlap) visible at a glance.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composer import Composer, FieldOperator
from streamable_genes import (
    StreamableGenome,
    iter_partial_states,
    pathway_hint,
    stream,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_tokens(tokens) -> str:
    return "[" + ", ".join(t.name for t in tokens) + "]"


# ---------------------------------------------------------------------------
# 1. Prefix organism: how composable hints evolve token-by-token before EOF.
# ---------------------------------------------------------------------------
def experiment_prefix_organism() -> None:
    banner("1. Prefix organism — composable hints grow before EOF")
    tokens = [
        "L:formula_graph",
        "L:graph_adjacency",
        "L:spatial_samples",
        "L:clause_pressure",
        "E",
    ]
    print(f"  source tokens ({len(tokens)}): {tokens}")
    print()
    for i, state in enumerate(iter_partial_states(tokens), start=1):
        last = tokens[i - 1]
        composable = state.composable_now()
        print(f"  step {i:>2}  feed {last!r:>28}  "
              f"composable={composable}")
    print()
    print("  Observation: each L: token monotonically extends the composable")
    print("  set. EOF is a no-op for composability — the planner can act on")
    print("  any prefix; the stream is genuinely streamable.")


# ---------------------------------------------------------------------------
# 2. Motif reuse — emitted vs source token length as a compression diagnostic.
# ---------------------------------------------------------------------------
def experiment_motif_reuse() -> None:
    banner("2. Motif reuse — compression ratio as a diagnostic")
    # Motif slot 1 = the "graph build" motif. Reused three times.
    tokens = [
        "D:1:formula_graph,graph_adjacency,spatial_samples",
        "M:1",
        "W:nested_clause_window",
        "M:1",
        "L:clause_pressure",
        "R",
        "M:1",
        "E",
    ]
    state = stream(tokens)

    emitted = state.emitted
    # Source cost: count payload-bearing tokens (exclude E).
    source_payload = [t for t in tokens if t != "E"]
    print(f"  source tokens ({len(source_payload)}): {source_payload}")
    print(f"  emitted literals ({len(emitted)}): "
          f"{[t.name for t in emitted]}")
    print(f"  composable_now (dedup'd): {state.composable_now()}")
    print()
    ratio = len(emitted) / max(1, len(source_payload))
    print(f"  expansion ratio = emitted / source_payload "
          f"= {len(emitted)}/{len(source_payload)} = {ratio:.2f}x")
    print()
    # Reuse ratio: literal expansions of motif body vs single define cost.
    motif_body_len = len(state.motifs[1])
    motif_uses = sum(1 for t in tokens if t == "M:1")
    literal_equiv = motif_body_len * motif_uses
    define_cost = 1
    print(f"  motif #1 body length = {motif_body_len}, uses = {motif_uses}")
    print(f"  literal-equivalent emissions = {literal_equiv}; "
          f"with motif: {motif_uses} M-refs + 1 D-define = "
          f"{motif_uses + define_cost}")
    print(f"  motif compression = "
          f"{literal_equiv}/{motif_uses + define_cost} = "
          f"{literal_equiv / (motif_uses + define_cost):.2f}x")
    print()
    print("  Observation: motifs are reusable composition structure. The")
    print("  same body re-emits inside different type windows without re-")
    print("  declaring it. composable_now stays minimal — reuse is reuse,")
    print("  not noise.")


# ---------------------------------------------------------------------------
# 3. Type windows — nested/open/reset, climates ride on the emitted token.
# ---------------------------------------------------------------------------
def experiment_type_windows() -> None:
    banner("3. Type windows — local climates carried on tokens")
    tokens = [
        "L:bare_token",
        "W:high_entropy",
        "L:in_high",
        "W:clause_pressure_climate",
        "L:in_clause",
        "R",                       # close inner
        "L:back_in_high",
        "W:low_entropy",           # sibling window after the nested one
        "L:in_low",
        "R",
        "R",                       # close outer
        "L:bare_again",
        "E",
    ]
    state = stream(tokens)
    print("  per-token (name -> window):")
    for t in state.emitted:
        print(f"    {t.name:>16}  window={t.window}")
    print()
    print(f"  final window stack: {state.window_stack}")
    print()
    # Diagnostic: how much locality? Fraction of tokens with a window.
    n = len(state.emitted)
    with_window = sum(1 for t in state.emitted if t.window is not None)
    print(f"  window locality coverage = {with_window}/{n} = "
          f"{with_window / n:.0%} of emitted tokens carry a local climate.")
    print()
    print("  Observation: windows are local. Closing the nested window")
    print("  restores the parent climate ('back_in_high' is back under")
    print("  high_entropy). A sibling window opened later does not leak")
    print("  into prior tokens. Stack fully unwinds before EOF.")


# ---------------------------------------------------------------------------
# 4. Attention — where hints land, and that they are single-use.
# ---------------------------------------------------------------------------
def experiment_attention() -> None:
    banner("4. Attention — single-use inheritance hints")
    tokens = [
        "A:carry_pressure",
        "L:clause_pressure",       # gets the hint
        "L:influence_lift",        # does NOT inherit
        "A:carry_locality",
        "W:nested",                # window does not consume
        "L:locality_probe",        # gets the hint
        "R",
        "A:carry_bridge",
        "A:override_bridge",       # second A: replaces pending hint
        "L:bridge_walker",         # gets override_bridge
        "E",
    ]
    state = stream(tokens)
    print("  per-token (name -> attention):")
    for t in state.emitted:
        marker = " <-- carried" if t.attention else ""
        print(f"    {t.name:>20}  attention={t.attention}{marker}")
    print()
    carried = sum(1 for t in state.emitted if t.attention is not None)
    print(f"  attention coverage = {carried}/{len(state.emitted)} "
          f"emitted tokens carry a hint")
    print()
    print("  Observation: A: is single-use. A second A: before any literal")
    print("  silently replaces the first (override behavior). Worth noting:")
    print("  there is no error and no warning when a hint is overwritten —")
    print("  could be intentional ('latest wins') or a future design pressure")
    print("  (warn? stack? carry both?).")


# ---------------------------------------------------------------------------
# 5. Composer overlap — pathway_hint coverage against a real Composer.
# ---------------------------------------------------------------------------
def experiment_composer_overlap() -> None:
    banner("5. Composer overlap — pathway_hint coverage")

    def make_op(name: str, outputs: tuple[str, ...]) -> FieldOperator:
        return FieldOperator(
            name=name,
            inputs=(),
            outputs=outputs,
            run=lambda _ctx, _outs=outputs: {key: None for key in _outs},
        )

    composer = Composer([
        make_op("formula_graph", ("formula_graph",)),
        make_op("graph_adjacency", ("graph_adjacency",)),
        make_op("spatial_samples", ("spatial_samples",)),
    ])
    known = tuple(composer._operators.keys())
    print(f"  composer knows: {known}")
    print()

    cases = {
        "all_known": [
            "L:formula_graph", "L:graph_adjacency", "L:spatial_samples", "E"
        ],
        "mixed": [
            "L:formula_graph", "L:unknown_alpha", "L:graph_adjacency",
            "L:unknown_beta", "E",
        ],
        "all_unknown": [
            "L:unknown_alpha", "L:unknown_beta", "L:unknown_gamma", "E"
        ],
        "motif_to_known": [
            "D:1:formula_graph,graph_adjacency", "M:1",
            "L:unknown_gamma", "E",
        ],
    }
    for label, tokens in cases.items():
        state = stream(tokens)
        visible = state.composable_now()
        hint = pathway_hint(state, known)
        cov = len(hint) / max(1, len(visible))
        print(f"  case={label!r}")
        print(f"    visible composable: {visible}")
        print(f"    pathway_hint:       {hint}")
        print(f"    coverage:           {len(hint)}/{len(visible)} = {cov:.0%}")
        print()
    print("  Observation: pathway_hint is a one-way filter (gene -> composer).")
    print("  It does not say which composer operators were *missed* by the")
    print("  gene stream. That asymmetry might be a feature (we trust the")
    print("  stream's choice) or a future diagnostic to add ('unused ops').")


# ---------------------------------------------------------------------------
# 6. Bonus probe: ambiguity & reset behavior on degenerate streams.
# ---------------------------------------------------------------------------
def experiment_edge_resets() -> None:
    banner("6. Edge probes — extra resets, empty motif, attention before EOF")

    # Extra R when no window is open: silent no-op (current behavior).
    state = stream(["R", "R", "L:bare", "E"])
    print(f"  extra-R: emitted={fmt_tokens(state.emitted)} "
          f"window_stack={state.window_stack}")

    # Empty-body motif define.
    g = StreamableGenome()
    g.feed("D:9:")  # empty body
    g.feed("M:9")   # expands to nothing
    g.feed("L:after_empty_motif")
    g.feed("E")
    print(f"  empty-motif: emitted={fmt_tokens(g.state.emitted)} "
          f"motifs={dict(g.state.motifs)}")

    # Attention left dangling at EOF: silently dropped.
    g = StreamableGenome()
    g.feed("A:dangling")
    g.feed("E")
    print(f"  dangling-A: emitted={fmt_tokens(g.state.emitted)} "
          f"pending_attention={g.state.pending_attention}")
    print()
    print("  Observation: stray R, empty motif body, and dangling A all")
    print("  fail-soft. Friendly to noisy/streamed input; may hide bugs.")
    print("  Worth a design call: should any of these surface a warning?")


def main() -> None:
    experiment_prefix_organism()
    experiment_motif_reuse()
    experiment_type_windows()
    experiment_attention()
    experiment_composer_overlap()
    experiment_edge_resets()
    print()
    print("done.")


if __name__ == "__main__":
    main()
