#!/usr/bin/env python3

from __future__ import annotations

import unittest

from composer import Composer, FieldOperator

from streamable_genes import (
    StreamableGenome,
    iter_partial_states,
    pathway_hint,
    stream,
)


class StreamableGeneTests(unittest.TestCase):
    def test_prefix_decode_emits_tokens_in_order(self) -> None:
        state = stream(["L:formula_graph", "L:graph_adjacency", "E"])

        self.assertEqual(
            [token.name for token in state.emitted],
            ["formula_graph", "graph_adjacency"],
        )
        self.assertTrue(state.ended)

    def test_partial_decode_exposes_composable_names_before_end(self) -> None:
        tokens = ["L:formula_graph", "L:graph_adjacency", "L:spatial_samples", "E"]

        snapshots = [state.composable_now() for state in iter_partial_states(tokens)]

        self.assertEqual(snapshots[0], ("formula_graph",))
        self.assertEqual(snapshots[1], ("formula_graph", "graph_adjacency"))
        self.assertEqual(
            snapshots[2],
            ("formula_graph", "graph_adjacency", "spatial_samples"),
        )
        # End-of-stream marker does not add new composable names.
        self.assertEqual(snapshots[3], snapshots[2])

    def test_motif_define_then_reuse_via_backreference(self) -> None:
        state = stream(
            [
                "D:1:formula_graph,graph_adjacency",
                "M:1",
                "L:spatial_samples",
                "M:1",
                "E",
            ]
        )

        self.assertEqual(
            [token.name for token in state.emitted],
            [
                "formula_graph",
                "graph_adjacency",
                "spatial_samples",
                "formula_graph",
                "graph_adjacency",
            ],
        )
        # composable_now collapses duplicates: motif reuse is reuse, not noise.
        self.assertEqual(
            state.composable_now(),
            ("formula_graph", "graph_adjacency", "spatial_samples"),
        )

    def test_undefined_motif_reference_raises(self) -> None:
        genome = StreamableGenome()
        with self.assertRaises(KeyError):
            genome.feed("M:7")

    def test_type_window_open_close_reset_is_local(self) -> None:
        tokens = [
            "L:outside",
            "W:high_entropy",
            "L:inside_a",
            "W:nested",
            "L:inside_b",
            "R",
            "L:inside_c",
            "R",
            "L:outside_again",
            "E",
        ]

        state = stream(tokens)
        windows = {token.name: token.window for token in state.emitted}

        self.assertIsNone(windows["outside"])
        self.assertEqual(windows["inside_a"], "high_entropy")
        self.assertEqual(windows["inside_b"], "nested")
        # After closing the nested window we are back to the outer climate.
        self.assertEqual(windows["inside_c"], "high_entropy")
        self.assertIsNone(windows["outside_again"])
        # Stack fully unwound.
        self.assertEqual(state.window_stack, [])

    def test_attention_hint_attaches_to_next_literal_only(self) -> None:
        state = stream(["A:carry_pressure", "L:clause_pressure", "L:influence_lift", "E"])

        first, second = state.emitted
        self.assertEqual(first.attention, "carry_pressure")
        self.assertIsNone(second.attention)

    def test_pathway_hint_filters_to_known_composer_operators(self) -> None:
        def make_op(name: str, outputs: tuple[str, ...]) -> FieldOperator:
            return FieldOperator(
                name=name,
                inputs=(),
                outputs=outputs,
                run=lambda _ctx, _outs=outputs: {key: None for key in _outs},
            )

        composer = Composer(
            [
                make_op("formula_graph", ("formula_graph",)),
                make_op("graph_adjacency", ("graph_adjacency",)),
            ]
        )
        known = composer._operators.keys()

        # Partial stream: one known, one unknown name visible so far.
        partial = next(
            state
            for index, state in enumerate(
                iter_partial_states(["L:formula_graph", "L:unknown_gene", "E"])
            )
            if index == 1
        )

        self.assertEqual(pathway_hint(partial, known), ("formula_graph",))

    def test_feed_after_end_raises(self) -> None:
        genome = StreamableGenome()
        genome.feed("E")
        with self.assertRaises(ValueError):
            genome.feed("L:late")

    def test_unknown_token_head_raises(self) -> None:
        genome = StreamableGenome()
        with self.assertRaises(ValueError):
            genome.feed("Z:nope")


if __name__ == "__main__":
    unittest.main()
