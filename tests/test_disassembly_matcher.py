#!/usr/bin/env python3
"""Structural tests for disassembly_matcher.

Assert *shapes and relationships* — Jaccard bounds, set inclusions,
stem-derivation invariants — not exact opcode strings or stem strings.
This module is not a decompiler; the tests guard the cue layer.
"""

from __future__ import annotations

import unittest
from collections import Counter

from disassembly_matcher import (
    DisassemblyMatch,
    NamingCues,
    disassembly_match,
    motif_similarity,
    naming_cues,
    operator_name_candidates,
    shared_motifs,
)
from sat_field import formula_graph, formula_graph_to_adjacency


def loop_sum(items):
    total = 0
    for x in items:
        total += x
    return total


def loop_product(items):
    total = 1
    for x in items:
        total *= x
    return total


def constant_seven():
    return 7


class MotifSimilarityTests(unittest.TestCase):
    def test_identical_distributions_are_one(self):
        dist = Counter({("A", "B", "C"): 2, ("B", "C", "D"): 1})
        self.assertEqual(motif_similarity(dist, dist), 1.0)

    def test_disjoint_distributions_are_zero(self):
        a = Counter({("A", "B", "C"): 1})
        b = Counter({("X", "Y", "Z"): 1})
        self.assertEqual(motif_similarity(a, b), 0.0)

    def test_empty_pair_is_zero(self):
        self.assertEqual(motif_similarity({}, {}), 0.0)

    def test_jaccard_in_unit_interval(self):
        # Same family of loops: similarity should be in [0, 1].
        a = disassembly_match(loop_sum, loop_product)
        self.assertGreaterEqual(a.jaccard, 0.0)
        self.assertLessEqual(a.jaccard, 1.0)

    def test_sibling_loops_more_similar_than_constant(self):
        # Two near-identical loop bodies should be more similar to each
        # other than either is to a constant-return function.
        sibling = disassembly_match(loop_sum, loop_product).jaccard
        far = disassembly_match(loop_sum, constant_seven).jaccard
        self.assertGreaterEqual(sibling, far)


class SharedMotifsTests(unittest.TestCase):
    def test_shared_motifs_subset_of_each_input(self):
        match = disassembly_match(loop_sum, loop_product)
        # Every motif reported as shared should be in both inputs.
        # Re-derive distributions to check.
        from bytecode_genes import static_bytecode_tokens
        from bytecode_gene_summary import motif_distribution, opname_sequence
        a = motif_distribution(opname_sequence(static_bytecode_tokens(loop_sum)))
        b = motif_distribution(opname_sequence(static_bytecode_tokens(loop_product)))
        for motif, score in match.shared:
            self.assertIn(motif, a)
            self.assertIn(motif, b)
            self.assertGreater(score, 0)

    def test_shared_motifs_sorted_descending(self):
        match = disassembly_match(loop_sum, loop_product)
        scores = [score for _, score in match.shared]
        self.assertEqual(scores, sorted(scores, reverse=True))


class NamingCuesTests(unittest.TestCase):
    def test_arg_names_match_argcount(self):
        cues = naming_cues(formula_graph)
        # formula_graph has one positional arg.
        self.assertEqual(len(cues.arg_names), 1)
        self.assertEqual(cues.arg_names[0], "formula")

    def test_call_targets_include_constructed_types(self):
        cues = naming_cues(formula_graph)
        # Best-effort heuristic surfaces FormulaGraph / FormulaGraphEdge.
        joined = " ".join(cues.call_targets)
        self.assertIn("FormulaGraph", joined)

    def test_name_stems_non_empty_when_body_non_trivial(self):
        cues = naming_cues(formula_graph)
        self.assertGreater(len(cues.name_stems), 0)
        for stem in cues.name_stems:
            self.assertGreaterEqual(len(stem), 2)
            self.assertEqual(stem, stem.lower())

    def test_const_kinds_are_type_names(self):
        cues = naming_cues(formula_graph)
        for k in cues.const_kinds:
            self.assertIsInstance(k, str)
            self.assertTrue(k.isidentifier())


class OperatorNameCandidatesTests(unittest.TestCase):
    def test_candidates_prefix_includes_qualname_stems(self):
        # The author-chosen qualname must seed the candidate list first.
        candidates = operator_name_candidates(formula_graph)
        self.assertTrue(candidates)
        # The qualname contains "formula" and "graph"; both should be early.
        head = list(candidates)
        self.assertIn("formula", head)
        self.assertIn("graph", head)

    def test_candidates_are_unique_and_lowercase(self):
        candidates = operator_name_candidates(formula_graph_to_adjacency)
        self.assertEqual(len(set(candidates)), len(candidates))
        for c in candidates:
            self.assertEqual(c, c.lower())

    def test_limit_respected(self):
        candidates = operator_name_candidates(formula_graph, limit=3)
        self.assertLessEqual(len(candidates), 3)


class DisassemblyMatchTests(unittest.TestCase):
    def test_self_match_is_identity_like(self):
        match = disassembly_match(formula_graph, formula_graph)
        self.assertEqual(match.jaccard, 1.0)
        self.assertEqual(match.left_only_calls, ())
        self.assertEqual(match.right_only_calls, ())

    def test_match_record_is_immutable(self):
        match = disassembly_match(loop_sum, loop_product)
        with self.assertRaises(Exception):
            match.jaccard = 0.5  # type: ignore[misc]

    def test_left_only_and_right_only_calls_are_disjoint(self):
        match = disassembly_match(formula_graph, formula_graph_to_adjacency)
        self.assertEqual(
            set(match.left_only_calls) & set(match.right_only_calls), set()
        )
        self.assertEqual(
            set(match.left_only_calls) & set(match.shared_calls), set()
        )


if __name__ == "__main__":
    unittest.main()
