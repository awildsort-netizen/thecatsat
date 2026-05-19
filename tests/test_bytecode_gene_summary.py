#!/usr/bin/env python3
"""Structural tests for bytecode_gene_summary.

Python bytecode is CPython-version-specific. These tests assert *shapes
and relationships* — counts, set inclusions, token grammar, decode
round-trips — and avoid pinning exact opcode strings.
"""

from __future__ import annotations

import unittest

from bytecode_gene_summary import (
    boundary_runs_from_records,
    boundary_summary_tokens,
    call_distribution,
    call_target_of,
    motif_dictionary,
    motif_distribution,
    opname_distribution,
    opname_of,
    opname_sequence,
)
from bytecode_genes import static_bytecode_tokens, trace_call
from streamable_genes import stream


def toy_loop(items):
    total = 0
    for x in items:
        if x > 0:
            total += x
    return total


def toy_uses_helper(items):
    result = list(items)
    return len(result)


class TokenParseTests(unittest.TestCase):
    def test_opname_of_extracts_head(self):
        self.assertEqual(opname_of("B:LOAD_FAST@12"), "LOAD_FAST")

    def test_opname_of_returns_none_for_non_bytecode_token(self):
        self.assertIsNone(opname_of("L:foo"))
        self.assertIsNone(opname_of("CALL:bar"))
        self.assertIsNone(opname_of("E"))

    def test_call_target_extraction(self):
        self.assertEqual(call_target_of("CALL:list"), "list")
        self.assertIsNone(call_target_of("B:LOAD_FAST@0"))


class OpnameDistributionTests(unittest.TestCase):
    def test_distribution_drops_offsets(self):
        tokens = static_bytecode_tokens(toy_loop)
        counts = opname_distribution(tokens)
        # Offsets must not appear in keys.
        for op in counts:
            self.assertNotIn("@", op)
        # Total count equals number of B: tokens.
        bytecode_tokens = [t for t in tokens if t.startswith("B:")]
        self.assertEqual(sum(counts.values()), len(bytecode_tokens))

    def test_distribution_keys_are_opname_set(self):
        tokens = static_bytecode_tokens(toy_loop)
        from bytecode_genes import static_opnames

        expected = set(static_opnames(toy_loop))
        self.assertEqual(set(opname_distribution(tokens).keys()), expected)


class CallDistributionTests(unittest.TestCase):
    def test_call_distribution_picks_up_resolved_calls(self):
        trace = trace_call(toy_uses_helper, [1, 2, -1])
        counts = call_distribution(trace.call_tokens())
        # All resolved calls show up; their names must be non-empty.
        self.assertTrue(counts)
        self.assertTrue(all(isinstance(k, str) and k for k in counts))


class OpnameSequenceTests(unittest.TestCase):
    def test_sequence_preserves_order(self):
        tokens = static_bytecode_tokens(toy_loop)
        seq = opname_sequence(tokens)
        # length matches number of B: entries
        self.assertEqual(
            len(seq), sum(1 for t in tokens if t.startswith("B:"))
        )
        # first opname equals opname of first B: token
        first_b = next(t for t in tokens if t.startswith("B:"))
        self.assertEqual(seq[0], opname_of(first_b))


class MotifDistributionTests(unittest.TestCase):
    def test_size_3_distribution_lengths(self):
        seq = opname_sequence(static_bytecode_tokens(toy_loop))
        dist = motif_distribution(seq, motif_size=3)
        # All keys are 3-tuples of opnames.
        for key in dist:
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 3)

    def test_empty_sequence_yields_empty_distribution(self):
        self.assertEqual(motif_distribution([], motif_size=3), {})


class MotifDictionaryTests(unittest.TestCase):
    def test_repeat_motif_gets_one_slot(self):
        # Construct a sequence with one obviously repeated 3-gram.
        opnames = ["A", "B", "C", "A", "B", "C", "X", "Y"]
        result = motif_dictionary(opnames, motif_size=3, min_repeats=2)
        self.assertEqual(len(result.slots), 1)
        slot_body = next(iter(result.slots.values()))
        self.assertEqual(slot_body, ("A", "B", "C"))

    def test_compressed_stream_round_trips_through_decoder(self):
        opnames = ["A", "B", "C", "A", "B", "C", "X"]
        result = motif_dictionary(opnames, motif_size=3, min_repeats=2)
        state = stream(result.compressed_tokens)
        decoded = tuple(token.name for token in state.emitted)
        self.assertEqual(decoded, tuple(opnames))

    def test_no_motif_when_min_repeats_not_met(self):
        opnames = ["A", "B", "C", "X", "Y", "Z"]
        result = motif_dictionary(opnames, motif_size=3, min_repeats=2)
        self.assertEqual(result.slots, {})
        # Stream still decodes back to the original opnames.
        state = stream(result.compressed_tokens)
        decoded = tuple(token.name for token in state.emitted)
        self.assertEqual(decoded, tuple(opnames))

    def test_motif_compression_shortens_stream_with_repeats(self):
        # Repeat the same 4-token motif 4 times — strong compression target.
        opnames = ["LOAD", "ADD", "STORE", "JUMP"] * 4
        result = motif_dictionary(opnames, motif_size=4, min_repeats=2)
        self.assertGreaterEqual(len(result.slots), 1)
        # D:<id>:... + M:<id>*4 + E < 16 bare L:<op>+E tokens.
        self.assertLess(
            len(result.compressed_tokens), len(opnames) + 1
        )

    def test_short_sequence_returns_no_motifs(self):
        result = motif_dictionary(["A"], motif_size=3)
        self.assertEqual(result.slots, {})


class BoundaryRunTests(unittest.TestCase):
    def test_runs_partition_records_by_qualname(self):
        trace = trace_call(toy_loop, [1, 2, 3])
        runs = boundary_runs_from_records(trace.records)
        self.assertTrue(runs)
        # qualnames should all be non-empty strings
        for run in runs:
            self.assertTrue(run.qualname)
        # at least one run is for the target function itself
        self.assertIn("toy_loop", {run.qualname for run in runs})

    def test_run_opname_counts_have_no_offsets(self):
        trace = trace_call(toy_loop, [1, 2, 3])
        runs = boundary_runs_from_records(trace.records)
        for run in runs:
            for key in run.opname_counts:
                self.assertNotIn("@", key)

    def test_boundary_summary_tokens_end_with_E(self):
        trace = trace_call(toy_loop, [1, 2, 3])
        tokens = boundary_summary_tokens(trace.records)
        self.assertEqual(tokens[-1], "E")
        for tok in tokens[:-1]:
            self.assertTrue(tok.startswith("L:"))

    def test_boundary_summary_decodes_through_streamable_genes(self):
        trace = trace_call(toy_loop, [1, 2, 3])
        tokens = boundary_summary_tokens(trace.records)
        state = stream(tokens)
        # Every emitted token names a real qualname seen in the trace.
        emitted_names = {tok.name for tok in state.emitted}
        record_qns = {r.qualname for r in trace.records}
        self.assertTrue(emitted_names.issubset(record_qns))
        self.assertTrue(state.ended)


if __name__ == "__main__":
    unittest.main()
