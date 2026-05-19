#!/usr/bin/env python3
"""Structural tests for bytecode_genes.

Python bytecode is CPython-version-specific. These tests assert *broad
properties* — token shape, ordering, subset relationships, qualname capture
— and deliberately avoid pinning exact opcode strings or offsets.
"""

from __future__ import annotations

import unittest

from bytecode_genes import (
    CodeBoundary,
    pathway_diff,
    static_bytecode_tokens,
    static_call_targets,
    static_opnames,
    trace_call,
)


def toy_loop(items):
    total = 0
    for x in items:
        if x > 0:
            total += x
    return total


def toy_uses_helper(items):
    result = list(items)
    return len(result)


class StaticBytecodeTests(unittest.TestCase):
    def test_static_tokens_have_expected_shape(self):
        tokens = static_bytecode_tokens(toy_loop)
        self.assertGreater(len(tokens), 0)
        for tok in tokens:
            self.assertTrue(tok.startswith("B:"))
            head, _, offset = tok[2:].rpartition("@")
            self.assertTrue(head.isupper() or "_" in head)
            self.assertTrue(offset.isdigit())

    def test_static_tokens_ordered_by_offset(self):
        tokens = static_bytecode_tokens(toy_loop)
        offsets = [int(t.rsplit("@", 1)[1]) for t in tokens]
        self.assertEqual(offsets, sorted(offsets))

    def test_static_opnames_unique_and_present(self):
        opnames = static_opnames(toy_loop)
        # uniqueness
        self.assertEqual(len(opnames), len(set(opnames)))
        # any loop body must contain *some* iteration opcode
        iter_like = {"FOR_ITER", "GET_ITER", "JUMP_BACKWARD"}
        self.assertTrue(
            iter_like.intersection(opnames),
            f"expected loop opcodes in {opnames!r}",
        )
        # any function with a return statement should have a return opcode
        return_like = {"RETURN_VALUE", "RETURN_CONST"}
        self.assertTrue(return_like.intersection(opnames))

    def test_static_call_targets_capture_global_calls(self):
        targets = static_call_targets(toy_uses_helper)
        # Heuristic walks LOAD_*/CALL pairs; ``list`` and ``len`` are both
        # loaded then immediately called, so both should surface.
        self.assertIn("list", targets)
        self.assertIn("len", targets)


class CodeBoundaryTests(unittest.TestCase):
    def test_boundary_captures_basic_code_object_fields(self):
        boundary = CodeBoundary.of(toy_loop)
        self.assertEqual(boundary.qualname, "toy_loop")
        self.assertEqual(boundary.argcount, 1)
        self.assertIn("items", boundary.varnames)
        self.assertIn("total", boundary.varnames)
        self.assertGreater(boundary.firstlineno, 0)
        self.assertTrue(boundary.filename.endswith(".py"))
        # No closure cells in a top-level function.
        self.assertEqual(boundary.freevars, ())

    def test_boundary_freevars_populated_for_closure(self):
        def outer():
            captured = 7

            def inner(x):
                return x + captured

            return inner

        inner = outer()
        boundary = CodeBoundary.of(inner)
        self.assertIn("captured", boundary.freevars)


class TraceCallTests(unittest.TestCase):
    def test_trace_records_opcode_events_for_target_only(self):
        result = trace_call(toy_loop, [1, -2, 3])
        self.assertEqual(result.return_value, 4)
        op_records = [r for r in result.records if r.kind == "opcode"]
        self.assertGreater(len(op_records), 0)
        # All opcode events must come from the target's qualname.
        self.assertEqual(
            {r.qualname for r in op_records},
            {"toy_loop"},
        )

    def test_trace_opcode_tokens_are_subset_of_static(self):
        result = trace_call(toy_loop, [1, 2, 3])
        static = set(static_bytecode_tokens(toy_loop))
        activated = set(result.opcode_tokens())
        self.assertTrue(
            activated.issubset(static),
            f"activated tokens not in static set: {activated - static}",
        )

    def test_trace_call_events_resolve_callable_names(self):
        # Force a call so the CALL event fires.
        result = trace_call(toy_uses_helper, [1, 2, -1])
        call_names = [r.qualname for r in result.records if r.kind == "call"]
        # The generator/sum dispatch resolves to *some* named callable.
        self.assertTrue(call_names, "expected at least one CALL event")
        self.assertTrue(all(isinstance(n, str) and n for n in call_names))

    def test_trace_without_opcodes_still_records_lines(self):
        result = trace_call(toy_loop, [1, 2, 3], opcodes=False)
        line_records = [r for r in result.records if r.kind == "line"]
        op_records = [r for r in result.records if r.kind == "opcode"]
        self.assertGreater(len(line_records), 0)
        self.assertEqual(op_records, [])


class PathwayDiffTests(unittest.TestCase):
    def test_empty_iterable_skips_loop_body_instructions(self):
        # Trivial AF: no items → loop body never enters.
        empty_trace = trace_call(toy_loop, [])
        empty_diff = pathway_diff(toy_loop, empty_trace)
        # Full AF: every branch fires.
        full_trace = trace_call(toy_loop, [1, 2, 3])
        full_diff = pathway_diff(toy_loop, full_trace)

        # Both diffs share the same static denominator.
        self.assertEqual(empty_diff.static_total, full_diff.static_total)
        # The full activation must cover at least as many instructions.
        self.assertGreaterEqual(
            full_diff.activation_ratio, empty_diff.activation_ratio
        )
        # And it should reach a substantial fraction of the body.
        self.assertGreater(full_diff.activation_ratio, 0.5)

    def test_negative_only_input_leaves_positive_branch_latent(self):
        # AF where x > 0 is never true: ``total += x`` body stays latent.
        trace = trace_call(toy_loop, [-1, -2, -3])
        diff = pathway_diff(toy_loop, trace)
        # The activation must miss *something* (the if-true branch).
        self.assertTrue(diff.static_only)
        self.assertLess(diff.activation_ratio, 1.0)

    def test_diff_shared_and_static_only_partition_static_set(self):
        trace = trace_call(toy_loop, [1, 2])
        diff = pathway_diff(toy_loop, trace)
        static = set(static_bytecode_tokens(toy_loop))
        self.assertEqual(set(diff.shared) | set(diff.static_only), static)
        self.assertEqual(set(diff.shared) & set(diff.static_only), set())


if __name__ == "__main__":
    unittest.main()
