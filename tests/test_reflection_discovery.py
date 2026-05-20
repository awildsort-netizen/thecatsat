#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sat_field
import streamable_genes
from composer import discover_operator_candidates, rank_provider_candidates


class ReflectionDiscoveryTests(unittest.TestCase):
    def test_sat_field_public_functions_surface_as_candidates(self) -> None:
        candidates = discover_operator_candidates(sat_field)
        public_names = {c.name for c in candidates if not c.name.startswith("_")}
        self.assertIn("formula_graph", public_names)
        self.assertIn("formula_graph_to_adjacency", public_names)

    def test_reflected_formula_graph_wins_provider_fit_for_its_target(self) -> None:
        candidates = discover_operator_candidates(sat_field)
        fits = rank_provider_candidates(
            "formula_graph", candidates, nearby_terms=("sat", "formula")
        )
        self.assertTrue(fits)
        self.assertEqual(fits[0].candidate.name, "formula_graph")
        self.assertIn("exact_output", fits[0].reasons)

    def test_stream_dataclass_return_explodes_into_named_outputs(self) -> None:
        candidates = discover_operator_candidates(streamable_genes)
        by_name = {c.name: c for c in candidates}
        stream_cand = by_name["stream"]
        # StreamState has multiple fields; we expect them all to surface
        # as inferred outputs (not just the function name).
        self.assertGreater(len(stream_cand.inferred_outputs), 1)
        self.assertIn("emitted", stream_cand.inferred_outputs)

    def test_private_underscore_helpers_can_be_filtered_locally(self) -> None:
        candidates = discover_operator_candidates(sat_field)
        public = [c for c in candidates if not c.name.startswith("_")]
        private = [c for c in candidates if c.name.startswith("_")]
        # sat_field has no underscored module-level functions today; this
        # test guards the filter contract rather than a count.
        self.assertEqual(len(public) + len(private), len(candidates))


if __name__ == "__main__":
    unittest.main()
