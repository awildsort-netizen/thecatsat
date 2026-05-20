#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_policies import (
    POLICIES,
    accumulate,
    first_wins,
    latest_wins,
    replay_attention_queues,
    resolve,
    stack_unique,
    strict,
)


class AttentionPolicyTests(unittest.TestCase):
    def test_latest_wins_matches_current_decoder_behavior(self) -> None:
        self.assertEqual(latest_wins([]), None)
        self.assertEqual(latest_wins(["a"]), "a")
        self.assertEqual(latest_wins(["a", "b", "c"]), "c")

    def test_first_wins_is_opposite_of_latest(self) -> None:
        self.assertEqual(first_wins([]), None)
        self.assertEqual(first_wins(["a", "b", "c"]), "a")

    def test_accumulate_carries_every_hint_in_order(self) -> None:
        self.assertEqual(accumulate(["a", "b", "a"]), ("a", "b", "a"))

    def test_stack_unique_dedups_but_preserves_order(self) -> None:
        self.assertEqual(stack_unique(["a", "b", "a", "c"]), ("a", "b", "c"))

    def test_strict_refuses_to_drop_silently(self) -> None:
        self.assertEqual(strict(["only"]), "only")
        with self.assertRaises(ValueError):
            strict(["a", "b"])

    def test_resolve_reports_dropped_hints(self) -> None:
        rec = resolve("latest_wins", ["a", "b", "c"])
        self.assertEqual(rec.resolved, "c")
        self.assertEqual(rec.dropped, ("a", "b"))

        rec = resolve("accumulate", ["a", "b"])
        self.assertEqual(rec.resolved, ("a", "b"))
        self.assertEqual(rec.dropped, ())

    def test_policies_registry_lists_all_known(self) -> None:
        self.assertEqual(
            set(POLICIES),
            {"latest_wins", "first_wins", "accumulate", "stack_unique", "strict"},
        )

    def test_replay_attention_queues_groups_hints_by_next_literal(self) -> None:
        tokens = [
            "A:carry_pressure",
            "L:clause_pressure",       # queue: ("carry_pressure",)
            "L:influence_lift",        # queue: ()
            "A:carry_locality",
            "W:nested",
            "L:locality_probe",        # queue: ("carry_locality",)
            "R",
            "A:carry_bridge",
            "A:override_bridge",
            "L:bridge_walker",         # queue: ("carry_bridge", "override_bridge")
            "E",
        ]
        queues = replay_attention_queues(tokens)
        self.assertEqual(
            queues,
            (
                ("carry_pressure",),
                (),
                ("carry_locality",),
                ("carry_bridge", "override_bridge"),
            ),
        )

    def test_policies_diverge_only_when_queue_has_multiple_hints(self) -> None:
        queues = replay_attention_queues(
            ["A:a", "L:x", "A:b", "A:c", "L:y", "E"]
        )
        per_policy = {
            name: tuple(policy(q) for q in queues)
            for name, policy in POLICIES.items()
            if name != "strict"
        }
        # Single-hint queue: scalar policies pick "a"; accumulating
        # policies wrap it as a 1-tuple. Either way the payload contains "a".
        for resolved in per_policy.values():
            self.assertIn(resolved[0], ("a", ("a",)))
        # Multi-hint queue diverges.
        self.assertEqual(per_policy["latest_wins"][1], "c")
        self.assertEqual(per_policy["first_wins"][1], "b")
        self.assertEqual(per_policy["accumulate"][1], ("b", "c"))
        self.assertEqual(per_policy["stack_unique"][1], ("b", "c"))


if __name__ == "__main__":
    unittest.main()
