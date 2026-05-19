#!/usr/bin/env python3
"""Tests for ``Composer.iterate``: bounded cycles as a first-class primitive."""

from __future__ import annotations

import unittest

from composer import Composer, FieldOperator, IterationStep


def _counter_op() -> FieldOperator:
    def _run(ctx):
        return {"next_n": ctx["n"] + 1}

    return FieldOperator(
        name="counter", inputs=("n",), outputs=("next_n",), run=_run,
    )


def _accumulate_op() -> FieldOperator:
    def _run(ctx):
        history = list(ctx.get("prev_history", []))
        history.append(ctx["next_n"])
        return {"history": history}

    return FieldOperator(
        name="accumulate",
        inputs=("next_n", "prev_history"),
        outputs=("history",),
        run=_run,
    )


def _emit_op() -> FieldOperator:
    """Always emits a fresh derived key — used to test stale-key dropping."""

    def _run(ctx):
        return {"emit": f"step:{ctx['next_n']}"}

    return FieldOperator(
        name="emit", inputs=("next_n",), outputs=("emit",), run=_run,
    )


class ComposerIterateTests(unittest.TestCase):
    def test_count_zero_returns_initial_context_unmodified(self) -> None:
        composer = Composer([_counter_op()])
        result = composer.iterate(
            ("next_n",), count=0, initial_context={"n": 5}
        )
        self.assertEqual(result.count, 0)
        self.assertEqual(result.steps, ())
        self.assertEqual(dict(result.context), {"n": 5})

    def test_negative_count_is_rejected(self) -> None:
        composer = Composer([_counter_op()])
        with self.assertRaises(ValueError):
            composer.iterate(("next_n",), count=-1, initial_context={"n": 0})

    def test_rename_map_carries_state_across_iterations(self) -> None:
        composer = Composer([_counter_op()])
        result = composer.iterate(
            ("next_n",),
            count=4,
            initial_context={"n": 0},
            rename_map={"next_n": "n"},
        )
        self.assertEqual(result.context["n"], 4)
        self.assertEqual(result.count, 4)

    def test_collect_records_per_step_snapshot(self) -> None:
        composer = Composer([_counter_op()])
        result = composer.iterate(
            ("next_n",),
            count=3,
            initial_context={"n": 10},
            rename_map={"next_n": "n"},
            collect=("next_n", "n"),
        )
        self.assertEqual(len(result.steps), 3)
        first, second, third = result.steps
        self.assertEqual(first.index, 0)
        # Snapshot is taken after composer.run but before rename, so ``n`` still
        # holds the prior step's value while ``next_n`` is the freshly produced
        # one — this is the most useful framing: both the input and output of
        # the iteration are visible.
        self.assertEqual(first.collected, {"next_n": 11, "n": 10})
        self.assertEqual(second.collected, {"next_n": 12, "n": 11})
        self.assertEqual(third.collected, {"next_n": 13, "n": 12})

    def test_step_key_stamps_the_iteration_index(self) -> None:
        composer = Composer([_counter_op()])
        result = composer.iterate(
            ("next_n",),
            count=3,
            initial_context={"n": 0},
            rename_map={"next_n": "n"},
            step_key="t",
            collect=("t",),
        )
        self.assertEqual([s.collected["t"] for s in result.steps], [0, 1, 2])
        self.assertEqual(result.context["t"], 2)

    def test_before_step_can_derive_inputs_from_prior_step(self) -> None:
        """``before_step`` runs after stale-drop and before the plan; use it
        for inputs that aren't simple key renames (e.g. ``[-1]`` of a list)."""

        composer = Composer([_counter_op(), _accumulate_op()])

        def before(ctx, index):
            # prev_history was carried forward by rename_map; before_step also
            # surfaces the most recent entry as a scalar for plans that want
            # it without indexing the list themselves.
            ctx["last_history"] = (
                ctx["prev_history"][-1] if ctx["prev_history"] else None
            )
            return {}

        result = composer.iterate(
            ("next_n", "history"),
            count=3,
            initial_context={"n": 0, "prev_history": []},
            rename_map={"next_n": "n", "history": "prev_history"},
            before_step=before,
            collect=("history", "last_history"),
        )
        self.assertEqual(result.context["prev_history"], [1, 2, 3])
        self.assertEqual(
            [s.collected["history"] for s in result.steps],
            [[1], [1, 2], [1, 2, 3]],
        )
        self.assertEqual(
            [s.collected["last_history"] for s in result.steps],
            [None, 1, 2],
        )

    def test_stale_per_step_outputs_are_dropped(self) -> None:
        composer = Composer([_counter_op(), _emit_op()])
        result = composer.iterate(
            ("next_n", "emit"),
            count=2,
            initial_context={"n": 0},
            rename_map={"next_n": "n"},
            collect=("emit",),
        )
        self.assertEqual(
            [s.collected["emit"] for s in result.steps],
            ["step:1", "step:2"],
        )
        self.assertEqual(result.context["emit"], "step:2")

    def test_rename_map_routes_a_running_accumulator(self) -> None:
        """A running sum needs the prior step's value under a different input
        key — the standard pattern for breaking a graph cycle into a counted
        iteration."""

        def _run(ctx):
            return {"sum": ctx["prev_sum"] + ctx["n"]}

        composer = Composer(
            [
                _counter_op(),
                FieldOperator(
                    name="sum_op",
                    inputs=("n", "prev_sum"),
                    outputs=("sum",),
                    run=_run,
                ),
            ]
        )
        result = composer.iterate(
            ("next_n", "sum"),
            count=4,
            initial_context={"n": 1, "prev_sum": 0},
            rename_map={"next_n": "n", "sum": "prev_sum"},
            collect=("sum",),
        )
        self.assertEqual([s.collected["sum"] for s in result.steps], [1, 3, 6, 10])
        self.assertEqual(result.context["prev_sum"], 10)

    def test_rename_destination_not_treated_as_stale(self) -> None:
        composer = Composer([_counter_op()])
        result = composer.iterate(
            ("next_n",),
            count=2,
            initial_context={"n": 0},
            rename_map={"next_n": "n"},
        )
        self.assertEqual(result.context["n"], 2)
        self.assertNotIn("next_n", result.context)

    def test_cycle_becomes_a_natural_loop(self) -> None:
        """A graph with a self-edge (n -> next_n -> n) is no longer a planner
        error; ``iterate`` turns it into a bounded loop by externalising the
        feedback through ``rename_map``."""

        composer = Composer([_counter_op()])
        plan = composer.plan(("next_n",), available_keys=("n",))
        self.assertEqual(plan.missing, ())
        result = composer.iterate(
            ("next_n",),
            count=10,
            initial_context={"n": 0},
            rename_map={"next_n": "n"},
        )
        self.assertEqual(result.context["n"], 10)

    def test_iteration_step_is_dataclass_with_index_and_collected(self) -> None:
        step = IterationStep(index=2, collected={"x": 1})
        self.assertEqual(step.index, 2)
        self.assertEqual(step.collected, {"x": 1})


if __name__ == "__main__":
    unittest.main()
