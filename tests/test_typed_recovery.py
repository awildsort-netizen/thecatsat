#!/usr/bin/env python3
"""Tests for ``typed_recovery``: lazy allocation and lazy recovery
across typed transformation boundaries.

The tests demonstrate the two consumption modes from the design note:

- a plan can complete without recovering an unused error-tagged
  object (the production climate forgets it);
- a recovery target can consume the same residue explicitly (the
  recovery climate lifts it back).
"""

from __future__ import annotations

import unittest

from typed_recovery import (
    CapacityTag,
    ErrorTag,
    Space,
    TaggedValue,
    plan_complete,
    recover_tagged,
    run_recovery,
    tag_error,
)


class TestTaggedValue(unittest.TestCase):
    def test_with_tag_does_not_mutate(self):
        v = TaggedValue(payload=7)
        w = v.with_tag(ErrorTag(reason="boom", origin="space-a"))
        self.assertEqual(v.tags, ())
        self.assertEqual(len(w.tags), 1)
        self.assertTrue(w.is_errored())
        self.assertFalse(v.is_errored())

    def test_error_tags_filters_to_error_only(self):
        v = TaggedValue(payload="x", tags=("plain", ErrorTag(reason="r")))
        errs = v.error_tags()
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0].reason, "r")


class TestTagError(unittest.TestCase):
    def test_tag_error_wraps_bare_payload(self):
        v = tag_error(42, reason="bad", origin="ingest")
        self.assertIsInstance(v, TaggedValue)
        self.assertEqual(v.payload, 42)
        self.assertTrue(v.is_errored())
        self.assertEqual(v.error_tags()[0].origin, "ingest")

    def test_tag_error_preserves_existing_tags(self):
        v = TaggedValue(payload=1, tags=("warm",))
        w = tag_error(v, reason="overflow")
        self.assertIn("warm", w.tags)
        self.assertTrue(w.is_errored())


class TestSpaceAdmission(unittest.TestCase):
    def test_admit_respects_capacity(self):
        s = Space("candidates", CapacityTag(limit=2))
        self.assertTrue(s.admit(1))
        self.assertTrue(s.admit(2))
        self.assertFalse(s.admit(3))
        self.assertEqual(len(s), 2)

    def test_admit_or_tag_returns_capacity_error_when_full(self):
        s = Space("candidates", CapacityTag(limit=1))
        s.admit(1)
        spilled = s.admit_or_tag(2)
        self.assertTrue(spilled.is_errored())
        self.assertEqual(spilled.error_tags()[0].reason, "capacity")
        self.assertEqual(spilled.error_tags()[0].origin, "candidates")
        self.assertEqual(len(s), 1)


class TestLazyPlanCompletion(unittest.TestCase):
    """A plan completes without rescuing residue the target does not need."""

    def test_target_reached_ignores_error_residue(self):
        s = Space("live", CapacityTag(limit=10))
        s.admit(TaggedValue(payload=10))
        s.admit(TaggedValue(payload=20))
        s.admit(tag_error(99, reason="conflict", origin="border-a"))

        def target(values):
            return sum(v.payload for v in values) >= 30

        self.assertTrue(plan_complete(target, s))
        self.assertEqual(len(s.errored_values()), 1)

    def test_forget_errors_drops_residue_when_target_done(self):
        s = Space("live", CapacityTag(limit=10))
        s.admit(TaggedValue(payload=10))
        s.admit(tag_error(99, reason="conflict"))
        self.assertEqual(s.forget_errors(), 1)
        self.assertEqual([v.payload for v in s.itervalues()], [10])

    def test_target_can_fail_without_touching_errors(self):
        s = Space("live", CapacityTag(limit=10))
        s.admit(TaggedValue(payload=5))
        s.admit(tag_error(100, reason="conflict"))

        def target(values):
            return sum(v.payload for v in values) >= 30

        self.assertFalse(plan_complete(target, s))
        self.assertEqual(len(s.errored_values()), 1)


class TestRecoveryClimate(unittest.TestCase):
    """A recovery operator can consume error-tagged residue explicitly."""

    def test_recover_tagged_passes_live_value_through(self):
        live = TaggedValue(payload=7)

        def recovery(_v):
            self.fail("recovery should not run on live values")

        self.assertIs(recover_tagged(live, recovery), live)

    def test_recovery_lifts_errored_value(self):
        v = tag_error(7, reason="conflict", origin="border-a")

        def recovery(tv):
            return TaggedValue(payload=tv.payload * 2)

        out = recover_tagged(v, recovery)
        self.assertIsNotNone(out)
        self.assertFalse(out.is_errored())
        self.assertEqual(out.payload, 14)

    def test_recovery_can_drop(self):
        v = tag_error(7, reason="unrecoverable")
        self.assertIsNone(recover_tagged(v, lambda _tv: None))

    def test_run_recovery_consumes_residue(self):
        s = Space("live", CapacityTag(limit=10))
        s.admit(TaggedValue(payload=1))
        s.admit(tag_error(2, reason="conflict", origin="border-a"))
        s.admit(tag_error(3, reason="conflict", origin="border-b"))

        def recovery(tv):
            if tv.error_tags()[0].origin == "border-a":
                return TaggedValue(payload=tv.payload + 100)
            return None

        recovered, dropped = run_recovery(s, recovery)
        self.assertEqual([v.payload for v in recovered], [102])
        self.assertEqual([v.payload for v in dropped], [3])
        self.assertEqual([v.payload for v in s.itervalues()], [1])

    def test_recovered_value_must_be_re_admitted_explicitly(self):
        s = Space("live", CapacityTag(limit=10))
        s.admit(tag_error(5, reason="conflict"))

        recovered, _dropped = run_recovery(s, lambda tv: TaggedValue(payload=tv.payload))
        self.assertEqual(len(s), 0)

        for v in recovered:
            s.admit(v)
        self.assertEqual([v.payload for v in s.itervalues()], [5])


class TestLazyAndRecoveryTogether(unittest.TestCase):
    """Same residue, two climates: production forgets, recovery consumes."""

    def _populate(self) -> Space:
        s = Space("live", CapacityTag(limit=5))
        s.admit(TaggedValue(payload=10))
        s.admit(TaggedValue(payload=20))
        s.admit(tag_error(99, reason="conflict", origin="border-a"))
        return s

    def test_production_climate_forgets_residue(self):
        s = self._populate()

        def target(values):
            return sum(v.payload for v in values) >= 30

        self.assertTrue(plan_complete(target, s))
        s.forget_errors()
        self.assertEqual(s.errored_values(), [])

    def test_recovery_climate_consumes_residue(self):
        s = self._populate()
        recovered, dropped = run_recovery(
            s, lambda tv: TaggedValue(payload=tv.payload - 99)
        )
        self.assertEqual([v.payload for v in recovered], [0])
        self.assertEqual(dropped, [])
        self.assertEqual(s.errored_values(), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
