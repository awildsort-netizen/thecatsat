#!/usr/bin/env python3
"""Typed transformation boundaries: tagged values, error tags, capacity tags.

A tiny reference implementation of the architecture described in
``docs/typed_transformation_boundaries.md``. The point is to make the
semantics of lazy allocation and lazy recovery concrete enough to test:

- ``TaggedValue`` carries a payload plus zero or more typed tags.
- ``ErrorTag`` marks a value as errored-but-preserved.
- ``CapacityTag`` marks a space's budget on how many values it will hold.
- ``tag_error`` and ``recover_tagged`` are the seam operators between
  live space and errored-but-preserved space.
- ``Space`` is a minimal capacity-tagged local space. Admission and
  recovery happen at its boundary; the interior assumes its local
  shape.

This module is intentionally not a heap manager and not a framework.
It is the smallest amount of code that shows the boundary semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator


@dataclass(frozen=True)
class ErrorTag:
    """Mark on a value that a transformation could not complete.

    ``reason`` is a short string; ``origin`` names the space that
    failed to admit or transform the value. Both fields are advisory
    — recovery operators dispatch on them, but the production
    climate is free to ignore them entirely.
    """

    reason: str
    origin: str = ""


@dataclass(frozen=True)
class CapacityTag:
    """Budget tag on a local space.

    ``limit`` is the number of values the space will hold before
    admission becomes a seam decision (drop, spill, or trigger
    recovery). A ``None`` limit means unbounded — useful only for
    test scaffolding.
    """

    limit: int | None


@dataclass
class TaggedValue:
    """Payload plus a stack of tags accumulated across boundaries."""

    payload: Any
    tags: tuple[Any, ...] = ()

    def with_tag(self, tag: Any) -> "TaggedValue":
        return TaggedValue(self.payload, self.tags + (tag,))

    def error_tags(self) -> tuple[ErrorTag, ...]:
        return tuple(t for t in self.tags if isinstance(t, ErrorTag))

    def is_errored(self) -> bool:
        return any(isinstance(t, ErrorTag) for t in self.tags)


def tag_error(value: TaggedValue | Any, reason: str, origin: str = "") -> TaggedValue:
    """Seam operator: move a value into errored-but-preserved space.

    Accepts a bare payload or an existing ``TaggedValue``. The
    original value is not mutated; a new tagged value is returned
    so the caller can decide whether to keep it.
    """

    if not isinstance(value, TaggedValue):
        value = TaggedValue(value)
    return value.with_tag(ErrorTag(reason=reason, origin=origin))


def recover_tagged(
    value: TaggedValue,
    recovery: Callable[[TaggedValue], TaggedValue | None],
) -> TaggedValue | None:
    """Seam operator from errored space back toward a live space.

    The ``recovery`` callable inspects the value (typically its
    ``error_tags()``) and returns either a recovered ``TaggedValue``
    or ``None`` to drop. Values that are not errored pass through
    unchanged — recovery on a live value is a no-op, by design.
    """

    if not value.is_errored():
        return value
    return recovery(value)


class Space:
    """A capacity-tagged local space holding tagged values.

    Admission is the only place where capacity is checked. The
    interior of the space assumes its local shape and pays nothing
    for the assumption — methods like ``itervalues`` do not
    re-validate tags.
    """

    def __init__(self, name: str, capacity: CapacityTag) -> None:
        self.name = name
        self.capacity = capacity
        self._values: list[TaggedValue] = []

    def admit(self, value: TaggedValue | Any) -> bool:
        """Admit a value if capacity allows. Returns True on admit."""

        if not isinstance(value, TaggedValue):
            value = TaggedValue(value)
        if self._at_capacity():
            return False
        self._values.append(value)
        return True

    def admit_or_tag(self, value: TaggedValue | Any) -> TaggedValue:
        """Admit if there's room; otherwise return a capacity-error tag.

        This is the seam operator used when a caller wants to keep
        the rejected value as recovery material rather than drop it.
        The returned tagged value is *not* in this space — it is
        owned by whoever called ``admit_or_tag``.
        """

        if self.admit(value):
            return value if isinstance(value, TaggedValue) else TaggedValue(value)
        return tag_error(value, reason="capacity", origin=self.name)

    def itervalues(self) -> Iterator[TaggedValue]:
        return iter(self._values)

    def live_values(self) -> list[TaggedValue]:
        return [v for v in self._values if not v.is_errored()]

    def errored_values(self) -> list[TaggedValue]:
        return [v for v in self._values if v.is_errored()]

    def forget_errors(self) -> int:
        """Drop all error-tagged values. Returns the number dropped.

        This is what a production climate does at the end of a plan
        if the target was reached without consulting the residue.
        """

        before = len(self._values)
        self._values = [v for v in self._values if not v.is_errored()]
        return before - len(self._values)

    def drain_errors(self) -> list[TaggedValue]:
        """Remove and return all error-tagged values.

        Used by a recovery climate to consume residue out of the
        production space without forcing the production climate to
        deal with it.
        """

        errored = self.errored_values()
        self._values = [v for v in self._values if not v.is_errored()]
        return errored

    def __len__(self) -> int:
        return len(self._values)

    def _at_capacity(self) -> bool:
        if self.capacity.limit is None:
            return False
        return len(self._values) >= self.capacity.limit


def plan_complete(
    target: Callable[[Iterable[TaggedValue]], bool],
    space: Space,
) -> bool:
    """A plan is complete when the target predicate is satisfied by
    live values alone. Error-tagged residue does not block completion.
    """

    return target(space.live_values())


def run_recovery(
    space: Space,
    recovery: Callable[[TaggedValue], TaggedValue | None],
) -> tuple[list[TaggedValue], list[TaggedValue]]:
    """Drain error-tagged values and run a recovery operator on each.

    Returns ``(recovered, dropped)``. Recovered values are *not*
    re-admitted to ``space`` automatically — that is the caller's
    decision, and is itself a seam crossing.
    """

    recovered: list[TaggedValue] = []
    dropped: list[TaggedValue] = []
    for value in space.drain_errors():
        result = recovery(value)
        if result is None:
            dropped.append(value)
        else:
            recovered.append(result)
    return recovered, dropped
