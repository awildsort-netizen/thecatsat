# Typed Transformation Boundaries

An architecture note on what *types* are for in this repo, and how
error/capacity tags let allocation and recovery stay lazy. Companion
to `interpretation_sieve.md` and `tests_as_activation_factors.md`.

## The claim

Types are not validators. Types are **transformation boundaries**.
A type marks a place where one local space hands an object to a
different local space, and the handover is the only point where the
shape of the object actually has to be checked or rewritten.

> A type is the seam between two ways of holding an object. The
> seam owns the conversion; the interior of each space does not.

Inside a local space, code can assume the local shape and pay
nothing for the assumption. Between spaces, a typed boundary names
the operator that moves an object across — including the operator
that moves an object out of a *failed* state without forcing the
failure to be dealt with right now.

## Local spaces and heap-indexed allocation

A useful analogy: a large heap, with many local spaces each holding
indices into that heap. Each space is a typed view — different
spaces interpret the same underlying index differently, and the
typed border defines how an object moves from one view to another.

- **Local space type** — a set of indices treated under one set of
  operator assumptions (e.g. "warm motif slots", "candidate
  assignments", "incompatibility frontier").
- **Cross-space operator** — a function defined exactly at the
  seam: given an object in space A's interpretation, hand it to
  space B's interpretation. This is where the cost is paid, once.
- **Heap as substrate** — the underlying storage is shared; the
  view is what differs. Moving objects between spaces is mostly
  re-tagging, not copying.

The same object can carry multiple type tags as it crosses
boundaries. The tags are the record of which spaces have admitted
it and under what assumptions.

## Error tags as typed operators

An error is also a type. When a transformation cannot complete,
the object does not vanish and does not raise an exception that
unwinds the whole plan. It is **tagged** with an error type and
left in place.

- `tag_error(obj, reason)` is a typed operator: it moves the
  object from "live" space into "errored-but-preserved" space.
- A recovery function is a typed operator in the other direction:
  it accepts objects in errored space and tries to lift them back
  to a live space, possibly via a different conversion path.
- Multiple recovery operators can specialize on different error
  tag families. The dispatch is by tag, not by stack trace.

The important property: the error tag preserves the object as
**potential recovery material**. It does not demand the system
deal with the failure right now. Whether the system deals with it
later depends on whether downstream goals need this object.

## Capacity tags and lazy allocation

A capacity tag describes a resource or budget constraint a space
is willing to hold: how many indices it will accept, how much
concentration it will track, how many error-tagged objects it
will keep around before triage.

- Capacity tags let allocation be **lazy**: you don't materialize
  an object until a space with capacity for it admits it.
- Capacity tags let preservation be **lazy**: a space can keep
  error-tagged objects without committing to recovering them, up
  to the tagged budget.
- When capacity is exceeded, the seam operator decides what to do
  — drop, spill to a coarser space, or trigger recovery.

Capacity is also a typed boundary. "This space has room" and
"this space is saturated" are different types; admission across
the seam respects the difference.

## Lazy recovery: forget what the target does not need

The architectural pivot:

> If a plan's target can be completed without an object, the
> object's error tag may be **forgotten**. The object is not
> rescued, not retried, not logged as a failure. It is left as
> recovery material for whoever asks.

This means error handling is not a control-flow concern that has
to be threaded through every operator. It is an **allocation
question**: did the target need this object to complete? If no,
the error tag is inert and the plan proceeds. If yes, a recovery
operator is invoked at the seam where the object would have been
consumed.

Two consumption modes for error-tagged objects:

1. **Plan completion** — the plan finishes without consulting the
   tagged object. The tag is harmless; the object decays with the
   local space's capacity policy.
2. **Recovery thread / operator** — a separate consumer reads
   error-tagged objects out of the space, runs a recovery typed
   operator on them, and either lifts them back into a live space
   or drops them. This is a *different* climate from the
   production climate, in the sense of `interpretation_sieve.md`.

Recovery, like development, is a rarer climate. The production
climate stays cheap because it does not have to negotiate with
errors that don't matter to it.

## Relation to Composer eligibility and planning

The Composer already operates at a typed seam: eligibility decides
which operators a gene is *willing* to lean on, and activation
decides which of those actually fire. Typed transformation
boundaries extend this in a few places:

- **Eligibility as type admission** — an operator becomes
  eligible when its input space admits the current object's type.
  Operators that would error on an object of the wrong type are
  simply not eligible, not invoked-and-caught.
- **Operator outputs as typed objects** — a composed step can
  emit a live object *or* an error-tagged object. Both are
  legitimate outputs; the next seam decides whether to consume.
- **Plan completion as target-reach** — the Composer's notion of
  "done" is target completion, not absence of errors. Tagged
  objects that the target does not depend on are correctly
  ignored.
- **Concentration channels are capacity-tagged spaces** — they
  hold a budgeted view over the bytecode/motif substrate. When
  channels saturate, that *is* a capacity boundary, and the
  decay/eviction policy is the seam operator.

This keeps types where they belong in this repo: at composer and
operator seams, not as repeated defensive checks inside kernels.
The interior of a kernel trusts its local space; the seam owns
the conversion and the failure handling.

## Relation to puzzle ecology and climates

The SAT layer makes the same shape concrete:

- **Typed borders of an instance** — clause groups, variable
  neighborhoods, and contested seams are local spaces with their
  own interpretation of which assignments matter.
- **Incompatibility distance** — moving an assignment across a
  border is a typed transformation; the metabolism metrics
  (`sat_metabolism.py`) record what that move cost.
- **Error-tagged candidates** — a partial assignment that
  conflicts on one border but is fine elsewhere can be tagged and
  set aside without forcing the solver to abandon it. A later
  recovery pass (under a wider climate) may revive it; if the
  target is reached without it, it decays.
- **Concentration and activation climates** — the production
  climate runs cheap and ignores most error-tagged residue; the
  reflection/recovery climate visits residue on purpose.

The result: the puzzle ecology gains a graceful failure surface
without paying for it on every turn. Errors become a kind of
inventory, not a kind of interruption.

## What this is not

A few things this architecture explicitly does **not** want:

- It is not a heap manager. The substrate is whatever the rest of
  the system already uses; the contribution is the typing of the
  views, not the storage.
- It is not exception-replacement-as-a-framework. The point is
  *fewer* defensive checks, not a new exception hierarchy.
- It is not eager recovery. Forgetting an error tag whose object
  the target did not need is the **expected** behavior, not a
  leak.
- It is not a static type system. The tags are runtime objects
  that move with values; they describe what the climate has
  admitted, in the spirit of `interpretation_sieve.md`'s type
  windows.

## Map to this repo's primitives

| Idea | Where it lives or would live |
| --- | --- |
| Local space type | a concentration channel or eligibility set, viewed as an admitted-objects space |
| Typed seam operator | composer step that admits / rejects / converts an object between spaces |
| Error tag | runtime tag on a value carrying reason + originating space |
| Recovery operator | typed function from `(error-tagged object)` → `live object \| dropped` |
| Capacity tag | budgeted size on a channel / space (concentration cap, eligibility width) |
| Lazy allocation | an object is only materialized when an admitting space has capacity |
| Forgettable error | a tagged value the target did not depend on; decays with capacity |
| Recovery climate | a reflection-style pass that consumes error-tagged residue |

## Practical consequence

When something fails inside an operator, the first question is not
"how do we handle this." It is: **does the target need this
object?** If no, tag it and move on. If yes, send it across the
recovery seam. Most failures should be answerable by the first
case; the second case is what the recovery climate is for.

Keep the seams thin and the interiors trusting. The types are
already doing the work at the boundary.
