# Representational bubbles: the next target after the transform litmus

A design note, **not yet an experiment**. This file names the framing
the user proposed after PR #9's negative-or-neutral result, sketches
how it connects back to the existing geometry stack
(`flattening_probe`, `riordan_probe`, `tangent_lift_probe`,
`transform_litmus`) and the `parser_evolver` work, and pins down a
single small next empirical target. It is deliberately doc-heavy and
code-light: PR #9 already gave us the right reason to slow down.

## The seed idea

A **representational bubble** is a small local address space inflated
around persistent collision or strain so that distinctions which were
previously colliding can breathe — i.e. so that two states which the
current coordinates were forced to call "the same point" get their own
nearby-but-distinct addresses, and a small protected interior of legal
moves opens up between them.

A bubble has four things going for it that a raw point does not:

- **local expansion** — what used to be one cell is now a neighborhood,
- **new interior** — the neighborhood contains moves that didn't exist
  before, not just relabellings of moves that did,
- **protected navigable volume** — the moves inside the bubble do not
  immediately leak back into the collision they were inflated to
  resolve,
- **a typed boundary** — the bubble has an edge that the rest of the
  representation can refer to without entering it (this is what makes
  it composable, not just a local fix).

A *concept*, in this framing, is a bubble that has stabilized: it has
an address, an interior, an edge, and the rest of the system has
learned to route through it. A *singularity* is a bubble trying to
nucleate but failing to find a stable interior — pressure without
volume. *Persistent strain* (in the `flattening_probe` / `riordan_probe`
sense — residual unsat that the search cannot dissolve) reads, under
this framing, as insufficient interior volume: the system is being
asked to host a distinction it has no room for.

## Relationship to the tangent lift (PR #8)

PR #8 is the cleanest example of a working bubble at continuous-math
scale. The scalar `tan(x)` blows up at `π/2` because in one dimension
there is nowhere for the singularity to *go*: the chart is forced to
report infinity. Lifted into `(sin x, cos x)`, the same point becomes
the typed boundary condition `cos x = 0`. The lift creates exactly the
four things above:

- local expansion (1-D → 2-D around the bad point),
- new interior (legal nearby states with finite, well-defined values),
- protected navigable volume (`max_finite_difference` drops by ~17
  orders of magnitude near the bad point, per `tangent_lift_probe`),
- a typed boundary (`cos x = 0` is a *named* edge, not a runtime
  explosion).

So PR #8 is not just "singularity is a coordinate artifact." It is also
the working positive control for bubble nucleation: a successful chart
turned a singularity into a typed boundary *and* preserved a stable
off-boundary interior the solver can keep navigating in. That is the
shape we want the SAT analogue to eventually have.

## Relationship to the transform litmus (PR #9)

PR #9 is the corresponding *negative* control. On the existing 22-instance
suite, `transform_litmus` measured whether the spectral / Pascal /
signed-Pascal / Sierpinski views in PR #7 produce localization
without solving — residual strain collapsing onto a small named set
even when the view didn't finish the instance. The honest finding:

- the two interesting middle verdicts — `localized_but_unstable` and
  `moved_singularity` — are **empty** on this suite,
- the dominant non-trivial verdict is `amplified_pathology` (31/88),
- `resolved_to_boundary` is real but rare (2/88).

Under the bubble framing, that result has a sharper reading than "the
transforms mostly don't help." It says: **the transforms in PR #7 are
not designed to nucleate bubbles**. They either solve outright (rare),
amplify the collision (common), or do nothing measurable. None of them
is built to take a persistent collision, inflate a local interior
around it, and hand the solver back a chart with one extra
addressable cell.

That gives us a concrete next target rather than just a research
mood: the next transform we add to the geometry stack should be
designed *specifically* to produce non-empty `localized_but_unstable`
or `localized_but_stable` verdicts — to take a collision and try to
give it room — and the litmus from PR #9 is already the instrument
that will measure whether it worked.

## The local collision test

A bubble seed is, operationally, a place where two states are *close
in current address* but *far in observed behavior*. We don't have to
build a bubble to detect one — we can ask:

- pick a pair of variables, clauses, or operator slots `(a, b)`,
- measure their **address distance** in the current view (the cheap
  coordinate-space distance the view defines: `flattening_probe` uses
  variable indices, `riordan_probe` uses the transformed coordinate
  vector, parser_evolver uses operator/span coordinates),
- measure their **behavioral distance** (how often flipping `a`
  changes the residual on clauses involving `b` vs. not — i.e. the
  strain off-diagonal),
- flag the pair as a **collision seed** when address distance is low
  and behavioral distance is high.

A seed is not yet a bubble. It is a place where the current
representation is being asked to do work it cannot — a candidate site
for inflation. The same test reads sensibly in parser_evolver: two
operator slots that occupy nearby positions but produce divergent
hallucination signatures are candidate seeds for a new typed
sub-address.

## Bubble lifecycle (sketched, not implemented)

1. **Seed** — a collision is detected (low address distance, high
   behavioral distance). This is the only step the litmus + a small
   pairwise scan can already implement on top of PR #9.
2. **Inflate** — a small local coordinate expansion is proposed
   around the seed. This is the transform-design step PR #9 said
   nothing currently does on this suite.
3. **Stabilize** — the inflated region is checked for an interior:
   are there moves inside the new address space that don't
   immediately leak back into the original collision? This is where
   `localized_but_stable` would come from.
4. **Route** — the rest of the system learns to address the bubble
   through its typed boundary, not by entering it. This is the
   composability step; it is the SAT/parser analogue of "concept."
5. **Merge / prune** — adjacent bubbles either fuse (sharing
   boundary) or one is shown to be redundant and collapsed back.
   We have no instrument for this yet.
6. **Plaque risk** — a bubble that does not stabilize, does not get
   routed, and is not pruned becomes **semantic plaque**: a region
   that holds address space without doing work, polluting the
   distance metric and degrading later collision tests. This is the
   exact failure mode named in
   [`parser_evolver/docs/hallucination_geometry.md`](../parser_evolver/docs/hallucination_geometry.md);
   we are recovering it here as the natural failure mode of the
   lifecycle.

## Connections to parser_evolver

The bubble framing is not SAT-specific. It maps cleanly onto the
artifacts the parser_evolver work already produces:

- **masked slots** — slots whose addresses are reserved by the
  attractor form but not yet filled by evidence are bubbles in the
  *seed* state: address allocated, interior not yet inflated.
- **hallucination regions** — the local-strain regions in
  `hallucination_geometry.md` are exactly the persistent-collision
  sites the local collision test is meant to surface. Hallucination
  is a bubble failing to stabilize.
- **remembered absence** — when the system records "no evidence here,
  but the shape requires something" it is keeping a bubble in
  *inflated-but-unstabilized* state instead of collapsing the address
  back to nothing. That memory is the thing that lets the bubble
  later be filled rather than re-nucleated from scratch.
- **archons** — load-bearing operator clusters can be read as
  *routed* bubbles: stable interiors that the rest of the system
  reaches through a fixed typed boundary. An archon that loses its
  boundary type is a bubble decaying into plaque.
- **semantic plaque** — the named failure mode in step 6 above. The
  bubble framing makes plaque a *predicted* outcome of the lifecycle,
  not a one-off pathology.
- **browser_oracle as developmental trace** — the developmental-trace
  model treats the browser session as a record of how the system came
  to know what it knows. In bubble terms, the trace is the history of
  which collisions nucleated which bubbles and which routes
  stabilized — i.e. the lifecycle log, replayed.
- **noncoding operator material** — operator material that is present
  but doesn't currently fire is, in this framing, *boundary tissue*:
  it holds the edge of stabilized bubbles open so that nearby
  collisions can find their seed without immediately running into a
  hard wall. We should not assume noncoding == waste.

## Conceptual type sketch (non-binding)

If the next experiment ends up wanting types, these are the shapes
the doc is pointing at. They are **not** to be added to
`geometry/__init__.py` until something actually consumes them; this
section is here so the next PR doesn't have to re-derive the
vocabulary.

```python
# Sketch only — do NOT add to geometry/__init__.py yet.

@dataclass(frozen=True)
class CollisionSeed:
    """A (low address distance, high behavioral distance) site."""
    address: tuple[int, ...]       # coordinates in the current view
    address_distance: float        # cheap metric distance
    behavioral_distance: float     # strain / divergence between paired items
    pair: tuple[int, int]          # the two items (vars, clauses, slots)
    view_name: str                 # which CoordinateView surfaced it


@dataclass(frozen=True)
class AddressBubble:
    """A local expansion proposed around a seed."""
    seed: CollisionSeed
    interior_size: int             # how many new addressable cells
    boundary_type: str             # named edge (e.g. "cos x = 0")
    routed: bool                   # whether the rest of the system addresses
                                   # this bubble through its boundary


@dataclass(frozen=True)
class BubbleLitmus:
    """Per-(view, seed) reading: did the inflation give the seed room?"""
    seed: CollisionSeed
    bubble: AddressBubble | None
    verdict: str                   # e.g. "localized_but_stable",
                                   # "localized_but_unstable",
                                   # "no_interior", "plaque_risk"
```

These are deliberately tiny and parallel to `StrainLocalization` /
`LitmusReading` in `geometry/transform_litmus.py`. If we end up
implementing this, the existing summarize/association machinery from
PR #9 should be reusable almost verbatim — the only new piece is the
inflation step itself.

## Pacing guardrail

It is tempting, after a clean framing, to ship a new metric, a new
probe, and a new driver in one PR. PR #9 is the reason not to.

The empirical next step is **one** small transform designed
*specifically* to create local interiors around a `CollisionSeed` —
not more metrics, not a new harness, not a generalized lifecycle
runner. The litmus from PR #9 already measures whether the transform
worked. If the new transform's verdict column has any non-empty
`localized_but_stable` (or `localized_but_unstable`) rows, the
framing earned its next PR. If it does not, we have learned something
just as important.

Concretely, the next PR should add:

- one new `CoordinateView` (call it e.g. `bubble_seed_view`) that
  takes a `ProbeResult` from the raw view, finds the top-1 collision
  seed by the local collision test, and projects coordinates so that
  the seed pair gets one extra addressable cell between them,
- an experiment driver that reuses the existing 22-instance suite
  and prints the same litmus association table as PR #9,
- the smallest test that pins the determinism contract (same shape
  as `tests/test_transform_litmus.py`).

That is the scope. Anything more is plaque.
