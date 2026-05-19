# Tests as Activation Factors

A reframing note on what a test *is* in this repo. Companion to
`conversation_metabolism.md`.

## The claim

Tests are **activation factors** (AFs). A test does not merely check
that a capability is present — it creates the ecological conditions
under which a compressed capability must **unfold** into an executable
pathway. Tests are ecological decompression environments.

The shorthand:

> **compressed motif + activation factors → executable pathway**

A streamable gene (`L:`, `W:…R`, `D:i:…`, `M:i`, `A:`) is the
compression: a small token stream that *stands in for* a larger
trajectory through operator space. On its own it does nothing. Run it
under a particular concentration field, climate window, and
eligibility set, and it decompresses into a sequence of actual
operator activations. That decompression is what produces work.

A test is the climate. It pins concentrations, fixes the eligibility
set, supplies inputs, and asks: *does this gene unfold into a stable
trajectory here?*

## Failing tests as unstable unfolding

A failing test is not (only) "the code is wrong." It is a statement
about the **gene/climate pair**: under this particular activation
climate, the current compression no longer unfolds stably. Either:

- the climate has shifted (eligibility narrowed, concentrations
  re-weighted, decay changed) and the old gene can no longer fix
  enough of its substructure from the local field, or
- the gene was always brittle and the new test simply happens to be a
  climate that exposes it.

In both cases the right question is not "patch the failure" but "what
climate is the test imposing, and is the gene we have the right
compression for that climate?"

## Different test suites are different species selectors

Two test suites that overlap on surface API but differ in setup,
fixtures, eligibility, or input distributions are **different
ecological pressures**. The same source code under suite A and suite B
becomes, effectively, two different species: the operators that stay
warm, the motifs that get reused, the decompression pathways that fire
are not the same. What survives in one climate may not survive in the
other. This is why "the tests pass" is not the same statement across
suites — each suite picks out a different population of viable
genes.

## Map to this repo's primitives

| Idea | Primitive |
| --- | --- |
| Compressed motif | streamable gene tokens (`streamable_genes.py`) |
| Activation factor | concentration field + climate window + eligibility set |
| Decompression | `Composer.iterate` running over a warmed `FieldContext` |
| "Did the unfolding stabilize?" | the test assertion + `OperatorTrace.active` |
| Climate-driven activation | `_trace_append_operator` in `sat_composer.py` thresholds `ctx["concentrations"]` against the uniform-prior floor — any enriched channel decompresses the corresponding traces, regardless of policy name |
| Species-by-suite | `experiments/sat_solver_metabolism.py` compares climates on the same instance: `gene_entropy` and `motif_reuse_count` already shift with the activation climate, not the policy label |

## Practical consequence

When you add a test, you are not just asking the code a question — you
are *building a climate* and selecting for the genes that survive it.
Choose the climate deliberately: which channels do you want enriched,
which operators eligible, what window? The test will preserve
whatever decompresses cleanly under those conditions and quietly
erode whatever doesn't.

This is also why `OperatorTrace.active` now reads concentration state
rather than policy identity — the trace's job is to record what the
climate actually decompressed, not what the policy nominally
authorized.
