# The Interpretation Sieve

A reframing note on what a gene string *is*, and how meaning survives
in this repo. Companion to `conversation_metabolism.md` and
`tests_as_activation_factors.md`.

## The claim

A gene string is **not** a stored program. It is a small piece of
compressed executable ecological potential — a bytecode of
*possibilities*. Many decompressions of it are syntactically legal.
The ones that actually exist are the ones that **survive** activation
under a climate. The other readings never run, so they never had a
referent.

> A gene's meaning is the residue of decompressions that survive
> execution pressure.

The sieve is the execution climate. It does not pick the "correct"
parse; it lets the parses that unfold stably keep their footing and
quietly erases the rest. Meaning is what is left.

## Bytecode as instruction potential

The token stream (`streamable_genes.py`, `bytecode_genes.py`) is a
compression artifact, not a script:

- `L:<operator>` — early header opcodes; the gene declares which
  operators it is willing to lean on. Eligibility is *advertised*
  here, not committed.
- `B:<OP>@<offset>` — body-level opcodes; a metabolic step keyed by
  operator and offset. The same `B:` token under two different
  climates is two different acts.
- `M:<i>` / motif markers — recurring substructures that, when reused
  enough, **fuse** into a compressed opcode (e.g. `B17`). Fused
  opcodes are local operators discovered by the system itself.
- `W:…R`, `D:i:…`, `A:` — windowing, decomposition, and activation
  markers, all of which constrain *which* decompressions can survive
  this turn.

Statically, the gene is a probability field over decompressions:
possible operator interpretations, possible call boundaries, possible
motif fusings, possible type windows. There is no canonical parse to
recover. There are only parses that the climate will and will not let
through.

## Operators as tendencies, not functions

In a stored-program view an operator is a function: same input, same
output, fully determined. In this repo an operator is a **tendency of
interpretation** — it raises the probability of certain
decompressions and lowers others, given the current concentration
field. Whether it actually fires, and what it actually means when it
fires, depends on:

- which channels are enriched above the uniform-prior floor,
- which eligibility set the climate has admitted,
- which motifs are already warm enough to be reused cheaply,
- which type window is currently collapsing or sharpening
  distinctions.

This is why `OperatorTrace.active` reads concentration state instead
of policy identity (see `tests_as_activation_factors.md`): the trace
records what the climate decompressed, not what the gene nominally
authorized.

## Why this does not become arbitrary soup

The risk of "meaning is whatever survives" is that anything goes.
What prevents that here is that survival is **expensive**.

- Tests / activation factors impose climates that punish unstable
  unfolding (see `tests_as_activation_factors.md`).
- Concentration fields decay; motifs that don't reuse pay full cost
  next time, so they don't fuse, so they don't persist.
- Solver metabolism (`sat_metabolism.py`) charges distance per
  incompatibility resolved, Hamming movement, revisits, entropy. A
  decompression that "works" but pays too much is selected against by
  the surrounding work, not by an explicit rule.

Adaptive bytecode is bounded by execution cost. The sieve is not
opinionated about which reading is right; it is opinionated about
which reading is cheap and stable enough to keep paying for itself.

## Activation climates, not invocations

A gene is dormant capability. It requires **local activation** to do
anything, and activation does not bypass eligibility — it only
realizes what eligibility already permitted.

Two climates matter, and they are not the same:

- **Production / execution climate** — narrow, biased toward already-
  warm operators, cheap. Most turns happen here. Recurring motifs
  compress into local operators and concentration priors;
  attention is inherited from the prior turn instead of rebuilt.
- **Development / reflection / mutation climate** — wider, rarer,
  exploratory. This is where new motifs get a chance to fuse, where
  type windows are renegotiated, where the gene's eligibility header
  may be edited.

The point of two climates is that we don't pay developmental cost on
every turn. The city stays mostly asleep.

## Type compression and windowing

Types in this repo are not static annotations. They are **persistent
compression attractors**: regions where many decompressions converge
on the same operator family, so the local sieve can collapse them
into one. A type window is the local/temporal extent over which a
distinction is being held open or being collapsed.

- A sharpened window distinguishes readings that the body of the gene
  otherwise treats as fungible.
- A collapsed window lets two formally distinct operators count as
  one for the duration of this activation.

Types are how the sieve buys cheap reuse without losing the option to
re-distinguish later.

## Puzzle ecology: fit, not fitness

The SAT layer is the worked example. Operators are **coastlines**,
not scalar scores. What matters is *fit* between a gene's
decompression and the instance's typed borders, contested seams, and
incompatibility landscape — not a one-dimensional fitness number.

Concretely, the SAT solver is doing **distance / geodesic
computation**: reduce incompatibility distance until a stable
embedding is found. Solver metabolism metrics (in `sat_metabolism.py`
and the benchmark CSVs) are diagnostic of how the sieve is paying:

- distance paid per incompatibility resolved,
- Hamming movement,
- revisits,
- motif reuse count,
- gene entropy.

A gene that wins by paying less distance per resolution has been
better sieved by this climate. A gene that wins by entropy collapse
without movement has overfit the climate.

## Observing the sieve: Python bytecode microscope

A practical note. Python's `dis` module is a useful **microscope** on
operator behavior — you can watch what a composed operator actually
executes turn by turn. But:

- Do *not* make our operators be Python bytecode. They are not.
  Our operators are tendencies of interpretation; CPython bytecode is
  one observation instrument we can point at them.
- CPython bytecode is **version-specific**. Anything we read off it
  is a measurement under a particular interpreter climate, not a
  truth about the operator. Treat microscope readings as evidence,
  not as definition.

## Map to this repo's primitives

| Idea | Primitive |
| --- | --- |
| Compressed instruction potential | `streamable_genes.py`, `bytecode_genes.py` token streams |
| Early opcodes | `L:` header tokens |
| Body-level opcodes | `B:<OP>@<offset>` tokens |
| Fused local opcodes | motif markers (`M:i`) that recur enough to compress into a `B…` slot |
| Sieve | climate + eligibility + concentrations applied during `Composer.iterate` |
| Survival pressure | tests as AFs + solver metabolism distance accounting |
| Tendencies | operators registered in `sat_composer.py`, gated by `_trace_append_operator` against the uniform-prior floor |
| Type windows | concentration channels held open or collapsed within a window |
| Distance computation | `sat_metabolism.py` metrics over composed runs |
| Microscope | `dis` / CPython bytecode inspection of composed operator bodies (read-only diagnostic) |

## Practical consequence

When you reach for "what does this gene mean," the honest answer is:
under which climate? The sieve is the operator of meaning. Build
climates deliberately, watch what survives, and let what doesn't
survive go without rescuing it. The genes that keep paying for
themselves are the ones we actually have.
