# Conversation Metabolism

## The problem

Current LLM-style systems wake the whole city for every request. Each
turn pulls the full context through every layer of the model — the same
billions of parameters are excited whether the user is asking for a
one-line rename or a novel architecture. Recurring motifs (the same
clarifying question, the same setup ritual, the same kind of bug) are
re-paid in full every time.

That is metabolically expensive. Tokens, latency, and context pressure
all scale with the size of the neighborhood we light up, not with the
size of the work the user actually needed.

## The aim

We want **local conversational metabolism**: recurring motifs should
compress into local operators or concentration priors that reactivate a
nearby ecology cheaply. Only the rarer, harder turns should pay for
broad activation.

Two climates, not one:

- **Execution climate** — cheap, narrow, biased toward the operators
  the local context has already warmed. Most turns happen here.
- **Development / reflection climate** — rarer, wider, exploratory.
  This is where the system grows new capability or rewires its priors.

Activation factors decide which climate a turn enters. The execution
climate must never be allowed to silently widen into the development
climate, and the development climate must never bypass eligibility.

## How the repo's primitives map to the idea

Nothing here is new infrastructure — the pieces already exist:

| Idea | Primitive |
| --- | --- |
| Streamable record of attention inheritance | `streamable_genes` (`L:`, `W:…R`, `D:i:…`, `M:i`, `A:`) |
| Local concentration prior over operators | `concentration_from_gene_tokens` → field dict |
| Closed-loop motif reinforcement | `run_rounds` (sample → harvest trace → warm next field → decay) |
| Bias without veto | `sample_provider` (concentration biases, eligibility gates) |
| Execution vs. development climate switching | `climate_tokens` + `window_scale` on `run_rounds` |
| Composition of warmed operators into work | `Composer.iterate` |
| Discovery of new operators worth promoting | reflection-discovery tests / `experiments/reflection_discovery.py` |
| Lock-in diagnostic for the loop | `RoundResult.fixation_index` |

## Why this could reduce cost

- A warmed local field selects from a small set of eligible operators
  rather than the global library. Sample paths are short and cheap.
- Recurring motifs become `D:i:…` / `M:i` backreferences — one symbol
  stands in for an entire substructure.
- Decay is a budget: motifs that stop paying off cool, freeing the
  field for new evidence. The metabolism stays bounded.
- Development-climate spikes (a `W:dev` window with a high
  `window_scale`) wake dormant capability only when needed. The rest of
  the city stays asleep.

## What `fixation_index` is for

The closed loop has a failure mode: a small initial bias amplifies into
near-total lock-in, and the system stops exploring. `fixation_index =
max(distribution.values()) / total` on each `RoundResult` is the
cheapest possible alarm — a single number per round that flags when the
local ecology has collapsed onto one path. Rising fixation across
rounds is the signature of the city forgetting how to wake any other
neighborhood; flat fixation is healthy ecology. Whether to respond by
raising decay, widening eligibility, or injecting a development climate
is the experimenter's call — the diagnostic only names the symptom.

## Boundaries

- Eligibility is the hard gate. Climates and concentrations bias, they
  never veto or summon.
- Activation factors must not bypass eligibility. A roaring `W:dev`
  climate over an ineligible operator still produces no choices of it.
- This is an ecology, not a framework. Resist building a metrics layer,
  a climate scheduler, or a policy registry around these primitives —
  the value is in their smallness.
