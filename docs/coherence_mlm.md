# Coherence-Regularized MLM Probe

## Framing

The conjecture being tested (informally): when a network learns an MLM
task, *being right about the masked token* is only one of the things
training pressures it toward. The other thing — the one this probe is
asking about — is whether the model arrives at the right token through
a *coherent* internal trajectory or a turbulent one.

The user's framing maps roughly like this:

| Concept                                  | MLM proxy used here                                     |
|------------------------------------------|---------------------------------------------------------|
| timing / settling-time field             | iterative-refinement `settle_steps`                     |
| rhythmic coherence under perturbation    | low cosine drift in masked-position hidden state under token-preserving noise |
| activation turbulence between layers     | mean L2 between hidden(L) and hidden(L-1), normalized   |
| attention drift / non-settling search    | mean Shannon entropy of attention rows                  |
| divergence / strain                      | spread of any of the above; jump in drift post-train    |
| training objective augmentation          | `MLM_CE + lambda * perturbation_drift`                  |

We are not claiming this regularizer is novel or that the proxies are
the *right* proxies. We are asking the smaller, testable question:
on a tiny synthetic MLM, does adding a coherence term move the proxies
in the predicted direction without destroying the task loss?

## What the probe does

- Generates a synthetic n-gram language (24 tokens, length 10) with a
  few "attractor" transitions so MLM has real structure to learn.
- Builds a tiny encoder: 2 transformer-ish layers, 2 heads, hidden 16.
  The encoder backbone is **frozen at random init**; only the token
  embedding (which is tied to the output projection) is trained. This
  keeps gradients tractable in numpy and isolates the experiment to
  *how the model represents its inputs*, not *how it propagates them*.
- Runs two training loops on the same initial weights, same masks,
  same seeds:
  - **baseline**: standard MLM cross-entropy on masked positions.
  - **coherence**: `MLM_CE + lambda * perturbation_drift`, where
    `perturbation_drift` is the mean cosine distance between the
    masked-position hidden state under the original masked input and
    under a token-preserving perturbation of one unmasked position.
- Logs at every step: MLM loss, perturbation drift, layer jitter,
  attention entropy, time-to-settle from iterative refinement.

Both gradients are computed by centered finite differences over the
embedding rows that the current batch actually uses. With a 25×16
embedding and a tiny batch this is fast enough for CPU; on the
configuration in `experiments/coherence_mlm.py` it runs in well under
a minute end-to-end.

## How to read the output

The driver prints two per-step tables (one per run) plus a summary
that averages the last third of training, then a held-out pre/post
drift probe on a fresh masked batch. Both runs use identical
masks/seeds, so the *delta* in held-out drift is what tells you the
regularizer is doing anything beyond noise.

If the conjecture has any teeth at all on this tiny model, the
coherence run should:

- not improve (and probably slightly worsen) MLM loss,
- reduce post-training perturbation drift relative to pre-training,
- reduce layer jitter,
- leave attention entropy roughly unchanged (the backbone is frozen,
  so there's a ceiling on how much attention smoothness can move).

A run that flat-lines all proxies just means the regularizer was too
weak (try a larger `lambda_coh`) or that this particular model is
already incoherent enough that single-position perturbations are
already maximally informative (try fewer mask positions, or a deeper
backbone).

## Limits of this probe

- The backbone is frozen, so the "internal propagation" we're
  regularizing is propagation *through a fixed random map*. A real
  test would push gradients through trainable attention.
- The perturbation we use is uniform token replacement on a single
  unmasked position. A more careful version would replace with a
  token that is *Markov-equivalent* (matches transition statistics)
  so the perturbation truly preserves the local distribution.
- All four coherence proxies are slightly different things. We treat
  them as a small panel rather than a single measurement. If you
  wanted one number, `perturbation_drift` is the one being directly
  optimized; the others are read-only.
- We do not claim this generalizes to real LLMs. The whole point is a
  modifiable, conceptually legible knob to play with the framing.

## Running

```
python experiments/coherence_mlm.py                # smoke (~25s)
python experiments/coherence_mlm.py --mode heavy   # ~3 min
```

Smoke writes `coherence_mlm_log.jsonl`. Heavy writes
`coherence_mlm_log.jsonl` (per-step and per-validation rows) and
`coherence_mlm_summary.csv` (one row per run). No dependencies beyond
numpy.

## Heavy mode — what it adds and why

The smoke run is fast enough to be a sanity check but small enough that
no signal is more than a hint. Heavy mode pushes the setup further:

- larger task (V=33, seq_len=12, num_sequences=192, batch=10),
  3-lambda × 2-seed sweep, 15 steps per run,
- a held-out validation split (25%) so all `step_to_*` and
  `early_stop_step` metrics are measured under a *matched* probe,
- every coherence run is paired with a same-seed/same-init baseline,
- per-step instrumentation of wall time and a forward-call counter,
  so comparisons can be normalized by *compute used* rather than by
  *steps taken*.

Heavy mode is still numpy-only and still uses centered finite-difference
gradients over the embedding rows; we just point them at a model big
enough that one training step costs ~0.8 s (baseline) or ~2.7 s (with
the coherence term, which adds a second finite-diff pass through the
same embedding rows).

## Early convergence / cycle-savings framing

The follow-up question being asked: can the coherence framing be used
as a *training-time early-convergence proxy* — i.e. does adding the
regularizer (or just watching the proxies) save cycles?

We track the following per-run metrics so this is a question we can
answer with numbers rather than vibes:

- `wall_time_s` per run and cumulative time-to-threshold,
- `forward_calls`, a compute proxy invariant to wall noise,
- `step_to_half_mlm` / `step_to_half_drift`: first held-out step at
  which val MLM loss / val drift has crossed halfway between its first
  and last observed values. Calibration-free convergence proxy.
- `step_to_mlm_eps05`: first step at which val MLM loss is within 0.05
  of its final value (plateau-reached proxy).
- `auc_train_mlm` and `auc_val_drift`: trapezoidal areas under the
  train-loss and held-out-drift curves. Lower = the run spent less of
  its compute in the high-loss / high-drift region.
- `early_stop_step`: step at which val MLM loss was best (matched
  early-stopping criterion across all runs).

### What would count as evidence

The strong claim — "coherence regularization gives early MLM
convergence" — would require, at matched compute and matched seeds:

- coherence runs reaching the same val MLM loss in *fewer forward
  calls* (not just fewer steps, since the regularizer makes each step
  more expensive), or
- coherence runs hitting a lower `auc_train_mlm` at the same step
  budget, or
- the held-out drift trajectory diverging well before MLM loss does,
  in a way that lets you stop the baseline early without sacrificing
  MLM loss.

A weaker, more useful claim — "drift itself is a stable training-time
proxy" — would require:

- val_drift moving smoothly and monotonically with lambda,
- the ordering by val_drift agreeing across seeds.

### What the current heavy run shows

From a representative heavy run (15 steps, seeds={1,2}, lambdas={0, 5, 20},
total wall time 3m04s):

| lambda | wall (s) | val_mlm | val_drift | half_mlm | half_drift | AUC_mlm | AUC_drift | early_stop |
|-------:|---------:|--------:|----------:|---------:|-----------:|--------:|----------:|-----------:|
| 0      |    12.4  |  3.671  |  0.021    |   3.0    |    4.5     |  51.35  |   0.269   |   15.0     |
| 5      |    39.7  |  3.675  |  0.018    |   4.5    |    3.5     |  51.45  |   0.244   |   15.0     |
| 20     |    39.9  |  3.701  |  0.012    |   4.5    |    5.5     |  51.86  |   0.190   |   13.0     |

Read row by row:

- **No early MLM convergence.** `auc_train_mlm` grows mildly with
  lambda; final val MLM loss is flat-to-worse at lam=20.
  `step_to_half_mlm` is *higher* under coherence (3.0 → 4.5), and only
  at lam=20/seed=2 did the held-out criterion pick an early-stop step
  before the end.
- **Clear drift effect, dose-dependent.** Final val_drift falls almost
  monotonically (0.021 → 0.018 → 0.012), and the drift-AUC (the
  *integrated* incoherence across training) drops by ~30% at lam=20.
  This holds across both seeds individually — the ordering
  lam=20 < lam=5 < lam=0 is consistent, not just present in the mean.
- **No cycle saving in compute units.** Coherence runs cost ~3.2×
  the baseline in wall time and forward calls. To match a baseline
  at the same forward-call budget, the coherence run would need to
  converge in roughly a third as many steps. It does not.
- **`step_to_half_drift` is noisy.** It does not order cleanly with
  lambda. The drift trajectories converge fast in absolute terms (a
  few steps to halfway) but the differences between regimes sit
  within-noise at this seed count.

### What that means for the user's question

The smoke run hinted that the regularizer moved drift in the predicted
direction; under the heavier setup that hint *strengthens*: the effect
is dose-dependent and seed-consistent, and the AUC of val_drift is a
less noisy summary than any single endpoint. So as a **measurement**
(drift correlates with the regularizer even when each step is more
expensive), the signal is real and arguably stronger than at the smoke
scale.

But as a **shortcut** (use coherence to converge faster), there is no
evidence of cycle saving in this regime. The regularizer slightly
worsens MLM loss and costs ~3× per step. The more promising read for
"early-convergence proxy" is the *baseline* run's drift curve, which
already drops smoothly with no coherence pressure at all — so drift
might be useful as a *passive* training-time signal without paying the
regularization cost. That is a cheaper follow-up than further sweeps.

### Caveats specific to this configuration

- The backbone is frozen; only the embedding moves. We are measuring a
  representation effect, not a propagation effect. Training attention
  is the obvious next experiment and would also narrow the gap between
  "drift drops" and "MLM improves".
- The absolute drift threshold (default 0.25) is satisfied at step 1
  for every run because hidden states are small at init. That is why
  heavy mode reports `step_to_half_drift` (calibration-free) instead;
  the absolute-threshold metric stays in the CSV so other thresholds
  can be evaluated post-hoc.
- 2 seeds × 3 lambdas is not enough to claim statistical separation on
  `auc_*`; report directional change and ordering, not magnitudes.
