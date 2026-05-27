#!/usr/bin/env python3
"""Coherence-regularized masked language modeling (tiny, numpy-only).

Framing
-------
The user's conjecture: latency / settling-time / activation jitter in a
network is a proxy for hidden energetic-geometric strain. A model that
predicts the right masked token but does so via a turbulent internal
trajectory is *locally* correct but *globally* incoherent — small
perturbations to the input (mask shifts, token-preserving noise) should
then produce disproportionate hidden-state drift.

This module gives us a knob to test that. We define:

- A standard MLM task on a tiny synthetic vocabulary with a clear
  *coherence-violating* failure mode (an n-gram language where guessing
  a masked token correctly is easy on average but the *internal
  trajectory* the model takes can be jittery).
- A tiny encoder with a **frozen random transformer backbone** plus
  trainable token embeddings and a tied output head. Freezing the
  backbone makes the only thing that can adapt the embeddings; this
  keeps gradients analytic and makes the coherence experiment isolate
  *how the inputs are represented*, not *how the encoder propagates
  them* (which would muddy the probe).
- Coherence proxies measured on the encoder's hidden trajectory:
    * ``perturbation_drift``: cosine distance between hidden states at
      the masked positions under the original input vs. an alternate
      same-meaning input (mask shifted by 1, or a token-preserving
      noise of one unrelated position).
    * ``layer_jitter``: mean L2 between hidden(layer L) and
      hidden(layer L-1) across layers, normalized by hidden(layer 0).
      Standing in for "activation turbulence between propagation steps".
    * ``attention_entropy``: mean Shannon entropy of attention rows in
      the (frozen) attention layers — a smoothness proxy for whether
      the encoder is "settling" or "thrashing".
    * ``settle_steps``: a time-to-settle proxy via iterative refinement
      — feed the encoder its own output as input k times and report
      the smallest k for which the masked-position hidden state stops
      moving (within tolerance).

Coherence loss
--------------
We optimize MLM cross-entropy plus a scalar multiple of
``perturbation_drift`` evaluated at the masked positions. This is the
only term we can differentiate end-to-end in closed form without
implementing full backprop through attention; we use the fact that the
encoder is frozen and linear in the embeddings *at the input layer* to
get a tractable surrogate gradient (see ``_coherence_grad``).

Nothing here claims to be SOTA. The point is a clean, modifiable knob:
``lambda_coh = 0`` recovers a baseline MLM; ``lambda_coh > 0`` adds
the regularizer and we read out whether the proxies move together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data: a tiny n-gram language with deliberate local-but-not-global
# structure. Vocabulary = {0..V-1}. Sequences are produced by a Markov chain
# with a few "attractor" tokens that should be easy to predict in context;
# the rest are noisier. This gives MLM something to learn quickly on CPU.
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    vocab_size: int = 24
    seq_len: int = 12
    num_sequences: int = 256
    mask_prob: float = 0.2
    seed: int = 0


def _markov_matrix(vocab_size: int, rng: np.random.Generator) -> np.ndarray:
    """Build a sparse-ish transition matrix with a couple of attractor tokens."""
    m = rng.dirichlet(np.ones(vocab_size) * 0.3, size=vocab_size)
    # Bias a few "attractor" transitions to be strongly predictable.
    attractors = [(1, 2), (2, 3), (5, 6), (10, 11), (15, 16)]
    attractors = [(a, b) for a, b in attractors if a < vocab_size and b < vocab_size]
    for a, b in attractors:
        m[a] *= 0.05
        m[a, b] += 0.85
        m[a] /= m[a].sum()
    return m


def make_dataset(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (sequences, transition_matrix). Sequences shape (N, L)."""
    rng = np.random.default_rng(cfg.seed)
    trans = _markov_matrix(cfg.vocab_size, rng)
    seqs = np.zeros((cfg.num_sequences, cfg.seq_len), dtype=np.int64)
    for i in range(cfg.num_sequences):
        seqs[i, 0] = rng.integers(0, cfg.vocab_size)
        for t in range(1, cfg.seq_len):
            seqs[i, t] = rng.choice(cfg.vocab_size, p=trans[seqs[i, t - 1]])
    return seqs, trans


def apply_mask(
    seqs: np.ndarray,
    mask_token: int,
    mask_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard MLM masking. Returns (masked_input, mask_positions_bool)."""
    mask = rng.random(seqs.shape) < mask_prob
    # Always mask at least one position per sequence.
    for i in range(seqs.shape[0]):
        if not mask[i].any():
            j = rng.integers(0, seqs.shape[1])
            mask[i, j] = True
    masked = seqs.copy()
    masked[mask] = mask_token
    return masked, mask


# ---------------------------------------------------------------------------
# Tiny encoder: frozen random transformer backbone + trainable embeddings.
# We expose the full hidden trajectory so the coherence probes can read it.
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int = 25  # +1 for [MASK]
    hidden_dim: int = 16
    num_layers: int = 2
    num_heads: int = 2
    seq_len: int = 12
    seed: int = 1


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class LayerWeights:
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    Wff1: np.ndarray
    bff1: np.ndarray
    Wff2: np.ndarray
    bff2: np.ndarray


@dataclass
class Model:
    cfg: ModelConfig
    embed: np.ndarray  # (V, H) — trainable
    pos: np.ndarray  # (L, H) — frozen
    layers: list[LayerWeights]  # frozen


# Module-level forward-call counter so the driver can report compute used
# as a unit independent of wall time.
_FORWARD_CALLS = 0


def reset_forward_calls() -> None:
    global _FORWARD_CALLS
    _FORWARD_CALLS = 0


def get_forward_calls() -> int:
    return _FORWARD_CALLS


def init_model(cfg: ModelConfig) -> Model:
    rng = np.random.default_rng(cfg.seed)
    scale = 1.0 / math.sqrt(cfg.hidden_dim)
    embed = rng.standard_normal((cfg.vocab_size, cfg.hidden_dim)) * scale
    pos = rng.standard_normal((cfg.seq_len, cfg.hidden_dim)) * scale
    layers = []
    for _ in range(cfg.num_layers):
        layers.append(
            LayerWeights(
                Wq=rng.standard_normal((cfg.hidden_dim, cfg.hidden_dim)) * scale,
                Wk=rng.standard_normal((cfg.hidden_dim, cfg.hidden_dim)) * scale,
                Wv=rng.standard_normal((cfg.hidden_dim, cfg.hidden_dim)) * scale,
                Wo=rng.standard_normal((cfg.hidden_dim, cfg.hidden_dim)) * scale,
                Wff1=rng.standard_normal((cfg.hidden_dim, cfg.hidden_dim * 2)) * scale,
                bff1=np.zeros(cfg.hidden_dim * 2),
                Wff2=rng.standard_normal((cfg.hidden_dim * 2, cfg.hidden_dim)) * scale,
                bff2=np.zeros(cfg.hidden_dim),
            )
        )
    return Model(cfg=cfg, embed=embed, pos=pos, layers=layers)


def _layer_forward(
    h: np.ndarray,
    w: LayerWeights,
    num_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Single transformer-ish layer. Returns (h_out, attention_probs).

    Pre-norm-free for simplicity. h shape (B, L, H).
    """
    B, L, H = h.shape
    head_dim = H // num_heads
    Q = h @ w.Wq
    K = h @ w.Wk
    V = h @ w.Wv
    Q = Q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
    scores = np.einsum("bhld,bhmd->bhlm", Q, K) / math.sqrt(head_dim)
    attn = _softmax(scores, axis=-1)
    out = np.einsum("bhlm,bhmd->bhld", attn, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, H)
    out = out @ w.Wo
    h2 = h + out
    ff = np.tanh(h2 @ w.Wff1 + w.bff1) @ w.Wff2 + w.bff2
    return h2 + ff, attn


def forward(
    model: Model,
    tokens: np.ndarray,
) -> dict:
    """Forward pass. tokens shape (B, L). Returns rich trace for probes."""
    global _FORWARD_CALLS
    _FORWARD_CALLS += 1
    B, L = tokens.shape
    h0 = model.embed[tokens] + model.pos[None, :L, :]
    trajectory = [h0]
    attentions = []
    h = h0
    for layer in model.layers:
        h, attn = _layer_forward(h, layer, model.cfg.num_heads)
        trajectory.append(h)
        attentions.append(attn)
    logits = h @ model.embed.T  # tied output projection
    return {
        "logits": logits,
        "hidden": h,
        "trajectory": trajectory,  # list len num_layers+1
        "attentions": attentions,
    }


# ---------------------------------------------------------------------------
# Losses and probes
# ---------------------------------------------------------------------------


def mlm_loss_and_grad(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Cross-entropy over masked positions. Returns (loss, dL/dlogits)."""
    B, L, V = logits.shape
    probs = _softmax(logits, axis=-1)
    one_hot = np.zeros_like(probs)
    bi, li = np.where(mask)
    one_hot[bi, li, targets[bi, li]] = 1.0
    eps = 1e-12
    log_probs = np.log(probs + eps)
    n = mask.sum()
    loss = -float((one_hot * log_probs).sum() / max(n, 1))
    grad = (probs - one_hot) * mask[..., None] / max(n, 1)
    return loss, grad


def perturbation_drift(
    model: Model,
    tokens: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Cosine distance between masked hidden states under the original input
    and a token-preserving perturbation (replace one unmasked position with
    another token that has similar transition behaviour — here we just resample
    uniformly; the goal is to test sensitivity, not preserve semantics).

    Returns (drift_scalar, hidden_orig, hidden_pert).
    """
    out_orig = forward(model, tokens)
    h_orig = out_orig["hidden"]
    perturbed = tokens.copy()
    B, L = tokens.shape
    for i in range(B):
        candidates = np.where(~mask[i])[0]
        if len(candidates) == 0:
            continue
        j = rng.choice(candidates)
        perturbed[i, j] = rng.integers(0, model.cfg.vocab_size - 1)
    out_pert = forward(model, perturbed)
    h_pert = out_pert["hidden"]
    # cosine distance at masked positions, averaged
    drifts = []
    for i in range(B):
        for j in np.where(mask[i])[0]:
            a = h_orig[i, j]
            b = h_pert[i, j]
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            drifts.append(1.0 - float(np.dot(a, b) / denom))
    drift = float(np.mean(drifts)) if drifts else 0.0
    return drift, h_orig, h_pert


def layer_jitter(trajectory: list[np.ndarray]) -> float:
    """Mean L2 between successive hidden layers, normalized by layer-0 norm.

    Stands in for activation turbulence between propagation steps.
    """
    base = float(np.linalg.norm(trajectory[0]) + 1e-12)
    diffs = []
    for k in range(1, len(trajectory)):
        diffs.append(float(np.linalg.norm(trajectory[k] - trajectory[k - 1])))
    return float(np.mean(diffs)) / base


def attention_entropy(attentions: list[np.ndarray]) -> float:
    """Mean Shannon entropy of attention rows across layers and heads."""
    ents = []
    for a in attentions:
        p = a + 1e-12
        ent = -(p * np.log(p)).sum(axis=-1)
        ents.append(float(ent.mean()))
    return float(np.mean(ents))


def settle_steps(
    model: Model,
    tokens: np.ndarray,
    mask: np.ndarray,
    max_iters: int = 6,
    tol: float = 1e-3,
) -> float:
    """Iterative refinement time-to-settle. We feed the argmax of the encoder's
    output back in at the masked positions and measure how many iterations
    until the masked-position hidden state changes by less than ``tol`` (L2).
    """
    current = tokens.copy()
    prev_h = None
    for k in range(1, max_iters + 1):
        out = forward(model, current)
        h = out["hidden"]
        preds = out["logits"].argmax(axis=-1)
        # only update masked positions
        current = np.where(mask, preds, current)
        if prev_h is not None:
            diff = float(
                np.linalg.norm((h - prev_h)[mask[..., None].repeat(h.shape[-1], -1).reshape(h.shape)])
            )
            if diff < tol:
                return float(k)
        prev_h = h
    return float(max_iters)


# ---------------------------------------------------------------------------
# Training step. Only the embedding is trainable; backprop is analytic
# because the encoder backbone is frozen — we still need to push the gradient
# of MLM loss back through the (frozen) layers, which we do by finite
# differences w.r.t. the embedding matrix. That sounds expensive but with
# V=25 and H=16 it's a 400-element Jacobian and the model is tiny; on CPU
# this is plenty fast for a probe. For the coherence term we use a cheaper
# surrogate gradient that pushes embeddings to make hidden states at masked
# positions less sensitive to a fixed perturbation direction.
# ---------------------------------------------------------------------------


def _finite_diff_grad_embed(
    model: Model,
    tokens: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-3,
) -> tuple[float, np.ndarray]:
    """Centered finite-difference gradient of MLM loss w.r.t. ``model.embed``.

    For tiny models this is the simplest correct thing. We probe only the
    rows of ``embed`` that are actually used by ``tokens`` to keep the work
    bounded by ``unique_tokens * hidden_dim``.
    """
    base_logits = forward(model, tokens)["logits"]
    base_loss, _ = mlm_loss_and_grad(base_logits, targets, mask)
    used = np.unique(tokens)
    grad = np.zeros_like(model.embed)
    H = model.cfg.hidden_dim
    for v in used:
        for d in range(H):
            orig = model.embed[v, d]
            model.embed[v, d] = orig + eps
            lp = mlm_loss_and_grad(forward(model, tokens)["logits"], targets, mask)[0]
            model.embed[v, d] = orig - eps
            lm = mlm_loss_and_grad(forward(model, tokens)["logits"], targets, mask)[0]
            model.embed[v, d] = orig
            grad[v, d] = (lp - lm) / (2 * eps)
    return base_loss, grad


def _coherence_grad(
    model: Model,
    tokens: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    eps: float = 1e-3,
) -> tuple[float, np.ndarray]:
    """Finite-difference gradient of perturbation_drift w.r.t. ``model.embed``.

    We hold the perturbation fixed across +eps and -eps probes (same rng
    state replay) so the gradient is consistent.
    """
    state = rng.bit_generator.state
    base_drift, _, _ = perturbation_drift(model, tokens, mask, np.random.default_rng(0))
    used = np.unique(tokens)
    grad = np.zeros_like(model.embed)
    H = model.cfg.hidden_dim
    for v in used:
        for d in range(H):
            orig = model.embed[v, d]
            model.embed[v, d] = orig + eps
            dp, _, _ = perturbation_drift(model, tokens, mask, np.random.default_rng(0))
            model.embed[v, d] = orig - eps
            dm, _, _ = perturbation_drift(model, tokens, mask, np.random.default_rng(0))
            model.embed[v, d] = orig
            grad[v, d] = (dp - dm) / (2 * eps)
    rng.bit_generator.state = state
    return base_drift, grad


@dataclass
class TrainConfig:
    steps: int = 30
    batch_size: int = 16
    lr: float = 0.05
    lambda_coh: float = 0.0
    seed: int = 7


def train(
    model: Model,
    data: np.ndarray,
    train_cfg: TrainConfig,
    mask_token: int,
    mask_prob: float,
    log_every: int = 1,
) -> list[dict]:
    """Train ``model.embed`` only. Returns per-step log records."""
    rng = np.random.default_rng(train_cfg.seed)
    log = []
    for step in range(1, train_cfg.steps + 1):
        idx = rng.choice(data.shape[0], size=train_cfg.batch_size, replace=False)
        batch = data[idx]
        masked, mask = apply_mask(batch, mask_token, mask_prob, rng)
        loss, g_mlm = _finite_diff_grad_embed(model, masked, batch, mask)
        if train_cfg.lambda_coh > 0:
            drift, g_coh = _coherence_grad(model, masked, mask, rng)
            grad = g_mlm + train_cfg.lambda_coh * g_coh
        else:
            drift, _, _ = perturbation_drift(model, masked, mask, np.random.default_rng(0))
            grad = g_mlm
        model.embed -= train_cfg.lr * grad
        if step % log_every == 0:
            out = forward(model, masked)
            traj = out["trajectory"]
            ent = attention_entropy(out["attentions"])
            jit = layer_jitter(traj)
            settle = settle_steps(model, masked, mask)
            log.append(
                {
                    "step": step,
                    "mlm_loss": float(loss),
                    "perturbation_drift": float(drift),
                    "layer_jitter": float(jit),
                    "attention_entropy": float(ent),
                    "settle_steps": float(settle),
                    "lambda_coh": float(train_cfg.lambda_coh),
                }
            )
    return log
