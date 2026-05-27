#!/usr/bin/env python3
"""Driver/report for the coherence-regularized MLM probe.

Two modes:

- ``--mode smoke``  (default): the original tiny two-run probe. Fast,
  ~25s, used as the regression-style sanity check.
- ``--mode heavy``: a more strenuous CPU experiment that asks the
  *follow-up* question — does coherence regularization give us
  early convergence or save compute cycles? Larger model, more steps,
  a lambda sweep, multiple seeds, and a held-out validation set so we
  can measure steps/time to reach matched thresholds.

The heavy mode logs per-step *and* per-run convergence metrics:
    - wall-clock time per run, cumulative time to reach an MLM loss
      threshold, and time to reach a held-out drift threshold,
    - step counts to those same thresholds,
    - area under the MLM-loss curve (compute-normalized),
    - early-stopping step under a matched held-out criterion,
    - total forward-pass calls (a model-size-agnostic compute proxy).

Output:
- prints per-run tables and a compact comparison summary,
- writes ``coherence_mlm_log.jsonl`` (per-step) and
  ``coherence_mlm_summary.csv`` (per-run) in the current working dir.

Run with::

    python experiments/coherence_mlm.py                 # smoke
    python experiments/coherence_mlm.py --mode heavy    # ~2-3 min
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.coherence_mlm import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    apply_mask,
    attention_entropy,
    forward,
    get_forward_calls,
    init_model,
    layer_jitter,
    make_dataset,
    mlm_loss_and_grad,
    perturbation_drift,
    reset_forward_calls,
    settle_steps,
    train,
)


# ---------------------------------------------------------------------------
# Held-out evaluation helpers
# ---------------------------------------------------------------------------


def _held_out_eval(
    model,
    val_seqs: np.ndarray,
    mask_token: int,
    mask_prob: float,
    eval_seed: int = 12345,
) -> dict:
    """Deterministic held-out probe. Uses a fixed seed so all runs see the
    same masks, which is what lets us compare *steps-to-threshold* fairly.
    """
    rng = np.random.default_rng(eval_seed)
    masked, mask = apply_mask(val_seqs, mask_token, mask_prob, rng)
    out = forward(model, masked)
    loss, _ = mlm_loss_and_grad(out["logits"], val_seqs, mask)
    drift, _, _ = perturbation_drift(
        model, masked, mask, np.random.default_rng(eval_seed + 1)
    )
    return {
        "val_mlm_loss": float(loss),
        "val_drift": float(drift),
        "val_jitter": float(layer_jitter(out["trajectory"])),
        "val_attn_entropy": float(attention_entropy(out["attentions"])),
    }


# ---------------------------------------------------------------------------
# Single instrumented run
# ---------------------------------------------------------------------------


def _train_with_validation(
    model,
    train_seqs: np.ndarray,
    val_seqs: np.ndarray,
    train_cfg: TrainConfig,
    mask_token: int,
    mask_prob: float,
    val_every: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Step-by-step training that also records held-out metrics every
    ``val_every`` steps. Returns (per_step_log, per_val_log).

    We do it ourselves instead of calling ``train`` so we can interleave
    held-out evals and capture per-step timing.
    """
    from geometry.coherence_mlm import (
        _coherence_grad,
        _finite_diff_grad_embed,
    )

    rng = np.random.default_rng(train_cfg.seed)
    step_log: list[dict] = []
    val_log: list[dict] = []
    for step in range(1, train_cfg.steps + 1):
        t0 = time.perf_counter()
        idx = rng.choice(train_seqs.shape[0], size=train_cfg.batch_size, replace=False)
        batch = train_seqs[idx]
        masked, mask = apply_mask(batch, mask_token, mask_prob, rng)
        loss, g_mlm = _finite_diff_grad_embed(model, masked, batch, mask)
        if train_cfg.lambda_coh > 0:
            drift, g_coh = _coherence_grad(model, masked, mask, rng)
            grad = g_mlm + train_cfg.lambda_coh * g_coh
        else:
            drift, _, _ = perturbation_drift(model, masked, mask, np.random.default_rng(0))
            grad = g_mlm
        model.embed -= train_cfg.lr * grad
        step_time = time.perf_counter() - t0
        step_log.append(
            {
                "step": step,
                "train_mlm_loss": float(loss),
                "train_drift": float(drift),
                "step_time_s": float(step_time),
            }
        )
        if step % val_every == 0 or step == train_cfg.steps:
            v = _held_out_eval(model, val_seqs, mask_token, mask_prob)
            v["step"] = step
            v["forward_calls"] = get_forward_calls()
            val_log.append(v)
    return step_log, val_log


# ---------------------------------------------------------------------------
# Convergence-metric extraction
# ---------------------------------------------------------------------------


def _first_step_reaching(
    log: list[dict], key: str, threshold: float, direction: str = "below"
) -> int | None:
    for row in log:
        v = row[key]
        if direction == "below" and v <= threshold:
            return int(row["step"])
        if direction == "above" and v >= threshold:
            return int(row["step"])
    return None


def _cumulative_time_to_step(step_log: list[dict], target_step: int | None) -> float | None:
    if target_step is None:
        return None
    total = 0.0
    for row in step_log:
        total += row["step_time_s"]
        if row["step"] >= target_step:
            return total
    return None


def _auc_loss(step_log: list[dict], key: str = "train_mlm_loss") -> float:
    """Trapezoidal AUC of the loss curve over steps. Lower = faster
    convergence (smaller area under loss curve)."""
    if len(step_log) < 2:
        return float(step_log[0][key]) if step_log else 0.0
    xs = [row["step"] for row in step_log]
    ys = [row[key] for row in step_log]
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return float(total)


def _early_stop_step(val_log: list[dict], key: str = "val_mlm_loss") -> int:
    """Step at which held-out metric was lowest (matched criterion)."""
    if not val_log:
        return 0
    best = min(val_log, key=lambda r: r[key])
    return int(best["step"])


def _step_to_within_eps(val_log: list[dict], key: str, eps: float) -> int | None:
    """First step at which ``key`` is within ``eps`` of its final value.
    Calibration-free convergence proxy: how soon does the curve flatten?"""
    if not val_log:
        return None
    final = val_log[-1][key]
    for row in val_log:
        if abs(row[key] - final) <= eps:
            return int(row["step"])
    return None


def _step_to_halfway(val_log: list[dict], key: str) -> int | None:
    """First step at which ``key`` reaches halfway between its first and
    final observed values. Direction-agnostic. Useful when absolute
    thresholds are uncalibrated."""
    if len(val_log) < 2:
        return None
    first = val_log[0][key]
    final = val_log[-1][key]
    target = 0.5 * (first + final)
    decreasing = final < first
    for row in val_log:
        if decreasing and row[key] <= target:
            return int(row["step"])
        if not decreasing and row[key] >= target:
            return int(row["step"])
    return None


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def _print_step_log_smoke(label: str, log: list[dict]) -> None:
    print(f"\n[{label}]  step  mlm_loss  drift   jitter  attn_H  settle")
    for row in log:
        print(
            f"           {row['step']:>4d}  "
            f"{row['mlm_loss']:.4f}   "
            f"{row['perturbation_drift']:.4f}  "
            f"{row['layer_jitter']:.4f}  "
            f"{row['attention_entropy']:.4f}  "
            f"{row['settle_steps']:.2f}"
        )


def run_smoke() -> int:
    """Original tiny probe — kept so existing behaviour is unchanged."""
    data_cfg = DataConfig(vocab_size=24, seq_len=10, num_sequences=128, mask_prob=0.2, seed=0)
    seqs, _ = make_dataset(data_cfg)
    mask_token = data_cfg.vocab_size
    model_cfg = ModelConfig(
        vocab_size=data_cfg.vocab_size + 1,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        seq_len=data_cfg.seq_len,
        seed=1,
    )
    base_model = init_model(model_cfg)
    runs = [
        ("baseline", TrainConfig(steps=15, batch_size=8, lr=0.10, lambda_coh=0.0, seed=7)),
        ("coherence", TrainConfig(steps=15, batch_size=8, lr=0.10, lambda_coh=5.0, seed=7)),
    ]
    all_logs: dict[str, list[dict]] = {}
    for label, tcfg in runs:
        model = copy.deepcopy(base_model)
        log = train(model, seqs, tcfg, mask_token=mask_token, mask_prob=data_cfg.mask_prob)
        all_logs[label] = log
        _print_step_log_smoke(label, log)
    jsonl_path = "coherence_mlm_log.jsonl"
    with open(jsonl_path, "w") as f:
        for label, log in all_logs.items():
            for row in log:
                f.write(json.dumps({"run": label, **row}) + "\n")
    print(f"\nWrote {jsonl_path}")
    return 0


def run_heavy(
    lambdas: list[float],
    seeds: list[int],
    steps: int,
    hidden_dim: int,
    num_layers: int,
    seq_len: int,
    num_sequences: int,
    val_fraction: float,
    batch_size: int,
    mlm_threshold: float,
    drift_threshold: float,
) -> int:
    """Heavier sweep: each (lambda, seed) pair is one run sharing the same
    base model init and the same train/val split. We instrument convergence
    metrics so the comparison is not just final-loss vs. final-loss.
    """
    data_cfg = DataConfig(
        vocab_size=32,
        seq_len=seq_len,
        num_sequences=num_sequences,
        mask_prob=0.2,
        seed=0,
    )
    seqs, _ = make_dataset(data_cfg)
    n_val = max(16, int(num_sequences * val_fraction))
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(seqs.shape[0])
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_seqs = seqs[val_idx]
    train_seqs = seqs[train_idx]
    mask_token = data_cfg.vocab_size

    summaries: list[dict] = []
    per_step_records: list[dict] = []
    per_val_records: list[dict] = []

    print(
        f"[heavy] V={data_cfg.vocab_size + 1} L={seq_len} N_train={len(train_idx)} "
        f"N_val={len(val_idx)} H={hidden_dim} layers={num_layers} steps={steps} "
        f"batch={batch_size} lambdas={lambdas} seeds={seeds}"
    )

    for seed in seeds:
        model_cfg = ModelConfig(
            vocab_size=data_cfg.vocab_size + 1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=2,
            seq_len=seq_len,
            seed=seed,
        )
        base_model = init_model(model_cfg)
        for lam in lambdas:
            label = f"seed{seed}_lam{lam:g}"
            model = copy.deepcopy(base_model)
            tcfg = TrainConfig(
                steps=steps,
                batch_size=batch_size,
                lr=0.08,
                lambda_coh=float(lam),
                seed=7 + seed,
            )
            reset_forward_calls()
            t_start = time.perf_counter()
            step_log, val_log = _train_with_validation(
                model,
                train_seqs,
                val_seqs,
                tcfg,
                mask_token=mask_token,
                mask_prob=data_cfg.mask_prob,
                val_every=max(1, steps // 10),
            )
            wall = time.perf_counter() - t_start
            fwd_calls = get_forward_calls()

            # Convergence metrics on held-out curve (matched across runs).
            step_to_mlm = _first_step_reaching(
                val_log, "val_mlm_loss", mlm_threshold, "below"
            )
            step_to_drift = _first_step_reaching(
                val_log, "val_drift", drift_threshold, "below"
            )
            time_to_mlm = _cumulative_time_to_step(step_log, step_to_mlm)
            time_to_drift = _cumulative_time_to_step(step_log, step_to_drift)
            auc_train = _auc_loss(step_log, "train_mlm_loss")
            es = _early_stop_step(val_log, "val_mlm_loss")
            step_to_mlm_eps = _step_to_within_eps(val_log, "val_mlm_loss", eps=0.05)
            step_to_half_drift = _step_to_halfway(val_log, "val_drift")
            step_to_half_mlm = _step_to_halfway(val_log, "val_mlm_loss")
            # AUC of held-out drift over (steps), as compute-normalized signal:
            #   how much accumulated incoherence the run carried.
            xs = [r["step"] for r in val_log]
            ys = [r["val_drift"] for r in val_log]
            auc_val_drift = 0.0
            for i in range(1, len(xs)):
                auc_val_drift += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])

            final_val = val_log[-1]
            summary = {
                "label": label,
                "seed": seed,
                "lambda_coh": float(lam),
                "wall_time_s": float(wall),
                "forward_calls": int(fwd_calls),
                "final_val_mlm_loss": final_val["val_mlm_loss"],
                "final_val_drift": final_val["val_drift"],
                "final_val_jitter": final_val["val_jitter"],
                "step_to_mlm_threshold": step_to_mlm,
                "time_to_mlm_threshold_s": time_to_mlm,
                "step_to_drift_threshold": step_to_drift,
                "time_to_drift_threshold_s": time_to_drift,
                "auc_train_mlm": auc_train,
                "auc_val_drift": float(auc_val_drift),
                "early_stop_step": es,
                "step_to_mlm_eps05": step_to_mlm_eps,
                "step_to_half_drift": step_to_half_drift,
                "step_to_half_mlm": step_to_half_mlm,
            }
            summaries.append(summary)
            for r in step_log:
                per_step_records.append({"run": label, **r})
            for r in val_log:
                per_val_records.append({"run": label, **r})
            print(
                f"  {label:<18s}  wall={wall:5.1f}s  fwd={fwd_calls:>6d}  "
                f"val_mlm={final_val['val_mlm_loss']:.3f}  "
                f"val_drift={final_val['val_drift']:.3f}  "
                f"half_mlm@{step_to_half_mlm}  half_drift@{step_to_half_drift}  "
                f"AUC_mlm={auc_train:.2f}  AUC_drift={auc_val_drift:.3f}  ES@{es}"
            )

    # ---- compact comparison: for each seed, baseline vs each lam ----
    print("\n=== Heavy summary (per seed, baseline vs coherence) ===")
    header = (
        f"{'run':<18s}  wall(s)  fwd     val_mlm  val_drift  "
        f"st_mlm  t_mlm   st_drift  t_drift  AUC     ES"
    )
    print(header)
    for s in summaries:
        t_mlm = f"{s['time_to_mlm_threshold_s']:.2f}" if s['time_to_mlm_threshold_s'] is not None else "-"
        t_drift = f"{s['time_to_drift_threshold_s']:.2f}" if s['time_to_drift_threshold_s'] is not None else "-"
        st_mlm = str(s['step_to_mlm_threshold']) if s['step_to_mlm_threshold'] is not None else "-"
        st_drift = str(s['step_to_drift_threshold']) if s['step_to_drift_threshold'] is not None else "-"
        print(
            f"{s['label']:<18s}  "
            f"{s['wall_time_s']:5.1f}    "
            f"{s['forward_calls']:>5d}  "
            f"{s['final_val_mlm_loss']:.3f}    "
            f"{s['final_val_drift']:.3f}     "
            f"{st_mlm:<6s}  "
            f"{t_mlm:<6s}  "
            f"{st_drift:<8s}  "
            f"{t_drift:<7s}  "
            f"{s['auc_train_mlm']:.2f}   "
            f"{s['early_stop_step']}"
        )

    # ---- aggregate by lambda (mean across seeds) ----
    print("\n=== Heavy summary (mean over seeds, per lambda) ===")
    by_lam: dict[float, list[dict]] = {}
    for s in summaries:
        by_lam.setdefault(s["lambda_coh"], []).append(s)
    print(
        f"{'lambda':<8s} wall(s)  val_mlm  val_drift  half_mlm  half_drift  "
        f"AUC_mlm  AUC_drift  ES"
    )
    for lam in sorted(by_lam.keys()):
        rows = by_lam[lam]
        wall = float(np.mean([r["wall_time_s"] for r in rows]))
        vm = float(np.mean([r["final_val_mlm_loss"] for r in rows]))
        vd = float(np.mean([r["final_val_drift"] for r in rows]))
        h_mlm = [r["step_to_half_mlm"] for r in rows if r["step_to_half_mlm"] is not None]
        h_drift = [r["step_to_half_drift"] for r in rows if r["step_to_half_drift"] is not None]
        auc = float(np.mean([r["auc_train_mlm"] for r in rows]))
        auc_d = float(np.mean([r["auc_val_drift"] for r in rows]))
        es_mean = float(np.mean([r["early_stop_step"] for r in rows]))
        h_mlm_str = f"{float(np.mean(h_mlm)):.1f}" if h_mlm else "-"
        h_drift_str = f"{float(np.mean(h_drift)):.1f}" if h_drift else "-"
        print(
            f"{lam:<8g} {wall:5.1f}    {vm:.3f}    {vd:.3f}      "
            f"{h_mlm_str:<8s}  {h_drift_str:<10s}  {auc:.2f}    {auc_d:.3f}      {es_mean:.1f}"
        )

    # ---- persistence ----
    with open("coherence_mlm_log.jsonl", "w") as f:
        for r in per_step_records:
            f.write(json.dumps({"kind": "step", **r}) + "\n")
        for r in per_val_records:
            f.write(json.dumps({"kind": "val", **r}) + "\n")
    sfields = [
        "label", "seed", "lambda_coh", "wall_time_s", "forward_calls",
        "final_val_mlm_loss", "final_val_drift", "final_val_jitter",
        "step_to_mlm_threshold", "time_to_mlm_threshold_s",
        "step_to_drift_threshold", "time_to_drift_threshold_s",
        "auc_train_mlm", "auc_val_drift", "early_stop_step",
        "step_to_mlm_eps05", "step_to_half_drift", "step_to_half_mlm",
    ]
    with open("coherence_mlm_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print("\nWrote coherence_mlm_log.jsonl and coherence_mlm_summary.csv")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("smoke", "heavy"), default="smoke")
    p.add_argument("--lambdas", type=str, default="0,5,20")
    p.add_argument("--seeds", type=str, default="1,2")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--num-sequences", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--mlm-threshold", type=float, default=2.7)
    p.add_argument("--drift-threshold", type=float, default=0.25)
    args = p.parse_args()
    if args.mode == "smoke":
        return run_smoke()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    return run_heavy(
        lambdas=lambdas,
        seeds=seeds,
        steps=args.steps,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        mlm_threshold=args.mlm_threshold,
        drift_threshold=args.drift_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
