"""
Train Behavior Cloning policy over a frozen Hand Action VAE.

Predicts:
  - arm_action  (dims 0-5): decode from {visual_feat, arm_state_feat}
  - hand_action (dims 6-11): VAE.decode(mu_prior + delta_z) — decoder frozen
    where delta_z is predicted from {visual_feat, hand_prior_feat} and can
    optionally also condition on arm_state_feat

Usage:
    python imitation_learning/behavior_clone/scripts/train.py \
        --train_dir data/20260327-11:10:43/demos/success/train \
        --test_dir  data/20260327-11:10:43/demos/success/test \
        --vae_ckpt  outputs/dim_2_best/checkpoint.pth \
        --output_dir outputs/bc_simple_v1
"""

import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BC_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))
_VAE_ROOT = os.path.join(_PROJ_ROOT, "vae")


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dataset_mod = _load(os.path.join(_BC_ROOT, "model/bc_dataset.py"), "bc_dataset")
_policy_mod = _load(os.path.join(_BC_ROOT, "model/bc_policy.py"), "bc_policy")
_utils_mod = _load(os.path.join(_VAE_ROOT, "model/utils.py"), "vae_utils")

BCDataset = _dataset_mod.BCDataset
compute_action_stats = _dataset_mod.compute_action_stats
BCPolicy = _policy_mod.BCPolicy
build_and_freeze_vae = _policy_mod.build_and_freeze_vae
trainable_params = _policy_mod.trainable_params
strip_vae_state_dict = _policy_mod.strip_vae_state_dict
cosine_scheduler = _utils_mod.cosine_scheduler


# ─── Args ──────────────────────────────────────────────────────────────────────


def get_args():
    p = argparse.ArgumentParser(description="Train BC policy with frozen VAE")
    # Data
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--vae_ckpt", type=str,
                   default="outputs/dim_2_best/checkpoint.pth")
    p.add_argument("--output_dir", type=str, default="outputs/bc_simple_v2")
    p.add_argument("--window_size", type=int, default=8,
                   help="VAE prior window size — must match VAE training")
    # Model
    p.add_argument("--feat_dim", type=int, default=128,
                   help="Per-branch feature width before fusion")
    p.add_argument("--fusion_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout probability in the arm/head MLPs")
    p.add_argument("--arm_gru_hidden", type=int, default=256,
                   help="Hidden dim for arm GRU decoder (chunk mode only)")
    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Lower than VAE's 2e-3 — from-scratch CNN can spike at higher LR")
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--noise_std_hand", type=float, default=0.1,
                   help="Gaussian noise std on past_hand_win during training. "
                        "Critical for AR robustness — without noise the model "
                        "drifts catastrophically in autoregressive mode.")
    p.add_argument("--noise_std_arm", type=float, default=0.1,
                   help="Gaussian noise std on z-score normalized arm state[:6] "
                        "during training. Mitigates AR drift on arm branch.")
    p.add_argument("--reg_drift", type=float, default=1.0,
                   help="Weight on the hand-drift regularizer "
                        "MSE(hand_action, hand_no_corr). hand_no_corr is the frozen "
                        "decoder output at mu_prior, so the regularizer penalizes how "
                        "far the hand branch pushes the latent away from the prior mean.")
    p.add_argument("--future_horizon", type=int, default=8,
                   help="Number of future frames to predict in chunk mode")
    p.add_argument("--num_workers", type=int, default=0,
                   help="0 = single process. Default is 0 because all data is "
                        "preloaded into RAM and forking workers can blow /dev/shm.")
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--print_freq", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=1000)
    p.add_argument("--save_freq", type=int, default=5000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─── Eval ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(policy: BCPolicy, loader: DataLoader, device: torch.device, chunk_mode: bool = False) -> dict:
    """Eval set MSE for arm, hand, and the no-correction baseline (delta_z = 0)."""
    policy.eval()

    sums = {"arm": 0.0, "hand_full": 0.0, "hand_no_corr": 0.0, "n": 0}

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        gt = batch["gt_action"]
        if chunk_mode:
            arm_gt = gt[:, :, :6]
            hand_gt = gt[:, :, 6:]
        else:
            arm_gt = gt[:, :6]
            hand_gt = gt[:, 6:]

        out = policy(
            img_main=batch["img_main"],
            img_extra=batch["img_extra"],
            state=batch["state"],
            past_hand_win=batch["past_hand_win"],
        )
        arm_mse = F.mse_loss(out["arm_action"], arm_gt).item()
        hand_mse_full = F.mse_loss(out["hand_action"], hand_gt).item()

        out_zero = policy(
            img_main=batch["img_main"],
            img_extra=batch["img_extra"],
            state=batch["state"],
            past_hand_win=batch["past_hand_win"],
            zero_delta=True,
        )
        hand_mse_no_corr = F.mse_loss(out_zero["hand_action"], hand_gt).item()

        batch_size = gt.shape[0]
        sums["arm"] += arm_mse * batch_size
        sums["hand_full"] += hand_mse_full * batch_size
        sums["hand_no_corr"] += hand_mse_no_corr * batch_size
        sums["n"] += batch_size

    policy.train()

    n = sums["n"]
    return {
        "arm_mse": sums["arm"] / n,
        "hand_mse_full": sums["hand_full"] / n,
        "hand_mse_no_correction": sums["hand_no_corr"] / n,
        "total_mse": (sums["arm"] + sums["hand_full"]) / n,
    }


# ─── Checkpoint ────────────────────────────────────────────────────────────────


def save_checkpoint(policy, optimizer, step, output_dir, action_mean, action_std, args,
                    chunk_mode=False, future_horizon=1):
    os.makedirs(output_dir, exist_ok=True)
    saved_args = vars(args).copy()
    saved_args["chunk_mode"] = chunk_mode
    saved_args["future_horizon"] = future_horizon
    state = {
        "model": strip_vae_state_dict(policy.state_dict()),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "action_mean": action_mean,
        "action_std": action_std,
        "args": saved_args,
    }
    torch.save(state, os.path.join(output_dir, f"checkpoint-{step}.pth"))
    torch.save(state, os.path.join(output_dir, "checkpoint.pth"))


# ─── Plot ──────────────────────────────────────────────────────────────────────


def save_training_curves(history, output_dir, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping training curves")
        return

    from numpy import convolve, ones

    steps = history["steps"]
    eval_steps = history["eval_steps"]
    window = max(1, min(100, len(steps) // 10))
    kernel = ones(window) / window

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"BC Training Curves  (lr={args.lr}, batch={args.batch_size}, "
        f"steps={args.total_steps})",
        fontsize=13,
    )

    def smooth(arr):
        return convolve(arr, kernel, mode="valid") if window > 1 and len(arr) >= window else arr

    # 1. Total train loss
    ax = axes[0, 0]
    ax.plot(steps, history["train_total"], alpha=0.3, color="blue", linewidth=0.5)
    sm = smooth(history["train_total"])
    ax.plot(steps[:len(sm)], sm, color="blue", linewidth=1.5, label="Smoothed")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(f"Train Total Loss  (arm + hand + {args.reg_drift}*drift)")
    ax.grid(True, alpha=0.3); ax.legend()

    # 2. Train arm vs hand vs drift
    ax = axes[0, 1]
    sm_arm = smooth(history["train_arm"])
    sm_hand = smooth(history["train_hand"])
    sm_drift = smooth(history["train_drift"])
    ax.plot(steps[:len(sm_arm)], sm_arm, color="green", linewidth=1.5, label="arm (smoothed)")
    ax.plot(steps[:len(sm_hand)], sm_hand, color="red", linewidth=1.5, label="hand (smoothed)")
    ax.plot(steps[:len(sm_drift)], sm_drift, color="purple", linewidth=1.5,
            label="drift (unweighted)", linestyle="--")
    ax.set_xlabel("Step"); ax.set_ylabel("MSE")
    ax.set_title("Train arm / hand / drift MSE")
    ax.grid(True, alpha=0.3); ax.legend()

    # 3. Val arm + hand_full + no_correction baseline
    ax = axes[0, 2]
    if eval_steps:
        ax.plot(eval_steps, history["val_arm"], "go-", markersize=4, label="val arm")
        ax.plot(eval_steps, history["val_hand_full"], "ro-", markersize=4, label="val hand (delta)")
        ax.plot(eval_steps, history["val_hand_no_corr"], "k--", linewidth=1, label="val hand (no corr)")
    ax.set_xlabel("Step"); ax.set_ylabel("MSE")
    ax.set_title("Validation MSE")
    ax.grid(True, alpha=0.3); ax.legend()

    # 4. Val total
    ax = axes[1, 0]
    if eval_steps:
        ax.plot(eval_steps, history["val_total"], "bo-", markersize=4, label="val total")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Validation Total MSE")
    ax.grid(True, alpha=0.3); ax.legend()

    # 5. LR schedule
    ax = axes[1, 1]
    ax.plot(steps, history["train_lr"], color="teal", linewidth=1.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate (Cosine)")
    ax.grid(True, alpha=0.3)

    # 6. Hand-MSE delta vs no-correction (proxy for "did vision help?")
    ax = axes[1, 2]
    if eval_steps:
        improvement = [b - a for a, b in zip(history["val_hand_full"], history["val_hand_no_corr"])]
        ax.plot(eval_steps, improvement, "mo-", markersize=4)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Step"); ax.set_ylabel("hand_no_corr - hand_full")
    ax.set_title("Vision Improvement Over No-Correction Baseline")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {fig_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Action stats from train split (one pass over all 12-dim action vectors) ──
    print(f"Computing action mean/std from {args.train_dir} ...")
    action_mean, action_std = compute_action_stats(args.train_dir)
    print(f"  action_mean = {[round(x, 3) for x in action_mean.tolist()]}")
    print(f"  action_std  = {[round(x, 3) for x in action_std.tolist()]}")

    # ── VAE: auto-detect type ──
    vae, vae_type = build_and_freeze_vae(args.vae_ckpt)
    chunk_mode = (vae_type == "chunk")
    future_horizon = args.future_horizon if chunk_mode else 1
    print(f"[VAE] type={vae_type}  chunk_mode={chunk_mode}  future_horizon={future_horizon}")

    # ── Datasets ──
    train_ds = BCDataset(
        args.train_dir, action_mean=action_mean, action_std=action_std,
        window_size=args.window_size,
        noise_std_hand=args.noise_std_hand,
        noise_std_arm=args.noise_std_arm,
        chunk_mode=chunk_mode,
        future_horizon=future_horizon,
    )
    test_ds = BCDataset(
        args.test_dir, action_mean=action_mean, action_std=action_std,
        window_size=args.window_size,
        noise_std_hand=0.0,   # never noise the eval split
        noise_std_arm=0.0,
        chunk_mode=chunk_mode,
        future_horizon=future_horizon,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model: frozen VAE wrapped inside BCPolicy ──
    policy = BCPolicy(
        vae=vae,
        arm_state_dim=6,
        feat_dim=args.feat_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        chunk_mode=chunk_mode,
        future_horizon=future_horizon,
        arm_gru_hidden=args.arm_gru_hidden,
        action_mean=action_mean if chunk_mode else None,
        action_std=action_std if chunk_mode else None,
    ).to(device)
    print(f"[Aug] noise_std_hand={args.noise_std_hand}  noise_std_arm={args.noise_std_arm}")
    if chunk_mode:
        print(f"[Arch] BC chunk — arm GRU decoder + chunk VAE hand, horizon={future_horizon}")
    else:
        print("[Arch] BC 3.0 — weakly-coupled arm/hand branches, hand conditioned on arm.")

    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy parameters: total={n_total:,}  trainable={n_train:,}  "
          f"frozen={n_total - n_train:,}")

    # ── Optimizer & LR schedule (no beta — BC has no KL) ──
    optimizer = torch.optim.AdamW(
        trainable_params(policy), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_schedule = cosine_scheduler(
        args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps,
    )

    # ── Step-0 sanity check ──
    # Verify zero-init: the hand delta-z head must output exactly 0 on a real batch.
    print("\n[Sanity] Step-0 zero-init check:")
    policy.eval()
    with torch.no_grad():
        peek_batch = next(iter(test_loader))
        peek_batch = {k: v.to(device) for k, v in peek_batch.items()}
        peek_out = policy(
            img_main=peek_batch["img_main"],
            img_extra=peek_batch["img_extra"],
            state=peek_batch["state"],
            past_hand_win=peek_batch["past_hand_win"],
        )
    dz_max = peek_out["delta_z"].abs().max().item()
    print(f"  |delta_z|_max = {dz_max:.2e}  (must be 0.00e+00)")
    assert dz_max == 0.0, "zero-init of hand delta-z head is broken"
    policy.train()

    print("[Sanity] Step-0 evaluation:")
    init_eval = evaluate(policy, test_loader, device, chunk_mode=chunk_mode)
    print(f"  arm_mse                = {init_eval['arm_mse']:.6f}")
    print(f"  hand_mse_full          = {init_eval['hand_mse_full']:.6f}")
    print(f"  hand_mse_no_correction = {init_eval['hand_mse_no_correction']:.6f}")
    print()

    # ── Training loop ──
    history = {
        "steps": [], "train_total": [], "train_arm": [], "train_hand": [],
        "train_drift": [], "train_lr": [],
        "eval_steps": [], "val_arm": [], "val_hand_full": [],
        "val_hand_no_corr": [], "val_total": [],
    }

    train_iter = iter(train_loader)
    policy.train()
    print(f"Training for {args.total_steps} steps  "
          f"(train={len(train_ds)}, test={len(test_ds)})\n")
    t0 = time.time()

    for step in range(args.total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[step]

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        gt = batch["gt_action"]

        out = policy(
            img_main=batch["img_main"],
            img_extra=batch["img_extra"],
            state=batch["state"],
            past_hand_win=batch["past_hand_win"],
        )
        if chunk_mode:
            arm_loss = F.mse_loss(out["arm_action"], gt[:, :, :6])
            hand_loss = F.mse_loss(out["hand_action"], gt[:, :, 6:])
        else:
            arm_loss = F.mse_loss(out["arm_action"], gt[:, :6])
            hand_loss = F.mse_loss(out["hand_action"], gt[:, 6:])
        # Drift regularizer: pull BC's hand prediction toward the decoder output
        # at the frozen prior mean. The decoder is frozen, so gradients flow only
        # through the corrected hand path.
        drift_loss = F.mse_loss(out["hand_action"], out["hand_no_corr"])
        loss = arm_loss + hand_loss + args.reg_drift * drift_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params(policy), args.clip_grad)
        optimizer.step()

        history["steps"].append(step)
        history["train_total"].append(loss.item())
        history["train_arm"].append(arm_loss.item())
        history["train_hand"].append(hand_loss.item())
        history["train_drift"].append(drift_loss.item())
        history["train_lr"].append(lr_schedule[step])

        if step % args.print_freq == 0:
            elapsed = time.time() - t0
            print(f"[Step {step:>5d}] total={loss.item():.6f}  "
                  f"arm={arm_loss.item():.6f}  hand={hand_loss.item():.6f}  "
                  f"drift={drift_loss.item():.6f}  "
                  f"lr={lr_schedule[step]:.2e}  ({elapsed:.0f}s)")

        if step > 0 and step % args.eval_freq == 0:
            ev = evaluate(policy, test_loader, device, chunk_mode=chunk_mode)
            history["eval_steps"].append(step)
            history["val_arm"].append(ev["arm_mse"])
            history["val_hand_full"].append(ev["hand_mse_full"])
            history["val_hand_no_corr"].append(ev["hand_mse_no_correction"])
            history["val_total"].append(ev["total_mse"])
            improved = ev["hand_mse_no_correction"] - ev["hand_mse_full"]
            print(f"           val arm={ev['arm_mse']:.6f}  "
                  f"hand={ev['hand_mse_full']:.6f}  "
                  f"no_corr={ev['hand_mse_no_correction']:.6f}  "
                  f"vision_gain={improved:+.6f}")

        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(policy, optimizer, step, args.output_dir,
                            action_mean, action_std, args,
                            chunk_mode=chunk_mode, future_horizon=future_horizon)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    # ── Final eval + save ──
    save_checkpoint(policy, optimizer, args.total_steps, args.output_dir,
                    action_mean, action_std, args,
                    chunk_mode=chunk_mode, future_horizon=future_horizon)
    ev = evaluate(policy, test_loader, device, chunk_mode=chunk_mode)
    history["eval_steps"].append(args.total_steps)
    history["val_arm"].append(ev["arm_mse"])
    history["val_hand_full"].append(ev["hand_mse_full"])
    history["val_hand_no_corr"].append(ev["hand_mse_no_correction"])
    history["val_total"].append(ev["total_mse"])

    print(f"\nFinal val: arm={ev['arm_mse']:.6f}  "
          f"hand={ev['hand_mse_full']:.6f}  "
          f"no_corr={ev['hand_mse_no_correction']:.6f}  "
          f"vision_gain={ev['hand_mse_no_correction'] - ev['hand_mse_full']:+.6f}")
    print(f"  Vision-corrected hand prediction "
          f"{'BEATS' if ev['hand_mse_full'] < ev['hand_mse_no_correction'] else 'DOES NOT beat'} "
          f"the no-correction baseline.")

    save_training_curves(history, args.output_dir, args)
    print(f"\nDone! Checkpoints + figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
