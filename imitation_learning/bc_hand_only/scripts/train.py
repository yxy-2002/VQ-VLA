"""
Train hand-only Behavior Cloning policy over a frozen Hand Action VAE.

Predicts delta_z correction so that z_ctrl = mu_prior + delta_z, then
hand_action = VAE.decode(z_ctrl). No arm branch — purely hand diagnostics.

Usage:
    python imitation_learning/bc_hand_only/scripts/train.py \
        --train_dir data/20260327-11:10:43/demos/success/train \
        --test_dir  data/20260327-11:10:43/demos/success/test \
        --vae_ckpt  outputs/dim_2_best/checkpoint.pth \
        --output_dir outputs/bc_hand_only_sweep/baseline
"""

import argparse
import importlib.util
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BC_HAND_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_HAND_ROOT, "..", ".."))
_VAE_ROOT = os.path.join(_PROJ_ROOT, "vae")


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dataset_mod = _load(os.path.join(_BC_HAND_ROOT, "model/bc_hand_dataset.py"), "bc_hand_dataset")
_policy_mod = _load(os.path.join(_BC_HAND_ROOT, "model/bc_hand_policy.py"), "bc_hand_policy")
_utils_mod = _load(os.path.join(_VAE_ROOT, "model/utils.py"), "vae_utils")

BCHandDataset = _dataset_mod.BCHandDataset
BCHandPolicy = _policy_mod.BCHandPolicy
build_and_freeze_vae = _policy_mod.build_and_freeze_vae
trainable_params = _policy_mod.trainable_params
strip_vae_state_dict = _policy_mod.strip_vae_state_dict
cosine_scheduler = _utils_mod.cosine_scheduler


# ─── Args ──────────────────────────────────────────────────────────────────────


def get_args():
    p = argparse.ArgumentParser(description="Train hand-only BC policy with frozen VAE")
    # Data
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--vae_ckpt", type=str,
                   default="outputs/dim_2_best/checkpoint.pth")
    p.add_argument("--output_dir", type=str, default="outputs/bc_hand_only")
    p.add_argument("--window_size", type=int, default=8,
                   help="VAE prior window size — must match VAE training")
    # Model
    p.add_argument("--feat_dim", type=int, default=128,
                   help="Per-branch feature width before fusion")
    p.add_argument("--fusion_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout probability in the hand delta-z head")
    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--noise_std_hand", type=float, default=0.0,
                   help="Gaussian noise std on past_hand_win during training")
    p.add_argument("--disable_vision", action="store_true",
                   help="Ablation: zero out CNN features and freeze CNN params")
    p.add_argument("--reg_drift", type=float, default=1.0,
                   help="Weight on drift regularizer MSE(hand_action, hand_no_corr)")
    p.add_argument("--num_workers", type=int, default=0)
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
def evaluate(policy: BCHandPolicy, loader: DataLoader, device: torch.device) -> dict:
    """Eval set MSE for hand (with correction) and no-correction baseline."""
    policy.eval()
    sums = {"hand_full": 0.0, "hand_no_corr": 0.0, "n": 0}

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        gt = batch["gt_hand_action"]

        out = policy(
            img_main=batch["img_main"],
            img_extra=batch["img_extra"],
            past_hand_win=batch["past_hand_win"],
        )
        hand_mse_full = F.mse_loss(out["hand_action"], gt).item()

        out_zero = policy(
            img_main=batch["img_main"],
            img_extra=batch["img_extra"],
            past_hand_win=batch["past_hand_win"],
            zero_delta=True,
        )
        hand_mse_no_corr = F.mse_loss(out_zero["hand_action"], gt).item()

        batch_size = gt.shape[0]
        sums["hand_full"] += hand_mse_full * batch_size
        sums["hand_no_corr"] += hand_mse_no_corr * batch_size
        sums["n"] += batch_size

    policy.train()
    n = sums["n"]
    return {
        "hand_mse_full": sums["hand_full"] / n,
        "hand_mse_no_correction": sums["hand_no_corr"] / n,
    }


# ─── Checkpoint ────────────────────────────────────────────────────────────────


def save_checkpoint(policy, optimizer, step, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "model": strip_vae_state_dict(policy.state_dict()),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Hand-Only BC Training  (lr={args.lr}, batch={args.batch_size}, "
        f"steps={args.total_steps}, reg_drift={args.reg_drift})",
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
    ax.set_title(f"Train Total Loss  (hand + {args.reg_drift}*drift)")
    ax.grid(True, alpha=0.3); ax.legend()

    # 2. Train hand vs drift
    ax = axes[0, 1]
    sm_hand = smooth(history["train_hand"])
    sm_drift = smooth(history["train_drift"])
    ax.plot(steps[:len(sm_hand)], sm_hand, color="red", linewidth=1.5, label="hand (smoothed)")
    ax.plot(steps[:len(sm_drift)], sm_drift, color="purple", linewidth=1.5,
            label="drift (unweighted)", linestyle="--")
    ax.set_xlabel("Step"); ax.set_ylabel("MSE")
    ax.set_title("Train hand / drift MSE")
    ax.grid(True, alpha=0.3); ax.legend()

    # 3. Val hand_full vs hand_no_corr
    ax = axes[1, 0]
    if eval_steps:
        ax.plot(eval_steps, history["val_hand_full"], "ro-", markersize=4,
                label="val hand (delta)")
        ax.plot(eval_steps, history["val_hand_no_corr"], "k--", linewidth=1,
                label="val hand (no corr)")
        improvement = [b - a for a, b in
                       zip(history["val_hand_full"], history["val_hand_no_corr"])]
        ax.plot(eval_steps, improvement, "mo-", markersize=3, alpha=0.6,
                label="vision gain (no_corr - full)")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("MSE")
    ax.set_title("Validation Hand MSE")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 4. LR schedule
    ax = axes[1, 1]
    ax.plot(steps, history["train_lr"], color="teal", linewidth=1.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate (Cosine)")
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

    # ── Datasets ──
    train_ds = BCHandDataset(
        args.train_dir,
        window_size=args.window_size,
        noise_std_hand=args.noise_std_hand,
    )
    test_ds = BCHandDataset(
        args.test_dir,
        window_size=args.window_size,
        noise_std_hand=0.0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ──
    vae = build_and_freeze_vae(args.vae_ckpt)
    policy = BCHandPolicy(
        vae=vae,
        feat_dim=args.feat_dim,
        fusion_dim=args.fusion_dim,
        disable_vision=args.disable_vision,
        dropout=args.dropout,
    ).to(device)
    if args.disable_vision:
        print("[Ablation] disable_vision=True — CNN outputs forced to 0, "
              "CNN params frozen. BC sees only past_hand_win via frozen VAE.")
    if args.noise_std_hand > 0:
        print(f"[Aug] noise_std_hand={args.noise_std_hand} — train past_hand_win "
              f"will get N(0, {args.noise_std_hand}) noise per __getitem__.")

    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy parameters: total={n_total:,}  trainable={n_train:,}  "
          f"frozen={n_total - n_train:,}")

    # ── Optimizer & LR schedule ──
    optimizer = torch.optim.AdamW(
        trainable_params(policy), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_schedule = cosine_scheduler(
        args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps,
    )

    # ── Step-0 sanity check ──
    print("\n[Sanity] Step-0 zero-init check:")
    policy.eval()
    with torch.no_grad():
        peek_batch = next(iter(test_loader))
        peek_batch = {k: v.to(device) for k, v in peek_batch.items()}
        peek_out = policy(
            img_main=peek_batch["img_main"],
            img_extra=peek_batch["img_extra"],
            past_hand_win=peek_batch["past_hand_win"],
        )
    dz_max = peek_out["delta_z"].abs().max().item()
    print(f"  |delta_z|_max = {dz_max:.2e}  (must be 0.00e+00)")
    assert dz_max == 0.0, "zero-init of hand delta-z head is broken"
    policy.train()

    print("[Sanity] Step-0 evaluation:")
    init_eval = evaluate(policy, test_loader, device)
    print(f"  hand_mse_full          = {init_eval['hand_mse_full']:.6f}")
    print(f"  hand_mse_no_correction = {init_eval['hand_mse_no_correction']:.6f}")
    print()

    # ── Training loop ──
    history = {
        "steps": [], "train_total": [], "train_hand": [],
        "train_drift": [], "train_lr": [],
        "eval_steps": [], "val_hand_full": [], "val_hand_no_corr": [],
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
        gt = batch["gt_hand_action"]

        out = policy(
            img_main=batch["img_main"],
            img_extra=batch["img_extra"],
            past_hand_win=batch["past_hand_win"],
        )
        hand_loss = F.mse_loss(out["hand_action"], gt)
        drift_loss = F.mse_loss(out["hand_action"], out["hand_no_corr"])
        loss = hand_loss + args.reg_drift * drift_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params(policy), args.clip_grad)
        optimizer.step()

        history["steps"].append(step)
        history["train_total"].append(loss.item())
        history["train_hand"].append(hand_loss.item())
        history["train_drift"].append(drift_loss.item())
        history["train_lr"].append(lr_schedule[step])

        if step % args.print_freq == 0:
            elapsed = time.time() - t0
            print(f"[Step {step:>5d}] total={loss.item():.6f}  "
                  f"hand={hand_loss.item():.6f}  "
                  f"drift={drift_loss.item():.6f}  "
                  f"lr={lr_schedule[step]:.2e}  ({elapsed:.0f}s)")

        if step > 0 and step % args.eval_freq == 0:
            ev = evaluate(policy, test_loader, device)
            history["eval_steps"].append(step)
            history["val_hand_full"].append(ev["hand_mse_full"])
            history["val_hand_no_corr"].append(ev["hand_mse_no_correction"])
            improved = ev["hand_mse_no_correction"] - ev["hand_mse_full"]
            print(f"           val hand={ev['hand_mse_full']:.6f}  "
                  f"no_corr={ev['hand_mse_no_correction']:.6f}  "
                  f"vision_gain={improved:+.6f}")

        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(policy, optimizer, step, args.output_dir, args)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    # ── Final eval + save ──
    save_checkpoint(policy, optimizer, args.total_steps, args.output_dir, args)
    ev = evaluate(policy, test_loader, device)
    history["eval_steps"].append(args.total_steps)
    history["val_hand_full"].append(ev["hand_mse_full"])
    history["val_hand_no_corr"].append(ev["hand_mse_no_correction"])

    print(f"\nFinal val: hand={ev['hand_mse_full']:.6f}  "
          f"no_corr={ev['hand_mse_no_correction']:.6f}  "
          f"vision_gain={ev['hand_mse_no_correction'] - ev['hand_mse_full']:+.6f}")
    print(f"  Vision-corrected hand prediction "
          f"{'BEATS' if ev['hand_mse_full'] < ev['hand_mse_no_correction'] else 'DOES NOT beat'} "
          f"the no-correction baseline.")

    save_training_curves(history, args.output_dir, args)
    print(f"\nDone! Checkpoints + figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
