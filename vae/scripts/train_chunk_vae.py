"""
Train a chunked hand-action VAE.

Usage:
    python vae/scripts/train_chunk_vae.py \
        --train_dir data/20260327-11:10:43/demos/success/train \
        --test_dir  data/20260327-11:10:43/demos/success/test \
        --output_dir outputs/chunk_vae
"""

import argparse
import importlib.util
import os

import torch
from torch.utils.data import DataLoader

_vae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionChunkVAE = _load("model/hand_chunk_vae.py", "hand_chunk_vae").HandActionChunkVAE
HandActionChunkDataset = _load("model/hand_chunk_dataset.py", "hand_chunk_dataset").HandActionChunkDataset
_utils = _load("model/utils.py", "utils")
cosine_scheduler = _utils.cosine_scheduler
beta_annealing_schedule = _utils.beta_annealing_schedule


def get_args():
    p = argparse.ArgumentParser(description="Train chunked hand-action Chunk VAE")
    # Data
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--future_horizon", type=int, default=8)
    p.add_argument("--noise_std", type=float, default=0.01)
    # Model
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--beta_warmup", type=int, default=2000)
    p.add_argument("--encoder_type", type=str, default="mlp", choices=["mlp", "causal_conv"])
    p.add_argument("--num_hidden_layers", type=int, default=1)
    p.add_argument("--free_bits", type=float, default=0.0)
    # Training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--print_freq", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=1000)
    p.add_argument("--save_freq", type=int, default=5000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (recon_loss, kl_loss, copy_baseline_mse)."""
    model.eval()
    total_recon, total_kl, total_copy, total_n = 0.0, 0.0, 0.0, 0

    for window, target in loader:
        window, target = window.to(device), target.to(device)
        _, recon_loss, kl_loss, _, _ = model(window, target)

        # Copy baseline: repeat last history frame for the whole chunk
        copy_pred = window[:, -1:, :].expand(-1, target.shape[1], -1)
        copy_mse = torch.nn.functional.mse_loss(copy_pred, target).item()

        B = window.shape[0]
        total_recon += recon_loss.item() * B
        total_kl += kl_loss.item() * B
        total_copy += copy_mse * B
        total_n += B

    model.train()
    return total_recon / total_n, total_kl / total_n, total_copy / total_n


def save_checkpoint(model, optimizer, step, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }
    torch.save(state, os.path.join(output_dir, f"checkpoint-{step}.pth"))
    torch.save(state, os.path.join(output_dir, "checkpoint.pth"))


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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Chunk-VAE Training  (window={args.window_size}, horizon={args.future_horizon}, "
        f"noise={args.noise_std}, β={args.beta}, latent={args.latent_dim}, "
        f"encoder={args.encoder_type})",
        fontsize=13,
    )

    window = min(100, len(steps) // 10)
    kernel = ones(window) / window if window > 1 else None

    def _smooth(vals):
        if kernel is not None and len(vals) > len(kernel):
            return convolve(vals, kernel, mode="valid"), steps[: len(convolve(vals, kernel, mode="valid"))]
        return vals, steps

    # 1. Total loss
    ax = axes[0, 0]
    ax.plot(steps, history["train_total"], alpha=0.3, color="blue", linewidth=0.5)
    sm, sm_s = _smooth(history["train_total"])
    if kernel is not None:
        ax.plot(sm_s, sm, color="blue", linewidth=1.5, label="Smoothed")
        ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title("Total Loss (Recon + β × KL)")
    ax.grid(True, alpha=0.3)

    # 2. Reconstruction loss
    ax = axes[0, 1]
    ax.plot(steps, history["train_recon"], alpha=0.3, color="green", linewidth=0.5)
    sm, sm_s = _smooth(history["train_recon"])
    if kernel is not None:
        ax.plot(sm_s, sm, color="green", linewidth=1.5, label="Train (smoothed)")
    if eval_steps:
        ax.plot(eval_steps, history["val_recon"], "ro-", markersize=4, linewidth=1.5, label="Validation")
        ax.axhline(y=history["val_copy"][0], color="gray", linestyle="--", linewidth=1, label="Copy Baseline")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Step")
    ax.set_title("Reconstruction Loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. KL divergence
    ax = axes[0, 2]
    ax.plot(steps, history["train_kl"], alpha=0.3, color="orange", linewidth=0.5)
    sm, sm_s = _smooth(history["train_kl"])
    if kernel is not None:
        ax.plot(sm_s, sm, color="orange", linewidth=1.5, label="Train (smoothed)")
    if eval_steps:
        ax.plot(eval_steps, history["val_kl"], "ro-", markersize=4, linewidth=1.5, label="Validation")
    ax.set_ylabel("KL Divergence")
    ax.set_xlabel("Step")
    ax.set_title("KL(posterior || learned prior)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Beta schedule
    ax = axes[1, 0]
    ax.plot(steps, history["train_beta"], color="purple", linewidth=1.5)
    ax.set_ylabel("β")
    ax.set_xlabel("Step")
    ax.set_title("KL Weight (β) Annealing")
    ax.grid(True, alpha=0.3)

    # 5. Learning rate schedule
    ax = axes[1, 1]
    ax.plot(steps, history["train_lr"], color="teal", linewidth=1.5)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_title("Learning Rate (Cosine Schedule)")
    ax.grid(True, alpha=0.3)

    # 6. Loss decomposition
    ax = axes[1, 2]
    beta_kl = [b * k for b, k in zip(history["train_beta"], history["train_kl"])]
    if kernel is not None:
        sm_r, sm_rs = _smooth(history["train_recon"])
        sm_bk, sm_bks = _smooth(beta_kl)
        ax.plot(sm_rs, sm_r, color="green", linewidth=1.5, label="Recon Loss")
        ax.plot(sm_bks, sm_bk, color="red", linewidth=1.5, label="β × KL")
    else:
        ax.plot(steps, history["train_recon"], color="green", linewidth=1, label="Recon Loss")
        ax.plot(steps, beta_kl, color="red", linewidth=1, label="β × KL")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title("Loss Decomposition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {fig_path}")


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    train_dataset = HandActionChunkDataset(
        args.train_dir,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        noise_std=args.noise_std,
    )
    test_dataset = HandActionChunkDataset(
        args.test_dir,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        noise_std=0.0,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = HandActionChunkVAE(
        action_dim=args.action_dim,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        encoder_type=args.encoder_type,
        num_hidden_layers=args.num_hidden_layers,
        free_bits=args.free_bits,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps)
    beta_schedule = beta_annealing_schedule(args.beta, args.total_steps, warmup_steps=args.beta_warmup)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    val_recon, val_kl, val_copy = evaluate(model, test_loader, device)
    print(f"Copy-baseline MSE: {val_copy:.6f}  (this is the bar to beat)\n")

    print(
        f"Training for {args.total_steps} steps | "
        f"window={args.window_size}, horizon={args.future_horizon}, "
        f"latent={args.latent_dim}, β={args.beta}, noise={args.noise_std}, device={args.device}"
    )

    # Metric history
    history = {
        "steps": [], "train_total": [], "train_recon": [], "train_kl": [],
        "train_beta": [], "train_lr": [],
        "eval_steps": [], "val_recon": [], "val_kl": [], "val_copy": [],
    }

    train_iter = iter(train_loader)
    best_val_recon = float("inf")

    for step in range(args.total_steps):
        try:
            window, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            window, target = next(train_iter)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[step]
        model.beta = beta_schedule[step]

        window, target = window.to(device), target.to(device)
        _, recon_loss, kl_loss, total_loss, _ = model(window, target)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()

        # Log
        history["steps"].append(step)
        history["train_total"].append(total_loss.item())
        history["train_recon"].append(recon_loss.item())
        history["train_kl"].append(kl_loss.item())
        history["train_beta"].append(model.beta)
        history["train_lr"].append(lr_schedule[step])

        if step % args.print_freq == 0:
            print(
                f"[Step {step:>5d}] total={total_loss.item():.6f}  "
                f"recon={recon_loss.item():.6f}  kl={kl_loss.item():.4f}  "
                f"beta={model.beta:.4f}  lr={lr_schedule[step]:.2e}"
            )

        if step % args.eval_freq == 0 and step > 0:
            vr, vk, vc = evaluate(model, test_loader, device)
            history["eval_steps"].append(step)
            history["val_recon"].append(vr)
            history["val_kl"].append(vk)
            history["val_copy"].append(vc)
            best_val_recon = min(best_val_recon, vr)
            print(
                f"           val_recon={vr:.6f}  val_kl={vk:.4f}  "
                f"copy_baseline={vc:.6f}  best_val_recon={best_val_recon:.6f}"
            )

        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args.output_dir, args)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    # Final
    save_checkpoint(model, optimizer, args.total_steps, args.output_dir, args)
    vr, vk, vc = evaluate(model, test_loader, device)
    history["eval_steps"].append(args.total_steps)
    history["val_recon"].append(vr)
    history["val_kl"].append(vk)
    history["val_copy"].append(vc)

    print(f"\nFinal: val_recon={vr:.6f}  val_kl={vk:.4f}  copy_baseline={vc:.6f}")
    print(f"  Chunk VAE {'beats' if vr < vc else 'does NOT beat'} copy baseline "
          f"({vr:.6f} vs {vc:.6f})")

    save_training_curves(history, args.output_dir, args)
    print(f"\nDone! Checkpoints + figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
