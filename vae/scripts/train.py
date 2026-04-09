"""
Train Hand Action VAE (Stage 1: pure state, no vision).

Usage:
    python vae/scripts/train.py \
        --train_dir data/20260327-11:10:43/demos/success/train \
        --test_dir data/20260327-11:10:43/demos/success/test \
        --output_dir outputs/hand_vae
"""

import argparse
import importlib.util
import os
import sys

import torch
from torch.utils.data import DataLoader

_vae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionVAE = _load("model/hand_vae.py", "hand_vae").HandActionVAE
HandActionWindowDataset = _load("model/hand_dataset.py", "hand_dataset").HandActionWindowDataset
_utils = _load("model/utils.py", "utils")
cosine_scheduler = _utils.cosine_scheduler
beta_annealing_schedule = _utils.beta_annealing_schedule


def get_args():
    p = argparse.ArgumentParser(description="Train Hand Action VAE")
    # Data
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/dim_2_best")
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--noise_std", type=float, default=0.01, help="Gaussian noise std on input window (0=off)")
    # Model — defaults match the empirically-best latent_dim=2 recipe (see README "Sweep findings").
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.001, help="Target KL weight")
    p.add_argument("--beta_warmup", type=int, default=2000, help="Steps to anneal beta from 0 to target")
    p.add_argument("--encoder_type", type=str, default="mlp", choices=["mlp", "causal_conv"])
    p.add_argument("--num_hidden_layers", type=int, default=1,
                   help="Number of hidden→hidden blocks in encoder/decoder MLPs (1 = original 2-Linear stack)")
    p.add_argument("--recon_aux_weight", type=float, default=0.0,
                   help="Weight for auxiliary input-reconstruction loss (0 = disabled)")
    p.add_argument("--free_bits", type=float, default=0.0,
                   help="Per-dim KL floor in nats (0 = standard VAE; 0.5 typical)")
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
    """Evaluate: return recon_loss, kl_loss, and copy-baseline MSE."""
    model.eval()
    total_recon, total_kl, total_copy, total_n = 0.0, 0.0, 0.0, 0

    for window, target in loader:
        window, target = window.to(device), target.to(device)
        _, recon_loss, kl_loss, _, _, _ = model(window, target)

        # Copy baseline: predict t+1 = t (last frame of window)
        copy_pred = window[:, -1, :]
        copy_mse = torch.nn.functional.mse_loss(copy_pred, target).item()

        B = window.shape[0]
        total_recon += recon_loss.item() * B
        total_kl += kl_loss.item() * B
        total_copy += copy_mse * B
        total_n += B

    model.train()
    return total_recon / total_n, total_kl / total_n, total_copy / total_n


def save_checkpoint(model, optimizer, step, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        # Diagnostic-only: free_bits doesn't affect inference, but it's nice to know how the model was trained.
        "recon_aux_weight": getattr(model, "recon_aux_weight", 0.0),
        "free_bits": getattr(model, "free_bits", 0.0),
    }
    torch.save(state, os.path.join(output_dir, f"checkpoint-{step}.pth"))
    torch.save(state, os.path.join(output_dir, "checkpoint.pth"))


def save_training_curves(history, output_dir, args):
    """Save training metric curves as figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping training curves")
        return

    steps = history["steps"]
    eval_steps = history["eval_steps"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"VAE Training Curves  (noise_std={args.noise_std}, beta={args.beta}, "
                 f"latent_dim={args.latent_dim}, encoder={args.encoder_type})", fontsize=13)

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(steps, history["train_total"], alpha=0.3, color="blue", linewidth=0.5)
    # Smoothed version
    window = min(100, len(steps) // 10)
    if window > 1:
        from numpy import convolve, ones
        kernel = ones(window) / window
        smoothed = convolve(history["train_total"], kernel, mode="valid")
        ax.plot(steps[:len(smoothed)], smoothed, color="blue", linewidth=1.5, label="Smoothed")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title("Total Loss (Reconstruction + β × KL Divergence)")
    ax.grid(True, alpha=0.3)
    if window > 1:
        ax.legend()

    # 2. Reconstruction Loss (train + val)
    ax = axes[0, 1]
    ax.plot(steps, history["train_recon"], alpha=0.3, color="green", linewidth=0.5)
    if window > 1:
        smoothed = convolve(history["train_recon"], kernel, mode="valid")
        ax.plot(steps[:len(smoothed)], smoothed, color="green", linewidth=1.5, label="Train (smoothed)")
    if eval_steps:
        ax.plot(eval_steps, history["val_recon"], "ro-", markersize=4, linewidth=1.5, label="Validation")
        ax.axhline(y=history["val_copy"][0], color="gray", linestyle="--", linewidth=1, label="Copy Baseline")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Step")
    ax.set_title("Reconstruction Loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. KL Divergence (train + val)
    ax = axes[0, 2]
    ax.plot(steps, history["train_kl"], alpha=0.3, color="orange", linewidth=0.5)
    if window > 1:
        smoothed = convolve(history["train_kl"], kernel, mode="valid")
        ax.plot(steps[:len(smoothed)], smoothed, color="orange", linewidth=1.5, label="Train (smoothed)")
    if eval_steps:
        ax.plot(eval_steps, history["val_kl"], "ro-", markersize=4, linewidth=1.5, label="Validation")
    ax.set_ylabel("KL Divergence")
    ax.set_xlabel("Step")
    ax.set_title("KL Divergence  KL(q(z|x) || N(0,1))")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Beta Schedule
    ax = axes[1, 0]
    ax.plot(steps, history["train_beta"], color="purple", linewidth=1.5)
    ax.set_ylabel("β")
    ax.set_xlabel("Step")
    ax.set_title("KL Weight (β) Annealing Schedule")
    ax.grid(True, alpha=0.3)

    # 5. Learning Rate Schedule
    ax = axes[1, 1]
    ax.plot(steps, history["train_lr"], color="teal", linewidth=1.5)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_title("Learning Rate (Cosine Schedule)")
    ax.grid(True, alpha=0.3)

    # 6. Weighted KL (beta * KL) vs Reconstruction
    ax = axes[1, 2]
    beta_kl = [b * k for b, k in zip(history["train_beta"], history["train_kl"])]
    if window > 1:
        smoothed_recon = convolve(history["train_recon"], kernel, mode="valid")
        smoothed_bkl = convolve(beta_kl, kernel, mode="valid")
        ax.plot(steps[:len(smoothed_recon)], smoothed_recon, color="green", linewidth=1.5, label="Reconstruction Loss")
        ax.plot(steps[:len(smoothed_bkl)], smoothed_bkl, color="red", linewidth=1.5, label="β × KL Divergence")
    else:
        ax.plot(steps, history["train_recon"], color="green", linewidth=1, label="Reconstruction Loss")
        ax.plot(steps, beta_kl, color="red", linewidth=1, label="β × KL Divergence")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title("Loss Decomposition (Reconstruction vs β × KL)")
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
    train_dataset = HandActionWindowDataset(args.train_dir, window_size=args.window_size, noise_std=args.noise_std)
    test_dataset = HandActionWindowDataset(args.test_dir, window_size=args.window_size, noise_std=0.0)  # no noise for eval
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = HandActionVAE(
        action_dim=args.action_dim,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        encoder_type=args.encoder_type,
        num_hidden_layers=args.num_hidden_layers,
        recon_aux_weight=args.recon_aux_weight,
        free_bits=args.free_bits,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & schedules
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps)
    beta_schedule = beta_annealing_schedule(args.beta, args.total_steps, warmup_steps=args.beta_warmup)

    # Initial evaluation (copy baseline)
    val_recon, val_kl, val_copy = evaluate(model, test_loader, device)
    print(f"Copy-baseline MSE: {val_copy:.6f}  (this is the bar to beat)\n")

    # Training loop with metric logging
    train_iter = iter(train_loader)
    model.train()

    print(f"Training for {args.total_steps} steps")
    print(f"  Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    print(f"  Latent dim: {args.latent_dim}, Beta: 0→{args.beta} over {args.beta_warmup} steps")
    print(f"  Noise std: {args.noise_std}\n")

    # Metric history for plotting
    history = {
        "steps": [], "train_total": [], "train_recon": [], "train_kl": [],
        "train_beta": [], "train_lr": [],
        "eval_steps": [], "val_recon": [], "val_kl": [], "val_copy": [],
    }

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
        pred, recon_loss, kl_loss, total_loss, mu, log_var = model(window, target)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()

        # Log train metrics
        history["steps"].append(step)
        history["train_total"].append(total_loss.item())
        history["train_recon"].append(recon_loss.item())
        history["train_kl"].append(kl_loss.item())
        history["train_beta"].append(model.beta)
        history["train_lr"].append(lr_schedule[step])

        if step % args.print_freq == 0:
            print(f"[Step {step:>5d}] total={total_loss.item():.6f}  "
                  f"recon={recon_loss.item():.6f}  kl={kl_loss.item():.4f}  "
                  f"beta={model.beta:.4f}  lr={lr_schedule[step]:.2e}")

        if step % args.eval_freq == 0 and step > 0:
            vr, vk, vc = evaluate(model, test_loader, device)
            history["eval_steps"].append(step)
            history["val_recon"].append(vr)
            history["val_kl"].append(vk)
            history["val_copy"].append(vc)
            print(f"           val_recon={vr:.6f}  val_kl={vk:.4f}  "
                  f"copy_baseline={vc:.6f}")

        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args.output_dir)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    # Final eval
    save_checkpoint(model, optimizer, args.total_steps, args.output_dir)
    vr, vk, vc = evaluate(model, test_loader, device)
    history["eval_steps"].append(args.total_steps)
    history["val_recon"].append(vr)
    history["val_kl"].append(vk)
    history["val_copy"].append(vc)

    print(f"\nFinal: val_recon={vr:.6f}  val_kl={vk:.4f}  copy_baseline={vc:.6f}")
    print(f"  VAE {'beats' if vr < vc else 'does NOT beat'} copy baseline "
          f"({vr:.6f} vs {vc:.6f})")

    # Save training curves
    save_training_curves(history, args.output_dir, args)
    print(f"\nDone! Checkpoints + figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
