"""
Train the improved chunk CVAE.
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


HandActionChunkCVAE = _load("model/hand_chunk_cvae.py", "hand_chunk_cvae").HandActionChunkCVAE
HandActionChunkDataset = _load("model/hand_chunk_dataset.py", "hand_chunk_dataset").HandActionChunkDataset
_utils = _load("model/utils.py", "utils")
cosine_scheduler = _utils.cosine_scheduler
beta_annealing_schedule = _utils.beta_annealing_schedule


def get_args():
    p = argparse.ArgumentParser(description="Train improved chunk CVAE")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--future_horizon", type=int, default=12)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--beta_warmup", type=int, default=2000)
    p.add_argument("--encoder_type", type=str, default="mlp", choices=["mlp", "causal_conv"])
    p.add_argument("--num_hidden_layers", type=int, default=1)
    p.add_argument("--free_bits", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_freq", type=int, default=500)
    p.add_argument("--eval_freq", type=int, default=1000)
    p.add_argument("--save_freq", type=int, default=5000)
    default_device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device", type=str, default=default_device)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_recon, total_kl, total_copy, total_n = 0.0, 0.0, 0.0, 0
    for window, target in loader:
        window, target = window.to(device), target.to(device)
        _, recon_loss, kl_loss, _, _ = model(window, target)

        copy_pred = window[:, -1:, :].expand(-1, target.shape[1], -1)
        copy_mse = torch.nn.functional.mse_loss(copy_pred, target).item()

        batch_size = window.shape[0]
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total_copy += copy_mse * batch_size
        total_n += batch_size

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


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

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

    model = HandActionChunkCVAE(
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
    print(f"Copy-baseline MSE: {val_copy:.6f}\n")
    print(
        f"Training for {args.total_steps} steps | window={args.window_size}, horizon={args.future_horizon}, "
        f"latent={args.latent_dim}, beta={args.beta}, noise_std={args.noise_std}, device={args.device}"
    )

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

        if step % args.print_freq == 0:
            print(
                f"[Step {step:>5d}] total={total_loss.item():.6f}  "
                f"recon={recon_loss.item():.6f}  kl={kl_loss.item():.4f}  "
                f"beta={model.beta:.4f}  lr={lr_schedule[step]:.2e}"
            )

        if step % args.eval_freq == 0 and step > 0:
            val_recon, val_kl, val_copy = evaluate(model, test_loader, device)
            best_val_recon = min(best_val_recon, val_recon)
            print(
                f"           val_recon={val_recon:.6f}  val_kl={val_kl:.4f}  "
                f"copy_baseline={val_copy:.6f}  best_val_recon={best_val_recon:.6f}"
            )

        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args.output_dir, args)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    save_checkpoint(model, optimizer, args.total_steps, args.output_dir, args)
    val_recon, val_kl, val_copy = evaluate(model, test_loader, device)
    print(f"\nFinal: val_recon={val_recon:.6f}  val_kl={val_kl:.4f}  copy_baseline={val_copy:.6f}")
    print(f"Done! Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
