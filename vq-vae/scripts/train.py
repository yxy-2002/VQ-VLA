"""
Training script for simplified Hand VQ-VAE.

Usage:
    python vq-vae/scripts/train.py \
        --train_dir data/20260327-11:10:43/demos/success/train \
        --test_dir data/20260327-11:10:43/demos/success/test \
        --output_dir outputs/hand_vqvae
"""

import argparse
import importlib.util
import os
import sys

import torch
from torch.utils.data import DataLoader

_vqvae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vqvae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionDataset = _load("model/hand_dataset.py", "hand_dataset").HandActionDataset
HandVQVAE = _load("model/hand_vqvae.py", "hand_vqvae").HandVQVAE
cosine_scheduler = _load("model/utils.py", "utils").cosine_scheduler


def get_args():
    p = argparse.ArgumentParser(description="Train Hand VQ-VAE")
    # Data
    p.add_argument("--train_dir", type=str, required=True, help="Path to train trajectory dir")
    p.add_argument("--test_dir", type=str, required=True, help="Path to test trajectory dir")
    p.add_argument("--output_dir", type=str, default="outputs/hand_vqvae")
    # Model
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_vq_layers", type=int, default=2)
    p.add_argument("--codebook_size", type=int, default=4)
    p.add_argument("--commitment_weight", type=float, default=5.0)
    # Training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--print_freq", type=int, default=100)
    p.add_argument("--eval_freq", type=int, default=500)
    p.add_argument("--save_freq", type=int, default=2000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        _, _, recon_loss, _, _ = model(batch)
        total_loss += recon_loss.item() * batch.shape[0]
        total_n += batch.shape[0]
    model.train()
    return total_loss / total_n


@torch.no_grad()
def analyze_codebook(model, loader, device):
    model.eval()
    all_indices = []
    for batch in loader:
        indices = model.encode(batch.to(device))
        all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices)

    codebook_size = model.vq.layers[0].codebook_size
    total_combos = codebook_size ** all_indices.shape[1]
    combos = all_indices[:, 0] * codebook_size + all_indices[:, 1]
    unique_combos = combos.unique()

    print(f"\n{'=' * 50}")
    print(f"Codebook utilization: {len(unique_combos)}/{total_combos} combos used")
    print(f"{'=' * 50}")
    for layer_idx in range(all_indices.shape[1]):
        used = all_indices[:, layer_idx].unique()
        print(f"  Layer {layer_idx}: {len(used)}/{codebook_size} entries used")

    print(f"\n  {'Combo':>6} {'L0':>3} {'L1':>3} {'Count':>7} {'Pct':>7}")
    print(f"  {'-' * 30}")
    for c in range(total_combos):
        count = (combos == c).sum().item()
        if count > 0:
            l0, l1 = c // codebook_size, c % codebook_size
            print(f"  {c:>6} {l0:>3} {l1:>3} {count:>7} {count / len(combos) * 100:>6.1f}%")
    model.train()


def save_checkpoint(model, optimizer, step, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    torch.save(state, os.path.join(output_dir, f"checkpoint-{step}.pth"))
    torch.save(state, os.path.join(output_dir, "checkpoint.pth"))


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = HandActionDataset(args.train_dir)
    test_dataset = HandActionDataset(args.test_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = HandVQVAE(
        action_dim=args.action_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        num_vq_layers=args.num_vq_layers, codebook_size=args.codebook_size,
        commitment_weight=args.commitment_weight,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps)

    train_iter = iter(train_loader)
    model.train()

    print(f"\nTraining for {args.total_steps} steps, batch_size={args.batch_size}")
    print(f"  Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    print(f"  Codebook: {args.num_vq_layers} layers x {args.codebook_size} entries = "
          f"{args.codebook_size ** args.num_vq_layers} combos\n")

    for step in range(args.total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[step]

        batch = batch.to(device)
        x_recon, indices, recon_loss, commit_loss, total_loss = model(batch)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()

        if step % args.print_freq == 0:
            print(f"[Step {step:>5d}] total={total_loss.item():.6f}  "
                  f"recon={recon_loss.item():.6f}  commit={commit_loss.item():.6f}  "
                  f"lr={lr_schedule[step]:.2e}")
        if step % args.eval_freq == 0:
            val_loss = evaluate(model, test_loader, device)
            print(f"           val_recon={val_loss:.6f}")
        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args.output_dir)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    save_checkpoint(model, optimizer, args.total_steps, args.output_dir)
    val_loss = evaluate(model, test_loader, device)
    print(f"\nFinal val_recon={val_loss:.6f}")
    analyze_codebook(model, test_loader, device)
    print(f"\nDone! Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
