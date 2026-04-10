"""
Train the stochastic recurrent state-space hand prior from trajectory starts.

Compared with random sliding-window training, this rollout-aligned setup keeps the
episode reset state distinct, so the prior can learn "early open hold" and then
transition into closing later in the sequence.
"""

import argparse
import importlib.util
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_vae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandTrajectoryChunkSequenceDataset = _load(
    "model/hand_trajectory_chunk_dataset.py",
    "hand_trajectory_chunk_dataset",
).HandTrajectoryChunkSequenceDataset
HandActionStateSpaceChunkPrior = _load(
    "model/hand_state_space_chunk_prior.py",
    "hand_state_space_chunk_prior",
).HandActionStateSpaceChunkPrior
_utils = _load("model/utils.py", "utils")
cosine_scheduler = _utils.cosine_scheduler


def get_args():
    p = argparse.ArgumentParser(description="Train state-space chunk prior from trajectory starts")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--future_horizon", type=int, default=12)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--recurrent_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--monotonic_start_idx", type=int, default=2)
    p.add_argument("--monotonic_output_mode", type=str, default="exp", choices=["exp", "plateau"])
    p.add_argument("--num_hidden_layers", type=int, default=1)
    p.add_argument("--latent_transition_scale", type=float, default=0.5)
    p.add_argument("--hidden_update_rate", type=float, default=0.35)
    p.add_argument("--rollout_mode", type=str, default="recon", choices=["teacher", "recon"])
    p.add_argument("--kl_weight", type=float, default=0.02)
    p.add_argument("--latent_step_weight", type=float, default=0.02)
    p.add_argument("--terminal_weight", type=float, default=0.02)
    p.add_argument("--first_chunk_weight", type=float, default=1.0)
    p.add_argument("--anti_drop_weight", type=float, default=0.0)
    p.add_argument("--anti_drop_threshold", type=float, default=0.04)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_freq", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=500)
    p.add_argument("--save_freq", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def collate_sequences(batch):
    batch_size = len(batch)
    window_size, action_dim = batch[0]["seed_window"].shape
    future_horizon = batch[0]["target_chunks"].shape[1]
    max_chunks = max(item["num_chunks"] for item in batch)

    seed_windows = torch.zeros(batch_size, window_size, action_dim, dtype=torch.float32)
    initial_states = torch.zeros(batch_size, action_dim, dtype=torch.float32)
    target_chunks = torch.zeros(batch_size, max_chunks, future_horizon, action_dim, dtype=torch.float32)
    chunk_mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool)

    for idx, item in enumerate(batch):
        num_chunks = item["num_chunks"]
        seed_windows[idx] = item["seed_window"]
        initial_states[idx] = item["initial_state"]
        target_chunks[idx, :num_chunks] = item["target_chunks"]
        if num_chunks < max_chunks:
            target_chunks[idx, num_chunks:] = item["target_chunks"][-1]
        chunk_mask[idx, :num_chunks] = True

    return seed_windows, initial_states, target_chunks, chunk_mask


def kl_divergence_per_sample(post_mu, post_log_var, prior_mu, prior_log_var):
    post_var = torch.exp(post_log_var)
    prior_var = torch.exp(prior_log_var)
    kl = prior_log_var - post_log_var
    kl = kl + (post_var + (post_mu - prior_mu).pow(2)) / prior_var
    kl = kl - 1.0
    return 0.5 * kl.sum(dim=-1)


def masked_mean(values, mask):
    weight = mask.float()
    denom = weight.sum().clamp_min(1.0)
    return (values * weight).sum() / denom


def directional_antidrop_per_sample(current_state, target_chunk, recon, threshold):
    target_delta = target_chunk[:, -1, :] - current_state
    direction = torch.sign(target_delta)
    active = (target_delta.abs() > threshold).float()

    prev = torch.cat([current_state.unsqueeze(1), recon[:, :-1]], dim=1)
    pred_delta = recon - prev
    signed_delta = pred_delta * direction.unsqueeze(1)
    penalty = F.relu(-signed_delta) * active.unsqueeze(1)

    denom = (active.sum(dim=-1) * recon.shape[1]).clamp_min(1.0)
    return penalty.sum(dim=(1, 2)) / denom


def compute_sequence_objective(model, seed_windows, initial_states, target_chunks, chunk_mask, args):
    hidden = model.encode_history(seed_windows)
    current_state = initial_states
    prev_z = torch.zeros(seed_windows.shape[0], model.latent_dim, device=seed_windows.device, dtype=seed_windows.dtype)

    objective_sum = torch.tensor(0.0, device=seed_windows.device)
    recon_sum = torch.tensor(0.0, device=seed_windows.device)
    kl_sum = torch.tensor(0.0, device=seed_windows.device)
    latent_step_sum = torch.tensor(0.0, device=seed_windows.device)
    terminal_sum = torch.tensor(0.0, device=seed_windows.device)
    anti_drop_sum = torch.tensor(0.0, device=seed_windows.device)
    copy_sum = torch.tensor(0.0, device=seed_windows.device)
    total_weight = torch.tensor(0.0, device=seed_windows.device)

    num_steps = target_chunks.shape[1]
    for step_idx in range(num_steps):
        active = chunk_mask[:, step_idx]
        if not torch.any(active):
            break

        target_chunk = target_chunks[:, step_idx]
        step_out = model.forward_step(hidden, current_state, prev_z, target_chunk)

        recon_ps = (step_out["recon"] - target_chunk).pow(2).mean(dim=(1, 2))
        kl_ps = kl_divergence_per_sample(
            step_out["post_mu"],
            step_out["post_log_var"],
            step_out["prior_mu"],
            step_out["prior_log_var"],
        )
        latent_step_ps = (step_out["post_mu"] - prev_z).pow(2).mean(dim=-1)
        terminal_ps = (step_out["recon"][:, -1] - target_chunk[:, -1]).pow(2).mean(dim=-1)
        anti_drop_ps = directional_antidrop_per_sample(
            current_state=current_state,
            target_chunk=target_chunk,
            recon=step_out["recon"],
            threshold=args.anti_drop_threshold,
        )
        copy_pred = current_state.unsqueeze(1).expand_as(target_chunk)
        copy_ps = (copy_pred - target_chunk).pow(2).mean(dim=(1, 2))

        step_weight = active.float()
        if step_idx == 0 and args.first_chunk_weight != 1.0:
            step_weight = step_weight * args.first_chunk_weight

        objective_ps = recon_ps
        objective_ps = objective_ps + args.kl_weight * kl_ps
        objective_ps = objective_ps + args.latent_step_weight * latent_step_ps
        objective_ps = objective_ps + args.terminal_weight * terminal_ps
        objective_ps = objective_ps + args.anti_drop_weight * anti_drop_ps

        objective_sum = objective_sum + (objective_ps * step_weight).sum()
        recon_sum = recon_sum + (recon_ps * step_weight).sum()
        kl_sum = kl_sum + (kl_ps * step_weight).sum()
        latent_step_sum = latent_step_sum + (latent_step_ps * step_weight).sum()
        terminal_sum = terminal_sum + (terminal_ps * step_weight).sum()
        anti_drop_sum = anti_drop_sum + (anti_drop_ps * step_weight).sum()
        copy_sum = copy_sum + (copy_ps * step_weight).sum()
        total_weight = total_weight + step_weight.sum()

        if args.rollout_mode == "teacher":
            next_hidden = model.transition_hidden(hidden, target_chunk, current_state, step_out["z"])
            next_state = target_chunk[:, -1]
        else:
            next_hidden = step_out["next_hidden"]
            next_state = step_out["next_state"]

        active_f = active.unsqueeze(-1)
        hidden = torch.where(active_f, next_hidden, hidden)
        current_state = torch.where(active_f, next_state, current_state)
        prev_z = torch.where(active_f, step_out["z"], prev_z)

    denom = total_weight.clamp_min(1.0)
    return {
        "objective": objective_sum / denom,
        "recon_loss": recon_sum / denom,
        "kl_loss": kl_sum / denom,
        "latent_step_loss": latent_step_sum / denom,
        "terminal_loss": terminal_sum / denom,
        "anti_drop_loss": anti_drop_sum / denom,
        "copy": copy_sum / denom,
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    totals = {
        "objective": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "latent_step_loss": 0.0,
        "terminal_loss": 0.0,
        "anti_drop_loss": 0.0,
        "copy": 0.0,
    }
    total_n = 0
    for seed_windows, initial_states, target_chunks, chunk_mask in loader:
        seed_windows = seed_windows.to(device)
        initial_states = initial_states.to(device)
        target_chunks = target_chunks.to(device)
        chunk_mask = chunk_mask.to(device)
        out = compute_sequence_objective(model, seed_windows, initial_states, target_chunks, chunk_mask, args)
        batch_size = seed_windows.shape[0]
        total_n += batch_size
        for key in totals:
            totals[key] += out[key].item() * batch_size
    model.train()
    return {key: value / total_n for key, value in totals.items()}


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

    train_dataset = HandTrajectoryChunkSequenceDataset(
        args.train_dir,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        noise_std=args.noise_std,
    )
    test_dataset = HandTrajectoryChunkSequenceDataset(
        args.test_dir,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        noise_std=0.0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_sequences,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )

    model = HandActionStateSpaceChunkPrior(
        action_dim=args.action_dim,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        hidden_dim=args.hidden_dim,
        recurrent_dim=args.recurrent_dim,
        latent_dim=args.latent_dim,
        monotonic_start_idx=args.monotonic_start_idx,
        monotonic_output_mode=args.monotonic_output_mode,
        num_hidden_layers=args.num_hidden_layers,
        latent_transition_scale=args.latent_transition_scale,
        hidden_update_rate=args.hidden_update_rate,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    val = evaluate(model, test_loader, device, args)
    print(f"Copy-baseline MSE: {val['copy']:.6f}\n")
    print(
        f"Training for {args.total_steps} steps | window={args.window_size}, horizon={args.future_horizon}, "
        f"latent={args.latent_dim}, recurrent={args.recurrent_dim}, rollout_mode={args.rollout_mode}, "
        f"kl_weight={args.kl_weight}, latent_step={args.latent_step_weight}, terminal={args.terminal_weight}, "
        f"first_chunk={args.first_chunk_weight}, anti_drop={args.anti_drop_weight}, "
        f"transition_scale={args.latent_transition_scale}, hidden_rate={args.hidden_update_rate}, "
        f"noise_std={args.noise_std}"
    )

    train_iter = iter(train_loader)
    best_objective = float("inf")

    for step in range(args.total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[step]

        seed_windows, initial_states, target_chunks, chunk_mask = [x.to(device) for x in batch]
        out = compute_sequence_objective(model, seed_windows, initial_states, target_chunks, chunk_mask, args)

        optimizer.zero_grad()
        out["objective"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()

        if step % args.print_freq == 0:
            print(
                f"[Step {step:>5d}] obj={out['objective'].item():.6f}  recon={out['recon_loss'].item():.6f}  "
                f"kl={out['kl_loss'].item():.6f}  latent_step={out['latent_step_loss'].item():.6f}  "
                f"terminal={out['terminal_loss'].item():.6f}  anti_drop={out['anti_drop_loss'].item():.6f}  "
                f"lr={lr_schedule[step]:.2e}"
            )

        if step % args.eval_freq == 0 and step > 0:
            val = evaluate(model, test_loader, device, args)
            best_objective = min(best_objective, val["objective"])
            print(
                f"           val_obj={val['objective']:.6f}  val_recon={val['recon_loss']:.6f}  "
                f"val_kl={val['kl_loss']:.6f}  val_latent_step={val['latent_step_loss']:.6f}  "
                f"val_terminal={val['terminal_loss']:.6f}  val_anti_drop={val['anti_drop_loss']:.6f}  "
                f"copy_baseline={val['copy']:.6f}  best_val_obj={best_objective:.6f}"
            )

        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args.output_dir, args)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    save_checkpoint(model, optimizer, args.total_steps, args.output_dir, args)
    val = evaluate(model, test_loader, device, args)
    print(
        f"\nFinal: val_obj={val['objective']:.6f}  val_recon={val['recon_loss']:.6f}  "
        f"val_kl={val['kl_loss']:.6f}  val_latent_step={val['latent_step_loss']:.6f}  "
        f"val_terminal={val['terminal_loss']:.6f}  val_anti_drop={val['anti_drop_loss']:.6f}  "
        f"copy_baseline={val['copy']:.6f}"
    )
    print(f"Done! Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
