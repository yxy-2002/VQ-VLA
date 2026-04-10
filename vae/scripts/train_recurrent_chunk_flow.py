"""
Train a recurrent latent flow prior with an internal hidden state.
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


HandActionChunkDataset = _load("model/hand_chunk_dataset.py", "hand_chunk_dataset").HandActionChunkDataset
HandActionRecurrentChunkFlow = _load(
    "model/hand_recurrent_chunk_flow.py",
    "hand_recurrent_chunk_flow",
).HandActionRecurrentChunkFlow
_utils = _load("model/utils.py", "utils")
cosine_scheduler = _utils.cosine_scheduler


def get_args():
    p = argparse.ArgumentParser(description="Train recurrent chunk flow prior")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--future_horizon", type=int, default=12)
    p.add_argument("--future_chunks", type=int, default=4)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--recurrent_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--monotonic_start_idx", type=int, default=2)
    p.add_argument("--num_hidden_layers", type=int, default=1)
    p.add_argument("--time_embed_dim", type=int, default=32)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--flow_weight", type=float, default=1.0)
    p.add_argument("--latent_reg_weight", type=float, default=0.02)
    p.add_argument("--rollout_mode", type=str, default="recon", choices=["teacher", "recon", "sample"])
    p.add_argument("--sample_align_weight", type=float, default=0.05)
    p.add_argument("--terminal_weight", type=float, default=0.05)
    p.add_argument("--integration_steps_train", type=int, default=24)
    p.add_argument("--integration_steps_eval", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_freq", type=int, default=1000)
    p.add_argument("--eval_freq", type=int, default=2000)
    p.add_argument("--save_freq", type=int, default=10000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def split_target_chunks(target, future_horizon, future_chunks):
    return [
        target[:, idx * future_horizon : (idx + 1) * future_horizon]
        for idx in range(future_chunks)
    ]


def choose_rollout_chunk(args, target_chunk, recon_chunk, sample_chunk):
    if args.rollout_mode == "teacher":
        return target_chunk
    if args.rollout_mode == "sample":
        return sample_chunk
    return recon_chunk


def compute_sequence_objective(model, history, target, args, integration_steps):
    chunks = split_target_chunks(target, args.future_horizon, args.future_chunks)
    hidden = model.encode_history(history)
    current_state = history[:, -1]

    base_terms = []
    recon_terms = []
    flow_terms = []
    reg_terms = []
    sample_terms = []
    terminal_terms = []

    for target_chunk in chunks:
        step_out = model.forward_step(hidden, current_state, target_chunk)
        base_terms.append(step_out["total_loss"])
        recon_terms.append(step_out["recon_loss"])
        flow_terms.append(step_out["flow_loss"])
        reg_terms.append(step_out["latent_reg"])

        sample_chunk = None
        if args.sample_align_weight > 0.0 or args.rollout_mode == "sample":
            sample_chunk, _, _, _ = model.sample_chunk(
                hidden,
                current_state,
                num_integration_steps=integration_steps,
                return_latent=True,
            )
            if args.sample_align_weight > 0.0:
                sample_terms.append(F.mse_loss(sample_chunk, target_chunk))

        if args.terminal_weight > 0.0:
            terminal_terms.append(F.mse_loss(step_out["recon"][:, -1], target_chunk[:, -1]))

        rollout_chunk = choose_rollout_chunk(args, target_chunk, step_out["recon"], sample_chunk)
        if args.rollout_mode == "recon":
            hidden = step_out["next_hidden"]
            current_state = step_out["next_state"]
        else:
            hidden = model.transition_hidden(hidden, rollout_chunk, current_state)
            current_state = rollout_chunk[:, -1]

    mean_base = torch.stack(base_terms).mean()
    mean_recon = torch.stack(recon_terms).mean()
    mean_flow = torch.stack(flow_terms).mean()
    mean_reg = torch.stack(reg_terms).mean()
    mean_sample = (
        torch.stack(sample_terms).mean()
        if sample_terms
        else torch.zeros((), device=history.device, dtype=history.dtype)
    )
    mean_terminal = (
        torch.stack(terminal_terms).mean()
        if terminal_terms
        else torch.zeros((), device=history.device, dtype=history.dtype)
    )
    objective = mean_base
    objective = objective + args.sample_align_weight * mean_sample
    objective = objective + args.terminal_weight * mean_terminal
    return {
        "objective": objective,
        "base_loss": mean_base,
        "recon_loss": mean_recon,
        "flow_loss": mean_flow,
        "latent_reg": mean_reg,
        "sample_loss": mean_sample,
        "terminal_loss": mean_terminal,
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    totals = {
        "objective": 0.0,
        "base_loss": 0.0,
        "recon_loss": 0.0,
        "flow_loss": 0.0,
        "latent_reg": 0.0,
        "sample_loss": 0.0,
        "terminal_loss": 0.0,
        "copy": 0.0,
    }
    total_n = 0
    for history, target in loader:
        history, target = history.to(device), target.to(device)
        out = compute_sequence_objective(model, history, target, args, integration_steps=args.integration_steps_eval)

        first_target = target[:, : args.future_horizon]
        copy_pred = history[:, -1:, :].expand(-1, first_target.shape[1], -1)
        copy_mse = F.mse_loss(copy_pred, first_target)

        batch_size = history.shape[0]
        total_n += batch_size
        for key in totals:
            if key == "copy":
                totals[key] += copy_mse.item() * batch_size
            else:
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

    train_dataset = HandActionChunkDataset(
        args.train_dir,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        noise_std=args.noise_std,
        future_chunks=args.future_chunks,
    )
    test_dataset = HandActionChunkDataset(
        args.test_dir,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        noise_std=0.0,
        future_chunks=args.future_chunks,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = HandActionRecurrentChunkFlow(
        action_dim=args.action_dim,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        hidden_dim=args.hidden_dim,
        recurrent_dim=args.recurrent_dim,
        latent_dim=args.latent_dim,
        monotonic_start_idx=args.monotonic_start_idx,
        num_hidden_layers=args.num_hidden_layers,
        time_embed_dim=args.time_embed_dim,
        recon_weight=args.recon_weight,
        flow_weight=args.flow_weight,
        latent_reg_weight=args.latent_reg_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    val = evaluate(model, test_loader, device, args)
    print(f"Copy-baseline MSE: {val['copy']:.6f}\n")
    print(
        f"Training for {args.total_steps} steps | window={args.window_size}, horizon={args.future_horizon}, "
        f"future_chunks={args.future_chunks}, latent={args.latent_dim}, recurrent={args.recurrent_dim}, "
        f"rollout_mode={args.rollout_mode}, sample_align={args.sample_align_weight}, "
        f"terminal={args.terminal_weight}, noise_std={args.noise_std}"
    )

    train_iter = iter(train_loader)
    best_objective = float("inf")

    for step in range(args.total_steps):
        try:
            history, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            history, target = next(train_iter)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[step]

        history, target = history.to(device), target.to(device)
        out = compute_sequence_objective(model, history, target, args, integration_steps=args.integration_steps_train)

        optimizer.zero_grad()
        out["objective"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()

        if step % args.print_freq == 0:
            print(
                f"[Step {step:>5d}] obj={out['objective'].item():.6f}  base={out['base_loss'].item():.6f}  "
                f"recon={out['recon_loss'].item():.6f}  flow={out['flow_loss'].item():.6f}  "
                f"latent_reg={out['latent_reg'].item():.6f}  sample={out['sample_loss'].item():.6f}  "
                f"terminal={out['terminal_loss'].item():.6f}  lr={lr_schedule[step]:.2e}"
            )

        if step % args.eval_freq == 0 and step > 0:
            val = evaluate(model, test_loader, device, args)
            best_objective = min(best_objective, val["objective"])
            print(
                f"           val_obj={val['objective']:.6f}  val_base={val['base_loss']:.6f}  "
                f"val_recon={val['recon_loss']:.6f}  val_flow={val['flow_loss']:.6f}  "
                f"val_latent_reg={val['latent_reg']:.6f}  val_sample={val['sample_loss']:.6f}  "
                f"val_terminal={val['terminal_loss']:.6f}  copy_baseline={val['copy']:.6f}  "
                f"best_val_obj={best_objective:.6f}"
            )

        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args.output_dir, args)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    save_checkpoint(model, optimizer, args.total_steps, args.output_dir, args)
    val = evaluate(model, test_loader, device, args)
    print(
        f"\nFinal: val_obj={val['objective']:.6f}  val_base={val['base_loss']:.6f}  "
        f"val_recon={val['recon_loss']:.6f}  val_flow={val['flow_loss']:.6f}  "
        f"val_latent_reg={val['latent_reg']:.6f}  val_sample={val['sample_loss']:.6f}  "
        f"val_terminal={val['terminal_loss']:.6f}  copy_baseline={val['copy']:.6f}"
    )
    print(f"Done! Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
