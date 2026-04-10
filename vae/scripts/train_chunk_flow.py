"""
Train the conditional latent flow-matching hand prior.
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
HandActionChunkFlow = _load("model/hand_chunk_flow.py", "hand_chunk_flow").HandActionChunkFlow
_utils = _load("model/utils.py", "utils")
cosine_scheduler = _utils.cosine_scheduler


def get_args():
    p = argparse.ArgumentParser(description="Train chunked latent flow-matching model")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--future_horizon", type=int, default=8)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--num_hidden_layers", type=int, default=1)
    p.add_argument("--time_embed_dim", type=int, default=32)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--flow_weight", type=float, default=1.0)
    p.add_argument("--latent_reg_weight", type=float, default=0.1)
    p.add_argument("--consistency_chunks", type=int, default=1)
    p.add_argument("--rollout_mode", type=str, default="sample", choices=["sample", "posterior"])
    p.add_argument("--consistency_weight", type=float, default=0.0)
    p.add_argument("--sample_weight", type=float, default=0.0)
    p.add_argument("--terminal_weight", type=float, default=0.0)
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


def split_target_chunks(target, future_horizon, num_chunks):
    return [
        target[:, idx * future_horizon : (idx + 1) * future_horizon]
        for idx in range(num_chunks)
    ]


def append_chunk_to_history(history, chunk, window_size):
    return torch.cat([history, chunk], dim=1)[:, -window_size:]


def compute_objective(model, history, target, args, integration_steps):
    target_chunks = split_target_chunks(target, args.future_horizon, args.consistency_chunks)
    first_target = target_chunks[0]
    out = model(history, first_target)

    sample_loss = torch.zeros((), device=history.device, dtype=history.dtype)
    consistency_loss = torch.zeros((), device=history.device, dtype=history.dtype)
    terminal_loss = torch.zeros((), device=history.device, dtype=history.dtype)

    if args.rollout_mode == "sample":
        if args.sample_weight > 0.0 or args.consistency_weight > 0.0 or args.terminal_weight > 0.0:
            cond = model.encode_history(history)
            pred_chunk = model.predict_with_cond(cond, num_integration_steps=integration_steps)
            if args.sample_weight > 0.0:
                sample_loss = F.mse_loss(pred_chunk, first_target)
            if args.terminal_weight > 0.0:
                terminal_loss = F.mse_loss(pred_chunk[:, -1], first_target[:, -1])

            if args.consistency_weight > 0.0 and args.consistency_chunks > 1:
                rollout_terms = []
                next_history = append_chunk_to_history(history, pred_chunk, args.window_size)
                for chunk_idx in range(1, args.consistency_chunks):
                    cond_next = model.encode_history(next_history)
                    pred_next = model.predict_with_cond(cond_next, num_integration_steps=integration_steps)
                    rollout_terms.append(F.mse_loss(pred_next, target_chunks[chunk_idx]))
                    next_history = append_chunk_to_history(next_history, pred_next, args.window_size)
                consistency_loss = torch.stack(rollout_terms).mean()
    else:
        if args.terminal_weight > 0.0:
            terminal_terms = [F.mse_loss(out["recon"][:, -1], first_target[:, -1])]
        else:
            terminal_terms = []

        if args.consistency_weight > 0.0 and args.consistency_chunks > 1:
            rollout_terms = []
            next_history = append_chunk_to_history(history, out["recon"], args.window_size)
            for chunk_idx in range(1, args.consistency_chunks):
                next_out = model(next_history, target_chunks[chunk_idx])
                rollout_terms.append(next_out["total_loss"])
                if args.terminal_weight > 0.0:
                    terminal_terms.append(F.mse_loss(next_out["recon"][:, -1], target_chunks[chunk_idx][:, -1]))
                next_history = append_chunk_to_history(next_history, next_out["recon"], args.window_size)
            consistency_loss = torch.stack(rollout_terms).mean()

        if terminal_terms:
            terminal_loss = torch.stack(terminal_terms).mean()

    total_loss = out["total_loss"]
    total_loss = total_loss + args.sample_weight * sample_loss
    total_loss = total_loss + args.consistency_weight * consistency_loss
    total_loss = total_loss + args.terminal_weight * terminal_loss

    return {
        **out,
        "sample_loss": sample_loss,
        "consistency_loss": consistency_loss,
        "terminal_loss": terminal_loss,
        "objective": total_loss,
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    total_recon, total_flow, total_reg, total_copy, total_n = 0.0, 0.0, 0.0, 0.0, 0
    total_sample, total_consistency, total_terminal, total_objective = 0.0, 0.0, 0.0, 0.0
    for history, target in loader:
        history, target = history.to(device), target.to(device)
        out = compute_objective(model, history, target, args, integration_steps=args.integration_steps_eval)
        first_target = target[:, : args.future_horizon]
        copy_pred = history[:, -1:, :].expand(-1, first_target.shape[1], -1)
        copy_mse = F.mse_loss(copy_pred, first_target).item()

        batch_size = history.shape[0]
        total_recon += out["recon_loss"].item() * batch_size
        total_flow += out["flow_loss"].item() * batch_size
        total_reg += out["latent_reg"].item() * batch_size
        total_copy += copy_mse * batch_size
        total_sample += out["sample_loss"].item() * batch_size
        total_consistency += out["consistency_loss"].item() * batch_size
        total_terminal += out["terminal_loss"].item() * batch_size
        total_objective += out["objective"].item() * batch_size
        total_n += batch_size
    model.train()
    return {
        "recon": total_recon / total_n,
        "flow": total_flow / total_n,
        "latent_reg": total_reg / total_n,
        "copy": total_copy / total_n,
        "sample": total_sample / total_n,
        "consistency": total_consistency / total_n,
        "terminal": total_terminal / total_n,
        "objective": total_objective / total_n,
    }


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
        future_chunks=args.consistency_chunks,
    )
    test_dataset = HandActionChunkDataset(
        args.test_dir,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        noise_std=0.0,
        future_chunks=args.consistency_chunks,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = HandActionChunkFlow(
        action_dim=args.action_dim,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
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
        f"latent={args.latent_dim}, latent_reg={args.latent_reg_weight}, noise_std={args.noise_std}, "
        f"rollout_mode={args.rollout_mode}, consistency_chunks={args.consistency_chunks}, "
        f"consistency_weight={args.consistency_weight}, "
        f"sample_weight={args.sample_weight}, terminal_weight={args.terminal_weight}"
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
        out = compute_objective(model, history, target, args, integration_steps=args.integration_steps_train)

        optimizer.zero_grad()
        out["objective"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()

        if step % args.print_freq == 0:
            print(
                f"[Step {step:>5d}] obj={out['objective'].item():.6f}  base={out['total_loss'].item():.6f}  "
                f"recon={out['recon_loss'].item():.6f}  flow={out['flow_loss'].item():.6f}  "
                f"latent_reg={out['latent_reg'].item():.6f}  sample={out['sample_loss'].item():.6f}  "
                f"cons={out['consistency_loss'].item():.6f}  terminal={out['terminal_loss'].item():.6f}  "
                f"lr={lr_schedule[step]:.2e}"
            )

        if step % args.eval_freq == 0 and step > 0:
            val = evaluate(model, test_loader, device, args)
            best_objective = min(best_objective, val["objective"])
            print(
                f"           val_obj={val['objective']:.6f}  val_recon={val['recon']:.6f}  "
                f"val_flow={val['flow']:.6f}  val_latent_reg={val['latent_reg']:.6f}  "
                f"val_sample={val['sample']:.6f}  val_cons={val['consistency']:.6f}  "
                f"val_terminal={val['terminal']:.6f}  copy_baseline={val['copy']:.6f}  "
                f"best_val_obj={best_objective:.6f}"
            )

        if step > 0 and step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args.output_dir, args)
            print(f"           checkpoint saved -> {args.output_dir}/checkpoint-{step}.pth")

    save_checkpoint(model, optimizer, args.total_steps, args.output_dir, args)
    val = evaluate(model, test_loader, device, args)
    print(
        f"\nFinal: val_obj={val['objective']:.6f}  val_recon={val['recon']:.6f}  "
        f"val_flow={val['flow']:.6f}  val_latent_reg={val['latent_reg']:.6f}  "
        f"val_sample={val['sample']:.6f}  val_cons={val['consistency']:.6f}  "
        f"val_terminal={val['terminal']:.6f}  copy_baseline={val['copy']:.6f}"
    )
    print(f"Done! Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
