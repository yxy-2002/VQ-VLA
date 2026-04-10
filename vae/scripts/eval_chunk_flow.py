"""
Evaluate the conditional latent flow-matching chunk prior.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

_vae_root = os.path.join(os.path.dirname(__file__), "..")

DEFAULT_INIT = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionChunkFlow = _load("model/hand_chunk_flow.py", "hand_chunk_flow").HandActionChunkFlow


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate chunk flow-matching prior")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--num_rollouts", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--integration_steps", type=int, default=24)
    parser.add_argument("--open_seed_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def infer_model_args(checkpoint):
    args = checkpoint.get("args")
    if args is None:
        raise ValueError("Checkpoint is missing saved args; re-train with the current script.")
    return {
        "action_dim": args["action_dim"],
        "window_size": args["window_size"],
        "future_horizon": args["future_horizon"],
        "hidden_dim": args["hidden_dim"],
        "latent_dim": args["latent_dim"],
        "num_hidden_layers": args["num_hidden_layers"],
        "time_embed_dim": args["time_embed_dim"],
        "recon_weight": args["recon_weight"],
        "flow_weight": args["flow_weight"],
        "latent_reg_weight": args["latent_reg_weight"],
    }


def finger_mean(actions):
    return actions[..., 2:6].mean(axis=-1)


def first_crossing(series, threshold):
    idx = np.where(series > threshold)[0]
    return int(idx[0]) if len(idx) > 0 else len(series)


@torch.no_grad()
def rollout_free_batch(model, seed, num_rollouts, max_steps, integration_steps):
    device = next(model.parameters()).device
    actions = torch.zeros(num_rollouts, max_steps, model.action_dim, device=device)
    actions[:, 0, :] = seed[0].to(device)
    chunk_latents = []

    t = 1
    while t < max_steps:
        window = actions[:, max(0, t - model.window_size) : t, :]
        if window.shape[1] < model.window_size:
            pad = actions[:, 0:1, :].expand(-1, model.window_size - window.shape[1], -1)
            window = torch.cat([pad, window], dim=1)
        pred, z = model.predict(window, num_integration_steps=integration_steps, return_latent=True)
        take = min(model.future_horizon, max_steps - t)
        actions[:, t : t + take, :] = pred[:, :take, :]
        chunk_latents.append(z.cpu().numpy())
        t += take

    return {
        "actions": actions.cpu().numpy(),
        "chunk_latents": np.stack(chunk_latents, axis=1) if chunk_latents else np.zeros((num_rollouts, 0, model.latent_dim)),
    }


def summarize_rollouts(actions, threshold):
    fm = finger_mean(actions)
    cross = fm > threshold
    ever = cross.any(axis=1)
    first = np.where(ever, cross.argmax(axis=1), fm.shape[1])
    final = fm[:, -1]
    return {
        "close_fraction_by_max_step": float(ever.mean()),
        "first_close_mean": float(first[ever].mean()) if np.any(ever) else None,
        "first_close_median": float(np.median(first[ever])) if np.any(ever) else None,
        "first_close_q90": float(np.quantile(first[ever], 0.9)) if np.any(ever) else None,
        "final_finger_mean_all": float(final.mean()),
        "final_finger_mean_closed": float(final[ever].mean()) if np.any(ever) else None,
        "still_open_count": int((~ever).sum()),
    }


def dataset_reference_stats(test_dir, threshold):
    onsets, finals, lengths = [], [], []
    for path in sorted(Path(test_dir).glob("trajectory_*_demo_expert.pt")):
        actions = torch.load(path, map_location="cpu", weights_only=False)["actions"][:, 0, 6:12].float().numpy()
        fm = finger_mean(actions)
        idx = np.where(fm > threshold)[0]
        onsets.append(int(idx[0]) if len(idx) > 0 else len(actions))
        finals.append(float(fm[-1]))
        lengths.append(len(actions))
    onsets = np.array(onsets)
    finals = np.array(finals)
    lengths = np.array(lengths)
    closed = finals > threshold
    return {
        "num_traj": int(len(onsets)),
        "onset_mean": float(onsets.mean()),
        "onset_median": float(np.median(onsets)),
        "onset_q90": float(np.quantile(onsets, 0.9)),
        "final_finger_mean": float(finals.mean()),
        "final_finger_mean_closed": float(finals[closed].mean()) if np.any(closed) else None,
        "never_close": int(np.sum(onsets == lengths)),
    }


def open_window_reference(test_dir, threshold, horizon):
    close_flags, first_steps, end_vals = [], [], []
    for split in ["train", "test"]:
        for path in sorted(Path(test_dir).parent.joinpath(split).glob("trajectory_*_demo_expert.pt")):
            actions = torch.load(path, map_location="cpu", weights_only=False)["actions"][:, 0, 6:12].float().numpy()
            total_steps = len(actions)
            for t in range(total_steps):
                start = t - 7
                if start < 0:
                    pad = np.repeat(actions[0:1], -start, axis=0)
                    window = np.concatenate([pad, actions[: t + 1]], axis=0)
                else:
                    window = actions[start : t + 1]
                if np.abs(window[:, 2:6]).max() >= 0.02:
                    continue
                future = []
                for off in range(1, horizon + 1):
                    idx = min(t + off, total_steps - 1)
                    future.append(actions[idx])
                future = np.stack(future, axis=0)
                fm = finger_mean(future)
                idx = np.where(fm > threshold)[0]
                close_flags.append(len(idx) > 0)
                first_steps.append((idx[0] + 1) if len(idx) > 0 else horizon + 1)
                end_vals.append(float(fm[-1]))
    close_flags = np.array(close_flags)
    first_steps = np.array(first_steps)
    end_vals = np.array(end_vals)
    return {
        "num_open_windows": int(len(close_flags)),
        "close_fraction_in_chunk": float(close_flags.mean()),
        "first_close_median_if_close": float(np.median(first_steps[close_flags])) if np.any(close_flags) else None,
        "first_close_q90_if_close": float(np.quantile(first_steps[close_flags], 0.9)) if np.any(close_flags) else None,
        "end_finger_mean": float(end_vals.mean()),
        "end_finger_std": float(end_vals.std()),
    }


@torch.no_grad()
def open_seed_chunk_stats(model, num_samples, threshold, integration_steps):
    seed = torch.tensor(DEFAULT_INIT, dtype=torch.float32).view(1, 1, -1)
    window = seed.expand(num_samples, model.window_size, -1).to(next(model.parameters()).device)
    pred = model.predict(window, num_integration_steps=integration_steps).cpu().numpy()
    fm = finger_mean(pred)
    close_flags = np.array([np.any(seq > threshold) for seq in fm])
    first_steps = np.array([first_crossing(seq, threshold) + 1 if np.any(seq > threshold) else len(seq) + 1 for seq in fm])
    return {
        "close_fraction_in_chunk": float(close_flags.mean()),
        "first_close_median_if_close": float(np.median(first_steps[close_flags])) if np.any(close_flags) else None,
        "first_close_q90_if_close": float(np.quantile(first_steps[close_flags], 0.9)) if np.any(close_flags) else None,
        "end_finger_mean": float(fm[:, -1].mean()),
        "end_finger_std": float(fm[:, -1].std()),
    }


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_kwargs = infer_model_args(ckpt)
    model = HandActionChunkFlow(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    seed = torch.tensor(DEFAULT_INIT, dtype=torch.float32).unsqueeze(0)
    rollout = rollout_free_batch(
        model,
        seed=seed,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        integration_steps=args.integration_steps,
    )
    rollout_stats = summarize_rollouts(rollout["actions"], threshold=args.threshold)
    dataset_ref = dataset_reference_stats(args.test_dir, threshold=args.threshold)
    open_ref = open_window_reference(args.test_dir, threshold=args.threshold, horizon=model.future_horizon)
    open_seed = open_seed_chunk_stats(model, args.open_seed_samples, threshold=args.threshold, integration_steps=args.integration_steps)

    payload = {
        "checkpoint": args.ckpt,
        "model_kwargs": model_kwargs,
        "rollout_stats": rollout_stats,
        "dataset_reference": dataset_ref,
        "open_window_reference": open_ref,
        "open_seed_chunk_stats": open_seed,
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
