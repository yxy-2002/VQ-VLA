"""
Evaluate a chunked hand-action VAE with free-run rollouts.
"""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

_vae_root = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(_vae_root, "..")

DEFAULT_INIT = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionChunkVAE = _load("model/hand_chunk_vae.py", "hand_chunk_vae").HandActionChunkVAE


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate chunked hand-action VAE")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--num_rollouts", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def infer_model_args(state_dict):
    kwargs = {}
    kwargs["latent_dim"] = state_dict["fc_mu.weight"].shape[0]
    kwargs["hidden_dim"] = state_dict["fc_mu.weight"].shape[1]

    decoder_indices = sorted(
        int(m.group(1)) for k in state_dict if (m := re.match(r"decoder\.(\d+)\.weight$", k))
    )
    last_dec_idx = decoder_indices[-1]
    decoder_out = state_dict[f"decoder.{last_dec_idx}.weight"].shape[0]

    if "encoder.net.0.conv.weight" in state_dict:
        kwargs["encoder_type"] = "causal_conv"
        raise ValueError("causal_conv inference not implemented for chunk eval")
    else:
        kwargs["encoder_type"] = "mlp"
        encoder_in = state_dict["encoder.0.weight"].shape[1]

    fc_out = state_dict["fc_mu.weight"].shape[0]
    kwargs["num_hidden_layers"] = max(0, len(decoder_indices) - 2)

    # action_dim is always 6 in this project, but infer it from the encoder input if possible.
    for action_dim in [6, 12, 16, 20]:
        if encoder_in % action_dim == 0 and decoder_out % action_dim == 0:
            window_size = encoder_in // action_dim
            future_horizon = decoder_out // action_dim
            if window_size > 0 and future_horizon > 0:
                kwargs["action_dim"] = action_dim
                kwargs["window_size"] = window_size
                kwargs["future_horizon"] = future_horizon
                return kwargs

    raise ValueError("Failed to infer action_dim/window_size/future_horizon from checkpoint")


def finger_mean(actions):
    return actions[..., 2:6].mean(axis=-1)


@torch.no_grad()
def rollout_free_batch(model, seed, num_rollouts, max_steps, window_size, future_horizon, deterministic=False):
    device = next(model.parameters()).device
    if seed.shape[0] != 1:
        raise ValueError("This evaluator expects a single-frame default seed")

    actions = torch.zeros(num_rollouts, max_steps, seed.shape[-1], device=device)
    actions[:, 0, :] = seed[0].to(device)
    t = 1

    while t < max_steps:
        window = actions[:, max(0, t - window_size) : t, :]
        if window.shape[1] == 0:
            raise RuntimeError("Unexpected empty rollout window")
        if window.shape[1] < window_size:
            pad = actions[:, 0:1, :].expand(-1, window_size - window.shape[1], -1)
            window = torch.cat([pad, window], dim=1)

        pred_chunk = model.predict(window, deterministic=deterministic)
        take = min(future_horizon, max_steps - t)
        actions[:, t : t + take, :] = pred_chunk[:, :take, :]
        t += take

    return actions.cpu().numpy()


def dataset_reference_stats(test_dir, threshold):
    onsets = []
    finals = []
    lengths = []
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
    closed_mask = finals > threshold
    return {
        "num_traj": int(len(onsets)),
        "onset_mean": float(onsets.mean()),
        "onset_median": float(np.median(onsets)),
        "onset_q90": float(np.quantile(onsets, 0.9)),
        "final_finger_mean": float(finals.mean()),
        "final_finger_mean_closed": float(finals[closed_mask].mean()) if np.any(closed_mask) else None,
        "never_close": int(np.sum(onsets == lengths)),
    }


def summarize_rollouts(runs, threshold):
    fm = finger_mean(runs)
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


def main():
    args = get_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_kwargs = infer_model_args(ckpt["model"])
    model = HandActionChunkVAE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    seed = torch.tensor(DEFAULT_INIT, dtype=torch.float32).unsqueeze(0)
    runs = rollout_free_batch(
        model,
        seed=seed,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        window_size=model_kwargs["window_size"],
        future_horizon=model_kwargs["future_horizon"],
        deterministic=False,
    )

    rollout_stats = summarize_rollouts(runs, threshold=args.threshold)
    ref_stats = dataset_reference_stats(args.test_dir, threshold=args.threshold)

    payload = {
        "checkpoint": args.ckpt,
        "model_kwargs": model_kwargs,
        "rollout_stats": rollout_stats,
        "dataset_reference": ref_stats,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
