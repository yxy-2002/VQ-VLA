"""
Evaluate a chunked hand-action VAE with free-run rollouts.

Usage:
    # Free-run from default initial state, 200 stochastic samples
    python vae/scripts/eval_chunk_vae.py \
        --ckpt outputs/chunk_vae/checkpoint.pth \
        --test_dir data/20260327-11:10:43/demos/success/test \
        --output_dir visualizations/chunk_vae_eval \
        --num_samples 200 --max_steps 100 --save_plot

    # Free-run with full-chunk stride (use all 8 predicted frames before re-predicting)
    python vae/scripts/eval_chunk_vae.py \
        --ckpt outputs/chunk_vae/checkpoint.pth \
        --test_dir data/20260327-11:10:43/demos/success/test \
        --rollout_stride 8 --save_plot
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

_vae_root = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(_vae_root, "..")

JOINT_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
DEFAULT_INIT = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionChunkVAE = _load("model/hand_chunk_vae.py", "hand_chunk_vae").HandActionChunkVAE


# ── Model loading ────────────────────────────────────────────────────────────


def load_model_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    kwargs = {
        "action_dim": args.get("action_dim", 6),
        "window_size": args.get("window_size", 8),
        "future_horizon": args.get("future_horizon", 8),
        "hidden_dim": args.get("hidden_dim", 256),
        "latent_dim": args.get("latent_dim", 2),
        "beta": args.get("beta", 0.001),
        "encoder_type": args.get("encoder_type", "mlp"),
        "num_hidden_layers": args.get("num_hidden_layers", 1),
        "free_bits": args.get("free_bits", 0.0),
    }
    model = HandActionChunkVAE(**kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, kwargs


# ── Rollout ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def rollout_free_batch(model, seed, num_rollouts, max_steps, rollout_stride, deterministic=False):
    """
    Batched chunk-based autoregressive rollout.

    At each step: predict future_horizon frames, use the first `rollout_stride`
    frames, then slide the window forward by stride.
    """
    device = next(model.parameters()).device
    window_size = model.window_size
    future_horizon = model.future_horizon
    action_dim = seed.shape[-1]

    actions = torch.zeros(num_rollouts, max_steps, action_dim, device=device)
    actions[:, 0, :] = seed[0].to(device)
    t = 1

    while t < max_steps:
        # Build window from existing actions
        window = actions[:, max(0, t - window_size) : t, :]
        if window.shape[1] < window_size:
            pad = actions[:, 0:1, :].expand(-1, window_size - window.shape[1], -1)
            window = torch.cat([pad, window], dim=1)

        pred_chunk = model.predict(window, deterministic=deterministic)
        take = min(max(1, rollout_stride), future_horizon, max_steps - t)
        actions[:, t : t + take, :] = pred_chunk[:, :take, :]
        t += take

    return actions.cpu().numpy()


# ── Statistics ───────────────────────────────────────────────────────────────


def finger_mean(actions):
    """Average of index/middle/ring/pinky (dims 2:6)."""
    return actions[..., 2:6].mean(axis=-1)


def summarize_rollouts(runs, threshold):
    fm = finger_mean(runs)
    cross = fm > threshold
    ever = cross.any(axis=1)
    first = np.where(ever, cross.argmax(axis=1), fm.shape[1])
    final = fm[:, -1]
    return {
        "close_fraction": float(ever.mean()),
        "first_close_mean": float(first[ever].mean()) if np.any(ever) else None,
        "first_close_median": float(np.median(first[ever])) if np.any(ever) else None,
        "final_finger_mean_all": float(final.mean()),
        "final_finger_mean_closed": float(final[ever].mean()) if np.any(ever) else None,
        "still_open_count": int((~ever).sum()),
    }


def dataset_reference_stats(test_dir, threshold):
    onsets, finals, lengths = [], [], []
    for path in sorted(Path(test_dir).glob("trajectory_*_demo_expert.pt")):
        acts = torch.load(path, map_location="cpu", weights_only=False)["actions"][:, 0, 6:12].float().numpy()
        fm = finger_mean(acts)
        idx = np.where(fm > threshold)[0]
        onsets.append(int(idx[0]) if len(idx) > 0 else len(acts))
        finals.append(float(fm[-1]))
        lengths.append(len(acts))

    onsets = np.array(onsets)
    finals = np.array(finals)
    closed = finals > threshold
    return {
        "num_traj": len(onsets),
        "onset_mean": float(onsets.mean()),
        "onset_median": float(np.median(onsets)),
        "final_finger_mean": float(finals.mean()),
        "never_close": int(np.sum(onsets == np.array(lengths))),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_free_run(runs, output_dir, label, seed_len):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_samples, max_steps, _ = runs.shape
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(
        f"Free-run: {max_steps} steps, {num_samples} stochastic samples  "
        f"(seed={seed_len} frames)",
        fontsize=14,
    )

    mean = runs.mean(axis=0)
    std = runs.std(axis=0)

    for i, (ax, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        for s in range(min(num_samples, 30)):
            ax.plot(range(max_steps), runs[s, :, i], color="steelblue", alpha=0.2, linewidth=0.8)
        ax.plot(range(max_steps), mean[:, i], "b-", label=f"Mean (n={num_samples})", linewidth=2)
        if num_samples > 1:
            ax.fill_between(
                range(max_steps),
                mean[:, i] - std[:, i],
                mean[:, i] + std[:, i],
                color="steelblue", alpha=0.2,
            )
        ax.axvline(x=seed_len - 0.5, color="gray", linestyle=":", alpha=0.4, label="Seed boundary")
        ax.set_ylabel(name)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")

    path = os.path.join(output_dir, f"free_run_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def get_args():
    p = argparse.ArgumentParser(description="Evaluate chunked hand-action VAE")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=os.path.join(_proj_root, "visualizations/chunk_vae_eval"))
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--rollout_stride", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--init_state", type=float, nargs=6, default=None)
    p.add_argument("--save_plot", action="store_true")
    p.add_argument("--label", type=str, default="default")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    model, model_kwargs = load_model_from_checkpoint(args.ckpt, device)
    print(f"Model: {args.ckpt}")
    print(f"  arch: window={model_kwargs['window_size']}  horizon={model_kwargs['future_horizon']}  "
          f"hidden={model_kwargs['hidden_dim']}  latent={model_kwargs['latent_dim']}  "
          f"encoder={model_kwargs['encoder_type']}  depth={model_kwargs['num_hidden_layers']}")
    print(f"  rollout_stride={args.rollout_stride}  "
          f"{'deterministic' if args.deterministic else f'stochastic ({args.num_samples} samples)'}")

    # Seed
    if args.init_state:
        seed = torch.tensor(args.init_state, dtype=torch.float32).unsqueeze(0)
    else:
        seed = torch.tensor(DEFAULT_INIT, dtype=torch.float32).unsqueeze(0)

    # Rollout
    runs = rollout_free_batch(
        model,
        seed=seed,
        num_rollouts=args.num_samples,
        max_steps=args.max_steps,
        rollout_stride=args.rollout_stride,
        deterministic=args.deterministic,
    )

    # Print summary
    print(f"\nFree-run: {args.max_steps} steps, {args.num_samples} samples")
    print(f"  Seed: {seed[0].numpy().round(3)}")
    print(f"  Final state (mean): {runs[:, -1, :].mean(axis=0).round(3)}")
    print(f"  Final state (std):  {runs[:, -1, :].std(axis=0).round(3)}")

    rollout_stats = summarize_rollouts(runs, threshold=args.threshold)
    ref_stats = dataset_reference_stats(args.test_dir, threshold=args.threshold)

    print(f"\n  Rollout stats: {json.dumps(rollout_stats, indent=4)}")
    print(f"  Dataset ref:   {json.dumps(ref_stats, indent=4)}")

    # Save NPZ
    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, f"free_run_{args.label}.npz")
    np.savez(npz_path, runs=runs, seed=seed.numpy())
    print(f"\n  Data saved: {npz_path}")

    # Save plot
    if args.save_plot:
        plot_free_run(runs, args.output_dir, args.label, seed_len=seed.shape[0])

    # Save JSON
    if args.output_json:
        payload = {
            "checkpoint": args.ckpt,
            "model_kwargs": model_kwargs,
            "rollout_stride": args.rollout_stride,
            "rollout_stats": rollout_stats,
            "dataset_reference": ref_stats,
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  JSON saved: {args.output_json}")


if __name__ == "__main__":
    main()
