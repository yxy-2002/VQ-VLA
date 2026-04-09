"""
Evaluate Hand Action VAE with two modes:

1. GT-comparison mode (default):
   Given real trajectories, compare model predictions (TF + AR) against ground truth.
   Supports multiple stochastic sampling runs to visualize prediction variance.

2. Free-run mode (--free_run):
   Given a seed (initial state or first N frames), roll out for --max_steps steps
   with no ground truth. Saves the generated trajectory.

Usage:
    # GT comparison on test set (default: stochastic sampling)
    python vae/scripts/eval.py --traj_id 105 --save_plot

    # GT comparison with N stochastic samples to see variance
    python vae/scripts/eval.py --traj_id 105 --num_samples 10 --save_plot

    # Deterministic (use μ, no sampling)
    python vae/scripts/eval.py --traj_id 105 --deterministic --save_plot

    # Free-run from default initial state for 100 steps
    python vae/scripts/eval.py --free_run --max_steps 100 --num_samples 5 --save_plot

    # Free-run from a specific trajectory's seed
    python vae/scripts/eval.py --free_run --traj_id 105 --max_steps 80 --save_plot
"""

import argparse
import importlib.util
import os
import glob
import re

import numpy as np
import torch

_vae_root = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(_vae_root, "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionVAE = _load("model/hand_vae.py", "hand_vae").HandActionVAE

JOINT_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
# Default initial hand state: thumb rotated to 0.4, others open
DEFAULT_INIT = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]


def infer_model_args(state_dict, default_window_size=8):
    """Infer HandActionVAE constructor kwargs from a checkpoint's state_dict.

    Reads weight tensor shapes (and key names) to recover action_dim, window_size,
    hidden_dim, latent_dim, encoder_type, and num_hidden_layers — so eval doesn't
    need to hard-code anything.
    """
    kwargs = {}

    # latent_dim / hidden_dim from fc_mu: Linear(hidden_dim, latent_dim)
    kwargs["latent_dim"] = state_dict["fc_mu.weight"].shape[0]
    kwargs["hidden_dim"] = state_dict["fc_mu.weight"].shape[1]

    # action_dim is the output dim of the LAST decoder Linear (not always index 4
    # — depends on num_hidden_layers). Find max even index in decoder.{i}.weight.
    decoder_indices = sorted(
        int(m.group(1)) for k in state_dict
        if (m := re.match(r"decoder\.(\d+)\.weight$", k))
    )
    last_dec_idx = decoder_indices[-1]
    kwargs["action_dim"] = state_dict[f"decoder.{last_dec_idx}.weight"].shape[0]

    # num_hidden_layers: decoder has (1 input proj) + N hidden→hidden + (1 output proj) Linears.
    # So num_hidden_layers = num_decoder_linears - 2.
    kwargs["num_hidden_layers"] = max(0, len(decoder_indices) - 2)

    # Auxiliary input-reconstruction head: only present if model was trained with recon_aux_weight > 0.
    # If keys exist in state_dict, we must construct the head so load_state_dict succeeds.
    if any(k.startswith("aux_recon_head.") for k in state_dict):
        kwargs["recon_aux_weight"] = 1.0  # any positive value triggers construction

    # Detect encoder type from key naming
    if "encoder.net.0.conv.weight" in state_dict:
        kwargs["encoder_type"] = "causal_conv"
        # CausalConv is sequence-length agnostic; window_size only matters for the dataset
        kwargs["window_size"] = default_window_size
    else:
        kwargs["encoder_type"] = "mlp"
        # MLP encoder.0: Linear(action_dim * window_size, hidden_dim)
        in_features = state_dict["encoder.0.weight"].shape[1]
        if in_features % kwargs["action_dim"] != 0:
            raise ValueError(
                f"Cannot infer window_size: encoder input dim {in_features} "
                f"not divisible by action_dim {kwargs['action_dim']}"
            )
        kwargs["window_size"] = in_features // kwargs["action_dim"]

    return kwargs


def load_trajectory(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["actions"][:, 0, 6:12].float()


# ─── Rollout functions ─────────────────────────────────────────────────────────

def rollout_teacher_forcing(model, gt_actions, window_size, deterministic):
    """Each step uses GT as input. Measures single-step prediction quality."""
    device = next(model.parameters()).device
    T = gt_actions.shape[0]
    pred = gt_actions.clone()
    for t in range(window_size, T):
        window = gt_actions[t - window_size:t].unsqueeze(0).to(device)
        pred[t] = model.predict(window, deterministic=deterministic).squeeze(0).cpu()
    return pred


def rollout_autoregressive(model, seed, total_steps, window_size, deterministic):
    """
    Autoregressive rollout from seed frames.
    seed: (S, 6) where S >= 1. If S < window_size, pad with first frame.
    total_steps: total output length (including seed).
    """
    device = next(model.parameters()).device
    S = seed.shape[0]

    # Pad seed if shorter than window_size
    if S < window_size:
        pad = seed[0:1].expand(window_size - S, -1)
        buffer = torch.cat([pad, seed], dim=0)
    else:
        buffer = seed.clone()

    # Roll out
    all_actions = list(buffer)
    for _ in range(total_steps - len(all_actions)):
        window = torch.stack(all_actions[-window_size:]).unsqueeze(0).to(device)
        pred = model.predict(window, deterministic=deterministic).squeeze(0).cpu()
        all_actions.append(pred)

    return torch.stack(all_actions[:total_steps])


# ─── Mode 1: GT comparison ────────────────────────────────────────────────────

def eval_gt_comparison(model, traj_path, window_size, deterministic, num_samples, verbose):
    """Compare model predictions against ground truth trajectory."""
    gt = load_trajectory(traj_path)
    T = gt.shape[0]
    traj_id = os.path.basename(traj_path).split("trajectory_")[1].split("_")[0]

    # Teacher forcing (always deterministic for clean comparison)
    pred_tf = rollout_teacher_forcing(model, gt, window_size, deterministic=True)
    mse_tf = ((pred_tf - gt) ** 2).mean(dim=1).numpy()

    # Autoregressive: run num_samples times
    ar_runs = []
    for _ in range(num_samples):
        pred_ar = rollout_autoregressive(model, gt[:window_size], T, window_size, deterministic)
        ar_runs.append(pred_ar.numpy())
    ar_runs = np.stack(ar_runs)  # (num_samples, T, 6)
    ar_mean = ar_runs.mean(axis=0)  # (T, 6)
    ar_std = ar_runs.std(axis=0)    # (T, 6)
    mse_ar_mean = ((ar_mean - gt.numpy()) ** 2).mean(axis=1)

    copy_mse = ((gt[window_size:] - gt[window_size - 1:-1]) ** 2).mean().item()

    if verbose:
        tf_region = mse_tf[window_size:].mean()
        ar_region = mse_ar_mean[window_size:].mean()
        print(f"\nTrajectory {traj_id}: {T} steps (seed={window_size}, predict={T - window_size})")
        print(f"  Teacher Forcing MSE:         {tf_region:.6f}")
        print(f"  Autoregressive MSE (mean):   {ar_region:.6f}  ({num_samples} samples)")
        print(f"  Copy Baseline MSE:           {copy_mse:.6f}")
        if num_samples > 1:
            print(f"  AR prediction std (mean):    {ar_std[window_size:].mean():.6f}")

    return {
        "traj_id": traj_id, "T": T, "gt": gt.numpy(),
        "pred_tf": pred_tf.numpy(), "mse_tf": mse_tf,
        "ar_runs": ar_runs, "ar_mean": ar_mean, "ar_std": ar_std,
        "mse_ar_mean": mse_ar_mean, "copy_mse": copy_mse,
        "num_samples": num_samples,
    }


# ─── Mode 2: Free-run ─────────────────────────────────────────────────────────

def eval_free_run(model, seed, max_steps, window_size, deterministic, num_samples, label="free"):
    """Generate trajectories from seed without ground truth."""
    runs = []
    for _ in range(num_samples):
        traj = rollout_autoregressive(model, seed, max_steps, window_size, deterministic)
        runs.append(traj.numpy())
    runs = np.stack(runs)  # (num_samples, max_steps, 6)

    print(f"\nFree-run: {max_steps} steps, {num_samples} samples")
    print(f"  Seed: {seed[0].numpy().round(3)}")
    print(f"  Final state (mean): {runs[:, -1, :].mean(axis=0).round(3)}")
    print(f"  Final state (std):  {runs[:, -1, :].std(axis=0).round(3)}")

    return {
        "label": label, "max_steps": max_steps,
        "seed": seed.numpy(), "runs": runs,
        "num_samples": num_samples,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_gt_comparison(result, output_dir, window_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = result["gt"]
    T = result["T"]
    traj_id = result["traj_id"]
    ar_runs = result["ar_runs"]
    ar_mean = result["ar_mean"]
    ar_std = result["ar_std"]
    pred_tf = result["pred_tf"]
    num_samples = result["num_samples"]
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Trajectory {traj_id}: Ground Truth vs Predictions  "
                 f"({num_samples} stochastic samples)", fontsize=14)

    for i, (ax, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        ax.plot(range(T), gt[:, i], "k-", label="Ground Truth", linewidth=2)
        ax.plot(range(T), pred_tf[:, i], "g--", label="Teacher Forcing", linewidth=1.5)

        # Plot individual AR samples (light)
        for s in range(min(num_samples, 20)):
            ax.plot(range(T), ar_runs[s, :, i], color="red", alpha=0.15, linewidth=0.8)

        # AR mean + std band
        ax.plot(range(T), ar_mean[:, i], "r-", label=f"Autoregressive (mean, n={num_samples})", linewidth=1.5)
        if num_samples > 1:
            ax.fill_between(range(T),
                            ar_mean[:, i] - ar_std[:, i],
                            ar_mean[:, i] + ar_std[:, i],
                            color="red", alpha=0.15)

        ax.axvline(x=window_size - 0.5, color="gray", linestyle=":", alpha=0.4, label="Seed boundary")
        ax.set_ylabel(name)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")

    path = os.path.join(output_dir, f"traj_{traj_id}_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_free_run(result, output_dir, window_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = result["runs"]
    num_samples, max_steps, _ = runs.shape
    label = result["label"]
    seed_len = result["seed"].shape[0]
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Free-run: {max_steps} steps, {num_samples} stochastic samples  "
                 f"(seed={seed_len} frames)", fontsize=14)

    mean = runs.mean(axis=0)
    std = runs.std(axis=0)

    for i, (ax, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        for s in range(min(num_samples, 30)):
            ax.plot(range(max_steps), runs[s, :, i], color="steelblue", alpha=0.2, linewidth=0.8)
        ax.plot(range(max_steps), mean[:, i], "b-", label=f"Mean (n={num_samples})", linewidth=2)
        if num_samples > 1:
            ax.fill_between(range(max_steps),
                            mean[:, i] - std[:, i],
                            mean[:, i] + std[:, i],
                            color="steelblue", alpha=0.2)
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hand Action VAE")
    parser.add_argument("--ckpt", type=str, default=os.path.join(_proj_root, "outputs/hand_vae/checkpoint.pth"))
    parser.add_argument("--test_dir", type=str, default=os.path.join(_proj_root, "data/20260327-11:10:43/demos/success/test"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(_proj_root, "visualizations/vae_eval"))
    parser.add_argument("--window_size", type=int, default=8)

    # Mode selection
    parser.add_argument("--free_run", action="store_true", help="Free-run mode (no GT comparison)")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps for free-run mode")

    # Trajectory selection (for GT mode, or seed source for free-run)
    parser.add_argument("--traj_id", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--init_state", type=float, nargs=6, default=None,
                        help="Custom initial state for free-run (6 floats)")

    # Sampling
    parser.add_argument("--deterministic", action="store_true", help="Use μ directly (no sampling)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of stochastic rollouts")

    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()

    # Load model — infer architecture from the checkpoint's state_dict so we
    # don't have to know hidden_dim/latent_dim/encoder_type ahead of time.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_kwargs = infer_model_args(ckpt["model"], default_window_size=args.window_size)
    model = HandActionVAE(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Sync window_size from the checkpoint (MLP encoder pins it; conv encoder keeps the CLI value)
    if model_kwargs["window_size"] != args.window_size:
        print(f"  Note: overriding --window_size {args.window_size} → {model_kwargs['window_size']} (from ckpt)")
        args.window_size = model_kwargs["window_size"]

    deterministic = args.deterministic
    print(f"Model: {args.ckpt}")
    print(f"  Inferred arch: action_dim={model_kwargs['action_dim']}  "
          f"window_size={model_kwargs['window_size']}  "
          f"hidden_dim={model_kwargs['hidden_dim']}  "
          f"latent_dim={model_kwargs['latent_dim']}  "
          f"encoder={model_kwargs['encoder_type']}  "
          f"depth={model_kwargs['num_hidden_layers']}  "
          f"aux_head={'yes' if model_kwargs.get('recon_aux_weight', 0) > 0 else 'no'}")
    print(f"Sampling: {'deterministic (μ)' if deterministic else f'stochastic ({args.num_samples} samples)'}")

    if args.free_run:
        # ── Free-run mode ──
        if args.init_state:
            seed = torch.tensor(args.init_state).float().unsqueeze(0)
            label = "custom"
        elif args.traj_id:
            all_files = sorted(glob.glob(os.path.join(args.test_dir, "trajectory_*_demo_expert.pt")))
            matches = [f for f in all_files if f"trajectory_{args.traj_id[0]}_" in f]
            if matches:
                gt = load_trajectory(matches[0])
                seed = gt[:args.window_size]
                label = f"seed_traj{args.traj_id[0]}"
            else:
                print(f"Trajectory {args.traj_id[0]} not found, using default init")
                seed = torch.tensor(DEFAULT_INIT).float().unsqueeze(0)
                label = "default"
        else:
            seed = torch.tensor(DEFAULT_INIT).float().unsqueeze(0)
            label = "default"

        result = eval_free_run(model, seed, args.max_steps, args.window_size,
                               deterministic, args.num_samples, label)

        # Save trajectory data
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"free_run_{label}.npz")
        np.savez(save_path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        print(f"  Trajectory data saved: {save_path}")

        if args.save_plot:
            plot_free_run(result, args.output_dir, args.window_size)

    else:
        # ── GT comparison mode ──
        all_files = sorted(glob.glob(os.path.join(args.test_dir, "trajectory_*_demo_expert.pt")))
        if args.traj_id:
            files = []
            for tid in args.traj_id:
                matches = [f for f in all_files if f"trajectory_{tid}_" in f]
                files.extend(matches)
        elif args.all:
            files = all_files
        else:
            files = all_files[:3]

        if not files:
            print("No trajectory files found!")
            return

        results = []
        for f in files:
            r = eval_gt_comparison(model, f, args.window_size, deterministic,
                                   args.num_samples, verbose=True)
            results.append(r)
            if args.save_plot:
                plot_gt_comparison(r, args.output_dir, args.window_size)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"Summary ({len(results)} trajectories, {args.num_samples} samples each)")
        print(f"{'=' * 60}")
        for r in results:
            W = args.window_size
            tf_mse = r["mse_tf"][W:].mean()
            ar_mse = r["mse_ar_mean"][W:].mean()
            print(f"  traj {r['traj_id']:>4}: TF={tf_mse:.6f}  AR={ar_mse:.6f}  "
                  f"copy={r['copy_mse']:.6f}  "
                  f"AR_std={r['ar_std'][W:].mean():.4f}")


if __name__ == "__main__":
    main()
