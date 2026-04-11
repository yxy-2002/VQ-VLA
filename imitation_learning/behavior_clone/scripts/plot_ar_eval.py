"""
AR evaluation with MC Dropout: run the BC policy in autoregressive mode
with dropout active (train mode) to produce stochastic rollouts.

For each test trajectory, generates a VAE-eval-style figure showing:
  - GT next-pose (black solid line)
  - Individual AR samples (light colored, low alpha)
  - AR mean (bold colored)
  - AR std fill band
  - No-correction baseline (dashed gray)

Also generates:
  - Per-trajectory MSE plot (arm + hand)
  - Summary bar chart across all trajectories
  - Latent diagnostic stats (delta_z magnitude, etc.)

Usage:
    python imitation_learning/behavior_clone/scripts/plot_ar_eval.py \
        --ckpt outputs/bc_v4_sweep/drop03/checkpoint.pth \
        --output_dir visualizations/bc_ar_eval/drop03 \
        --num_samples 5
"""

import argparse
import glob
import importlib.util
import json
import os
from typing import Optional

import numpy as np
import torch

# ─── Path & module loading ────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BC_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_policy_mod = _load(os.path.join(_BC_ROOT, "model/bc_policy.py"), "bc_policy")
_dataset_mod = _load(os.path.join(_BC_ROOT, "model/bc_dataset.py"), "bc_dataset")
BCPolicy = _policy_mod.BCPolicy
build_and_freeze_vae = _policy_mod.build_and_freeze_vae
apply_state_mask = _dataset_mod.apply_state_mask

ARM_NAMES = [f"arm_{i}" for i in range(6)]
HAND_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
JOINT_NAMES = ARM_NAMES + HAND_NAMES


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_trajectory(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    actions = data["actions"][:, 0, :].float()
    main = data["curr_obs"]["main_images"][:, 0].permute(0, 3, 1, 2).contiguous()
    extra = data["curr_obs"]["extra_view_images"][:, 0, 0].permute(0, 3, 1, 2).contiguous()
    traj_id = os.path.basename(path).split("trajectory_")[1].split("_")[0]
    return {
        "traj_id": traj_id,
        "actions": actions,
        "imgs_main": main,
        "imgs_extra": extra,
        "T": int(actions.shape[0]),
    }


def build_past_window(hand_seq: torch.Tensor, t: int, window_size: int = 8):
    start = t - window_size + 1
    if start < 0:
        pad_len = -start
        return torch.cat([hand_seq[0:1].expand(pad_len, -1), hand_seq[0:t + 1]], dim=0)
    return hand_seq[start:t + 1]


# ─── MC Dropout AR rollout ────────────────────────────────────────────────────


def rollout_ar_mc_dropout(
    policy: BCPolicy,
    traj: dict,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    num_samples: int,
    device: torch.device,
    window_size: int = 8,
    state_mask: str = "all",
) -> dict:
    """Run AR rollout with MC Dropout (policy in train mode for stochastic output).

    Each of the num_samples rollouts gets its own dropout mask sequence,
    producing diverse trajectories. The VAE stays frozen in eval mode
    (BCPolicy.train() override ensures this).

    Returns dict with:
        ar_runs:    (num_samples, T, 12) predicted actions
        no_corr:    (T, 12) deterministic no-correction baseline
        delta_z:    (num_samples, T, latent_dim)
        mu_prior:   (num_samples, T, latent_dim)
        z_ctrl:     (num_samples, T, latent_dim)
    """
    T = traj["T"]
    actions = traj["actions"].to(device)
    main_imgs = traj["imgs_main"].to(device)
    extra_imgs = traj["imgs_extra"].to(device)
    action_mean_d = action_mean.to(device)
    action_std_d = action_std.to(device)
    latent_dim = int(policy.latent_dim)

    # Enable dropout for stochastic rollouts (VAE stays eval via override)
    policy.train()

    ar_runs = torch.zeros((num_samples, T, 12), device=device)
    delta_z_all = torch.zeros((num_samples, T, latent_dim), device=device)
    mu_prior_all = torch.zeros((num_samples, T, latent_dim), device=device)
    z_ctrl_all = torch.zeros((num_samples, T, latent_dim), device=device)

    # Run each sample independently for independent dropout masks per step
    with torch.no_grad():
        for s in range(num_samples):
            action_seq = actions.clone()  # (T, 12) — will be replaced in AR

            for t in range(T):
                state = ((action_seq[t:t + 1] - action_mean_d) / action_std_d)
                state = apply_state_mask(state, state_mask)

                window = build_past_window(action_seq[:, 6:12], t, window_size)
                window = window.unsqueeze(0)  # (1, 8, 6)

                img_main = (main_imgs[t].float() / 255.0).unsqueeze(0)
                img_extra = (extra_imgs[t].float() / 255.0).unsqueeze(0)

                out = policy(
                    img_main=img_main,
                    img_extra=img_extra,
                    state=state,
                    past_hand_win=window,
                )

                ar_runs[s, t, :] = out["action_pred"][0]
                delta_z_all[s, t, :] = out["delta_z"][0]
                mu_prior_all[s, t, :] = out["mu_prior"][0]
                z_ctrl_all[s, t, :] = out["z_ctrl"][0]

                # Feed back prediction to next step (AR)
                if t + 1 < T:
                    action_seq[t + 1, :] = out["action_pred"][0]

    # No-correction baseline: deterministic (eval mode, delta_z=0)
    policy.eval()
    no_corr = torch.zeros((T, 12), device=device)
    with torch.no_grad():
        action_seq = actions.clone()
        for t in range(T):
            state = ((action_seq[t:t + 1] - action_mean_d) / action_std_d)
            state = apply_state_mask(state, state_mask)
            window = build_past_window(action_seq[:, 6:12], t, window_size).unsqueeze(0)
            img_main = (main_imgs[t].float() / 255.0).unsqueeze(0)
            img_extra = (extra_imgs[t].float() / 255.0).unsqueeze(0)
            out = policy(
                img_main=img_main, img_extra=img_extra,
                state=state, past_hand_win=window, zero_delta=True,
            )
            no_corr[t, :] = out["action_pred"][0]
            if t + 1 < T:
                action_seq[t + 1, :] = out["action_pred"][0]

    return {
        "ar_runs": ar_runs.cpu().numpy(),
        "no_corr": no_corr.cpu().numpy(),
        "delta_z": delta_z_all.cpu().numpy(),
        "mu_prior": mu_prior_all.cpu().numpy(),
        "z_ctrl": z_ctrl_all.cpu().numpy(),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────


def plot_trajectory_actions(traj_id, T, gt_target, ar_runs, no_corr, num_samples,
                            output_dir, onset_step=None):
    """VAE-eval style: 4x3 grid, individual samples + mean + std + GT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_mean = ar_runs.mean(axis=0)   # (T, 12)
    ar_std = ar_runs.std(axis=0)     # (T, 12)
    steps = np.arange(T)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True)
    fig.suptitle(
        f"Trajectory {traj_id}:  AR rollout with MC Dropout  "
        f"(T={T}, n={num_samples})",
        fontsize=14,
    )

    for i, name in enumerate(JOINT_NAMES):
        ax = axes.flat[i]

        # Individual AR samples (light colored)
        for s in range(num_samples):
            ax.plot(steps, ar_runs[s, :, i],
                    color="steelblue", alpha=0.25, linewidth=0.8)

        # AR mean (bold)
        ax.plot(steps, ar_mean[:, i],
                color="royalblue", linewidth=2.0,
                label=f"AR mean (n={num_samples})")

        # AR std fill
        ax.fill_between(steps,
                         ar_mean[:, i] - ar_std[:, i],
                         ar_mean[:, i] + ar_std[:, i],
                         color="steelblue", alpha=0.15)

        # GT next-pose (black)
        ax.plot(steps, gt_target[:, i],
                "k-", linewidth=2.0, label="GT next-pose", zorder=10)

        # No-correction baseline (dashed gray)
        ax.plot(steps, no_corr[:, i],
                color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
                label="No correction (VAE prior)")

        # Onset marker
        if onset_step is not None:
            ax.axvline(onset_step, color="red", linestyle=":", linewidth=1.0,
                       alpha=0.6)

        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Decision step t", fontsize=10)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"traj_{traj_id}_ar_actions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trajectory_mse(traj_id, T, gt_target, ar_runs, no_corr, output_dir,
                        onset_step=None):
    """Per-step MSE plot: arm and hand side-by-side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Per-step MSE averaged over samples
    ar_arm_mse = ((ar_runs[:, :, :6] - gt_target[:, :6]) ** 2).mean(axis=2).mean(axis=0)
    ar_hand_mse = ((ar_runs[:, :, 6:] - gt_target[:, 6:]) ** 2).mean(axis=2).mean(axis=0)
    nc_arm_mse = ((no_corr[:, :6] - gt_target[:, :6]) ** 2).mean(axis=1)
    nc_hand_mse = ((no_corr[:, 6:] - gt_target[:, 6:]) ** 2).mean(axis=1)

    # Copy baseline
    copy_arm = ((gt_target[:-1, :6] - gt_target[1:, :6]) ** 2).mean(axis=1)
    copy_hand = ((gt_target[:-1, 6:] - gt_target[1:, 6:]) ** 2).mean(axis=1)
    # Pad last step
    copy_arm = np.concatenate([copy_arm, copy_arm[-1:]])
    copy_hand = np.concatenate([copy_hand, copy_hand[-1:]])

    steps = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Trajectory {traj_id}: Per-step MSE", fontsize=13)

    for ax, side, ar_mse, nc_mse, copy_mse in [
        (axes[0], "Arm", ar_arm_mse, nc_arm_mse, copy_arm),
        (axes[1], "Hand", ar_hand_mse, nc_hand_mse, copy_hand),
    ]:
        ax.plot(steps, ar_mse, color="royalblue", linewidth=1.5,
                label=f"AR (mean={ar_mse.mean():.5f})")
        ax.plot(steps, nc_mse, color="gray", linestyle="--", linewidth=1.2,
                label=f"No-corr (mean={nc_mse.mean():.5f})")
        ax.plot(steps, copy_mse, "k:", linewidth=1.0, alpha=0.5,
                label=f"Copy baseline (mean={copy_mse.mean():.5f})")
        if onset_step is not None:
            ax.axvline(onset_step, color="red", linestyle=":", linewidth=1.0,
                       alpha=0.6)
        ax.set_xlabel("Decision step t")
        ax.set_ylabel("MSE")
        ax.set_title(f"{side} MSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    path = os.path.join(output_dir, f"traj_{traj_id}_ar_mse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(all_results, output_dir, num_samples):
    """Summary bar chart: mean MSE across all trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_arm_vals = [r["ar_arm_mse"] for r in all_results]
    ar_hand_vals = [r["ar_hand_mse"] for r in all_results]
    nc_arm_vals = [r["nc_arm_mse"] for r in all_results]
    nc_hand_vals = [r["nc_hand_mse"] for r in all_results]
    copy_arm_vals = [r["copy_arm_mse"] for r in all_results]
    copy_hand_vals = [r["copy_hand_mse"] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"BC AR Evaluation Summary  ({len(all_results)} trajectories, "
        f"n={num_samples} MC Dropout samples)",
        fontsize=13,
    )

    for ax, side, ar_v, nc_v, cp_v in [
        (axes[0], "Arm", ar_arm_vals, nc_arm_vals, copy_arm_vals),
        (axes[1], "Hand", ar_hand_vals, nc_hand_vals, copy_hand_vals),
    ]:
        labels = ["AR (MC Drop)", "No Correction", "Copy Baseline"]
        values = [np.mean(ar_v), np.mean(nc_v), np.mean(cp_v)]
        colors = ["royalblue", "gray", "black"]
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mean MSE")
        ax.set_title(f"{side} MSE")
        ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1.0)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "summary_ar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot: {path}")


def plot_latent_diagnostics(all_results, output_dir, num_samples):
    """Delta_z magnitude and mu_prior distribution across all trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Latent Diagnostics  ({len(all_results)} trajectories, n={num_samples})",
        fontsize=13,
    )

    # Collect per-step stats across all trajectories
    all_delta_z_norm = []
    all_mu_prior_norm = []
    all_delta_z_dim0 = []
    all_delta_z_dim1 = []

    for r in all_results:
        dz = r["delta_z"]          # (n, T, latent_dim)
        mu = r["mu_prior"]         # (n, T, latent_dim)
        dz_norm = np.linalg.norm(dz, axis=-1).flatten()
        mu_norm = np.linalg.norm(mu, axis=-1).flatten()
        all_delta_z_norm.extend(dz_norm.tolist())
        all_mu_prior_norm.extend(mu_norm.tolist())
        all_delta_z_dim0.extend(dz[:, :, 0].flatten().tolist())
        if dz.shape[-1] > 1:
            all_delta_z_dim1.extend(dz[:, :, 1].flatten().tolist())

    # 1. Delta_z norm histogram
    ax = axes[0, 0]
    ax.hist(all_delta_z_norm, bins=80, color="steelblue", alpha=0.7, density=True)
    ax.axvline(np.mean(all_delta_z_norm), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(all_delta_z_norm):.4f}")
    ax.set_xlabel("|delta_z|")
    ax.set_title("delta_z magnitude distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. mu_prior norm histogram
    ax = axes[0, 1]
    ax.hist(all_mu_prior_norm, bins=80, color="gray", alpha=0.7, density=True)
    ax.axvline(np.mean(all_mu_prior_norm), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(all_mu_prior_norm):.4f}")
    ax.set_xlabel("|mu_prior|")
    ax.set_title("mu_prior magnitude distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Delta_z scatter (dim 0 vs dim 1)
    ax = axes[1, 0]
    if all_delta_z_dim1:
        ax.scatter(all_delta_z_dim0[:5000], all_delta_z_dim1[:5000],
                   s=2, alpha=0.3, color="steelblue")
        ax.set_xlabel("delta_z[0]")
        ax.set_ylabel("delta_z[1]")
        ax.set_title("delta_z scatter (first 5k points)")
    else:
        ax.hist(all_delta_z_dim0, bins=80, color="steelblue", alpha=0.7)
        ax.set_xlabel("delta_z[0]")
        ax.set_title("delta_z[0] distribution")
    ax.grid(True, alpha=0.3)

    # 4. Delta_z norm over time (averaged across all trajs)
    ax = axes[1, 1]
    max_T = max(r["T"] for r in all_results)
    dz_norm_by_t = [[] for _ in range(max_T)]
    for r in all_results:
        dz = r["delta_z"]   # (n, T, latent_dim)
        norms = np.linalg.norm(dz, axis=-1)  # (n, T)
        for t in range(r["T"]):
            dz_norm_by_t[t].extend(norms[:, t].tolist())

    mean_by_t = [np.mean(v) if v else 0 for v in dz_norm_by_t]
    std_by_t = [np.std(v) if v else 0 for v in dz_norm_by_t]
    steps = np.arange(len(mean_by_t))
    ax.plot(steps, mean_by_t, color="royalblue", linewidth=1.5,
            label="mean |delta_z|")
    ax.fill_between(steps,
                     np.array(mean_by_t) - np.array(std_by_t),
                     np.array(mean_by_t) + np.array(std_by_t),
                     color="steelblue", alpha=0.2)
    ax.set_xlabel("Decision step t")
    ax.set_ylabel("|delta_z|")
    ax.set_title("delta_z magnitude over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "latent_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Latent diagnostics: {path}")


def plot_per_trajectory_bar(all_results, output_dir):
    """Per-trajectory bar chart: AR hand MSE sorted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_results = sorted(all_results, key=lambda r: r["ar_hand_mse"])
    traj_ids = [r["traj_id"] for r in sorted_results]
    ar_hand = [r["ar_hand_mse"] for r in sorted_results]
    nc_hand = [r["nc_hand_mse"] for r in sorted_results]

    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(traj_ids) * 0.5), 5))
    x = np.arange(len(traj_ids))
    w = 0.35
    ax.bar(x - w / 2, ar_hand, w, color="royalblue", alpha=0.8, label="AR hand")
    ax.bar(x + w / 2, nc_hand, w, color="gray", alpha=0.6, label="No-corr hand")
    ax.set_xticks(x)
    ax.set_xticklabels(traj_ids, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Hand MSE")
    ax.set_title("Per-trajectory AR hand MSE (sorted)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(output_dir, "per_trajectory_hand_mse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-trajectory bar: {path}")


# ─── Onset detection (reuse from eval.py) ────────────────────────────────────

def detect_grasp_onset(actions, threshold=0.02, lookahead=3, min_count=2):
    if actions.shape[0] <= 1:
        return None
    delta = actions[1:, 6:] - actions[:-1, 6:]
    norm = np.linalg.norm(delta, axis=1)
    delta_norm = np.concatenate([norm, np.zeros((1,))])
    for t in range(actions.shape[0] - 1):
        window = delta_norm[t:min(actions.shape[0] - 1, t + lookahead)]
        if int((window > threshold).sum()) >= min_count:
            return int(t)
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="AR eval with MC Dropout visualization")
    parser.add_argument("--ckpt", type=str, required=True, help="BC checkpoint path")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Test data directory (default: read from checkpoint args)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of MC Dropout AR rollouts per trajectory")
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load checkpoint ──
    print(f"Loading: {args.ckpt}")
    bc_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    action_mean = bc_ckpt["action_mean"]
    action_std = bc_ckpt["action_std"]
    bc_args = bc_ckpt.get("args", {})

    vae_path = (
        args.vae_ckpt
        or bc_args.get("vae_ckpt")
        or os.path.join(_PROJ_ROOT, "outputs/dim_2_best/checkpoint.pth")
    )
    vae = build_and_freeze_vae(vae_path)
    policy = BCPolicy(
        vae=vae,
        state_dim=12,
        arm_state_dim=bc_args.get("arm_state_dim", 6),
        feat_dim=bc_args.get("feat_dim", 128),
        fusion_dim=bc_args.get("fusion_dim", 256),
        disable_vision=bc_args.get("disable_vision", False),
        dropout=bc_args.get("dropout", 0.0),
        hand_condition_on_arm=bc_args.get("hand_condition_on_arm", False),
    ).to(device)
    missing, unexpected = policy.load_state_dict(bc_ckpt["model"], strict=False)
    bc_only_missing = [k for k in missing if not k.startswith("vae.")]
    if bc_only_missing or unexpected:
        raise RuntimeError(f"Checkpoint incompatible: missing={bc_only_missing}, unexpected={unexpected}")

    dropout_rate = bc_args.get("dropout", 0.0)
    state_mask = bc_args.get("state_mask", "all")
    print(
        f"Loaded step {bc_ckpt.get('step', '?')}, dropout={dropout_rate}, "
        f"state_mask={state_mask}, hand_condition_on_arm={bc_args.get('hand_condition_on_arm', False)}"
    )

    if dropout_rate == 0:
        print("WARNING: This checkpoint has dropout=0. All MC Dropout samples will "
              "be identical. Consider using a checkpoint trained with dropout > 0.")

    # ── Load test trajectories ──
    test_dir = args.test_dir or bc_args.get("test_dir")
    if test_dir is None:
        test_dir = os.path.join(_PROJ_ROOT, "data/20260327-11:10:43/demos/success/test")
    files = sorted(glob.glob(os.path.join(test_dir, "trajectory_*_demo_expert.pt")))
    print(f"Found {len(files)} test trajectories in {test_dir}")

    # ── Evaluate each trajectory ──
    all_results = []
    for fi, f in enumerate(files):
        traj = load_trajectory(f)
        traj_id = traj["traj_id"]
        T = traj["T"]
        gt = traj["actions"].numpy()
        gt_target = np.concatenate([gt[1:], gt[-1:]], axis=0)  # actions[t+1]

        print(f"\n[{fi + 1}/{len(files)}] Trajectory {traj_id} (T={T})")

        rollout = rollout_ar_mc_dropout(
            policy=policy,
            traj=traj,
            action_mean=action_mean,
            action_std=action_std,
            num_samples=args.num_samples,
            device=device,
            state_mask=state_mask,
        )

        ar_runs = rollout["ar_runs"]   # (n, T, 12)
        no_corr = rollout["no_corr"]   # (T, 12)

        # MSE metrics
        ar_arm_mse = float(((ar_runs[:, :, :6] - gt_target[:, :6]) ** 2).mean())
        ar_hand_mse = float(((ar_runs[:, :, 6:] - gt_target[:, 6:]) ** 2).mean())
        nc_arm_mse = float(((no_corr[:, :6] - gt_target[:, :6]) ** 2).mean())
        nc_hand_mse = float(((no_corr[:, 6:] - gt_target[:, 6:]) ** 2).mean())
        copy_arm_mse = float(((gt[:, :6] - gt_target[:, :6]) ** 2).mean())
        copy_hand_mse = float(((gt[:, 6:] - gt_target[:, 6:]) ** 2).mean())

        onset_step = detect_grasp_onset(gt)

        result = {
            "traj_id": traj_id,
            "T": T,
            "onset_step": onset_step,
            "ar_arm_mse": ar_arm_mse,
            "ar_hand_mse": ar_hand_mse,
            "nc_arm_mse": nc_arm_mse,
            "nc_hand_mse": nc_hand_mse,
            "copy_arm_mse": copy_arm_mse,
            "copy_hand_mse": copy_hand_mse,
            "delta_z": rollout["delta_z"],
            "mu_prior": rollout["mu_prior"],
            "z_ctrl": rollout["z_ctrl"],
        }
        all_results.append(result)

        print(f"  AR  arm={ar_arm_mse:.6f}  hand={ar_hand_mse:.6f}")
        print(f"  NC  arm={nc_arm_mse:.6f}  hand={nc_hand_mse:.6f}")
        print(f"  onset_step={onset_step}")

        # Per-trajectory plots
        plot_trajectory_actions(
            traj_id, T, gt_target, ar_runs, no_corr, args.num_samples,
            args.output_dir, onset_step=onset_step,
        )
        plot_trajectory_mse(
            traj_id, T, gt_target, ar_runs, no_corr, args.output_dir,
            onset_step=onset_step,
        )

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(all_results)} trajectories, {args.num_samples} MC Dropout samples)")
    print("=" * 70)
    mean_ar_arm = np.mean([r["ar_arm_mse"] for r in all_results])
    mean_ar_hand = np.mean([r["ar_hand_mse"] for r in all_results])
    mean_nc_arm = np.mean([r["nc_arm_mse"] for r in all_results])
    mean_nc_hand = np.mean([r["nc_hand_mse"] for r in all_results])
    mean_copy_arm = np.mean([r["copy_arm_mse"] for r in all_results])
    mean_copy_hand = np.mean([r["copy_hand_mse"] for r in all_results])
    print(f"  AR mean:        arm={mean_ar_arm:.6f}  hand={mean_ar_hand:.6f}")
    print(f"  No-corr mean:   arm={mean_nc_arm:.6f}  hand={mean_nc_hand:.6f}")
    print(f"  Copy baseline:  arm={mean_copy_arm:.6f}  hand={mean_copy_hand:.6f}")

    # Delta_z stats
    all_dz = np.concatenate([r["delta_z"].reshape(-1, r["delta_z"].shape[-1])
                              for r in all_results], axis=0)
    dz_norm = np.linalg.norm(all_dz, axis=-1)
    print(f"\n  delta_z norm:   mean={dz_norm.mean():.4f}  std={dz_norm.std():.4f}  "
          f"max={dz_norm.max():.4f}")

    # Summary JSON
    summary = {
        "ckpt": args.ckpt,
        "num_samples": args.num_samples,
        "dropout": dropout_rate,
        "num_trajectories": len(all_results),
        "ar_arm_mse": float(mean_ar_arm),
        "ar_hand_mse": float(mean_ar_hand),
        "nc_arm_mse": float(mean_nc_arm),
        "nc_hand_mse": float(mean_nc_hand),
        "copy_arm_mse": float(mean_copy_arm),
        "copy_hand_mse": float(mean_copy_hand),
        "delta_z_norm_mean": float(dz_norm.mean()),
        "delta_z_norm_std": float(dz_norm.std()),
        "per_trajectory": [
            {
                "traj_id": r["traj_id"],
                "T": r["T"],
                "onset_step": r["onset_step"],
                "ar_arm_mse": r["ar_arm_mse"],
                "ar_hand_mse": r["ar_hand_mse"],
                "nc_arm_mse": r["nc_arm_mse"],
                "nc_hand_mse": r["nc_hand_mse"],
            }
            for r in all_results
        ],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON: {summary_path}")

    # Summary plots
    plot_summary(all_results, args.output_dir, args.num_samples)
    plot_per_trajectory_bar(all_results, args.output_dir)
    plot_latent_diagnostics(all_results, args.output_dir, args.num_samples)

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
