"""
AR evaluation for BC 3.0: autoregressive rollout with visualization.

For each test trajectory, generates:
  - traj_{id}_ar_actions.png: 4x3 grid (12 joints) with GT / AR / no-corr
  - traj_{id}_ar_mse.png: per-step arm + hand MSE
  - summary_ar.png: bar chart across all trajectories
  - per_trajectory_hand_mse.png: sorted per-trajectory bars
  - latent_diagnostics.png: delta_z / mu_prior distributions
  - summary.json: aggregated metrics

Usage:
    python imitation_learning/behavior_clone/scripts/eval.py \
        --ckpt outputs/bc_v6_noise/noise01/checkpoint.pth \
        --output_dir visualizations/bc_v6_noise/noise01_full \
        --num_samples 5
"""

import argparse
import glob
import importlib.util
import json
import os

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


# ─── AR rollout ──────────────────────────────────────────────────────────────


def rollout_ar(
    policy: BCPolicy,
    traj: dict,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    num_samples: int,
    device: torch.device,
    window_size: int = 8,
    rollout_stride: int = 1,
) -> dict:
    """AR rollout: both arm and hand predictions fed back to next step.

    In chunk mode, `rollout_stride` controls how many frames from each predicted
    chunk to execute before re-predicting. stride=1 matches single-step behavior
    (re-predict every frame); stride=future_horizon matches the VAE-eval setting
    (execute the full chunk, re-predict every H frames).

    Returns dict with:
        ar_runs:    (num_samples, T, 12)
        no_corr:    (T, 12) deterministic no-correction baseline
        delta_z:    (num_samples, T, latent_dim)  [frame the prediction was made at]
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
    chunk_mode = getattr(policy, "chunk_mode", False)
    H = getattr(policy, "future_horizon", 1) if chunk_mode else 1
    stride = max(1, min(rollout_stride, H))

    policy.train()  # enable dropout if any

    ar_runs = torch.zeros((num_samples, T, 12), device=device)
    delta_z_all = torch.zeros((num_samples, T, latent_dim), device=device)
    mu_prior_all = torch.zeros((num_samples, T, latent_dim), device=device)
    z_ctrl_all = torch.zeros((num_samples, T, latent_dim), device=device)

    with torch.no_grad():
        for s in range(num_samples):
            action_seq = actions.clone()

            t = 0
            while t < T:
                state = (action_seq[t:t + 1] - action_mean_d) / action_std_d
                window = build_past_window(action_seq[:, 6:12], t, window_size).unsqueeze(0)
                img_main = (main_imgs[t].float() / 255.0).unsqueeze(0)
                img_extra = (extra_imgs[t].float() / 255.0).unsqueeze(0)

                out = policy(
                    img_main=img_main, img_extra=img_extra,
                    state=state, past_hand_win=window,
                )

                raw_pred = out["action_pred"][0]  # (12,) single or (H, 12) chunk
                if chunk_mode and raw_pred.dim() == 2:
                    # Take first `stride` frames of the chunk, write to action_seq.
                    take = min(stride, T - t)
                    chunk = raw_pred[:take]  # (take, 12)
                    for k in range(take):
                        ar_runs[s, t + k, :] = chunk[k]
                        delta_z_all[s, t + k, :] = out["delta_z"][0]
                        mu_prior_all[s, t + k, :] = out["mu_prior"][0]
                        z_ctrl_all[s, t + k, :] = out["z_ctrl"][0]
                        if t + k + 1 < T:
                            action_seq[t + k + 1, :] = chunk[k]
                    t += take
                else:
                    # Single-step mode
                    pred = raw_pred
                    ar_runs[s, t, :] = pred
                    delta_z_all[s, t, :] = out["delta_z"][0]
                    mu_prior_all[s, t, :] = out["mu_prior"][0]
                    z_ctrl_all[s, t, :] = out["z_ctrl"][0]
                    if t + 1 < T:
                        action_seq[t + 1, :] = pred
                    t += 1

    # No-correction baseline (eval mode, delta_z=0)
    policy.eval()
    no_corr = torch.zeros((T, 12), device=device)
    with torch.no_grad():
        action_seq = actions.clone()
        t = 0
        while t < T:
            state = (action_seq[t:t + 1] - action_mean_d) / action_std_d
            window = build_past_window(action_seq[:, 6:12], t, window_size).unsqueeze(0)
            img_main = (main_imgs[t].float() / 255.0).unsqueeze(0)
            img_extra = (extra_imgs[t].float() / 255.0).unsqueeze(0)
            out = policy(
                img_main=img_main, img_extra=img_extra,
                state=state, past_hand_win=window, zero_delta=True,
            )
            raw_pred = out["action_pred"][0]
            if chunk_mode and raw_pred.dim() == 2:
                take = min(stride, T - t)
                chunk = raw_pred[:take]
                for k in range(take):
                    no_corr[t + k, :] = chunk[k]
                    if t + k + 1 < T:
                        action_seq[t + k + 1, :] = chunk[k]
                t += take
                continue
            pred = raw_pred
            no_corr[t, :] = pred
            if t + 1 < T:
                action_seq[t + 1, :] = pred
            t += 1

    return {
        "ar_runs": ar_runs.cpu().numpy(),
        "no_corr": no_corr.cpu().numpy(),
        "delta_z": delta_z_all.cpu().numpy(),
        "mu_prior": mu_prior_all.cpu().numpy(),
        "z_ctrl": z_ctrl_all.cpu().numpy(),
    }


# ─── Onset detection ─────────────────────────────────────────────────────────


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


# ─── Plotting ─────────────────────────────────────────────────────────────────


def plot_trajectory_actions(traj_id, T, gt_target, ar_runs, no_corr,
                            num_samples, output_dir, onset_step=None,
                            xlim_max=None, xtick_step=5):
    """4x3 grid: 12 joints with GT / AR samples / mean / std / no-corr."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_mean = ar_runs.mean(axis=0)
    ar_std = ar_runs.std(axis=0)
    steps = np.arange(T)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True)
    fig.suptitle(
        f"Trajectory {traj_id}:  AR rollout  (T={T}, n={num_samples})",
        fontsize=14,
    )

    for i, name in enumerate(JOINT_NAMES):
        ax = axes.flat[i]
        for s in range(num_samples):
            ax.plot(steps, ar_runs[s, :, i],
                    color="steelblue", alpha=0.25, linewidth=0.8)
        ax.plot(steps, ar_mean[:, i],
                color="royalblue", linewidth=2.0,
                label=f"AR mean (n={num_samples})")
        ax.fill_between(steps,
                         ar_mean[:, i] - ar_std[:, i],
                         ar_mean[:, i] + ar_std[:, i],
                         color="steelblue", alpha=0.15)
        ax.plot(steps, gt_target[:, i],
                "k-", linewidth=2.0, label="GT next-pose", zorder=10)
        ax.plot(steps, no_corr[:, i],
                color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
                label="No correction (VAE prior)")
        if onset_step is not None:
            ax.axvline(onset_step, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        if xlim_max is not None:
            ax.set_xlim(0, xlim_max)
            ax.set_xticks(np.arange(0, xlim_max + 1, xtick_step))
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Decision step t", fontsize=10)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"traj_{traj_id}_ar_actions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_mse(traj_id, T, gt_target, ar_runs, no_corr,
                        output_dir, onset_step=None,
                        xlim_max=None, xtick_step=5):
    """Per-step MSE: arm and hand side-by-side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_arm_mse = ((ar_runs[:, :, :6] - gt_target[:, :6]) ** 2).mean(axis=2).mean(axis=0)
    ar_hand_mse = ((ar_runs[:, :, 6:] - gt_target[:, 6:]) ** 2).mean(axis=2).mean(axis=0)
    nc_arm_mse = ((no_corr[:, :6] - gt_target[:, :6]) ** 2).mean(axis=1)
    nc_hand_mse = ((no_corr[:, 6:] - gt_target[:, 6:]) ** 2).mean(axis=1)
    copy_arm = np.concatenate([((gt_target[:-1, :6] - gt_target[1:, :6]) ** 2).mean(axis=1),
                                np.zeros(1)])
    copy_hand = np.concatenate([((gt_target[:-1, 6:] - gt_target[1:, 6:]) ** 2).mean(axis=1),
                                 np.zeros(1)])
    steps = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Trajectory {traj_id}: Per-step MSE", fontsize=13)

    for ax, side, ar_mse, nc_mse, cp_mse in [
        (axes[0], "Arm", ar_arm_mse, nc_arm_mse, copy_arm),
        (axes[1], "Hand", ar_hand_mse, nc_hand_mse, copy_hand),
    ]:
        ax.plot(steps, ar_mse, color="royalblue", linewidth=1.5,
                label=f"AR (mean={ar_mse.mean():.5f})")
        ax.plot(steps, nc_mse, color="gray", linestyle="--", linewidth=1.2,
                label=f"No-corr (mean={nc_mse.mean():.5f})")
        ax.plot(steps, cp_mse, "k:", linewidth=1.0, alpha=0.5,
                label=f"Copy baseline (mean={cp_mse.mean():.5f})")
        if onset_step is not None:
            ax.axvline(onset_step, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Decision step t")
        ax.set_ylabel("MSE")
        ax.set_title(f"{side} MSE")
        ax.grid(True, alpha=0.3)
        if xlim_max is not None:
            ax.set_xlim(0, xlim_max)
            ax.set_xticks(np.arange(0, xlim_max + 1, xtick_step))
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    path = os.path.join(output_dir, f"traj_{traj_id}_ar_mse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(all_results, output_dir, num_samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"BC AR Evaluation Summary  ({len(all_results)} trajectories, n={num_samples})",
        fontsize=13,
    )
    for ax, side in [(axes[0], "arm"), (axes[1], "hand")]:
        ar_v = np.mean([r[f"ar_{side}_mse"] for r in all_results])
        nc_v = np.mean([r[f"nc_{side}_mse"] for r in all_results])
        cp_v = np.mean([r[f"copy_{side}_mse"] for r in all_results])
        labels = ["AR", "No Correction", "Copy Baseline"]
        values = [ar_v, nc_v, cp_v]
        colors = ["royalblue", "gray", "black"]
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mean MSE")
        ax.set_title(f"{side.title()} MSE")
        ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1.0)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "summary_ar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot: {path}")


def plot_per_trajectory_bar(all_results, output_dir):
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


def plot_latent_diagnostics(all_results, output_dir, num_samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Latent Diagnostics  ({len(all_results)} trajectories, n={num_samples})",
        fontsize=13,
    )

    all_dz_norm, all_mu_norm, all_dz_d0, all_dz_d1 = [], [], [], []
    for r in all_results:
        dz, mu = r["delta_z"], r["mu_prior"]
        all_dz_norm.extend(np.linalg.norm(dz, axis=-1).flatten().tolist())
        all_mu_norm.extend(np.linalg.norm(mu, axis=-1).flatten().tolist())
        all_dz_d0.extend(dz[:, :, 0].flatten().tolist())
        if dz.shape[-1] > 1:
            all_dz_d1.extend(dz[:, :, 1].flatten().tolist())

    ax = axes[0, 0]
    ax.hist(all_dz_norm, bins=80, color="steelblue", alpha=0.7, density=True)
    ax.axvline(np.mean(all_dz_norm), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(all_dz_norm):.4f}")
    ax.set_xlabel("|delta_z|"); ax.set_title("delta_z magnitude")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(all_mu_norm, bins=80, color="gray", alpha=0.7, density=True)
    ax.axvline(np.mean(all_mu_norm), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(all_mu_norm):.4f}")
    ax.set_xlabel("|mu_prior|"); ax.set_title("mu_prior magnitude")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if all_dz_d1:
        ax.scatter(all_dz_d0[:5000], all_dz_d1[:5000], s=2, alpha=0.3, color="steelblue")
        ax.set_xlabel("delta_z[0]"); ax.set_ylabel("delta_z[1]")
        ax.set_title("delta_z scatter (first 5k)")
    else:
        ax.hist(all_dz_d0, bins=80, color="steelblue", alpha=0.7)
        ax.set_xlabel("delta_z[0]"); ax.set_title("delta_z[0] distribution")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    max_T = max(r["T"] for r in all_results)
    dz_by_t = [[] for _ in range(max_T)]
    for r in all_results:
        norms = np.linalg.norm(r["delta_z"], axis=-1)
        for t in range(r["T"]):
            dz_by_t[t].extend(norms[:, t].tolist())
    mean_t = [np.mean(v) if v else 0 for v in dz_by_t]
    std_t = [np.std(v) if v else 0 for v in dz_by_t]
    ax.plot(mean_t, color="royalblue", linewidth=1.5, label="mean |delta_z|")
    ax.fill_between(range(len(mean_t)),
                     np.array(mean_t) - np.array(std_t),
                     np.array(mean_t) + np.array(std_t),
                     color="steelblue", alpha=0.2)
    ax.set_xlabel("Decision step t"); ax.set_ylabel("|delta_z|")
    ax.set_title("delta_z magnitude over time")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "latent_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Latent diagnostics: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="BC 3.0 AR evaluation")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--rollout_stride", type=int, default=1,
                        help="Frames to execute from each predicted chunk "
                             "before re-predicting (chunk mode only). "
                             "stride=1: step-by-step; stride=future_horizon: full-chunk")
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
    vae, vae_type = build_and_freeze_vae(vae_path)
    chunk_mode = bc_args.get("chunk_mode", False)
    future_horizon = bc_args.get("future_horizon", 8) if chunk_mode else 1
    policy = BCPolicy(
        vae=vae,
        arm_state_dim=bc_args.get("arm_state_dim", 6),
        feat_dim=bc_args.get("feat_dim", 128),
        fusion_dim=bc_args.get("fusion_dim", 256),
        dropout=bc_args.get("dropout", 0.0),
        chunk_mode=chunk_mode,
        future_horizon=future_horizon,
        arm_gru_hidden=bc_args.get("arm_gru_hidden", 256),
        action_mean=action_mean if chunk_mode else None,
        action_std=action_std if chunk_mode else None,
    ).to(device)
    missing, unexpected = policy.load_state_dict(bc_ckpt["model"], strict=False)
    bc_only_missing = [k for k in missing if not k.startswith("vae.")]
    if bc_only_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint incompatible: missing={bc_only_missing}, unexpected={unexpected}"
        )

    dropout_rate = bc_args.get("dropout", 0.0)
    print(f"Loaded step {bc_ckpt.get('step', '?')}, dropout={dropout_rate}")
    if dropout_rate == 0:
        print("WARNING: dropout=0 — all samples will be identical.")

    # ── Load test trajectories ──
    test_dir = args.test_dir or bc_args.get("test_dir")
    if test_dir is None:
        test_dir = os.path.join(_PROJ_ROOT, "data/20260327-11:10:43/demos/success/test")
    files = sorted(glob.glob(os.path.join(test_dir, "trajectory_*_demo_expert.pt")))
    print(f"Found {len(files)} test trajectories in {test_dir}")

    # Pre-scan trajectory lengths so every per-traj plot can share the same x-axis.
    # Round up to the next multiple of xtick_step for clean ticks.
    xtick_step = 5
    max_T_raw = 0
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        max_T_raw = max(max_T_raw, int(data["actions"].shape[0]))
    global_xlim = int(np.ceil(max_T_raw / xtick_step) * xtick_step)
    print(f"Shared x-axis: xlim=[0, {global_xlim}]  (max T={max_T_raw}, step={xtick_step})")

    # ── Evaluate each trajectory ──
    all_results = []
    for fi, f in enumerate(files):
        traj = load_trajectory(f)
        traj_id = traj["traj_id"]
        T = traj["T"]
        gt = traj["actions"].numpy()
        gt_target = np.concatenate([gt[1:], gt[-1:]], axis=0)

        print(f"\n[{fi + 1}/{len(files)}] Trajectory {traj_id} (T={T})")

        rollout = rollout_ar(
            policy=policy, traj=traj,
            action_mean=action_mean, action_std=action_std,
            num_samples=args.num_samples, device=device,
            rollout_stride=args.rollout_stride,
        )

        ar_runs = rollout["ar_runs"]
        no_corr = rollout["no_corr"]

        ar_arm_mse = float(((ar_runs[:, :, :6] - gt_target[:, :6]) ** 2).mean())
        ar_hand_mse = float(((ar_runs[:, :, 6:] - gt_target[:, 6:]) ** 2).mean())
        nc_arm_mse = float(((no_corr[:, :6] - gt_target[:, :6]) ** 2).mean())
        nc_hand_mse = float(((no_corr[:, 6:] - gt_target[:, 6:]) ** 2).mean())
        copy_arm_mse = float(((gt[:, :6] - gt_target[:, :6]) ** 2).mean())
        copy_hand_mse = float(((gt[:, 6:] - gt_target[:, 6:]) ** 2).mean())

        onset_step = detect_grasp_onset(gt)

        result = {
            "traj_id": traj_id, "T": T, "onset_step": onset_step,
            "ar_arm_mse": ar_arm_mse, "ar_hand_mse": ar_hand_mse,
            "nc_arm_mse": nc_arm_mse, "nc_hand_mse": nc_hand_mse,
            "copy_arm_mse": copy_arm_mse, "copy_hand_mse": copy_hand_mse,
            "delta_z": rollout["delta_z"],
            "mu_prior": rollout["mu_prior"],
            "z_ctrl": rollout["z_ctrl"],
        }
        all_results.append(result)

        print(f"  AR  arm={ar_arm_mse:.6f}  hand={ar_hand_mse:.6f}")
        print(f"  NC  arm={nc_arm_mse:.6f}  hand={nc_hand_mse:.6f}")

        plot_trajectory_actions(
            traj_id, T, gt_target, ar_runs, no_corr, args.num_samples,
            args.output_dir, onset_step=onset_step,
            xlim_max=global_xlim, xtick_step=xtick_step,
        )
        plot_trajectory_mse(
            traj_id, T, gt_target, ar_runs, no_corr, args.output_dir,
            onset_step=onset_step,
            xlim_max=global_xlim, xtick_step=xtick_step,
        )

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(all_results)} trajectories, {args.num_samples} samples)")
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

    all_dz = np.concatenate([r["delta_z"].reshape(-1, r["delta_z"].shape[-1])
                              for r in all_results], axis=0)
    dz_norm = np.linalg.norm(all_dz, axis=-1)
    print(f"\n  delta_z norm:   mean={dz_norm.mean():.4f}  std={dz_norm.std():.4f}  "
          f"max={dz_norm.max():.4f}")

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
                "traj_id": r["traj_id"], "T": r["T"], "onset_step": r["onset_step"],
                "ar_arm_mse": r["ar_arm_mse"], "ar_hand_mse": r["ar_hand_mse"],
                "nc_arm_mse": r["nc_arm_mse"], "nc_hand_mse": r["nc_hand_mse"],
            }
            for r in all_results
        ],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON: {summary_path}")

    plot_summary(all_results, args.output_dir, args.num_samples)
    plot_per_trajectory_bar(all_results, args.output_dir)
    plot_latent_diagnostics(all_results, args.output_dir, args.num_samples)

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
