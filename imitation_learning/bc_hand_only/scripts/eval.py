"""
AR evaluation for hand-only BC: MC Dropout rollouts with visualization.

For each test trajectory, generates:
  - traj_{id}_ar_actions.png: 2x3 grid (6 hand joints) with GT / AR / no-corr
  - traj_{id}_ar_mse.png: per-step hand MSE comparison
  - summary_ar.png: bar chart across all trajectories
  - per_trajectory_hand_mse.png: sorted per-trajectory bars
  - latent_diagnostics.png: delta_z / mu_prior distributions
  - summary.json: aggregated metrics

Usage:
    python imitation_learning/bc_hand_only/scripts/eval.py \
        --ckpt outputs/bc_hand_only_sweep/baseline/checkpoint.pth \
        --output_dir visualizations/bc_hand_only_sweep/baseline \
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
_BC_HAND_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_HAND_ROOT, "..", ".."))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_policy_mod = _load(os.path.join(_BC_HAND_ROOT, "model/bc_hand_policy.py"), "bc_hand_policy")
BCHandPolicy = _policy_mod.BCHandPolicy
build_and_freeze_vae = _policy_mod.build_and_freeze_vae

HAND_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_trajectory(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    actions = data["actions"][:, 0, :].float()
    main = data["curr_obs"]["main_images"][:, 0].permute(0, 3, 1, 2).contiguous()
    extra = data["curr_obs"]["extra_view_images"][:, 0, 0].permute(0, 3, 1, 2).contiguous()
    traj_id = os.path.basename(path).split("trajectory_")[1].split("_")[0]
    return {
        "traj_id": traj_id,
        "actions": actions,     # (T, 12) full actions
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


def rollout_ar_hand_only(
    policy: BCHandPolicy,
    traj: dict,
    num_samples: int,
    device: torch.device,
    window_size: int = 8,
) -> dict:
    """AR rollout for hand-only BC. Arm state always GT; only hand is fed back.

    Returns dict with:
        ar_hand:    (num_samples, T, 6) predicted hand actions
        no_corr:    (T, 6) deterministic no-correction baseline
        delta_z:    (num_samples, T, latent_dim)
        mu_prior:   (num_samples, T, latent_dim)
        z_ctrl:     (num_samples, T, latent_dim)
    """
    T = traj["T"]
    actions = traj["actions"].to(device)       # (T, 12)
    main_imgs = traj["imgs_main"].to(device)
    extra_imgs = traj["imgs_extra"].to(device)
    latent_dim = int(policy.latent_dim)

    policy.train()  # enable dropout for MC samples

    ar_hand = torch.zeros((num_samples, T, 6), device=device)
    delta_z_all = torch.zeros((num_samples, T, latent_dim), device=device)
    mu_prior_all = torch.zeros((num_samples, T, latent_dim), device=device)
    z_ctrl_all = torch.zeros((num_samples, T, latent_dim), device=device)

    with torch.no_grad():
        for s in range(num_samples):
            # Hand history: starts as GT, overwritten by predictions in AR
            hand_seq = actions[:, 6:12].clone()  # (T, 6)

            for t in range(T):
                window = build_past_window(hand_seq, t, window_size).unsqueeze(0)
                img_main = (main_imgs[t].float() / 255.0).unsqueeze(0)
                img_extra = (extra_imgs[t].float() / 255.0).unsqueeze(0)

                out = policy(
                    img_main=img_main,
                    img_extra=img_extra,
                    past_hand_win=window,
                )

                ar_hand[s, t, :] = out["hand_action"][0]
                delta_z_all[s, t, :] = out["delta_z"][0]
                mu_prior_all[s, t, :] = out["mu_prior"][0]
                z_ctrl_all[s, t, :] = out["z_ctrl"][0]

                # Feed back hand prediction to next step
                if t + 1 < T:
                    hand_seq[t + 1, :] = out["hand_action"][0]

    # No-correction baseline (eval mode, delta_z=0)
    policy.eval()
    no_corr_hand = torch.zeros((T, 6), device=device)
    with torch.no_grad():
        hand_seq = actions[:, 6:12].clone()
        for t in range(T):
            window = build_past_window(hand_seq, t, window_size).unsqueeze(0)
            img_main = (main_imgs[t].float() / 255.0).unsqueeze(0)
            img_extra = (extra_imgs[t].float() / 255.0).unsqueeze(0)
            out = policy(
                img_main=img_main, img_extra=img_extra,
                past_hand_win=window, zero_delta=True,
            )
            no_corr_hand[t, :] = out["hand_action"][0]
            if t + 1 < T:
                hand_seq[t + 1, :] = out["hand_action"][0]

    return {
        "ar_hand": ar_hand.cpu().numpy(),
        "no_corr": no_corr_hand.cpu().numpy(),
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


def plot_trajectory_actions(traj_id, T, gt_hand_target, ar_hand, no_corr,
                            num_samples, output_dir, onset_step=None):
    """2x3 grid: 6 hand joints with GT / AR samples / mean / std / no-corr."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_mean = ar_hand.mean(axis=0)   # (T, 6)
    ar_std = ar_hand.std(axis=0)     # (T, 6)
    steps = np.arange(T)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle(
        f"Trajectory {traj_id}:  Hand-Only AR rollout  "
        f"(T={T}, n={num_samples})",
        fontsize=14,
    )

    for i, name in enumerate(HAND_NAMES):
        ax = axes.flat[i]

        for s in range(num_samples):
            ax.plot(steps, ar_hand[s, :, i],
                    color="steelblue", alpha=0.25, linewidth=0.8)

        ax.plot(steps, ar_mean[:, i],
                color="royalblue", linewidth=2.0,
                label=f"AR mean (n={num_samples})")
        ax.fill_between(steps,
                         ar_mean[:, i] - ar_std[:, i],
                         ar_mean[:, i] + ar_std[:, i],
                         color="steelblue", alpha=0.15)
        ax.plot(steps, gt_hand_target[:, i],
                "k-", linewidth=2.0, label="GT next-pose", zorder=10)
        ax.plot(steps, no_corr[:, i],
                color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
                label="No correction (VAE prior)")

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


def plot_trajectory_mse(traj_id, T, gt_hand_target, ar_hand, no_corr,
                        output_dir, onset_step=None):
    """Per-step hand MSE plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_hand_mse = ((ar_hand - gt_hand_target) ** 2).mean(axis=2).mean(axis=0)
    nc_hand_mse = ((no_corr - gt_hand_target) ** 2).mean(axis=1)
    copy_hand = ((gt_hand_target[:-1] - gt_hand_target[1:]) ** 2).mean(axis=1)
    copy_hand = np.concatenate([copy_hand, copy_hand[-1:]])

    steps = np.arange(T)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(f"Trajectory {traj_id}: Per-step Hand MSE", fontsize=13)

    ax.plot(steps, ar_hand_mse, color="royalblue", linewidth=1.5,
            label=f"AR (mean={ar_hand_mse.mean():.5f})")
    ax.plot(steps, nc_hand_mse, color="gray", linestyle="--", linewidth=1.2,
            label=f"No-corr (mean={nc_hand_mse.mean():.5f})")
    ax.plot(steps, copy_hand, "k:", linewidth=1.0, alpha=0.5,
            label=f"Copy baseline (mean={copy_hand.mean():.5f})")
    if onset_step is not None:
        ax.axvline(onset_step, color="red", linestyle=":", linewidth=1.0,
                   alpha=0.6)
    ax.set_xlabel("Decision step t")
    ax.set_ylabel("MSE")
    ax.set_title("Hand MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    path = os.path.join(output_dir, f"traj_{traj_id}_ar_mse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(all_results, output_dir, num_samples):
    """Summary bar chart: hand MSE across all trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_hand_vals = [r["ar_hand_mse"] for r in all_results]
    nc_hand_vals = [r["nc_hand_mse"] for r in all_results]
    copy_hand_vals = [r["copy_hand_mse"] for r in all_results]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(
        f"Hand-Only BC AR Summary  ({len(all_results)} trajectories, "
        f"n={num_samples} MC Dropout samples)",
        fontsize=13,
    )

    labels = ["AR (MC Drop)", "No Correction", "Copy Baseline"]
    values = [np.mean(ar_hand_vals), np.mean(nc_hand_vals), np.mean(copy_hand_vals)]
    colors = ["royalblue", "gray", "black"]
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Hand MSE")
    ax.set_title("Hand MSE")
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "summary_ar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot: {path}")


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


def plot_latent_diagnostics(all_results, output_dir, num_samples):
    """Delta_z and mu_prior distributions across all trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Latent Diagnostics  ({len(all_results)} trajectories, n={num_samples})",
        fontsize=13,
    )

    all_delta_z_norm = []
    all_mu_prior_norm = []
    all_delta_z_dim0 = []
    all_delta_z_dim1 = []

    for r in all_results:
        dz = r["delta_z"]
        mu = r["mu_prior"]
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

    # 4. Delta_z norm over time
    ax = axes[1, 1]
    max_T = max(r["T"] for r in all_results)
    dz_norm_by_t = [[] for _ in range(max_T)]
    for r in all_results:
        dz = r["delta_z"]
        norms = np.linalg.norm(dz, axis=-1)
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


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Hand-only AR eval with MC Dropout")
    parser.add_argument("--ckpt", type=str, required=True, help="BC hand-only checkpoint")
    parser.add_argument("--test_dir", type=str, default=None)
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
    bc_args = bc_ckpt.get("args", {})

    vae_path = (
        args.vae_ckpt
        or bc_args.get("vae_ckpt")
        or os.path.join(_PROJ_ROOT, "outputs/dim_2_best/checkpoint.pth")
    )
    vae = build_and_freeze_vae(vae_path)
    policy = BCHandPolicy(
        vae=vae,
        feat_dim=bc_args.get("feat_dim", 128),
        fusion_dim=bc_args.get("fusion_dim", 256),
        disable_vision=bc_args.get("disable_vision", False),
        dropout=bc_args.get("dropout", 0.0),
    ).to(device)
    missing, unexpected = policy.load_state_dict(bc_ckpt["model"], strict=False)
    bc_only_missing = [k for k in missing if not k.startswith("vae.")]
    if bc_only_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint incompatible: missing={bc_only_missing}, unexpected={unexpected}"
        )

    dropout_rate = bc_args.get("dropout", 0.0)
    print(
        f"Loaded step {bc_ckpt.get('step', '?')}, dropout={dropout_rate}"
    )
    if dropout_rate == 0:
        print("WARNING: dropout=0 — all MC Dropout samples will be identical.")

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
        gt = traj["actions"].numpy()           # (T, 12)
        gt_hand = gt[:, 6:]                    # (T, 6)
        gt_hand_target = np.concatenate([gt_hand[1:], gt_hand[-1:]], axis=0)  # actions[t+1]

        print(f"\n[{fi + 1}/{len(files)}] Trajectory {traj_id} (T={T})")

        rollout = rollout_ar_hand_only(
            policy=policy,
            traj=traj,
            num_samples=args.num_samples,
            device=device,
        )

        ar_hand = rollout["ar_hand"]     # (n, T, 6)
        no_corr = rollout["no_corr"]     # (T, 6)

        ar_hand_mse = float(((ar_hand - gt_hand_target) ** 2).mean())
        nc_hand_mse = float(((no_corr - gt_hand_target) ** 2).mean())
        copy_hand_mse = float(((gt_hand - gt_hand_target) ** 2).mean())

        onset_step = detect_grasp_onset(gt)

        result = {
            "traj_id": traj_id,
            "T": T,
            "onset_step": onset_step,
            "ar_hand_mse": ar_hand_mse,
            "nc_hand_mse": nc_hand_mse,
            "copy_hand_mse": copy_hand_mse,
            "delta_z": rollout["delta_z"],
            "mu_prior": rollout["mu_prior"],
            "z_ctrl": rollout["z_ctrl"],
        }
        all_results.append(result)

        print(f"  AR  hand={ar_hand_mse:.6f}")
        print(f"  NC  hand={nc_hand_mse:.6f}")
        print(f"  onset_step={onset_step}")

        plot_trajectory_actions(
            traj_id, T, gt_hand_target, ar_hand, no_corr, args.num_samples,
            args.output_dir, onset_step=onset_step,
        )
        plot_trajectory_mse(
            traj_id, T, gt_hand_target, ar_hand, no_corr, args.output_dir,
            onset_step=onset_step,
        )

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(all_results)} trajectories, {args.num_samples} MC Dropout samples)")
    print("=" * 70)
    mean_ar_hand = np.mean([r["ar_hand_mse"] for r in all_results])
    mean_nc_hand = np.mean([r["nc_hand_mse"] for r in all_results])
    mean_copy_hand = np.mean([r["copy_hand_mse"] for r in all_results])
    print(f"  AR mean:        hand={mean_ar_hand:.6f}")
    print(f"  No-corr mean:   hand={mean_nc_hand:.6f}")
    print(f"  Copy baseline:  hand={mean_copy_hand:.6f}")

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
        "ar_hand_mse": float(mean_ar_hand),
        "nc_hand_mse": float(mean_nc_hand),
        "copy_hand_mse": float(mean_copy_hand),
        "delta_z_norm_mean": float(dz_norm.mean()),
        "delta_z_norm_std": float(dz_norm.std()),
        "per_trajectory": [
            {
                "traj_id": r["traj_id"],
                "T": r["T"],
                "onset_step": r["onset_step"],
                "ar_hand_mse": r["ar_hand_mse"],
                "nc_hand_mse": r["nc_hand_mse"],
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
