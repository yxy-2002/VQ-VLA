"""
Evaluate Behavior Cloning policy over a frozen Hand-Action VAE.

BC operates per-step on obs (image + state), so unlike `vae/scripts/eval.py`
there is no env-simulating "free run" mode. Instead we expose three rollouts
that all consume GT obs at every step but vary how the past_hand_window is fed:

  TF (teacher-forced):  GT past_hand_window at every step.
                        Best-case BC accuracy.

  AR-hand (auto-hand):  past_hand_window is filled with the BC's OWN predicted
                        hand actions (replaced step by step). Tests whether
                        the hand predictions stay coherent when fed back —
                        the closest analogue to deployment we can get without
                        a simulator.

  NO-CORR (baseline):   GT past_hand_window, but delta_mu=delta_log_var=0
                        (BC delta heads forced to zero, hand decoded from VAE
                        prior alone). This is the bar that BC's vision-driven
                        hand correction must beat to demonstrate any value.

Each mode is run with --num_samples stochastic reparameterize draws so we can
report per-step variance (the VAE README explicitly warns deterministic mode
is broken — see vae/README.md trick #5).

Usage:
    python imitation_learning/behavior_clone/scripts/eval.py \
        --ckpt outputs/bc_simple_v1/checkpoint.pth \
        --num_samples 5 \
        --traj_id 105 107 \
        --output_dir visualizations/bc_eval/v1

Outputs (in --output_dir):
    traj_<id>_eval.npz       raw per-trajectory data (gt, runs, means, stds, mses)
    traj_<id>_actions.png    12-joint comparison plot (GT vs all 3 modes)
    traj_<id>_mse.png        per-step arm/hand MSE plot
    summary.json             aggregate metrics across trajectories
    summary.png              bar chart of per-mode mean MSE
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


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_policy_mod = _load(os.path.join(_BC_ROOT, "model/bc_policy.py"), "bc_policy")
BCPolicy = _policy_mod.BCPolicy
build_and_freeze_vae = _policy_mod.build_and_freeze_vae


# ─── Constants ────────────────────────────────────────────────────────────────

ARM_NAMES = [f"arm_{i}" for i in range(6)]
HAND_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
JOINT_NAMES = ARM_NAMES + HAND_NAMES  # 12 dims total

MODES = ["tf", "ar", "no_corr"]
MODE_LABELS = {
    "tf":      "Teacher-Forced",
    "ar":      "AR-hand",
    "no_corr": "No Correction (VAE prior)",
}
MODE_COLORS = {
    "tf":      "tab:green",
    "ar":      "tab:red",
    "no_corr": "tab:blue",
}


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_trajectory_full(path: str) -> dict:
    """Load actions + obs from a trajectory .pt file (matches BCDataset)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    actions = data["actions"][:, 0, :].float()                # (T, 12)
    states = data["curr_obs"]["states"][:, 0, :].float()      # (T, 24)
    main = data["curr_obs"]["main_images"][:, 0]              # (T, H, W, 3) uint8
    main = main.permute(0, 3, 1, 2).contiguous()              # (T, 3, H, W)
    extra = data["curr_obs"]["extra_view_images"][:, 0, 0]    # (T, H, W, 3)
    extra = extra.permute(0, 3, 1, 2).contiguous()            # (T, 3, H, W)
    return {
        "actions": actions, "states": states,
        "imgs_main": main, "imgs_extra": extra,
        "T": int(actions.shape[0]),
    }


def build_past_window(hand_seq: torch.Tensor, t: int, window_size: int = 8) -> torch.Tensor:
    """Past hand window ENDING at t-1 (upper bound is t, not t+1).

    Left-padded with hand_seq[0] when t < window_size. Matches the offset rule
    in bc_dataset.BCDataset — see its docstring for why this is critical.
    """
    start = t - window_size
    if start < 0:
        pad_len = -start
        return torch.cat(
            [hand_seq[0:1].expand(pad_len, -1), hand_seq[0:t]],
            dim=0,
        )
    return hand_seq[start:t]


# ─── Rollout (batched over stochastic samples) ────────────────────────────────


@torch.no_grad()
def rollout_n_samples(
    policy, traj, state_mean, state_std, mode, num_samples, device, window_size=8,
) -> np.ndarray:
    """Run BC over the full trajectory, batched over `num_samples` stochastic draws.

    For TF / no_corr the past_hand_window comes from GT (same across samples).
    For AR mode each sample has its own progressively-replaced hand history.

    Returns:
        runs: (num_samples, T, 12) numpy array of predicted actions
    """
    T = traj["T"]
    actions = traj["actions"].to(device)              # (T, 12) GT
    states = traj["states"].to(device)                # (T, 24)
    main_imgs = traj["imgs_main"].to(device)          # (T, 3, H, W) uint8
    extra_imgs = traj["imgs_extra"].to(device)        # (T, 3, H, W) uint8
    state_mean_d = state_mean.to(device)
    state_std_d = state_std.to(device)

    hand_gt = actions[:, 6:12]                        # (T, 6)
    # Per-sample hand histories. Cloning is essential — `expand` returns a view
    # and AR mode mutates in place.
    hand_seqs = hand_gt.unsqueeze(0).expand(num_samples, -1, -1).clone()  # (n, T, 6)

    zero_delta = (mode == "no_corr")
    replace_hand = (mode == "ar")

    runs = torch.zeros((num_samples, T, 12), device=device)

    for t in range(T):
        # Per-sample windows: (n, window_size, 6)
        windows = torch.stack(
            [build_past_window(hand_seqs[s], t, window_size) for s in range(num_samples)],
            dim=0,
        )

        # Replicate (deterministic) obs across samples; use repeat to ensure contiguity.
        img_main = (main_imgs[t].float() / 255.0).unsqueeze(0).repeat(num_samples, 1, 1, 1)
        img_extra = (extra_imgs[t].float() / 255.0).unsqueeze(0).repeat(num_samples, 1, 1, 1)
        state = ((states[t] - state_mean_d) / state_std_d).unsqueeze(0).repeat(num_samples, 1)

        out = policy(
            img_main=img_main,
            img_extra=img_extra,
            state=state,
            past_hand_win=windows,
            zero_delta=zero_delta,
        )
        runs[:, t, :6] = out["arm_action"]
        runs[:, t, 6:] = out["hand_action"]

        if replace_hand:
            hand_seqs[:, t, :] = out["hand_action"]

    return runs.cpu().numpy()


# ─── Per-trajectory eval ──────────────────────────────────────────────────────


def eval_trajectory(
    policy, traj_path, state_mean, state_std, num_samples, device,
    window_size=8, verbose=True,
) -> dict:
    """Run all 3 modes on one trajectory. Returns a dict of arrays + scalars."""
    traj = load_trajectory_full(traj_path)
    gt = traj["actions"].numpy()                                  # (T, 12)
    T = traj["T"]
    traj_id = os.path.basename(traj_path).split("trajectory_")[1].split("_")[0]

    result = {"traj_id": traj_id, "T": T, "gt": gt, "num_samples": num_samples}

    for mode in MODES:
        runs = rollout_n_samples(
            policy, traj, state_mean, state_std, mode, num_samples, device, window_size,
        )                                                         # (n, T, 12)
        mean = runs.mean(axis=0)                                  # (T, 12)
        std = runs.std(axis=0)                                    # (T, 12)
        # Per-step MSE: average over reparam samples and over the relevant joint dims
        per_step_arm = ((runs[:, :, :6] - gt[:, :6]) ** 2).mean(axis=2).mean(axis=0)
        per_step_hand = ((runs[:, :, 6:] - gt[:, 6:]) ** 2).mean(axis=2).mean(axis=0)

        result[f"{mode}_runs"] = runs
        result[f"{mode}_mean"] = mean
        result[f"{mode}_std"] = std
        result[f"{mode}_mse_arm"] = per_step_arm
        result[f"{mode}_mse_hand"] = per_step_hand
        result[f"{mode}_mse_arm_mean"] = float(per_step_arm.mean())
        result[f"{mode}_mse_hand_mean"] = float(per_step_hand.mean())

    if verbose:
        print(f"\nTrajectory {traj_id}: {T} steps")
        print(f"  {'mode':<28} | {'arm_mse':>10} | {'hand_mse':>10}")
        print(f"  {'-' * 28} | {'-' * 10} | {'-' * 10}")
        for mode in MODES:
            print(
                f"  {MODE_LABELS[mode]:<28} | "
                f"{result[f'{mode}_mse_arm_mean']:>10.6f} | "
                f"{result[f'{mode}_mse_hand_mean']:>10.6f}"
            )

    return result


# ─── Plotting ─────────────────────────────────────────────────────────────────


def plot_trajectory_actions(result, output_dir):
    """12 subplots (4 rows × 3 cols): GT vs each mode (mean + std band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = result["gt"]
    T = result["T"]
    traj_id = result["traj_id"]
    n = result["num_samples"]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    fig.suptitle(
        f"Trajectory {traj_id}: GT vs BC predictions  (T={T}, n_samples={n})",
        fontsize=14,
    )

    for i, name in enumerate(JOINT_NAMES):
        ax = axes.flat[i]
        ax.plot(range(T), gt[:, i], "k-", label="GT", linewidth=2.0, zorder=10)
        for mode in MODES:
            mean = result[f"{mode}_mean"][:, i]
            std = result[f"{mode}_std"][:, i]
            ax.plot(
                range(T), mean, color=MODE_COLORS[mode],
                label=MODE_LABELS[mode], linewidth=1.3,
            )
            if n > 1:
                ax.fill_between(
                    range(T), mean - std, mean + std,
                    color=MODE_COLORS[mode], alpha=0.15, linewidth=0,
                )
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Step")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"traj_{traj_id}_actions.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_trajectory_mse(result, output_dir):
    """Per-step MSE for arm and hand, all 3 modes side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = result["T"]
    traj_id = result["traj_id"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Trajectory {traj_id}: per-step MSE", fontsize=14)

    for ax, side, title in [
        (axes[0], "arm", "Arm (dims 0–5)"),
        (axes[1], "hand", "Hand (dims 6–11)"),
    ]:
        for mode in MODES:
            mse = result[f"{mode}_mse_{side}"]
            mean = float(mse.mean())
            ax.plot(
                range(T), mse, color=MODE_COLORS[mode],
                label=f"{MODE_LABELS[mode]} (mean={mean:.5f})",
                linewidth=1.5, alpha=0.85,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"traj_{traj_id}_mse.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_summary(all_results, output_dir):
    """Bar chart: mean arm/hand MSE per mode across all evaluated trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm = {m: float(np.mean([r[f"{m}_mse_arm_mean"] for r in all_results])) for m in MODES}
    hand = {m: float(np.mean([r[f"{m}_mse_hand_mean"] for r in all_results])) for m in MODES}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"BC eval summary across {len(all_results)} trajectories", fontsize=13)

    for ax, side, vals in [(axes[0], "arm", arm), (axes[1], "hand", hand)]:
        labels = [MODE_LABELS[m] for m in MODES]
        values = [vals[m] for m in MODES]
        colors = [MODE_COLORS[m] for m in MODES]
        bars = ax.bar(labels, values, color=colors)
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:.5f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylabel("Mean MSE")
        ax.set_title(f"{side} MSE")
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy over frozen VAE")
    parser.add_argument(
        "--ckpt", type=str,
        default=os.path.join(_PROJ_ROOT, "outputs/bc_simple_v1/checkpoint.pth"),
        help="BC checkpoint path",
    )
    parser.add_argument(
        "--vae_ckpt", type=str, default=None,
        help="VAE checkpoint (default: read from BC ckpt's args)",
    )
    parser.add_argument(
        "--test_dir", type=str,
        default=os.path.join(_PROJ_ROOT, "data/20260327-11:10:43/demos/success/test"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(_PROJ_ROOT, "visualizations/bc_eval/v1"),
    )
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument(
        "--traj_id", type=int, nargs="+", default=None,
        help="Specific trajectory IDs (default: first 3 in test_dir)",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all test trajectories")
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Stochastic reparameterize draws per mode per trajectory",
    )
    parser.add_argument("--no_plot", action="store_true", help="Skip plots (save raw .npz only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load BC checkpoint ──
    print(f"Loading BC checkpoint: {args.ckpt}")
    bc_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_mean = bc_ckpt["state_mean"]
    state_std = bc_ckpt["state_std"]
    bc_args = bc_ckpt.get("args", {})

    # ── Load VAE (frozen) ──
    vae_ckpt_path = (
        args.vae_ckpt
        or bc_args.get("vae_ckpt")
        or os.path.join(_PROJ_ROOT, "outputs/dim_2_best/checkpoint.pth")
    )
    vae = build_and_freeze_vae(vae_ckpt_path)

    # ── Build BC policy and load weights ──
    policy = BCPolicy(
        vae=vae,
        state_dim=24,
        feat_dim=bc_args.get("feat_dim", 128),
        fusion_dim=bc_args.get("fusion_dim", 256),
    ).to(device)
    missing, unexpected = policy.load_state_dict(bc_ckpt["model"], strict=False)
    bc_only_missing = [k for k in missing if not k.startswith("vae.")]
    if bc_only_missing:
        print(f"WARNING: missing BC keys at load: {bc_only_missing}")
    if unexpected:
        print(f"WARNING: unexpected keys at load: {unexpected}")
    policy.eval()
    print(f"Loaded BC weights at step {bc_ckpt.get('step', '?')}")

    # ── Select trajectories ──
    all_files = sorted(glob.glob(os.path.join(args.test_dir, "trajectory_*_demo_expert.pt")))
    if args.traj_id:
        files = []
        for tid in args.traj_id:
            matches = [f for f in all_files if f"trajectory_{tid}_" in f]
            if not matches:
                print(f"  WARNING: trajectory {tid} not found in {args.test_dir}")
            files.extend(matches)
    elif args.all:
        files = all_files
    else:
        files = all_files[:3]

    if not files:
        print("No trajectory files found!")
        return
    print(f"Evaluating {len(files)} trajectories with {args.num_samples} samples per mode")

    # ── Per-trajectory eval ──
    all_results = []
    for f in files:
        result = eval_trajectory(
            policy, f, state_mean, state_std,
            args.num_samples, device, args.window_size,
            verbose=True,
        )
        all_results.append(result)

        # Save raw .npz (skip non-array scalars)
        npz_path = os.path.join(args.output_dir, f"traj_{result['traj_id']}_eval.npz")
        np_save = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        np.savez(npz_path, **np_save)

        if not args.no_plot:
            plot_trajectory_actions(result, args.output_dir)
            plot_trajectory_mse(result, args.output_dir)

    # ── Aggregate summary ──
    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(all_results)} trajectories, {args.num_samples} samples each)")
    print("=" * 70)
    print(f"  {'mode':<28} | {'arm_mse':>10} | {'hand_mse':>10}")
    print(f"  {'-' * 28} | {'-' * 10} | {'-' * 10}")

    summary = {
        "ckpt": args.ckpt,
        "test_dir": args.test_dir,
        "num_trajectories": len(all_results),
        "num_samples_per_mode": args.num_samples,
        "modes": {},
        "per_trajectory": [
            {
                "traj_id": r["traj_id"],
                "T": r["T"],
                **{f"{m}_arm": r[f"{m}_mse_arm_mean"] for m in MODES},
                **{f"{m}_hand": r[f"{m}_mse_hand_mean"] for m in MODES},
            }
            for r in all_results
        ],
    }
    for mode in MODES:
        arm = float(np.mean([r[f"{mode}_mse_arm_mean"] for r in all_results]))
        hand = float(np.mean([r[f"{mode}_mse_hand_mean"] for r in all_results]))
        print(f"  {MODE_LABELS[mode]:<28} | {arm:>10.6f} | {hand:>10.6f}")
        summary["modes"][mode] = {"arm_mse": arm, "hand_mse": hand}

    vg_arm = summary["modes"]["no_corr"]["arm_mse"] - summary["modes"]["tf"]["arm_mse"]
    vg_hand = summary["modes"]["no_corr"]["hand_mse"] - summary["modes"]["tf"]["hand_mse"]
    ar_drift_arm = summary["modes"]["ar"]["arm_mse"] - summary["modes"]["tf"]["arm_mse"]
    ar_drift_hand = summary["modes"]["ar"]["hand_mse"] - summary["modes"]["tf"]["hand_mse"]
    summary["vision_gain_arm"] = vg_arm
    summary["vision_gain_hand"] = vg_hand
    summary["ar_drift_arm"] = ar_drift_arm
    summary["ar_drift_hand"] = ar_drift_hand
    print(f"\n  Vision gain (no_corr - tf):  arm={vg_arm:+.6f}  hand={vg_hand:+.6f}")
    print(f"  AR drift (ar - tf):          arm={ar_drift_arm:+.6f}  hand={ar_drift_hand:+.6f}")
    print(
        f"  → Vision-corrected hand "
        f"{'BEATS' if vg_hand > 0 else 'DOES NOT beat'} the no-correction baseline."
    )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON saved: {summary_path}")

    if not args.no_plot:
        plot_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
