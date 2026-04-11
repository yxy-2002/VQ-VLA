"""
Evaluate Behavior Cloning policy over a frozen Hand-Action VAE.

Compared with the original evaluator, this version adds offline diagnostics for
stage-2 failure analysis:

- image counterfactuals (`--image_mode`): normal / zero / stale / shuffle / swap
- partial feedback (`--feedback_horizon`): only feed back the first K predicted
  actions in AR mode, then fall back to GT
- onset slices: report metrics on pre-onset / onset-band / post-onset segments
- latent debug dumps (`--save_debug_latent`): save prior/controlled latent stats

The evaluator is still offline: even in AR mode it consumes dataset images at
all steps. Only the action state / hand-history feedback is altered.
The current BC 3.0 policy only consumes the first 6 state dims for the arm
branch; the hand branch conditions on the VAE prior and optionally the encoded
arm-state latent.
"""

import argparse
import glob
import importlib.util
import json
import os
from typing import Dict, List, Optional

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
_dataset_mod = _load(os.path.join(_BC_ROOT, "model/bc_dataset.py"), "bc_dataset")
BCPolicy = _policy_mod.BCPolicy
build_and_freeze_vae = _policy_mod.build_and_freeze_vae
apply_state_mask = _dataset_mod.apply_state_mask


# ─── Constants ────────────────────────────────────────────────────────────────

ARM_NAMES = [f"arm_{i}" for i in range(6)]
HAND_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
JOINT_NAMES = ARM_NAMES + HAND_NAMES

MODES = ["tf", "ar", "no_corr"]
MODE_LABELS = {
    "tf": "Teacher-Forced",
    "ar": "AR-hand",
    "no_corr": "No Correction (VAE prior)",
}
MODE_COLORS = {
    "tf": "tab:green",
    "ar": "tab:red",
    "no_corr": "tab:blue",
}
IMAGE_MODES = ["normal", "zero", "stale", "shuffle", "swap"]
SLICE_NAMES = ["pre_onset", "onset_band", "post_onset"]


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_trajectory_full(path: str) -> dict:
    """Load actions + images from a trajectory .pt file (matches BCDataset)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    actions = data["actions"][:, 0, :].float()
    main = data["curr_obs"]["main_images"][:, 0].permute(0, 3, 1, 2).contiguous()
    extra = data["curr_obs"]["extra_view_images"][:, 0, 0].permute(0, 3, 1, 2).contiguous()
    traj_id = os.path.basename(path).split("trajectory_")[1].split("_")[0]
    return {
        "path": path,
        "traj_id": traj_id,
        "actions": actions,
        "imgs_main": main,
        "imgs_extra": extra,
        "T": int(actions.shape[0]),
    }


def build_past_window(hand_seq: torch.Tensor, t: int, window_size: int = 8) -> torch.Tensor:
    """Past hand window [a_{t-7}..a_t] inclusive of current frame."""
    start = t - window_size + 1
    if start < 0:
        pad_len = -start
        return torch.cat([hand_seq[0:1].expand(pad_len, -1), hand_seq[0:t + 1]], dim=0)
    return hand_seq[start:t + 1]


# ─── Onset diagnostics ────────────────────────────────────────────────────────


def compute_hand_delta_norm(actions: np.ndarray) -> np.ndarray:
    """Per-step hand delta norm aligned to decision step t (predicting t+1)."""
    if actions.shape[0] <= 1:
        return np.zeros((actions.shape[0],), dtype=np.float32)
    delta = actions[1:, 6:] - actions[:-1, 6:]
    norm = np.linalg.norm(delta, axis=1).astype(np.float32)
    return np.concatenate([norm, np.zeros((1,), dtype=np.float32)], axis=0)


def detect_grasp_onset(
    actions: np.ndarray,
    threshold: float = 0.02,
    lookahead: int = 3,
    min_count: int = 2,
) -> Optional[int]:
    """Detect the first decision step where hand motion persistently increases."""
    delta_norm = compute_hand_delta_norm(actions)
    if delta_norm.shape[0] == 0:
        return None
    last_valid = max(0, actions.shape[0] - 1)
    for t in range(last_valid):
        window = delta_norm[t:min(last_valid, t + lookahead)]
        if int((window > threshold).sum()) >= min_count:
            return int(t)
    return None


def build_slice_masks(
    T: int,
    onset_step: Optional[int],
    onset_pre: int = 2,
    onset_post: int = 4,
) -> Dict[str, np.ndarray]:
    masks = {name: np.zeros((T,), dtype=bool) for name in SLICE_NAMES}
    if onset_step is None:
        masks["pre_onset"][:] = True
        return masks

    band_start = max(0, onset_step - onset_pre)
    band_end = min(T, onset_step + onset_post + 1)
    masks["pre_onset"][:band_start] = True
    masks["onset_band"][band_start:band_end] = True
    masks["post_onset"][band_end:] = True
    return masks


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if not mask.any():
        return None
    return float(arr[mask].mean())


# ─── Image / feedback plans ───────────────────────────────────────────────────


def parse_feedback_horizon(value: str) -> Optional[int]:
    if value == "full":
        return None
    horizon = int(value)
    if horizon < 0:
        raise ValueError("feedback_horizon must be >= 0 or 'full'")
    return horizon


def build_image_plan(
    traj: dict,
    image_mode: str,
    seed: int,
    swap_traj: Optional[dict] = None,
) -> dict:
    T = traj["T"]
    if image_mode == "normal":
        return {"source": "self", "indices": np.arange(T, dtype=np.int64)}
    if image_mode == "stale":
        return {"source": "self", "indices": np.zeros((T,), dtype=np.int64)}
    if image_mode == "shuffle":
        rng = np.random.default_rng(seed + int(traj["traj_id"]))
        return {"source": "self", "indices": rng.permutation(T).astype(np.int64)}
    if image_mode == "swap":
        if swap_traj is None:
            raise ValueError("image_mode='swap' requires a donor trajectory")
        donor_idx = np.minimum(np.arange(T, dtype=np.int64), swap_traj["T"] - 1)
        return {"source": swap_traj["traj_id"], "indices": donor_idx}
    if image_mode == "zero":
        return {"source": "zero", "indices": np.full((T,), -1, dtype=np.int64)}
    raise ValueError(f"Unsupported image_mode={image_mode!r}")


# ─── Rollout (batched over stochastic samples) ────────────────────────────────


@torch.no_grad()
def rollout_n_samples(
    policy: BCPolicy,
    traj: dict,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    mode: str,
    num_samples: int,
    device: torch.device,
    window_size: int = 8,
    image_mode: str = "normal",
    feedback_horizon: Optional[int] = None,
    state_mask: str = "all",
    save_debug_latent: bool = False,
    swap_traj: Optional[dict] = None,
    seed: int = 42,
) -> dict:
    """Run BC over the full trajectory, batched over stochastic samples."""
    T = traj["T"]
    actions = traj["actions"].to(device)
    main_imgs = traj["imgs_main"].to(device)
    extra_imgs = traj["imgs_extra"].to(device)
    swap_main = swap_extra = None
    if swap_traj is not None:
        swap_main = swap_traj["imgs_main"].to(device)
        swap_extra = swap_traj["imgs_extra"].to(device)

    action_mean_d = action_mean.to(device)
    action_std_d = action_std.to(device)
    image_plan = build_image_plan(traj, image_mode, seed, swap_traj=swap_traj)

    action_seqs = actions.unsqueeze(0).expand(num_samples, -1, -1).clone()
    zero_delta = (mode == "no_corr")
    replace = (mode == "ar")

    runs = torch.zeros((num_samples, T, 12), device=device)
    debug = None
    if save_debug_latent:
        latent_dim = int(policy.latent_dim)
        debug = {
            "delta_z": torch.zeros((num_samples, T, latent_dim), device=device),
            "mu_prior": torch.zeros((num_samples, T, latent_dim), device=device),
            "log_var_prior": torch.zeros((num_samples, T, latent_dim), device=device),
            "z_ctrl": torch.zeros((num_samples, T, latent_dim), device=device),
            "z_no_corr": torch.zeros((num_samples, T, latent_dim), device=device),
            "hand_action": torch.zeros((num_samples, T, 6), device=device),
            "hand_no_corr": torch.zeros((num_samples, T, 6), device=device),
        }

    for t in range(T):
        state = (action_seqs[:, t, :] - action_mean_d) / action_std_d
        state = apply_state_mask(state, state_mask)

        windows = torch.stack(
            [build_past_window(action_seqs[s, :, 6:12], t, window_size) for s in range(num_samples)],
            dim=0,
        )

        if image_mode == "zero":
            shape = (num_samples,) + tuple(main_imgs.shape[1:])
            img_main = torch.zeros(shape, device=device, dtype=torch.float32)
            img_extra = torch.zeros(shape, device=device, dtype=torch.float32)
        else:
            img_idx = int(image_plan["indices"][t])
            if image_mode == "swap":
                src_main = swap_main
                src_extra = swap_extra
            else:
                src_main = main_imgs
                src_extra = extra_imgs
            img_main = (src_main[img_idx].float() / 255.0).unsqueeze(0).repeat(num_samples, 1, 1, 1)
            img_extra = (src_extra[img_idx].float() / 255.0).unsqueeze(0).repeat(num_samples, 1, 1, 1)

        out = policy(
            img_main=img_main,
            img_extra=img_extra,
            state=state,
            past_hand_win=windows,
            zero_delta=zero_delta,
        )
        runs[:, t, :6] = out["arm_action"]
        runs[:, t, 6:] = out["hand_action"]

        if debug is not None:
            for key in [
                "delta_z", "mu_prior", "log_var_prior", "z_ctrl", "z_no_corr",
            ]:
                debug[key][:, t, :] = out[key]
            debug["hand_action"][:, t, :] = out["hand_action"]
            debug["hand_no_corr"][:, t, :] = out["hand_no_corr"]

        if replace and t + 1 < T and (feedback_horizon is None or t < feedback_horizon):
            action_seqs[:, t + 1, :6] = out["arm_action"]
            action_seqs[:, t + 1, 6:] = out["hand_action"]

    payload = {
        "runs": runs.cpu().numpy(),
        "image_indices": image_plan["indices"],
        "image_source": image_plan["source"],
    }
    if debug is not None:
        payload["debug"] = {k: v.cpu().numpy() for k, v in debug.items()}
    return payload


# ─── Per-trajectory eval ──────────────────────────────────────────────────────


def eval_trajectory(
    policy: BCPolicy,
    traj: dict,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    num_samples: int,
    device: torch.device,
    window_size: int = 8,
    verbose: bool = True,
    image_mode: str = "normal",
    feedback_horizon: Optional[int] = None,
    state_mask: str = "all",
    save_debug_latent: bool = False,
    swap_traj: Optional[dict] = None,
    seed: int = 42,
    onset_threshold: float = 0.02,
    onset_lookahead: int = 3,
    onset_min_count: int = 2,
) -> dict:
    gt = traj["actions"].numpy()
    T = traj["T"]
    traj_id = traj["traj_id"]

    gt_target = np.concatenate([gt[1:], gt[-1:]], axis=0)
    copy_arm = ((gt[:, :6] - gt_target[:, :6]) ** 2).mean(axis=1)
    copy_hand = ((gt[:, 6:] - gt_target[:, 6:]) ** 2).mean(axis=1)

    onset_step = detect_grasp_onset(
        gt,
        threshold=onset_threshold,
        lookahead=onset_lookahead,
        min_count=onset_min_count,
    )
    hand_delta_norm = compute_hand_delta_norm(gt)
    slice_masks = build_slice_masks(T, onset_step)

    result = {
        "traj_id": traj_id,
        "T": T,
        "gt": gt,
        "gt_target": gt_target,
        "num_samples": num_samples,
        "copy_arm_mse": copy_arm,
        "copy_hand_mse": copy_hand,
        "copy_arm_mean": float(copy_arm.mean()),
        "copy_hand_mean": float(copy_hand.mean()),
        "hand_delta_norm": hand_delta_norm,
        "onset_step": int(onset_step) if onset_step is not None else None,
        "onset_found": onset_step is not None,
        "image_mode": image_mode,
        "feedback_horizon": "full" if feedback_horizon is None else int(feedback_horizon),
        "state_mask": state_mask,
        "pre_onset_mask": slice_masks["pre_onset"].astype(np.uint8),
        "onset_band_mask": slice_masks["onset_band"].astype(np.uint8),
        "post_onset_mask": slice_masks["post_onset"].astype(np.uint8),
        "slices": {},
    }

    for slice_name, mask in slice_masks.items():
        result["slices"][slice_name] = {
            "n_steps": int(mask.sum()),
            "copy_arm_mse": masked_mean(copy_arm, mask),
            "copy_hand_mse": masked_mean(copy_hand, mask),
            "modes": {},
        }

    for mode in MODES:
        rollout = rollout_n_samples(
            policy=policy,
            traj=traj,
            action_mean=action_mean,
            action_std=action_std,
            mode=mode,
            num_samples=num_samples,
            device=device,
            window_size=window_size,
            image_mode=image_mode,
            feedback_horizon=feedback_horizon,
            state_mask=state_mask,
            save_debug_latent=save_debug_latent,
            swap_traj=swap_traj,
            seed=seed,
        )
        runs = rollout["runs"]
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)
        per_step_arm = ((runs[:, :, :6] - gt_target[:, :6]) ** 2).mean(axis=2).mean(axis=0)
        per_step_hand = ((runs[:, :, 6:] - gt_target[:, 6:]) ** 2).mean(axis=2).mean(axis=0)

        result[f"{mode}_runs"] = runs
        result[f"{mode}_mean"] = mean
        result[f"{mode}_std"] = std
        result[f"{mode}_mse_arm"] = per_step_arm
        result[f"{mode}_mse_hand"] = per_step_hand
        result[f"{mode}_mse_arm_mean"] = float(per_step_arm.mean())
        result[f"{mode}_mse_hand_mean"] = float(per_step_hand.mean())
        result[f"{mode}_image_indices"] = rollout["image_indices"]
        result[f"{mode}_image_source"] = rollout["image_source"]

        if save_debug_latent:
            for key, value in rollout["debug"].items():
                result[f"{mode}_{key}"] = value

        for slice_name, mask in slice_masks.items():
            result["slices"][slice_name]["modes"][mode] = {
                "arm_mse": masked_mean(per_step_arm, mask),
                "hand_mse": masked_mean(per_step_hand, mask),
            }

    if verbose:
        print(f"\nTrajectory {traj_id}: {T} steps  onset={result['onset_step']}")
        print(f"  image_mode={image_mode}  feedback_horizon={result['feedback_horizon']}  state_mask={state_mask}")
        print(f"  {'mode':<28} | {'arm_mse':>10} | {'hand_mse':>10}")
        print(f"  {'-' * 28} | {'-' * 10} | {'-' * 10}")
        print(
            f"  {'Copy baseline (a[t+1]=a[t])':<28} | "
            f"{result['copy_arm_mean']:>10.6f} | {result['copy_hand_mean']:>10.6f}"
        )
        for mode in MODES:
            print(
                f"  {MODE_LABELS[mode]:<28} | "
                f"{result[f'{mode}_mse_arm_mean']:>10.6f} | "
                f"{result[f'{mode}_mse_hand_mean']:>10.6f}"
            )
        for slice_name in SLICE_NAMES:
            n_steps = result["slices"][slice_name]["n_steps"]
            tf_hand = result["slices"][slice_name]["modes"]["tf"]["hand_mse"]
            no_corr_hand = result["slices"][slice_name]["modes"]["no_corr"]["hand_mse"]
            print(
                f"  slice={slice_name:<10} n={n_steps:<3d}  "
                f"tf_hand={tf_hand if tf_hand is not None else float('nan'):.6f}  "
                f"no_corr_hand={no_corr_hand if no_corr_hand is not None else float('nan'):.6f}"
            )

    return result


# ─── Summary aggregation ──────────────────────────────────────────────────────


def summarize_results(
    all_results: List[dict],
    ckpt_path: str,
    test_dir: str,
    image_mode: str,
    feedback_horizon: Optional[int],
    state_mask: str,
    num_samples: int,
) -> dict:
    summary = {
        "ckpt": ckpt_path,
        "test_dir": test_dir,
        "num_trajectories": len(all_results),
        "num_samples_per_mode": num_samples,
        "image_mode": image_mode,
        "feedback_horizon": "full" if feedback_horizon is None else int(feedback_horizon),
        "state_mask": state_mask,
        "modes": {},
        "copy_baseline": {},
        "slices": {},
        "per_trajectory": [],
    }

    copy_arm_agg = float(np.mean([r["copy_arm_mean"] for r in all_results]))
    copy_hand_agg = float(np.mean([r["copy_hand_mean"] for r in all_results]))
    summary["copy_baseline"] = {"arm_mse": copy_arm_agg, "hand_mse": copy_hand_agg}

    for mode in MODES:
        arm = float(np.mean([r[f"{mode}_mse_arm_mean"] for r in all_results]))
        hand = float(np.mean([r[f"{mode}_mse_hand_mean"] for r in all_results]))
        summary["modes"][mode] = {"arm_mse": arm, "hand_mse": hand}

    onset_steps = [r["onset_step"] for r in all_results if r["onset_step"] is not None]
    summary["onset_stats"] = {
        "found": len(onset_steps),
        "missing": len(all_results) - len(onset_steps),
        "mean_step": float(np.mean(onset_steps)) if onset_steps else None,
        "median_step": float(np.median(onset_steps)) if onset_steps else None,
    }

    for slice_name in SLICE_NAMES:
        count = int(sum(r["slices"][slice_name]["n_steps"] for r in all_results))
        slice_entry = {
            "num_steps": count,
            "copy_baseline": {"arm_mse": None, "hand_mse": None},
            "modes": {},
            "vision_gain_arm": None,
            "vision_gain_hand": None,
        }
        if count > 0:
            copy_arm = sum(
                r["slices"][slice_name]["copy_arm_mse"] * r["slices"][slice_name]["n_steps"]
                for r in all_results
                if r["slices"][slice_name]["copy_arm_mse"] is not None
            ) / count
            copy_hand = sum(
                r["slices"][slice_name]["copy_hand_mse"] * r["slices"][slice_name]["n_steps"]
                for r in all_results
                if r["slices"][slice_name]["copy_hand_mse"] is not None
            ) / count
            slice_entry["copy_baseline"] = {
                "arm_mse": float(copy_arm),
                "hand_mse": float(copy_hand),
            }
            for mode in MODES:
                arm = sum(
                    r["slices"][slice_name]["modes"][mode]["arm_mse"] * r["slices"][slice_name]["n_steps"]
                    for r in all_results
                    if r["slices"][slice_name]["modes"][mode]["arm_mse"] is not None
                ) / count
                hand = sum(
                    r["slices"][slice_name]["modes"][mode]["hand_mse"] * r["slices"][slice_name]["n_steps"]
                    for r in all_results
                    if r["slices"][slice_name]["modes"][mode]["hand_mse"] is not None
                ) / count
                slice_entry["modes"][mode] = {"arm_mse": float(arm), "hand_mse": float(hand)}
            slice_entry["vision_gain_arm"] = (
                slice_entry["modes"]["no_corr"]["arm_mse"] - slice_entry["modes"]["tf"]["arm_mse"]
            )
            slice_entry["vision_gain_hand"] = (
                slice_entry["modes"]["no_corr"]["hand_mse"] - slice_entry["modes"]["tf"]["hand_mse"]
            )
        summary["slices"][slice_name] = slice_entry

    vg_arm = summary["modes"]["no_corr"]["arm_mse"] - summary["modes"]["tf"]["arm_mse"]
    vg_hand = summary["modes"]["no_corr"]["hand_mse"] - summary["modes"]["tf"]["hand_mse"]
    ar_drift_arm = summary["modes"]["ar"]["arm_mse"] - summary["modes"]["tf"]["arm_mse"]
    ar_drift_hand = summary["modes"]["ar"]["hand_mse"] - summary["modes"]["tf"]["hand_mse"]
    summary["vision_gain_arm"] = vg_arm
    summary["vision_gain_hand"] = vg_hand
    summary["ar_drift_arm"] = ar_drift_arm
    summary["ar_drift_hand"] = ar_drift_hand

    for r in all_results:
        summary["per_trajectory"].append({
            "traj_id": r["traj_id"],
            "T": r["T"],
            "onset_step": r["onset_step"],
            "copy_arm": r["copy_arm_mean"],
            "copy_hand": r["copy_hand_mean"],
            **{f"{m}_arm": r[f"{m}_mse_arm_mean"] for m in MODES},
            **{f"{m}_hand": r[f"{m}_mse_hand_mean"] for m in MODES},
            "slices": r["slices"],
        })

    return summary


# ─── Plotting ─────────────────────────────────────────────────────────────────


def plot_trajectory_actions(result, output_dir):
    """12 subplots (4 rows × 3 cols): GT vs each mode (mean + std band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_target = result["gt_target"]
    T = result["T"]
    traj_id = result["traj_id"]
    n = result["num_samples"]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    fig.suptitle(
        f"Trajectory {traj_id}: GT next-pose vs BC predictions  (T={T}, n_samples={n})",
        fontsize=14,
    )

    for i, name in enumerate(JOINT_NAMES):
        ax = axes.flat[i]
        ax.plot(range(T), gt_target[:, i], "k-", label="GT next pose", linewidth=2.0, zorder=10)
        for mode in MODES:
            mean = result[f"{mode}_mean"][:, i]
            std = result[f"{mode}_std"][:, i]
            ax.plot(range(T), mean, color=MODE_COLORS[mode], label=MODE_LABELS[mode], linewidth=1.3)
            if n > 1:
                ax.fill_between(range(T), mean - std, mean + std, color=MODE_COLORS[mode], alpha=0.15, linewidth=0)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Decision step t  (BC predicts action[t+1])")

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
        copy_mse = result[f"copy_{side}_mse"]
        ax.plot(
            range(T), copy_mse, "k--",
            label=f"Copy baseline (mean={float(copy_mse.mean()):.5f})",
            linewidth=1.0, alpha=0.6,
        )
        onset_step = result.get("onset_step")
        if onset_step is not None:
            ax.axvline(onset_step, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Decision step t")
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
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
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


# ─── Policy loading / selection helpers ───────────────────────────────────────


def load_policy_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    vae_ckpt_override: Optional[str] = None,
):
    print(f"Loading BC checkpoint: {ckpt_path}")
    bc_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    action_mean = bc_ckpt["action_mean"]
    action_std = bc_ckpt["action_std"]
    bc_args = bc_ckpt.get("args", {})

    vae_ckpt_path = (
        vae_ckpt_override
        or bc_args.get("vae_ckpt")
        or os.path.join(_PROJ_ROOT, "outputs/dim_2_best/checkpoint.pth")
    )
    vae = build_and_freeze_vae(vae_ckpt_path)

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
        legacy_hint = any(
            key.startswith("delta_mu_head") or key.startswith("delta_log_var_head")
            for key in unexpected
        )
        msg = (
            "BC checkpoint is incompatible with the current BC 3.0 weakly-coupled policy. "
            f"missing={bc_only_missing}, unexpected={unexpected}."
        )
        if legacy_hint:
            msg += " This looks like an older delta_mu/delta_log_var checkpoint; retrain BC with the new architecture before evaluation."
        raise RuntimeError(msg)
    policy.eval()
    print(f"Loaded BC weights at step {bc_ckpt.get('step', '?')}")
    return policy, action_mean, action_std, bc_args, bc_ckpt


def select_trajectory_files(test_dir: str, traj_ids: Optional[List[int]], use_all: bool) -> List[str]:
    all_files = sorted(glob.glob(os.path.join(test_dir, "trajectory_*_demo_expert.pt")))
    if traj_ids:
        files = []
        for tid in traj_ids:
            matches = [f for f in all_files if f"trajectory_{tid}_" in f]
            if not matches:
                print(f"  WARNING: trajectory {tid} not found in {test_dir}")
            files.extend(matches)
        return files
    if use_all:
        return all_files
    return all_files[:3]


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy over frozen VAE")
    parser.add_argument(
        "--ckpt", type=str,
        default=os.path.join(_PROJ_ROOT, "outputs/bc_simple_v2/checkpoint.pth"),
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
        default=os.path.join(_PROJ_ROOT, "visualizations/bc_eval/v2"),
    )
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument(
        "--traj_id", type=int, nargs="+", default=None,
        help="Specific trajectory IDs (default: first 3 in test_dir)",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all test trajectories")
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Repeated rollout copies per mode; current hand path is deterministic",
    )
    parser.add_argument("--image_mode", type=str, default="normal", choices=IMAGE_MODES)
    parser.add_argument(
        "--feedback_horizon", type=str, default="full",
        help="AR feedback horizon: full or an integer K. Only the first K predictions are fed back.",
    )
    parser.add_argument("--save_debug_latent", action="store_true")
    parser.add_argument(
        "--state_mask", type=str, default=None,
        help="Override the state-mask ablation. Default: read from checkpoint args.",
    )
    parser.add_argument("--onset_threshold", type=float, default=0.02)
    parser.add_argument("--onset_lookahead", type=int, default=3)
    parser.add_argument("--onset_min_count", type=int, default=2)
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

    policy, action_mean, action_std, bc_args, _ = load_policy_from_checkpoint(
        args.ckpt, device=device, vae_ckpt_override=args.vae_ckpt,
    )
    state_mask = args.state_mask or bc_args.get("state_mask", "all")
    if state_mask not in ["all", "arm_only", "none"]:
        raise ValueError(f"Unsupported state_mask={state_mask!r}")
    feedback_horizon = parse_feedback_horizon(args.feedback_horizon)

    files = select_trajectory_files(args.test_dir, args.traj_id, args.all)
    if not files:
        print("No trajectory files found!")
        return

    loaded = [load_trajectory_full(f) for f in files]
    if args.image_mode == "swap" and len(loaded) < 2:
        raise ValueError("image_mode='swap' requires at least two trajectories")

    print(
        f"Evaluating {len(files)} trajectories with {args.num_samples} samples per mode  "
        f"(image_mode={args.image_mode}, feedback_horizon={args.feedback_horizon}, state_mask={state_mask})"
    )

    all_results = []
    for idx, traj in enumerate(loaded):
        swap_traj = None
        if args.image_mode == "swap":
            swap_traj = loaded[(idx + 1) % len(loaded)]
        result = eval_trajectory(
            policy=policy,
            traj=traj,
            action_mean=action_mean,
            action_std=action_std,
            num_samples=args.num_samples,
            device=device,
            window_size=args.window_size,
            verbose=True,
            image_mode=args.image_mode,
            feedback_horizon=feedback_horizon,
            state_mask=state_mask,
            save_debug_latent=args.save_debug_latent,
            swap_traj=swap_traj,
            seed=args.seed,
            onset_threshold=args.onset_threshold,
            onset_lookahead=args.onset_lookahead,
            onset_min_count=args.onset_min_count,
        )
        all_results.append(result)

        npz_path = os.path.join(args.output_dir, f"traj_{result['traj_id']}_eval.npz")
        np_save = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        np.savez(npz_path, **np_save)

        if not args.no_plot:
            plot_trajectory_actions(result, args.output_dir)
            plot_trajectory_mse(result, args.output_dir)

    summary = summarize_results(
        all_results=all_results,
        ckpt_path=args.ckpt,
        test_dir=args.test_dir,
        image_mode=args.image_mode,
        feedback_horizon=feedback_horizon,
        state_mask=state_mask,
        num_samples=args.num_samples,
    )

    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(all_results)} trajectories, {args.num_samples} samples each)")
    print("=" * 70)
    print(f"  {'mode':<28} | {'arm_mse':>10} | {'hand_mse':>10}")
    print(f"  {'-' * 28} | {'-' * 10} | {'-' * 10}")
    print(
        f"  {'Copy baseline (a[t+1]=a[t])':<28} | "
        f"{summary['copy_baseline']['arm_mse']:>10.6f} | {summary['copy_baseline']['hand_mse']:>10.6f}"
    )
    for mode in MODES:
        print(
            f"  {MODE_LABELS[mode]:<28} | "
            f"{summary['modes'][mode]['arm_mse']:>10.6f} | {summary['modes'][mode]['hand_mse']:>10.6f}"
        )

    print(
        f"\n  Vision gain (no_corr - tf):  arm={summary['vision_gain_arm']:+.6f}  "
        f"hand={summary['vision_gain_hand']:+.6f}"
    )
    print(
        f"  AR drift (ar - tf):          arm={summary['ar_drift_arm']:+.6f}  "
        f"hand={summary['ar_drift_hand']:+.6f}"
    )
    print(
        f"  Onset detections: found={summary['onset_stats']['found']}  "
        f"missing={summary['onset_stats']['missing']}  "
        f"mean_step={summary['onset_stats']['mean_step']}"
    )
    for slice_name in SLICE_NAMES:
        s = summary['slices'][slice_name]
        if s['num_steps'] == 0:
            continue
        print(
            f"  slice={slice_name:<10} steps={s['num_steps']:<4d}  "
            f"tf_hand={s['modes']['tf']['hand_mse']:.6f}  "
            f"no_corr_hand={s['modes']['no_corr']['hand_mse']:.6f}  "
            f"vision_gain_hand={s['vision_gain_hand']:+.6f}"
        )

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON saved: {summary_path}")

    if not args.no_plot:
        plot_summary(all_results, args.output_dir)


if __name__ == '__main__':
    main()
