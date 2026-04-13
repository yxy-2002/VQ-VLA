"""
Quick AR comparison across multiple BC checkpoints.

Runs AR rollout on all test trajectories for each checkpoint and reports
arm_mse, hand_mse, hand_no_corr, vision_gain. Does NOT generate per-traj plots.
"""

import glob
import importlib.util
import json
import os

import numpy as np
import torch

_PROJ_ROOT = "/home/yxy/VQ-VLA"
_BC_ROOT = os.path.join(_PROJ_ROOT, "imitation_learning/behavior_clone")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


policy_mod = _load(os.path.join(_BC_ROOT, "model/bc_policy.py"), "bc_policy")
eval_mod = _load(os.path.join(_BC_ROOT, "scripts/eval.py"), "bc_eval")
BCPolicy = policy_mod.BCPolicy
build_and_freeze_vae = policy_mod.build_and_freeze_vae
load_trajectory = eval_mod.load_trajectory
rollout_ar = eval_mod.rollout_ar


def evaluate_ckpt(ckpt_path, test_files, device, num_samples=1, rollout_stride=1):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    action_mean = ckpt["action_mean"]
    action_std = ckpt["action_std"]
    bc_args = ckpt.get("args", {})
    chunk_mode = bc_args.get("chunk_mode", False)
    future_horizon = bc_args.get("future_horizon", 8) if chunk_mode else 1

    vae_path = bc_args.get("vae_ckpt") or os.path.join(_PROJ_ROOT, "outputs/dim_2_best/checkpoint.pth")
    vae, _ = build_and_freeze_vae(vae_path)
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
    policy.load_state_dict(ckpt["model"], strict=False)
    policy.eval()

    ar_arm_mses, ar_hand_mses = [], []
    nc_arm_mses, nc_hand_mses = [], []

    for f in test_files:
        traj = load_trajectory(f)
        T = traj["T"]
        gt = traj["actions"].numpy()
        gt_target = np.concatenate([gt[1:], gt[-1:]], axis=0)

        rollout = rollout_ar(
            policy=policy, traj=traj,
            action_mean=action_mean, action_std=action_std,
            num_samples=num_samples, device=device,
            rollout_stride=rollout_stride,
        )
        ar_runs = rollout["ar_runs"]  # (S, T, 12)
        no_corr = rollout["no_corr"]  # (T, 12)

        ar_mean = ar_runs.mean(axis=0)
        ar_arm_mses.append(((ar_mean[:, :6] - gt_target[:, :6]) ** 2).mean())
        ar_hand_mses.append(((ar_mean[:, 6:] - gt_target[:, 6:]) ** 2).mean())
        nc_arm_mses.append(((no_corr[:, :6] - gt_target[:, :6]) ** 2).mean())
        nc_hand_mses.append(((no_corr[:, 6:] - gt_target[:, 6:]) ** 2).mean())

    return {
        "ar_arm_mse": float(np.mean(ar_arm_mses)),
        "ar_hand_mse": float(np.mean(ar_hand_mses)),
        "nc_arm_mse": float(np.mean(nc_arm_mses)),
        "nc_hand_mse": float(np.mean(nc_hand_mses)),
        "vision_gain": float(np.mean(nc_hand_mses) - np.mean(ar_hand_mses)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = os.path.join(_PROJ_ROOT, "data/20260327-11:10:43/demos/success/test")
    test_files = sorted(glob.glob(os.path.join(test_dir, "trajectory_*_demo_expert.pt")))
    print(f"Found {len(test_files)} test trajectories")

    ckpts = [
        # Architecture sweep
        ("A1_h128", "outputs/bc_chunk_sweep/A1_h128/checkpoint.pth"),
        ("A2_h256", "outputs/bc_chunk_sweep/A2_h256/checkpoint.pth"),
        ("A3_h256_wide", "outputs/bc_chunk_sweep/A3_h256_wide/checkpoint.pth"),
        # Hyperparameter sweep
        ("T1_arm_n=0.05", "outputs/bc_chunk_hp_sweep/T1/checkpoint.pth"),
        ("T3_arm_n=0.2", "outputs/bc_chunk_hp_sweep/T3/checkpoint.pth"),
        ("T4_drift=0.5", "outputs/bc_chunk_hp_sweep/T4/checkpoint.pth"),
        ("T5_drift=2.0", "outputs/bc_chunk_hp_sweep/T5/checkpoint.pth"),
        ("T6_hand_n=0.05", "outputs/bc_chunk_hp_sweep/T6/checkpoint.pth"),
        # Final best
        ("BEST(T4)", "outputs/bc_chunk_best/checkpoint.pth"),
    ]

    results = {}
    for name, ckpt_path in ckpts:
        full_path = os.path.join(_PROJ_ROOT, ckpt_path)
        if not os.path.exists(full_path):
            print(f"[SKIP] {name}: {full_path} not found")
            continue
        print(f"\n[{name}]  {ckpt_path}")
        r1 = evaluate_ckpt(full_path, test_files, device, num_samples=1, rollout_stride=1)
        r8 = evaluate_ckpt(full_path, test_files, device, num_samples=1, rollout_stride=8)
        results[name] = {"stride_1": r1, "stride_8": r8}
        print(f"  stride=1: arm={r1['ar_arm_mse']:.6f}  hand={r1['ar_hand_mse']:.6f}  "
              f"NC hand={r1['nc_hand_mse']:.6f}  gain={r1['vision_gain']:+.6f}")
        print(f"  stride=8: arm={r8['ar_arm_mse']:.6f}  hand={r8['ar_hand_mse']:.6f}  "
              f"NC hand={r8['nc_hand_mse']:.6f}  gain={r8['vision_gain']:+.6f}")

    print("\n" + "=" * 100)
    print(f"{'Config':<18} | {'stride=1 arm':>12} {'s=1 hand':>10} {'s=1 gain':>10} | "
          f"{'s=8 arm':>10} {'s=8 hand':>10} {'s=8 gain':>10}")
    print("=" * 100)
    for name, rr in results.items():
        r1, r8 = rr["stride_1"], rr["stride_8"]
        print(f"{name:<18} | {r1['ar_arm_mse']:>12.6f} {r1['ar_hand_mse']:>10.6f} "
              f"{r1['vision_gain']:>+10.6f} | "
              f"{r8['ar_arm_mse']:>10.6f} {r8['ar_hand_mse']:>10.6f} "
              f"{r8['vision_gain']:>+10.6f}")

    with open("/tmp/bc_ckpt_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
