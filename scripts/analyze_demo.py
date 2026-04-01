"""
Analyze zero/near-zero action frames across all trajectories.
Determine whether data cleaning is needed for VQ-VAE training.
"""

from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path("/home/admin01/yxy/VQ-VLA/data/20260327-11:10:43/demos")


def main():
    thresholds = [0.0, 1e-4, 1e-3, 1e-2, 5e-2]

    # Per-trajectory stats
    traj_stats = []
    all_actions = []
    all_arm_norms = []
    all_hand_norms = []

    for i in range(152):
        d = torch.load(DATA_DIR / f"trajectory_{i}_demo_expert.pt", map_location="cpu", weights_only=False)
        actions = d["actions"][:, 0, :]  # (T, 12)
        T = actions.shape[0]

        arm = actions[:, :6]
        hand = actions[:, 6:]
        full_norm = actions.norm(dim=1)       # L2 norm per step
        arm_norm = arm.norm(dim=1)
        hand_norm = hand.norm(dim=1)

        # Count exact zeros
        exact_zero = (full_norm == 0).sum().item()
        # Count zero arm + zero hand separately
        arm_zero = (arm_norm == 0).sum().item()
        hand_zero = (hand_norm == 0).sum().item()

        traj_stats.append({
            "id": i,
            "T": T,
            "exact_zero": exact_zero,
            "arm_zero": arm_zero,
            "hand_zero": hand_zero,
        })

        all_actions.append(actions)
        all_arm_norms.append(arm_norm)
        all_hand_norms.append(hand_norm)

    all_actions = torch.cat(all_actions, dim=0)
    all_arm_norms = torch.cat(all_arm_norms, dim=0)
    all_hand_norms = torch.cat(all_hand_norms, dim=0)
    all_full_norms = all_actions.norm(dim=1)
    N = all_actions.shape[0]

    # ============ Global summary ============
    print("=" * 70)
    print(f"Total frames: {N}  (152 trajectories)")
    print("=" * 70)

    print("\n--- Full action (12-dim) near-zero frame counts ---")
    print(f"  {'threshold':>12} {'count':>8} {'percent':>8}")
    for th in thresholds:
        cnt = (all_full_norms <= th).sum().item()
        print(f"  {th:>12.4f} {cnt:>8d} {cnt/N*100:>7.1f}%")

    print("\n--- Arm action (dim 0-5) near-zero frame counts ---")
    print(f"  {'threshold':>12} {'count':>8} {'percent':>8}")
    for th in thresholds:
        cnt = (all_arm_norms <= th).sum().item()
        print(f"  {th:>12.4f} {cnt:>8d} {cnt/N*100:>7.1f}%")

    print("\n--- Hand action (dim 6-11) near-zero frame counts ---")
    print(f"  {'threshold':>12} {'count':>8} {'percent':>8}")
    for th in thresholds:
        cnt = (all_hand_norms <= th).sum().item()
        print(f"  {th:>12.4f} {cnt:>8d} {cnt/N*100:>7.1f}%")

    # ============ Position in trajectory ============
    print("\n--- Where do zero-action frames appear? ---")
    head_zero = 0   # first 3 steps
    tail_zero = 0   # last 3 steps
    mid_zero = 0    # middle steps

    for i in range(152):
        d = torch.load(DATA_DIR / f"trajectory_{i}_demo_expert.pt", map_location="cpu", weights_only=False)
        actions = d["actions"][:, 0, :]
        T = actions.shape[0]
        norms = actions.norm(dim=1)
        for t in range(T):
            if norms[t] == 0:
                if t < 3:
                    head_zero += 1
                elif t >= T - 3:
                    tail_zero += 1
                else:
                    mid_zero += 1

    total_zero = head_zero + tail_zero + mid_zero
    print(f"  Exact zero frames: {total_zero}")
    print(f"    Head (first 3 steps): {head_zero} ({head_zero/max(total_zero,1)*100:.1f}%)")
    print(f"    Tail (last 3 steps):  {tail_zero} ({tail_zero/max(total_zero,1)*100:.1f}%)")
    print(f"    Middle:               {mid_zero} ({mid_zero/max(total_zero,1)*100:.1f}%)")

    # ============ Per-trajectory zero ratio ============
    print("\n--- Per-trajectory zero-action ratio distribution ---")
    zero_ratios = [s["exact_zero"] / s["T"] for s in traj_stats]
    zero_ratios = np.array(zero_ratios)
    print(f"  mean:   {zero_ratios.mean():.3f}")
    print(f"  median: {np.median(zero_ratios):.3f}")
    print(f"  max:    {zero_ratios.max():.3f}")
    print(f"  >50% zero: {(zero_ratios > 0.5).sum()} trajectories")
    print(f"  >30% zero: {(zero_ratios > 0.3).sum()} trajectories")
    print(f"  >10% zero: {(zero_ratios > 0.1).sum()} trajectories")

    # Show worst trajectories
    worst_idx = np.argsort(zero_ratios)[-10:][::-1]
    print(f"\n  Top-10 highest zero-ratio trajectories:")
    print(f"  {'traj':>6} {'steps':>6} {'zeros':>6} {'ratio':>8} {'arm_z':>6} {'hand_z':>7}")
    for idx in worst_idx:
        s = traj_stats[idx]
        print(f"  {s['id']:>6} {s['T']:>6} {s['exact_zero']:>6} "
              f"{s['exact_zero']/s['T']:>8.3f} {s['arm_zero']:>6} {s['hand_zero']:>7}")

    # ============ Consecutive zero runs ============
    print("\n--- Consecutive zero-action runs ---")
    run_lengths = []
    for i in range(152):
        d = torch.load(DATA_DIR / f"trajectory_{i}_demo_expert.pt", map_location="cpu", weights_only=False)
        actions = d["actions"][:, 0, :]
        norms = actions.norm(dim=1)
        is_zero = (norms == 0).numpy()
        run = 0
        for z in is_zero:
            if z:
                run += 1
            else:
                if run > 0:
                    run_lengths.append(run)
                run = 0
        if run > 0:
            run_lengths.append(run)

    if run_lengths:
        run_lengths = np.array(run_lengths)
        print(f"  Total zero-runs: {len(run_lengths)}")
        print(f"  Run length: mean={run_lengths.mean():.1f}, median={np.median(run_lengths):.0f}, "
              f"max={run_lengths.max()}, min={run_lengths.min()}")
        # Distribution
        for maxlen in [1, 2, 3, 5, 10, 20]:
            cnt = (run_lengths <= maxlen).sum()
            print(f"    runs <= {maxlen:2d} steps: {cnt}/{len(run_lengths)}")

    # ============ Action magnitude distribution ============
    print("\n--- Action L2 norm distribution (non-zero frames only) ---")
    nonzero_mask = all_full_norms > 0
    nz_norms = all_full_norms[nonzero_mask].numpy()
    if len(nz_norms) > 0:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"  N = {len(nz_norms)} non-zero frames")
        for p in percentiles:
            print(f"    P{p:02d}: {np.percentile(nz_norms, p):.4f}")

    # ============ Recommendation ============
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    zero_pct = total_zero / N * 100
    print(f"\n  Zero-action frames: {total_zero}/{N} ({zero_pct:.1f}%)")
    if zero_pct > 20:
        print("  -> Significant portion. Recommend cleaning.")
    elif zero_pct > 5:
        print("  -> Moderate portion. Consider cleaning head/tail zeros.")
    else:
        print("  -> Small portion. Cleaning is optional.")


if __name__ == "__main__":
    main()
