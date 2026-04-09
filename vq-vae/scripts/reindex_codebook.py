"""
Reindex VQ-VAE codebook tokens by PCA projection for downstream policy use.

After VQ-VAE training, token indices are arbitrary. This script:
1. Enumerates all token combinations and decodes to hand actions
2. Projects actions to 1D via PCA
3. Sorts by projection value so index order has physical meaning
4. Saves sorted codebook as .npy for downstream policy training

Usage:
    python scripts/reindex_codebook.py
    python scripts/reindex_codebook.py --ckpt outputs/hand_vqvae_v2/checkpoint.pth
"""

import argparse
import importlib.util
import os

import numpy as np
import torch
from sklearn.decomposition import PCA

_vqvae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vqvae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandVQVAE = _load("model/hand_vqvae.py", "hand_vqvae").HandVQVAE

JOINT_UPPER_RAD = [2.356, 0.698, 1.518, 1.571, 1.571, 1.545]
JOINT_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]


def main():
    parser = argparse.ArgumentParser(description="Reindex VQ-VAE codebook via PCA sorting")
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(_vqvae_root, "../outputs/hand_vqvae/checkpoint.pth"))
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npy path (default: same dir as ckpt)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.ckpt), "sorted_codebook.npy")

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HandVQVAE()
    model.load_state_dict(ckpt["model"])
    model.eval()

    codebook_size = model.vq.layers[0].codebook_size
    num_layers = model.vq.num_layers
    total = codebook_size ** num_layers

    # Step 1: Enumerate all token combinations and decode
    all_indices = []
    for i in range(codebook_size):
        for j in range(codebook_size):
            all_indices.append([i, j])
    all_indices = torch.tensor(all_indices)

    with torch.no_grad():
        actions_raw = model.decode_from_indices(all_indices).numpy()  # (16, 6)

    # Clamp to valid range [0, 1] before sorting
    # VQ-VAE decoder may produce small out-of-range values
    actions = np.clip(actions_raw, 0.0, 1.0)

    # Step 2: Compute grip score = mean of dims 1-5 (exclude thumb rotation)
    # Thumb rotation (dim 0) doesn't reflect grip openness,
    # dims 1-5 (thumb bend + 4 fingers) directly indicate how closed the hand is
    grip_weights = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2])
    grip_score = actions @ grip_weights  # (16,)

    print(f"Sorting by grip score (weighted mean of dims 1-5)")
    print(f"Grip weights: {grip_weights}")

    # Step 3: Sort by grip score (0=fully open, 1=fully closed)
    sorted_idx = np.argsort(grip_score)

    sorted_actions = actions[sorted_idx]           # (16, 6)
    sorted_orig_indices = all_indices[sorted_idx]  # (16, 2) original L0,L1
    sorted_grip = grip_score[sorted_idx]

    # Step 4: Save
    np.save(args.output, sorted_actions)

    # Also save metadata for reference
    meta_path = args.output.replace(".npy", "_meta.npz")
    np.savez(meta_path,
             sorted_actions=sorted_actions,
             original_indices=sorted_orig_indices.numpy(),
             grip_score=sorted_grip,
             grip_weights=grip_weights)

    # Print results
    print(f"\n{'=' * 90}")
    print(f"Sorted codebook: {total} entries")
    print(f"{'=' * 90}")
    print(f"\n{'new':>4} {'orig':>6} {'grip':>8} | " +
          " | ".join(f"{n:>10}" for n in JOINT_NAMES))
    print("-" * 90)

    for new_id in range(total):
        orig = sorted_orig_indices[new_id]
        act = sorted_actions[new_id]
        rads = act * np.array(JOINT_UPPER_RAD)
        degs = np.rad2deg(rads)
        act_str = " | ".join(f"{d:>7.1f}deg" for d in degs)
        print(f"{new_id:>4} L{orig[0]},{orig[1]}  {sorted_grip[new_id]:>8.4f} | {act_str}")

    print(f"\nSaved: {args.output}")
    print(f"Meta:  {meta_path}")
    print(f"\nDownstream usage:")
    print(f"  codebook = np.load('{os.path.basename(args.output)}')  # ({total}, 6)")
    print(f"  # hand_action → nearest index → normalize to [-1,1]")
    print(f"  # policy predicts 1 scalar instead of 6-dim action")


if __name__ == "__main__":
    main()
