"""
Decode all token combinations from trained Hand VQ-VAE and print corresponding actions.

Usage:
    python scripts/decode_codebook.py
    python scripts/decode_codebook.py --ckpt outputs/hand_vqvae_v2/checkpoint.pth
"""

import argparse
import importlib.util
import os

import torch

_vqvae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vqvae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandVQVAE = _load("model/hand_vqvae.py", "hand_vqvae").HandVQVAE

# URDF joint upper limits (rad), from RuiYan_Hand_Left_Mimic.urdf
JOINT_UPPER_RAD = [2.356, 0.698, 1.518, 1.571, 1.571, 1.545]

JOINT_NAMES = [
    "thumb_rot",   # joint_1_1
    "thumb_bend",  # joint_1_2
    "index",       # joint_2_1
    "middle",      # joint_3_1
    "ring",        # joint_4_1
    "pinky",       # joint_5_1
]


def main():
    parser = argparse.ArgumentParser(description="Decode all VQ-VAE token combos to hand joint angles")
    parser.add_argument("--ckpt", type=str, default=os.path.join(_vqvae_root, "../outputs/hand_vqvae/checkpoint.pth"))
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HandVQVAE()
    model.load_state_dict(ckpt["model"])
    model.eval()

    codebook_size = model.vq.layers[0].codebook_size
    num_layers = model.vq.num_layers
    total = codebook_size ** num_layers

    # Build all index pairs
    indices = []
    for i in range(codebook_size):
        for j in range(codebook_size):
            indices.append([i, j])
    indices = torch.tensor(indices)

    # Decode
    with torch.no_grad():
        actions = model.decode_from_indices(indices)

    # Print
    header = f"{'combo':>6} {'L0':>3} {'L1':>3} | " + " | ".join(f"{n:>12}" for n in JOINT_NAMES)
    print(header)
    print("-" * len(header))
    for c in range(total):
        l0, l1 = indices[c]
        raw = actions[c].tolist()
        rads = [v * u for v, u in zip(raw, JOINT_UPPER_RAD)]
        vals = " | ".join(f"{r:>9.3f}rad" for r in rads)
        print(f"{c:>6} {l0.item():>3} {l1.item():>3} | {vals}")


if __name__ == "__main__":
    main()
