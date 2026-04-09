"""
Visualize all 16 VQ-VAE token combinations as hand poses using PyBullet.

Generates two sets of images:
  1. Original order (L0, L1 grid) - codebook_grid_original.png
  2. PCA-sorted order (0-15 linear) - codebook_grid_sorted.png

Usage:
    python scripts/visualize_codebook.py
    python scripts/visualize_codebook.py --ckpt outputs/hand_vqvae_v2/checkpoint.pth
    python scripts/visualize_codebook.py --sorted_codebook outputs/hand_vqvae_v2/sorted_codebook.npy
"""

import argparse
import importlib.util
import os

import numpy as np
import pybullet as p
import pybullet_data
import torch
from PIL import Image, ImageDraw, ImageFont

_vqvae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vqvae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandVQVAE = _load("model/hand_vqvae.py", "hand_vqvae").HandVQVAE

URDF_PATH = os.path.join(_vqvae_root, "../RuiYan_Hand_Left_Mimic/RuiYan_Hand_Left_Mimic.urdf")
JOINT_UPPER_RAD = np.array([2.356, 0.698, 1.518, 1.571, 1.571, 1.545])
JOINT_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]


# ─── PyBullet helpers ──────────────────────────────────────────────────────────

def get_actuated_joints(robot_id):
    actuated = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        if info[2] == p.JOINT_REVOLUTE:
            if any(name.endswith(s) for s in ["_1_1", "_1_2", "_2_1", "_3_1", "_4_1", "_5_1"]):
                actuated.append((i, name))
    return actuated


def set_joint_angles(robot_id, actuated_joints, joint_angles):
    for (idx, _), angle in zip(actuated_joints, joint_angles):
        p.resetJointState(robot_id, idx, angle)
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        if info[2] == p.JOINT_REVOLUTE:
            if name.endswith("_1_3"):
                p.resetJointState(robot_id, i, joint_angles[1] * 1.675)
            elif any(name.endswith(f"_{f}_2") for f in ["2", "3", "4", "5"]):
                fnum = name.split("_")[-2]
                fmap = {"2": 2, "3": 3, "4": 4, "5": 5}
                if fnum in fmap:
                    p.resetJointState(robot_id, i, joint_angles[fmap[fnum]])


def render_view(robot_id, eye, target, width, height):
    view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.01, farVal=2.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, view, proj, renderer=p.ER_TINY_RENDERER,
        lightDirection=[0.4, -0.4, 0.8], lightColor=[1, 1, 1], lightDistance=0.5,
        lightAmbientCoeff=0.5, lightDiffuseCoeff=0.7, lightSpecularCoeff=0.3)
    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
    rgb = img[:, :, :3].copy()
    gray_mask = np.all(np.abs(rgb.astype(int) - 201) < 15, axis=-1)
    rgb[gray_mask] = 255
    return rgb


def render_combo(robot_id, actuated_joints, angles_rad, sz=400):
    set_joint_angles(robot_id, actuated_joints, angles_rad)
    front = render_view(robot_id, eye=[0.35, -0.20, 0.24], target=[0, 0, 0.05], width=sz, height=sz)
    side = render_view(robot_id, eye=[0.0, -0.38, 0.20], target=[0, 0, 0.05], width=sz, height=sz)
    combined = np.ones((sz, sz * 2, 3), dtype=np.uint8) * 255
    combined[:, :sz] = front
    combined[:, sz:] = side
    return combined


# ─── Grid assembly ─────────────────────────────────────────────────────────────

def build_grid(images, labels, codebook_size, sz):
    """Build a 4×4 grid image with labels under each cell."""
    cell_w = sz * 2
    cell_h = sz
    label_h = 50
    ncols = codebook_size
    nrows = codebook_size
    grid = np.ones(((cell_h + label_h) * nrows, cell_w * ncols, 3), dtype=np.uint8) * 255

    for c in range(len(images)):
        row, col = c // ncols, c % ncols
        y0 = row * (cell_h + label_h)
        x0 = col * cell_w
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = images[c]

    grid_pil = Image.fromarray(grid)
    draw = ImageDraw.Draw(grid_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    for c in range(len(images)):
        row, col = c // ncols, c % ncols
        x0 = col * cell_w + 4
        y0 = row * (cell_h + label_h) + cell_h + 2
        title, subtitle = labels[c]
        draw.text((x0, y0), title, fill=(0, 0, 0), font=font)
        draw.text((x0, y0 + 16), subtitle, fill=(80, 80, 80), font=font_sm)
        # View labels
        yt = row * (cell_h + label_h) + 4
        draw.text((col * cell_w + 4, yt), "front", fill=(60, 60, 60), font=font_sm)
        draw.text((col * cell_w + sz + 4, yt), "side", fill=(60, 60, 60), font=font_sm)

    return grid_pil


def build_strip(images, labels, sz):
    """Build a single-row horizontal strip for sorted codebook (1×16)."""
    cell_w = sz * 2
    cell_h = sz
    label_h = 50
    n = len(images)
    # 2 rows of 8 for better aspect ratio
    ncols = 8
    nrows = (n + ncols - 1) // ncols
    strip = np.ones(((cell_h + label_h) * nrows, cell_w * ncols, 3), dtype=np.uint8) * 255

    for c in range(n):
        row, col = c // ncols, c % ncols
        y0 = row * (cell_h + label_h)
        x0 = col * cell_w
        strip[y0:y0 + cell_h, x0:x0 + cell_w] = images[c]

    strip_pil = Image.fromarray(strip)
    draw = ImageDraw.Draw(strip_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    for c in range(n):
        row, col = c // ncols, c % ncols
        x0 = col * cell_w + 4
        y0 = row * (cell_h + label_h) + cell_h + 2
        title, subtitle = labels[c]
        draw.text((x0, y0), title, fill=(0, 0, 0), font=font)
        draw.text((x0, y0 + 16), subtitle, fill=(80, 80, 80), font=font_sm)
        yt = row * (cell_h + label_h) + 4
        draw.text((col * cell_w + 4, yt), "front", fill=(60, 60, 60), font=font_sm)
        draw.text((col * cell_w + sz + 4, yt), "side", fill=(60, 60, 60), font=font_sm)

    return strip_pil


def deg_label(actions_norm):
    """Convert normalized action to degree string."""
    degs = np.rad2deg(np.clip(actions_norm, 0, 1) * JOINT_UPPER_RAD)
    return "  ".join(f"{JOINT_NAMES[i]}={degs[i]:.0f}" for i in range(6)) + " (deg)"


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE codebook (original + sorted)")
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(_vqvae_root, "../outputs/hand_vqvae/checkpoint.pth"))
    parser.add_argument("--sorted_codebook", type=str, default=None,
                        help="Path to sorted_codebook.npy (default: same dir as ckpt)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_vqvae_root, "../visualizations/codebook_poses"))
    parser.add_argument("--cell_size", type=int, default=400)
    args = parser.parse_args()

    if args.sorted_codebook is None:
        args.sorted_codebook = os.path.join(os.path.dirname(args.ckpt), "sorted_codebook.npy")

    os.makedirs(args.output_dir, exist_ok=True)
    sz = args.cell_size

    # ── Load model and decode original codebook ──
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HandVQVAE()
    model.load_state_dict(ckpt["model"])
    model.eval()

    codebook_size = model.vq.layers[0].codebook_size
    total = codebook_size ** model.vq.num_layers

    all_indices = []
    for i in range(codebook_size):
        for j in range(codebook_size):
            all_indices.append([i, j])
    all_indices = torch.tensor(all_indices)

    with torch.no_grad():
        orig_actions = model.decode_from_indices(all_indices).numpy()  # (16, 6) normalized

    # ── Load sorted codebook ──
    sorted_actions = np.load(args.sorted_codebook)  # (16, 6) normalized
    meta_path = args.sorted_codebook.replace(".npy", "_meta.npz")
    meta = np.load(meta_path) if os.path.exists(meta_path) else None

    # ── Setup PyBullet ──
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    robot_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0], useFixedBase=True)
    actuated_joints = get_actuated_joints(robot_id)
    for i in range(-1, p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, i, rgbaColor=[0.82, 0.72, 0.63, 1.0])

    # ── Render original order ──
    print(f"Rendering {total} combos (original order)...")
    orig_images = []
    orig_labels = []
    for c in range(total):
        l0, l1 = all_indices[c]
        angles_rad = np.clip(orig_actions[c] * JOINT_UPPER_RAD, 0, JOINT_UPPER_RAD)
        rgb = render_combo(robot_id, actuated_joints, angles_rad, sz=sz)
        orig_images.append(rgb)
        orig_labels.append((
            f"#{c}  L0={l0.item()} L1={l1.item()}",
            deg_label(orig_actions[c]),
        ))
        Image.fromarray(rgb).save(os.path.join(args.output_dir, f"original_{c:02d}_L{l0}_{l1}.png"))
        print(f"  original {c:2d} (L0={l0.item()}, L1={l1.item()}) done")

    # ── Render sorted order ──
    print(f"\nRendering {total} combos (PCA-sorted order)...")
    sorted_images = []
    sorted_labels = []
    for c in range(total):
        angles_rad = np.clip(sorted_actions[c] * JOINT_UPPER_RAD, 0, JOINT_UPPER_RAD)
        rgb = render_combo(robot_id, actuated_joints, angles_rad, sz=sz)
        sorted_images.append(rgb)

        # Find original index for label
        orig_info = ""
        if meta is not None:
            orig_idx = meta["original_indices"][c]
            orig_info = f"  (was L{orig_idx[0]},{orig_idx[1]})"
        sorted_labels.append((
            f"sorted #{c}{orig_info}",
            deg_label(sorted_actions[c]),
        ))
        Image.fromarray(rgb).save(os.path.join(args.output_dir, f"sorted_{c:02d}.png"))
        print(f"  sorted {c:2d} done")

    p.disconnect()

    # ── Assemble grids ──
    # Original: 4×4 grid (rows=L0, cols=L1)
    grid_orig = build_grid(orig_images, orig_labels, codebook_size, sz)
    grid_orig_path = os.path.join(args.output_dir, "codebook_grid_original.png")
    grid_orig.save(grid_orig_path)

    # Sorted: 2×8 strip (index 0→15 left to right, top to bottom)
    grid_sorted = build_strip(sorted_images, sorted_labels, sz)
    grid_sorted_path = os.path.join(args.output_dir, "codebook_grid_sorted.png")
    grid_sorted.save(grid_sorted_path)

    print(f"\nOriginal grid: {grid_orig_path}")
    print(f"Sorted grid:   {grid_sorted_path}")
    print(f"Individual images: {args.output_dir}/original_*.png, sorted_*.png")


if __name__ == "__main__":
    main()
