"""
Dataset for Hand-Only Behavior Cloning over a frozen Hand-Action VAE.

Simplified version of the full BC dataset — no state normalization, no arm
state output. The model only sees images and the VAE prior latent.

NAMING CONVENTION (CRITICAL):
  `actions` = robot ABSOLUTE POSE at each timestep (12 dims = 6 arm + 6 hand).
  The hand-only BC predicts `actions[t+1, 6:12]` (next hand pose).

Each sample at trajectory frame t yields:

  img_main        : (3, 128, 128)  float32 in [0, 1]
  img_extra       : (3, 128, 128)  float32 in [0, 1]
  past_hand_win   : (8, 6)         float32, hand actions [a_{t-7}..a_t]
  gt_hand_action  : (6,)           float32, target = actions[t+1, 6:12]
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class BCHandDataset(Dataset):
    """Hand-only BC dataset: images + past hand window -> next hand pose.

    Args:
        data_dir:    directory with trajectory_*_demo_expert.pt files
        window_size: VAE prior window size (must match VAE training; default 8)
        noise_std_hand: Gaussian noise std added to past_hand_win (train only)
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 8,
        noise_std_hand: float = 0.0,
    ):
        data_dir = Path(data_dir)
        traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files in {data_dir}")

        self.window_size = window_size
        self.noise_std_hand = float(noise_std_hand)

        self.actions = []      # list of (T, 12) float32
        self.imgs_main = []    # list of (T, 3, 128, 128) uint8
        self.imgs_extra = []   # list of (T, 3, 128, 128) uint8
        self.samples = []      # flat index: list of (traj_idx, t)

        for traj_idx, f in enumerate(traj_files):
            data = torch.load(f, map_location="cpu", weights_only=False)

            actions = data["actions"][:, 0, :].float()                # (T, 12)
            main = data["curr_obs"]["main_images"][:, 0]              # (T, H, W, 3) uint8
            main = main.permute(0, 3, 1, 2).contiguous()              # (T, 3, H, W)
            extra = data["curr_obs"]["extra_view_images"][:, 0, 0]    # (T, H, W, 3) uint8
            extra = extra.permute(0, 3, 1, 2).contiguous()            # (T, 3, H, W)

            self.actions.append(actions)
            self.imgs_main.append(main)
            self.imgs_extra.append(extra)
            for t in range(actions.shape[0]):
                self.samples.append((traj_idx, t))

        n_frames = len(self.samples)
        bytes_imgs = sum(m.numel() + e.numel() for m, e in zip(self.imgs_main, self.imgs_extra))
        print(
            f"BCHandDataset: {len(traj_files)} trajectories, {n_frames} frames "
            f"from {data_dir}  (image RAM ~{bytes_imgs / 1e6:.0f} MB uint8)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        traj_idx, t = self.samples[idx]

        actions = self.actions[traj_idx]      # (T, 12)
        main = self.imgs_main[traj_idx]       # (T, 3, H, W) uint8
        extra = self.imgs_extra[traj_idx]     # (T, 3, H, W) uint8
        T = actions.shape[0]

        # ── Images at time t ──
        img_main = main[t].float() / 255.0    # (3, H, W)
        img_extra = extra[t].float() / 255.0  # (3, H, W)

        # ── Frozen VAE input: past hand window [a_{t-7}..a_t] ──
        hand = actions[:, 6:12]
        start = t - self.window_size + 1      # = t - 7
        if start < 0:
            pad_len = -start
            past_hand_win = torch.cat(
                [hand[0:1].expand(pad_len, -1), hand[0:t + 1]],
                dim=0,
            )
        else:
            past_hand_win = hand[start:t + 1]
        assert past_hand_win.shape == (self.window_size, 6), (
            f"past_hand_win shape {past_hand_win.shape} at traj={traj_idx} t={t}"
        )
        if self.noise_std_hand > 0:
            past_hand_win = past_hand_win + torch.randn_like(past_hand_win) * self.noise_std_hand

        # ── Target: hand pose at t+1 (or t for last frame) ──
        if t + 1 < T:
            gt_hand_action = actions[t + 1, 6:12]
        else:
            gt_hand_action = actions[t, 6:12]

        return {
            "img_main": img_main,
            "img_extra": img_extra,
            "past_hand_win": past_hand_win,
            "gt_hand_action": gt_hand_action,
        }
