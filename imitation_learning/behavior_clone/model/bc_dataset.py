"""
Dataset for Behavior Cloning over a frozen Hand-Action VAE.

NAMING CONVENTION (CRITICAL):
  In this dataset the field `actions` represents the robot's ABSOLUTE POSE at
  each timestep (12 dims = 6 arm + 6 hand). So `actions[t]` IS the robot
  state at time t. The BC policy predicts the NEXT pose `actions[t+1]`.

  ─────────────────────────────────────────────────────────
   time t            : observe state[t] = actions[t]
   BC predicts       : actions[t+1]   (the next absolute pose)
  ─────────────────────────────────────────────────────────

This matches the frozen VAE's training convention exactly: the VAE was
trained to predict `a_{t+1}` from window `[a_{t-7}..a_t]`, so we feed it
the same window and use its output as a prior over the next hand pose.

Each sample at trajectory frame t (0 <= t < T) yields:

  img_main      : (3, 128, 128)  float32 in [0, 1]
  img_extra     : (3, 128, 128)  float32 in [0, 1]
  state         : (12,)          float32, = actions[t] standardized by
                                  per-dim mean/std (z-score), then optionally
                                  masked for ablations. BC 3.0 only consumes
                                  the first 6 arm dims; the hand dims are kept
                                  for backward-compatibility with older logs.
  past_hand_win : (8, 6)         float32, hand actions [a_{t-7}..a_t]
                                  inclusive of current frame, padded with
                                  hand[0] when t < 7. Matches the VAE's
                                  HandActionWindowDataset window EXACTLY.
  gt_action     : (12,)          float32, target = actions[t+1] (raw, NOT
                                  normalized — the VAE decoder is trained
                                  in raw action space). For t == T-1 we use
                                  actions[T-1] (hold last pose).

IMPORTANT — past_hand_win is NOT a BC trainable input:
  The BC network's trainable layers only see (img_main, img_extra, state).
  past_hand_win is consumed exclusively by the FROZEN VAE encoder to compute
  a prior (mu_p, log_var_p) over the next hand pose. The BC's job is to predict
  a small delta_z correction so that z_ctrl = mu_prior + delta_z.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


def compute_action_stats(data_dir: str):
    """One pass over the train split: per-dim mean/std of 12-dim action vectors.

    Returns:
        mean: (12,) float32
        std:  (12,) float32  (clamped to >= 1e-6 to avoid divide-by-zero)
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
    if not files:
        raise FileNotFoundError(f"No trajectory files in {data_dir}")

    chunks = []
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        chunks.append(data["actions"][:, 0, :].float())  # (T, 12)
    all_actions = torch.cat(chunks, dim=0)
    mean = all_actions.mean(dim=0)
    std = all_actions.std(dim=0).clamp(min=1e-6)
    return mean, std


class BCDataset(Dataset):
    """BC dataset: per-step observation + next-action target.

    Args:
        data_dir:    directory with trajectory_*_demo_expert.pt files
        action_mean: (12,) per-dim mean for state standardization
        action_std:  (12,) per-dim std (already clamped)
        window_size: VAE prior window size (must match VAE training; default 8)
    """

    def __init__(
        self,
        data_dir: str,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        window_size: int = 8,
        noise_std_hand: float = 0.1,
        noise_std_arm: float = 0.0,
    ):
        data_dir = Path(data_dir)
        traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files in {data_dir}")

        self.window_size = window_size
        self.action_mean = action_mean.clone().float()
        self.action_std = action_std.clone().float()
        self.noise_std_hand = float(noise_std_hand)
        self.noise_std_arm = float(noise_std_arm)

        # Per-trajectory tensors (preloaded into RAM)
        self.actions = []     # list of (T, 12) float32
        self.imgs_main = []   # list of (T, 3, 128, 128) uint8
        self.imgs_extra = []  # list of (T, 3, 128, 128) uint8

        # Flat sample index: list of (traj_idx, t)
        self.samples = []

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
            f"BCDataset: {len(traj_files)} trajectories, {n_frames} frames "
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

        # ── BC trainable inputs at time t ──
        img_main = main[t].float() / 255.0    # (3, H, W)
        img_extra = extra[t].float() / 255.0  # (3, H, W)
        # state = actions[t] (current absolute pose), z-score normalized and
        # optionally ablated to test whether timing is leaking through state.
        state = (actions[t] - self.action_mean) / self.action_std  # (12,)
        # Optional Gaussian noise on arm state (training only — set 0 for test)
        if self.noise_std_arm > 0:
            state = state.clone()
            state[:6] = state[:6] + torch.randn(6) * self.noise_std_arm

        # ── Frozen VAE input (NOT a BC trainable input) ──
        # past_hand_win = 8 frames [a_{t-7}..a_t] inclusive of current frame.
        # Padded with hand[0] when t < window_size - 1. Matches VAE training.
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
        # Optional Gaussian noise on past_hand_win (training only — set 0 for test)
        if self.noise_std_hand > 0:
            past_hand_win = past_hand_win + torch.randn_like(past_hand_win) * self.noise_std_hand

        # ── Target: actions[t+1] (or actions[t] for last frame) ──
        if t + 1 < T:
            gt_action = actions[t + 1]
        else:
            gt_action = actions[t]

        return {
            "img_main": img_main,
            "img_extra": img_extra,
            "state": state,
            "past_hand_win": past_hand_win,
            "gt_action": gt_action,
        }
