"""
Dataset for Behavior Cloning over a frozen Hand-Action VAE.

Each sample at trajectory frame t yields:
  img_main      : (3, 128, 128)  float32 in [0, 1]
  img_extra     : (3, 128, 128)  float32 in [0, 1]
  state         : (24,)          float32, standardized by train mean/std
  past_hand_win : (8, 6)         float32, hand actions [a_{t-8} .. a_{t-1}]
                  left-padded with actions[0] when t < 8.
  gt_action     : (12,)          float32, raw action at frame t

CRITICAL: past_hand_win ENDS AT t-1 (exclusive of t). The frozen VAE was
trained to predict a_{t+1} from window [a_{t-7} .. a_t], so to use it as a
prior over a_t we must shift its input one step earlier — otherwise the
target leaks into the prior. This is the single most important detail in
the BC dataset.

Images are kept in RAM as uint8 (~500 MB total) and cast to float32 only
inside __getitem__ to avoid 4 GB RAM blowup.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


def compute_state_stats(data_dir: str):
    """One pass over the train split: per-dim mean/std of curr_obs.states.

    Returns:
        mean: (24,) float32
        std:  (24,) float32  (clamped to >= 1e-6 to avoid divide-by-zero)
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
    if not files:
        raise FileNotFoundError(f"No trajectory files in {data_dir}")

    chunks = []
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        chunks.append(data["curr_obs"]["states"][:, 0, :].float())  # (T, 24)
    all_states = torch.cat(chunks, dim=0)  # (sum_T, 24)
    mean = all_states.mean(dim=0)
    std = all_states.std(dim=0).clamp(min=1e-6)
    return mean, std


class BCDataset(Dataset):
    """Behavior cloning dataset: images + state + past hand window + GT action.

    Args:
        data_dir:    directory with trajectory_*_demo_expert.pt files
        state_mean:  (24,) tensor — per-dim mean for state standardization
        state_std:   (24,) tensor — per-dim std (already clamped)
        window_size: VAE prior window size (must match VAE training; default 8)
    """

    def __init__(
        self,
        data_dir: str,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        window_size: int = 8,
    ):
        data_dir = Path(data_dir)
        traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files in {data_dir}")

        self.window_size = window_size
        self.state_mean = state_mean.clone().float()
        self.state_std = state_std.clone().float()

        # Per-trajectory tensors (pre-loaded into RAM)
        self.actions = []     # list of (T, 12) float32
        self.states = []      # list of (T, 24) float32
        self.imgs_main = []   # list of (T, 3, 128, 128) uint8
        self.imgs_extra = []  # list of (T, 3, 128, 128) uint8

        # Flat sample index: list of (traj_idx, t)
        self.samples = []

        for traj_idx, f in enumerate(traj_files):
            data = torch.load(f, map_location="cpu", weights_only=False)

            actions = data["actions"][:, 0, :].float()                  # (T, 12)
            states = data["curr_obs"]["states"][:, 0, :].float()        # (T, 24)
            # main_images:  (T, 1, H, W, 3) -> (T, 3, H, W) uint8
            main = data["curr_obs"]["main_images"][:, 0]                # (T, H, W, 3) uint8
            main = main.permute(0, 3, 1, 2).contiguous()                # (T, 3, H, W)
            # extra_view_images: (T, 1, 1, H, W, 3) -> (T, 3, H, W) uint8
            extra = data["curr_obs"]["extra_view_images"][:, 0, 0]      # (T, H, W, 3) uint8
            extra = extra.permute(0, 3, 1, 2).contiguous()              # (T, 3, H, W)

            T = actions.shape[0]
            self.actions.append(actions)
            self.states.append(states)
            self.imgs_main.append(main)
            self.imgs_extra.append(extra)

            for t in range(T):
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
        states = self.states[traj_idx]        # (T, 24)
        main = self.imgs_main[traj_idx]       # (T, 3, H, W) uint8
        extra = self.imgs_extra[traj_idx]     # (T, 3, H, W) uint8

        # --- Past hand window ENDING AT t-1 (NOT t). See module docstring. ---
        hand = actions[:, 6:12]               # (T, 6)
        start = t - self.window_size          # may be negative
        if start < 0:
            pad_len = -start
            past_hand_win = torch.cat(
                [hand[0:1].expand(pad_len, -1), hand[0:t]],
                dim=0,
            )
        else:
            past_hand_win = hand[start:t]
        # If t == 0, hand[0:0] is empty -> all 8 slots come from the pad branch.
        assert past_hand_win.shape == (self.window_size, 6), (
            f"past_hand_win shape {past_hand_win.shape} at traj={traj_idx} t={t}"
        )

        # --- Observations at frame t ---
        img_main = main[t].float() / 255.0    # (3, H, W) in [0, 1]
        img_extra = extra[t].float() / 255.0  # (3, H, W) in [0, 1]
        state = (states[t] - self.state_mean) / self.state_std  # (24,)
        gt_action = actions[t]                # (12,) raw

        return {
            "img_main": img_main,
            "img_extra": img_extra,
            "state": state,
            "past_hand_win": past_hand_win,
            "gt_action": gt_action,
        }
