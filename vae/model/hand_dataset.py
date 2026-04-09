"""
Dataset for hand action VAE training.

Each sample is a (window, target) pair:
  window: (window_size, 6) past hand actions [t-7 : t]
  target: (6,) ground truth hand action at t+1

Padding behavior is auto-detected from the data:
  - Absolute actions (per-dim |mean| > 0.05): pad with first frame
  - Delta actions (per-dim |mean| < 0.05): pad with zeros (= "no motion before t=0")
For the last frame of each trajectory, target = action at t (hold).
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class HandActionWindowDataset(Dataset):
    """Builds sliding window samples from all trajectories.

    Args:
        noise_std: if > 0, add Gaussian noise to the input window during __getitem__.
                   Target is always clean GT. Only effective during training.
    """

    def __init__(self, data_dir: str, window_size: int = 8, noise_std: float = 0.0):
        data_dir = Path(data_dir)
        traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files found in {data_dir}")

        self.window_size = window_size
        self.noise_std = noise_std
        self.samples = []  # list of (window, target) tensors

        # First pass: load all trajectories' actions
        all_actions = []
        for f in traj_files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            all_actions.append(data["actions"][:, 0, 6:12].float())  # (T, 6)

        # Auto-detect action mode from per-dim mean
        concat = torch.cat(all_actions, dim=0)
        per_dim_abs_mean = concat.mean(dim=0).abs()
        self.is_delta = bool(per_dim_abs_mean.max().item() < 0.05)
        pad_mode = "zero" if self.is_delta else "first_frame"

        # Second pass: build sliding-window samples
        for actions in all_actions:
            T = actions.shape[0]
            for t in range(T):
                start = t - window_size + 1
                if start < 0:
                    pad_len = -start
                    if self.is_delta:
                        pad = torch.zeros(pad_len, actions.shape[1])
                    else:
                        pad = actions[0:1].expand(pad_len, -1)
                    window = torch.cat([pad, actions[0:t + 1]], dim=0)
                else:
                    window = actions[start:t + 1]

                if t + 1 < T:
                    target = actions[t + 1]
                else:
                    target = actions[t]

                self.samples.append((window, target))

        print(f"Loaded {len(traj_files)} trajectories, {len(self.samples)} samples "
              f"(window={window_size}, action_mode={'delta' if self.is_delta else 'absolute'}, "
              f"pad={pad_mode}) from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window, target = self.samples[idx]
        if self.noise_std > 0:
            window = window + torch.randn_like(window) * self.noise_std
        return window, target  # (window_size, 6), (6,)
