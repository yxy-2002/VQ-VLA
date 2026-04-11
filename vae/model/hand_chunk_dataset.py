"""
Dataset for chunked hand-action prediction.

Each sample is a (past_window, future_chunk) pair:
  window: (window_size, 6) history ending at time t
  target: (future_horizon, 6) future actions from t+1 onward

The future chunk is right-padded with the last available action so every target
has the same length. This lets the chunk decoder learn "hold after close" naturally.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class HandActionChunkDataset(Dataset):
    """Build sliding-window / future-chunk pairs from all trajectories."""

    def __init__(
        self,
        data_dir: str,
        window_size: int = 8,
        future_horizon: int = 8,
        noise_std: float = 0.0,
    ):
        data_dir = Path(data_dir)
        traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files found in {data_dir}")

        self.window_size = window_size
        self.future_horizon = future_horizon
        self.noise_std = noise_std
        self.samples = []

        for path in traj_files:
            data = torch.load(path, map_location="cpu", weights_only=False)
            actions = data["actions"][:, 0, 6:12].float()
            total_steps = actions.shape[0]

            for t in range(total_steps):
                # Build history window ending at t (left-pad with first frame)
                start = t - window_size + 1
                if start < 0:
                    pad = actions[0:1].expand(-start, -1)
                    window = torch.cat([pad, actions[: t + 1]], dim=0)
                else:
                    window = actions[start : t + 1]

                # Build future chunk from t+1 (right-pad with last frame)
                future = []
                for offset in range(1, future_horizon + 1):
                    idx = min(t + offset, total_steps - 1)
                    future.append(actions[idx])
                target = torch.stack(future, dim=0)

                self.samples.append((window, target))

        print(
            f"Loaded {len(traj_files)} trajectories, {len(self.samples)} samples "
            f"(window={window_size}, horizon={future_horizon}) from {data_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, target = self.samples[idx]
        if self.noise_std > 0:
            window = window + torch.randn_like(window) * self.noise_std
        return window, target
