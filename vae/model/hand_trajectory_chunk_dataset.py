"""
Trajectory-start dataset for chunked hand-prior training.

Each sample keeps the episode reset state explicit:
  - `seed_window`: repeated first action, shape (window_size, 6)
  - `initial_state`: first action, shape (6,)
  - `target_chunks`: non-overlapping future chunks from t=0 onward,
    shape (num_chunks, future_horizon, 6)

This lets the model learn the rollout dynamics from the true episode start,
instead of treating all-open windows from different absolute times as the same state.
"""

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


class HandTrajectoryChunkSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        window_size: int = 8,
        future_horizon: int = 12,
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
            num_chunks = max(1, math.ceil((total_steps - 1) / future_horizon))

            seed_window = actions[0:1].expand(window_size, -1).clone()
            target_chunks = []
            for chunk_idx in range(num_chunks):
                start = chunk_idx * future_horizon + 1
                future = []
                for offset in range(future_horizon):
                    idx = min(start + offset, total_steps - 1)
                    future.append(actions[idx])
                target_chunks.append(torch.stack(future, dim=0))

            self.samples.append(
                {
                    "seed_window": seed_window,
                    "initial_state": actions[0].clone(),
                    "target_chunks": torch.stack(target_chunks, dim=0),
                    "num_chunks": num_chunks,
                    "length": total_steps,
                }
            )

        print(
            f"Loaded {len(traj_files)} trajectories "
            f"(window={window_size}, horizon={future_horizon}) from {data_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seed_window = sample["seed_window"]
        if self.noise_std > 0:
            seed_window = seed_window + torch.randn_like(seed_window) * self.noise_std
        return {
            "seed_window": seed_window,
            "initial_state": sample["initial_state"],
            "target_chunks": sample["target_chunks"],
            "num_chunks": sample["num_chunks"],
            "length": sample["length"],
        }
