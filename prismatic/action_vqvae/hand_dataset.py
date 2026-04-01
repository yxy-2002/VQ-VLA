"""
Dataset for loading dexterous hand actions from trajectory .pt files.
Extracts action dims 6:12 (hand joints) from each timestep.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class HandActionDataset(Dataset):
    """Loads all trajectories from a directory, extracts hand actions as flat (N, 6) tensor."""

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files found in {data_dir}")

        chunks = []
        for f in traj_files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            actions = data["actions"][:, 0, 6:12]  # (T, 6) hand action dims
            chunks.append(actions)

        self.data = torch.cat(chunks, dim=0).float()  # (N, 6)
        print(f"Loaded {len(traj_files)} trajectories, {len(self.data)} samples from {data_dir}")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
