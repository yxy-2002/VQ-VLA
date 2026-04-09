"""
Simplified VQ-VAE for dexterous hand action tokenization.

Architecture: MLP Encoder → 2-layer Residual VQ (4 entries each) → MLP Decoder
Input/Output: single-timestep 6-dim hand action
Token space: 4 × 4 = 16 possible discrete combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVQ(nn.Module):
    """Single-layer vector quantizer with EMA codebook updates and dead code reset."""

    def __init__(self, dim: int, codebook_size: int = 4, ema_decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.ema_decay = ema_decay
        self.eps = eps

        # Codebook embeddings (not a Parameter — updated via EMA, not gradient)
        self.register_buffer("codebook", torch.randn(codebook_size, dim))
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_weight", torch.randn(codebook_size, dim))
        self.register_buffer("initialized", torch.tensor(False))

    def _init_from_data(self, x: torch.Tensor):
        """Initialize codebook from random data samples (run once on first forward)."""
        n = min(x.shape[0], self.codebook_size)
        idx = torch.randperm(x.shape[0], device=x.device)[:n]
        self.codebook[:n].copy_(x[idx])
        # If fewer samples than codebook entries, fill remaining with noise around mean
        if n < self.codebook_size:
            mean = x.mean(dim=0)
            for i in range(n, self.codebook_size):
                self.codebook[i].copy_(mean + torch.randn_like(mean) * 0.1)
        self.ema_weight.copy_(self.codebook.clone())
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    def _ema_update(self, x: torch.Tensor, indices: torch.Tensor):
        """Update codebook via Exponential Moving Average."""
        # Count assignments per codebook entry
        one_hot = F.one_hot(indices, self.codebook_size).float()  # (B, K)
        counts = one_hot.sum(dim=0)  # (K,)
        sum_embeddings = one_hot.T @ x  # (K, D)

        # EMA update
        self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(sum_embeddings, alpha=1 - self.ema_decay)

        # Normalize
        n = self.ema_count.clamp(min=self.eps).unsqueeze(1)
        self.codebook.copy_(self.ema_weight / n)

    def _reset_dead_codes(self, x: torch.Tensor):
        """Replace unused codebook entries with random encoder outputs."""
        dead_mask = self.ema_count < 1.0  # entries with very few assignments
        n_dead = dead_mask.sum().item()
        if n_dead > 0 and x.shape[0] > 0:
            idx = torch.randperm(x.shape[0], device=x.device)[:n_dead]
            self.codebook[dead_mask] = x[idx[:n_dead]].detach()
            self.ema_weight[dead_mask] = self.codebook[dead_mask].clone()
            self.ema_count[dead_mask] = 1.0

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, D) continuous latent vectors
        Returns:
            quantized: (B, D) quantized vectors (with STE gradient)
            indices:   (B,)   codebook indices
            commit_loss: scalar commitment loss
        """
        # Initialize codebook from first batch of data
        if not self.initialized:
            self._init_from_data(x)

        # L2 distance to codebook: (B, K)
        dists = torch.cdist(x, self.codebook)

        # Nearest neighbor
        indices = dists.argmin(dim=-1)  # (B,)
        quantized = self.codebook[indices]  # (B, D) -- no nn.Embedding, direct index

        # EMA codebook update (training only)
        if self.training:
            self._ema_update(x.detach(), indices)
            self._reset_dead_codes(x.detach())

        # Commitment loss: push encoder output toward codebook entries
        commit_loss = F.mse_loss(x, quantized.detach())

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, indices, commit_loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B,) -> embeddings: (B, D)"""
        return self.codebook[indices]


class SimpleResidualVQ(nn.Module):
    """Multi-layer residual vector quantizer. Each layer quantizes the residual of the previous."""

    def __init__(self, dim: int, num_layers: int = 2, codebook_size: int = 4, ema_decay: float = 0.99):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            SimpleVQ(dim, codebook_size, ema_decay=ema_decay) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, D) continuous latent vectors
        Returns:
            quantized_out: (B, D) sum of quantized residuals
            indices:       (B, num_layers) codebook indices per layer
            total_loss:    scalar total commitment loss
        """
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices = []
        total_loss = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            quantized, indices, commit_loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            total_loss = total_loss + commit_loss

        all_indices = torch.stack(all_indices, dim=-1)  # (B, num_layers)
        return quantized_out, all_indices, total_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, num_layers) codebook indices per layer
        Returns:
            quantized: (B, D) sum of codebook lookups
        """
        quantized = torch.zeros(indices.shape[0], self.layers[0].dim, device=indices.device)
        for i, layer in enumerate(self.layers):
            quantized = quantized + layer.lookup(indices[:, i])
        return quantized


class HandVQVAE(nn.Module):
    """
    VQ-VAE for dexterous hand actions.

    Encodes 6-dim hand action → 32-dim latent → 2-layer residual VQ → decoded 6-dim action.
    Produces 2 discrete tokens (one per VQ layer), each in {0,1,2,3}.
    """

    def __init__(
        self,
        action_dim: int = 6,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_vq_layers: int = 2,
        codebook_size: int = 4,
        commitment_weight: float = 5.0,
    ):
        super().__init__()
        self.commitment_weight = commitment_weight

        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.vq = SimpleResidualVQ(dim=latent_dim, num_layers=num_vq_layers, codebook_size=codebook_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 6) hand action
        Returns:
            x_recon:     (B, 6) reconstructed action
            indices:     (B, 2) VQ token indices
            recon_loss:  scalar MSE reconstruction loss
            commit_loss: scalar commitment loss
            total_loss:  scalar recon_loss + commitment_weight * commit_loss
        """
        z = self.encoder(x)                            # (B, 32)
        z_q, indices, commit_loss = self.vq(z)         # (B, 32), (B, 2), scalar
        x_recon = self.decoder(z_q)                    # (B, 6)

        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + self.commitment_weight * commit_loss

        return x_recon, indices, recon_loss, commit_loss, total_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode actions to discrete token indices. Returns (B, 2)."""
        z = self.encoder(x)
        _, indices, _ = self.vq(z)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices back to actions. indices: (B, 2) -> (B, 6)."""
        z_q = self.vq.decode_indices(indices)
        return self.decoder(z_q)
