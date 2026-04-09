"""
Hand Action VAE: encode past 8 frames of hand actions → latent (μ, σ) → predict t+1 action.

Input:  (B, 8, 6)  past 8 frames of 6-dim hand action
Output: (B, 6)     predicted next-frame hand action

Supports two encoder architectures:
  - "mlp":        flatten (8,6)→48 → MLP
  - "causal_conv": 1D causal convolutions preserving temporal order
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding: output only depends on past inputs."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))  # left-pad only
        return self.conv(x)


class CausalConvEncoder(nn.Module):
    """
    1D Causal Conv encoder: (B, T, 6) → (B, hidden_dim).
    Takes the last timestep's hidden state as the summary.
    """

    def __init__(self, action_dim=6, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(action_dim, 64, kernel_size=3),
            nn.SiLU(),
            CausalConv1d(64, 128, kernel_size=3),
            nn.SiLU(),
            CausalConv1d(128, hidden_dim, kernel_size=3),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (B, T, action_dim) → transpose to (B, action_dim, T) for Conv1d
        h = self.net(x.transpose(1, 2))  # (B, hidden_dim, T)
        return h[:, :, -1]               # (B, hidden_dim) — last timestep


class MLPEncoder(nn.Sequential):
    """MLP encoder: flatten (B, T, 6) → (B, T*6) → MLP → (B, hidden_dim).

    `num_hidden_layers` is the number of (hidden_dim → hidden_dim) blocks added
    after the input projection. Default = 1 reproduces the original 2-Linear
    structure exactly so old checkpoints still load.
    """

    def __init__(self, action_dim=6, window_size=8, hidden_dim=256, num_hidden_layers=1):
        layers = [nn.Linear(action_dim * window_size, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        super().__init__(*layers)

    def forward(self, x):
        # x: (B, T, action_dim)
        return super().forward(x.reshape(x.shape[0], -1))  # (B, hidden_dim)


class HandActionVAE(nn.Module):
    """
    VAE for hand action prediction.

    Args:
        encoder_type: "mlp" or "causal_conv"
    """

    def __init__(
        self,
        action_dim: int = 6,
        window_size: int = 8,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        beta: float = 0.01,
        encoder_type: str = "mlp",
        num_hidden_layers: int = 1,
        recon_aux_weight: float = 0.0,
        free_bits: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder_type = encoder_type
        self.num_hidden_layers = num_hidden_layers
        # If > 0, decoder gains an auxiliary head that reconstructs the flat input window.
        # This forces the encoder to preserve input information (raises R²).
        self.recon_aux_weight = recon_aux_weight
        # If > 0, applies per-dim KL floor in nats — prevents low-info dims from collapsing.
        self.free_bits = free_bits

        # Encoder
        if encoder_type == "causal_conv":
            self.encoder = CausalConvEncoder(action_dim, hidden_dim)
        else:
            self.encoder = MLPEncoder(action_dim, window_size, hidden_dim, num_hidden_layers)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent → hidden → [hidden → hidden] × num_hidden_layers → action_dim.
        # Default num_hidden_layers=1 matches the original 3-Linear structure (decoder.0/2/4).
        decoder_layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_layers):
            decoder_layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        decoder_layers.append(nn.Linear(hidden_dim, action_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Auxiliary input-reconstruction head (only built when needed, so old
        # checkpoints without this head still load cleanly).
        if recon_aux_weight > 0:
            self.aux_recon_head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, action_dim * window_size),
            )
        else:
            self.aux_recon_head = None

    def encode(self, x: torch.Tensor):
        """
        Args:
            x: (B, window_size, action_dim)
        Returns:
            mu: (B, latent_dim), log_var: (B, latent_dim)
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        pred = self.decode(z)

        recon_loss = F.mse_loss(pred, target)

        # KL with optional free bits — average over batch first, then apply
        # the per-dim floor, then average over dims (matches the original
        # mean-over-(B, D) magnitude when free_bits=0).
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())   # (B, latent_dim)
        kl_per_dim_avg = kl_per_dim.mean(dim=0)                          # (latent_dim,)
        if self.free_bits > 0:
            floor = torch.full_like(kl_per_dim_avg, self.free_bits)
            kl_per_dim_avg = torch.maximum(kl_per_dim_avg, floor)
        kl_loss = kl_per_dim_avg.mean()

        total_loss = recon_loss + self.beta * kl_loss

        # Auxiliary input-reconstruction loss: forces encoder to preserve
        # input information in latent (raises R²).
        if self.aux_recon_head is not None:
            x_flat = x.reshape(x.shape[0], -1)
            aux_pred = self.aux_recon_head(z)
            aux_recon_loss = F.mse_loss(aux_pred, x_flat)
            total_loss = total_loss + self.recon_aux_weight * aux_recon_loss

        return pred, recon_loss, kl_loss, total_loss, mu, log_var

    @torch.no_grad()
    def predict(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mu, log_var = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, log_var)
        return self.decode(z)
