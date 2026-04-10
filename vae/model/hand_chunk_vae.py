"""
Chunked hand-action VAE.

Encodes a history window of hand states and decodes a fixed-length future chunk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConvEncoder(nn.Module):
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
        h = self.net(x.transpose(1, 2))
        return h[:, :, -1]


class MLPEncoder(nn.Sequential):
    def __init__(self, action_dim=6, window_size=8, hidden_dim=256, num_hidden_layers=1):
        layers = [nn.Linear(action_dim * window_size, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        super().__init__(*layers)

    def forward(self, x):
        return super().forward(x.reshape(x.shape[0], -1))


class HandActionChunkVAE(nn.Module):
    def __init__(
        self,
        action_dim: int = 6,
        window_size: int = 8,
        future_horizon: int = 8,
        hidden_dim: int = 256,
        latent_dim: int = 2,
        beta: float = 0.001,
        encoder_type: str = "mlp",
        num_hidden_layers: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.future_horizon = future_horizon
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder_type = encoder_type
        self.num_hidden_layers = num_hidden_layers

        if encoder_type == "causal_conv":
            self.encoder = CausalConvEncoder(action_dim, hidden_dim)
        else:
            self.encoder = MLPEncoder(action_dim, window_size, hidden_dim, num_hidden_layers)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        decoder_layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_layers):
            decoder_layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        decoder_layers.append(nn.Linear(hidden_dim, future_horizon * action_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.view(z.shape[0], self.future_horizon, self.action_dim)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        pred = self.decode(z)

        recon_loss = F.mse_loss(pred, target)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return pred, recon_loss, kl_loss, total_loss, mu, log_var

    @torch.no_grad()
    def predict(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mu, log_var = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, log_var)
        return self.decode(z)
