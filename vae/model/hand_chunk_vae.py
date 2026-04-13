"""
Chunked hand-action VAE.

Key design choices:
1) Training uses a conditional posterior q(z | history, future_chunk),
   inference uses a learned prior p(z | history).
2) The decoder rolls out the future chunk autoregressively via GRU,
   so chunk boundaries are not generated as unrelated blocks.
3) The decoder predicts raw residual updates in the action space directly,
   without logit-space bounded residual (no sigmoid). This avoids the
   sigmoid flattening issue at basin edges (state≈0 or ≈1), which caused
   stride<H rollout to fail.

Output of each step: next_state = prev_state + raw_delta  (unconstrained).
Training will naturally keep outputs in [0, 1] via MSE loss on real data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────────


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


class MLP(nn.Sequential):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers=1):
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)


# ── Model ────────────────────────────────────────────────────────────────────


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
        free_bits: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.future_horizon = future_horizon
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder_type = encoder_type
        self.num_hidden_layers = num_hidden_layers
        self.free_bits = free_bits

        # History encoder
        if encoder_type == "causal_conv":
            self.history_encoder = CausalConvEncoder(action_dim, hidden_dim)
        else:
            self.history_encoder = MLPEncoder(action_dim, window_size, hidden_dim, num_hidden_layers)

        # Posterior: q(z | history, future_chunk) — used during training
        posterior_in = hidden_dim + action_dim + future_horizon * action_dim
        self.posterior_net = MLP(posterior_in, hidden_dim, hidden_dim, num_hidden_layers)
        self.posterior_mu = nn.Linear(hidden_dim, latent_dim)
        self.posterior_log_var = nn.Linear(hidden_dim, latent_dim)

        # Prior: p(z | history) — used during inference
        self.prior_net = MLP(hidden_dim + action_dim, hidden_dim, hidden_dim, num_hidden_layers)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder: GRU autoregressive
        self.decoder_init = MLP(
            hidden_dim + latent_dim + action_dim, hidden_dim, hidden_dim, num_hidden_layers,
        )
        self.decoder_cell = nn.GRUCell(
            input_size=action_dim + hidden_dim + latent_dim,
            hidden_size=hidden_dim,
        )
        self.decoder_out = MLP(hidden_dim, hidden_dim, action_dim, num_hidden_layers=1)

    # ── Encoder ──────────────────────────────────────────────────────────────

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history)

    def prior(self, history: torch.Tensor):
        current_state = history[:, -1]
        hist_feat = self.encode_history(history)
        stats = self.prior_net(torch.cat([hist_feat, current_state], dim=-1))
        mu = self.prior_mu(stats)
        log_var = self.prior_log_var(stats).clamp(-6.0, 3.0)
        return mu, log_var, hist_feat, current_state

    def posterior(self, history: torch.Tensor, future_chunk: torch.Tensor):
        prior_mu, prior_log_var, hist_feat, current_state = self.prior(history)
        future_flat = future_chunk.reshape(future_chunk.shape[0], -1)
        stats = self.posterior_net(torch.cat([hist_feat, current_state, future_flat], dim=-1))
        mu = self.posterior_mu(stats)
        log_var = self.posterior_log_var(stats).clamp(-6.0, 3.0)
        return {
            "prior_mu": prior_mu,
            "prior_log_var": prior_log_var,
            "post_mu": mu,
            "post_log_var": log_var,
            "hist_feat": hist_feat,
            "current_state": current_state,
        }

    # ── Reparameterize ───────────────────────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    # ── Decoder ──────────────────────────────────────────────────────────────

    def _linear_residual_update(self, prev_state: torch.Tensor, raw_delta: torch.Tensor) -> torch.Tensor:
        """Raw residual in action space (no sigmoid). Outputs are unconstrained."""
        return prev_state + raw_delta

    def decode(
        self,
        z: torch.Tensor,
        hist_feat: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.decoder_init(torch.cat([hist_feat, z, current_state], dim=-1))
        prev_state = current_state
        outputs = []
        for _ in range(self.future_horizon):
            cell_in = torch.cat([prev_state, hist_feat, z], dim=-1)
            hidden = self.decoder_cell(cell_in, hidden)
            raw_delta = self.decoder_out(hidden)
            next_state = self._linear_residual_update(prev_state, raw_delta)
            outputs.append(next_state)
            prev_state = next_state
        return torch.stack(outputs, dim=1)

    # ── KL divergence ────────────────────────────────────────────────────────

    def kl_divergence(
        self,
        post_mu: torch.Tensor,
        post_log_var: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_log_var: torch.Tensor,
    ) -> torch.Tensor:
        post_var = torch.exp(post_log_var)
        prior_var = torch.exp(prior_log_var)
        kl = prior_log_var - post_log_var
        kl = kl + (post_var + (post_mu - prior_mu).pow(2)) / prior_var
        kl = kl - 1.0
        kl_per_dim = 0.5 * kl.mean(dim=0)
        if self.free_bits > 0:
            floor = torch.full_like(kl_per_dim, self.free_bits)
            kl_per_dim = torch.maximum(kl_per_dim, floor)
        return kl_per_dim.mean()

    # ── Forward / predict ────────────────────────────────────────────────────

    def forward(self, history: torch.Tensor, target: torch.Tensor):
        enc = self.posterior(history, target)
        z = self.reparameterize(enc["post_mu"], enc["post_log_var"])
        pred = self.decode(z, enc["hist_feat"], enc["current_state"])

        recon_loss = F.mse_loss(pred, target)
        kl_loss = self.kl_divergence(
            enc["post_mu"], enc["post_log_var"],
            enc["prior_mu"], enc["prior_log_var"],
        )
        total_loss = recon_loss + self.beta * kl_loss
        return pred, recon_loss, kl_loss, total_loss, enc

    @torch.no_grad()
    def predict(self, history: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        prior_mu, prior_log_var, hist_feat, current_state = self.prior(history)
        z = prior_mu if deterministic else self.reparameterize(prior_mu, prior_log_var)
        return self.decode(z, hist_feat, current_state)
