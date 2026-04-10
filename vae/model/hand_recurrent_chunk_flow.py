"""
Recurrent latent flow prior for hand future chunks.

This model keeps a hidden state across chunk rollouts so it can represent
"internal phase" even when the visible hand state still looks almost identical.
The sampled latent stays 2D for later policy biasing / visualization.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers=1):
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class HandActionRecurrentChunkFlow(nn.Module):
    def __init__(
        self,
        action_dim: int = 6,
        window_size: int = 8,
        future_horizon: int = 12,
        hidden_dim: int = 256,
        recurrent_dim: int = 128,
        latent_dim: int = 2,
        monotonic_start_idx: int = 2,
        num_hidden_layers: int = 1,
        time_embed_dim: int = 32,
        recon_weight: float = 1.0,
        flow_weight: float = 1.0,
        latent_reg_weight: float = 0.02,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.future_horizon = future_horizon
        self.hidden_dim = hidden_dim
        self.recurrent_dim = recurrent_dim
        self.latent_dim = latent_dim
        self.monotonic_start_idx = monotonic_start_idx
        self.num_hidden_layers = num_hidden_layers
        self.time_embed_dim = time_embed_dim
        self.recon_weight = recon_weight
        self.flow_weight = flow_weight
        self.latent_reg_weight = latent_reg_weight

        self.history_rnn = nn.GRU(
            input_size=action_dim,
            hidden_size=recurrent_dim,
            num_layers=1,
            batch_first=True,
        )
        self.cond_proj = MLP(
            in_dim=recurrent_dim + action_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.future_encoder = MLP(
            in_dim=future_horizon * action_dim + hidden_dim + action_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.decoder = MLP(
            in_dim=latent_dim + hidden_dim + action_dim,
            hidden_dim=hidden_dim,
            out_dim=future_horizon * action_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.summary_encoder = MLP(
            in_dim=future_horizon * action_dim + action_dim,
            hidden_dim=hidden_dim,
            out_dim=recurrent_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.transition = nn.GRUCell(recurrent_dim, recurrent_dim)

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.flow_net = MLP(
            in_dim=latent_dim + hidden_dim + time_embed_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            num_hidden_layers=num_hidden_layers + 1,
        )

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history_rnn(history)
        return hidden[-1]

    def make_condition(self, hidden: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        return self.cond_proj(torch.cat([hidden, current_state], dim=-1))

    def encode_future(
        self,
        future_chunk: torch.Tensor,
        cond: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        flat = future_chunk.reshape(future_chunk.shape[0], -1)
        return self.future_encoder(torch.cat([flat, cond, current_state], dim=-1))

    def flow_vector_field(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        return self.flow_net(torch.cat([z_t, cond, t_embed], dim=-1))

    def latent_reg_loss(self, z: torch.Tensor) -> torch.Tensor:
        mean = z.mean(dim=0)
        std = z.std(dim=0) + 1e-6
        return mean.pow(2).mean() + (std - 1.0).pow(2).mean()

    def decode(self, z: torch.Tensor, cond: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(torch.cat([z, cond, current_state], dim=-1))
        raw = raw.view(z.shape[0], self.future_horizon, self.action_dim)

        free_dim = min(self.monotonic_start_idx, self.action_dim)
        mono_dim = self.action_dim - free_dim
        outputs = []
        eps = 1e-4

        if free_dim > 0:
            start_free = current_state[:, :free_dim].clamp(eps, 1.0 - eps)
            start_logit = torch.log(start_free) - torch.log1p(-start_free)
            free = torch.sigmoid(start_logit.unsqueeze(1) + raw[:, :, :free_dim])
            outputs.append(free)

        if mono_dim > 0:
            start_mono = current_state[:, free_dim:].clamp(0.0, 1.0).unsqueeze(1)
            deltas = F.softplus(raw[:, :, free_dim:])
            cum_delta = torch.cumsum(deltas, dim=1)
            mono = 1.0 - (1.0 - start_mono) * torch.exp(-cum_delta)
            outputs.append(mono.clamp(0.0, 1.0))

        return torch.cat(outputs, dim=-1)

    def summarize_chunk(self, chunk: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        flat = chunk.reshape(chunk.shape[0], -1)
        return self.summary_encoder(torch.cat([flat, current_state], dim=-1))

    def transition_hidden(
        self,
        hidden: torch.Tensor,
        chunk: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        summary = self.summarize_chunk(chunk, current_state)
        return self.transition(summary, hidden)

    def forward_step(
        self,
        hidden: torch.Tensor,
        current_state: torch.Tensor,
        target_chunk: torch.Tensor,
    ):
        cond = self.make_condition(hidden, current_state)
        z1 = self.encode_future(target_chunk, cond, current_state)
        recon = self.decode(z1, cond, current_state)
        recon_loss = F.mse_loss(recon, target_chunk)

        z0 = torch.randn_like(z1)
        t = torch.rand(z1.shape[0], 1, device=z1.device, dtype=z1.dtype)
        z_t = (1.0 - t) * z0 + t * z1
        target_v = z1 - z0
        pred_v = self.flow_vector_field(z_t, t, cond)
        flow_loss = F.mse_loss(pred_v, target_v)

        latent_reg = self.latent_reg_loss(z1)
        total_loss = (
            self.recon_weight * recon_loss
            + self.flow_weight * flow_loss
            + self.latent_reg_weight * latent_reg
        )

        next_hidden = self.transition_hidden(hidden, recon, current_state)
        next_state = recon[:, -1]
        return {
            "cond": cond,
            "z1": z1,
            "recon": recon,
            "recon_loss": recon_loss,
            "flow_loss": flow_loss,
            "latent_reg": latent_reg,
            "total_loss": total_loss,
            "next_hidden": next_hidden,
            "next_state": next_state,
        }

    def sample_latent_with_cond(
        self,
        cond: torch.Tensor,
        num_integration_steps: int = 24,
        z0: torch.Tensor | None = None,
    ):
        if z0 is None:
            z = torch.randn(cond.shape[0], self.latent_dim, device=cond.device, dtype=cond.dtype)
        else:
            z = z0.clone()

        dt = 1.0 / num_integration_steps
        for step in range(num_integration_steps):
            t = torch.full(
                (cond.shape[0], 1),
                fill_value=(step + 0.5) * dt,
                device=cond.device,
                dtype=cond.dtype,
            )
            z = z + dt * self.flow_vector_field(z, t, cond)
        return z

    def sample_chunk(
        self,
        hidden: torch.Tensor,
        current_state: torch.Tensor,
        num_integration_steps: int = 24,
        z0: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        cond = self.make_condition(hidden, current_state)
        z = self.sample_latent_with_cond(cond, num_integration_steps=num_integration_steps, z0=z0)
        chunk = self.decode(z, cond, current_state)
        next_hidden = self.transition_hidden(hidden, chunk, current_state)
        next_state = chunk[:, -1]
        if return_latent:
            return chunk, z, next_hidden, next_state
        return chunk, next_hidden, next_state
