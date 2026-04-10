"""
Conditional latent flow-matching model for hand future chunks.

History window -> conditional flow in a 2D latent space -> decoder reconstructs a
future chunk. The future chunk encoder is deterministic so the latent space stays
easy to visualize.
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


class HandActionChunkFlow(nn.Module):
    def __init__(
        self,
        action_dim: int = 6,
        window_size: int = 8,
        future_horizon: int = 8,
        hidden_dim: int = 256,
        latent_dim: int = 2,
        num_hidden_layers: int = 1,
        time_embed_dim: int = 32,
        recon_weight: float = 1.0,
        flow_weight: float = 1.0,
        latent_reg_weight: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.future_horizon = future_horizon
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers
        self.time_embed_dim = time_embed_dim
        self.recon_weight = recon_weight
        self.flow_weight = flow_weight
        self.latent_reg_weight = latent_reg_weight

        self.history_encoder = MLP(
            in_dim=window_size * action_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.future_encoder = MLP(
            in_dim=future_horizon * action_dim + hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.decoder = MLP(
            in_dim=latent_dim + hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=future_horizon * action_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.flow_net = MLP(
            in_dim=latent_dim + hidden_dim + time_embed_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            num_hidden_layers=num_hidden_layers + 1,
        )

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history.reshape(history.shape[0], -1))

    def encode_future(self, future_chunk: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if cond is None:
            cond = torch.zeros(
                future_chunk.shape[0],
                self.hidden_dim,
                device=future_chunk.device,
                dtype=future_chunk.dtype,
            )
        future_flat = future_chunk.reshape(future_chunk.shape[0], -1)
        return self.future_encoder(torch.cat([future_flat, cond], dim=-1))

    def decode(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if cond is None:
            cond = torch.zeros(z.shape[0], self.hidden_dim, device=z.device, dtype=z.dtype)
        out = torch.sigmoid(self.decoder(torch.cat([z, cond], dim=-1)))
        return out.view(z.shape[0], self.future_horizon, self.action_dim)

    def flow_vector_field(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        inp = torch.cat([z_t, cond, t_embed], dim=-1)
        return self.flow_net(inp)

    def latent_reg_loss(self, z: torch.Tensor) -> torch.Tensor:
        mean = z.mean(dim=0)
        std = z.std(dim=0) + 1e-6
        return mean.pow(2).mean() + (std - 1.0).pow(2).mean()

    def forward(self, history: torch.Tensor, target: torch.Tensor):
        cond = self.encode_history(history)
        z1 = self.encode_future(target, cond=cond)
        recon = self.decode(z1, cond=cond)
        recon_loss = F.mse_loss(recon, target)

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
        return {
            "recon": recon,
            "z1": z1,
            "cond": cond,
            "recon_loss": recon_loss,
            "flow_loss": flow_loss,
            "latent_reg": latent_reg,
            "total_loss": total_loss,
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

    def predict_with_cond(
        self,
        cond: torch.Tensor,
        num_integration_steps: int = 24,
        z0: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        z = self.sample_latent_with_cond(cond, num_integration_steps=num_integration_steps, z0=z0)
        chunk = self.decode(z, cond=cond)
        if return_latent:
            return chunk, z
        return chunk

    @torch.no_grad()
    def sample_latent(
        self,
        history: torch.Tensor,
        num_integration_steps: int = 24,
        z0: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ):
        if cond is None:
            cond = self.encode_history(history)
        return self.sample_latent_with_cond(cond, num_integration_steps=num_integration_steps, z0=z0)

    @torch.no_grad()
    def predict(
        self,
        history: torch.Tensor,
        num_integration_steps: int = 24,
        z0: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        cond = self.encode_history(history)
        return self.predict_with_cond(
            cond,
            num_integration_steps=num_integration_steps,
            z0=z0,
            return_latent=return_latent,
        )
