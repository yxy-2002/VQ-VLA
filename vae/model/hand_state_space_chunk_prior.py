"""
Stochastic recurrent state-space prior for hand future chunks.

The model keeps:
- a deterministic hidden state `h_t` for rollout memory
- a 2D stochastic latent state `z_t` that persists across chunks

This is meant to separate:
- internal phase / readiness (hidden + stochastic latent)
- controllable low-dimensional latent used later by a visual policy
"""

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


class HandActionStateSpaceChunkPrior(nn.Module):
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
        latent_transition_scale: float = 0.5,
        hidden_update_rate: float = 0.35,
        monotonic_output_mode: str = "exp",
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
        self.latent_transition_scale = latent_transition_scale
        self.hidden_update_rate = hidden_update_rate
        self.monotonic_output_mode = monotonic_output_mode
        if monotonic_output_mode not in {"exp", "plateau"}:
            raise ValueError(f"Unsupported monotonic_output_mode: {monotonic_output_mode}")

        self.history_rnn = nn.GRU(
            input_size=action_dim,
            hidden_size=recurrent_dim,
            num_layers=1,
            batch_first=True,
        )
        cond_in_dim = recurrent_dim + action_dim + latent_dim
        self.cond_proj = MLP(
            in_dim=cond_in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.prior_net = MLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim * 2,
            num_hidden_layers=num_hidden_layers,
        )
        self.posterior_net = MLP(
            in_dim=future_horizon * action_dim + hidden_dim + action_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim * 2,
            num_hidden_layers=num_hidden_layers,
        )
        mono_dim = max(0, action_dim - min(monotonic_start_idx, action_dim))
        decoder_out_dim = future_horizon * action_dim
        if monotonic_output_mode == "plateau" and mono_dim > 0:
            decoder_out_dim += mono_dim
        self.decoder = MLP(
            in_dim=latent_dim + hidden_dim + action_dim,
            hidden_dim=hidden_dim,
            out_dim=decoder_out_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.summary_encoder = MLP(
            in_dim=future_horizon * action_dim + action_dim + latent_dim,
            hidden_dim=hidden_dim,
            out_dim=recurrent_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self.transition = nn.GRUCell(recurrent_dim, recurrent_dim)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history_rnn(history)
        return hidden[-1]

    def make_condition(
        self,
        hidden: torch.Tensor,
        current_state: torch.Tensor,
        prev_z: torch.Tensor,
    ) -> torch.Tensor:
        return self.cond_proj(torch.cat([hidden, current_state, prev_z], dim=-1))

    def prior(self, cond: torch.Tensor, prev_z: torch.Tensor):
        stats = self.prior_net(cond)
        delta_mu, log_var = stats.chunk(2, dim=-1)
        mu = prev_z + self.latent_transition_scale * torch.tanh(delta_mu)
        log_var = log_var.clamp(-6.0, 3.0)
        return mu, log_var

    def posterior(
        self,
        future_chunk: torch.Tensor,
        cond: torch.Tensor,
        current_state: torch.Tensor,
        prev_z: torch.Tensor,
    ):
        flat = future_chunk.reshape(future_chunk.shape[0], -1)
        stats = self.posterior_net(torch.cat([flat, cond, current_state], dim=-1))
        delta_mu, log_var = stats.chunk(2, dim=-1)
        mu = prev_z + self.latent_transition_scale * torch.tanh(delta_mu)
        log_var = log_var.clamp(-6.0, 3.0)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(torch.cat([z, cond, current_state], dim=-1))
        free_dim = min(self.monotonic_start_idx, self.action_dim)
        mono_dim = self.action_dim - free_dim
        target_raw = None
        if self.monotonic_output_mode == "plateau" and mono_dim > 0:
            target_raw = raw[:, self.future_horizon * self.action_dim :]
            raw = raw[:, : self.future_horizon * self.action_dim]
        raw = raw.view(z.shape[0], self.future_horizon, self.action_dim)

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
            if self.monotonic_output_mode == "plateau":
                mono_target = current_state[:, free_dim:] + (1.0 - current_state[:, free_dim:]) * torch.sigmoid(target_raw)
                progress = 1.0 - torch.exp(-cum_delta)
                mono = start_mono + (mono_target.unsqueeze(1) - start_mono) * progress
            else:
                mono = 1.0 - (1.0 - start_mono) * torch.exp(-cum_delta)
            outputs.append(mono.clamp(0.0, 1.0))

        return torch.cat(outputs, dim=-1)

    def summarize_chunk(
        self,
        chunk: torch.Tensor,
        current_state: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        flat = chunk.reshape(chunk.shape[0], -1)
        return self.summary_encoder(torch.cat([flat, current_state, z], dim=-1))

    def transition_hidden(
        self,
        hidden: torch.Tensor,
        chunk: torch.Tensor,
        current_state: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        summary = self.summarize_chunk(chunk, current_state, z)
        updated = self.transition(summary, hidden)
        return hidden + self.hidden_update_rate * (updated - hidden)

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
        return 0.5 * kl.sum(dim=-1).mean()

    def forward_step(
        self,
        hidden: torch.Tensor,
        current_state: torch.Tensor,
        prev_z: torch.Tensor,
        target_chunk: torch.Tensor,
    ):
        cond = self.make_condition(hidden, current_state, prev_z)
        prior_mu, prior_log_var = self.prior(cond, prev_z)
        post_mu, post_log_var = self.posterior(target_chunk, cond, current_state, prev_z)
        z = self.reparameterize(post_mu, post_log_var)
        recon = self.decode(z, cond, current_state)
        next_hidden = self.transition_hidden(hidden, recon, current_state, z)
        next_state = recon[:, -1]

        return {
            "cond": cond,
            "prior_mu": prior_mu,
            "prior_log_var": prior_log_var,
            "post_mu": post_mu,
            "post_log_var": post_log_var,
            "z": z,
            "recon": recon,
            "next_hidden": next_hidden,
            "next_state": next_state,
        }

    @torch.no_grad()
    def sample_step(
        self,
        hidden: torch.Tensor,
        current_state: torch.Tensor,
        prev_z: torch.Tensor,
        deterministic: bool = False,
        return_stats: bool = False,
    ):
        cond = self.make_condition(hidden, current_state, prev_z)
        prior_mu, prior_log_var = self.prior(cond, prev_z)
        if deterministic:
            z = prior_mu
        else:
            z = self.reparameterize(prior_mu, prior_log_var)
        chunk = self.decode(z, cond, current_state)
        next_hidden = self.transition_hidden(hidden, chunk, current_state, z)
        next_state = chunk[:, -1]
        if return_stats:
            return chunk, z, next_hidden, next_state, prior_mu, prior_log_var
        return chunk, z, next_hidden, next_state
