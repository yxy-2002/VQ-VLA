"""
Behavior cloning policy over a frozen Hand-Action VAE.

Architecture (simple v1: CNN + MLP):
  Inputs:
    img_main, img_extra : (B, 3, 128, 128)
    state               : (B, 24)   already standardized
    past_hand_win       : (B, 8, 6)

  Branches:
    cnn_main, cnn_extra (separate weights, identical 4-block CNN, GroupNorm)
    state_encoder       : MLP (24 → 128 → 128)

  Fusion + heads:
    fusion_mlp          : Linear(384, 256) → ReLU → Linear(256, 256) → ReLU
    arm_head            : Linear(256, 128) → ReLU → Linear(128, 6)        [direct]
    delta_mu_head       : Linear(256, 64)  → ReLU → Linear(64, 2)         [zero-init last]
    delta_log_var_head  : Linear(256, 64)  → ReLU → Linear(64, 2)         [zero-init last]

  Hand action path (VAE frozen):
    mu_p, lv_p = vae.encode(past_hand_win)            # no_grad
    mu_corr    = mu_p + delta_mu
    lv_corr    = clamp(lv_p + delta_log_var, -10, 2)  # safety
    z          = vae.reparameterize(mu_corr, lv_corr) # stochastic; grad flows through
    hand       = vae.decode(z)                        # grad flows through

  Output:
    arm_action  : (B, 6)
    hand_action : (B, 6)
    action_pred : (B, 12) = concat(arm, hand)

CRITICAL frozen-VAE hygiene:
  - VAE weights: requires_grad_(False)
  - Override train(mode) so VAE never re-enters train mode
  - Optimizer must filter by requires_grad to skip VAE params
  - delta_mu_head and delta_log_var_head: final Linear is zero-initialized,
    so at step 0 the policy reproduces the no-correction VAE rollout exactly.
"""

import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── VAE loader ────────────────────────────────────────────────────────────────

_BC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))
_VAE_ROOT = os.path.join(_PROJ_ROOT, "vae")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hand_vae_mod = _load_module(os.path.join(_VAE_ROOT, "model/hand_vae.py"), "hand_vae")
_eval_mod = _load_module(os.path.join(_VAE_ROOT, "scripts/eval.py"), "vae_eval")
HandActionVAE = _hand_vae_mod.HandActionVAE
infer_model_args = _eval_mod.infer_model_args


def build_and_freeze_vae(ckpt_path: str) -> nn.Module:
    """Load HandActionVAE from checkpoint, auto-detect arch, freeze all params."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kwargs = infer_model_args(ckpt["model"])
    vae = HandActionVAE(**kwargs)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(
        f"Loaded frozen VAE from {ckpt_path}: "
        f"latent_dim={kwargs['latent_dim']}, hidden_dim={kwargs['hidden_dim']}, "
        f"encoder={kwargs['encoder_type']}, window_size={kwargs['window_size']}"
    )
    return vae


# ─── Vision encoder ────────────────────────────────────────────────────────────


class SimpleCNN(nn.Module):
    """4-block CNN with GroupNorm. Input (B, 3, 128, 128) → (B, out_dim)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),    # 128 → 64
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 64 → 32
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 → 16
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 → 8
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── BC policy ─────────────────────────────────────────────────────────────────


class BCPolicy(nn.Module):
    """Behavior cloning policy with arm head + VAE-corrected hand head."""

    LV_CLAMP_MIN = -10.0
    LV_CLAMP_MAX = 2.0

    def __init__(
        self,
        vae: nn.Module,
        state_dim: int = 24,
        feat_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.vae = vae  # registered as a submodule, but kept frozen
        self.latent_dim = vae.latent_dim

        # Two separate vision encoders
        self.cnn_main = SimpleCNN(out_dim=feat_dim)
        self.cnn_extra = SimpleCNN(out_dim=feat_dim)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feat_dim * 3, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )

        # Heads
        self.arm_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
        self.delta_mu_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.latent_dim),
        )
        self.delta_log_var_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.latent_dim),
        )

        self._zero_init_delta_heads()

    def _zero_init_delta_heads(self):
        """Zero out the FINAL Linear of both delta heads.

        At step 0 the BC delta is exactly (0, 0), so the policy reproduces
        the no-correction VAE rollout — a clean baseline for v1 sanity checks.
        """
        for head in (self.delta_mu_head, self.delta_log_var_head):
            last_linear = head[-1]
            assert isinstance(last_linear, nn.Linear)
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def train(self, mode: bool = True):
        """Override so the wrapped frozen VAE never re-enters train mode."""
        super().train(mode)
        self.vae.eval()
        return self

    # ─── Forward path ──────────────────────────────────────────────────────────

    def encode_obs(
        self,
        img_main: torch.Tensor,
        img_extra: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        f_main = self.cnn_main(img_main)
        f_extra = self.cnn_extra(img_extra)
        f_state = self.state_encoder(state)
        return self.fusion_mlp(torch.cat([f_main, f_extra, f_state], dim=-1))

    def forward(
        self,
        img_main: torch.Tensor,
        img_extra: torch.Tensor,
        state: torch.Tensor,
        past_hand_win: torch.Tensor,
        zero_delta: bool = False,
    ):
        """
        Args:
            img_main, img_extra: (B, 3, 128, 128) float in [0, 1]
            state              : (B, 24) standardized
            past_hand_win      : (B, 8, 6) past hand actions ending at t-1
            zero_delta         : if True, force delta_mu = delta_log_var = 0
                                 (used by evaluate() for the no-correction baseline)

        Returns:
            dict with arm_action, hand_action, action_pred, mu_prior, log_var_prior,
                      delta_mu, delta_log_var, mu_corr, log_var_corr
        """
        h = self.encode_obs(img_main, img_extra, state)

        arm_action = self.arm_head(h)  # (B, 6)

        # VAE prior — no gradient (mu_p, lv_p are constants w.r.t. BC params)
        with torch.no_grad():
            mu_p, lv_p = self.vae.encode(past_hand_win)  # (B, latent), (B, latent)

        if zero_delta:
            delta_mu = torch.zeros_like(mu_p)
            delta_lv = torch.zeros_like(lv_p)
        else:
            delta_mu = self.delta_mu_head(h)
            delta_lv = self.delta_log_var_head(h)

        mu_corr = mu_p + delta_mu
        lv_corr = torch.clamp(lv_p + delta_lv, self.LV_CLAMP_MIN, self.LV_CLAMP_MAX)

        z = self.vae.reparameterize(mu_corr, lv_corr)  # gradients flow through
        hand_action = self.vae.decode(z)               # gradients flow through

        action_pred = torch.cat([arm_action, hand_action], dim=-1)  # (B, 12)

        return {
            "arm_action": arm_action,
            "hand_action": hand_action,
            "action_pred": action_pred,
            "mu_prior": mu_p,
            "log_var_prior": lv_p,
            "delta_mu": delta_mu,
            "delta_log_var": delta_lv,
            "mu_corr": mu_corr,
            "log_var_corr": lv_corr,
        }


def trainable_params(module: nn.Module):
    """Iterator over module parameters with requires_grad=True. Use this when
    constructing the optimizer so VAE params are excluded automatically.
    """
    return [p for p in module.parameters() if p.requires_grad]


def strip_vae_state_dict(state_dict: dict) -> dict:
    """Drop keys starting with 'vae.' so the BC checkpoint doesn't bloat with
    frozen VAE weights. The VAE is reloaded at inference from --vae_ckpt.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith("vae.")}
