"""
Hand-only behavior cloning policy over a frozen Hand-Action VAE.

Decoupled hand branch extracted from the full BC 3.0 policy for diagnostic
purposes. No arm components — only vision + VAE prior -> delta_z correction.

Pipeline:
  img_main, img_extra -> CNNs -> visual_fusion -> visual_feat (feat_dim)
  past_hand_win       -> frozen VAE encoder -> (mu_prior, log_var_prior)
                      -> hand_prior_encoder -> hand_prior_feat (feat_dim)
  [visual_feat, hand_prior_feat] -> hand_delta_z_head -> delta_z
  z_ctrl = mu_prior + delta_z
  hand_action = frozen_vae.decode(z_ctrl)
"""

import importlib.util
import os

import torch
import torch.nn as nn


# ─── VAE loader ────────────────────────────────────────────────────────────────

_BC_HAND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_HAND_ROOT, "..", ".."))
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
    """4-block CNN with GroupNorm. Input (B, 3, 128, 128) -> (B, out_dim)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Hand-only BC policy ─────────────────────────────────────────────────────


class BCHandPolicy(nn.Module):
    """Hand-only behavior cloning policy: vision + VAE prior -> delta_z."""

    def __init__(
        self,
        vae: nn.Module,
        feat_dim: int = 128,
        fusion_dim: int = 256,
        disable_vision: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vae = vae
        self.latent_dim = vae.latent_dim
        self.feat_dim = feat_dim
        self.disable_vision = disable_vision

        self.cnn_main = SimpleCNN(out_dim=feat_dim)
        self.cnn_extra = SimpleCNN(out_dim=feat_dim)
        if disable_vision:
            for p in self.cnn_main.parameters():
                p.requires_grad_(False)
            for p in self.cnn_extra.parameters():
                p.requires_grad_(False)

        self.visual_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        self.hand_prior_encoder = nn.Sequential(
            nn.Linear(2 * self.latent_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        hand_layers = [
            nn.Linear(feat_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            hand_layers.append(nn.Dropout(dropout))
        hand_layers.append(nn.Linear(64, self.latent_dim))
        self.hand_delta_z_head = nn.Sequential(*hand_layers)

        self._zero_init_delta_head()

    def _zero_init_delta_head(self):
        """Zero out the final Linear of the hand delta-z head."""
        last_linear = self.hand_delta_z_head[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def train(self, mode: bool = True):
        """Override so the wrapped frozen VAE never re-enters train mode."""
        super().train(mode)
        self.vae.eval()
        return self

    def encode_visual(
        self,
        img_main: torch.Tensor,
        img_extra: torch.Tensor,
    ) -> torch.Tensor:
        if self.disable_vision:
            batch_size = img_main.shape[0]
            return torch.zeros(
                batch_size,
                self.feat_dim,
                device=img_main.device,
                dtype=img_main.dtype,
            )
        f_main = self.cnn_main(img_main)
        f_extra = self.cnn_extra(img_extra)
        return self.visual_fusion(torch.cat([f_main, f_extra], dim=-1))

    def forward(
        self,
        img_main: torch.Tensor,
        img_extra: torch.Tensor,
        past_hand_win: torch.Tensor,
        zero_delta: bool = False,
    ):
        """Run the hand-only BC policy.

        Args:
            img_main, img_extra: (B, 3, 128, 128) float in [0, 1]
            past_hand_win: (B, 8, 6) past hand actions [a_{t-7}..a_t]
            zero_delta: if True, force delta_z = 0 for the no-correction baseline

        Returns:
            dict with hand_action, hand_no_corr, mu_prior, log_var_prior,
            delta_z, z_ctrl, z_no_corr, visual_feat, hand_prior_feat.
        """
        visual_feat = self.encode_visual(img_main, img_extra)

        with torch.no_grad():
            mu_p, lv_p = self.vae.encode(past_hand_win)
        hand_prior_feat = self.hand_prior_encoder(torch.cat([mu_p, lv_p], dim=-1))

        if zero_delta:
            delta_z = torch.zeros_like(mu_p)
        else:
            hand_latent_input = torch.cat([visual_feat, hand_prior_feat], dim=-1)
            delta_z = self.hand_delta_z_head(hand_latent_input)

        z_ctrl = mu_p + delta_z
        z_no_corr = mu_p

        hand_action = self.vae.decode(z_ctrl)
        hand_no_corr = self.vae.decode(z_no_corr)

        return {
            "hand_action": hand_action,
            "hand_no_corr": hand_no_corr,
            "mu_prior": mu_p,
            "log_var_prior": lv_p,
            "delta_z": delta_z,
            "z_ctrl": z_ctrl,
            "z_no_corr": z_no_corr,
            "visual_feat": visual_feat,
            "hand_prior_feat": hand_prior_feat,
        }


def trainable_params(module: nn.Module):
    """Iterator over module parameters with requires_grad=True."""
    return [p for p in module.parameters() if p.requires_grad]


def strip_vae_state_dict(state_dict: dict) -> dict:
    """Drop keys starting with 'vae.' so checkpoints exclude frozen VAE weights."""
    return {k: v for k, v in state_dict.items() if not k.startswith("vae.")}
