"""
Behavior cloning policy over a frozen single-step Hand-Action VAE.

Architecture:
  - arm_head MLP → arm_action (B, 6)
  - hand delta_z → frozen VAE.decode(z_ctrl) → hand_action (B, 6)
"""

import importlib.util
import os

import torch
import torch.nn as nn


# ─── Module loaders ──────────────────────────────────────────────────────────

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


# ─── VAE loader ──────────────────────────────────────────────────────────────


def build_and_freeze_vae(ckpt_path: str):
    """Load single-step VAE from checkpoint and freeze all params."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kwargs = infer_model_args(ckpt["model"])
    vae = HandActionVAE(**kwargs)
    print(
        f"Loaded frozen VAE from {ckpt_path}: "
        f"latent_dim={kwargs['latent_dim']}, hidden_dim={kwargs['hidden_dim']}, "
        f"encoder={kwargs['encoder_type']}, window_size={kwargs['window_size']}"
    )

    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ─── Vision encoder ──────────────────────────────────────────────────────────


class SimpleCNN(nn.Module):
    """4-block CNN with GroupNorm. Input (B, 3, 128, 128) → (B, out_dim)."""

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


# ─── BC policy ───────────────────────────────────────────────────────────────


class BCPolicy(nn.Module):
    """Single-step BC policy with weakly-coupled arm and hand branches.

    Predicts one next action (B, 12) per call.
    """

    def __init__(
        self,
        vae: nn.Module,
        arm_state_dim: int = 6,
        feat_dim: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vae = vae
        self.latent_dim = vae.latent_dim
        self.arm_state_dim = arm_state_dim
        self.feat_dim = feat_dim

        # Shared encoders
        self.cnn_main = SimpleCNN(out_dim=feat_dim)
        self.cnn_extra = SimpleCNN(out_dim=feat_dim)

        self.visual_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        self.arm_state_encoder = nn.Sequential(
            nn.Linear(arm_state_dim, feat_dim),
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

        # ── Arm branch: MLP head ──
        arm_layers = [
            nn.Linear(feat_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            arm_layers.append(nn.Dropout(dropout))
        arm_layers.extend([
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
        ])
        if dropout > 0:
            arm_layers.append(nn.Dropout(dropout))
        arm_layers.append(nn.Linear(128, 6))
        self.arm_head = nn.Sequential(*arm_layers)

        # ── Hand branch ──
        hand_layers = [
            nn.Linear(feat_dim * 3, fusion_dim),
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
        f_main = self.cnn_main(img_main)
        f_extra = self.cnn_extra(img_extra)
        return self.visual_fusion(torch.cat([f_main, f_extra], dim=-1))

    def forward(
        self,
        img_main: torch.Tensor,
        img_extra: torch.Tensor,
        state: torch.Tensor,
        past_hand_win: torch.Tensor,
        zero_delta: bool = False,
    ):
        """Run the BC policy.

        Returns dict with arm_action (B,6), hand_action (B,6), hand_no_corr (B,6),
        action_pred (B,12), mu_prior, log_var_prior, delta_z, z_ctrl, z_no_corr.
        """
        visual_feat = self.encode_visual(img_main, img_extra)
        arm_state = state[..., :self.arm_state_dim]
        arm_state_feat = self.arm_state_encoder(arm_state)

        arm_action = self.arm_head(torch.cat([visual_feat, arm_state_feat], dim=-1))

        with torch.no_grad():
            mu_p, lv_p = self.vae.encode(past_hand_win)
        hand_prior_feat = self.hand_prior_encoder(torch.cat([mu_p, lv_p], dim=-1))

        if zero_delta:
            delta_z = torch.zeros_like(mu_p)
        else:
            hand_input = torch.cat([visual_feat, hand_prior_feat, arm_state_feat], dim=-1)
            delta_z = self.hand_delta_z_head(hand_input)

        z_ctrl = mu_p + delta_z
        z_no_corr = mu_p

        hand_action = self.vae.decode(z_ctrl)
        hand_no_corr = self.vae.decode(z_no_corr)
        action_pred = torch.cat([arm_action, hand_action], dim=-1)

        return {
            "arm_action": arm_action,
            "hand_action": hand_action,
            "hand_no_corr": hand_no_corr,
            "action_pred": action_pred,
            "mu_prior": mu_p,
            "log_var_prior": lv_p,
            "delta_z": delta_z,
            "z_ctrl": z_ctrl,
            "z_no_corr": z_no_corr,
            "visual_feat": visual_feat,
            "arm_state_feat": arm_state_feat,
            "hand_prior_feat": hand_prior_feat,
        }


def trainable_params(module: nn.Module):
    """Iterator over module parameters with requires_grad=True."""
    return [p for p in module.parameters() if p.requires_grad]


def strip_vae_state_dict(state_dict: dict) -> dict:
    """Drop keys starting with 'vae.' so BC checkpoints exclude frozen VAE weights."""
    return {k: v for k, v in state_dict.items() if not k.startswith("vae.")}
