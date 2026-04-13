"""
Behavior cloning policy over a frozen single-step Hand-Action VAE.

Visual encoder: shared pretrained ResNet-18 (HuggingFace) per view → (B, 512) each.
Two views concatenated into visual_feat (B, 1024). No fusion MLP.

Arm branch: MLP → arm_action (B, 6)
Hand branch: MLP → delta_z → frozen VAE.decode(mu_prior + delta_z) → hand_action (B, 6)
"""

import importlib.util
import os

import torch
import torch.nn as nn

from transformers import ResNetModel


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


# ─── Visual backbone ─────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DEFAULT_RESNET_PATH = os.path.join(_PROJ_ROOT, "pretrained_model/resnet-18")


class ResNet18Backbone(nn.Module):
    """Shared pretrained ResNet-18 backbone (HF format).

    Input: pixel values (B, 3, H, W) in [0, 1] range (will be ImageNet-normalized here).
    Output: (B, 512) pooled features.
    """

    def __init__(self, pretrained_path: str = _DEFAULT_RESNET_PATH, freeze: bool = False):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained(pretrained_path)
        self.freeze_backbone = freeze
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        if freeze:
            for p in self.resnet.parameters():
                p.requires_grad_(False)
            self.resnet.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.resnet.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1]; normalize with ImageNet stats
        x = (x - self._mean) / self._std
        out = self.resnet(pixel_values=x)
        return out.pooler_output.flatten(1)  # (B, 512)


# ─── State / prior encoder factory ───────────────────────────────────────────


def build_state_encoder(in_dim: int, encoder_type: str):
    """Return (module, out_dim) for the given encoder variant.

    Variants:
      - "mlp": 2-layer MLP in_dim → 128 → 128 (ReLU)          [out_dim=128]
      - "linear64": single Linear in_dim → 64 + ReLU           [out_dim=64]
      - "raw": Identity passthrough                            [out_dim=in_dim]
    """
    if encoder_type == "mlp":
        mod = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        return mod, 128
    if encoder_type == "linear64":
        mod = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(inplace=True))
        return mod, 64
    if encoder_type == "raw":
        return nn.Identity(), in_dim
    raise ValueError(f"Unknown state_encoder_type: {encoder_type!r}")


# ─── BC policy ───────────────────────────────────────────────────────────────


class BCPolicy(nn.Module):
    """Single-step BC policy with weakly-coupled arm and hand branches.

    Visual features: shared ResNet-18 on each view → (B, 512); two views concatenated
    → visual_feat (B, 1024). No fusion MLP.
    """

    VISUAL_DIM = 1024  # 2 views x 512

    def __init__(
        self,
        vae: nn.Module,
        backbone: nn.Module,
        arm_state_dim: int = 6,
        state_encoder_type: str = "mlp",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vae = vae
        self.backbone = backbone
        self.latent_dim = vae.latent_dim
        self.arm_state_dim = arm_state_dim
        self.state_encoder_type = state_encoder_type

        # Arm / hand-prior encoders (variant-driven)
        self.arm_state_encoder, arm_feat_dim = build_state_encoder(
            arm_state_dim, state_encoder_type,
        )
        hand_prior_in = 2 * self.latent_dim  # [mu, log_var]
        self.hand_prior_encoder, hand_prior_feat_dim = build_state_encoder(
            hand_prior_in, state_encoder_type,
        )
        self.arm_feat_dim = arm_feat_dim
        self.hand_prior_feat_dim = hand_prior_feat_dim

        # ── Arm head ──  input: [visual_feat, arm_state_feat]
        arm_in = self.VISUAL_DIM + arm_feat_dim
        arm_layers = [
            nn.Linear(arm_in, 512),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            arm_layers.append(nn.Dropout(dropout))
        arm_layers.extend([
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        ])
        if dropout > 0:
            arm_layers.append(nn.Dropout(dropout))
        arm_layers.append(nn.Linear(256, 6))
        self.arm_head = nn.Sequential(*arm_layers)

        # ── Hand delta_z head ──  input: [visual_feat, hand_prior_feat, arm_state_feat]
        hand_in = self.VISUAL_DIM + hand_prior_feat_dim + arm_feat_dim
        hand_layers = [
            nn.Linear(hand_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            hand_layers.append(nn.Dropout(dropout))
        hand_layers.append(nn.Linear(64, self.latent_dim))
        self.hand_delta_z_head = nn.Sequential(*hand_layers)

        self._zero_init_delta_head()

    def _zero_init_delta_head(self):
        last_linear = self.hand_delta_z_head[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def train(self, mode: bool = True):
        """Keep frozen VAE (and optionally frozen backbone) in eval mode."""
        super().train(mode)
        self.vae.eval()
        if getattr(self.backbone, "freeze_backbone", False):
            self.backbone.resnet.eval()
        return self

    def encode_visual(
        self,
        img_main: torch.Tensor,
        img_extra: torch.Tensor,
    ) -> torch.Tensor:
        f_main = self.backbone(img_main)    # (B, 512)
        f_extra = self.backbone(img_extra)  # (B, 512)
        return torch.cat([f_main, f_extra], dim=-1)  # (B, 1024)

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
        visual_feat = self.encode_visual(img_main, img_extra)   # (B, 1024)
        arm_state = state[..., :self.arm_state_dim]
        arm_state_feat = self.arm_state_encoder(arm_state)

        arm_action = self.arm_head(torch.cat([visual_feat, arm_state_feat], dim=-1))

        with torch.no_grad():
            mu_p, lv_p = self.vae.encode(past_hand_win)
        hand_prior_feat = self.hand_prior_encoder(torch.cat([mu_p, lv_p], dim=-1))

        if zero_delta:
            delta_z = torch.zeros_like(mu_p)
        else:
            hand_input = torch.cat(
                [visual_feat, hand_prior_feat, arm_state_feat], dim=-1,
            )
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
    return [p for p in module.parameters() if p.requires_grad]


def strip_vae_state_dict(state_dict: dict) -> dict:
    """Drop keys starting with 'vae.' so BC checkpoints exclude frozen VAE weights."""
    return {k: v for k, v in state_dict.items() if not k.startswith("vae.")}
