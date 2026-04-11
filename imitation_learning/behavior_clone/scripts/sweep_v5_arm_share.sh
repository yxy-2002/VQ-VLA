#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yxy/VQ-VLA"
TRAIN_DIR="$ROOT/data/20260327-11:10:43/demos/success/train"
TEST_DIR="$ROOT/data/20260327-11:10:43/demos/success/test"
VAE_CKPT="$ROOT/outputs/dim_2_best/checkpoint.pth"
TRAIN_SCRIPT="$ROOT/imitation_learning/behavior_clone/scripts/train.py"
EVAL_SCRIPT="$ROOT/imitation_learning/behavior_clone/scripts/eval.py"
OUT_ROOT="$ROOT/outputs/bc_v5_arm_share"
VIS_ROOT="$ROOT/visualizations/bc_v5_arm_share"

COMMON_TRAIN_ARGS=(
  --train_dir "$TRAIN_DIR"
  --test_dir "$TEST_DIR"
  --vae_ckpt "$VAE_CKPT"
  --feat_dim 128
  --fusion_dim 256
  --dropout 0.3
  --reg_drift 1.0
  --lr 5e-4
  --min_lr 1e-5
  --weight_decay 1e-4
  --batch_size 128
  --total_steps 20000
  --warmup_steps 500
  --clip_grad 1.0
  --eval_freq 1000
  --save_freq 5000
  --print_freq 500
  --device cuda
)

COMMON_EVAL_ARGS=(
  --all
  --num_samples 1
  --save_debug_latent
  --no_plot
  --device cuda
)

conda run --no-capture-output -n serl python "$TRAIN_SCRIPT"   "${COMMON_TRAIN_ARGS[@]}"   --output_dir "$OUT_ROOT/no_arm_latent"

conda run --no-capture-output -n serl python "$TRAIN_SCRIPT"   "${COMMON_TRAIN_ARGS[@]}"   --output_dir "$OUT_ROOT/share_arm_latent"   --hand_condition_on_arm

conda run --no-capture-output -n serl python "$EVAL_SCRIPT"   --ckpt "$OUT_ROOT/no_arm_latent/checkpoint.pth"   --output_dir "$VIS_ROOT/no_arm_latent_eval"   "${COMMON_EVAL_ARGS[@]}"

conda run --no-capture-output -n serl python "$EVAL_SCRIPT"   --ckpt "$OUT_ROOT/share_arm_latent/checkpoint.pth"   --output_dir "$VIS_ROOT/share_arm_latent_eval"   "${COMMON_EVAL_ARGS[@]}"

conda run --no-capture-output -n serl python - <<'PY2'
import json
from pathlib import Path
import numpy as np

base = Path('/home/yxy/VQ-VLA/visualizations/bc_v5_arm_share')
variants = {
    'no_arm_latent': base / 'no_arm_latent_eval',
    'share_arm_latent': base / 'share_arm_latent_eval',
}
comparison = {}
for name, root in variants.items():
    summary = json.loads((root / 'summary.json').read_text())
    tf_dz = []
    ar_dz = []
    for npz_path in sorted(root.glob('traj_*_eval.npz')):
        arr = np.load(npz_path)
        tf_dz.append(np.linalg.norm(arr['tf_delta_z'], axis=-1).reshape(-1))
        ar_dz.append(np.linalg.norm(arr['ar_delta_z'], axis=-1).reshape(-1))
    comparison[name] = {
        'tf_arm': summary['modes']['tf']['arm_mse'],
        'tf_hand': summary['modes']['tf']['hand_mse'],
        'ar_arm': summary['modes']['ar']['arm_mse'],
        'ar_hand': summary['modes']['ar']['hand_mse'],
        'vision_gain_hand': summary['vision_gain_hand'],
        'onset_tf_hand': summary['slices']['onset_band']['modes']['tf']['hand_mse'],
        'onset_vision_gain_hand': summary['slices']['onset_band']['vision_gain_hand'],
        'tf_delta_z_norm_mean': float(np.concatenate(tf_dz).mean()),
        'ar_delta_z_norm_mean': float(np.concatenate(ar_dz).mean()),
    }
out = base / 'comparison.json'
out.write_text(json.dumps(comparison, indent=2))
print(json.dumps(comparison, indent=2))
print(f'Wrote {out}')
PY2
