#!/usr/bin/env bash
# sweep_delta_vae.sh — Re-run all 3 VAE sweeps on delta-action data.
# 12 unique training runs (arch:5 + loss:3 + recon:4), each 20k steps.
# Mirrors the absolute-action sweeps in EXPERIMENTS.md exactly.
set -euo pipefail

PY=/home/cxl/miniconda3/envs/serl/bin/python
TRAIN=data/delta_action_20260327_11_10_43/train
TEST=data/delta_action_20260327_11_10_43/test
OUT=outputs/delta
SEED=42

COMMON="--latent_dim 2 --beta 0.001 --noise_std 0.01 --total_steps 20000 \
  --batch_size 256 --lr 2e-3 --seed $SEED --train_dir $TRAIN --test_dir $TEST"

echo "=== Delta-action VAE sweep ==="
echo "Train: $TRAIN"
echo "Test:  $TEST"
echo "Output root: $OUT"
echo ""

# ──────────────────────────────────────────────────────────────────
# Sweep 1: Architecture (5 cells)
# ──────────────────────────────────────────────────────────────────
echo ">>> SWEEP 1: Architecture (5 cells)"

echo "[1/12] A: h256 d1 mlp (baseline)"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 0.0 --free_bits 0.0 \
  --output_dir $OUT/arch_sweep/A_h256_d1_mlp

echo "[2/12] B: h512 d1 mlp"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 512 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 0.0 --free_bits 0.0 \
  --output_dir $OUT/arch_sweep/B_h512_d1_mlp

echo "[3/12] C: h256 d2 mlp"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 2 --encoder_type mlp \
  --recon_aux_weight 0.0 --free_bits 0.0 \
  --output_dir $OUT/arch_sweep/C_h256_d2_mlp

echo "[4/12] D: h512 d2 mlp"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 512 --num_hidden_layers 2 --encoder_type mlp \
  --recon_aux_weight 0.0 --free_bits 0.0 \
  --output_dir $OUT/arch_sweep/D_h512_d2_mlp

echo "[5/12] E: h256 d1 causal_conv"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type causal_conv \
  --recon_aux_weight 0.0 --free_bits 0.0 \
  --output_dir $OUT/arch_sweep/E_h256_d1_cnn

# ──────────────────────────────────────────────────────────────────
# Sweep 2: Loss ablation (3 cells; baseline = reuse A)
# ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> SWEEP 2: Loss ablation (3 cells)"

echo "[6/12] F: recon_aux=0.1"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 0.1 --free_bits 0.0 \
  --output_dir $OUT/loss_sweep/F_recon_only

echo "[7/12] G: free_bits=0.5"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 0.0 --free_bits 0.5 \
  --output_dir $OUT/loss_sweep/G_freebits_only

echo "[8/12] H: recon_aux=0.1 + free_bits=0.5"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 0.1 --free_bits 0.5 \
  --output_dir $OUT/loss_sweep/H_both

# ──────────────────────────────────────────────────────────────────
# Sweep 3: Recon aux weight (4 new cells; W000=A, W010=F reused)
# ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> SWEEP 3: Recon aux weight (4 cells)"

echo "[9/12] W005: recon_aux=0.05"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 0.05 --free_bits 0.0 \
  --output_dir $OUT/recon_sweep/W005

echo "[10/12] W030: recon_aux=0.30"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 0.30 --free_bits 0.0 \
  --output_dir $OUT/recon_sweep/W030

echo "[11/12] W050: recon_aux=0.50"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 0.50 --free_bits 0.0 \
  --output_dir $OUT/recon_sweep/W050

echo "[12/12] W100: recon_aux=1.00"
$PY vae/scripts/train.py $COMMON \
  --hidden_dim 256 --num_hidden_layers 1 --encoder_type mlp \
  --recon_aux_weight 1.00 --free_bits 0.0 \
  --output_dir $OUT/recon_sweep/W100

echo ""
echo "=== All 12 training runs complete ==="
echo ""

# ──────────────────────────────────────────────────────────────────
# Eval pass: free-run 200 samples for each cell
# ──────────────────────────────────────────────────────────────────
echo ">>> Running free-run eval (200 samples each)"

EVAL_COMMON="--test_dir $TEST --data_mode delta --free_run --num_samples 200 --save_plot --max_steps 100"

for sweep_dir in arch_sweep/A_h256_d1_mlp arch_sweep/B_h512_d1_mlp arch_sweep/C_h256_d2_mlp \
                 arch_sweep/D_h512_d2_mlp arch_sweep/E_h256_d1_cnn \
                 loss_sweep/F_recon_only loss_sweep/G_freebits_only loss_sweep/H_both \
                 recon_sweep/W005 recon_sweep/W030 recon_sweep/W050 recon_sweep/W100; do
  cell=$(basename $sweep_dir)
  sweep=$(dirname $sweep_dir)
  ckpt="$OUT/$sweep_dir/checkpoint.pth"
  if [ -f "$ckpt" ]; then
    echo "  eval: $sweep_dir"
    # Pick a test trajectory for seeding — use first available
    traj_id=$(ls $TEST/trajectory_*_demo_expert.pt | head -1 | grep -oP 'trajectory_\K[0-9]+')
    $PY vae/scripts/eval.py --ckpt "$ckpt" \
      --output_dir "visualizations/delta_eval/$sweep_dir" \
      $EVAL_COMMON --traj_id "$traj_id"
  else
    echo "  [skip] $ckpt not found"
  fi
done

echo ""
echo "=== Sweep + eval complete ==="
