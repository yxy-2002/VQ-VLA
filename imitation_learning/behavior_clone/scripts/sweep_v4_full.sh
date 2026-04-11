#!/usr/bin/env bash
# Full BC v2.0 validation sweep: reg_drift, dropout, capacity, lr/wd, ablations.
# 14 runs, ~2.5 hours on a single RTX 4090.
#
# Usage:
#     bash imitation_learning/behavior_clone/scripts/sweep_v4_full.sh
#
# Outputs:
#     outputs/bc_v4_sweep/<tag>/checkpoint.pth
#     outputs/bc_v4_sweep/<tag>.train.log
#     visualizations/bc_v4_sweep/<tag>/summary.json
#     outputs/bc_v4_sweep/<tag>.eval.log
#     outputs/bc_v4_sweep/V4_SWEEP_SUMMARY.txt

set -e
set -u

PYTHON=/home/cxl/miniconda3/envs/serl/bin/python
TRAIN=imitation_learning/behavior_clone/scripts/train.py
EVAL=imitation_learning/behavior_clone/scripts/eval.py
TRAIN_DIR=data/20260327-11:10:43/demos/success/train
TEST_DIR=data/20260327-11:10:43/demos/success/test
VAE_CKPT=outputs/dim_2_best/checkpoint.pth

OUT_ROOT=outputs/bc_v4_sweep
VIS_ROOT=visualizations/bc_v4_sweep
mkdir -p "$OUT_ROOT" "$VIS_ROOT"

run_one() {
    local TAG="$1"
    shift
    local OUT="$OUT_ROOT/$TAG"
    local TRAIN_LOG="$OUT_ROOT/${TAG}.train.log"
    local EVAL_LOG="$OUT_ROOT/${TAG}.eval.log"

    echo
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] BEGIN  $TAG  |  extra flags: $@"
    echo "================================================================"

    PYTHONUNBUFFERED=1 "$PYTHON" "$TRAIN" \
        --train_dir "$TRAIN_DIR" \
        --test_dir "$TEST_DIR" \
        --vae_ckpt "$VAE_CKPT" \
        --output_dir "$OUT" \
        "$@" \
        > "$TRAIN_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] train done -> $TRAIN_LOG"

    PYTHONUNBUFFERED=1 "$PYTHON" "$EVAL" \
        --ckpt "$OUT/checkpoint.pth" \
        --output_dir "$VIS_ROOT/$TAG" \
        --all --num_samples 1 \
        > "$EVAL_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] eval done  -> $EVAL_LOG"
}

T_START=$(date +%s)

# ── G1: Baseline ─────────────────────────────────────────────────────────────
run_one "baseline"      --reg_drift 1.0

# ── G2: reg_drift confirmation ───────────────────────────────────────────────
run_one "reg0"          --reg_drift 0.0
run_one "reg10"         --reg_drift 10.0

# ── G3: Dropout sweep ────────────────────────────────────────────────────────
run_one "drop01"        --reg_drift 1.0 --dropout 0.1
run_one "drop03"        --reg_drift 1.0 --dropout 0.3
run_one "drop05"        --reg_drift 1.0 --dropout 0.5

# ── G4: Capacity sweep ───────────────────────────────────────────────────────
run_one "small"         --reg_drift 1.0 --feat_dim 64 --fusion_dim 128
run_one "small_drop"    --reg_drift 1.0 --feat_dim 64 --fusion_dim 128 --dropout 0.3
run_one "mid_drop"      --reg_drift 1.0 --feat_dim 96 --fusion_dim 192 --dropout 0.3

# ── G5: Training hyperparameters ─────────────────────────────────────────────
run_one "lr_low"        --reg_drift 1.0 --lr 2e-4 --min_lr 5e-6
run_one "wd_high"       --reg_drift 1.0 --weight_decay 1e-3
run_one "lr_wd_drop"    --reg_drift 1.0 --lr 2e-4 --weight_decay 1e-3 --dropout 0.3

# ── G6: Ablation controls ───────────────────────────────────────────────────
run_one "no_vision"     --reg_drift 1.0 --disable_vision
run_one "no_state"      --reg_drift 1.0 --state_mask none

T_ELAPSED=$(( $(date +%s) - T_START ))

# ── Consolidated summary ─────────────────────────────────────────────────────
SUMMARY="$OUT_ROOT/V4_SWEEP_SUMMARY.txt"
ALL_TAGS="baseline reg0 reg10 drop01 drop03 drop05 small small_drop mid_drop lr_low wd_high lr_wd_drop no_vision no_state"
{
    echo "BC v2.0 full sweep summary"
    echo "Generated: $(date)"
    echo "Total wall time: ${T_ELAPSED}s"
    echo
    printf "%-14s | %-6s | %-8s | %-8s | %-10s | %-10s | %-12s | %-10s | %-10s | %-10s\n" \
        "tag" "drop" "feat" "fusion" "tf_arm" "tf_hand" "no_corr_hand" "vis_gain" "ar_arm" "ar_hand"
    printf "%s\n" "-------------- + ------ + -------- + -------- + ---------- + ---------- + ------------ + ---------- + ---------- + ----------"
    for TAG in $ALL_TAGS; do
        SUMMARY_JSON="$VIS_ROOT/$TAG/summary.json"
        if [ ! -f "$SUMMARY_JSON" ]; then
            printf "%-14s | (missing summary.json)\n" "$TAG"
            continue
        fi
        CKPT_FILE="$OUT_ROOT/$TAG/checkpoint.pth"
        "$PYTHON" - "$TAG" "$SUMMARY_JSON" "$CKPT_FILE" <<'PYEOF'
import json, sys, torch
tag, s_path, c_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(s_path) as f:
    d = json.load(f)
m = d["modes"]
try:
    ckpt = torch.load(c_path, map_location="cpu", weights_only=False)
    ca = ckpt.get("args", {})
    drop = ca.get("dropout", 0.0)
    feat = ca.get("feat_dim", 128)
    fusion = ca.get("fusion_dim", 256)
except Exception:
    drop, feat, fusion = "?", "?", "?"
print(f"{tag:<14} | "
      f"{drop:<6} | "
      f"{feat:<8} | "
      f"{fusion:<8} | "
      f"{m['tf']['arm_mse']:<10.6f} | "
      f"{m['tf']['hand_mse']:<10.6f} | "
      f"{m['no_corr']['hand_mse']:<12.6f} | "
      f"{d['vision_gain_hand']:<+10.6f} | "
      f"{m['ar']['arm_mse']:<10.6f} | "
      f"{m['ar']['hand_mse']:<10.6f}")
PYEOF
    done
    echo
    echo "Legend:"
    echo "  tf_arm / tf_hand   = teacher-forced MSE (best-case accuracy)"
    echo "  no_corr_hand       = VAE prior alone (delta_z=0)"
    echo "  vis_gain           = no_corr_hand - tf_hand  (positive = BC delta helps)"
    echo "  ar_arm / ar_hand   = autoregressive MSE (deployment proxy)"
} | tee "$SUMMARY"

echo
echo "[$(date +%H:%M:%S)] SWEEP COMPLETE — ${T_ELAPSED}s total"
echo "Summary: $SUMMARY"
