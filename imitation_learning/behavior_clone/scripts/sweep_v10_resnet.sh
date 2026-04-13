#!/usr/bin/env bash
# v10 sweep: ResNet-18 backbone + state_encoder × freeze_backbone
#
# 3 state encoder variants × 2 freeze settings = 6 runs
# Uses AMP + batch=256 + lr=5e-4 + backbone_lr_scale=0.1 when trainable.
# Total ~80 min on a single 4090.
set -euo pipefail

PROJ_ROOT="/home/yxy/VQ-VLA"
cd "$PROJ_ROOT"

PY="conda run -n serl --no-capture-output python"
TRAIN_SCRIPT="imitation_learning/behavior_clone/scripts/train.py"
EVAL_SCRIPT="imitation_learning/behavior_clone/scripts/eval.py"

TRAIN_DIR="data/20260327-11:10:43/demos/success/train"
TEST_DIR="data/20260327-11:10:43/demos/success/test"

OUT_ROOT="outputs/bc_v10_resnet"
VIZ_ROOT="visualizations/bc_v10_resnet"
mkdir -p "$OUT_ROOT" "$VIZ_ROOT"

TOTAL_STEPS=10000
BATCH=256
WARMUP=500
LR=5e-4
BLR_SCALE=0.1

run_one () {
    local tag="$1"
    local state_enc="$2"
    local freeze_flag="$3"   # either "" or "--freeze_backbone"
    local outdir="$OUT_ROOT/$tag"
    local vizdir="$VIZ_ROOT/$tag"
    mkdir -p "$outdir" "$vizdir"

    echo "============================================================"
    echo "[$tag] state_encoder=$state_enc  freeze=${freeze_flag:-no}"
    echo "============================================================"
    $PY $TRAIN_SCRIPT \
        --train_dir "$TRAIN_DIR" \
        --test_dir  "$TEST_DIR" \
        --output_dir "$outdir" \
        --state_encoder "$state_enc" \
        --batch_size "$BATCH" \
        --total_steps "$TOTAL_STEPS" \
        --warmup_steps "$WARMUP" \
        --lr "$LR" \
        --backbone_lr_scale "$BLR_SCALE" \
        $freeze_flag \
        2>&1 | tee "$outdir/train.log"

    echo "[$tag] eval ..."
    $PY $EVAL_SCRIPT \
        --ckpt "$outdir/checkpoint.pth" \
        --output_dir "$vizdir" \
        --num_samples 3 \
        2>&1 | tee "$outdir/eval.log"
}

# Core 6 runs
run_one "mlp_trainable"      "mlp"      ""
run_one "mlp_frozen"         "mlp"      "--freeze_backbone"
run_one "linear64_trainable" "linear64" ""
run_one "linear64_frozen"    "linear64" "--freeze_backbone"
run_one "raw_trainable"      "raw"      ""
run_one "raw_frozen"         "raw"      "--freeze_backbone"

echo "============================================================"
echo "All v10 sweep runs complete."
echo "Summary script: imitation_learning/behavior_clone/scripts/summarize_v10.py"
echo "============================================================"
