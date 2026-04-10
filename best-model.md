# Best Model

当前最推荐的模型是 `outputs/chunk_cvae_h12_b1e3/checkpoint.pth`，在可视化里对应
`outputs/chunk_fix_viz_20260410/chunk_cvae_h12_b1e3_s12`。

## Why this model

- 模型：`Chunk-CVAE`
- 推理设置：`rollout_stride=12`
- 观感上比旧 `chunk_h12_b1e3` 更连续、更平滑，基本消除了固定周期的 chunk 边界尖刺。
- 分布指标上也更接近 GT：open-like 条件下的触发时机分布、hazard、终态闭合值分布都明显优于旧 chunk，也优于当前的 `vae_step`。

## Files

- checkpoint: `outputs/chunk_cvae_h12_b1e3/checkpoint.pth`
- training script: `vae/scripts/train_chunk_cvae.py`
- eval script: `vae/scripts/eval_chunk_cvae.py`
- visualization script: `vae/scripts/visualize_competitive_models.py`
- model definition: `vae/model/hand_chunk_cvae.py`

## Training command

```bash
python vae/scripts/train_chunk_cvae.py \
  --train_dir success/train \
  --test_dir success/test \
  --output_dir outputs/chunk_cvae_h12_b1e3 \
  --device mps \
  --window_size 8 \
  --future_horizon 12 \
  --hidden_dim 256 \
  --latent_dim 2 \
  --beta 0.001 \
  --batch_size 256 \
  --total_steps 6000 \
  --warmup_steps 500 \
  --beta_warmup 2000 \
  --eval_freq 1000 \
  --save_freq 2000 \
  --print_freq 500
```

## Eval command

```bash
python vae/scripts/eval_chunk_cvae.py \
  --ckpt outputs/chunk_cvae_h12_b1e3/checkpoint.pth \
  --test_dir success/test \
  --device mps \
  --rollout_stride 12
```

## Visualization command

生成当前对比图（包含 `chunk_cvae_h12_b1e3_s12`）用：

```bash
python vae/scripts/visualize_competitive_models.py \
  --test_dir success/test \
  --models_json outputs/comparisons/models_chunk_fix_20260410.json \
  --output_dir outputs/chunk_fix_viz_20260410 \
  --device mps \
  --lang zh
```

如果只想单独保留这个 best model，可以准备一个只含 `chunk_cvae_h12_b1e3_s12` 的 `models_json` 再跑同一个脚本。
