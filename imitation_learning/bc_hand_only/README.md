# Hand-Only Behavior Cloning (Decoupled Hand Branch)

## 动机

完整的 BC 3.0 模型（`imitation_learning/behavior_clone/`）同时预测 arm 和 hand，但两个分支的性能都很差。为了定位问题来源，这里把 hand 分支单独拆出来训练和评估：

- 如果 hand-only 的 AR MSE 和 BC 3.0 的 hand AR MSE 差不多 → **问题出在 delta_z + frozen VAE 方案本身**
- 如果 hand-only 明显更好 → **问题出在 arm/hand 耦合**

## 架构

```text
Inputs
  img_main      (B, 3, 128, 128)   主视角 RGB
  img_extra     (B, 3, 128, 128)   附加视角 RGB
  past_hand_win (B, 8, 6)          过去 8 帧手部动作 [a_{t-7}..a_t]
        ↓
  cnn_main / cnn_extra      4 conv blocks + GroupNorm → (B, 128) ×2 (独立权重)
        ↓ concat
  visual_fusion             Linear(256→128) → ReLU → Linear(128→128) → ReLU
        ↓                                                    visual_feat (128)
  ┌─────────────────────────────────────────────────────────┐
  │ frozen VAE encoder                                       │
  │   past_hand_win → (mu_prior, log_var_prior)             │
  │   → hand_prior_encoder → hand_prior_feat (128)          │
  └─────────────────────────────────────────────────────────┘
        ↓ concat [visual_feat, hand_prior_feat]
  hand_delta_z_head         Linear(256→fusion_dim) → ReLU
                            → Linear(fusion_dim→64) → ReLU
                            → [Dropout] → Linear(64→latent_dim)  (zero-init)
        ↓
  delta_z                   (B, 2)
  z_ctrl = mu_prior + delta_z
        ↓
  frozen VAE decoder        → hand_action (B, 6)
```

**与 BC 3.0 的区别**：没有 arm_state_encoder、arm_head、hand_condition_on_arm。模型不接收 robot state，只看图像和 VAE prior。

## Loss

```python
hand_loss  = MSE(hand_action, gt_hand)           # 手部预测精度
drift_loss = MSE(hand_action, hand_no_corr)       # delta_z 正则化
total_loss = hand_loss + reg_drift * drift_loss
```

## Quick Start

### 训练

```bash
/home/cxl/miniconda3/envs/serl/bin/python \
    imitation_learning/bc_hand_only/scripts/train.py \
    --train_dir data/20260327-11:10:43/demos/success/train \
    --test_dir  data/20260327-11:10:43/demos/success/test \
    --vae_ckpt  outputs/dim_2_best/checkpoint.pth \
    --output_dir outputs/bc_hand_only_sweep/baseline
```

### 评估（AR + 可视化）

```bash
/home/cxl/miniconda3/envs/serl/bin/python \
    imitation_learning/bc_hand_only/scripts/eval.py \
    --ckpt outputs/bc_hand_only_sweep/baseline/checkpoint.pth \
    --output_dir visualizations/bc_hand_only_sweep/baseline \
    --num_samples 5
```

### 全量 Sweep

```bash
bash imitation_learning/bc_hand_only/scripts/sweep.sh
```

输出：`outputs/bc_hand_only_sweep/SWEEP_SUMMARY.txt`

## 评估输出

每条测试轨迹生成：
- `traj_{id}_ar_actions.png` — 2×3 grid，6 个手部关节的 GT / AR / no-corr 对比
- `traj_{id}_ar_mse.png` — 逐步 hand MSE

汇总：
- `summary_ar.png` — AR vs no-corr vs copy baseline 柱状图
- `per_trajectory_hand_mse.png` — 按轨迹排序的 hand MSE
- `latent_diagnostics.png` — delta_z / mu_prior 分布诊断
- `summary.json` — 完整数值指标

## 如何解读结果

1. **vision_gain = nc_hand - ar_hand**：正值说明视觉修正有帮助
2. **no_vision ablation**：如果 disable_vision 和有视觉的 AR MSE 差不多，说明视觉修正没有贡献
3. **与 BC 3.0 对比**：直接比较 `ar_hand_mse`，看 hand-only 是否更好

## 超参默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lr | 5e-4 | 与 BC 3.0 一致 |
| batch_size | 128 | |
| total_steps | 20000 | |
| reg_drift | 1.0 | drift 正则化权重 |
| feat_dim | 128 | 特征维度 |
| fusion_dim | 256 | MLP 隐层宽度 |
| dropout | 0.0 | hand delta-z head 的 dropout |

## 文件结构

```
bc_hand_only/
├── model/
│   ├── bc_hand_policy.py      # BCHandPolicy 模型定义
│   └── bc_hand_dataset.py     # BCHandDataset 数据集
├── scripts/
│   ├── train.py               # 训练脚本
│   ├── eval.py                # AR 评估 + 可视化
│   └── sweep.sh               # 实验 sweep
├── README.md
└── EXPERIMENTS.md
```
