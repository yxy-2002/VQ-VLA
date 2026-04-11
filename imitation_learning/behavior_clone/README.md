# Behavior Cloning over a Frozen Hand-Action VAE

## Overview

两阶段 pipeline 的 **Stage 2**：`vae/` 先学一个只基于动作历史的 hand latent prior；本模块利用视觉和当前状态，在这个 prior 上做修正，同时预测机械臂动作。

## 架构 (BC 3.0)

弱耦合 arm/hand 分支，视觉共享，arm_state 和 hand_prior 分别编码。hand 分支始终接收 arm_state_feat 作为额外条件。

```text
Inputs
  img_main      (B, 3, 128, 128)   主视角 RGB
  img_extra     (B, 3, 128, 128)   附加视角 RGB
  state         (B, 12)            本体感知（z-score 标准化，仅前 6 维 arm 被使用）
  past_hand_win (B, 8, 6)          过去 8 帧手部动作 [a_{t-7}..a_t]
        ↓
  cnn_main / cnn_extra      4 conv blocks + GroupNorm → (B, 128) ×2
        ↓ concat
  visual_fusion             Linear(256→128) → ReLU → Linear(128→128) → ReLU
        ↓                                                    visual_feat (128)
  arm_state_encoder         state[:6] → Linear(6→128) → ReLU → Linear(128→128)
        ↓                                                    arm_state_feat (128)
  frozen VAE encoder        past_hand_win → (mu_prior, log_var_prior)
  hand_prior_encoder        [mu_p, lv_p] → Linear(4→128) → ReLU → Linear(128→128)
        ↓                                                    hand_prior_feat (128)

  ARM BRANCH:   [visual_feat, arm_state_feat] → arm_head → arm_action (6)
  HAND BRANCH:  [visual_feat, hand_prior_feat, arm_state_feat]
                → hand_delta_z_head → delta_z (2)
                → z_ctrl = mu_prior + delta_z
                → frozen VAE decoder → hand_action (6)
```

## Loss

```python
arm_loss   = MSE(arm_action, gt[:, :6])
hand_loss  = MSE(hand_action, gt[:, 6:])
drift_loss = MSE(hand_action, hand_no_corr)
total_loss = arm_loss + hand_loss + reg_drift * drift_loss
```

## 关键训练技巧：past_hand_win 噪声注入

训练时必须给 `past_hand_win` 添加高斯噪声（`--noise_std_hand`，默认 0.1），否则模型在 AR 模式下因误差累积而完全崩溃。

| noise_std | AR arm MSE | AR hand MSE | hand vision_gain |
|-----------|-----------|------------|-----------------|
| 0.00 | 0.061 | 0.086 (= no_corr) | -0.037 |
| 0.03 | 0.091 | 0.023 | +0.063 |
| 0.05 | 0.088 | 0.020 | +0.066 |
| **0.10** | 0.098 | **0.014** | **+0.072** |

## Quick Start

### 训练

```bash
/home/cxl/miniconda3/envs/serl/bin/python \
    imitation_learning/behavior_clone/scripts/train.py \
    --train_dir data/20260327-11:10:43/demos/success/train \
    --test_dir  data/20260327-11:10:43/demos/success/test \
    --vae_ckpt  outputs/dim_2_best/checkpoint.pth \
    --output_dir outputs/bc_v6_noise/noise01
```

默认已包含 `--noise_std_hand 0.1` 和 `--reg_drift 1.0`。

## 超参默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| noise_std_hand | **0.1** | past_hand_win 训练噪声，AR 鲁棒性关键参数 |
| reg_drift | 1.0 | drift 正则化权重 |
| lr | 5e-4 | |
| batch_size | 128 | |
| total_steps | 20000 | |
| feat_dim | 128 | 特征维度 |
| fusion_dim | 256 | MLP 隐层宽度 |
| dropout | 0.0 | arm/hand head 的 dropout |

## 数据语义约定

数据集里的 `actions` 是一个 **(T, 12)** 张量，表示每一帧机器人的绝对位姿（前 6 维机械臂，后 6 维灵巧手）。

```text
actions[t]   == 当前时刻 t 的 state
actions[t+1] == BC 需要预测的目标
```

## 文件结构

```
behavior_clone/
├── model/
│   ├── bc_policy.py       # BCPolicy 模型定义
│   └── bc_dataset.py      # BCDataset 数据集
├── scripts/
│   └── train.py           # 训练脚本
├── README.md
└── EXPERIMENTS.md
```

评估脚本位于 `bc_hand_only/scripts/eval.py`（hand-only 版本）。完整模型的 AR 评估可参考该脚本或使用已保存的可视化结果（`visualizations/bc_v6_noise/`）。
