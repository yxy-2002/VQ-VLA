# Behavior Cloning over a Frozen Hand-Action VAE

## Overview

行为克隆（BC）策略，作为两阶段 pipeline 的 **Stage 2 视觉网络**，建立在 `vae/` 模块训练好的冻结 `HandActionVAE`（默认 `outputs/dim_2_best/checkpoint.pth`，latent_dim=2）之上。

- **Stage 1（`vae/` 模块）**：纯动作历史 → latent (μ, log_var) → decoder 预测 t+1 动作。已收敛。
- **Stage 2（本模块）**：相机图像 + 本体感知 state + 过去手部动作 → 直接预测机械臂 6 维动作 + 给 VAE 先验输出 (Δμ, Δlog_var) 修正灵巧手动作。**VAE 权重全程冻结**。

```
Inputs
  img_main      (B, 3, 128, 128)   主视角 RGB
  img_extra     (B, 3, 128, 128)   附加视角 RGB
  state         (B, 24)            本体感知（已 z-score 标准化）
  past_hand_win (B, 8, 6)          过去 8 帧手部动作 [a_{t-8}..a_{t-1}]   ← 注意：终止于 t-1
        ↓
  cnn_main / cnn_extra      4 conv blocks + GroupNorm → (B, 128) ×2 (独立权重)
  state_encoder             MLP (24 → 128 → 128)
        ↓ concat
  fusion_mlp                Linear(384, 256) → ReLU → Linear(256, 256) → ReLU
        ↓
  ┌────────────────────┬────────────────────────────────────────┐
  │ arm_head           │ delta_mu_head / delta_log_var_head     │
  │ MLP → (B, 6)       │ MLP → (B, 2) ×2  (final layer zero-init)│
  │  = arm_action      │                                         │
  └────────────────────┴────────────────────┬───────────────────┘
                                            │
                       (frozen) VAE.encode(past_hand_win) → mu_p, lv_p
                                            │
                       mu_corr  = mu_p + delta_mu
                       lv_corr  = clamp(lv_p + delta_log_var, -10, 2)
                       z        = VAE.reparameterize(mu_corr, lv_corr)   # 必须 stochastic
                       hand_action = (frozen) VAE.decode(z)              # (B, 6)
                                            │
                  action_pred = concat(arm_action, hand_action)          # (B, 12)
```

## 文件结构

```
imitation_learning/behavior_clone/
├── model/
│   ├── bc_dataset.py     # BCDataset + compute_state_stats
│   └── bc_policy.py      # BCPolicy + SimpleCNN + build_and_freeze_vae
├── scripts/
│   └── train.py          # 训练脚本（默认即 v1 配置）
├── README.md             # 本文档
└── EXPERIMENTS.md        # 实验日志，按时间倒序
```

## Quick Start

训练（默认 v1 配置）：

```bash
/home/cxl/miniconda3/envs/serl/bin/python imitation_learning/behavior_clone/scripts/train.py \
    --train_dir data/20260327-11:10:43/demos/success/train \
    --test_dir  data/20260327-11:10:43/demos/success/test \
    --vae_ckpt  outputs/dim_2_best/checkpoint.pth \
    --output_dir outputs/bc_simple_v1
# checkpoints + training_curves.png → outputs/bc_simple_v1/
```

20k 步在单卡 GPU 上约需要 5-10 分钟。

## 模型 API

```python
import importlib.util, torch

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

ds_mod  = _load("imitation_learning/behavior_clone/model/bc_dataset.py", "bc_dataset")
pol_mod = _load("imitation_learning/behavior_clone/model/bc_policy.py",  "bc_policy")

# 1. Build frozen VAE + BC policy
vae = pol_mod.build_and_freeze_vae("outputs/dim_2_best/checkpoint.pth")
policy = pol_mod.BCPolicy(vae=vae)

# 2. Load BC checkpoint (state_mean / state_std are saved alongside)
ckpt = torch.load("outputs/bc_simple_v1/checkpoint.pth", map_location="cpu", weights_only=False)
policy.load_state_dict(ckpt["model"], strict=False)   # strict=False because vae.* keys are dropped
policy.eval()

# 3. Inference on a single sample
out = policy(
    img_main      = torch.rand(1, 3, 128, 128),
    img_extra     = torch.rand(1, 3, 128, 128),
    state         = torch.zeros(1, 24),                # already standardized
    past_hand_win = torch.zeros(1, 8, 6),
)
action = out["action_pred"]                            # (1, 12) = [arm(6), hand(6)]
```

## 冻结 VAE 守则（must do）

| 项 | 实现 |
|---|---|
| 关闭 VAE 梯度 | `for p in vae.parameters(): p.requires_grad_(False)` (在 `build_and_freeze_vae` 中) |
| 优化器只收 BC 参数 | `torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], ...)` |
| 永远不让 VAE 进 train mode | `BCPolicy.train(mode)` 被覆盖，调用后立即 `self.vae.eval()` |
| BC checkpoint 不存 VAE 权重 | `strip_vae_state_dict()` 把所有 `vae.*` key 过滤掉 |
| log_var 安全 clamp | `torch.clamp(lv_p + delta_lv, -10, 2)`（σ 范围 [0.007, 2.7]） |
| **采样必须 stochastic** | 用 `vae.reparameterize(mu, lv)`，**不要**用 deterministic μ（见 vae/README.md trick #5） |

## 关键设计

### 过去窗口偏移（最重要的细节）

VAE 训练时的窗口约定：给定 `[a_{t-7} .. a_t]` 预测 `a_{t+1}`。

BC 在第 `t` 步要输出 `a_t`，因此送给 VAE 的窗口必须**终止于 `a_{t-1}`**：

```python
past_hand_win = hand_actions[max(0, t-8) : t]    # 上界是 t，不是 t+1
```

如果直接复用 `vae/model/hand_dataset.py` 的 `HandActionWindowDataset`（窗口终止于 `a_t`），就会把目标 `a_t` 泄漏到 prior 里——decoder 自己就给出近似答案，BC 的 delta 退化成 trivial，且训练/推理分布不一致。`bc_dataset.py` 的 `BCDataset` 自己实现了正确的偏移。

### Delta heads zero-init

`delta_mu_head` 和 `delta_log_var_head` 的**最后一层 Linear**（weight + bias）都被零初始化。所以 step 0 时 `delta_mu == delta_log_var == 0`，policy 完全等价于"无修正的 VAE rollout"——这是天然的 baseline。训练脚本会在 step 0 显式 assert `|delta|_max == 0`。

### CNN 选用 GroupNorm 而非 BatchNorm

batch=128 + 5k 帧的小数据集下 BN 的 running stats 噪声大；GroupNorm（每组 8 通道）在小批量上更稳。

### 两个视角分别独立的 CNN（不共享权重）

主视角和附加视角内容分布不同，共享权重会强迫卷积学习"两个视角都好用"的特征（更难）。两路 CNN 各 ~160k 参数，整体仍在 ~1M 量级。

### 不归一化 hand action 目标

VAE decoder 是在原始动作空间训练的，hand 输出范围 ~[0, 0.6]。如果对 hand GT 做 z-score，BC 就会强迫 VAE decoder 输出标准化值——它从来没见过这个空间。**arm 同理，保留原始尺度**。

## 默认超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--feat_dim` | 128 | 每路分支输出特征宽度 |
| `--fusion_dim` | 256 | 融合 MLP 隐层宽度 |
| `--lr` | **5e-4** | 比 VAE 的 2e-3 小，从头训的 CNN 在 2e-3 容易尖峰 |
| `--min_lr` | 1e-5 | cosine 终值 |
| `--warmup_steps` | 500 | 线性 warmup |
| `--weight_decay` | 1e-4 | AdamW |
| `--total_steps` | 20000 | 训练步数 |
| `--batch_size` | 128 | |
| `--clip_grad` | 1.0 | 若早期 grad norm 报警，先放到 5.0 再说 |
| `--eval_samples` | 3 | 验证集每个 batch 重复采样 3 次取均值，减小方差 |
| `--num_workers` | **0** | 全部数据已预载入 RAM，多进程会撑爆 /dev/shm |

## 验证指标解读

每次 eval 报告 4 个数：

| 指标 | 含义 |
|---|---|
| `arm_mse` | 机械臂 6 维 MSE，越低越好 |
| `hand_mse_full` | 完整 policy 的 hand 预测 MSE（3 次 reparam 平均） |
| `hand_mse_no_correction` | δμ=δlog_var=0 时的 hand MSE — 即 **VAE 自己的 prior 自由 rollout 的得分**，是 BC 修正必须超过的下限 |
| `total_mse = arm_mse + hand_mse_full` | 训练目标 |

**判定**：`hand_mse_full < hand_mse_no_correction` 才能证明视觉信号真的在驱动手部动作；否则 BC delta 退化为零功效，是 bug 信号。

## Stage 集成关系

```
vae/  (Stage 1, frozen)
  ↓ outputs/dim_2_best/checkpoint.pth
imitation_learning/behavior_clone/  (Stage 2, this module)
  ↓ outputs/bc_simple_v1/checkpoint.pth
[downstream policy rollout / robot deployment]
```
