# Behavior Cloning over a Frozen Hand-Action VAE

## Overview

这个模块是两阶段 pipeline 的 **Stage 2**。`vae/` 先学一个只基于动作历史的 hand latent prior；`imitation_learning/behavior_clone/` 再利用视觉和当前状态，在这个 prior 上做修正。

这份 README 现在同时保留三套信息：

- **旧版 / 1.x 历史说明**：保留原有的直观框架图、旧实验语义和旧版训练接口，方便对照 `outputs/bc_sweep_v2/`、`outputs/bc_v3_ablations/` 这些历史结果。
- **上一版 / 2.0 说明**：记录 shared trunk + split arm/hand + `delta_z` 的过渡架构，对照 `outputs/bc_v4_sweep/`。
- **当前 / 3.0 说明**：记录现在代码里已经实现的弱耦合结构：视觉共享、`arm_state` 和 `hand_prior` 分别编码，hand 分支可选是否接收 `arm latent`。

如果你是在看老实验，请优先看“旧版 / 1.x 历史保留”；如果你要对照 v4，请看“上一版 / 2.0”；如果你要用当前代码训练或评估，请看“当前 / 3.0”。

## 数据语义约定

数据集里的 `actions` 是一个 **(T, 12)** 张量，表示每一帧机器人的**绝对位姿**：

- 前 6 维：机械臂
- 后 6 维：灵巧手

所以：

```text
actions[t]   == 当前时刻 t 的 state[t]
actions[t+1] == BC 需要预测的目标
```

BC 的输入 `state` 实际上就是标准化后的 `actions[t]`。

## 旧版 / 1.x 历史保留

这一节保留的是你原来 README 里更直观的框架图和说明，便于对照历史实验。这里描述的是 **`delta_mu + delta_log_var`** 的旧版结构，不代表当前代码。

### 旧版概览

行为克隆（BC）策略，作为两阶段 pipeline 的 **Stage 2 视觉网络**，建立在 `vae/` 模块训练好的冻结 `HandActionVAE`（默认 `outputs/dim_2_best/checkpoint.pth`，latent_dim=2）之上。

- **Stage 1（`vae/` 模块）**：纯动作历史 → latent (μ, log_var) → decoder 预测 t+1 动作。已收敛。
- **Stage 2（旧版）**：相机图像 + 本体感知 state + 过去手部动作 → 直接预测机械臂 6 维动作 + 给 VAE 先验输出 `(Δμ, Δlog_var)` 修正灵巧手动作。**VAE 权重全程冻结**。

### 旧版直观框架图

```text
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

### 旧版文件结构

```text
imitation_learning/behavior_clone/
├── model/
│   ├── bc_dataset.py     # BCDataset + compute_state_stats
│   └── bc_policy.py      # BCPolicy + SimpleCNN + build_and_freeze_vae
├── scripts/
│   └── train.py          # 训练脚本（默认即 v1 配置）
├── README.md             # 本文档
└── EXPERIMENTS.md        # 实验日志，按时间倒序
```

### 旧版 Quick Start

```bash
/home/cxl/miniconda3/envs/serl/bin/python imitation_learning/behavior_clone/scripts/train.py \
    --train_dir data/20260327-11:10:43/demos/success/train \
    --test_dir  data/20260327-11:10:43/demos/success/test \
    --vae_ckpt  outputs/dim_2_best/checkpoint.pth \
    --output_dir outputs/bc_simple_v1
# checkpoints + training_curves.png -> outputs/bc_simple_v1/
```

20k 步在单卡 GPU 上约需要 5-10 分钟。

### 旧版关键设计

#### 过去窗口偏移（最重要的细节）

VAE 训练时的窗口约定：给定 `[a_{t-7} .. a_t]` 预测 `a_{t+1}`。

BC 在第 `t` 步要输出 `a_t`，因此送给 VAE 的窗口必须**终止于 `a_{t-1}`**：

```python
past_hand_win = hand_actions[max(0, t-8) : t]    # 上界是 t，不是 t+1
```

如果直接复用 `vae/model/hand_dataset.py` 的 `HandActionWindowDataset`（窗口终止于 `a_t`），就会把目标 `a_t` 泄漏到 prior 里。`bc_dataset.py` 的 `BCDataset` 当时就是为了解决这个偏移问题。

#### Delta heads zero-init

`delta_mu_head` 和 `delta_log_var_head` 的最后一层 Linear 都被零初始化。所以 step 0 时 `delta_mu == delta_log_var == 0`，policy 完全等价于“无修正的 VAE rollout”。

#### CNN 选用 GroupNorm 而非 BatchNorm

batch=128 + 5k 帧的小数据集下 BN 的 running stats 噪声大；GroupNorm（每组 8 通道）在小批量上更稳。

#### 两个视角分别独立的 CNN（不共享权重）

主视角和附加视角内容分布不同，共享权重会强迫卷积学习“两个视角都好用”的特征。两路 CNN 各约 160k 参数，整体仍在约 1M 量级。

#### 不归一化 hand action 目标

VAE decoder 是在原始动作空间训练的，hand 输出范围约 `[0, 0.6]`。如果对 hand GT 做 z-score，会强迫 decoder 输出它没见过的空间。

### 旧版默认超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--feat_dim` | 128 | 每路分支输出特征宽度 |
| `--fusion_dim` | 256 | 融合 MLP 隐层宽度 |
| `--lr` | `5e-4` | 比 VAE 的 `2e-3` 小，从头训的 CNN 在更高 LR 上容易尖峰 |
| `--min_lr` | `1e-5` | cosine 终值 |
| `--warmup_steps` | 500 | 线性 warmup |
| `--weight_decay` | `1e-4` | AdamW |
| `--total_steps` | 20000 | 训练步数 |
| `--batch_size` | 128 | |
| `--clip_grad` | 1.0 | 若早期 grad norm 报警，先放到 5.0 |
| `--eval_samples` | 3 | 验证集每个 batch 重复采样 3 次取均值 |
| `--num_workers` | 0 | 全部数据预载入 RAM，多进程会撑爆 `/dev/shm` |

### 旧版验证指标解读

| 指标 | 含义 |
|---|---|
| `arm_mse` | 机械臂 6 维 MSE |
| `hand_mse_full` | 完整 policy 的 hand 预测 MSE（多次 reparam 平均） |
| `hand_mse_no_correction` | `delta_mu=delta_log_var=0` 时的 hand MSE，即 VAE prior 自由 rollout 的得分 |
| `total_mse` | `arm_mse + hand_mse_full` |

判定：`hand_mse_full < hand_mse_no_correction` 才能说明视觉修正真的在帮 hand。

### 旧版 Stage 集成关系

```text
vae/  (Stage 1, frozen)
  ↓ outputs/dim_2_best/checkpoint.pth
imitation_learning/behavior_clone/  (Stage 2, old 1.x)
  ↓ outputs/bc_simple_v1/checkpoint.pth
[downstream policy rollout / robot deployment]
```

## 上一版 / 2.0

下面这一节描述的是 v4 时期代码实现的 2.0 版本，保留下来作为历史对照。

### 2.0 核心变化

2.0 不是推翻旧版，而是针对旧版暴露出来的问题做收缩和重构：

- 旧版里，`arm` 和 `hand` 的任务性质不同，但共享高层 head，容易互相拖拽。
- hand 通过 `Δμ / Δlog_var` 去间接控制冻结 decoder，路径太绕。
- `delta_log_var` 会把“控制量”和“分布形状”混在一起。
- 视觉更像是在决定“何时开始变化”，而不是重新参数化一个完整的 latent 高斯。

所以 2.0 的设计目标是：

- `arm` 继续做标准的条件回归。
- `hand` 直接在 Stage 1 已经学到的 latent prior 上做残差控制。
- 保留 `log_var_prior` 作为输入上下文，但不再让 BC 直接修改它。

### 2.0 架构图

```text
BC trainable inputs:
  img_main   (B, 3, 128, 128)
  img_extra  (B, 3, 128, 128)
  state      (B, 12) = normalized actions[t]

Frozen VAE-only input:
  past_hand_win (B, 8, 6) = hand[a_{t-7} .. a_t]

Shared observation trunk:
  img_main  -> cnn_main  ->
  img_extra -> cnn_extra -> concat -> fusion_mlp -> shared_h
  state     -> state_mlp ->

Split heads after shared_h:
  arm_head(shared_h) -> arm_action (B, 6)

  hand_obs_head(shared_h) -> hand_obs (B, 64)
  vae.encode(past_hand_win) -> mu_prior, log_var_prior   [frozen, no_grad]
  concat(hand_obs, mu_prior, log_var_prior) -> hand_delta_z_head -> delta_z
  z_ctrl = mu_prior + delta_z
  hand_action = vae.decode(z_ctrl)

No-correction baseline:
  z_no_corr = mu_prior
  hand_no_corr = vae.decode(z_no_corr)

Final output:
  action_pred = concat(arm_action, hand_action)
```

### 1.x -> 2.0 对照

| 项 | 旧版 / 1.x | 2.0 |
|---|---|---|
| 高层结构 | shared trunk + arm head + 两个 delta head | shared trunk + 独立 arm branch + 独立 hand branch |
| hand 控制接口 | `delta_mu`, `delta_log_var` | `delta_z` |
| latent 生成 | `reparameterize(mu_corr, log_var_corr)` | `z_ctrl = mu_prior + delta_z` |
| hand 路径 | stochastic | deterministic |
| no-correction baseline | 修正关闭后的 sampled prior decode | `decode(mu_prior)` |
| 主要问题 / 改进方向 | 控制路径绕，方差和控制耦合 | 更直接，便于诊断 |

### 2.0 代码改动整理

- `bc_policy.py`
  - 新增 split `arm` / `hand` 分支
  - 删除 `delta_mu_head` / `delta_log_var_head`
  - 新增 `hand_obs_head` / `hand_delta_z_head`
  - hand 路径改成 `z_ctrl = mu_prior + delta_z`
- `train.py`
  - zero-init sanity check 改成检查 `delta_z == 0`
  - eval 路径改成确定性 hand 前向，不再依赖重复采样
  - drift regularizer 改成约束 `hand_action` 相对 `decode(mu_prior)` 的偏离
- `eval.py`
  - latent debug 字段改成 `delta_z / z_ctrl / z_no_corr`
  - 加载旧版 checkpoint 时会直接报不兼容
- `diagnose_bc_stage2.py`
  - latent 统计从 `delta_mu / delta_log_var` 改成 `delta_z`

### 当前文件结构

```text
imitation_learning/behavior_clone/
├── model/
│   ├── bc_dataset.py
│   └── bc_policy.py
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── diagnose_bc_stage2.py
├── README.md
└── EXPERIMENTS.md
```

### 2.0 Quick Start

训练 2.0 结构的 BC：

```bash
conda run -n serl python imitation_learning/behavior_clone/scripts/train.py \
    --train_dir data/20260327-11:10:43/demos/success/train \
    --test_dir  data/20260327-11:10:43/demos/success/test \
    --vae_ckpt  outputs/dim_2_best/checkpoint.pth \
    --output_dir outputs/bc_split_delta_z_v1
```

评估：

```bash
conda run -n serl python imitation_learning/behavior_clone/scripts/eval.py \
    --ckpt outputs/bc_split_delta_z_v1/checkpoint.pth \
    --output_dir visualizations/bc_eval/bc_split_delta_z_v1 \
    --all --num_samples 1
```

批量诊断：

```bash
conda run -n serl python imitation_learning/behavior_clone/scripts/diagnose_bc_stage2.py \
    --ckpt outputs/bc_split_delta_z_v1/checkpoint.pth \
    --all \
    --output_json visualizations/bc_diagnosis/bc_split_delta_z_v1.json
```

### 2.0 兼容性说明

**旧的 BC checkpoint 与当前 2.0 代码不兼容。**

原因不是数据集或 VAE 变了，而是 Stage 2 的可训练 head 结构变了：

- 旧版：`delta_mu_head` + `delta_log_var_head`
- 2.0：`hand_obs_head` + `hand_delta_z_head`

因此像下面这些旧 checkpoint：

- `outputs/bc_sweep_v2/reg_0p0/checkpoint.pth`
- `outputs/bc_sweep_v2/reg_1p0/checkpoint.pth`
- `outputs/bc_sweep_v2/reg_10p0/checkpoint.pth`
- `outputs/bc_v3_ablations/handnoise/checkpoint.pth`
- `outputs/bc_v3_ablations/stateonly/checkpoint.pth`

都不能直接拿来评估 2.0 代码。`scripts/eval.py` 现在会显式报错，而不是静默半加载。

Stage 1 的 VAE checkpoint 仍然沿用：

- `outputs/dim_2_best/checkpoint.pth`

### 2.0 模型 API

```python
import torch
from imitation_learning.behavior_clone.model.bc_policy import BCPolicy, build_and_freeze_vae

bc_ckpt = torch.load(
    'outputs/bc_split_delta_z_v1/checkpoint.pth',
    map_location='cpu',
    weights_only=False,
)
action_mean = bc_ckpt['action_mean']
action_std = bc_ckpt['action_std']

vae = build_and_freeze_vae(bc_ckpt['args']['vae_ckpt'])
policy = BCPolicy(
    vae=vae,
    state_dim=12,
    feat_dim=bc_ckpt['args'].get('feat_dim', 128),
    fusion_dim=bc_ckpt['args'].get('fusion_dim', 256),
    disable_vision=bc_ckpt['args'].get('disable_vision', False),
)
missing, unexpected = policy.load_state_dict(bc_ckpt['model'], strict=False)
assert not [k for k in missing if not k.startswith('vae.')]
assert not unexpected
policy.eval()

img_main = torch.rand(1, 3, 128, 128)
img_extra = torch.rand(1, 3, 128, 128)
state_raw = torch.zeros(1, 12)
state = (state_raw - action_mean) / action_std
past_hand_win = torch.zeros(1, 8, 6)

out = policy(
    img_main=img_main,
    img_extra=img_extra,
    state=state,
    past_hand_win=past_hand_win,
)

print(out['action_pred'].shape)   # (1, 12)
print(out['delta_z'].shape)       # (1, latent_dim)
print(out['z_ctrl'].shape)        # (1, latent_dim)
```

### 2.0 训练目标

2.0 的训练 loss 仍然是：

```text
total_loss = arm_loss + hand_loss + reg_drift * drift_loss
```

其中：

- `arm_loss = MSE(arm_action, gt_arm)`
- `hand_loss = MSE(hand_action, gt_hand)`
- `drift_loss = MSE(hand_action, hand_no_corr)`

但 `drift_loss` 的物理意义和旧版不同：

- **旧版**：约束“修正后的 sampled hand”和“未修正 sampled hand”的差异
- **2.0**：约束 `hand_action` 相对 `decode(mu_prior)` 的偏离

所以在 2.0 里，drift regularizer 更接近于：

**限制 hand branch 对 prior latent 的控制幅度。**

### 2.0 评估模式

`scripts/eval.py` 保留了 3 种模式：

- `tf`：teacher-forced，state 和 hand history 都来自 GT
- `ar`：autoregressive，前一步预测动作会回灌到后续 state / hand history
- `no_corr`：强制 `delta_z = 0`，只看 `decode(mu_prior)` 的 baseline

另外还支持这些诊断开关：

- `--image_mode {normal, zero, stale, shuffle, swap}`
- `--feedback_horizon`
- `--state_mask {all, arm_only, none}`
- `--save_debug_latent`
- `--disable_vision`
- `--noise_std_hand`

注意：`--num_samples` 仍然保留，但在 2.0 里 hand 路径已经是确定性的，它更多只是为了兼容旧脚本接口。

## 当前 / 3.0

3.0 是在 2.0 的基础上继续往“弱耦合”方向推进的一版：不再让 hand 分支吃一个混合了 arm / hand / vision 的共享高层 trunk，而是把 **arm state** 和 **hand prior** 分开编码，并给 hand 分支保留一个明确的开关：要不要额外看 `arm latent`。

### 3.0 核心变化

- 视觉仍然共享：`img_main / img_extra -> CNN -> visual_feat`
- `state` 仍然沿用标准化后的 `actions[t]`，但 **3.0 只使用前 6 维 arm state**
- hand 分支不再看原始 hand state，而是看冻结 VAE 给出的 `mu_prior / log_var_prior`
- hand 分支输出仍然是 `delta_z`，控制接口保持 `z_ctrl = mu_prior + delta_z`
- 新增接口：`hand_condition_on_arm`
  - `False`：hand 只看 `{visual_feat, hand_prior_feat}`
  - `True`：hand 看 `{visual_feat, hand_prior_feat, arm_state_feat}`

### 3.0 架构图

```text
Trainable inputs:
  img_main   (B, 3, 128, 128)
  img_extra  (B, 3, 128, 128)
  state      (B, 12) = normalized actions[t]
                └─ BC 3.0 only uses state[:, :6] as arm_state

Frozen VAE-only input:
  past_hand_win (B, 8, 6) = hand[a_{t-7} .. a_t]

Shared vision path:
  img_main  -> cnn_main  ->
  img_extra -> cnn_extra -> concat -> visual_fusion -> visual_feat

Split latent encoders:
  arm_state[:6]                -> arm_state_encoder  -> arm_state_feat
  vae.encode(past_hand_win)    -> mu_prior, log_var_prior  [frozen, no_grad]
  concat(mu_prior, log_var_prior) -> hand_prior_encoder -> hand_prior_feat

Arm branch:
  concat(visual_feat, arm_state_feat) -> arm_head -> arm_action (B, 6)

Hand branch:
  concat(visual_feat, hand_prior_feat[, arm_state_feat]) -> hand_delta_z_head -> delta_z
  z_ctrl = mu_prior + delta_z
  hand_action = vae.decode(z_ctrl)

No-correction baseline:
  z_no_corr = mu_prior
  hand_no_corr = vae.decode(z_no_corr)

Final output:
  action_pred = concat(arm_action, hand_action)
```

### 2.0 -> 3.0 对照

| 项 | 2.0 | 当前 / 3.0 |
|---|---|---|
| 高层结构 | vision + full state 先进 shared trunk，再分 arm/hand | vision 共享，但 `arm_state` / `hand_prior` 分别编码 |
| hand 分支输入 | `shared_h + mu_prior + log_var_prior` | `visual_feat + hand_prior_feat (+ arm_state_feat)` |
| state 使用方式 | 默认 12 维一起进 trunk | 只消费前 6 维 arm state |
| arm -> hand 接口 | 隐式混在 shared trunk 里 | 显式开关 `hand_condition_on_arm` |
| 目标 | split head | weakly coupled branches |

### 3.0 架构比较：`hand_condition_on_arm` 关 / 开

固定同一组超参（沿用 v4 最优：`dropout=0.3`, `reg_drift=1.0`, `feat_dim=128`, `fusion_dim=256`, 20k steps）做了两组对照：

| variant | `hand_condition_on_arm` | TF arm | TF hand | hand no_corr | vision_gain | AR arm | AR hand | onset TF hand |
|---|---|---|---|---|---|---|---|---|
| `no_arm_latent` | False | 0.02664 | 0.00297 | 0.00295 | -0.000016 | 0.07479 | 0.08570 | 0.00839 |
| `share_arm_latent` | True | 0.02829 | 0.00285 | 0.00295 | +0.000100 | 0.07984 | 0.04803 | 0.00788 |

从这组结果看，**共享 arm latent 对 hand 是有帮助的**：

- `TF hand`: `0.00297 -> 0.00285`
- `AR hand`: `0.08570 -> 0.04803`
- onset 切片的 `vision_gain_hand`: `-0.000090 -> +0.000421`

但代价也很明确：

- `TF arm`: `0.02664 -> 0.02829`
- `AR arm`: `0.07479 -> 0.07984`

也就是说，`hand_condition_on_arm=True` 更像是在用一点 arm 精度，换 hand 分支更稳定的视觉 timing control。如果当前目标优先级是“让 hand 更可靠地在正确时机开始变化”，3.0 里更推荐先用 **`hand_condition_on_arm=True`**。

完整对比产物：

- 训练输出：`outputs/bc_v5_arm_share/{no_arm_latent,share_arm_latent}/`
- 全测试集评估：`visualizations/bc_v5_arm_share/{no_arm_latent_eval,share_arm_latent_eval}/summary.json`
- 聚合对比：`visualizations/bc_v5_arm_share/comparison.json`
- 复现实验脚本：`scripts/sweep_v5_arm_share.sh`

### 当前 Quick Start（3.0）

训练 3.0 结构的 BC：

```bash
conda run -n serl python imitation_learning/behavior_clone/scripts/train.py     --train_dir data/20260327-11:10:43/demos/success/train     --test_dir  data/20260327-11:10:43/demos/success/test     --vae_ckpt  outputs/dim_2_best/checkpoint.pth     --output_dir outputs/bc_v5_arm_share/share_arm_latent     --dropout 0.3 --reg_drift 1.0     --hand_condition_on_arm
```

评估：

```bash
conda run -n serl python imitation_learning/behavior_clone/scripts/eval.py     --ckpt outputs/bc_v5_arm_share/share_arm_latent/checkpoint.pth     --output_dir visualizations/bc_v5_arm_share/share_arm_latent_eval     --all --num_samples 1 --save_debug_latent --no_plot
```

### 当前兼容性说明（3.0）

**2.0 及更早的 BC checkpoint 与当前 3.0 代码不兼容。**

原因是可训练层已经从“shared trunk + hand_obs_head”切到了“`visual_fusion + arm_state_encoder + hand_prior_encoder`”这一套新接口。

因此像下面这些 checkpoint 都不能直接加载进 3.0：

- `outputs/bc_v4_sweep/drop03/checkpoint.pth`
- `outputs/bc_v4_sweep/reg10/checkpoint.pth`
- `outputs/bc_sweep_v2/*/checkpoint.pth`
- `outputs/bc_v3_ablations/*/checkpoint.pth`

`eval.py` 现在会直接报不兼容，而不会静默半加载。

### 当前模型 API（3.0）

```python
import torch
from imitation_learning.behavior_clone.model.bc_policy import BCPolicy, build_and_freeze_vae

bc_ckpt = torch.load(
    'outputs/bc_v5_arm_share/share_arm_latent/checkpoint.pth',
    map_location='cpu',
    weights_only=False,
)
action_mean = bc_ckpt['action_mean']
action_std = bc_ckpt['action_std']

vae = build_and_freeze_vae(bc_ckpt['args']['vae_ckpt'])
policy = BCPolicy(
    vae=vae,
    state_dim=12,
    arm_state_dim=6,
    feat_dim=bc_ckpt['args'].get('feat_dim', 128),
    fusion_dim=bc_ckpt['args'].get('fusion_dim', 256),
    disable_vision=bc_ckpt['args'].get('disable_vision', False),
    dropout=bc_ckpt['args'].get('dropout', 0.0),
    hand_condition_on_arm=bc_ckpt['args'].get('hand_condition_on_arm', False),
)
missing, unexpected = policy.load_state_dict(bc_ckpt['model'], strict=False)
assert not [k for k in missing if not k.startswith('vae.')]
assert not unexpected
policy.eval()

img_main = torch.rand(1, 3, 128, 128)
img_extra = torch.rand(1, 3, 128, 128)
state_raw = torch.zeros(1, 12)
state = (state_raw - action_mean) / action_std
past_hand_win = torch.zeros(1, 8, 6)

out = policy(
    img_main=img_main,
    img_extra=img_extra,
    state=state,
    past_hand_win=past_hand_win,
)

print(out['action_pred'].shape)     # (1, 12)
print(out['delta_z'].shape)         # (1, latent_dim)
print(out['arm_state_feat'].shape)  # (1, feat_dim)
print(out['hand_prior_feat'].shape) # (1, feat_dim)
```

### 当前训练 / 评估补充说明

- `state` 仍然保持 `(B, 12)` 的接口，方便沿用旧数据集和日志，但 **3.0 只使用前 6 维 arm state**。
- 因此在 3.0 下：
  - `state_mask=all` 与 `state_mask=arm_only` 是等价的
  - `state_mask=none` 才会真正去掉 arm state 条件
- `hand_condition_on_arm=False` 更强调结构解耦
- `hand_condition_on_arm=True` 更强调 hand timing 的可控性

## Frozen VAE Rules

无论是旧版还是 2.0，这几个守则都不变：

- VAE 参数全部 `requires_grad_(False)`。
- `BCPolicy.train(mode)` 被覆盖，防止 VAE 回到 train mode。
- optimizer 只接收 `requires_grad=True` 的 BC 参数。
- BC checkpoint 不保存 `vae.*` 权重。
- `vae.encode(past_hand_win)` 放在 `torch.no_grad()` 下，prior 对 BC 来说是常量输入。

## Recommended Next Step

既然代码已经切到 2.0，下一步应该重新训练一版 Stage 2，并优先关注：

- `vision_gain_hand`
- onset slice 上的 hand MSE
- `delta_z` 的量级是否稳定
- `arm` 和 `hand` 是否出现更清晰的任务分工

如果继续做实验，建议在 `EXPERIMENTS.md` 里把这条线单独记成：

- `bc_split_delta_z_v1`
