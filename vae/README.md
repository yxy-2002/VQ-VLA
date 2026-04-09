# Hand Action VAE: State-based Action Prediction Prior

## Overview

基于过去 8 帧灵巧手动作历史，预测下一帧动作的 VAE。作为两阶段 pipeline 的 Stage 1 基座网络：

- **Stage 1（本模块）**：纯动作历史 → latent (μ, log_var) → decoder 预测 t+1 动作
- **Stage 2（后续）**：视觉网络输出 Δμ, Δlog_var 修正先验 → 冻结的 decoder 解码修正后的动作

```
Input: past 8 frames (B, 8, 6)                          [a_{t-7}, ..., a_t]
        ↓
   MLP Encoder      Linear(48,256) → SiLU → Linear(256,256) → SiLU
        ↓
   fc_mu, fc_log_var       (B, 256) → (B, latent_dim)  ×2
        ↓
   Reparameterize           z = μ + σ ⊙ ε,  ε ~ N(0, I)
        ↓
   MLP Decoder      Linear(latent_dim,256) → SiLU → Linear(256,256) → SiLU → Linear(256,6)
        ↓
Output: predicted t+1 action (B, 6)                     [â_{t+1}]
```

**Latent space 当前默认为 2 维**（参考 README 末尾的 Sweep findings 一节，这是在 latent_dim 严格为 2 的约束下经验最优的配置）。

## 文件结构

```
vae/
├── model/
│   ├── hand_vae.py        # HandActionVAE + MLPEncoder + CausalConvEncoder
│   ├── hand_dataset.py    # HandActionWindowDataset (滑动窗口)
│   └── utils.py           # cosine_scheduler, beta_annealing_schedule
├── scripts/
│   ├── train.py           # 训练脚本（默认即 best 配置）
│   └── eval.py            # 评估脚本（自动从 ckpt 推断架构）
└── README.md
```

## Quick Start

训练（默认配置即 best）：

```bash
python vae/scripts/train.py \
    --train_dir /home/yxy/data/20260327-11:10:43/demos/success/train \
    --test_dir  /home/yxy/data/20260327-11:10:43/demos/success/test
# checkpoints + training_curves.png → outputs/dim_2_best/
```

评估（自动推断架构，无需手动指定 hidden_dim/latent_dim）：

```bash
python vae/scripts/eval.py \
    --ckpt outputs/dim_2_best/checkpoint.pth \
    --output_dir visualizations/vae_eval/dim_2_best \
    --free_run --num_samples 200 --save_plot
```

## 模型 API

```python
from vae.model.hand_vae import HandActionVAE

model = HandActionVAE(
    hidden_dim=256, latent_dim=2, num_hidden_layers=1, beta=0.001
)
ckpt = torch.load("outputs/dim_2_best/checkpoint.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# 推理（注意：必须用 stochastic）
window = torch.randn(1, 8, 6)
pred = model.predict(window, deterministic=False)   # (1, 6)

# 获取 latent 分布（供 Stage 2 视觉网络修正）
mu, log_var = model.encode(window)                    # (1, 2), (1, 2)
```

或直接让 eval 脚本从 ckpt 自动推断：

```python
from vae.scripts.eval import infer_model_args
ckpt = torch.load("outputs/dim_2_best/checkpoint.pth", map_location="cpu")
model = HandActionVAE(**infer_model_args(ckpt["model"]))
model.load_state_dict(ckpt["model"])
```

`infer_model_args` 会从 state_dict 的权重 shape + key 名反推出 `action_dim, window_size, hidden_dim, latent_dim, encoder_type, num_hidden_layers`，所以**无论是哪一组超参训出的 ckpt 都不用改 eval 脚本**。

## 与标准 VAE 的区别

|                | 标准 VAE | 本模型 |
|----------------|---------|--------|
| Encoder 输入   | x       | 过去 8 帧动作 (8, 6) |
| Decoder 输出   | 重建 x  | **预测 t+1 时刻的动作** (6,) |
| 重建 Loss      | MSE(out, x) | MSE(out, GT_{t+1}) |
| KL Loss        | 同      | 同 |
| 目的           | 学习数据分布 | 学习时序动态先验 + 可修正的 latent space |

## 训练 Loss 解释

```
total_loss = recon_loss + β × kl_loss
```

| 指标 | 含义 |
|------|------|
| `recon` | MSE(预测 t+1, 真实 t+1)，预测精度 |
| `kl` | KL(q(z\|x) ‖ N(0,I))，latent 正则化 |
| `beta` | KL 权重系数，β-annealing 从 0 线性升到目标值 |
| `copy_baseline` | 直接用 a_t 作为 a_{t+1} 预测的 MSE，模型必须显著低于此值才有意义 |

## 关键设计

### β-Annealing
KL 权重从 0 线性升到 `beta`（默认 2000 步），避免训练早期 KL 项压制重建学习。

### 首帧填充
轨迹开头不足 8 帧时，用第一帧复制填充（**不**用 0 填充——0 在手部动作里有物理含义"完全松开"，零填充会引入虚假信号）。

### 最后帧处理
轨迹最后一帧的 t+1 目标 = t 时刻动作（保持不变）。

## 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window_size` | 8 | 输入历史帧数 |
| `--latent_dim` | **2** | Latent 维度（极窄瓶颈） |
| `--hidden_dim` | **256** | encoder/decoder 中间层宽度 |
| `--num_hidden_layers` | **1** | hidden→hidden 块数（1 = 原始 2-Linear 结构） |
| `--beta` | **0.001** | 目标 KL 权重 |
| `--beta_warmup` | 2000 | β 从 0 升到目标的步数 |
| `--noise_std` | **0.01** | 训练时输入窗口的高斯噪声 std |
| `--lr` | 2e-3 | 学习率 |
| `--total_steps` | 20000 | 训练步数 |
| `--batch_size` | 256 | 批大小 |
| `--encoder_type` | mlp | 也支持 `causal_conv` |

---

## Sweep findings（latent_dim=2 + MLP 严格约束下的实验结论）

在 `latent_dim=2`、`encoder_type=mlp` 的硬约束下，对 hidden_dim / depth / β / 训练步数做了系统 sweep，对比指标是 **free-run 100 步终态时 4 个手指（index/middle/ring/pinky）的均值**——参考 dim_32 baseline 这个值约为 0.78，即握紧。

### Top configs（每条都是 3 次 stochastic eval 平均）

| Config | hidden | depth | β | finger 终态均值 | 备注 |
|--------|--------|-------|---|----------------|------|
| dim_32 reference | 256 | 1 | 0.01 | ~0.78 | latent=32 上限 |
| **default (E1)** | **256** | **1** | **0.001** | **~0.68** | 🏆 latent=2 最优 |
| h256 d1 β=5e-4 | 256 | 1 | 5e-4 | ~0.62 | |
| h512 d1 β=1e-3 | 512 | 1 | 1e-3 | ~0.59 | 加宽无收益 |
| h256 d1 β=2e-3 | 256 | 1 | 2e-3 | ~0.57 | |
| h256 d1 β=1e-4 | 256 | 1 | 1e-4 | ~0.55 | β 过低开始过拟合 |
| h256 d1 β=1e-3 (40k 步) | 256 | 1 | 1e-3 | ~0.43 | 长训反而退化 |
| h384 d1 β=1e-3 | 384 | 1 | 1e-3 | ~0.40 + **NaN** | 不稳定 |
| h256 d2 β=1e-3 | 256 | **2** | 1e-3 | ~0.34 | **加深害死它** |
| h512 d2 β=5e-4 | 512 | 2 | 5e-4 | ~0.35 | |
| h512 d2 β=1e-4 | 512 | 2 | 1e-4 | ~0.17 | |
| 原 dim_2 baseline | 128 | 1 | 0.01 | ~0.50 | β 太大、hidden 太窄 |

best 配置在保留 dim_32 reference **约 87% 表征质量**的同时把 latent 严格压到 2 维。

### 5 条反直觉的 tricks（建议保留）

1. **β 是最关键旋钮——而且要按 latent_dim 反比缩放**
   β=0.01 对 latent=32 合理，但对 latent=2 来说**单维 KL 压力是 32 维时的 16 倍**，会把仅有的 2 个 latent 维度压扁。把 β 从 0.01 降到 **0.001** 是单一最大改进。但**不能再低**（β=1e-4 时 train recon 持续下降但 val recon 反弹，过拟合）。

2. **加宽 hidden_dim 没有收益**
   hidden=512 比 256 略差，hidden=384 甚至训出 NaN。瓶颈在 latent=2 这个 16x 压缩，加宽中间层不能跨过它，反而引入训练不稳定。**256 已经够用**。

3. **加深 MLP（num_hidden_layers ≥ 2）显著有害**
   fingers 终态从 ~0.68 暴跌到 ~0.34。猜测：极窄瓶颈下更深的网络梯度信号被稀释，更难找到能"穿过"latent 的稳定路径。**坚持 depth=1**。

4. **训练 40k 步比 20k 步更差**
   在窄瓶颈下长训练会让 AR rollout 退化（recon loss 变化很小，但 latent 几何漂移）。**20k 是甜点**。

5. **deterministic 模式（用 μ）下所有模型都"瘫痪"在初始状态** ⚠️
   这是最重要的隐藏现象：用 `model.predict(deterministic=True)` 时，free-run 出来的轨迹**几乎不动**——4 个手指停在 0 附近，thumb_rot 停在初始 0.4。
   原因：模型学到的不是用 μ 推动轨迹，而是用 reparameterize 注入的 ε 噪声驱动 rollout。`fc_mu` 输出对 history 的依赖很弱，"前进的能量"主要来自 sampling 噪声。

   **三个直接后果**：
   - **eval 必须用 stochastic 模式**（`--num_samples >> 1`，不要加 `--deterministic`）。
   - 单次 stochastic eval 有方差（指尖终态 ±0.05），公正比较时务必跑多次取均值。
   - **Stage 2 视觉网络必须能修正 σ（log_var），不能只修正 μ**——否则视觉信号无法驱动 rollout。

## Stage 2 集成接口

```python
# 1. 从 8 帧历史拿到先验分布
mu, log_var = model.encode(past_8_frames)        # (B, 2), (B, 2)

# 2. 视觉网络输出修正（σ 必须可改！见 trick #5）
delta_mu, delta_log_var = vision_net(image)       # (B, 2), (B, 2)
corrected_mu      = mu      + delta_mu
corrected_log_var = log_var + delta_log_var

# 3. 采样（必须 stochastic）并用冻结的 decoder 解码
z = model.reparameterize(corrected_mu, corrected_log_var)
pred_action = model.decode(z)                     # (B, 6) — decoder 权重冻结
```
