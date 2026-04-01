# Hand VQ-VAE: Dexterous Hand Action Tokenizer

## Overview

将 6 维灵巧手连续动作离散化为 2 个 token（每个取值 0-3），共 4×4=16 种组合。用于下游 VLA 模型的动作 token 化。

```
Hand Action (6-dim, continuous)
        ↓
   MLP Encoder (6 → 128 → 128 → 32)
        ↓
   Residual VQ (2 layers × 4 entries)
        ↓  ← 输出 2 个离散 token
   MLP Decoder (32 → 128 → 128 → 6)
        ↓
Reconstructed Action (6-dim)
```

## Quick Start

```bash
# 训练
python train_vae/scripts/train_hand_vqvae.py \
    --train_dir data/20260327-11:10:43/demos/success/train \
    --test_dir data/20260327-11:10:43/demos/success/test \
    --output_dir outputs/hand_vqvae

# 自定义参数
python train_vae/scripts/train_hand_vqvae.py \
    --train_dir data/20260327-11:10:43/demos/success/train \
    --test_dir data/20260327-11:10:43/demos/success/test \
    --output_dir outputs/hand_vqvae_v2 \
    --codebook_size 8 \
    --num_vq_layers 3 \
    --total_steps 20000 \
    --lr 5e-4
```

## 模型使用

```python
from prismatic.action_vqvae.hand_vqvae import HandVQVAE

# 加载模型
model = HandVQVAE()
ckpt = torch.load("outputs/hand_vqvae/checkpoint.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# 编码: 连续动作 → 离散 token
action = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0]])  # (B, 6)
tokens = model.encode(action)          # (B, 2), 每个值 ∈ {0,1,2,3}

# 解码: 离散 token → 连续动作
recon = model.decode_from_indices(tokens)  # (B, 6)
```

## 文件结构

```
prismatic/action_vqvae/
├── hand_vqvae.py         # 模型: SimpleVQ, SimpleResidualVQ, HandVQVAE
├── hand_dataset.py       # 数据集: HandActionDataset
├── ...                   # 原始仓库代码 (未修改)

train_vae/scripts/
├── train_hand_vqvae.py   # 训练脚本
├── ...                   # 原始仓库代码 (未修改)
```

## Architecture Details

### Encoder / Decoder

3 层 MLP，SiLU 激活函数：

```
Encoder: Linear(6→128) → SiLU → Linear(128→128) → SiLU → Linear(128→32)
Decoder: Linear(32→128) → SiLU → Linear(128→128) → SiLU → Linear(128→6)
```

总参数量: ~43K

### Residual Vector Quantization

2 层残差量化，每层维护一个 4×32 的 codebook：

```
Layer 0: 量化 z             → quantized_0, index_0
         residual = z - quantized_0

Layer 1: 量化 residual      → quantized_1, index_1

输出:    quantized_0 + quantized_1 (用于解码)
         [index_0, index_1]       (2 个离散 token)
```

第一层捕获主要结构（如手张开/闭合），第二层修正残差细节。

## 关键 Tricks

### 1. EMA Codebook 更新 (替代 learnable codebook)

Codebook 不通过梯度下降更新，而是用 Exponential Moving Average 跟踪被分配到每个 entry 的编码器输出的均值：

```python
ema_count = decay * ema_count + (1 - decay) * assignment_count
ema_weight = decay * ema_weight + (1 - decay) * sum_of_assigned_vectors
codebook = ema_weight / ema_count
```

**为什么**: Learnable codebook 在小 codebook (4 entries) 下极易坍塌——梯度只更新被选中的 entry，未被选中的 entry 永远不动，导致所有样本都映射到同一个 entry。EMA 更稳定，因为它直接用统计量更新。

### 2. 数据驱动初始化 (替代随机初始化)

首次 forward 时，从输入 batch 中随机抽取样本作为 codebook 初始向量：

```python
if not self.initialized:
    idx = torch.randperm(x.shape[0])[:codebook_size]
    self.codebook = x[idx]
```

**为什么**: 随机初始化的 codebook 向量可能远离数据分布，导致初始阶段所有样本都 map 到同一个最近的 entry，后续 EMA 更新也无法恢复。数据初始化确保每个 entry 从一开始就在数据流形上。

### 3. Dead Code 重置

每步检查 EMA 计数，将几乎没被使用的 entry (count < 1) 替换为当前 batch 的随机编码器输出：

```python
dead_mask = self.ema_count < 1.0
self.codebook[dead_mask] = random_encoder_outputs
```

**为什么**: 即使初始化得当，训练过程中某些 entry 也可能逐渐失去吸引力（被其他 entry 抢走样本）。Dead code 重置给了它们"第二次机会"，确保 codebook 利用率最大化。

### 4. Straight-Through Estimator (STE)

Argmin 操作不可微，通过 STE 让梯度直接从量化后的向量传到编码器：

```python
quantized = x + (quantized - x).detach()
# 前向: quantized 的值
# 反向: 梯度等于 x 的梯度 (跳过 argmin)
```

### 5. Commitment Loss

迫使编码器输出靠近 codebook 向量，防止编码器"逃离" codebook 的覆盖范围：

```
total_loss = MSE(x, x_recon) + 5.0 * MSE(z_encoder, z_codebook.detach())
```

权重 5.0 沿用原始仓库的设定。如果训练不稳定可降低到 1.0 或 0.25。

## 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 256 | ~16 steps/epoch |
| `lr` | 1e-3 | AdamW 学习率 |
| `min_lr` | 1e-5 | Cosine 衰减终点 |
| `total_steps` | 10,000 | ~600 epochs |
| `warmup_steps` | 500 | 线性预热 |
| `weight_decay` | 1e-4 | L2 正则 |
| `clip_grad` | 1.0 | 梯度裁剪 |
| `latent_dim` | 32 | VQ bottleneck 维度 |
| `hidden_dim` | 128 | MLP 隐层宽度 |
| `codebook_size` | 4 | 每层 entry 数 |
| `num_vq_layers` | 2 | 残差 VQ 层数 |
| `commitment_weight` | 5.0 | VQ loss 权重 |

## 训练结果 (默认配置)

```
Test MSE:              0.0019
Codebook utilization:  15/16 (93.8%)
Layer 0:               4/4 entries used
Layer 1:               4/4 entries used
Dominant combo:        (L0=1, L1=3) = 72.6%  ← 手部静止状态 [0.4, 0, 0, 0, 0, 0]
```

Combo 分布不均匀是正常的，反映了数据本身的分布：大部分时刻手处于静止握持状态，仅少数时刻执行抓取动作。

## 扩展方向

如果 16 种 token 组合不足以达到所需的重建精度：

| 方案 | Token 组合数 | 改动 |
|------|-------------|------|
| 增大 codebook | 8×8=64 | `--codebook_size 8` |
| 增加 VQ 层数 | 4×4×4=64 | `--num_vq_layers 3` |
| 两者结合 | 8×8×8=512 | `--codebook_size 8 --num_vq_layers 3` |

增大后注意监控 codebook 利用率，避免 entry 浪费。
