# Hand VQ-VAE: Dexterous Hand Action Tokenizer

## Overview

将 Ruiyan 灵巧手 6 维连续动作离散化为 2 个 token（每个取值 0-3），共 4×4=16 种组合，并通过 grip score 排序为 0-15 的有序索引（0=全开, 15=全握）。用于下游 VLA 模型的动作 token 化。

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
        ↓
   Grip-score Reindex → 有序索引 0-15
```

## 文件结构

```
vq-vae/
├── model/
│   ├── hand_vqvae.py       # 模型: SimpleVQ, SimpleResidualVQ, HandVQVAE
│   ├── hand_dataset.py     # 数据集: HandActionDataset (.pt → hand actions)
│   └── utils.py            # 工具: cosine_scheduler
├── scripts/
│   ├── train.py             # 训练 VQ-VAE
│   ├── decode_codebook.py   # 打印 16 种 token → 关节角度 (rad)
│   ├── reindex_codebook.py  # Grip-score 排序 → sorted_codebook.npy
│   ├── visualize_codebook.py# PyBullet 可视化 (原始 + 排序 两组图)
│   └── prepare_dataset.py   # 数据清洗 + train/test 划分
└── README.md
```

## Pipeline

### Step 1: 数据准备

```bash
python vq-vae/scripts/prepare_dataset.py \
    --input_dir data/20260327-11:10:43/demos
```

输出:
- `demos/success/train/` — 120 条成功轨迹（去除开头零帧）
- `demos/success/test/` — 30 条成功轨迹
- `demos/failure/` — 2 条失败轨迹

可选参数: `--train_ratio 0.9`, `--strip_mode both`, `--seed 123`

### Step 2: 训练 VQ-VAE

```bash
python vq-vae/scripts/train.py \
    --train_dir data/20260327-11:10:43/demos/success/train \
    --test_dir data/20260327-11:10:43/demos/success/test \
    --output_dir outputs/hand_vqvae
```

输出: `outputs/hand_vqvae/checkpoint.pth`

### Step 3: 查看 Codebook

```bash
# 打印 16 种 token 对应的关节角度
python vq-vae/scripts/decode_codebook.py

# 指定其他 checkpoint
python vq-vae/scripts/decode_codebook.py --ckpt outputs/hand_vqvae_v2/checkpoint.pth
```

### Step 4: Token 重排 (Grip-score Reindex)

```bash
python vq-vae/scripts/reindex_codebook.py
```

输出:
- `outputs/hand_vqvae/sorted_codebook.npy` — 排序后的 16×6 动作表
- `outputs/hand_vqvae/sorted_codebook_meta.npz` — 排序元信息

排序逻辑: 忽略拇指旋转 (dim 0)，按其余 5 个关节弯曲程度的均值排序。0=全开, 15=全握。

### Step 5: 可视化

```bash
# 需要在 vqvla conda 环境下运行 (依赖 pybullet)
/home/admin01/anaconda3/envs/vqvla/bin/python vq-vae/scripts/visualize_codebook.py
```

输出:
- `visualizations/codebook_poses/codebook_grid_original.png` — 原始 4×4 网格
- `visualizations/codebook_poses/codebook_grid_sorted.png` — 排序后 2×8 网格
- `visualizations/codebook_poses/sorted_*.png` — 16 张单独图片

## 模型使用

```python
import torch
from vq-vae.model.hand_vqvae import HandVQVAE  # 或用 importlib 加载

model = HandVQVAE()
ckpt = torch.load("outputs/hand_vqvae/checkpoint.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# 编码: 连续动作 → 离散 token
action = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0]])
tokens = model.encode(action)               # (B, 2), 每个值 ∈ {0,1,2,3}

# 解码: 离散 token → 连续动作
recon = model.decode_from_indices(tokens)    # (B, 6)

# 下游策略使用排序后的 codebook
import numpy as np
codebook = np.load("outputs/hand_vqvae/sorted_codebook.npy")  # (16, 6)
# hand_action → find nearest in codebook → index ∈ [0,15] → normalize to [-1,1]
# 策略预测 1 个标量而不是 6 维动作
```

## 动作空间

6 维输入对应 Ruiyan 灵巧手的 6 个电机 (归一化 [0, 1]):

| Dim | 关节 | URDF Joint | 满量程角度 |
|-----|------|------------|-----------|
| 0 | 拇指旋转 | `joint_1_1` | 135° (2.356 rad) |
| 1 | 拇指弯曲 | `joint_1_2` | 40° (0.698 rad) |
| 2 | 食指弯曲 | `joint_2_1` | 87° (1.518 rad) |
| 3 | 中指弯曲 | `joint_3_1` | 90° (1.571 rad) |
| 4 | 无名指弯曲 | `joint_4_1` | 90° (1.571 rad) |
| 5 | 小指弯曲 | `joint_5_1` | 89° (1.545 rad) |

转换: `joint_angle_rad = action_value × joint_upper_limit`

## 模型架构

### Encoder / Decoder

3 层 MLP，SiLU 激活:
```
Encoder: Linear(6→128) → SiLU → Linear(128→128) → SiLU → Linear(128→32)
Decoder: Linear(32→128) → SiLU → Linear(128→128) → SiLU → Linear(128→6)
```

### Residual Vector Quantization

2 层残差量化，每层 4×32 的 codebook:
```
Layer 0: 量化 z             → quantized_0, index_0
         residual = z - quantized_0
Layer 1: 量化 residual      → quantized_1, index_1
输出:    quantized_0 + quantized_1 → 解码
         [index_0, index_1]       → 2 个 token
```

### 关键 Tricks

1. **EMA Codebook 更新**: 指数移动平均跟踪样本均值，避免梯度稀疏导致 codebook 坍塌
2. **数据驱动初始化**: 首批数据初始化 codebook entries，确保每个 entry 从起点就在数据分布内
3. **Dead Code 重置**: 检测未使用的 entry 并用随机编码器输出替换
4. **Straight-Through Estimator**: `quantized = x + (quantized - x).detach()` 让梯度跳过 argmin
5. **Commitment Loss**: `total_loss = MSE_recon + 5.0 × MSE_commit`

## 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 2e-3 | AdamW 学习率 (经实验验证最稳定) |
| `--min_lr` | 1e-5 | Cosine 衰减终点 |
| `--total_steps` | 10,000 | 训练步数 (~600 epochs) |
| `--warmup_steps` | 500 | 线性预热 |
| `--batch_size` | 256 | 批大小 |
| `--latent_dim` | 32 | VQ bottleneck 维度 |
| `--hidden_dim` | 128 | MLP 隐层宽度 |
| `--codebook_size` | 4 | 每层 entry 数 |
| `--num_vq_layers` | 2 | 残差 VQ 层数 |
| `--commitment_weight` | 5.0 | VQ loss 权重 |

## 训练结果

```
Test MSE:              0.0019
Codebook utilization:  15/16 (93.8%)
Layer 0:               4/4 entries used
Layer 1:               4/4 entries used
```

## 扩展

| 方案 | Token 组合数 | 改动 |
|------|-------------|------|
| 增大 codebook | 8×8=64 | `--codebook_size 8` |
| 增加 VQ 层数 | 4×4×4=64 | `--num_vq_layers 3` |
| 两者结合 | 8×8×8=512 | `--codebook_size 8 --num_vq_layers 3` |
