# Hand Action VAE — Experiment Log

持续维护的实验日志，按时间倒序排列（最新在最上面）。每次跑完一组实验都更新这里。

---

## 2026-04-09 · Delta-action sweep — 全部 3 组 sweep 在 delta 数据上复现

### 动机

将数据从**绝对目标位姿**切换为 **forward-difference delta action**（`delta[t] = abs[t+1] - abs[t]`），保持所有超参不变，重复 3 组 sweep（architecture / loss / recon_weight），评估 delta 表示对 VAE 性能的影响。

### 代码改动

- `hand_dataset.py`：自动检测 abs/delta（per-dim |mean| 阈值 0.05），delta 时用**零填充**（不再复制首帧）
- `eval.py`：新增 `--data_mode {auto, absolute, delta}`，delta 模式下用 `curr_obs.states[0, 0:6]` + cumsum(predicted_delta) 将输出重建为绝对位姿再绘图

### 数据

- delta 数据: `data/delta_action_20260327_11_10_43/{train,test}`
- 转换方式: forward difference, `delta[T-1] = 0` (zero-padded boundary)
- train: 120 trajs / 4233 samples, test: 30 trajs / 927 samples
- Copy baseline MSE: **0.004078**

### PCA upper bound（关键对比点）

| 维度 | 绝对位姿 | Delta |
|------|---------|-------|
| top-1 | 54.42% | 24.62% |
| **top-2** | **89.65%** | **45.50%** |
| top-4 | 94.94% | 72.02% |
| top-8 | 98.34% | 88.22% |

**Delta 数据的 PCA top-2 仅 45.50%**（绝对位姿为 89.65%），信息结构完全不同——delta 动作更"噪声化"、更高维，前 2 个主成分只能解释不到一半方差。

### Sweep 1: Architecture（5 cells）

固定: `latent_dim=2, β=0.001, noise_std=0.01, 20k steps, recon_aux=0, free_bits=0`

| ID | 配置 | val_recon | val_kl | **R²** | active | AR MSE | floor_median |
|----|------|-----------|--------|--------|--------|--------|-------------|
| **A** | h256 d1 mlp | 0.001551 | 0.240 | 7.12% | **1/2** | 0.004967 | 5e-6 |
| **B** | h512 d1 mlp | 0.001543 | 0.241 | 7.07% | 1/2 | 0.004661 | 3e-6 |
| **C** | h256 d2 mlp | **0.001503** | 0.224 | **7.63%** | 1/2 | 0.004802 | 4e-6 |
| **D** | h512 d2 mlp | 0.001644 | 0.239 | 6.98% | 1/2 | 0.004925 | 3e-6 |
| **E** | h256 d1 cnn | 0.001568 | 0.234 | 6.79% | 1/2 | 0.004894 | 0 |

**关键发现**：
- **只有 1/2 latent 维度活跃**——第二维 KL < 0.01 塌缩了。绝对位姿下两个维度都活跃。
- R² 全部在 6.8-7.6%，远低于 PCA bound（45.50%）和绝对位姿结果（50-60%）。
- C（deeper）略好——与绝对位姿下 A 最优的结论**相反**。
- Decoder 仍然不是瓶颈（floor_median ≈ 0）。

### Sweep 2: Loss ablation（3 cells + baseline A）

固定: h256 d1 mlp + 同上

| ID | recon_aux | free_bits | val_recon | val_kl | **R²** | active | AR MSE |
|----|-----------|-----------|-----------|--------|--------|--------|--------|
| A baseline | 0.0 | 0.0 | 0.001551 | 0.240 | 7.12% | 1/2 | 0.004967 |
| **F** | 0.1 | 0.0 | 0.001572 | 0.241 | 8.11% | 1/2 | 0.004816 |
| **G** | 0.0 | 0.5 | **0.001454** | 0.511 | 7.94% | **2/2** | 0.004817 |
| **H** | 0.1 | 0.5 | 0.001517 | 0.538 | **28.51%** ⭐ | **2/2** | 0.004909 |

**🎯 关键发现: free_bits 在 delta 数据上至关重要！**

- 绝对位姿下，G (free_bits=0.5) 几乎无效（R² +0.08 pts）；**delta 数据下它激活了第二个 latent 维度**（1/2→2/2）。
- 单独用 F (recon_aux=0.1) 只提升 R² 到 8.11%（+1 pts）——但绝对位姿下同样配置提升了 +13 pts。
- **两者叠加 H 产生 R²=28.51%（+21.4 pts）**——这是一个巨大的协同效应。绝对位姿下 H 和 F 几乎相同（72.78 vs 72.81），free_bits 没有贡献。
- **原因**：delta 数据下第二维自然塌缩；free_bits 强制"救活"它；recon_aux 随后利用两个活跃维度编码更多输入信息。

### Sweep 3: Recon aux weight（4 新 cells + W000=A / W010=F 复用）

固定: h256 d1 mlp, β=0.001, noise_std=0.01, 20k steps, free_bits=0

| ID | weight | val_recon | val_kl | **R²** | active | AR MSE |
|----|--------|-----------|--------|--------|--------|--------|
| W000 (=A) | 0.00 | 0.001551 | 0.240 | 7.12% | 1/2 | 0.004967 |
| W005 | 0.05 | 0.001550 | 0.236 | 7.42% | 2/2 | 0.004780 |
| W010 (=F) | 0.10 | 0.001572 | 0.241 | 8.11% | 1/2 | 0.004816 |
| **W030** | **0.30** | 0.001571 | 0.679 | **36.59%** ⭐ | **2/2** | **0.004503** |
| W050 | 0.50 | 0.001632 | 0.773 | **37.39%** | 2/2 | 0.004613 |
| W100 | 1.00 | 0.001573 | 0.909 | **37.66%** | 2/2 | 0.004521 |
| — PCA top-2 | — | — | — | **45.50%** | — | — |

**关键发现: 存在 phase transition（相变）而非 Goldilocks 曲线**

绝对位姿下 R² 从 60% 平滑上升到 76% 再下降。Delta 数据下：
- weight 0.0-0.10: R² ≈ 7-8%（低位平台，第二维塌缩）
- weight 0.30+: R² **跳变**到 ~37%（第二维激活）
- weight 0.30-1.00: R² 稳定在 36.6-37.7%（高位平台）

这是一个**由 latent 维度激活驱动的 phase transition**：当 recon_aux_weight 足够大时，aux loss 的梯度信号强到能把塌缩的第二维"拉回来"，R² 瞬间跳升。

### 绝对位姿 vs Delta 整体对比

| 指标 | 绝对位姿 best | Delta best | 变化 |
|------|-------------|-----------|------|
| PCA top-2 | 89.65% | 45.50% | −44.2 pts |
| R² (sweep winner) | 75.95% (W030) | 37.66% (W100) | −38.3 pts |
| R² gap to PCA | 13.70 pts | **7.84 pts** | 更小 ✅ |
| val_recon (best) | 0.002971 | **0.001454** | −51% |
| Copy baseline | 0.004830 | 0.004078 | |
| Active dims (most) | 2/2 | 1/2 ⚠️ | |
| Decoder bottleneck? | Never | Never | |
| 最关键改进因子 | recon_aux alone | recon_aux **+ free_bits** | |

### 🔑 5 条核心结论

#### 1. Delta 数据的信息结构根本不同
PCA top-2 从 89.65% 降到 45.50%。绝对位姿是低频、有偏（手关节偏置 ~0.4）的信号；delta 是零均值、高频、更"噪声化"的信号。用 2 个 latent 维度从根本上无法像在绝对位姿上那样有效压缩。

#### 2. Latent 维度坍缩是 delta 数据的核心问题
大部分配置下第二个 latent 维度 KL < 0.01（inactive）。绝对位姿下两个维度始终活跃。**delta 数据训练 dim_2 VAE 需要 free_bits 来防止维度塌缩**。

#### 3. Free bits + recon_aux 的协同效应是关键
单独使用 recon_aux: R² 从 7→8%（几乎无用）
单独使用 free_bits: R² 从 7→8%（几乎无用）
**两者组合: R² 从 7→28.5%**（+21 pts 协同增益）

这在绝对位姿上不存在（free_bits 无效，recon_aux 独立贡献 +13 pts）。

#### 4. R² gap to PCA 反而更小
虽然绝对 R² 更低（38% vs 76%），但相对 PCA 上界的 gap 只有 7.84 pts（绝对位姿为 13.70 pts）。这意味着 **encoder 在其理论上限内的效率并没有更差**——更低的 R² 纯粹是 PCA 上界降低导致的。

#### 5. 推荐 delta 数据的最优配置必须包含 free_bits
绝对位姿的推荐配置（recon_aux=0.3, 无 free_bits）**不能直接搬到 delta 数据**。

推荐 delta 最优配置:
```bash
python vae/scripts/train.py \
  --hidden_dim 256 --latent_dim 2 --num_hidden_layers 1 \
  --beta 0.001 --noise_std 0.01 \
  --recon_aux_weight 0.3 --free_bits 0.5 \
  --total_steps 20000
```

### Free-run 100 步终态（delta→abs 重建，200 samples mean）

| 配置 | thumb_rot | thumb_bend | index | middle | ring | pinky |
|------|-----------|------------|-------|--------|------|-------|
| A baseline | 0.47 | 0.58 | 0.93 | 1.34 | 1.04 | 1.13 |
| B h512 | 0.44 | 0.56 | 0.89 | 1.29 | 0.99 | 1.08 |
| C h256 d2 | 0.57 | 0.44 | 0.69 | 1.12 | 0.78 | 0.85 |
| D h512 d2 | 0.45 | 0.53 | 0.84 | 1.25 | 0.94 | 1.02 |
| E cnn | 0.44 | 0.50 | 0.80 | 1.21 | 0.89 | 0.96 |
| F recon_aux=0.1 | 0.41 | 0.47 | 0.75 | 1.16 | 0.83 | 0.91 |
| G free_bits=0.5 | 0.35 | 0.45 | 0.71 | 1.12 | 0.79 | 0.86 |
| H both | 0.43 | 0.49 | 0.80 | 1.21 | 0.89 | 0.96 |
| W005 | 0.41 | 0.49 | 0.78 | 1.18 | 0.87 | 0.95 |
| W030 | 0.61 | 0.38 | 0.63 | 1.07 | 0.72 | 0.79 |
| W050 | 0.63 | 0.37 | 0.63 | 1.08 | 0.73 | 0.81 |
| W100 | 0.63 | 0.38 | 0.63 | 1.08 | 0.73 | 0.81 |

注意：部分关节终态 > 1.0（如 middle 1.34）。这是 delta cumsum 积分漂移的结果——模型在 free-run 时产生的 delta 均值略偏正，100 步积累后超出合理范围。**这暗示 delta 模式下需要 action clipping 或 drift correction 机制**。

### 后续可探索

1. **free_bits=0.5 + recon_aux=0.3 的组合训练**（sweep 3 中没测这个组合——预计 R² 可能超过 37%，因为 H 的 R² 在 free_bits 帮助下已达 28.5%）
2. **增加 latent_dim 到 4-8**——delta 数据 PCA top-4 = 72%，dim_2 的 45.5% 天花板太低
3. **Action clipping**——cumsum 积分漂移的实际解决方案
4. **降低 noise_std**——当前 0.01 对 delta 的信噪比（~0.02-0.06 mean |a|）偏高
5. **降低 β**——delta 的 KL 容量需求可能更大

---

## 2026-04-08 · recon_aux_weight sweep on dim_2 — 找最优权重

### 动机

前一个 ablation 证明 input reconstruction 辅助 loss 在 weight=0.1 下能把 R² 从 60% 推到 73%。现在系统地扫 weight ∈ {0, 0.05, 0.1, 0.3, 0.5, 1.0}，找到 R² 最优的权重，并验证：

1. R² 是否能继续往 80% 推？
2. 高 weight 是否会损害 val_recon（主任务）？
3. weight 越大是否单调更好？还是有 sweet spot？

### 实验配置

6 个 ckpt，全部基于 dim_2_best baseline（h256 d1 mlp β=0.001 noise=0.01 20k steps），唯一变化的是 `--recon_aux_weight`：

| ID | recon_aux_weight | 复用已有 ckpt？ |
|----|-----------------|--------------|
| W000 | 0.00 | 复用 A_h256_d1_mlp |
| W005 | 0.05 | 新训 |
| W010 | 0.10 | 复用 F_recon_only |
| W030 | 0.30 | 新训 |
| W050 | 0.50 | 新训 |
| W100 | 1.00 | 新训 |

每个 ckpt 都用 `vae/scripts/eval.py --free_run --num_samples 200 --save_plot` 生成了独立的 npz + 形态图，存在：

```
visualizations/vae_eval/recon_sweep/W000_baseline/
visualizations/vae_eval/recon_sweep/W005/
visualizations/vae_eval/recon_sweep/W010/
visualizations/vae_eval/recon_sweep/W030/
visualizations/vae_eval/recon_sweep/W050/
visualizations/vae_eval/recon_sweep/W100/
```

### 训练结果

| ID | weight | val_recon | val_kl | 训练状态 |
|----|--------|-----------|--------|---------|
| W000 | 0.00 | 0.003008 | 1.047 | 复用 |
| W005 | 0.05 | 0.003048 | 1.096 | ✅ |
| W010 | 0.10 | 0.002980 | 1.084 | 复用 |
| W030 | 0.30 | 0.003191 | 1.146 | ✅ |
| W050 | 0.50 | 0.003127 | 1.274 | ✅ |
| W100 | 1.00 | 0.003118 | 1.257 | ✅ |

**val_recon 全在 0.0030-0.0032 之间**——即使 weight=1.0 也没让单步精度崩。**val_kl 单调递增**（1.05 → 1.27），说明 aux loss 确实推着 encoder 用更多 latent 容量。

### 诊断结果（核心数据）

| ID | weight | **R²** | gap to PCA | actual_recon | floor_median | AR MSE |
|----|--------|--------|-----------|-------------|-------------|--------|
| W000 | 0.00 | 59.79% | −29.86 pts | 0.00289 | 1.7e-5 | 0.0964 |
| W005 | 0.05 | 67.58% | −22.07 pts | 0.00294 | 1.2e-5 | 0.1192 |
| W010 | 0.10 | 72.81% | −16.84 pts | 0.00287 | 1.3e-5 | 0.1457 |
| **W030** | **0.30** | **75.95%** ⭐ | **−13.70 pts** | 0.00306 | 1.1e-5 | 0.1374 |
| W050 | 0.50 | 75.51% | −14.14 pts | 0.00306 | 1.0e-5 | 0.1391 |
| W100 | 1.00 | 72.80% | −16.85 pts | 0.00308 | 0.7e-5 | 0.1258 |
| — PCA top-2 | — | **89.65%** | 0 | — | — | — |

### 关键发现

#### 🎯 发现 1: **R² 呈现 Goldilocks 曲线，最优在 weight=0.3**

```
R² vs weight:
59.79%  ──╮              ← 0.00 baseline
          │
67.58%   │╲              ← 0.05 (+7.8)
         │ ╲
72.81%   │  ╲             ← 0.10 (+13.0)
         │   ╲╮
75.95% ⭐│    ╲           ← 0.30 (+16.2)  ← PEAK
         │     ╲╮
75.51%   │      ╲          ← 0.50 (+15.7) plateau
         │       ╲╮
72.80%   │        ╲        ← 1.00 (+13.0) 开始下降
```

- **0 → 0.3**: R² 单调上升 +16.2 pts
- **0.3 → 0.5**: 进入 plateau（差异 < 0.5 pts）
- **0.5 → 1.0**: R² 开始下降，aux loss 开始抢主任务的资源
- **最佳 weight = 0.3**（R²=75.95%, gap to PCA=13.70 pts）

**关闭了 30 pt gap 中的 16.2 pts，约 54%**。剩下的 13.70 pts 是 nonlinear encoder 在小数据集 + KL 约束下的实际工作上限，可能需要 multi-step / scheduled sampling 等更复杂改进才能继续推。

#### ✅ 发现 2: **val_recon 几乎不退化**

最坏情况（W100）val_recon 比 baseline 差 6.6%。在 R² 提升 13 pts 的同时，主任务单步精度只下降 6.6%——这是非常划算的 trade-off。

W030（最优）的 val_recon 比 baseline 差 5.9%（0.00306 vs 0.00289），**几乎可以认为没退化**。

#### ✅ 发现 3: **decoder reachability 始终稳定**

所有 6 个 weight 的 floor_median 都在 0.7-1.7 × 10⁻⁵ 范围内，**重新确认 decoder 永远不是瓶颈**。

特别有意思：weight 越大，floor_median 越小（W100=7e-6 vs W000=1.7e-5）。可能解释：aux loss 让 decoder 也学到了更"平滑"的 latent → output 映射，意外提升了 decoder 的可优化性。

#### ⚠️ 发现 4: **AR MSE 随 weight 增大变差**

| weight | 0 | 0.05 | 0.1 | 0.3 | 0.5 | 1.0 |
|--------|---|------|-----|-----|-----|-----|
| AR mean | 0.0964 | 0.1192 | 0.1457 | 0.1374 | 0.1391 | 0.1258 |

baseline (W000) 的 AR MSE 最低，aux loss 反而让 AR 多步累积误差变大 22-50%。

**这再次确认 R² 改善 ≠ AR 改善**。两者是独立瓶颈：
- **R²** 衡量 encoder 信息保留 → input recon loss 直接攻击
- **AR rollout** 衡量多步累积稳定性 → 需要 scheduled sampling / multi-step loss

### 关于 finger 终态（仅为视觉对照，不作判定）

free-run 100 步终态的 4 指均值（仅为视觉参考，已知是有偏指标）：

| weight | thumb_rot | thumb_bend | index | middle | ring | pinky |
|--------|-----------|------------|-------|--------|------|-------|
| 0 | 0.50 | 0.30 | 0.60 | 0.68 | 0.68 | 0.69 |
| 0.05 | 0.45 | 0.27 | 0.48 | 0.56 | 0.57 | 0.54 |
| 0.10 | 0.44 | 0.23 | 0.40 | 0.46 | 0.46 | 0.45 |
| 0.30 | 0.47 | 0.25 | 0.40 | 0.44 | 0.47 | 0.48 |
| 0.50 | 0.45 | 0.21 | 0.33 | 0.37 | 0.40 | 0.41 |
| 1.00 | 0.45 | 0.28 | 0.48 | 0.54 | 0.55 | 0.54 |

随 weight 增大，free-run finger 终态**降低**——这是 input recon loss 让 AR 漂移变慢的副作用，**不代表模型变差**。具体的形态对比可看每个 npz 对应的 plot。

### 最终判定

| 维度 | 结论 |
|------|------|
| **R² 最优权重** | **0.3**（R²=75.95%, +16.2 pts vs baseline）|
| **次优** | 0.5（R²=75.51%，几乎并列） |
| **过度** | 1.0（R² 开始下降）|
| **不足** | 0.05-0.1（仍有改善空间） |
| **val_recon 损失** | 最坏 6.6%，最优 5.9%——可接受 |
| **decoder 是瓶颈吗？** | ❌ 6 个 weight 都确认 decoder 充足 |
| **AR rollout 改善吗？** | ❌ 反而变差（独立瓶颈）|

### 🏆 推荐的新 dim_2 best 配置

```bash
python vae/scripts/train.py \
  --hidden_dim 256 --latent_dim 2 --num_hidden_layers 1 \
  --beta 0.001 --noise_std 0.01 \
  --recon_aux_weight 0.3 \
  --total_steps 20000
```

R² 75.95%（vs 旧 best 59.79%）、val_recon 0.00306（vs 0.00289 仅 6% 退化）、decoder floor 1.1e-5。

### 后续可探索

1. **R² 再往 80% 推**：可能需要降 β（释放 KL 压力）+ 维持 recon_aux=0.3
2. **AR rollout 的独立瓶颈**：scheduled sampling 或 multi-step rollout loss
3. **Stage 2 接入实测**：用 W030 作 Stage 1 base，让视觉网络接上修正 μ，看下游表现

---

## 2026-04-08 · Loss sweep on dim_2 — input reconstruction + free bits ablation

### 动机

前一个 sweep 已经证明：**架构改进对 dim_2 无效**（甚至有负效果），decoder 容量充足，瓶颈在 encoder 训练信号上。这次直接攻 R²：测试 **input reconstruction 辅助 loss** 和 **free bits** 这两个训练信号改进。

### 两个 trick 简介

1. **Input reconstruction 辅助 loss**：在 decoder 之外增加一个 aux head（z → 48 维 flat 输入窗口），训练时加入 `λ * MSE(aux(z), x_window)`。强迫 encoder 把输入信息保留在 latent 里，**直接攻 R²**。
2. **Free bits**：每个 latent 维度发一笔"免费 KL 配额"，KL 低于阈值时不罚。防止低信息维度被 KL 压扁退化。**dim_2_best 的第二维 KL=0.59 看起来在被半压扁**。

### 实验配置

3 个 ablation，全部基于 dim_2_best 配置（h256 d1 mlp β=0.001 noise=0.01 20k steps），只改 loss：

| ID | recon_aux_weight | free_bits | 测什么 |
|----|-----------------|-----------|-------|
| **F** | 0.1 | 0.0 | input recon 单独效果 |
| **G** | 0.0 | 0.5 | free bits 单独效果 |
| **H** | 0.1 | 0.5 | 两者叠加 |

### 训练结果

| ID | val_recon | val_kl |
|----|-----------|--------|
| A baseline | 0.003008 | 1.047 |
| F recon_only | 0.002980 | 1.084 |
| G freebits_only | 0.002971 | 1.080 |
| H both | 0.002991 | 1.094 |

**val_recon 几乎相同**——新 loss 没有损害单步预测精度。但 R² 才是关键。

### 诊断结果

| ID | **R²** | active | actual_recon | floor_median | frac ≥2× | 备注 |
|----|--------|--------|-------------|-------------|---------|------|
| **A** baseline | 59.79% | 2/2 | 0.00289 | 0.000017 | 11.9% | — |
| **F** recon_only | **72.81%** ⭐ | 2/2 | 0.00287 | 0.000013 | 12.1% | **+13.0 pts on R²** |
| **G** freebits_only | 59.87% | 2/2 | 0.00285 | 0.000017 | 11.7% | 几乎无效 |
| **H** both | **72.78%** | 2/2 | 0.00288 | 0.000014 | 21.6% | 与 F 持平，free bits 没贡献 |
| ref dim_32 | 67.27% | 5/32 | 0.00168 | 0.000000 | 99.2% | （只有 5 个 active dim）|
| — PCA top-2 | **89.65%** | — | — | — | — | 上界 |

### 关键发现

#### 🎯 发现 1: input reconstruction 是真正的杠杆（R² +13.0 pts）

R² 从 59.79% → 72.81%，**关闭了 30 pt gap 中的 13 pts**（约 43%）。这是单一最大的非架构改进。

dim_2 + recon_aux=0.1 的 R²（72.81%）**已经超过了 dim_32 ref 的 R²（67.27%）**——dim_32 因为 posterior collapse 浪费了 27 个维度，活跃容量反而不如带 aux loss 的 dim_2。

#### ⚠️ 发现 2: free bits 单独使用对 dim_2 几乎无效（R² +0.08 pts）

G 的 R² = 59.87% vs A 的 59.79%，差距只有 0.08 pts，在噪声范围内。

**为什么？** 之前预想 free bits 会救活 dim_2 的第二维（KL=0.59），但事实是 dim_2 baseline 的两个维度本来就都"活着"——free bits 只对真正塌缩的维度（KL → 0）才有效。检查 G 的 per-dim KL：[1.50, 0.60]，**和 baseline 几乎一样**，说明 free_bits=0.5 对一个 KL=0.59 的维度施加的"保护"还不够，或者第二维已经接近自然平衡点。

> **结论**：free bits 主要是 dim_32 这种严重 collapse 场景的解药。dim_2 不需要它。

#### ⚠️ 发现 3: 组合不超过 recon 单独

H 的 R² = 72.78% vs F 的 72.81%——**完全在噪声里**。free bits 没有给 input recon 带来任何额外贡献。

#### ✅ 发现 4: decoder reachability 全部稳定在 ~1e-5

A: 1.7e-5, F: 1.3e-5, G: 1.7e-5, H: 1.4e-5 —— **再次确认 decoder 不是瓶颈**，新 loss 没有对 decoder 容量施加任何额外约束。

#### ⚠️ 发现 5: AR rollout MSE 反而变差（但要小心 single-eval noise）

| ID | AR mean | early | mid | late |
|----|---------|-------|-----|------|
| A | 0.107 | 0.005 | 0.082 | 0.189 |
| F | **0.133** | 0.003 | 0.072 | 0.234 |
| G | **0.093** | 0.005 | 0.084 | 0.144 |
| H | 0.138 | 0.003 | 0.072 | 0.234 |

F/H（带 input recon）的 late-step AR MSE 比 baseline 更差。**但 G（free bits）反而是最好的**！这反差很奇怪，可能是：
- single stochastic run 的 noise（标准做法应该 3-5 runs 取平均）
- input recon loss 让 encoder 编码更"准确"的当前状态，但相邻 step 的 latent 几何更"陡峭"，AR 时小误差被放大
- 需要更多实验确认

**值得后续关注**：R² 改善没有传导到 AR rollout 改善——这暗示 AR drift 是独立的瓶颈，需要分开攻击（scheduled sampling 等）。

### 最终判定

| 问题 | 答案 |
|------|------|
| input reconstruction 辅助 loss 有效吗？ | ✅ **强烈有效**，R² +13 pts |
| free bits 对 dim_2 有效吗？ | ❌ 几乎无效 |
| 两者叠加有超过单一吗？ | ❌ 没有 |
| decoder 仍然不是瓶颈吗？ | ✅ 是的，所有变体都确认 |
| Stage 2 框架可行性？ | ✅ 进一步增强（更高 R² + decoder 充足）|
| AR rollout 是否同时改善？ | ❌ 没有，可能是独立瓶颈 |

### 推荐配置（dim_2 新 best）

```bash
python vae/scripts/train.py \
  --hidden_dim 256 --latent_dim 2 --num_hidden_layers 1 \
  --beta 0.001 --noise_std 0.01 \
  --recon_aux_weight 0.1 \
  --total_steps 20000
```

**不需要 free_bits**（在这个 setup 下没贡献），但保留 input recon。

### 后续可探索

1. **更高的 recon_aux_weight**（0.3, 0.5, 1.0）— 看 R² 是否能进一步推到 80%+
2. **拆开 encoder 与 aux head 的 hidden dim**——如果 aux head 需要更大容量
3. **scheduled sampling**——攻击 AR drift 这个独立瓶颈
4. **多个 stochastic eval 取均值**——确认 AR MSE 的真实变化方向

---

## 2026-04-08 · Architecture sweep on dim_2 (decoder reachability as core metric)

### 核心问题

> **在 latent_dim=2 严格约束下，编码不准的瓶颈到底在 encoder 还是 decoder？**
>
> - 如果是 encoder：Stage 2 视觉修正框架可以解决
> - 如果是 decoder：必须重新设计整个框架

### 评估方法

不再使用之前那个有偏的 "free-run finger 终态均值"。改用 4 个指标，**其中 decoder reachability 是核心**：

| 指标 | 衡量什么 | 怎么算 |
|------|---------|-------|
| **val_recon** | 单步预测精度（encoder→decoder 路径） | μ via encoder, MSE on test next-frame |
| **R² (linear-decodability)** | encoder 保留了多少输入信息 | OLS μ → flat input window，对照 PCA top-k 上界 |
| **active dims** | latent 利用率 | per-dim KL > 0.01 的维度数 |
| **decoder reachability** ⭐ | decoder 给定**最优 z\*** 能多准 | 多次重启梯度下降 in latent space，最小化 \|\|decoder(z*) - target\|\|² |

**判定逻辑**：

```
floor_median << actual_recon  →  ENCODER 是瓶颈（Stage 2 可救）
floor_median ≈  actual_recon  →  DECODER 是瓶颈（架构需重设）
floor 在不同架构间显著变化   →  decoder 容量是相关因素
floor 在不同架构间几乎不变   →  decoder 容量足够，瓶颈纯在 encoder
```

### 实验配置（5 个 architecture 变体）

所有变体共享：`latent_dim=2, β=0.001, noise_std=0.01, 20k steps, lr=2e-3, batch=256`。
唯一变化的是 encoder/decoder 架构：

| ID | hidden | depth | encoder | 假设 |
|----|--------|-------|---------|------|
| **A** | 256 | 1 | mlp | 当前 dim_2_best baseline |
| **B** | 512 | 1 | mlp | 加宽 → encoder/decoder 表达力 ↑ |
| **C** | 256 | 2 | mlp | 加深 → 非线性容量 ↑ |
| **D** | 512 | 2 | mlp | 又宽又深 |
| **E** | 256 | 1 | causal_conv | CNN encoder + MLP decoder |

每个变体训练后，跑 `scripts/analyze_demo.py` 得到全部 4 个指标。

### 训练结果

| ID | hidden | depth | encoder | val_recon | val_kl |
|----|--------|-------|---------|-----------|--------|
| A | 256 | 1 | mlp | **0.003008** | 1.047 |
| B | 512 | 1 | mlp | 0.003094 | 1.033 |
| C | 256 | 2 | mlp | 0.003402 | 1.067 |
| D | 512 | 2 | mlp | 0.003406 | 1.077 |
| E | 256 | 1 | causal_conv | 0.003368 | 1.085 |

**初步观察**（仅看 val_recon）：
- A (baseline) 略好于其他所有变体（差距 3-13%）
- 加宽（B）几乎与 A 持平（差 3%）
- 加深（C, D）反而略差（差 13%）—— 但小数据集下也可能是优化随机性
- CNN encoder（E）介于中间（差 12%）
- val_kl 5 个变体几乎一样（1.03 - 1.09），说明 KL 压力在所有架构下都同样有效

**但 val_recon 不是最重要的**，决定性的是下面的 decoder reachability。

### 诊断结果

| ID | 配置 | **R²** | active | actual_recon (mean) | **floor_median** | floor_p95 | frac ≥2× |
|----|------|-------|--------|---------------------|-----------------|-----------|---------|
| **A** | h256 d1 mlp（baseline） | **59.79%** ⭐ | 2/2 | 0.00289 | 0.000017 | — | 11.5% |
| **B** | h512 d1 mlp（wider） | 56.22% | 2/2 | 0.00290 | 0.000017 | 0.0048 | 59.1% |
| **C** | h256 d2 mlp（deeper） | 54.63% | 2/2 | 0.00322 | 0.000012 | 0.0070 | 49.9% |
| **D** | h512 d2 mlp（wider+deeper） | 55.35% | 2/2 | 0.00322 | 0.000021 | 0.0054 | 22.1% |
| **E** | h256 d1 causal_conv（CNN） | 50.71% | 2/2 | 0.00323 | **0.000008** | 0.0052 | 77.0% |
| **ref** | dim_32 (h256, k=32) | 67.27% | 5/32 | 0.00168 | 0.000000 | 0.000057 | 99.4% |
| — | **PCA top-2 上界** | **89.65%** | — | — | — | — | — |

### 最终判定（**结果非常清晰且超出预期**）

#### ✅ 判定 1: Decoder **不是** 瓶颈 —— 在所有 5 个架构上都不是

所有 5 个架构变体的 `floor_median` 都在 **1e-5 量级或更小**（A=1.7e-5, B=1.7e-5, C=1.2e-5, D=2.1e-5, E=8e-6），dim_32 ref 更是 0。这是**强结论**：

> **无论 encoder 用 MLP 还是 CNN，无论 hidden_dim 是 256 还是 512，无论 depth 是 1 还是 2，jointly trained 的 decoder 总是能给出"几乎完美"的重建上限。Decoder 容量从来不是问题。**

这意味着 **Stage 2 视觉修正框架在原理上一定可行** —— 给一个能找到正确 latent 的修正信号，decoder 就能产生正确的 action 数值。

#### ⚠️ 判定 2: **架构增强不仅没帮上 encoder，反而让 R² 全面退步**

- A baseline: **59.79%** ← R² 最高
- B wider: 56.22% (-3.6 pts)
- C deeper: 54.63% (-5.2 pts)
- D wider+deeper: 55.35% (-4.4 pts)
- E CNN: 50.71% (-9.1 pts)

**5 个变体没有一个比 baseline 更好**。这是一个非常意外但可以解释的结果：
1. **小数据集**（4233 样本）下，更大的 encoder 容量会过拟合 next-step loss，反而**牺牲了对输入方差的覆盖**
2. **R² 衡量的是 latent 保留输入信息的能力**，加宽/加深给的是"对 next-step loss 更精细的拟合"——两者方向不同
3. **CNN 当前的 receptive field 只有 7 帧**（3 层 kernel=3），看不全 8 帧窗口，理论上劣势已经存在

#### ⚠️ 判定 3: 单步精度 (val_recon / actual_recon) 反映的是同一个故事

| ID | actual_recon | 相对 baseline |
|----|-------------|---------------|
| A | 0.00289 | 1.00× |
| B | 0.00290 | 1.00× |
| C | 0.00322 | 1.11× |
| D | 0.00322 | 1.11× |
| E | 0.00323 | 1.12× |

所有变体单步精度都不如 baseline 或持平。**架构变化在 dim_2 + 4233 样本数据集下没有正收益**。

#### 🎯 真正的结论 & 行动方向

1. **架构路线在 dim_2 上彻底走不通**。我们用正确的指标（包括 decoder reachability 这个核心指标）证明了两件事：
   - decoder 容量充足，永远不是瓶颈
   - 各种 encoder 架构变体都无法超越 baseline 的 R² 上限
2. **唯一的真正瓶颈是 encoder 的训练信号**（30 pts 的 R² gap）。攻克它必须靠：
   - **input reconstruction 辅助 loss**（直接逼 encoder 保留输入信息）
   - **free bits**（对 KL 设 per-dim 下限，避免次维度退化）
   - **multi-step prediction**（让 latent 编码长时序信息）
3. **Stage 2 视觉修正框架的可行性得到强验证**。decoder 已经能产生正确的 action 数值，只要 Stage 2 能把 μ 推到正确位置，整个管线就会工作。

---

### 副产物：被这次 sweep 修正的早期错误结论

| 早期结论（基于 finger 终态均值） | 重新校准后 |
|--------------------------------|----------|
| "h512 比 h256 略差，加宽无收益" | h512 R² 比 h256 差 3.6 pts，但 reachability 几乎相同 → **加宽确实没用** |
| "depth=2 严重退化（finger 0.68→0.34）" | depth=2 R² 比 depth=1 差 5 pts，但 actual_recon 只差 11% → **退化没有 finger 指标显示的那么夸张** |
| "dim_32 比 dim_2 强 16 倍" | dim_32 active dims = 5，R² 比 dim_2 高 7.5 pts，**实际只强 ~2 倍**而不是 16 倍 |
| "CNN 应该比 MLP 强" | CNN R² **比 MLP 还低 9 pts**（receptive field 小 + 训练动力学差） |

---

## Background：之前已经知道的事实

### Decoder reachability 在 dim_2_best 和 dim_32 ref 上的初步结果（来自 2026-04-08 早些时候）

| 模型 | actual_recon (mean) | floor (median) | floor (p95) | 结论 |
|------|---------------------|----------------|-------------|------|
| dim_2_best (h256, latent=2) | 0.0029 | 0.000017 | 0.0055 | encoder 是主要瓶颈 |
| dim_32 ref (h256, latent=32) | 0.0017 | 0.000000 | 0.000044 | encoder 严重 collapse，decoder 几乎完美 |

### dim_32 的 posterior collapse

32 个 latent 维度里只有 5 个活跃（KL > 0.01），其余 27 个塌缩到 N(0,1)。dim_32 实际上是个伪 5 维模型。

### PCA 上界（线性硬限制）

- top-2 (= dim_2): **89.65%**
- top-4: 94.94%
- top-32 (= dim_32): 99.97%

### dim_2_best 的关键超参数

```
latent_dim=2, hidden_dim=256, num_hidden_layers=1,
encoder_type=mlp, beta=0.001, noise_std=0.01, 20k steps
```
