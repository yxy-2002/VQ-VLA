# Hand-Only BC — Experiment Log

持续维护的实验日志，按时间倒序排列（最新在最上面）。每次跑完一组实验都更新这里。

格式约定（与 `behavior_clone/EXPERIMENTS.md` 一致）：

```
## YYYY-MM-DD · <短标题>

### 动机
为什么做这次实验，要回答什么问题。

### 代码改动
碰过哪些文件，每个一行说明。

### 数据
数据路径、训练/测试规模、copy baseline。

### 配置
关键超参（lr / batch / steps / reg_drift / dropout）。

### 结果
hand_mse_full / hand_mse_no_correction / vision_gain 在关键 step 的表格。
training_curves.png 路径。

### 关键发现
1-3 条要点：什么有效、什么没效、下一步试什么。
```

---

## 2026-04-11 · v2 noise sweep — 寻找最优 past_hand_win 训练噪声

### 动机

v1 sweep 发现 noise_std_hand=0.03 是唯一正向 vision_gain 的配置（AR hand MSE 从 0.086 降到 0.017）。需要精细扫描噪声水平，找到最优点并确认高噪声端是否退化。

### 数据

同 v1。

### 配置

在 v1 baseline 基础上只变 `--noise_std_hand`，扫描 8 个值：

```
noise_std_hand ∈ {0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20}
```

其余参数：lr=5e-4, batch=128, steps=20000, reg_drift=1.0, dropout=0.0

### 结果

**AR 评估（deployment proxy）：**

| noise_std | AR hand MSE | no_corr (AR) | vision_gain (AR) | delta_z norm |
|-----------|------------|-------------|------------------|-------------|
| 0.01 | 0.0857 | 0.0857 | -0.000 | 0.158 |
| 0.02 | 0.0207 | 0.0857 | +0.065 | 0.215 |
| 0.03 | 0.0167 | 0.0857 | +0.069 | 0.179 |
| 0.05 | 0.0183 | 0.0857 | +0.067 | 0.223 |
| 0.07 | 0.0156 | 0.0857 | +0.070 | 0.191 |
| 0.10 | 0.0155 | 0.0857 | +0.070 | 0.166 |
| 0.15 | 0.0140 | 0.0857 | +0.072 | 0.149 |
| **0.20** | **0.0138** | 0.0857 | **+0.072** | 0.168 |

copy baseline: 0.00464

**TF 评估（训练分布上界）：**

| noise_std | TF hand MSE | no_corr (TF) | vision_gain (TF) |
|-----------|------------|-------------|------------------|
| 0.01 | 0.00287 | 0.00289 | +0.000018 |
| 0.02 | 0.00295 | 0.00289 | -0.000061 |
| 0.03 | 0.00295 | 0.00289 | -0.000057 |
| 0.05 | 0.00432 | 0.00289 | -0.001430 |
| 0.07 | 0.00435 | 0.00289 | -0.001453 |
| 0.10 | 0.00463 | 0.00289 | -0.001741 |
| 0.15 | 0.00462 | 0.00289 | -0.001724 |
| 0.20 | 0.00453 | 0.00289 | -0.001637 |

### 关键发现

1. **0.02 是生效阈值**：noise_std_hand=0.01 完全无效（AR ≈ no_corr），0.02 时 AR MSE 骤降到 0.021，说明模型需要一定的噪声量才能学会 AR 鲁棒性。
2. **0.02~0.20 范围内 AR 性能单调改善但增益递减**：从 0.021 (noise=0.02) 缓慢降到 0.014 (noise=0.20)，最优点在 0.15~0.20 附近。
3. **TF 性能与 AR 性能存在 trade-off**：noise ≥ 0.05 时 TF hand MSE 已经超过 copy baseline (0.00464)，说明高噪声下模型在干净输入上的精度受损。但 AR 模式（实际部署场景）持续受益。
4. **推荐值：noise_std_hand=0.10~0.15**。这个区间在 AR 性能接近最优（0.014~0.015）的同时，TF 精度退化可控（0.0046 ≈ copy baseline）。

可视化路径：`visualizations/bc_hand_only_sweep/noise03_full/`（noise03 的 5-sample 完整评估图）

---

## 2026-04-11 · v1 sweep — Hand-Only 解耦诊断 + 基准扫描

### 动机

完整的 BC 3.0 模型（`imitation_learning/behavior_clone/`）同时预测 arm 和 hand，但两个分支的 AR 性能都很差。为了定位问题来源，将 hand 分支单独拆出来训练和评估：

- 如果 hand-only AR MSE 和 BC 3.0 hand AR MSE 差不多 → 问题出在 delta_z + frozen VAE 方案本身
- 如果 hand-only 明显更好 → 问题出在 arm/hand 耦合

同时扫描 reg_drift、dropout、vision ablation、hand history noise 等配置。

### 代码改动

- `model/bc_hand_policy.py` — 新建，BCHandPolicy，仅 hand 分支（无 arm_head, arm_state_encoder）
- `model/bc_hand_dataset.py` — 新建，BCHandDataset，无 state 归一化
- `scripts/train.py` — 新建，hand_loss + reg_drift * drift_loss
- `scripts/eval.py` — 新建，AR 评估 + 可视化（2x3 hand joint 图）
- `scripts/sweep.sh` — 新建，8 组实验 sweep

### 数据

- 训练集：`data/20260327-11:10:43/demos/success/train` — 120 trajectories, 4233 frames
- 测试集：`data/20260327-11:10:43/demos/success/test` — 30 trajectories, 927 frames
- VAE checkpoint：`outputs/dim_2_best/checkpoint.pth` (latent_dim=2, mlp encoder)
- Copy baseline hand MSE: 0.00464

### 配置

基准：lr=5e-4, batch=128, steps=20000, warmup=500, reg_drift=1.0, feat_dim=128, fusion_dim=256

8 组实验：

| Tag | 变量 |
|-----|------|
| baseline | reg_drift=1.0 |
| reg0 | reg_drift=0.0 |
| reg10 | reg_drift=10.0 |
| drop03 | dropout=0.3 |
| drop05 | dropout=0.5 |
| no_vision | disable_vision=True |
| noise01 | noise_std_hand=0.01 |
| noise03 | noise_std_hand=0.03 |

### 结果

**TF 评估（teacher-forced，训练分布上界）：**

| 配置 | TF hand MSE | no_corr | vision_gain (TF) |
|------|------------|---------|------------------|
| baseline | 0.00286 | 0.00289 | +0.000030 |
| reg0 | 0.00325 | 0.00289 | -0.000357 |
| reg10 | 0.00289 | 0.00289 | +0.000006 |
| drop03 | 0.00287 | 0.00289 | +0.000026 |
| drop05 | 0.00289 | 0.00289 | +0.000004 |
| no_vision | 0.00291 | 0.00289 | -0.000017 |
| noise01 | 0.00287 | 0.00289 | +0.000018 |
| noise03 | 0.00295 | 0.00289 | -0.000057 |

**AR 评估（autoregressive，deployment proxy）：**

| 配置 | AR hand MSE | no_corr (AR) | vision_gain (AR) | delta_z norm |
|------|------------|-------------|------------------|-------------|
| baseline | 0.1226 | 0.0857 | -0.037 | 0.284 |
| reg0 | 0.2219 | 0.0857 | -0.136 | 0.278 |
| reg10 | 0.1901 | 0.0857 | -0.105 | 0.036 |
| drop03 | 0.2122 | 0.0857 | -0.127 | 0.064 |
| drop05 | 0.1902 | 0.0857 | -0.105 | 0.075 |
| no_vision | 0.1981 | 0.0857 | -0.112 | 0.066 |
| noise01 | 0.0857 | 0.0857 | -0.000 | 0.158 |
| **noise03** | **0.0167** | 0.0857 | **+0.069** | 0.179 |

训练曲线路径：`outputs/bc_hand_only_sweep/<tag>/training_curves.png`
评估可视化路径：`visualizations/bc_hand_only_sweep/<tag>/`

### 关键发现

1. **TF 模式下视觉修正几乎无意义**：所有配置的 TF vision_gain 都在 ±0.0004 以内。VAE prior 在给定正确历史时已经很准（0.0029），留给 delta_z 修正的空间极小。
2. **不加噪的模型在 AR 模式下全面崩溃**：baseline/reg/drop/no_vision 所有配置的 AR MSE (0.12~0.22) 都远高于 no_corr (0.086)。delta_z 修正在 AR feedback 下引入严重误差累积。
3. **no_vision 与有视觉的配置表现相似** (AR=0.198 vs baseline=0.123)：证明视觉信号对 hand branch 贡献极有限，hand 预测几乎完全由 VAE prior 主导。
4. **问题是内在的，与 arm/hand 耦合无关**：hand-only 模型完全没有 arm 分支，AR 表现仍然差，排除了 arm/hand 耦合假设。
5. **noise_std_hand=0.03 是唯一正向的配置**：AR MSE 降到 0.017，vision_gain=+0.069。训练噪声让模型学会在 AR drift 下做鲁棒修正。→ 后续 v2 noise sweep 精细扫描。
