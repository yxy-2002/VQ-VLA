# Behavior Clone — Experiment Log

持续维护的实验日志，按时间倒序排列（最新在最上面）。每次跑完一组实验都更新这里。

格式约定（参考 `vae/EXPERIMENTS.md`）：

```
## YYYY-MM-DD · <短标题>

### 动机
为什么做这次实验，要回答什么问题。

### 代码改动
碰过哪些文件，每个一行说明。

### 数据
数据路径、训练/测试规模、copy baseline。

### 配置
关键超参（lr / batch / steps / 模型宽度）。

### 结果
arm_mse / hand_mse_full / hand_mse_no_correction 在关键 step 的表格。
training_curves.png 路径。

### 关键发现
1-3 条要点：什么有效、什么没效、下一步试什么。
```

---

## 2026-04-09 · v1 baseline — 简单 CNN+MLP，BC over frozen dim_2_best VAE

### 动机

第一次跑通 BC over 冻结 VAE 的 pipeline。目标只是看损失能不能收敛、视觉信号能不能让 hand_mse 跌破 no-correction 基线，而不是冲性能。一次最小可信验证。

### 代码改动

新增模块（之前是空目录）：

- `model/bc_dataset.py` — `BCDataset` + `compute_state_stats`。预载入图像为 uint8、`__getitem__` 内现转 float32，避免 4 GB RAM 爆掉。**关键偏移**：`past_hand_win = hand[max(0,t-8):t]`（终止于 `t-1`），不能复用 `vae/model/hand_dataset.py` 的 `HandActionWindowDataset`，否则目标 `a_t` 会泄漏到 prior。
- `model/bc_policy.py` — `BCPolicy` + `SimpleCNN` + `build_and_freeze_vae` + `strip_vae_state_dict`。两路独立 CNN，state MLP，融合 MLP，arm head + 两个 zero-init 的 delta head。覆盖了 `train(mode)` 让 VAE 永远在 eval。
- `scripts/train.py` — 镜像 `vae/scripts/train.py` 的结构，删去 β 退火，新增 step-0 zero-init assert 和 `hand_mse_no_correction` 基线日志。

### 数据

- 数据集：`/home/yxy/VQ-VLA/data/20260327-11:10:43/demos/success/{train,test}` —— 与 VAE 训练完全相同
- train: 120 trajectories / **4233 帧**
- test:  30 trajectories / **927 帧**
- 图像：每帧 main + extra 共 2 路 128×128×3 uint8，preload 后 RAM 占用 ~510 MB
- state: 24 维，per-dim z-score（mean/std 从 train 计算一次，存进 ckpt）

无图像增广，无 action 归一化。

### 配置

| 项 | 值 |
|---|---|
| VAE ckpt | `outputs/dim_2_best/checkpoint.pth` (latent_dim=2, hidden=256, mlp, window=8) |
| BC 参数总量 | 917,524 (trainable 770,058 + frozen VAE 147,466) |
| feat_dim / fusion_dim | 128 / 256 |
| optimizer | AdamW, lr=5e-4, wd=1e-4 |
| LR schedule | cosine 5e-4 → 1e-5, warmup 500 |
| total_steps | 20000 |
| batch_size | 128 |
| clip_grad | 1.0 |
| eval_samples | 3 (验证时每个 batch 重复采样 3 次取均值) |
| seed | 42 |
| device | cuda |
| num_workers | 0 (preload 数据 + DataLoader 多进程会撑爆 /dev/shm) |

### 结果

输出目录：`outputs/bc_simple_v1/`
- `checkpoint.pth` — 最终权重
- `checkpoint-{5000,10000,15000,20000}.pth` — 中间快照
- `training_curves.png` — 6 子图：train total / train arm vs hand / val arm+hand+no_corr / val total / lr / vision improvement
- 训练用时：约 11 分钟 / 20k 步（单机 GPU + 7s/200 step）
- step-0 zero-init assert：`|delta_mu|_max = 0.00e+00`，`|delta_log_var|_max = 0.00e+00` ✅

| Step | train arm | train hand | val arm | val hand_full | val hand_no_corr | vision_gain |
|---|---|---|---|---|---|---|
| 0 (sanity) | 0.157 | 0.001313 | 0.138572 | 0.002820 | 0.002860 | +0.000040 (≈ 噪声) |
| 1000 | 0.014 | 0.001469 | 0.033066 | 0.002695 | 0.002860 | **+0.000165** |
| 2000 | 0.003 | 0.001443 | 0.037949 | 0.002692 | 0.002860 | **+0.000168** |
| 3000 | 0.001 | 0.000712 | 0.038421 | 0.002807 | 0.002860 | +0.000053 |
| 4000 | 0.001 | 0.001076 | 0.038440 | 0.002867 | 0.002860 | **−0.000007** |
| 5000 | 0.000 | 0.000922 | 0.037574 | 0.002914 | 0.002860 | −0.000054 |
| 10000 | 0.000 | 0.000987 | 0.036563 | 0.003069 | 0.002860 | −0.000209 |
| 15000 | 0.000 | 0.001257 | 0.036302 | 0.003079 | 0.002860 | −0.000219 |
| 20000 (final) | 0.000 | 0.000967 | **0.036414** | **0.003091** | 0.002860 | **−0.000231** |

### 关键发现

1. **训练侧完全收敛**：train arm MSE 从 0.157 → ~2e-6（5 个数量级的下降），train hand MSE 从 1.3e-3 → ~1e-3。Pipeline 跑通，无 NaN，无梯度爆炸，step-0 zero-init assert 通过。
2. **严重过拟合**：val arm MSE 从 step 1000 (0.033) 起就基本不动，最终 0.036；而 train arm MSE 已经掉到 ~2e-6——**train/val 差距 >18000×**。CNN+MLP + 4233 帧 + 无任何正则化 = 教科书级 memorize-train。
3. **🚨 视觉修正反向有害**：`vision_gain` 在 step ~2000 达到峰值 +0.000168，之后单调下降，**step ~4000 起转负**，最终稳定在 −0.000231。也就是说 BC delta head 学到的修正反而让 hand 比 VAE 自己的 prior 差。这有两个可能解释：
   - delta heads 在 train 上对每帧 memorize 出一个修正方向，到 val 上完全没有泛化；
   - VAE prior（不依赖图像）已经是 hand_mse ≈ 0.00286 这个水位的最优解，BC 只能添加噪声。
   下一版**必须**加正则化或显式约束 delta 不能太离谱。
4. **arm 和 hand 的过拟合幅度差异巨大**：arm 把 train 完全压掉了（5 个数量级），但 train hand 只从 1.3e-3 → 1e-3——猜测原因是 hand 路径有 stochastic reparameterize 注入的噪声，相当于天然的训练时正则化，限制了过拟合速度；而 arm 是确定性 MLP head，没有任何噪声，立即 memorize。

### v2 计划（已知调整方向）

按收益预期排序：

1. **加 dropout** 在 fusion_mlp 和 head 前（p=0.1~0.3）。最便宜的反过拟合手段。
2. **加 delta L2 正则化**：`λ * (||delta_mu||² + ||delta_log_var||²)`，λ ≈ 1e-3 起跑。直接限制 BC 不能把 latent 推太远。
3. **早停**：基于 val 的最佳 checkpoint 在 step ~2000 附近就出现了 (vision_gain=+0.000168)，应该把"best by val total"逻辑加进训练脚本。
4. **数据增广**：图像 random crop / color jitter；state Gaussian noise (类似 VAE 的 `noise_std=0.01`)。
5. **缩小模型**：770k trainable 对 4233 帧太多了；feat_dim=64 + fusion_dim=128 应该更合适，把可训参数压到 ~200k。
6. （可选）**分别 lr**：CNN 用 5e-4，head 用 1e-4——head 学得太快是过拟合的一大来源。

### 已观察到的细节

- **训练时长**：20k 步约 670s ≈ 11 分钟。eval 占用很少。
- **VAE prior 的 hand MSE 基线 0.00286** 比想象中低很多——意味着 VAE 自己（即使没看过当前帧的图像）已经能给出相当合理的 hand 预测。BC 想要超过这个值，必须从图像中提取真正有用的信号，而不是 memorize。
- **stochastic reparam 的方差影响**：看 train hand MSE 在不同 step 在 5e-4 和 2e-3 之间晃，说明每个 step 的 hand_loss 受 reparam noise 影响明显。eval 用 3-sample 均值已经能稳定到 ±2e-5 的方差水平。
