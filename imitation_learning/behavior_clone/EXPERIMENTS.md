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

## 2026-04-11 · v5 sweep — BC 3.0 弱耦合结构，对比 hand 是否共享 arm latent

### 动机

基于 v4 的 2.0 架构，我们已经确认了两个现象：

1. hand 分支在 2.0 下仍然比较像“prior 主导 + 视觉小修正”
2. arm / hand 虽然分头输出，但仍然通过 shared trunk 和 AR feedback 强耦合

因此这次把结构进一步改成 **BC 3.0**：

- vision 仍然共享
- `arm_state` 单独编码
- `hand_prior = (mu_prior, log_var_prior)` 单独编码
- arm 分支从 `{visual_feat, arm_state_feat}` 直接回归 6 维 arm action
- hand 分支从 `{visual_feat, hand_prior_feat}` 预测 `delta_z`
- 并保留一个显式开关：**hand 是否额外接收 `arm_state_feat`**

要回答的问题是：

1. 去掉 shared state trunk 以后，hand 是否更容易利用视觉？
2. `hand_condition_on_arm=False/True` 哪个更合理？
3. “让 hand 看 arm latent”到底是在帮 timing，还是只是把 arm 误差继续带进 hand？

### 代码改动

- `model/bc_policy.py`
  - shared trunk 改成 **shared vision + split latent encoders**
  - 新增 `visual_fusion`, `arm_state_encoder`, `hand_prior_encoder`
  - `state` 仍保持 12 维接口，但 **只消费前 6 维 arm state**
  - 新增 `hand_condition_on_arm`，控制 hand head 是否拼接 `arm_state_feat`
- `scripts/train.py`
  - 新 CLI `--hand_condition_on_arm`
  - 模型构造接入 `arm_state_dim=6`
  - 日志里打印当前是 BC 3.0 以及 hand-arm conditioning 开关
- `scripts/eval.py`
  - checkpoint loader 接入 `arm_state_dim` / `hand_condition_on_arm`
  - 不兼容报错升级为 BC 3.0 语义
- `scripts/plot_ar_eval.py`
  - checkpoint loader 接入 `hand_condition_on_arm`
- `model/bc_dataset.py`
  - 文档注释补充：BC 3.0 只消费前 6 维 arm state
- `scripts/sweep_v5_arm_share.sh`
  - 新增 2-run 对比脚本（train + eval + comparison json）

### 数据

与 v4 完全相同：120 train / 30 test trajectories，4233 / 927 帧。

### 配置

两组实验都沿用 v4 的最佳训练超参：

| tag | `hand_condition_on_arm` | dropout | reg_drift | feat_dim | fusion_dim | lr | steps |
|---|---|---|---|---|---|---|---|
| `no_arm_latent` | False | 0.3 | 1.0 | 128 | 256 | 5e-4 | 20k |
| `share_arm_latent` | True | 0.3 | 1.0 | 128 | 256 | 5e-4 | 20k |

输出目录：

- `outputs/bc_v5_arm_share/no_arm_latent/`
- `outputs/bc_v5_arm_share/share_arm_latent/`
- `visualizations/bc_v5_arm_share/no_arm_latent_eval/`
- `visualizations/bc_v5_arm_share/share_arm_latent_eval/`
- `visualizations/bc_v5_arm_share/comparison.json`

### 结果

训练结束后的 deterministic 全测试集评估：

| tag | TF arm | TF hand | no_corr hand | vision_gain | AR arm | AR hand | onset TF hand | onset vision_gain |
|---|---|---|---|---|---|---|---|---|
| `no_arm_latent` | 0.02664 | 0.00297 | 0.00295 | -0.000016 | 0.07479 | 0.08570 | 0.00839 | -0.000090 |
| `share_arm_latent` | 0.02829 | **0.00285** | 0.00295 | **+0.000100** | 0.07984 | **0.04803** | **0.00788** | **+0.000421** |

额外聚合统计（来自 `comparison.json`）：

| tag | TF `|delta_z|` | AR `|delta_z|` | TF trajs beating no_corr |
|---|---|---|---|
| `no_arm_latent` | 0.2009 | 0.0824 | 10 / 30 |
| `share_arm_latent` | 0.2053 | 0.0964 | 18 / 30 |

### 关键发现

#### ⭐ 发现 1：共享 arm latent 明显改善了 hand，而不是只改善了平均噪声

和 `no_arm_latent` 相比，`share_arm_latent`：

- `TF hand`: `0.00297 -> 0.00285`
- `AR hand`: `0.08570 -> 0.04803`
- onset 切片 `TF hand`: `0.00839 -> 0.00788`
- onset 切片 `vision_gain_hand`: `-0.000090 -> +0.000421`

这说明 hand 分支额外看到 `arm_state_feat` 之后，提升不只发生在全轨迹平均上，而是**恰好发生在最需要知道“什么时候开始变化”的 onset 段**。

#### ⭐ 发现 2：共享 arm latent 的代价是 arm 自己会更差一点

`share_arm_latent` 相比 `no_arm_latent`：

- `TF arm`: `0.02664 -> 0.02829`
- `AR arm`: `0.07479 -> 0.07984`

也就是说，这个接口不是“免费提升”。它更像是：

**用一点 arm 精度，换 hand 更可靠的视觉 timing control。**

如果任务目标里 hand onset 的重要性更高，这个 tradeoff 是值得的；如果 arm 的几何精度优先级更高，就要继续想办法把这条 arm->hand 条件路径做得更干净。

#### ⭐ 发现 3：`hand_condition_on_arm=False` 的解耦更干净，但 hand 视觉增益又退回去了

`no_arm_latent` 的结果说明，单靠 `{visual_feat, hand_prior_feat}` 这一组输入，3.0 还不足以让 hand 稳定利用视觉：

- 全局 `vision_gain_hand = -0.000016`
- onset `vision_gain_hand = -0.000090`
- `TF hand` 只在 10/30 条轨迹上优于 no-correction baseline

也就是说，**完全切断 hand 对 arm pose 的感知会让结构更干净，但当前数据和任务设定下，它会让 hand 丢掉一部分真正有用的 timing 条件。**

#### 发现 4：共享 arm latent 后，AR 下的 hand 修正幅度更大

- `no_arm_latent`: AR `|delta_z| mean = 0.0824`
- `share_arm_latent`: AR `|delta_z| mean = 0.0964`

这和指标变化是一致的：共享 arm latent 后，hand 分支在 rollout 中更愿意继续做修正，而不是快速塌回 prior。

#### 发现 5：有些“无 onset”轨迹上，共享 arm latent 能显著减少错误闭合

例如 `traj_100`：

- `no_arm_latent`: AR hand = 0.1532
- `share_arm_latent`: AR hand = 0.0035

这类轨迹说明 hand 分支在缺少 arm pose 条件时，很容易在 rollout 里随机闭合；加上 arm latent 后，至少在一部分轨迹上它更能稳定维持“不该动时不动”。

### 当前结论

如果问题表述是：

> 哪种 3.0 架构更适合“视觉决定 hand timing，VAE prior 提供动作趋势”？

当前更合理的答案是：

- **推荐默认**：`hand_condition_on_arm=True`
- 原因：它在 hand 上给出了更清楚的正向视觉增益，尤其在 onset 和 AR hand 上
- 代价：arm 会轻微退化，需要在下一轮里想办法把“提供 hand 条件信息”和“把 arm 误差污染给 hand”区分开

### 下一步

1. **试 `detach(arm_state_feat)` 后再喂 hand**
   看 hand 是否还能保留 timing 增益，同时减少 arm 分支被反向拖拽。
2. **试更弱的 arm-to-hand 接口**
   例如只给 hand 一个更小维度的 arm summary，而不是完整的 `arm_state_feat`。
3. **保留 `share_arm_latent=True` 作为当前默认，再做 scheduled sampling**
   现在 hand 已经比 `no_arm_latent` 更会用视觉，下一步值得先攻 TF-AR gap。

---

## 2026-04-11 · v4 sweep — 2.0 架构验证 + dropout / 容量 / 超参全面搜索

### 动机

v2/v3 使用的是旧版 **delta_mu + delta_log_var → reparameterize → decode** 的随机 hand 路径。代码已重构为 2.0 架构：

- hand 控制接口从 `(delta_mu, delta_log_var)` 改为确定性 `delta_z`：`z_ctrl = mu_prior + delta_z`
- state 输入维度不变（12 维 `actions[t]`），target 不变（`actions[t+1]`）
- past_hand_win 窗口不变（`[a_{t-7}..a_t]` 含当前帧）

需要回答：
1. 2.0 重构是否保持了 v2 的性能水平？
2. 确定性 delta_z 路径下，最优 `reg_drift` 是否改变？
3. **dropout 能否解决 arm 5000× 过拟合**（v1 就发现的核心问题，一直没解决）？
4. 模型容量（feat_dim / fusion_dim）和训练超参（lr / weight_decay）对 AR 稳定性的影响？

### 代码改动

- `model/bc_policy.py` — 删除 `delta_mu_head` / `delta_log_var_head`，新增 `hand_obs_head` + `hand_delta_z_head`；hand 路径改为 `z_ctrl = mu_prior + delta_z`；**新增 `dropout` 参数**，在 `fusion_mlp`（末尾）、`arm_head`（ReLU 后）、`hand_obs_head`（ReLU 后）插入 `nn.Dropout(p)`
- `model/bc_dataset.py` — 修复 docstring（`delta_mu/delta_log_var` → `delta_z`）
- `scripts/train.py` — 新 CLI `--dropout`，传入 BCPolicy 构造
- `scripts/eval.py` — `load_policy_from_checkpoint` 读 `dropout` 参数；旧版 checkpoint 显式报错
- `scripts/sweep_v4_full.sh` — 14 实验串行 sweep + eval + 汇总表
- `scripts/plot_ar_eval.py` — 新增 MC Dropout AR 评估脚本，生成 VAE-eval 风格的逐轨迹可视化

### 数据

与 v1/v2/v3 完全相同：120 train / 30 test trajectories，4233 / 927 帧。

### 配置

14 组实验，分 6 个组：

| 组 | Tag | reg_drift | dropout | feat_dim | fusion_dim | 其它 |
|---|---|---|---|---|---|---|
| G1 基准 | baseline | 1.0 | 0.0 | 128 | 256 | — |
| G2 正则强度 | reg0 | 0.0 | 0.0 | 128 | 256 | — |
| | reg10 | 10.0 | 0.0 | 128 | 256 | — |
| G3 Dropout | drop01 | 1.0 | 0.1 | 128 | 256 | — |
| | **drop03** | **1.0** | **0.3** | **128** | **256** | — |
| | drop05 | 1.0 | 0.5 | 128 | 256 | — |
| G4 容量 | small | 1.0 | 0.0 | 64 | 128 | — |
| | small_drop | 1.0 | 0.3 | 64 | 128 | — |
| | mid_drop | 1.0 | 0.3 | 96 | 192 | — |
| G5 超参 | lr_low | 1.0 | 0.0 | 128 | 256 | lr=2e-4, min_lr=5e-6 |
| | wd_high | 1.0 | 0.0 | 128 | 256 | weight_decay=1e-3 |
| | lr_wd_drop | 1.0 | 0.3 | 128 | 256 | lr=2e-4, wd=1e-3 |
| G6 消融 | no_vision | 1.0 | 0.0 | 128 | 256 | `--disable_vision` |
| | no_state | 1.0 | 0.0 | 128 | 256 | `--state_mask none` |

每组 20k 步 + eval，总用时 12177s ≈ 3.4 小时。

### 结果

`outputs/bc_v4_sweep/V4_SWEEP_SUMMARY.txt` 完整表：

| tag | drop | feat | TF arm | TF hand | no_corr | vis_gain | AR arm | AR hand |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.0 | 128 | 0.02584 | 0.00301 | 0.00295 | −0.000063 | 0.10895 | 0.10282 |
| reg0 | 0.0 | 128 | 0.02628 | 0.00331 | 0.00295 | −0.000356 | 0.10820 | 0.12303 |
| reg10 | 0.0 | 128 | 0.02552 | 0.00295 | 0.00295 | +0.000002 | 0.07537 | 0.03842 |
| drop01 | 0.1 | 128 | 0.02613 | 0.00290 | 0.00295 | +0.000055 | 0.08391 | 0.05217 |
| **drop03** | **0.3** | **128** | **0.02546** | **0.00290** | **0.00295** | **+0.000053** | **0.07286** | **0.01892** ⭐ |
| drop05 | 0.5 | 128 | 0.02590 | 0.00283 | 0.00295 | +0.000123 | 0.09165 | 0.12805 |
| small | 0.0 | 64 | 0.02752 | 0.00293 | 0.00295 | +0.000022 | 0.08100 | 0.03395 |
| small_drop | 0.3 | 64 | 0.02689 | 0.00293 | 0.00295 | +0.000017 | 0.07806 | 0.03895 |
| mid_drop | 0.3 | 96 | 0.02684 | 0.00290 | 0.00295 | +0.000048 | 0.08159 | 0.03739 |
| lr_low | 0.0 | 128 | 0.02888 | 0.00306 | 0.00295 | −0.000105 | 0.08959 | 0.05544 |
| wd_high | 0.0 | 128 | 0.02549 | 0.00310 | 0.00295 | −0.000148 | 0.08753 | 0.07213 |
| lr_wd_drop | 0.3 | 128 | 0.02755 | 0.00287 | 0.00295 | +0.000075 | 0.08136 | 0.04027 |
| no_vision | 0.0 | 128 | 0.03741 | 0.00291 | 0.00295 | +0.000044 | 5.91952 💥 | 0.82954 |
| no_state | 0.0 | 128 | 0.06553 | 0.00300 | 0.00295 | −0.000054 | 0.06553 | 0.14907 |

可视化：
- 训练曲线：`outputs/bc_v4_sweep/<tag>/training_curves.png`
- eval 可视化：`visualizations/bc_v4_sweep/<tag>/`（每个 tag 30 条轨迹 × 3 种图 + summary）
- **MC Dropout AR 逐轨迹图**：`visualizations/bc_ar_eval/drop03/traj_<id>_ar_actions.png`（×30）
- **MC Dropout AR MSE 图**：`visualizations/bc_ar_eval/drop03/traj_<id>_ar_mse.png`（×30）
- latent 诊断图：`visualizations/bc_ar_eval/drop03/latent_diagnostics.png`
- 逐轨迹柱状图：`visualizations/bc_ar_eval/drop03/per_trajectory_hand_mse.png`
- 汇总柱状图：`visualizations/bc_ar_eval/drop03/summary_ar.png`

### MC Dropout AR 诊断（drop03，5 samples × 30 trajectories）

drop03 使用 MC Dropout（推理时保持 dropout 激活）在所有 test 轨迹上跑 5 次全 AR rollout：

| 方法 | Arm MSE | Hand MSE |
|---|---|---|
| AR (MC Dropout, n=5) | 0.0821 | **0.0514** |
| No Correction (delta_z=0) | 0.0810 | 0.0857 |
| Copy baseline (a[t+1]=a[t]) | 0.0297 | 0.0046 |

delta_z 统计：`|delta_z|` 均值=0.122，std=0.113，max=1.033。`|mu_prior|` 均值=0.790。delta_z 约为 prior 的 15%。

30 条轨迹中 **约 20 条 AR hand < no_corr hand**（BC 视觉修正有效）。少数轨迹（100, 68, 105）AR 反而更差，属于 AR 漂移积累的失败案例。

delta_z 随时间衰减：初始步骤修正较大（对齐 onset 前的 prior 偏差），后期趋于稳定。

### 关键发现

#### ⭐ 发现 1：2.0 重构验证通过

baseline 的 TF arm=0.0258、TF hand=0.00301 与 v2 reg=1 的历史值（arm=0.0260、hand=0.00341）基本吻合。确定性 delta_z 没有破坏性能。no_corr hand=0.00295 也与 v2 的 0.00308 一致（微小差异来自 eval 的 deterministic vs stochastic 路径）。

#### ⭐ 发现 2：确定性路径下 reg_drift=10 优于 reg_drift=1

| reg_drift | AR hand (v2.0) | AR hand (v2 旧版) |
|---|---|---|
| 0.0 | 0.12303 | 0.15854 |
| 1.0 | 0.10282 | 0.05933 |
| 10.0 | **0.03842** | 0.08804 |

v2 旧版（随机路径）最优在 reg=1，但 v2.0（确定性路径）最优移到了 reg=10。原因：旧版的 reparameterize 采样噪声起到了天然正则化效果，确定性路径没有这层缓冲，需要更强的显式约束。

但这个结论被 dropout 实验推翻——见发现 3。

#### ⭐⭐ 发现 3：dropout=0.3 是全局最优，AR hand 提升 5.4 倍

**`drop03`（reg=1.0, dropout=0.3）的 AR hand=0.01892，是 14 组实验中的绝对最优**：

| 对比 | AR hand | 相对 drop03 |
|---|---|---|
| drop03 (最优) | **0.01892** | 1.0× |
| small (2nd) | 0.03395 | 1.8× |
| reg10 (3rd) | 0.03842 | 2.0× |
| baseline (无 dropout) | 0.10282 | **5.4×** |
| v2 reg=1 (旧版最优) | 0.05933 | 3.1× |

dropout 的效果：
- **AR hand**：从 0.103 → 0.019（5.4 倍改善）
- **AR arm**：从 0.109 → 0.073（1.5 倍改善）
- **TF 几乎无损**：hand 0.00290 vs 0.00301，arm 0.0255 vs 0.0258
- **vision_gain 转正**：从 −0.000063 → **+0.000053**

v1 就在"v2 计划"中提出了加 dropout（p=0.1~0.3）作为"最便宜的反过拟合手段"，但一直到 v4 才真正实验。事后看来应该在 v2 就做。

dropout=0.5 过度正则化（AR hand 反弹到 0.128）；dropout=0.1 效果不够（AR hand=0.052）。**0.3 是甜区**。

#### ⭐ 发现 4：vision_gain 转正是 dropout 的直接后果

| 有无 dropout | vision_gain |
|---|---|
| 所有 dropout=0 的实验 | −0.000063 ~ −0.000356（负值） |
| 所有 dropout>0 的实验 | **+0.000017 ~ +0.000123**（正值） |

v1-v3 时代反复观察到的"BC delta 对 hand 无贡献甚至有害"（vision_gain 为负），**根因是过拟合**。delta head 在 train 上 memorize 修正方向，到 val 上变成噪声。dropout 阻止了 memorize，使 delta 只在真正有视觉信号时激活。

这回答了 v1 提出的开放问题："BC 只能添加噪声，还是真的能从图像中提取有用信号？"——**真的能，但前提是控制过拟合**。

#### 发现 5：小模型也很有效

`small`（feat=64, fusion=128, 无 dropout）的 AR hand=0.034，比 baseline 好 3 倍。说明 768k 参数对 4233 帧确实太多。但 `drop03`（大模型 + dropout）仍然优于 `small`（0.019 vs 0.034），表明**在 dropout 控制过拟合的前提下，更大的容量能提取更好的视觉特征**。

#### 发现 6：lr 和 weight_decay 不如 dropout 有效

`lr_low`（lr=2e-4）和 `wd_high`（weight_decay=1e-3）单独使用效果一般（AR hand=0.055 和 0.072），远不如 dropout=0.3（0.019）。组合（`lr_wd_drop`）的 AR hand=0.040，反而不如单纯 dropout=0.3，说明降低 lr 会限制模型学习有用视觉特征的能力。

#### 发现 7：消融结果与 v3 一致

- `no_vision`：AR arm 崩溃到 5.9（与 v3 的 24.5 同属发散级别，量级差异来自随机种子）。**视觉仍是 AR arm 的唯一锚**。
- `no_state`：TF arm=0.066（vs 0.026），state 对 arm 精度贡献 ~60%。有趣的是 AR arm=0.066 完全不发散——因为纯视觉模型的 arm 不依赖 state 反馈，不存在误差累积。

### v4 后的整体图景

| 能力 | v3 水平 | v4 水平 (drop03) | 变化 |
|---|---|---|---|
| Arm TF | 0.026 | 0.025 | 基本不变 |
| Hand TF | 0.0034 | 0.0029 | ↓ 15% |
| vision_gain | −0.00033 | **+0.000053** | **转正** ⭐ |
| **Arm AR** | 0.090 | **0.073** | ↓ 19% |
| **Hand AR** | 0.059 | **0.019** | **↓ 68%** ⭐⭐ |

### 选定参数

新推荐配置：

```bash
python imitation_learning/behavior_clone/scripts/train.py \
    --reg_drift 1.0 --dropout 0.3 \
    --feat_dim 128 --fusion_dim 256 \
    --lr 5e-4 --weight_decay 1e-4
```

### 下一步

1. **Scheduled sampling**：dropout 把 AR hand 从 0.103 降到 0.019，但 TF hand=0.003 仍然是 6× 的差距。剩余 gap 的根因是 train/deploy 分布不一致，只有让 BC 在训练时看到自己的预测才能进一步缩小。
2. **Early stopping**：检查 drop03 的训练曲线，看 val 的最优 step 是否远早于 20k。如果是，加入 best-by-val 保存逻辑。
3. **更精细的 onset-slice 分析**：drop03 的 MC Dropout 评估显示 delta_z 在 onset 前修正量较大。可以按 onset 分段报告 MSE，找出模型在 grasp 阶段是否真正利用了视觉信号。

---

## 2026-04-10 · v3 ablations — past_hand_win 加噪声 + state-only 消融

### 动机

v2 sweep 选定 `reg_drift=1.0` 后，剩下两个最重要的开放问题：
1. **past_hand_win 加噪声能否缓解 AR drift？**（对标 VAE 的 `noise_std=0.01` 约定）
2. **视觉输入到底有没有贡献？**（state-only 消融）

两个实验都保持 `--reg_drift 1.0`，只改一个变量。

### 代码改动

- `model/bc_dataset.py` —— 新增 `noise_std_hand` 构造参数，`__getitem__` 里给 past_hand_win 加 `N(0, noise_std_hand)` 噪声（test split 固定 0）
- `model/bc_policy.py` —— 新增 `disable_vision` 构造参数；为 True 时 CNN 输出强制为零、CNN 参数冻结（trainable 从 768k 降到 249k）
- `scripts/train.py` —— 新 CLI `--noise_std_hand` / `--disable_vision`
- `scripts/eval.py` —— 从 ckpt args 读 `disable_vision`
- `scripts/sweep_v3_ablations.sh` —— 两实验串行 + eval + 对比表

### 数据

与 v1/v2 完全相同。

### 配置

| 项 | handnoise | stateonly |
|---|---|---|
| `--reg_drift` | 1.0 | 1.0 |
| `--noise_std_hand` | **0.01** | 0 |
| `--disable_vision` | No | **Yes** |
| trainable params | 768,522 | **249,354** |
| output | `outputs/bc_v3_ablations/handnoise/` | `outputs/bc_v3_ablations/stateonly/` |

### 结果

`outputs/bc_v3_ablations/V3_ABLATION_SUMMARY.txt`，对比 v2 reg=1 baseline：

| run | TF arm | TF hand | no_corr | vis_gain | AR arm | AR hand | copy hand |
|---|---|---|---|---|---|---|---|
| **v2_reg1** (baseline) | 0.02595 | 0.00341 | 0.00308 | −0.00033 | 0.09041 | 0.05933 | 0.00464 |
| **handnoise** (+noise) | 0.02625 | **0.00310** ↓ | 0.00308 | **−0.00003** ↑ | 0.08786 | 0.06614 | 0.00464 |
| **stateonly** (no vision) | **0.03778** ↑↑ | 0.00310 | 0.00308 | −0.00003 | **24.507** 💥 | 0.20423 | 0.00464 |

可视化：
- `outputs/bc_v3_ablations/{handnoise,stateonly}/training_curves.png`
- `visualizations/bc_v3_ablations/{handnoise,stateonly}/traj_<id>_actions.png`（×30）
- `visualizations/bc_v3_ablations/{handnoise,stateonly}/summary.png`

### 关键发现

#### ⭐ 结论 1：视觉对 arm 确有贡献（+45%）

State-only 消融给出了**最清晰的正面结论**：

| | with vision | without vision | 差异 |
|---|---|---|---|
| TF arm MSE | 0.02595 | **0.03778** | **+45%** worse without vision |

12 维 state 自己能做到 0.038，加了视觉能降到 0.026。**视觉贡献了约 30% 的 arm 预测精度**。此前我们一直怀疑视觉是否有用，现在消融明确回答了：**有用**。

#### ⭐ 结论 2：视觉对 hand 几乎无贡献

| | with vision | without vision |
|---|---|---|
| TF hand MSE | 0.00341 | 0.00310 |

两者差距仅 0.0003，且**无视觉反而还好一丁点**——原因是 stateonly 的 delta head 更保守（vision_gain 也接近 0）。hand 预测几乎完全由 VAE prior 决定（no_corr = 0.00308），BC delta head 对 hand 的贡献微乎其微。

#### ⭐ 结论 3：在 AR 反馈下视觉是唯一的"锚点"

State-only 的 AR arm 完全爆炸到 **24.5**（正常值 ~0.09）。轨迹可视化里 arm 预测发散到 −100 量级。

因果链：没有视觉 → arm TF 精度下降 → AR 模式把稍差的 arm 预测灌回 state → 下一步预测更差 → 正反馈死亡螺旋。**视觉在 AR 场景下是防止 arm 发散的唯一锚**。

#### 结论 4：past_hand_win 加噪声是有效的 TF 正则化但不治 AR

| | v2_reg1 | handnoise | 变化 |
|---|---|---|---|
| TF hand | 0.00341 | **0.00310** | **−9%** ⭐ |
| vision_gain | −0.00033 | **−0.00003** | 几乎中性了 |
| AR hand | 0.05933 | 0.06614 | +11%（略差） |

noise 的好处：训练时给 past_hand_win 加 N(0, 0.01) 噪声让 BC 更保守——delta head 不再过拟合 train 上的精确历史，TF hand 降了 9%，vision_gain 从显著负值恢复到接近中性。

noise 的局限：AR drift 反而稍微变差。原因是 AR 模式的分布偏移远大于 N(0, 0.01) 的扰动量级，加噪声只覆盖了"小扰动"但对"大偏移"无效。要真正解决 AR drift，必须让 BC 在训练时看到自己的预测（scheduled sampling / DAgger）。

### v3 后的整体图景

| 能力 | 当前水平 | 瓶颈在哪 |
|---|---|---|
| Arm TF 预测 | 0.026（比 copy baseline 0.030 低 13%） | 过拟合（train arm ~2e-6 vs val 0.026） |
| Hand TF 预测 | 0.0031（接近 VAE prior 0.00308 的极限） | BC delta 无法超越 VAE prior |
| Arm AR robustness | 0.090（with vision 时不会崩） | train/deploy 分布不一致 |
| Hand AR robustness | 0.059（reg=1 最佳） | 同上 |

**最值得投入的方向**：scheduled sampling（训练时概率灌入 BC 自己的预测到 state / past_hand_win）。这是唯一能从根本上缩小 TF-AR gap 的方法。

---

## 2026-04-10 · v2 sweep — state 改用 12 维 actions[t]，引入 hand drift 正则

### 动机

v1 暴露了两个问题：
1. **state 输入用错**：v1 用了 `curr_obs.states`（24 维，包含速度等乱七八糟），语义模糊。改用数据集里 `actions[t]`（机器人当前 12 维绝对位姿）作为 state，预测目标同步改为 `actions[t+1]`。这跟 VAE 训练约定完全一致（VAE 也是给 8 帧 `[a_{t-7}..a_t]` 预测 `a_{t+1}`），过去窗口的 offset 也回归 v1 的"含当前帧"。
2. **BC delta head 在 train 上 memorize**：v1 vision_gain 在 step ~2000 见顶后掉负，最终 −0.000506。需要正则项把 BC 修正后的 hand 拉回 VAE prior 输出附近。

新增 **drift 正则**：在 `BCPolicy.forward` 里 inline reparameterize 用**同一份 ε** 同时算 corrected 路径和 no-correction 路径，loss 加上 `λ_drift · MSE(hand_action, hand_no_corr)`。共享 ε 让 drift 完全只反映 (δμ, δlv) 的修正幅度，采样噪声两边互相抵消。

### 代码改动

- `model/bc_dataset.py` —— state 从 24 维 `curr_obs.states` 改为 12 维 `actions[t]`，标准化用 `compute_action_stats`；past_hand_win 改回 `hand[a_{t-7}..a_t]`（v1 是截止到 t-1）；target = `actions[t+1]`。
- `model/bc_policy.py` —— `state_dim` 默认 24→12；forward 里 inline reparameterize，多返回 `hand_no_corr`。
- `scripts/train.py` —— 新 CLI `--reg_drift`，loss = `arm + hand + λ·drift`，新 history 字段 `train_drift`，training_curves 加 drift 子图。
- `scripts/eval.py` —— rollout state 输入改用 `actions[t]`，AR mode 现在把整个 12 维 prediction（含 arm）灌回下一步 state（v1 只灌 hand 因为 state 是另一个东西）。新增 copy baseline。
- `scripts/sweep_reg_drift.sh` —— 5 值串行 sweep + eval + summary 表。
- README.md —— 重写架构图（BC 可训练框 vs 冻结 VAE 框，past_hand_win 明确画在 BC 之外）；新增训练 loss 章节；写入 sweep findings。

### 数据

跟 v1 完全相同：`/home/yxy/VQ-VLA/data/20260327-11:10:43/demos/success/{train,test}`，120 / 30 trajectories，4233 / 927 帧。

### 配置

| 项 | 值 |
|---|---|
| state 输入 | 12 维 `actions[t]`，z-score 标准化 |
| past_hand_win | `hand[a_{t-7}..a_t]`（含当前帧，匹配 VAE 训练） |
| target | `actions[t+1]`（最后一帧用 `actions[T-1]` 占位） |
| BC 参数总量 | 915,988（trainable 768,522 + frozen 147,466） |
| optimizer | AdamW lr=5e-4 wd=1e-4 |
| LR schedule | cosine 5e-4 → 1e-5, warmup 500 |
| total_steps | 20000 |
| batch_size | 128 |
| sweep 维度 | `--reg_drift ∈ {0.0, 1.0, 10.0, 100.0, 1000.0}` |

每个 reg 训练 ~11 分钟，整 sweep（5 训练 + 5 eval）总共 56 分钟。

### 结果

`outputs/bc_sweep_v2/SWEEP_SUMMARY.txt` 完整表：

| `reg_drift` | TF arm | TF hand | no_corr | vision gain | **AR hand** | AR arm | copy hand |
|---|---|---|---|---|---|---|---|
| 0.0    | 0.02592 | 0.00358 | 0.00308 | −0.00051 | **0.15854** | 0.10574 | 0.00464 |
| **1.0**| 0.02595 | 0.00341 | 0.00308 | −0.00033 | **0.05933** ⭐ | 0.06445 | 0.00464 |
| 10.0   | 0.02626 | **0.00306** ⭐ | 0.00308 | **+0.00001** | 0.08804 | 0.07203 | 0.00464 |
| 100.0  | 0.02571 | 0.00308 | 0.00308 | +0.00000 | 0.09204 | 0.07026 | 0.00464 |
| 1000.0 | 0.02579 | 0.00308 | 0.00308 | −0.00000 | 0.09354 | — | 0.00464 |

可视化：
- 训练曲线：`outputs/bc_sweep_v2/reg_<X>/training_curves.png`
- 12 关节对比图：`visualizations/bc_sweep_v2/reg_<X>/traj_<id>_actions.png`（×30）
- per-step MSE 图：`visualizations/bc_sweep_v2/reg_<X>/traj_<id>_mse.png`（×30）
- 总览柱状图：`visualizations/bc_sweep_v2/reg_<X>/summary.png`

### 关键发现

1. **drift 正则确实减轻 AR drift**：reg=1 把 hand AR drift 从 0.159 降到 0.059，降低 2.7×。这是部署场景下最实用的改进。
2. **reg 与 TF/AR 之间是 U 型**：TF hand 在 reg=10 处取最低（0.00306），AR hand 在 reg=1 处取最低（0.059）。两个最优不在同一个 reg 上。
3. **`reg=10` 是首个 vision_gain 转正的值**（+0.00001），但增益微小到可视为噪声。**实质含义：BC delta 在 reg=10 时已经被基本压平**，TF hand 等于 no_corr hand，BC 退化成"几乎不修正"。
4. **`reg=1.0` 是最佳实用平衡点**（已设为新 default）：
   - AR robustness 取最优
   - TF hand 比 reg=0 降 5%
   - BC delta 仍有学习容量（没被完全压平）
5. **TF arm 在所有 reg 都几乎不变**（0.0257-0.0263）。drift 正则只通过共享 backbone 间接影响 arm，影响极小。
6. **arm 仍然过拟合**：所有 reg 下 train arm MSE 都掉到 ~2e-6，但 val arm 卡在 ~0.026。说明 arm 路径需要独立的正则化（dropout / 数据增广 / 缩小 arm head）。
7. **AR drift 不能靠 drift 正则解决根本**：reg=1 已经把 hand AR drift 降到 0.059，但仍然比 TF（0.0034）差 17×。这是因为 BC 训练时只看过 GT 的 past_hand_win，**根因是训练分布与 AR 部署分布不一致**。下一步要解决必须改训练策略：
   - **scheduled sampling** / DAgger：训练时按概率灌入 BC 自己的预测
   - **训练时给 past_hand_win 加噪声**（类似 VAE 的 `noise_std=0.01`）
   - **closed-loop fine-tuning**：在已收敛的 BC 上做一轮 AR 模式微调

### 选定参数

新默认 `--reg_drift 1.0`。后续实验都基于这个值再叠加其它改进。

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
