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

## 当前开发目标（给下一次对话的启动上下文）

这个子项目当前的核心目标，不是把“抓玩具时张开手再合拢”这个单任务拟合到最好看，而是做一个**通用的手部状态先验模型**：

- 输入：过去一段时间的 `state chunk`
- 输出：下一时刻或下一段 future 的**状态分布**
- 这里的 `state` / `action` 在当前阶段都指**手各关节的 joint angle**

更具体地说：

1. **先验模型只负责手的内在状态演化分布**
   - 希望它学的是“给定当前手部历史，未来手部状态大概会怎么分布”。
   - 它应该刻画多峰、不确定、阶段切换这类时序规律。

2. **视觉 policy 只负责引导这个先验，而不是直接替代它**
   - 目标是让视觉 policy 输出一个对 latent 分布的 guidance / bias / sampling signal。
   - 不希望视觉 policy 直接输出完整手部动作，否则维度高、训练不稳，也会让 TCP pose 的权重被稀释。

3. **TCP pose 和手部先验要明确区分**
   - TCP pose、视觉、上下文都应该看作“指导先验怎么选”的外部因素。
   - **不要把 TCP pose 显式塞进手部先验模型本体里**，否则先验会和任务/场景强耦合，泛化变差。

4. **低维 latent 是必须的，不只是为了好看**
   - 当前重点是二维 latent，因为后续视觉 policy 可以在这个二维分布上施加 bias。
   - 理想状态下，视觉 policy 预测的是“如何在低维手部先验上引导采样”，而不是再额外回归高维手关节角。

一句话：**要做的是“手部状态分布先验 + 视觉引导”，不是“视觉直接回归手动作”。**

## 通用性约束（非常重要）

当前数据里“张开手 -> 某个随机时刻开始合拢 -> 保持合拢”只是一个 demo，用来观察先验是否学到了未来分布。后面任务可能会出现：

- 多次开合手
- 合到特定程度而不是完全闭合
- 不同节奏、不同子阶段

因此主线方法必须满足：

- **不要依赖 task-specific 的“精准动刀” loss**
  - 例如显式 anti-drop、显式“平台掉落惩罚”、专门为了当前 demo 修锯齿/去尖刺的损失。
- **不要为了当前 demo 在架构里硬编码动作规律**
  - 例如默认“后几维必须单调上升”“一定收敛到 plateau”“只允许闭合不允许再张开”这类强偏置。
- **如果出现尖刺/掉落，优先怀疑 chunk handling / autoregressive 接法 / 状态更新 bug**
  - 不要直接把“尖刺不允许”写成 loss。

换句话说：**当前例子可以用来暴露问题，但不能反过来定义方法本身。**

## 当前任务下的评估原则

### 为什么旧式指标不够好

如果只看“从 reset 开始的一条 rollout 像不像一条好看的闭手轨迹”，会错误地偏爱那些带强时序模板的模型。  
但这不等于它真的学会了：

- 给定当前手部历史，未来有哪些可能
- 每种可能出现的概率是多少
- 多峰未来分布是否和数据一致

尤其在当前数据里，很多“看起来还是全 0 / 近 0”的窗口，未来既可能继续 open，也可能马上开始 close。  
所以**只看 reset 首次合拢时间**是不够的。

### 应该重点看的指标

1. **open-like 条件未来分布**
   - 在当前观测窗口仍然是近 0 / open-like 的情况下：
     - 未来一段 horizon 内继续保持 open 的概率是多少？
     - 开始 close 的概率是多少？
     - 第一次离开 open basin 的延迟分布 / PMF / hazard 是什么？
   - 这是当前最重要的指标，因为它直接测“当前 state chunk -> 未来分布”。

2. **延迟分布是否接近近似泊松 / 几何式触发**
   - 对当前玩具抓取数据，GT 中 open-like 窗口的未来 close 触发应呈随机触发特性。
   - 因此模型也应该在 open-like 条件下给出接近的数据分布，而不是“必定马上合”或“永远不合”。

3. **数据支持集约束（dataset support constraint）**
   - 在当前数据中，GT 不存在明显的 `close -> open` 转变。
   - 所以在当前数据集上，模型 rollout 的 `reopen rate` 应接近 0。
   - 注意：这只是**当前数据集的评估约束**，不是未来所有任务都应写进架构的硬规则。

4. **闭合尾段幅度校准**
   - 一旦模型确实选择 close，最终的闭合均值不应系统性偏大或偏小。
   - 需要比较闭合尾段均值/关节均值与 GT closed distribution 的差距。

### 明确不推荐的指标/做法

- 不要把“尖刺式掉落”“平台回落”直接定义成主指标
- 不要用专门惩罚当前 demo 症状的 loss 当主线方法
- 不要只看 deterministic mean rollout
- 不要只看从 reset 开始的 first-close time
- 不要把“生成一条很好看的闭手曲线”误认为“学到了通用先验”

## 当前主线保留的模型家族

出于上面的通用性要求，当前主线保留的是这些相对中性的家族：

- `vae_step`
- `chunk_h12_b1e3`
- `flow_post_best`
- `state_space_rw_best`
- `traj_hold`

而下面这些方向目前**只当作特定 demo 的参考现象，不当作通用主线**：

- `plateau_*`
- `*_m1_*`
- `*_anti-drop*`

## 推荐优先查看的脚本

如果下一次对话要快速恢复上下文，优先看这些脚本：

- `vae/scripts/eval_distributional_metrics.py`
  - 按 open-like 条件分布、hazard、reopen rate、closed-tail calibration 评估模型
- `vae/scripts/visualize_competitive_models.py`
  - 生成“5 条 GT + 50 次 AR”以及 latent closed map
- `vae/scripts/generate_general_report.py`
  - 生成去掉 task-specific 精准修补后的通用版报告

建议默认把这三者当作当前主线评估入口，而不是旧的 anti-drop / spike 风格脚本。

## 新对话启动包（详细版）

这一节是专门写给“下一次新对话”用的。  
如果后续有新的 assistant / 新的协作者接手，请优先读完这一节，再决定训练、改模型或重做评估。

### 1. 用户真正想要解决的问题

用户当前不是在做一个“单任务动作回归器”，而是在做一个**手部状态先验（hand state prior）**：

- 输入：过去一段时间的手部状态历史，也就是 `state chunk`
- 输出：未来下一时刻或下一段时间的**状态分布**
- 当前实验里，`state` 和 `action` 都可以理解为手各关节的 joint angle

这里“先验”的含义很明确：

- 它应该描述“在没有视觉引导时，手在当前内部状态下，未来可能如何演化”
- 它应该支持**多峰未来**
- 它应该能表达“此时仍然 open / 也可能马上触发 close”的不确定性
- 它最好在一个**低维 latent** 空间里表达这些可能性，便于后续视觉 policy 去引导采样

### 2. 为什么必须把 hand prior 和视觉 / TCP pose 分开

用户多次强调：**TCP pose、视觉、context 都不应该直接写进 hand prior 本体**。

理由不是“现在的数据里用不上”，而是更底层的建模假设：

- 手部先验是更“先天”的内部运动规律
- 视觉 / context / TCP pose 是“如何选择或偏置这个先验”的外部条件
- 如果把它们混进 hand prior 本体，模型会很快学成“任务绑定的条件动作器”，而不是“可被引导的通用先验”

可以把职责划分理解为：

- **hand prior**：给出当前手部历史下，未来手状态的合理支持集和概率分布
- **visual policy**：在这个分布上施加偏置，决定这次具体走哪条分支

用户给过一个非常关键的直觉类比：  
盲人和刚出生的婴儿也会使用手，所以“手部先验”不应等同于“看到了什么物体之后的手动作回归”。

### 3. 为什么低维 latent 非常重要

低维 latent 不是为了可视化好看，而是这个方案成立的关键：

1. **降低视觉 policy 的输出维度**
   - 如果手有 16 自由度，而视觉 policy 还要同时输出 TCP pose，那么直接回归 `6 + 16` 维动作很容易让 TCP pose 被淹没
   - 用户明确担心这一点：TCP 估不准会直接导致任务失败

2. **让视觉 policy 做“分布引导”而不是“动作硬回归”**
   - 更理想的形式是：视觉 policy 输出对 latent 分布的 bias / guidance / sampling signal
   - 而不是直接把手动作从图像里硬回归出来

3. **便于理解和调试**
   - 当前二维 latent 正好可以：
     - 画出哪些区域是 open-like
     - 哪些区域是 closed-like
     - AR rollout 在 latent 空间中如何移动

### 4. 当前 demo 数据到底在验证什么

当前选用的 demo 任务是“抓起一个玩具”，因此手的典型行为是：

- 一开始张开手
- 过一段时间之后开始合拢
- 一旦开始合拢，在当前数据里通常不会重新张开
- 最后在一个闭合程度附近保持

这个 demo 的价值，不是为了把“单次张开→合拢”做到完美，而是为了验证以下更一般的问题：

- 模型能否从 open-like 状态学出**多峰未来分布**
- 模型能否学出“什么时候触发阶段切换”的分布，而不是只输出均值
- 模型能否在 AR rollout 中维持合理的未来状态演化

### 5. 当前数据集的关键统计特性

用户明确指出了一个核心现象：

- 在很多时间步上，当前观测窗口看起来几乎都是全 0 / 近 0
- 但它们的下一步未来并不唯一：
  - 有些会继续保持 open
  - 有些会开始 close

因此对这类 open-like 条件，未来实际上是一个**双峰 / 多分支分布**。

更进一步，在当前玩具抓取数据里，用户期望观察到：

- 不同 sample 会在不同时间步开始合手
- 这个“开始合手的时间”应该呈现接近**泊松 / 几何式随机触发**的分布
- 一旦进入 close，相比 open 概率会持续朝闭合方向演化
- 最终会在一个与 GT 近似的闭合均值附近停住

### 6. 用户明确反对的方向

这一部分非常重要，后续任何新对话都不应再反复走错。

#### 6.1 不要做“精准动刀”的 task-specific 修补

用户已经明确否定过这类做法：

- anti-drop loss
- spike-specific penalty
- 平台掉落惩罚
- 只为当前 demo 现象设计的局部补丁

原因是：

- 这些做法是对“当前症状”动刀，不是对“分布建模原理”动刀
- 它们会导致方法对当前单个示例过拟合
- 后续如果任务出现多次开合手、部分闭合、复杂节奏，这些规则会立刻变成错误先验

#### 6.2 不要把当前 demo 里的单向闭合写成硬架构偏置

用户也不希望主线架构默认带这样的限制：

- 后几维单调上升
- 一定收敛到 plateau
- 一旦 close 就结构上不允许 reopen

注意：  
对于当前数据集，`reopen` 作为**评估约束**是合理的；  
但把“不允许 reopen”直接硬编码进通用模型结构，则不合理。

#### 6.3 尖刺 / 掉落应优先被看作 bug 症状

用户明确说过：

- 某些关节出现尖刺式掉落，是不允许的
- 但这不应被直接变成一个专门的 loss
- 更合理的理解是：这是 autoregressive chunk handling、状态更新方式、边界拼接或模型表达上的 bug / 症状

所以遇到这类现象的默认处理顺序应该是：

1. 先检查 rollout 接法
2. 再检查 chunk 边界和 hidden state 更新
3. 再检查 latent / decoder 是否存在结构性平均化
4. 最后才考虑是否需要更一般的建模改动

而不是直接：

- “我们加个 anti-drop loss 吧”

### 7. 评估时最容易犯的错误

用户已经明确指出过几类错误评估思路：

#### 错误一：只看 reset 开始的一条均值轨迹漂不漂亮

这会偏爱那些学了“时间模板”的模型，比如：

- 从 episode 开头先固定 hold
- 然后在某个固定时机关闭

这种模型在当前 demo 上也许“看起来像”，但它不一定真的学会了：

- 当前 state chunk 对应哪些未来分支
- 每个分支概率是多少

#### 错误二：把“首次合拢时间”当成唯一核心指标

如果只从 reset 看 first-close time，会混淆两个问题：

- 当前观测是否具有歧义
- 未来触发的条件概率是否学对

用户已经指出：  
很多“马上要合拢”的窗口，当前看起来仍然和初始 open 几乎一样。  
所以仅看“从 reset 到首次合拢的时间”不够。

#### 错误三：把当前症状当成 loss 目标

例如：

- 看到尖刺，就优化“尖刺分”
- 看到平台掉落，就优化“平台保持率”

这在当前例子上也许能改图，但会把主线方法带歪。

### 8. 推荐的评估框架

后续主线评估，应优先采用下面这个框架。

#### 8.1 open-like 条件未来分布

对所有满足以下条件的窗口进行统计：

- 当前 history window 仍是 open-like / near-zero

然后比较模型与 GT 的：

- `未来 H 步内继续保持 open 的概率`
- `未来 H 步内开始 close 的概率`
- `首次离开 open basin 的延迟分布`
- `每一步触发 close 的离散 hazard`

这是当前最重要的评估，因为它最贴近用户真正要的东西：

> 根据当前 state chunk，预测未来合理分布

#### 8.2 长滚动 support consistency

从统一的 open seed 做长 AR rollout，重点观察：

- 是否出现 GT 不支持的 `close -> open`
- 是否系统性过早 / 过晚关闭
- survival curve 是否像 GT

#### 8.3 闭合尾段均值校准

对确实进入 close 的 rollout，检查：

- 最后的闭合程度是否和 GT 相近
- 不应系统性偏大
- 不应系统性偏小

#### 8.4 可视化仍然很重要，但只能做辅助

用户仍然非常重视肉眼可见的轨迹形态，因此可视化必须保留：

- 5 条 GT 轨迹的 6 个关节变化曲线
- 从初始 chunk 出发的 50 次 AR rollout 曲线
- latent closed/open 区域图
- AR 完整轨迹在 latent 空间里的路径

但这些图只能作为**辅助理解**，不能替代分布型指标。

### 9. 当前应优先使用的指标，而不是旧指标

#### 应优先用

- open-like conditional close probability
- open-like delay PMF
- open-like hazard
- reset survival curve
- reopen rate
- closed-tail calibration

#### 不应作为主导

- spike score
- anti-drop score
- 仅从 reset 看的 first-close median
- 单条 deterministic mean rollout 是否好看

### 10. 当前主线模型家族的定位

当前主线保留：

- `vae_step`
  - 最基础的一步预测基线
  - 常见问题是明显均值化

- `chunk_h12_b1e3`
  - 改善长滚动稳定性
  - 但可能会把 open-like 多峰未来压成“几乎必关”

- `flow_post_best`
  - 在通用家族里更有“连续分布建模”的味道
  - 但当前实现中要重点盯 `reopen`

- `state_space_rw_best`
  - 在 open-like 条件分布上比较有竞争力
  - 但要警惕闭合幅度过大、以及结构里残留的偏置

- `traj_hold`
  - 更像“保留 episode 起点语义”的版本
  - reset 轨迹可能更像数据
  - 但容易对任意 open-like chunk 给不出合理触发分布

#### 不再作为通用主线的方向

- `plateau_*`
- `*_m1_*`
- `*_anti-drop*`

这些方向可以保留作历史参考，但不要再作为“当前最佳通用方案”推荐。

### 11. 当前项目里最重要的脚本

如果下一次新对话要快速进入状态，优先看以下文件：

- `vae/scripts/eval_distributional_metrics.py`
  - 当前主线评估入口
  - 负责 open-like conditional distribution、hazard、reopen、closed-tail calibration

- `vae/scripts/visualize_competitive_models.py`
  - 当前主线可视化入口
  - 负责 5 条 GT + 50 次 AR、以及 latent closed map

- `vae/scripts/generate_general_report.py`
  - 生成去掉 task-specific 精准修补后的通用版报告

### 12. 新对话中 assistant 的默认行为建议

如果后续由新的 assistant 接手，建议默认遵守：

1. 默认使用中文回答
2. 优先读 `vae/README.md`
3. 默认把“通用性”放在“当前 demo 图好不好看”之前
4. 遇到尖刺/掉落，先排查建模或 rollout bug，不要直接上精准 loss
5. 比较模型时，先给出**分布型指标**，再给可视化结论
6. 不要把旧的 `plateau / anti-drop` 版本重新包装成“当前最优”

### 13. 一句总结（给下一次对话的开场白）

如果要用一句话概括当前项目状态，可以直接这样理解：

> 这是一个“二维低维手部状态先验 + 未来视觉引导”的项目；当前最重要的是学对 open-like 条件下的未来分布，而不是继续用 task-specific 的 anti-drop / plateau 技巧把单个 demo 修得更像。

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
