# 实验 5: Actor 输出温度缩放

> 不改 GeMS，不改 Actor 训练过程，只在推理（act）时乘一个温度系数

---

## 1. 动机

### 1.1 问题

从 `beta_ablation_repreduce` 的实验数据和静态云分析可知：

| 量 | mix | topdown |
|----|-----|---------|
| 数据云中心 `z_cloud`（标签到 dataset_center 的 L2） | ~1.80 | ~2.51 |
| Actor 输出位置 `z_actor`（Step 500 → 100k 几乎不变） | ~2.68 | ~3.39 |
| Actor 到云中心的距离 | ~0.88 | ~0.88 |
| 云内部宽度（标签间典型 L2） | ~1.03 | ~0.96 |

Actor 从一开始就站在云外约 0.88 的 L2 距离处，训练 10 万步几乎没动。Advantage 信号试图把 Actor 往云的方向拉，但力不够——Adv 在 5k-11k 步内消失，之后 Actor 锁死在原位。

### 1.2 为什么温度缩放可能有效

Actor 的输出链路：

```
obs → GRU → s_actor (20维)
  → TanhGaussianActor.forward()
    → mu (32维, 无约束)
    → raw_action = tanh(mu) ∈ [-1, +1]^32
    → latent = raw_action * action_scale + action_center  ← 反归一化到 GeMS 空间
    → GeMS.rank(latent) → slate
```

`raw_action` 的每个分量在 [-1, +1] 之间。`action_scale` 约 0.34（mix）或 0.56（topdown）。`action_center` 约 -0.008（mix）或 -0.005（topdown）。

Actor 的 `z_center` 很大不是因为 `raw_action` 在某个方向偏得离谱一而是因为 32 维里很多维同时取接近 ±1 的值，L2 范数自然就大。随机初始化的 tanh 网络确实会产生这种效果。

**温度缩放的思路**：在推理时把 `raw_action` 乘一个 < 1 的系数：

```python
# 原来
latent_action = raw_action * self.action_scale + self.action_center

# 改为
latent_action = raw_action * temperature * self.action_scale + self.action_center
```

这样 Actor 输出的方向不变（方向由 tanh(mu) 各维的相对大小决定，是训练学到的），但幅度减小——离 `action_center` 更近，也就离数据云更近。

### 1.3 为什么不改训练、只改推理

训练时 Actor 学的是 `true_actions`（数据云里的点）。`true_actions` 本身离 `dataset_center` 平均 ~2.76（mix）。Actor 学会的是"输出离中心 ~2.76 的向量"。在这个距离上，Actor 无法精确命中云内部（云内部宽度只有 ~1.03）。

温度缩放在推理时把输出"拉近"，是一种后处理：训练不变、GeMS 不变、只改 decode 前的最后一步。如果有效，说明问题的确就是"Actor 站得太远"——是几何问题而非学习问题。

### 1.4 已知证据

在 `beta_ablation_repreduce` 的所有 10 个 run 中：

z_center 从 Step 500 到 Step 100k 变化 < 5%。Actor 的全局位置几乎不随训练改变。

Advantage 信号在 5k-11k 步后枯竭。AWR 权重退化为均匀。此后训练退化为"在云外的确定性模仿"。

Slate 多样性持续崩塌（global_unique 从 300-500 降到 13-34）。

温度缩放开的是"把站在云外的 Actor 拉到云附近"这条路径。如果拉近后 decode 的 slate 能恢复多样性，就证明"云外→锁死→崩塌"的因果链成立。

---

## 2. 实验设计

### 2.1 变量

温度系数 T ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 1.0}

T=1.0 等于不改动（baseline 对照）。T=0.5 表示把 raw_action 的幅度减半，相应地把 latent_action 到 center 的距离也大致减半。

### 2.2 基准实验

使用 `beta_ablation_repreduce` 中已完成的两个代表性 run 的 checkpoint：

- mix_b8（终值 72.9，unique=34。典型的"没有明显峰但多样性崩塌"）
- td_b8（终值 87.0，unique=464。唯一保持多样性的 run，U 型恢复）

**为什么选这两个**：
- mix_b8 代表了 mix 环境里大多数 run 的模式——没有剧烈冲高崩塌，但多样性最终崩了
- td_b8 是唯一对"崩塌"有抵抗力的 run——看看温度缩放能否进一步改善它
- β=8 是"中间偏强"的 AWR 权重，在旧实验中展示了最明显的"冲高→崩塌"动态

### 2.3 评估方式

不重新训练。直接用训练好的 checkpoint，在 `scripts/eval.py` 里加温度参数，跑 100 个 episode 评估。每个温度值跑一次。

**需要的改动**：在 `IQLAgent.act()` 方法的反归一化处加温度参数。

```python
# src/agents/iql/agent.py, act() 方法
# 当前 (L1514):
latent_action = raw_action * self.action_scale + self.action_center

# 改为:
temperature = getattr(self, '_eval_temperature', 1.0)
latent_action = raw_action * self.action_scale * temperature + self.action_center
```

在 `scripts/eval.py` 加 `--temperature` 参数。

### 2.4 评估指标

每个 (run, T) 组合记录：

| 指标 | 含义 |
|------|------|
| mean_reward | 平均 episode reward |
| median_reward / iqm_reward | 稳健 reward |
| global_unique_items | slate 多样性恢复程度 |
| combo_soft_hit_rate | decode 结果是否落在训练集分布内 |
| z_center（从 SwanLab 训练指标推算） | Actor 被拉到多近 |

### 2.5 命令模板

```bash
# 对 mix_b8 的 final checkpoint, T=0.5
python scripts/eval.py \
    --algo iql \
    --env_name mix_divpen \
    --dataset_quality b5 \
    --gems_embedding_mode ideal_init \
    --checkpoint checkpoints/agents/beta_ablation_repreduce/mix_b8/iql_final.pt \
    --temperature 0.5 \
    --episodes 100
```

---

## 3. 预期与解读

### 3.1 如果温度缩放有效

随着 T 从 1.0 降到 0.3，应该能看到：

reward 先升后降——存在最优 T。T 太小会丢失方向信息（所有维都缩到接近 0），T 太大等于没改。

global_unique_items 在中低 T 时显著恢复。从 34 恢复到 > 100 就是强信号。

combo_hit_rate 在中低 T 时上升。说明"离云更近"果真让 decode 结果更像训练数据。

### 3.2 如果温度缩放无效

所有 T 下 reward 和 unique 都没变化（或只有随机波动）。说明问题不在"站得远"——Actor 虽然位置在云外，但方向已经不指向高 reward 的 slate。单纯拉近没用，需要改训练过程（BC 兜底、改 GeMS 等）。

所有 T 下 reward 都明显下降。说明 Actor 学到的方向本身是有意义的——偏离才能命中高 reward slate。这反而说明"云外"可能是最优策略有意选择的——但这个结论和当前数据（cloud_center 1.80, Actor 2.68）不太吻合，可能性较低。

### 3.3 如果只对 topdown 有效、对 mix 无效

mix 的云有效维只有 2，topdown 有 4。当 Actor 被拉近后：

- mix：云太扁，即使拉近也难命中——可能需要先拓宽云
- topdown：云有 4 维可以"容错"，拉近后更容易碰到好 slate

如果出现这种模式，说明温度缩放和 GeMS 结构（云宽度）需要联动——单独的几何修正不够。

---

## 4. 实施步骤

1. 在 `IQLAgent.act()` 中加 `_eval_temperature` 属性
2. 在 `scripts/eval.py` 中加 `--temperature` CLI 参数
3. 对 mix_b8 和 td_b8 的 final checkpoint，扫 T ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 1.0}
4. 对 best checkpoint（IQM 最优）重复步骤 3
5. 汇总对比表，判断温度缩放是否有效

---

## 5. 如果要扩展到训练阶段

如果推理阶段的温度缩放有效，下一步是**在训练阶段也加温度**，让 Actor 从一开始就被拉近云：

在 `IQLAgent.train()` 的 Actor loss 计算前，对 actor_mu 也乘 temperature。或者在 `act()` 中同时影响训练和推理路径。

但这是 Phase 2——先看推理侧有效再说。
