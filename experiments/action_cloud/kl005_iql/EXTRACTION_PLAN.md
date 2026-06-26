# KL=0.05 IQL 实验: 指标提取与分析计划

> **参考文档**:
> - `experiments/action_cloud/total_summary/README.md` — KL=1.0 baseline 完整分析 + KL 消融静态点云结构
> - `logs/agents/beta_ablation_repreduce/hypothesis_chain_detailed.md` — 10 个 run 的逐步指标表
> - `logs/agents/beta_ablation_repreduce/ANALYSIS_PLAN.md` — 分析框架

---

## 一、我在关注什么

### KL=1.0 baseline（beta_ablation_repreduce）已经确定的事

10/10 个 run 全部 reward 坍塌。关键特征：

- **所有 run 都有上升期**：peak 在 500-3500 步，此时 adv_q90 > 0.2
- **所有 run 都有下降期**：peak 之后 reward 跌 39%-85%，adv_q90 在 5k-11k 步之间死亡
- **ood_det 从头降到尾**：哪怕 reward 已经崩到谷底，ood_det 仍然在降（Actor 一直在"靠近 batch 数据"，但学的是噪声）
- **z_center 始终稳定在 2.65-3.41**：Actor 离云心的距离从开始到结束几乎不变——它在壳面上打转，从未进云
- **Topdown 比 Mix 坍塌更严重**：终值最低至 24.5，几乎是随机策略水平

### KL=1.0 的点云结构（total_summary §4 已分析）

- PC80=5.5-7（32 维只有五分之一有信号）
- p10≈p50（空心壳：最近 25% 的点全部集中在同一个距离上）
- 条件数千万级（某些方向方差趋近于零，梯度在这些方向上爆炸）
- E_cos=0.34-0.40（item 表示丧失区分度）

### KL=0.05 的点云结构（实验二已分析）

- PC80=15-16（是 KL=1.0 的 3 倍）
- p10≠p50（偏实心：有真正的近心点存在）
- E_cos=0.18-0.26（item 表示更多样）
- 唯一瑕疵：cond 偶尔飙高需要额外保护

### 本次实验要回答的核心问题

**如果把 GeMS 从 KL=1.0 换成 KL=0.05（云宽 3 倍），IQL 训练是否不再坍塌？**

更具体地说：

| 假设 | 如果 KL=0.05 有效 | 如果 KL=0.05 无效 |
|------|------------------|-----------------|
| **z_center 会下降吗** | 从 2.65-3.41 降到 1.5-2.0（Actor 进云了） | 还是 2.65-3.41（Actor 还是壳面打转） |
| **adv 活得久吗** | 死亡步数从 5k-11k 推迟到 20k+ | 跟 KL=1.0 一样 5k-11k 死亡 |
| **reward 还崩吗** | 终值明显高于 KL=1.0 的终值 | 终值和 KL=1.0 差不多（跌幅 39%-85%） |

---

## 二、要提取哪些指标

参照 beta_ablation_repreduce 分析框架，提取以下 6 个核心指标：

### 1. eval_reward — 最终裁判

SwanLab: `00_Eval/Reward/mean`

Actor 在 RecSim 模拟器上跑 50 个 episode 的平均 reward。这是所有分析的起点——先看 reward 曲线长什么样，有没有峰值、有没有坍塌。

### 2. adv_q90 — Advantage 信号的强度

SwanLab: `11_Train-Value/Adv-Quantile/q90`

当前 batch 里 Advantage = Q(s,a) - V(s) 的 90 分位。看的是"最好的那些动作比平均好多少"。

- > 0.2：信号强，AWR 能区分好坏动作
- < 0.1：信号死亡，AWR 权重退化为均匀分布
- 死亡步数：adv_q90 首次跌到 < 0.1 且后续不再回升的步数

### 3. near_zero (near_zero_rate) — Advantage 死亡的程度

SwanLab: `11_Train-Value/Adv-Shape/near_zero_rate`

|Advantage| < 0.1 的样本占比。> 0.75 表示四分之三以上的动作已经无区分度，AWR 变成均匀 BC。

### 4. ood_det — Actor 到当前 batch 标签的局部距离

SwanLab: `12_Train-Policy/Geo/ood_distance_mean_det`

Actor 确定性输出与同一条样本的标签 x 的 L2 距离。只看**当前 batch 的 256 条**，不看 100 万全局。持续下降表示 Actor 在局部靠近数据。

⚠️ 注意：KL=1.0 中 ood_det 下降 ≠ 学好了。这个指标在 KL=0.05 中也可能继续下降——关键看它和 z_center 的关系是否改变。

### 5. z_center — Actor 到全局定标中心的距离

SwanLab: `12_Train-Policy/Geo/z_to_dataset_center_mean`

Actor 确定性输出到 10k 样本定标中点的 L2 距离。看全局位置——Actor 在整张图里走进云了没有。

- KL=1.0: 始终 2.65-3.41，100k 步几乎不变
- KL=0.05 预测：会明显下降（云变宽了，Actor 能找到路进云）

### 6. floor_hit — 策略锁死的程度

SwanLab: `12_Train-Policy/LogStd/floor_hit_rate`

Actor 的 32 维 log_std 中有多少维被压在下限 LOG_STD_MIN=-5。从 ~0% 涨到 60%+ 表示探索能力持续丧失。

- KL=1.0: 从 1% 涨到 60-75%
- KL=0.05 预测：如果 adv 活得久，Actor 不用锁死来应对 adv 退化 → floor_hit 可能更低

---

## 三、数据提取方案

### 两组对比

| 组 | GeMS | 对比 |
|----|------|------|
| 旧 baseline | ideal_init KL=1.0 | `beta_ablation_repreduce` 已有（10 个 run 的数据已在 total_summary §2 和 hypothesis_chain_detailed.md） |
| **A 组** | ideal_init KL=0.05 | 本次实验（10 个 run） |
| **B 组** | mf_init KL=0.05 | 本次实验（10 个 run） |

### 提取方法

从本地 log 文件逐 run 提取 6 个指标在关键步（500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 7000, 11000, 15000, 20000, 30000, 50000, 100000）的数值。

参照 hypothesis_chain_detailed.md 的格式：每张 run 子表包含 6 列 `| step | eval_reward | adv_q90 | near_zero | ood_det | z_center | floor_hit |`。

### 输出物

1. **10 张逐 run 子表**（A 组 5 个 mix + 5 个 topdown，B 组同理）— 格式与 hypothesis_chain_detailed.md 一致
2. **汇总对比表** — 类似 total_summary §2.5 的格式：peak reward, peak step, final reward, drop%, 以及 adv 死亡步数、z_center @100k、floor_hit @100k
3. **核心验证表** — KL=0.05 vs KL=1.0 的逐指标对比：z_center 是否下降、adv 是否活得更久、reward 是否不崩

---

## 四、预期分析路径

1. **先看 A 组 vs baseline**（ideal_init KL=0.05 vs ideal_init KL=1.0）：这是最干净的对比，只改变 GeMS KL，其他完全一致。

2. **再看 B 组 vs A 组**（mf_init KL=0.05 vs ideal_init KL=0.05）：看 embedding 初始化是否有额外影响。如果 A 组已经明显优于 baseline、B 组跟 A 组差不多，则说明 KL 是主导变量。

3. **最后交叉看 mix vs topdown**：两个环境中 KL=0.05 的效果是否一致。如果 topdown 从 KL=0.05 获益比 mix 少，说明 topdown 的坍塌可能有 KL 之外的额外原因。
