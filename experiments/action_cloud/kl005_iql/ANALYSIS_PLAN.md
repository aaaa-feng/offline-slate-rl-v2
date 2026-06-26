# KL=0.05 IQL 实验分析计划

> 验证：拓宽 GeMS 云（KL=0.05, PC80=14-18）后，IQL 训练是否不再崩塌？

---

## 1. 对比逻辑

| 组 | GeMS | 云宽度 | 对比对象 |
|----|------|--------|---------|
| 旧 baseline | ideal_init KL=1.0 | PC80=2-5 | `beta_ablation_repreduce` 已有 |
| **A: ideal_init KL=0.05** | 新训 | PC80=14-18 | vs 旧 baseline |
| **B: mf_init KL=0.05** | 新训 | PC80=14-18 | vs A（embedding 是否有额外影响） |

每组: 2 env (mix, topdown) × 5 beta (0,2,5,8,10) = 10 runs
共 20 runs

---

## 2. 核心问题

**如果云变宽了，Actor 能进靶子吗？**

- 旧结论：Actor z_center 从 2.68（mix）/ 3.39（topdown）全程不动，Adv 消失后 reward 崩塌
- 预期：云变宽后，Actor 更容易进云 → z_center 下降 → reward 不崩塌（或崩塌更慢）

---

## 3. 关键指标提取

### 3.1 Reward 轨迹（回答"是否还崩塌"）

从 SwanLab 提取每个 run 的 eval reward 曲线，对比：

| 指标 | 旧 baseline (KL=1.0) | KL=0.05 预期 |
|------|---------------------|-------------|
| peak reward | mix: ~110, td: ~140 | 更高 or 持平？ |
| peak step | mix: ~11k, td: ~500 | 更晚？ |
| 终值 reward @100k | mix: ~70, td: ~40 | **明显更高？** |
| 终值/峰值比 | mix: ~0.64, td: ~0.29 | **更高（不崩塌）？** |

**输出表**：

| run | peak reward | peak step | 终值 @100k | 终值/峰值 | 是否崩塌 |
|-----|------------|----------|-----------|---------|---------|
| mix_b0 (β=0, BC) | | | | — | — |
| mix_b2 | | | | | |
| mix_b5 | | | | | |
| mix_b8 | | | | | |
| mix_b10 | | | | | |
| td_b0 (β=0, BC) | | | | — | — |
| td_b2 | | | | | |
| td_b5 | | | | | |
| td_b8 | | | | | |
| td_b10 | | | | | |

### 3.2 Actor 位置（回答"Actor 进云了吗"）

从 SwanLab 提取 z_center 曲线：

| 指标 | 旧 baseline | KL=0.05 预期 |
|------|-----------|-------------|
| z_center @500 | mix: ~2.68, td: ~3.39 | 更低？ |
| z_center @100k | mix: ~2.75, td: ~3.42 | **明显更低？** |
| z_center 变化 | mix: +0.07, td: +0.05 | **下降（进云）？** |

**输出表**：

| run | z_center @500 | z_center @10k | z_center @50k | z_center @100k | 进云？ |
|-----|--------------|--------------|--------------|---------------|-------|
| mix_b0 | | | | | |
| mix_b2 | | | | | |
| ... | | | | | |

### 3.3 Advantage 生命周期（回答"Adv 活得久吗"）

从 SwanLab 提取 adv_q90 和 near_zero_rate：

| 指标 | 旧 baseline | KL=0.05 预期 |
|------|-----------|-------------|
| adv_q90 死亡步数 | ~15k | **更晚？** |
| near_zero_rate >0.75 步数 | ~15k | **更晚？** |

**输出表**：

| run | adv_q90 < 0.1 步数 | near_zero >0.75 步数 | Adv 活得久？ |
|-----|------------------|-------------------|------------|
| mix_b0 | — | — | — |
| mix_b2 | | | |
| ... | | | |

### 3.4 Slate 多样性（回答"多样性维持更久吗"）

从 SwanLab 提取 global_unique_items 曲线：

| 指标 | 旧 baseline | KL=0.05 预期 |
|------|-----------|-------------|
| global_unique @11k | mix: ~40, td: ~400 | 更高？ |
| global_unique @50k | mix: ~30, td: ~300 | **更高？** |
| global_unique @100k | mix: ~25, td: ~250 | **更高？** |
| 崩塌步数（unique 跌破 50） | mix: ~20k, td: ~40k | **更晚？** |

**输出表**：

| run | unique @11k | unique @50k | unique @100k | 崩塌步数 |
|-----|-----------|-----------|------------|---------|
| mix_b0 | | | | |
| mix_b2 | | | | |
| ... | | | | |

---

## 4. 对比分析框架

### 4.1 主对比：KL=0.05 vs KL=1.0

对每个 env × beta，对比：

| 维度 | 旧 (KL=1.0) | 新 (KL=0.05) | 改善？ |
|------|-----------|------------|-------|
| 终值 reward | | | |
| z_center @100k | | | |
| adv 死亡步数 | | | |
| unique @100k | | | |

### 4.2 子对比：ideal_init vs mf_init (都 KL=0.05)

看 embedding 初始化是否有额外影响：

| 维度 | ideal_init KL=0.05 | mf_init KL=0.05 | 差异？ |
|------|------------------|---------------|-------|
| 终值 reward | | | |
| z_center @100k | | | |
| adv 死亡步数 | | | |
| unique @100k | | | |

---

## 5. 预期结论

**如果 KL=0.05 有效**：
- 终值 reward 明显高于 KL=1.0（比如 mix_b8 终值从 70 → 90+）
- z_center 下降（Actor 进云）
- adv 死亡步数更晚（比如从 15k → 30k+）
- unique 维持更久（崩塌步数从 20k → 40k+）

**如果 KL=0.05 无效**：
- 终值 reward 跟 KL=1.0 差不多
- z_center 还是不动
- adv 还是 15k 死亡
- unique 还是 20k 崩塌

→ 说明问题不在云宽度，而在别的地方（比如 Actor 输出归一化、BC 兜底）

---

## 6. 下一步

根据结果决定：
- **有效** → 继续优化 KL=0.05 的超参（beta, lambda_bc 等）
- **无效** → 转向实验 3（embedding 正交化）或实验 4（真实 click）
