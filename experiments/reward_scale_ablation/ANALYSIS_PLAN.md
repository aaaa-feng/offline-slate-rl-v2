# reward_scale ablation 分析计划

> 实验: 5 runs = reward_scale (1, 5, 10, 20, 50) × beta=8 × mix_divpen
> 数据来源: SwanLab project `Offline_Slate_RL_202606`, tag `reward_scale_ablation/rs{N}`
> 关键步: 500, 5000, 11000, 15000, 20000, 30000, 50000, 100000

---

## 核心问题

**不改 GeMS 动作空间，能否通过调节 Q/V 学习速度（reward_scale）来延长 Advantage 信号的生命周期？**

---

## §1 reward_scale 对 Q/V 量级的因果链

**假说**: reward_scale 越大（reward 越小），Q/V 学得越慢，Advantage 存活越久。

因果链:
```
reward_scale↑ → target_q↓ → TD error↓ → Q/V 收敛变慢 → Advantage 存活更久
reward_scale↓ → target_q↑ → TD error↑ → Q/V 收敛变快 → Advantage 更快消失
```

### 表 1-1: Q/V 收敛速度 vs reward_scale

| 指标 | rs1 | rs5 | rs10 | rs20 | rs50 |
|------|-----|-----|------|------|------|
| v_value_mean @5000 | | | | | |
| v_value_mean @100k | | | | | |
| q_value_mean @5000 | | | | | |
| q_value_mean @100k | | | | | |
| V-Q gap @5000 | | | | | |
| V-Q gap @100k | | | | | |

预期: rs1 的 Q/V 值最大、学得最快；rs50 的 Q/V 值最小、学得最慢。

---

## §2 Advantage 生命周期 vs reward_scale

**核心观测**: adv_q90 死亡步数是否随 reward_scale 单调变化。

### 表 2-1: adv_q90 关键步追踪

| Step | rs1 | rs5 | rs10 | rs20 | rs50 |
|------|-----|-----|------|------|------|
| 500 | | | | | |
| 5000 | | | | | |
| 11000 | | | | | |
| 15000 | | | | | |
| 20000 | | | | | |
| 30000 | | | | | |
| 50000 | | | | | |
| 100000 | | | | | |

### 表 2-2: adv 死亡时间

| 指标 | rs1 | rs5 | rs10 | rs20 | rs50 |
|------|-----|-----|------|------|------|
| adv_q90 < 0.1 的 step | | | | | |
| near_zero_rate > 0.75 的 step | | | | | |
| awr_entropy_norm > 0.95 的 step | | | | | |

预期: 如果假说成立，rs50 的 adv 死亡步数应该最大（最晚死），rs1 的死亡步数最小（最早死）。

---

## §3 Reward 轨迹 vs reward_scale

**核心观测**: reward_scale 变大后，reward 值域缩小，但峰值/终值的**相对关系**是否改变。

### 表 3-1: eval reward mean 关键步

| Step | rs1 | rs5 | rs10 | rs20 | rs50 |
|------|-----|-----|------|------|------|
| 500 | | | | | |
| 5000 | | | | | |
| 11000 | | | | | |
| 15000 | | | | | |
| 20000 | | | | | |
| 30000 | | | | | |
| 50000 | | | | | |
| 100000 | | | | | |

### 表 3-2: 峰值与终值

| 指标 | rs1 | rs5 | rs10 | rs20 | rs50 |
|------|-----|-----|------|------|------|
| peak reward | | | | | |
| peak step | | | | | |
| 终值 reward @100k | | | | | |
| 终值/峰值比 | | | | | |
| adv 死亡时 reward | | | | | |

预期: rs50 可能 peak 更低（学习慢，冲不高），但终值/峰值比更高（Adv 活得久，崩得少）。rs1 可能 peak 最高（冲得猛），但崩得更彻底。

---

## §4 策略锁死 vs reward_scale

**核心观测**: reward_scale 变大后，策略是否锁死得更慢。

### 表 4-1: floor_hit_rate

| Step | rs1 | rs5 | rs10 | rs20 | rs50 |
|------|-----|-----|------|------|------|
| 500 | | | | | |
| 15000 | | | | | |
| 30000 | | | | | |
| 100000 | | | | | |

预期: rs50 的 floor_hit 增长最慢（Adv 活得久，策略有更多探索机会）。rs1 的 floor_hit 增长最快。

---

## §5 判定矩阵

填完表 2-2 和 3-2 后:

| 结论 | 判定条件 | 成立？ |
|------|---------|--------|
| reward_scale 越大，adv 死得越晚 | 表 2-2 死亡步数单调递增 | |
| reward_scale 越大，终值/峰值比越高 | 表 3-2 比值单调递增 | |
| reward_scale 越大，策略锁死越慢 | 表 4-1 rs50 floor_hit_100k < rs1 floor_hit_100k | |
| 存在最优 reward_scale | 终值 reward 对 reward_scale 是倒 U 形 | |
| Adv 消失是 IQL 必然结果，与量级无关 | 所有 rs 的 adv 死亡步数差异 < 20% | |

### 决策分支

- 如果 5 个 rs 的 adv 死亡步数差异 > 30% → reward_scale 对 Adv 生命周期有显著因果影响 → 进入 §6
- 如果 5 个 rs 的 adv 死亡步数差异 < 10% → reward_scale 几乎不影响 Adv 生命周期 → Q/V 量级不是 Adv 消失的主因 → 验证了"Adv 消失是 IQL expectile 机制的必然结果"
- 如果存在最优 rs → 下一步做 rs × λ_bc 的 2D 消融

---

## §6（如果 §5 成立）最优 reward_scale 的下一步

1. 最优 rs 在 topdown_divpen 上验证（跨环境确认）
2. 最优 rs + λ_bc=0.3 组合实验（Adv 延迟 + BC 兜底），预期终值超过 rs10 + λ_bc=0
3. 最优 rs + λ_bc=0.3 在 200k 步长跑，看长期是否不再崩塌

---

## 附录: 指标 → SwanLab 字段

| 分析用名 | SwanLab 路径 |
|---------|-------------|
| eval reward mean | `00_Eval/Reward/mean` |
| adv_q90 | `11_Train-Value/Adv-Quantile/q90` |
| adv_near_zero_rate | `11_Train-Value/Adv-Shape/near_zero_rate` |
| awr_entropy_norm | `10_Train-Opt/AWR-Entropy/normalized` |
| v_value_mean | `11_Train-Value/V-Summary/mean` |
| q_value_mean | `11_Train-Value/Q-Summary/mean` |
| vq_gap_q50 | `11_Train-Value/VQ-Gap-Quantile/q50` |
| floor_hit_rate | `12_Train-Policy/LogStd/floor_hit_rate` |
| z_center | `12_Train-Policy/Geo/z_to_dataset_center_mean` |
