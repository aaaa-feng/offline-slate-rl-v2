# reward_scale ablation: Q/V 尺度 × Advantage 生命周期

## 动机

### 背景：Advantage 在 ~15k 步消失

在 beta_ablation_repreduce 实验中，所有 β 在 mix_divpen + ideal_init GeMS + λ_bc=0 设定下，Advantage 信号在 ~15k-20k 步衰减到接近零（adv_q90 < 0.1，near_zero_rate > 0.75）。AWR 权重退化为均匀分布，策略失去区分好坏动作的能力，reward 从峰值崩回 BC 水平。

旧假设链（瓶颈分析报告 §3）：
```
Q/V 在 10k-15k 步学稳
  → Q(s,a) ≈ V(s) （IQL expectile 机制的自然结果）
  → Advantage ≈ 0
  → AWR exp(βA) ≈ 1（均匀权重）
  → 策略锁死在数据云外
  → combo_hit 崩 → reward 崩
```

### 核心问题（回答导师 Q3）

**不改 GeMS 动作空间，能否通过调节 Q/V 学习速度来延长 Advantage 信号的生命周期？**

### 机制假说

在 IQL 中，reward 经过 `reward_scale` 缩放后进入 Bellman 备份：

```
target_q = reward / reward_scale + γ · next_v
```

`reward_scale` 同时缩放 Q 和 V 的绝对量级，但不直接改变两者的**相对收敛速度**。然而：

- **较小的 reward_scale（大 reward）**：TD error 更大 → Q/V 学得更快 → 两者更快收敛到一致 → Advantage 可能更早消失
- **较大的 reward_scale（小 reward）**：TD error 更小 → Q/V 学得更慢 → 两者收敛更慢 → Advantage 可能存活更久

这不是"改 reward_scale 让 Advantage 不消失"，而是**调 Q/V 学习速度看 Advantage 消失时机是否偏移**。如果 reward_scale=50 时 adv 在 30k 步才消失（而非 15k），则证实"Adv 消失是 Q/V 学习速度的产物"；如果所有 reward_scale 下 adv 都在 ~15k 步消失，则证实"Adv 消失是 IQL expectile 机制的必然结果，与 Q/V 量级无关"。

### 实验设计

| 参数 | 值 | 说明 |
|------|-----|------|
| 环境 | mix_divpen, b5 | 主环境 |
| GeMS | ideal_init | 与旧实验一致 |
| beta | 8 | 旧实验中峰值最明显的 beta |
| λ_bc | 0.0 | 关掉 BC 安全网，隔离 reward_scale 效应 |
| reward_scale | 1, 5, 10, 20, 50 | 5 个值，10 是当前 baseline |
| max_timesteps | 100k | 足够观察完整生命周期 |

### 预期观测

| reward_scale | reward 量级 | 预期 adv 消失时机 | 预期 peak reward |
|-------------|-----------|-----------------|-----------------|
| rs1 | ~0-10 | 最早（~10k?） | 可能最高 |
| rs5 | ~0-2 | 较早 | |
| rs10 | ~0-1 | baseline（~15k） | baseline |
| rs20 | ~0-0.5 | 较晚 | |
| rs50 | ~0-0.2 | 最晚（~30k?） | 可能最低但终值更高 |

### 回答 Q3 的具体方式

如果 reward_scale=50 让 adv 在 30k 步才消失，且 100k 终值高于 rs10，则证明：
- **不改 GeMS 空间，只调 Q/V 学习速度，可以延长 Advantage 信号窗口**
- 但这仍然不是永久解决——Adv 最终还是会消失

下一步可以结合 λ_bc > 0 + reward_scale 找到最优组合。

### 文件结构

```
experiments/reward_scale_ablation/
├── config.yaml
├── run.sh
└── README.md (本文件)

checkpoints/agents/reward_scale_ablation/
├── rs1/
├── rs5/
├── rs10/
├── rs20/
└── rs50/

logs/agents/reward_scale_ablation/
├── rs1.log
├── rs5.log
├── rs10.log
├── rs20.log
└── rs50.log
```

### 分析方法

1. 5 条 eval reward 曲线叠加在同一张图
2. adv_q90 死亡步数 vs reward_scale 散点图
3. 终值 reward vs reward_scale 散点图
4. 如果有最优 reward_scale → 在 geom_probe_v1 中做 2 env × 2 beta × 最优 scale 的确认实验
