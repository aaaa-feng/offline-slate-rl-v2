# GRU Belief Geometry 诊断结果

> 数据: 50 个 `*_belief.npz`（kl005 / kl1 / kl005_mf，mix + topdown 等 run）
> 每 run 对比: `iql_best.pt` vs `iql_final.pt`，**actor / critic_v 双流**
> 定量指标: `outputs/belief_metrics_summary.json`（100 行 = 50 npz × 2 stream）
> 图: `outputs/figures/belief/{kl}/{run}/`（约 209 张）

---

## 1. 指标定义（与 action cloud 对齐）

| 指标 | 含义 |
|------|------|
| `to_centroid` | policy belief 到 dataset belief 质心的平均 L2 距离 |
| `nn_dist` | policy 点到最近 dataset 邻居的平均距离 |
| `std_ratio` | policy per-dim std / dataset per-dim std |
| `in_cloud_p90` | policy 点落在 dataset p90 半径内的比例 |
| `reward` / `item_freq_pct` | eval 时同步记录的 step reward 与 item 频率百分位 |

**灰云**: `dataset_belief_{actor,critic_v}` — 离线数据按 episode 顺序过 GRU 得到的状态轨迹  
**彩点**: `policy_belief_{actor,critic_v}` — deterministic eval 时同步记录的双流 belief

---

## 2. 核心 run 定量表（actor stream）

| KL | Run | Tag | to_centroid | nn_dist | std_ratio | in_p90% | reward | item_freq |
|----|-----|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.05** | mix_b0 | best | 1.55 | 0.92 | 0.79 | 81% | 2.22 | 5.30 |
| | | final | 2.15 | 1.20 | **0.25** | **0%** | 0.62 | 0.06 |
| **0.05** | mix_b8 | best | 1.51 | 0.75 | 0.74 | 95% | 2.06 | 6.32 |
| | | final | 2.67 | 1.15 | 0.61 | **2%** | 0.72 | 0.08 |
| **0.05** | mix_b10 | best | 1.47 | 0.57 | 0.88 | 89% | 2.05 | 6.22 |
| | | final | 2.26 | 1.29 | 1.05 | 69% | 0.88 | 1.91 |
| **1.0** | mix_b8 | best | 1.63 | 0.23 | 0.78 | 96% | 2.01 | 6.13 |
| | | final | 3.31 | 0.32 | **0.13** | **1%** | 0.69 | 0.03 |
| **1.0** | mix_b10 | best | 2.14 | 0.48 | 0.93 | 94% | 1.49 | 4.37 |
| | | final | 2.68 | 0.15 | **0.19** | **100%** | 0.95 | 1.34 |

critic_v stream 在 best 阶段与 actor 接近；final 阶段出现显著**双流分歧**（见第 4 节）。

---

## 3. 聚合：best → final 变化（n=10 run/组）

| KL | Stream | Δto_centroid | Δin_p90% | Δstd_ratio |
|----|--------|:---:|:---:|:---:|
| **0.05** | actor | **+0.54** | **-70** | -0.09 |
| **0.05** | critic_v | +0.76 | -15 | -0.19 |
| **1.0** | actor | +0.71 | -47 | **-0.65** |
| **1.0** | critic_v | +0.93 | -26 | **-0.66** |

**解读**:
- **actor belief 的 in_cloud 跌幅远大于 critic_v**（-70 vs -15 @ KL=0.05），说明策略实际访问的状态（actor 路径）比 value 网络"以为"的状态更 OOD。
- **KL=1.0 的 std_ratio 崩塌更剧烈**（-0.65），与 action latent 的"维度坍缩"一致。
- **KL=0.05 的 to_centroid 漂移明显**（+0.5~0.8），与 action 的"空间漂移"一致。

**reward 跌幅 vs actor in_p90 跌幅**: Pearson r = **0.84**（n=5 主 mix run），状态分布漂移是 reward 塌的最强几何信号之一。

---

## 4. Actor vs CriticV 双流分歧（final checkpoint）

`qv_shared_detach` 配置下 actor / critic_v 是**独立 GRU + 独立 item embedding**，eval 时用 actor 出动作，但两条 stream 的 hidden state 会分叉。

### 典型模式 A：actor OOD，critic_v 仍在云内（KL=0.05 多数 run）

| Run | actor in_p90 | critic_v in_p90 | gap | reward |
|-----|:---:|:---:|:---:|:---:|
| mix_b0 | 0% | 61% | +61 | 0.62 |
| mix_b8 | 2% | 58% | +56 | 0.72 |
| topdown_b10 | 4% | 93% | +90 | 0.63 |

**含义**: Value 网络仍"以为"自己在数据支持区内做 bootstrap，但 actor 实际把轨迹带到了 OOD 状态 → Advantage 估计失真，加剧恶性循环。

### 典型模式 B：actor 仍在云内，critic_v OOD（KL=1.0 部分 run）

| Run | actor in_p90 | critic_v in_p90 | gap | reward |
|-----|:---:|:---:|:---:|:---:|
| mix_b0 | 100% | 3% | -97 | 1.03 |
| mix_b10 | 100% | 2% | -98 | 0.95 |
| mix_b5 | 100% | 5% | -95 | 0.79 |

**含义**: Actor 的 belief 仍落在 dataset 支持区（甚至 100% in_cloud），但 critic_v 的表征已坍缩/漂移到另一子空间 → Q/V 在错误的状态几何上拟合，与 action 侧"latent 仍在云内但 std_ratio 坍缩"（kl1/mix_b8 action final in_p90=100%）形成对照。

### 例外：kl1/mix_b8 final

- actor: in_p90=**1%**, std_ratio=0.13（严重坍缩+漂移）
- critic_v: in_p90=**99.5%**（几乎全在云内）
- 与 action 侧 paradox 一致：action latent 仍在云内 100%，但 belief actor 已塌到角落

---

## 5. Belief vs Action 联合解读

| 现象 | Action 几何 | Belief 几何 | 最可能机制 |
|------|------------|------------|-----------|
| KL=0.05 mix_b0 塌 | 漂到云外，std↓ | actor 0% in_cloud | **状态+动作双漂移**，policy-induced shift |
| KL=1.0 mix_b8 塌 | 仍在云内 100%，std↓↓ | actor 1% in_cloud | **状态表征坍缩**比动作 latent 更严重；decode 到低频 item |
| KL=0.05 mix_b10 相对稳 | in_p90 仍 29% | actor 69% in_cloud | 漂移较轻，item_freq 仍 1.91（非零） |
| reward↓ 最强相关 | item_freq_pct | actor in_p90↓ | 三者同源：偏离数据流形 → 低频 item → 低 reward |

**决策树（组会用）**:

```
final reward 塌了？
├─ actor belief in_p90 大降 ──→ 状态分布 shift（主因）
│   ├─ action 也漂 ──→ 动作+状态正反馈环
│   └─ action 未漂 ──→ decode / item support 局部问题
├─ actor 仍 in_cloud，critic_v 漂 ──→ Value 几何错位（KL=1.0 常见）
└─ 两者都坍缩（std_ratio<<1）──→ 表征维度锁死
```

---

## 6. 图示要点（PCA, per-checkpoint projection）

已查看代表图：`kl005/mix_b0`, `kl1/mix_b0`, `kl005/mix_b8`, `kl1/mix_b8`

**mix_b0（KL=0.05 & KL=1.0）**:
- BEST: actor/critic_v 彩点散布在灰云主体，reward ~1.7–2.2
- FINAL: 彩点收缩为 1–2 个小簇，远离灰云主密度；reward ~0.6–1.0

**mix_b8 KL=1.0**:
- BEST: 覆盖整个灰云
- FINAL: actor 塌到左下角单点（in_p90=1%），视觉上是"状态锁死"

**mix_b8 KL=0.05**:
- FINAL: 彩点成 3–4 个紧致簇漂到云边缘（漂移型，非单点坍缩）

图路径:
```
outputs/figures/belief/{kl005,kl1,kl005_mf}/{run}/
├── pca_actor_{reward,combo_hit,item_freq_pct}.png
├── pca_critic_v_{reward,combo_hit,item_freq_pct}.png
├── tsne_actor_*.png
└── tsne_critic_v_*.png
```

---

## 7. 结论（接入 total_summary 叙事）

1. **Reward 先升后塌伴随明确的 state distribution shift** — 不是单纯 decode 问题，GRU 访问的状态在 final 阶段系统性偏离 dataset belief cloud。

2. **KL=0.05**: 主要 failure mode 是 **actor belief 空间漂移**（in_p90 平均 -70%），critic_v 相对滞后（-15%）→ Advantage 基于"过时"的状态几何。

3. **KL=1.0**: 主要 failure mode 是 **双流表征坍缩/错位** — actor 或 critic_v 之一严重 OOD，std_ratio 平均 -0.65。

4. **与 action 几何互补**: Action 回答"输出什么"；Belief 回答"在什么状态下决策"。两者同时漂 → 正反馈崩塌环；仅 action 漂 → 偏 decode；仅 belief 漂 → 策略在 OOD 状态下仍试图输出"看似合理"的动作。

5. **对后续方向**: Flow Matching / Diffusion Policy 需同时约束 **action cloud** 与 **belief cloud**；仅修 GeMS 宽度不够，需 BC/行为克隆锚定状态轨迹，或统一 actor-critic 的 GRU 表征（减少 qv_shared_detach 带来的几何分裂）。

---

## 8. 复现

```bash
cd experiments/action_cloud/policy_geometry_tsne/gru_belief
/data/liyuefeng/miniconda3/envs/gems/bin/python generate_all_belief_plots.py --extract --plot

# 仅重算指标
/data/liyuefeng/miniconda3/envs/gems/bin/python3 -c "
import json, numpy as np
from pathlib import Path
# ... 见 outputs/belief_metrics_summary.json 生成逻辑
"
```
