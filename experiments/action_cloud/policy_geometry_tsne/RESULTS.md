# Policy Geometry t-SNE 诊断结果

> 数据: 5 个 mix run (KL=0.05 × mix_b0/b8/b10, KL=1.0 × mix_b8/b10)
> 每 run 对比: iql_best.pt vs iql_final.pt
> 定量指标定义见下表, 原始数据: `outputs/metrics_summary.json`

---

## 1. 定量指标定义

| 指标 | 含义 | 怎么算 |
|------|------|--------|
| `to_centroid_mean` | 策略点群到数据云质心的平均 L2 距离 | raw latent 归一化后, `‖p - centroid‖` 均值 |
| `nn_dist_mean` | 策略点到最近云邻居的平均 L2 距离 | sklearn NearestNeighbors(k=1) |
| `per_dim_std_ratio` | 策略 per-dim std / 云 per-dim std | `p_n.std(axis=0).mean() / d_n.std(axis=0).mean()` |
| `in_cloud_p90` | 策略点落在云 p90 半径内的比例 | 到 centroid 距离 ≤ cloud p90 → in cloud |
| `combo_hit` | 解码 slate 命中训练集 top-1000 combo 的比例 | `tuple(slate) in combo_freq_counter` |
| `item_freq_pct_mean` | 解码 slate 内 item 的训练集频率百分位均值 | `mean(item_freq / item_total * 100)` |

---

## 2. 核心结果表

| KL | Run | Tag | Step | to_centroid | nn_dist | std_ratio | in_p90% | reward |
|----|-----|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.05** | mix_b0 | best | 40k | 1.412 | 0.878 | **0.85** | 83% | 2.2 |
| | | final | 100k | 2.168 | 1.480 | **0.24** | **3%** | 0.6 |
| **0.05** | mix_b8 | best | 14.5k | 1.276 | 0.688 | **0.86** | 100% | 2.1 |
| | | final | 100k | 1.964 | 1.267 | **0.73** | **38%** | 0.7 |
| **0.05** | mix_b10 | best | 2.5k | 1.043 | 0.525 | **0.78** | 100% | 2.0 |
| | | final | 100k | 1.970 | 1.378 | **1.28** | **29%** | 0.9 |
| **1.0** | mix_b8 | best | 41k | 1.054 | 0.134 | **0.62** | 94% | 2.0 |
| | | final | 100k | 1.088 | 0.247 | **0.19** | **100%** | 0.7 |
| **1.0** | mix_b10 | best | 7k | 1.372 | 0.285 | **0.88** | 95% | 1.5 |
| | | final | 100k | 2.284 | 0.354 | **0.24** | **3%** | 0.9 |

---

## 3. Slate 质量指标（离散空间）

> 口径：与 `eval_env.py` / `train_agent.py` 一致 — top-1000 combo 字典 + 全局 item 频率百分位。
> 数据来源：`augment_slate_metrics.py` 对 10 个 `.npz` 的后处理结果。

| KL | Run | Tag | Step | combo_hit | item_freq_pct | reward |
|----|-----|-----|:---:|:---:|:---:|:---:|
| **0.05** | mix_b0 | best | 40k | 0% | 5.30 | 2.2 |
| | | final | 100k | 0% | **0.06** | 0.6 |
| **0.05** | mix_b8 | best | 14.5k | 12.3% | 6.32 | 2.1 |
| | | final | 100k | 0% | **0.08** | 0.7 |
| **0.05** | mix_b10 | best | 2.5k | 58.2% | 6.22 | 2.0 |
| | | final | 100k | 0% | **1.91** | 0.9 |
| **1.0** | mix_b8 | best | 41k | 56.7% | 6.13 | 2.0 |
| | | final | 100k | 0% | **0.03** | 0.7 |
| **1.0** | mix_b10 | best | 7k | 45.3% | 4.37 | 1.5 |
| | | final | 100k | **47.0%** | **1.34** | 0.9 |

### 发现

1. **所有 final 的 `item_freq_pct` 都暴跌** — best 在 4.4-6.3，final 在 0.03-1.91（10-200× 下降）。策略最终推的 item 在训练集中几乎没见过。这与 latent 空间漂移/坍缩互为因果：latent 偏了 → 解码出稀有/不存在 item；推了稀有 item → GRU 状态偏离 → 进一步推偏。

2. **大部分 final 的 `combo_hit` 归零** — 唯一例外是 **KL=1.0 mix_b10 final 维持 47%**（best 是 45.3%，略有上升）。这说明 KL=1.0 mix_b10 的"坍缩+偏移"模式下，策略虽然维度坍缩严重（std_ratio=0.24），但输出的 slate 仍落在训练集高频组合上 — **坍缩到了高频 combo 上**，但 reward 从 1.5→0.9，说明这些高频 combo 在 RecSim 环境下并不好。

3. **KL=0.05 mix_b0 final 的 `item_freq_pct=0.06`** — 几乎为零，配合 PCA 中策略完全漂到云外，确认了 KL=0.05 的"空间漂移"不仅是 latent 层面的现象，更是离散 slate 层面的后果。

4. **best 阶段的 combo_hit 与 beta 相关** — mix_b0 (beta=3.0, no BC) best 已经是 0%，而 mix_b8/mix_b10 (beta=8/10) best 在 12-58%。说明 Actor 的 beta 越高，策略在 best checkpoint 时越偏向数据集中已有的组合。但 beta 无法阻止 final 崩塌。

### 二维着色图

30 张 PCA/t-SNE 图已生成，覆盖所有 3 种着色方式：

```
outputs/figures/{kl005,kl1}/{mix_b0,mix_b8,mix_b10}/
├── pca_reward.png         # 连续着色（RdYlGn, 0-200）
├── pca_combo_hit.png      # 离散着色（蓝=miss, 红=hit）
├── pca_item_freq_pct.png  # 连续着色（viridis）
├── tsne_reward.png
├── tsne_combo_hit.png
└── tsne_item_freq_pct.png
```

一键重新生成：
```bash
/data/liyuefeng/miniconda3/envs/gems/bin/python generate_all_plots.py --augment --force
```

---

## 4. 结论

### 两种不同的崩塌模式

**KL=1.0（维度坍缩型）**：
- BEST→FINAL 的 `std_ratio` 从 0.62-0.88 降到 0.19-0.24（3-4× 坍缩）
- 策略输出被压扁到极少数维度上（"锁死"）
- mix_b8 在空间上没漂（in_cloud 94%→100%），但维度坍缩后策略无操作空间
- cloud 本身的 `per_dim_std` 也不高（0.21），KL=1.0 把云和策略都压扁了

**KL=0.05（空间漂移型）**：
- BEST→FINAL 的 `to_centroid` 从 ~1.1-1.4 涨到 ~2.0（+50%+）
- `in_cloud_p90` 从 83-100% 暴跌到 3-38%
- 策略漂到数据云外面去了
- 但 `std_ratio` 维持在 0.73-1.28 —— 维度利用仍然合理
- **KL=0.05 解决了维度坍缩，但引入了空间漂移**

### Best checkpoint 的步数差异

- **KL=0.05 mix_b10 best 仅在 2,500 步** — 几乎初始即巅峰，后续 97,500 步全在退化
- **KL=0.05 mix_b0 best 在 40,000 步** — 峰值最晚，但仍挡不住 final 崩塌
- **KL=1.0 的 best 步数（7k-41k）普遍晚于 KL=0.05 的同配 run** — KL=1.0 的策略收敛更慢但更稳定（mix_b8 final 仍在云内）
- 所有 final 均在 100k 步 — 在 best 之后的数万步里，策略一直在恶化而非收敛

### 一句话

KL=0.05 没有"完全解决"动作空间问题——它把问题从**低维锁死**（KL=1.0, 策略被压扁无法探索）变成了**空间漂移**（策略维持了维度但后期漂到了云外）。两个 failure mode 都需要额外修复——BC 兜底、Actor 温度缩放、或更大的 KL 值介于两者之间。

---

## 5. 产出索引

```
experiments/action_cloud/policy_geometry_tsne/
├── PLAN.md                                    # 实验计划
├── RESULTS.md                                 # 本文件（action cloud 结果）
├── action_cloud/                              # === Action Latent 分析 ===
│   ├── extract_policy_geometry.py             # 提取脚本
│   ├── plot_policy_geometry.py                # 画图脚本（--color_by）
│   ├── augment_slate_metrics.py               # 后处理：补充 slate 指标
│   ├── generate_all_plots.py                  # 批量生成 action 图
│   └── extract_all_experiments.py             # 批量提取所有实验
├── gru_belief/                                # === GRU Belief 分析 ===
│   ├── README.md                              # 流程说明
│   ├── RESULTS.md                             # belief 诊断结论（见该文件）
│   ├── extract_belief_geometry.py             # 双流抽取
│   ├── plot_belief_geometry.py                # PCA/t-SNE（--stream actor|critic_v）
│   └── generate_all_belief_plots.py           # 批量生成
└── outputs/
    ├── metrics_summary.json                   # action 定量
    ├── belief_metrics_summary.json            # belief 定量（actor/critic_v）
    ├── kl005/{run}/{best,final}_geometry.npz
    ├── kl005/{run}/{best,final}_belief.npz
    ├── kl005_mf/{run}/{best,final}_geometry.npz
    ├── kl1/{run}/{best,final}_geometry.npz
    └── figures/
        ├── action/{kl}/{run}/*.png
        └── belief/{kl}/{run}/*.png    (~209 张, 2 stream × 2 method × 3 color)
```

### 一键生成

```bash
cd /data/liyuefeng/offline-slate-rl-v2
python3 action_cloud/generate_all_plots.py --augment
python3 action_cloud/extract_all_experiments.py --extract --plot
```
