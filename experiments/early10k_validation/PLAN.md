# Early10K Validation 实验计划与当前诊断

> 这是从旧 `early10k_dense_validation` 计划收敛后的复盘版。核心目标仍然是用 0~10k 的高密度评估、det/samp 双轨、静态基线和动作/GRU 几何，解释早期高分到底来自 RL 学习、采样方差，还是 GeMS 潜空间甜区。

## 1. 实验动机

前置实验各有缺口：

- `pre50k_kl_sweep_geometry` 覆盖长程训练和 checkpoint 几何，但没有 step0，0~250 步太稀，无法解释早期尖峰。
- `rl_effectiveness_validation` 有 frozen actor、random latent、cloud latent 等无 RL 基线，但矩阵较窄，没有覆盖 KL/env/beta 全组合。

本实验只看 0~10k 的因果窗口，不重复证明 50k retain。要把高 reward 拆成三类来源：

```text
1. GeMS latent space 本身有甜区：无状态随机/云采样/云质心也能高分。
2. Gaussian actor 的 samp 方差撞到甜区：samp 高，但 det 低或不稳定。
3. Actor 均值 μ 真正学到 state-conditioned action：det 和 samp 同步改善，且不靠低 unique / 高 combo repeat。
```

## 2. 核心问题

1. 0~250 步的早期高分是否真实存在，还是旧实验 eval 稀疏造成的错觉？
2. det 与 samp 的分叉能否由 `log_std`、`combo_hit`、`global_unique`、动作云位置解释？
3. RL 训练相对静态基线的净增益是多少？
4. beta=0/2/5/8 的差异是否体现为 Advantage 加权带来的稳定收益，还是都被同一个潜空间甜区主导？
5. 动作 latent 与 GRU belief 演变是否支持 in-cloud learning，还是策略坍缩到窄区域？

## 3. 实验矩阵

训练矩阵是 2 个 KL × 2 个环境 × 4 个 beta：

```text
KL:        0.01, 0.05
Env:       mix_divpen, topdown
Beta:      0, 2, 5, 8
Dataset:   b5
Seed:      58407201
Init:      ideal_init GeMS
Eval:      det + samp dense timeline
```

对应 16 个训练 run。`kl005_topdown_b0_ideal_init` 没有完整跑到 10k，当前 checkpoint 到 4000，timeline 到 4600；绘图时按缺失 step 自动跳过。

静态基线按 KL × env 配对，包括 frozen init actor、random latent、cloud sample latent、cloud centroid latent。它们的作用是判断训练曲线相对“完全不学”“无状态乱采样”“云内乱采样”“固定质心甜区”到底多了多少真实 RL 增益。

## 4. 可视化设计

联排图使用统一 checkpoint tag，方便横向比较：

```text
step0, step25, step50, step125, step250, step1000, step3000, step5000, step8000, step10000
```

`kl005_topdown_b0_ideal_init` 因缺 checkpoint，只画已有的 `step0/25/50/125/250/1000/3000`。

产物分两类：

- action latent panel：灰色是数据集 action latent cloud，彩色是当前 policy rollout action latent。
- GRU belief panel：`dataset_panels_*` 看数据集轨迹在训练 GRU 空间里的变化，`policy_vs_dataset_panels_*` 看测试策略 belief 相对数据集 belief 的位置。

## 5. 当前产物状态

当前已经落盘的可视化素材：

```text
geometry_exports/action: 346 个 .npz（包含每个 run/mode 的 dataset_cloud.npz）
geometry_exports/belief: 314 个 .npz
analysis/figures:        240 张 .png
```

重建脚本：

```text
experiments/early10k_validation/scripts/run_registry.py
experiments/early10k_validation/scripts/plot_action_trajectory.py
experiments/early10k_validation/scripts/plot_belief_trajectory.py
experiments/early10k_validation/scripts/replot_curated_from_exports.sh
experiments/early10k_validation/scripts/postprocess_curated.sh
experiments/early10k_validation/scripts/postprocess_curated_parallel.sh
```

重新画全部图：

```bash
bash experiments/early10k_validation/scripts/replot_curated_from_exports.sh all
```

## 6. 初步判断

早期高分是真实存在的，不是旧实验 eval 稀疏造成的错觉；但它更像 IQL/AWR 快速把 actor 推进 GeMS/env 的高 reward 窄甜区，而不是稳定、多样化的 state-conditioned RL 增益。

关键现象：

- 很多 run 在 25~175 步已经到达 samp peak，说明早期变化很快。
- peak 往往伴随 `combo_hit` 高、`global_unique` 低、`log_std` 下降，说明高分和重复/坍缩相关。
- 多数 run 到 10k final 没有稳定超过 cloud sample latent 基线。
- topdown 的先升后降非常清楚，说明高 reward attractor 存在，但训练后期无法稳定留住。

## 7. 与旧预设对照

```text
H1: kl001_mix_b5 @250 是 det/samp 双高，不是纯 σ
结论：部分支持。确实存在 det/samp 双高阶段，但 unique 很低，更像 μ 进入窄甜区。

H2: kl005_mix_b8 @250 是 samp 高、det 低的采样效应
结论：支持。该 run 存在非常清楚的 samp>det 分叉。

H3: 0~250 若渐变则支持早期快变；若单点尖峰则是 eval 稀疏假象
结论：支持早期快变，否定“只是稀疏 eval 假象”。但快变不是稳健学习，更像甜区搜索/坍缩。

H4: samp 稳定段约等于 B2 且高于 B0
结论：弱支持。部分 run final samp 高于 B0，但多数没有稳定高于 cloud sample。
```

## 8. 下一步

1. 画 timeline 总图：det/samp reward、samp-det gap、combo/unique、log_std、adv_q90。
2. 人工标注 action/belief panel：peak 处是否为小岛，final 是否回到数据云，samp 是否只是覆盖甜区。
3. 下一轮 ablation 优先测试：降低 AWR clip/beta、加 entropy 或 log_std floor、加入 diversity-aware checkpoint metric、对 repeated slate/high combo 加 penalty。
