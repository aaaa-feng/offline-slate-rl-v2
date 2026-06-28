# Early10K Validation

本目录用于分析 0~10k 步内 IQL actor 的早期上升、回落和几何演变。重点不是长程 retain，而是解释早期高分来自真实 state-conditioned RL、采样撞甜区，还是 GeMS latent 空间本身的高 reward 窄区域。

主要入口：

- `PLAN.md`：实验动机、矩阵、图的设计和当前诊断。
- `analysis/RESULTS.md`：当前结果摘要。
- `analysis/summary/focused_topdown_mix_dynamics.md`：重点解释 topdown 与 mix beta=0/8 的先升后降现象。
- `plot_tags_manifest.json`：联排图使用的 checkpoint step。
- `scripts/replot_curated_from_exports.sh`：从已有 `.npz` 重新生成 action/belief panel。

现有产物：

```text
geometry_exports/action/   action latent .npz
geometry_exports/belief/   GRU belief .npz
analysis/figures/action/   action latent 联排图
analysis/figures/belief/   GRU belief 联排图
eval_timeline/             det/samp dense timeline
baseline_results/          静态基线评估结果
```

重新画全部图：

```bash
bash experiments/early10k_validation/scripts/replot_curated_from_exports.sh all
```

只画单个 run：

```bash
bash experiments/early10k_validation/scripts/replot_curated_from_exports.sh kl001_mix_b8_ideal_init
```
