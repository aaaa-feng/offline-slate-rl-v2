# Dataset Action Cloud Quality（静态 PCA 图）

对每个 **GeMS checkpoint**，把离线数据集 slate 编码成 32 维 latent，在共享 PCA 平面上按 **质量标签** 着色：

- `reward` — 离线 step reward
- `item_freq_pct` — slate 内 item 频率百分位
- `combo_hit` — 是否命中 top-1000 combo

与 IQL 训练无关；回答的是：**在这个 GeMS 定义的动作空间里，甜区和垃圾区分别在哪**。

## 目录结构

```text
static_analysis/
├── cloud_quality_common.py   # 抽取 + 画图公共逻辑
├── extract_cloud_quality.py  # GeMS ckpt → exports/{slug}/dataset_cloud.npz
├── plot_cloud_quality.py     # npz → figures/{slug}/dataset_action_quality_*.png
├── run.sh                      # 一键批量或单个 slug
├── exports/{slug}/             # 中间数据（latent + reward/combo_hit/item_freq）
└── figures/{slug}/             # 输出 PNG
```

`slug` 命名：`{env}_b5_beta{kl}_{embed}`，例如 `mix_divpen_b5_beta0.05_ideal_init`。

每个 slug 下 4 张图：

| 文件 | 内容 |
|------|------|
| `dataset_action_quality_panels.png` | reward / item_freq / combo_hit 三合一 |
| `dataset_action_quality_reward.png` | 单图 reward |
| `dataset_action_quality_item_freq_pct.png` | 单图 item_freq |
| `dataset_action_quality_combo_hit.png` | 单图 combo hit |

## 用法

```bash
cd offline-slate-rl-v2

# 全部 b5 GeMS（mix + topdown × β × ideal_init/mf_init，约 20 个）
bash experiments/action_cloud/static_analysis/run.sh 0

# 只重画图（已有 exports）
bash experiments/action_cloud/static_analysis/run.sh 0 plot-only

# 单个 slug
bash experiments/action_cloud/static_analysis/run.sh 0 mix_divpen_b5_beta0.1_ideal_init

# 或分步
python experiments/action_cloud/static_analysis/extract_cloud_quality.py \
  --ckpt checkpoints/gems/GeMS_mix_divpen_b5_pretrained_latent32_beta0.1_click1.0_seed58407201_ideal_init.ckpt
python experiments/action_cloud/static_analysis/plot_cloud_quality.py \
  --slug mix_divpen_b5_beta0.1_ideal_init
```

## 数据说明

- **GeMS**：`checkpoints/gems/GeMS_{env}_b5_pretrained_latent32_beta{kl}_..._{embed}.ckpt`
- **离线数据**：`data/datasets/offline/{env}/{env}_b5_data_d4rl.npz`
- **采样**：seed=58407201，20000 transitions，`clicks=fake_zero` 过 GeMS inference
- **PCA**：在该 slug 的 20k latent 上 fit 2D PCA；画图 subsample 至 12000 点

## exports npz 字段

- `dataset_latent_raw` — [N, 32]
- `dataset_reward`, `dataset_combo_hit`, `dataset_item_freq_pct_mean`
- `dataset_center`, `sample_indices`, `metadata`（json：env、beta、ckpt 路径等）
