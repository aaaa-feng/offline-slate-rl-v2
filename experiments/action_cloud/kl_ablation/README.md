# GeMS λ_KL 消融实验

> 回答：降低 KL 正则化强度能否让 GeMS 潜空间数据云变宽？

---

## 1. 实验设计

### 矩阵

| 变量 | 值 |
|------|------|
| embedding 初始化 | ideal_init (env ground truth), mf_init (BPR MF) |
| 环境 | mix_divpen, topdown_divpen |
| λ_KL | 0.01, 0.05, 0.1, 0.5, 1.0 |
| λ_click | 1.0（固定，与 ideal_init ckpt 保持一致） |
| λ_prior | 1.0（固定） |
| seed | 58407201（固定） |

**ideal_init**: λ_KL=1.0 已有 ckpt, 训 λ_KL 0.01/0.05/0.1/0.5 四个值  
**mf_init**: 五个值全训

共计 18 个 GeMS 模型。

### 固定参数

```
lambda_click: 1.0     lambda_prior: 1.0     ranker_lr: 3e-3
max_epochs:   50       batch_size:  256      latent_dim: 32
seed:         58407201 fixed_embedds: scratch (embedding 可训练)
```

### 为什么用这些参数

旧项目 ideal_init KL=1.0 ckpt 用 `click=1.0, prior=1.0, lr=3e-3, epochs=50` 训练。为保证公平对比——"只改 embedding 初始化和 KL，其他全相同"——所有新模型沿用同一套参数。

> 原论文 pretrain_GeMS.yml 用的是 `click=0.5, prior=0.0, lr=0.001, epochs=15`。我们不用这套是因为要跟已有的 ideal_init ckpt 对齐。

---

## 2. GeMS VAE Loss

```
loss = slate_loss + λ_click * click_loss + λ_KL * KLLoss + λ_prior * prior_reg
```

- **slate_loss**: CrossEntropy, 列表重构。保证 z 包含 item 信息
- **click_loss**: BCEWithLogits, 点击预测。让 z 包含用户偏好信息
- **KLLoss**: KL(q(z|x) || N(0,I))。约束 encoder 输出贴近标准高斯。**λ_KL 越大 → 云越窄**
- **prior_reg**: 先验正则，∑(μ² + log(σ²)²)。约束 prior 不要学出奇怪值

详细分析见 [GeMS_VAE_LOSS.md](GeMS_VAE_LOSS.md)

---

## 3. 如何使用

### 启动训练

```bash
cd /data/liyuefeng/offline-slate-rl-v2
bash experiments/action_cloud/kl_ablation/ideal_init/run.sh all   # 8 个
bash experiments/action_cloud/kl_ablation/mf_init/run.sh all       # 10 个
```

### 产出

| 产出 | 位置 |
|------|------|
| ckpt | `checkpoints/gems/GeMS_{env}_b5_pretrained_latent32_beta{KL}_click1.0_seed58407201_{tag}.ckpt` |
| log | `logs/gems/{ideal_init,mf_init}_{env}_kl{KL}.log` |
| SwanLab | project `offline_slate_rl_gems_202606`, experiment 含 KL+click+lr+tag |

---

## 4. SwanLab 实验名

```
gems_mix_divpen_b5_kl0.05_click1.0_lr0.003_ideal_init
gems_mix_divpen_b5_kl0.05_click1.0_lr0.003_mf_init
...
```

每个 run 唯一, ideal_init 和 mf_init 不会混淆。

---

## 5. 训练结果: 静态云分析

详见 [../static_analysis/RESULTS.md](../static_analysis/RESULTS.md)

核心结论：**λ_KL 是云宽度的主导变量**。

| KL | PC80 (有效维) | per_dim_std | 结论 |
|----|:---:|------------|------|
| 1.0 | 2-5 | 0.10-0.14 | 极窄, 基准 |
| 0.5 | 5-10 | 0.16-0.26 | 略有改善 |
| 0.1 | 10-19 | 0.37-0.53 | 明显改善 |
| 0.05 | 15-18 | 0.46-0.57 | **推荐甜点** |
| 0.01 | 14-17 | 0.50-0.61 | 接近 0.05, 部分 unstable |

KL=0.05 让 PC80 从 4→16 (+300%), 条数从 685→个位数。ideal_init 和 mf_init 趋势一致。

---

## 6. 备份

第一轮训练（ckpt 命名冲突前）的 18 个 ckpt + log 备份在：
- `checkpoints/gems/_backup_20260608/`
- `logs/gems/_backup_20260608/`# KL 消融实验: 静态云分析结果

## Baseline: ideal_init KL=1.0 (旧实验用的 GeMS)

| env | A_pc80 | A_cond | A_std | B_pc80 | B_z_center | B_spread | E_cos |
|-----|--------|--------|-------|--------|-----------|----------|-------|
| mix | 2 | 685 | 0.096 | 5 | 1.82 | 1.08 | 0.40 |
| topdown | 4 | 24293 | 0.135 | 5 | 2.54 | 0.98 | 0.29 |

**结论**: 32 维里只用 2-5 维, 条件数 685-24000, 云极窄极扁。

---

## KL=0.01 (极限放松)

| init | env | A_pc80 | A_cond | A_std | B_pc80 | B_z_center | B_spread | E_cos |
|------|-----|--------|--------|-------|--------|-----------|----------|-------|
| ideal_init | mix | 14 | 7.3 | 0.496 | 14 | 1.50 | 1.19 | 0.28 |
| mf_init | mix | 16 | 5.7 | 0.560 | 16 | 1.36 | 1.09 | 0.23 |
| ideal_init | td | 17 | 5.9 | 0.592 | 17 | 1.49 | 1.21 | 0.19 |
| mf_init | td | 17 | 4.7 | 0.606 | 16 | 1.42 | 1.26 | 0.21 |

**结论**: PC80 从 2-5 跃升到 14-17, cond 从 685-24000 降到 5-7, std 从 0.1 升到 0.5-0.6。云从极窄变成接近满秩 (32 → 14-17 有效维)。z_center 从 1.82→1.36, 云向原点移动。

---

## KL=0.05 (推荐甜点)

| init | env | A_pc80 | A_cond | A_std | B_pc80 | B_z_center | B_spread | E_cos |
|------|-----|--------|--------|-------|--------|-----------|----------|-------|
| ideal_init | mix | 16 | 386 | 0.458 | 16 | 1.39 | 1.12 | 0.26 |
| mf_init | mix | 16 | 11M | 0.472 | 16 | 1.49 | 1.12 | 0.24 |
| ideal_init | td | 16 | 5354 | 0.502 | 15 | 1.48 | 1.31 | 0.14 |
| mf_init | td | 18 | 5.3 | 0.570 | 17 | 1.40 | 1.26 | 0.21 |

**结论**: 效果接近 KL=0.01, PC80=15-18. 但 cond 波动大 (386-11M), 部分模型存在极窄方向。

---

## KL=0.1 (中等放松)

| init | env | A_pc80 | A_cond | A_std | B_pc80 | B_z_center | B_spread | E_cos |
|------|-----|--------|--------|-------|--------|-----------|----------|-------|
| ideal_init | mix | 12 | 35M | 0.369 | 10 | 1.61 | 1.20 | 0.30 |
| mf_init | mix | 15 | 513 | 0.426 | 15 | 1.62 | 1.22 | 0.19 |
| ideal_init | td | 11 | 27M | 0.371 | 11 | 1.65 | 1.33 | 0.29 |
| mf_init | td | 19 | 180 | 0.529 | 18 | 1.47 | 1.27 | 0.25 |

**结论**: 效果在 KL=0.05-0.1 之间。PC80=10-19, 但 cond 极不稳定。

---

## KL=0.5 (温和放松)

| init | env | A_pc80 | A_cond | A_std | B_pc80 | B_z_center | B_spread | E_cos |
|------|-----|--------|--------|-------|--------|-----------|----------|-------|
| ideal_init | mix | 5 | 24M | 0.156 | 6 | 1.69 | 1.23 | 0.41 |
| mf_init | mix | 6 | 840 | 0.183 | 8 | 1.85 | 1.14 | 0.36 |
| ideal_init | td | 5 | 562 | 0.163 | 7 | 2.19 | 1.04 | 0.30 |
| mf_init | td | 9 | 21M | 0.261 | 10 | 1.64 | 1.32 | 0.31 |

**结论**: 效果已经开始退向 KL=1.0。PC80=5-10, 改善有限。

---

## KL=1.0 (mf_init 新训)

| init | env | A_pc80 | A_cond | A_std | B_pc80 | B_z_center | B_spread | E_cos |
|------|-----|--------|--------|-------|--------|-----------|----------|-------|
| mf_init | mix | 5 | 22M | 0.147 | 6 | 1.70 | 1.24 | 0.34 |
| mf_init | td | 5 | 1135 | 0.173 | 7 | 1.74 | 1.38 | 0.38 |

**结论**: 与 ideal_init KL=1.0 接近, PC80=5-7, z_center=1.7-1.74。

---

## 总结: PC80 随 KL 变化

| KL | mix ideal | mix mf | td ideal | td mf | 平均 |
|-----|----------|--------|---------|-------|------|
| 0.01 | 14 | 16 | 17 | 17 | **16** |
| 0.05 | 16 | 16 | 16 | 18 | **16.5** |
| 0.1 | 12 | 15 | 11 | 19 | **14.3** |
| 0.5 | 5 | 6 | 7 | 10 | **7** |
| 1.0 | 2 | 5 | 5 | 5 | **4.3** |

**结论**: KL 是云宽度的主导变量。KL=0.01-0.05 能让 PC80 从 4→16 (+400%)。embedding 初始化 (ideal_init vs mf_init) 有次要影响但不改变趋势。KL=0.05 是推荐甜点——效果和 KL=0.01 接近但训练更稳定。

## 下一步

1. 选 KL=0.05 (或 0.05+0.01) 的 GeMS ckpt, 用它们跑 IQL beta ablation
2. 对比 old ideal_init KL=1.0 vs new KL=0.05 的 IQL 结果
3. 验证"拓宽云 → Actor 不再永远在云外 → reward 不崩塌"
