# 实验 4: 真实 click 替代 fake_zero

> 优先级 P1 | 不改 GeMS 模型 | 只改 IQL 标签生成

---

## 1. 背景

当前所有实验使用 `label_click_mode="fake_zero"`。在 IQL 训练中，每个 slate 的 GeMS 标签是这样生成的：

```python
# iql/agent.py train() 中
label_clicks = torch.zeros_like(slates, dtype=torch.float32)  # 全填 0
true_actions, _ = self.ranker.run_inference(flat_slates, label_clicks)
```

**fake_zero 的后果**：GeMS encoder 的输入是 `[item_embeddings(slate) | clicks]`。clicks 全为 0 意味着：同一个 slate 在任何上下文中都产生完全相同的 latent_mu。1M 条数据里可能有大量重复 slate，但因为 clicks 恒为 0，它们的 latent 完全一样 — 动作云被"复制粘贴"挤压了。

**真实 click 的不同**：数据集里的 clicks 来自在线 SAC+GeMS 策略与 RecSim 环境的实际交互。同一个 slate 被不同用户在不同 boredom 状态下看到，点击了不同位置 — 因此同一 slate 对应不同的 click pattern。GeMS encoder 同时看到 slate embedding 和 click pattern，会产生不同的 latent_mu。这意味着：动作云中同一个 slate 不再是一个点，而是一个分布 — 云的有效体积变大了。

旧报告估计这个变化的多样性提升约 +28%。

---

## 2. 假设

换成 `label_click_mode="real"` 后：

假设 1（云变宽）：同一 slate 在不同 click 下的 latent_mu 方差增大，有效 PC 数增加，条件数降低。
假设 2（训练改善）：Actor 有更大的目标区域可以命中 → 策略锁死缓解（floor_hit 涨得更慢或终值更低），slate 多样性崩塌推迟或减轻。
假设 3（reward 提升）：eval reward 的终值高于 fake_zero 基线，尤其是 β>0 的 run。

---

## 3. 改动范围

只改一个 CLI 参数。当前实验 `config.yaml` 中：

```yaml
label_click_mode: fake_zero
```

改为：

```yaml
label_click_mode: real
```

代码侧：`train_agent.py` 的 action normalization 和 IQLAgent.train() 都已经支持 `label_click_mode="real"` — 会从数据集中读取真实的 clicks 而非填零。不需要改任何代码。

---

## 4. 实验矩阵

| run | env | β | GeMS | label_click_mode | max_timesteps | 对比基线 |
|-----|-----|---|------|-----------------|---------------|---------|
| mix_b0_real | mix_divpen | 0 | ideal_init | real | 100k | mix_b0 (beta_ablation_repreduce) |
| mix_b5_real | mix_divpen | 5 | ideal_init | real | 100k | mix_b5 |
| mix_b8_real | mix_divpen | 8 | ideal_init | real | 100k | mix_b8 |

选 β=0, 5, 8 的原因：
- β=0（BC）：看纯模仿下 real click 是否比 fake_zero 表现更好
- β=5：中等 β，看 real click 是否延长 Adv 有效窗口
- β=8：高 β，看 real click 是否改变"早期冲高→崩塌"的模式

3 个 run，先快速跑一轮看效果。如果方向对，再扩展到 topdown 和更多 β。

---

## 5. 要对比的关键指标

从 `beta_ablation_repreduce` 的 fake_zero 基线中提取对应 β 的数据，与 real click 版本逐项对比：

### 静态动作云指标（训练前，离线分析）

| 指标 | 量什么 | 预期变化 |
|------|--------|---------|
| 同一 slate 的 latent 方差 | 取 top-10 高频 slate，看它们在 real click 下的 latent_mu 分布 | real > fake_zero，且 > 0 |
| SVD 有效 PC 数 | 1M 条 real click 标签的有效维 | real > fake_zero（2 → ?） |
| 条件数 | 最大/最小奇异值比 | real < fake_zero（761 → ?） |
| 每维 std 均值 | 32 维各自的 std | real > fake_zero（0.13 → ?） |
| L2 质心散布 | 标签间典型 L2 距离 | real > fake_zero（1.16 → ?） |

### 训练侧指标（从 SwanLab 提取，每 500 步）

| 指标 | SwanLab 路径 | 量什么 |
|------|-------------|--------|
| eval reward | `00_Eval/Reward/mean` + `iqm` | 最终裁判 |
| adv_q90 | `11_Train-Value/Adv-Quantile/q90` | Adv 信号强度与消亡时间 |
| z_center | `12_Train-Policy/Geo/z_to_dataset_center_mean` | 策略是否更靠近云 |
| floor_hit | `12_Train-Policy/LogStd/floor_hit_rate` | 策略锁死程度 |
| ood_det | `12_Train-Policy/Geo/ood_distance_mean_det` | 连续空间局部对齐 |
| global_unique | `00_Eval/Task/global_unique_items` | slate 多样性 |
| combo_hit | `00_Eval/Task/combo_soft_hit_rate` | slate 是否在训练数据流形上 |

### 关键步

沿用分析计划的 8 个关键步：500, 5000, 11000, 15000, 20000, 30000, 50000, 100000

---

## 6. 对比表模板

### 表 R-1: real vs fake_zero 静态云指标

| 指标 | fake_zero (基线) | real (mix) | 变化 |
|------|-----------------|-----------|------|
| SVD 80% PC 数 | 2 | | |
| 条件数 | 761 | | |
| 每维 std 均值 | 0.13 | | |
| L2 质心散布 | 1.16 | | |
| 同一 slate 的 latent std | 0（确定性） | | |

### 表 R-2: 训练指标逐 run 对比（关键步 5k/11k/20k/100k）

| step | 指标 | mix_b0_fake | mix_b0_real | mix_b5_fake | mix_b5_real | mix_b8_fake | mix_b8_real |
|------|------|-----------|-----------|-----------|-----------|-----------|-----------|
| 5000 | eval_reward | | | | | | |
| | adv_q90 | | | | | | |
| | floor_hit | | | | | | |
| | global_unique | | | | | | |
| 11000 | eval_reward | | | | | | |
| | adv_q90 | | | | | | |
| | floor_hit | | | | | | |
| | global_unique | | | | | | |
| 20000 | eval_reward | | | | | | |
| | adv_q90 | | | | | | |
| | floor_hit | | | | | | |
| | global_unique | | | | | | |
| 100000 | eval_reward | | | | | | |
| | adv_q90 | | | | | | |
| | floor_hit | | | | | | |
| | global_unique | | | | | | |

### 表 R-3: 终值汇总

| run | 终值 reward | IQM | Unique items | 峰值 reward | 峰值 step |
|-----|-----------|-----|:---:|-----------|----------|
| mix_b0_fake | 101.9 | 101.9 | 23 | 109.7 | 11k |
| mix_b0_real | | | | | |
| mix_b5_fake | 81.8 | 81.8 | 87 | 112.4 | 50k |
| mix_b5_real | | | | | |
| mix_b8_fake | 72.6 | 76.9 | 34 | 116.2 | 50k |
| mix_b8_real | | | | | |

---

## 7. 预期

如果假设成立：

- 静态云分析：real click 使有效 PC 从 2 增加到 3-4，条件数从 761 降到 200-400，每维 std 从 0.13 升到 0.18-0.25
- 训练侧：real click 的 eval reward 在 peak 和终值上均高于 fake_zero，floor_hit 涨得更慢，global_unique 崩塌得更慢
- 尤其是 β=5 和 β=8 的后期崩塌应该被缓解 — 因为云宽了，Actor 更容易命中有效 slate 组合

如果假设不成立（real click 无明显改善）：

- 说明"fake_zero 挤压云"这个假设本身需要修正
- 窄云的根因更可能在 GeMS encoder 结构和 KL 正则化，而非输入信号的多样性
- 后续重点回到实验 2（λ_KL 消融）和实验 3（embedding 正交化）

---

## 8. execution

```bash
cd /data/liyuefeng/offline-slate-rl-v2
bash experiments/action_cloud/real_click/run.sh
```
