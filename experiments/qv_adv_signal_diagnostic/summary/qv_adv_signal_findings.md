# Q/V/Adv 信号诊断结论（说人话版）

## 我到底做了什么

我没有重新训练模型，而是拿已经训练好的 checkpoint 做离线体检。

具体做法是：对同一批 replay states，准备几种不同的 action，然后问 critic：

```text
你觉得哪个 action 的 Q 更高？
```

几种 action 是：

| 名字 | 人话解释 |
|---|---|
| `data_action` | 数据集里这个 state 原本对应的动作 |
| `policy_mu` | actor 当前最想选的动作 |
| `policy_sample` | actor 随机采样出来的动作 |
| `elite_data_action` | 数据集中高 reward 样本对应的动作 |
| `random_latent` | latent 空间里随机乱采的动作 |
| `shuffled_data_action` | 别的 state 的数据动作，作为错配对照 |

如果 critic 是靠谱的，理论上应该大致满足：

```text
elite_data_action > data_action > random_latent
```

也就是：高 reward 数据动作应该比普通数据动作好，普通数据动作应该比随机乱采好。

---

## 我看了哪些指标

### 1. `data_action Adv q90`

这个指标的意思是：

```text
数据动作里前 10% 比较好的动作，比当前默认水平 V 高多少
```

也就是 AWR 还能不能分清“哪些动作值得重点模仿”。

如果它高，说明 Advantage 还有信号；如果它低，说明 AWR 越来越像普通 BC。

第一轮结果：

```text
mix:      0.365 -> 0.135
topdown: 0.330 -> 0.116
```

结论：**Adv 后期确实变弱了。**

### 2. `adv_near_zero_rate`

这个指标的意思是：

```text
有多少动作的 Q - V 接近 0
```

接地气说，就是 critic 觉得“这些动作都差不多，没有谁特别值得学”。

第一轮结果：

```text
mix 10k:      约 69% 动作接近 0
topdown 10k: 约 79% 动作接近 0
```

结论：**后期大多数训练动作对 AWR 来说没有明显区分度。**

### 3. `q_gap_vs_data_mean`

这个指标的意思是：

```text
换一个 action 后，critic 给的 Q 比原始 data_action 高还是低
```

例如：

```text
random_latent q_gap_vs_data_mean = +0.297
```

意思就是 critic 觉得随机 latent 的 Q 平均比数据动作高 `0.297`。

这很危险，因为随机动作不应该被系统性估得比数据动作更好。

---

## 最关键的发现

### 发现 1：Adv 后期确实没信号

`data_action Adv q90` 从早期三四成掉到 10k 的一成左右。

这说明后期 actor update 的 AWR 权重会越来越平均，策略更新越来越像普通 BC。

### 发现 2：不是 beta 导致 Adv 消失

同一个 env 里，`b0` 和 `b8` 的 critic/value/Adv 形状几乎一样。

例如 mix：

```text
mix_b8: Adv q90 0.365 -> 0.135
mix_b0: Adv q90 0.365 -> 0.135
```

所以 beta 主要影响 actor 怎么用 Advantage，而不是 Advantage 本身为什么消失。

### 发现 3：Q 对 action 的排序也不稳

如果 Q 很靠谱，`elite_data_action` 应该明显高于 `data_action`。

但第一轮结果里：

```text
elite_data_action 后期没有稳定高于 data_action
```

更糟的是 mix 后期：

```text
random_latent 比 data_action 的平均 Q 更高
10k random_latent q_gap_vs_data_mean ≈ +0.297
```

这说明问题不只是 `V` 把 `Q-V` 压小了，critic 自己对“哪个 action 更好”的排序也开始不可靠。

---

## 最终结论

原来我们以为问题可能只是：

```text
V 追上 Q -> Q - V 变小 -> AWR 没信号
```

现在更准确的说法是：

```text
V 确实把 Adv 压小了；
但 Q 对 action 的局部排序也不够可信。
```

所以不能简单粗暴地“让 Adv 一直变大”。

因为如果 Q 排序错了，放大 Adv 只会让 actor 更认真地学错东西，比如被拉向 random latent。

---

## 下一步我建议做什么

下一步不要只改 `beta`，也不要只压低 `V`。

更合理的是做一个小训练 ablation：

```text
baseline: 当前 IQL
variant A: 降低 value_lr / 延迟 V 更新
variant B: 加 action ranking / contrastive critic regularizer
variant C: A + B
```

核心目标是让 critic 学会：

```text
高 reward 动作 > 普通数据动作 > 随机动作
```

只有这个排序先变可靠，再让 Advantage 保持信号才有意义。

---

## 文件位置

```text
experiments/qv_adv_signal_diagnostic/PLAN.md
experiments/qv_adv_signal_diagnostic/scripts/probe_qv_action_ranking.py
experiments/qv_adv_signal_diagnostic/outputs/qv_action_ranking.csv
experiments/qv_adv_signal_diagnostic/summary/qv_adv_signal_findings.md
```

注意：`outputs/qv_action_ranking.csv` 会被完整 probe 重新覆盖生成。
