# Q/V/Adv 信号诊断结论

## 0. 最重要的结论

这个实验想回答一句话：

```text
后期 AWR 没信号，是因为 Adv 被 V 压没了，还是因为 critic 本身已经不会判断哪个 action 更好？
```

答案是：**两件事都发生了，而且 Q 排序出问题的时间比 Adv 完全消失更早。**

最关键的横向对比：

```text
step50：Adv 还有信号，Q 排序正常
  elite_data_action > data_action > random_latent

step250：Adv 仍然很高，但 Q 排序已经坏了
  random_latent ≈/ > data_action，elite_data_action 反而不稳定

step10000：Adv 变弱，Q 排序也没恢复
  mix 里 random_latent 明显高于 data_action
```

所以不能简单粗暴地“让 Adv 一直变大”。如果 Q 排序错了，放大 Adv 只会让 actor 更认真地学错东西。

---

## 1. 这个 probe 到底做了什么

没有重新训练，只加载已有 checkpoint 做离线诊断。

对同一批 replay states，准备几类 action，然后问 critic：

```text
这些 action 里，你觉得谁的 Q 更高？
```

几类 action：

| action | 人话解释 |
| --- | --- |
| `data_action` | 数据集里这个 state 原本对应的动作 |
| `elite_data_action` | 高 reward dataset transition 对应的动作 proxy |
| `random_latent` | latent 空间随机乱采的动作 |
| `policy_mu` | actor 当前均值动作 |
| `policy_sample` | actor 当前随机采样动作 |
| `shuffled_data_action` | 别的 state 的数据动作，错配对照 |

这里最关心前三个：

```text
elite_data_action / data_action / random_latent
```

如果 critic 的 action ranking 健康，应该大致看到：

```text
elite_data_action > data_action > random_latent
```

注意：`elite_data_action` 不是“当前 state 的 oracle 最优动作”，而是高 reward 数据动作的 proxy。它的作用不是证明逐 state 最优，而是检查 critic 是否至少能识别“高 reward 数据动作整体上比随机动作更像好动作”。

---

## 2. 看哪些指标

### `data_action Adv q90`

含义：数据动作里前 10% 比较好的动作，比默认水平 `V(state)` 高多少。

```text
Adv = Q(data_action) - V(state)
```

人话：AWR 还能不能分清“哪些数据动作值得重点模仿”。

### `adv_near_zero_rate`

含义：有多少动作的 `Q - V` 接近 0。

人话：critic 觉得多少动作“都差不多，没有谁特别值得学”。

### `Q 排序 / q_gap_vs_data_mean`

含义：换一个 action 后，它的平均 Q 比 `data_action` 高还是低。

例子：

```text
random_latent q_gap_vs_data_mean = +0.297
```

人话：critic 觉得随机 latent 平均比数据动作还好。这是明显危险信号。

---

## 3. 横向对比：Adv 高的时候 vs 训练最后

下面只列最关键的 `b8` 两条 run；`b0` 的 critic/value/Adv 形状和同 env 的 `b8` 基本一样。

| Run | step | data Adv q90 | near-zero | elite Q | data Q | random Q | Q 排序 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mix_b8 | 50 | 0.365 | 0.37 | 0.332 | 0.278 | 0.159 | elite > data > random |
| mix_b8 | 250 | 0.465 | 0.33 | 0.511 | 0.570 | 0.573 | random > data > elite |
| mix_b8 | 10000 | 0.135 | 0.69 | 7.406 | 7.411 | 7.708 | random > data > elite |
| topdown_b8 | 50 | 0.330 | 0.40 | 0.329 | 0.221 | 0.101 | elite > data > random |
| topdown_b8 | 250 | 0.363 | 0.41 | 0.372 | 0.418 | 0.474 | random > data > elite |
| topdown_b8 | 10000 | 0.116 | 0.79 | 4.946 | 4.992 | 4.990 | data > random > elite |

### 怎么读这个表

**step50 是最健康的阶段。**  
这时 Adv 还有信号，Q 排序也符合直觉：`elite > data > random`。

**step250 是关键转折点。**  
这时 `data Adv q90` 还很高，甚至 mix_b8 从 0.365 涨到 0.465；但 Q 排序已经坏了，`random_latent` 已经接近或超过 `data_action`，`elite_data_action` 不再稳定排在最前。

**step10000 是两个问题叠加。**  
Adv 明显变弱：mix_b8 `0.135`，topdown_b8 `0.116`；near-zero 很高：mix `69%`，topdown `79%`。同时 Q 排序也没有恢复，mix 里 `random_latent` 明显高于 `data_action`。

---

## 4. beta=0 和 beta=8 的关系

同一个 env 下，`b0` 和 `b8` 的 critic/value/Adv 形状几乎一样。

例如 mix：

```text
mix_b8 data Adv q90: 0.365 -> 0.135
mix_b0 data Adv q90: 0.365 -> 0.135
```

topdown 也是同样模式：

```text
topdown_b8 data Adv q90: 0.330 -> 0.116
topdown_b0 data Adv q90: 0.330 -> 0.116
```

这说明：

```text
beta 主要影响 actor 怎么用 Advantage；
Adv 变弱和 Q 排序变坏，主要来自 critic/value 训练链路本身。
```

---

## 5. 结论：三个问题的优先级

### 1. Adv 后期确实变弱

`data_action Adv q90` 从早期三四成掉到 10k 的一成左右；`near_zero_rate` 到 10k 变得很高。

这说明后期 AWR 权重越来越平均，actor update 越来越像 BC。

### 2. 但 Q 排序更早就出问题

step250 时 Adv 还高，但 `random_latent` 已经能接近或超过 `data_action`。

这说明问题不只是 `V` 把 `Q - V` 压小，而是 critic 在 action 维度上已经开始不可靠。

### 3. 只“让 Adv 变大”不安全

如果 Q 排序不可信，强行让 `Q - V` 变大，可能只是把错误排序放大。

最危险的例子是 mix 后期：

```text
random_latent q_gap_vs_data_mean ≈ +0.297
```

如果这类错误 Q 被 AWR 放大，actor 可能会被拉向 OOD/random latent。

---

## 6. 下一步建议

下一步不要只调 `beta`，也不要只压低 `V`。

更合理的小 ablation：

```text
baseline: 当前 IQL
variant A: 降低 value_lr / 延迟 V 更新
variant B: 加 action ranking / contrastive critic regularizer
variant C: A + B
```

训练时额外约束 critic 至少学会：

```text
Q(elite_data_action) > Q(data_action) > Q(random_latent)
```

目标不是让 Adv 数值变大，而是让 **Adv 的排序方向正确**。

---

## 7. 文件位置

```text
experiments/qv_adv_signal_diagnostic/PLAN.md
experiments/qv_adv_signal_diagnostic/scripts/probe_qv_action_ranking.py
experiments/qv_adv_signal_diagnostic/outputs/qv_action_ranking.csv
experiments/qv_adv_signal_diagnostic/summary/qv_adv_signal_findings.md
```
