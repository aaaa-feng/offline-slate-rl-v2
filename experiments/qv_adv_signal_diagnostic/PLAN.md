# Q/V/Adv Signal Diagnostic

## 这次实验想回答什么

前面 `early10k_validation` 里看到一个问题：训练前期 reward 很快上去，后期 `Adv = Q - V` 信号越来越弱，AWR 变得像普通 BC。

这次实验不重新训练，只加载已有 checkpoint，问 critic 一个很直接的问题：

```text
同一个 state 下，critic 到底觉得哪个 action 更好？
```

如果 critic 排序靠谱，应该大致满足：

```text
高 reward 数据动作 > 普通数据动作 > 随机 latent 动作
```

如果这个排序不成立，就不能简单说“让 Adv 一直变大”，因为放大错误的 Adv 会把 actor 拉向错的 action。

## Probe 怎么做

对同一批 replay states，构造几类 action：

| action source | 人话解释 |
|---|---|
| `data_action` | 数据集里这个 state 原本对应的 action |
| `policy_mu` | actor 当前最想选的均值 action |
| `policy_sample` | actor 当前随机采样出来的 action |
| `elite_data_action` | 数据集中高 reward transition 对应的 action proxy |
| `random_latent` | 在 latent 空间里随机乱采的 action |
| `shuffled_data_action` | 同一个 batch 里别的 state 的 data action，作为错配对照 |

然后记录：

```text
Q(action)
V(state)
Adv(action) = Q(action) - V(state)
Q gap vs data_action
Adv near-zero rate
```

## 最关键的三个指标

### 1. `data_action Adv q90`

数据动作里“前 10% 比较好的动作”比默认水平 `V` 高多少。

它大，说明 AWR 还能分清哪些动作更值得模仿；它小，说明 AWR 快变成普通 BC。

### 2. `adv_near_zero_rate`

有多少动作的 `Q - V` 接近 0。

接地气说，就是 critic 觉得这些动作“都差不多，没有谁特别值得学”。

### 3. `q_gap_vs_data_mean`

换一个 action 后，critic 给的平均 Q 比原始 `data_action` 高还是低。

例如：

```text
random_latent q_gap_vs_data_mean > 0
```

就表示 critic 居然觉得随机 latent 比数据动作更好，这是危险信号。

## 第一轮运行配置

```text
runs:
  kl001_mix_b8_ideal_init
  kl001_mix_b0_ideal_init
  kl001_topdown_b8_ideal_init
  kl001_topdown_b0_ideal_init

steps:
  50, 250, 1000, 5000, 10000

batch-size:
  32 episodes

max-transitions:
  3000

python:
  /data/liyuefeng/miniconda3/envs/gems/bin/python
```

## 已生成 / 需要恢复的文件

```text
experiments/qv_adv_signal_diagnostic/
  PLAN.md
  scripts/probe_qv_action_ranking.py
  outputs/qv_action_ranking.csv
  outputs/per_checkpoint/{run}/step{step}_qv_action_ranking.json
  summary/qv_adv_signal_findings.md
```

当前 outputs 还在，脚本和 summary 需要补回。

## 已得到的第一轮结论

1. `data_action Adv q90` 后期明显下降：mix 从约 `0.365` 到 `0.135`，topdown 从约 `0.330` 到 `0.116`。
2. `near_zero_rate` 后期很高：mix 10k 约 `69%`，topdown 10k 约 `79%`。
3. `elite_data_action` 后期没有稳定比 `data_action` 更高 Q。
4. mix 后期 `random_latent` 的平均 Q 甚至高于 data action，10k 约 `+0.297`。

所以结论是：

```text
Adv 后期确实变弱；
但更麻烦的是 Q 对 action 的排序也不可靠。
```

因此下一步不应该只是“让 Adv 数值变大”，而是要同时做：

```text
减慢 V 追 Q
  + 加 action ranking / contrastive critic 约束
```

目标是让 critic 学会：

```text
高 reward 动作 > 普通数据动作 > 随机动作
```
