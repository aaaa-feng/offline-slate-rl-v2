# Topdown 与 Mix 早期上升后下降现象分析

## 1. 结论先说

`topdown` 的“先上升后下降”最明显，是因为 actor 很快进入了一个高 reward 的窄 latent 区域，但这个区域不能稳定保持：继续训练后 `log_std` 收缩、动作多样性下降、Advantage 信号变弱，策略从短期甜区滑出或坍缩到低质量动作。

`mix beta=0` 也会先升后降，说明即使没有强 Advantage 加权，训练也能把 actor 从初始分布推向 GeMS 高分区域；但它的峰值更低、下降更快，因为缺少 beta=8 那种持续放大高 Advantage 动作的推力。

`mix beta=8` 能保持更高峰值优势，是因为 AWR 权重更强，早期更容易把 policy 拉到高 reward latent sweet spot 附近；但这不等于它学到了稳定推荐策略，因为高分同时伴随 combo repeat 和 unique 下降。

## 2. 看了哪些指标

```text
det_iqm_reward:
  不采样，只用 actor 均值 μ 评估。它高，说明 μ 本身到了高分区域。

samp_iqm_reward:
  从 actor 高斯分布采样评估。它高但 det 低，说明主要靠采样撞到甜区。

samp_iqm - det_iqm:
  gap 大，就是采样比均值强很多；gap 小且双高，说明 μ 和采样都在高分区。

combo_hit:
  推荐列表是否重复撞到少数高分组合。高 reward + 高 combo_hit 通常意味着策略窄。

global_unique:
  rollout 中出现了多少不同 slate。越低越像坍缩。

log_std_mean:
  actor 采样标准差的 log。越低，策略越确定，探索/覆盖越窄。

train_adv_q90:
  Advantage 的高分位信号。下降说明 AWR 后期能分辨“好动作”的信号变弱。
```

用接地气的话说：reward 告诉我们“分高不高”，combo/unique 告诉我们“是不是靠重复少数套路拿分”，log_std 告诉我们“策略还散不散”，Advantage 告诉我们“训练还知不知道哪些动作更值得学”。

## 3. 关键数字

几个典型 run 的早期 peak 和最终表现：

```text
kl001_mix_b0:
  early peak step125 samp=230.3 det=210.5
  global peak step725  samp=233.8
  final step10000      samp=151.3 det=99.5

kl001_mix_b8:
  early peak step175 samp=246.6 det=242.4
  global peak step850 samp=248.2
  final step10000     samp=180.9 det=115.4

kl005_topdown_b0:
  early/global peak step75 samp=237.4 det=113.4
  last step4600           samp=71.4  det=47.9

kl005_topdown_b8:
  early/global peak step50 samp=303.6 det=336.8
  final step10000          samp=119.5 det=61.4
```

这组数字说明两件事：

1. 早期上升是真实的，发生在 25~175 步，不是稀疏 eval 的假象。
2. 后期回落也是真实的，尤其 topdown peak 到 final 的跌幅非常大。

## 4. 为什么 topdown 更明显

topdown 的高分区域更像短期 attractor：actor 一旦被 AWR 推过去，det 和 samp 可以同时暴涨。但这个区域很窄，后续训练继续更新 Q/V/actor 后，策略不一定还能稳定待在那里。

从图上应重点看：

- action latent panel：peak 时彩色 policy 点是否集中成小岛。
- belief panel：policy belief 是否偏离或压缩到 dataset belief 的局部区域。
- combo 图：peak 的高 reward 是否伴随高 combo hit。

如果高 reward 小岛很窄，同时 unique 很低，那它更像“撞到固定套路”，不是泛化能力强。

## 5. 为什么 mix b0 会更低、更快掉

beta=0 基本弱化了 Advantage 权重，actor 更像在做均匀模仿/普通训练。它仍能上升，是因为 GeMS latent 空间本身有高 reward 区，训练会把策略从随机初始化推近数据云或甜区。

但没有强 Advantage 权重时：

- 推向高分动作的选择性更弱，所以 peak 更低。
- 一旦 Q/V/Adv 信号后期变弱，policy 没有足够力量留在高分区域，所以下降更快。

## 6. 为什么 mix b8 能保持优势

beta=8 放大正 Advantage 动作的模仿权重，早期能更强地把 actor 拉到高 reward latent 区域。和 beta=0 相比，它更容易做到 det/samp 双高，也更能在一段时间内维持高分。

但这份优势有条件：如果高分主要来自窄甜区，那 beta=8 也会加剧重复和 `log_std` 收缩。也就是说，beta=8 保住的是“更靠近高分甜区”的优势，不一定是“更稳定的个性化推荐”优势。

## 7. 对算法问题的判断

现在的主要问题不是“完全没学到”，而是“学得太容易被窄甜区吸走”：

```text
早期 Advantage/AWR 有效
  -> 快速提高 reward
  -> 同时压低多样性，combo repeat 上升
  -> log_std 收缩，采样空间变窄
  -> 后期 Advantage 信号弱化或 Q ranking 不稳
  -> peak 无法稳定保留
```

后续改法应该围绕三条线：

1. 让 Advantage 信号更可靠：检查 Q 对 data/elite/random/OOD action 的排序，必要时加 ranking/negative/OOD conservative 约束。
2. 控制 AWR 推力：降低 beta 或 clip，加入 warmup，避免早期被单一甜区吸走。
3. 保留策略覆盖：加 entropy/log_std floor，或者在 checkpoint metric 中加入 diversity/unique 惩罚项。
