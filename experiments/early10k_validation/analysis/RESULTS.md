# Early10K Validation 初步结果分析

## 1. 一句话结论

早期高分是真实存在的，不是旧实验 eval 稀疏造成的错觉；但它目前更像 **IQL/AWR 很快把 actor 推入 GeMS/env 的高 reward 窄甜区**，而不是稳定、多样化的 state-conditioned RL 增益。多数 run 在 10k 末端没有稳定超过 cloud sample latent 基线。

## 2. 当前产物

```text
训练 timeline: 16 个 run
action .npz:   346 个（含 dataset_cloud.npz）
belief .npz:   314 个
panel png:     240 张
```

图片目录：

```text
experiments/early10k_validation/analysis/figures/action/
experiments/early10k_validation/analysis/figures/belief/
```

## 3. 早期 peak 很快出现

从 dense timeline 看，很多 run 在 25~175 步已经到达早期或全局高点：

```text
kl001_mix_b0:       early peak step125 samp=230.3 det=210.5
kl001_mix_b2:       early peak step125 samp=232.4 det=209.8
kl001_mix_b5:       early peak step175 samp=240.0 det=247.6
kl001_mix_b8:       early peak step175 samp=246.6 det=242.4
kl005_topdown_b0:   early peak step75  samp=237.4 det=113.4
kl005_topdown_b2:   early peak step100 samp=271.8 det=61.4
kl005_topdown_b5:   early peak step100 samp=286.0 det=205.9
kl005_topdown_b8:   early peak step50  samp=303.6 det=336.8
```

这说明旧实验里看到的“早期高分”不是 250-step 稀疏 eval 的单点偶然，而是 0~250 内真实发生的快速迁移。

## 4. 但高分不像稳健 RL

核心问题在于，高分经常和以下信号同时出现：

- `combo_hit` 高：大量重复组合或撞到少数高分 slate 模板。
- `global_unique` 低：策略输出多样性不足。
- `log_std` 快速下降：samp 空间变窄，策略更容易固化。
- 后期 det/samp 回落：peak 没有形成稳定 plateau。

因此这批实验更支持“快速进入高 reward 窄甜区/策略坍缩”，而不是“稳定学会了 state-conditioned 推荐”。

## 5. mix 的主要问题

`mix` 环境本身的 latent space 很强，cloud centroid 和 cloud sample baseline 已经能拿到高 reward。因此 `mix` 早期 230~260 的 IQM peak 不能直接解释为 RL 成功。

`kl001_mix_b8` 的形态最典型：beta=8 能维持更高的早期峰值，说明 Advantage 加权确实能把 actor 更强地推向高分区域；但 final 仍明显回落，说明这个优势没有转化成长期稳定、丰富的策略。

`kl001_mix_b0` 也会先升后降，说明即使没有强 Advantage 加权，训练本身也会把 actor 从初始分布推向 GeMS 高分区域；只是 peak 较低、下降更快，说明没有 beta=8 那种“留在甜区附近”的推力。

## 6. topdown 的主要问题

`topdown` 的先升后降更极端：

```text
kl005_topdown_b8: early/global peak step50 samp=303.6 det=336.8
final:            step10000 samp=119.5 det=61.4
```

这说明 topdown 里存在非常强的短期高分 attractor，actor 很快能进入；但继续训练后，策略无法稳定保持在该区域，或者该区域与长期状态转移、多样性、GRU belief 分布不兼容。

## 7. 与旧预设是否一致

```text
H1: kl001_mix_b5 @250 是 det/samp 双高，不是纯 σ
判断：部分支持。确实有 det/samp 双高阶段，但 unique 低、combo 高，像 μ 进窄甜区。

H2: kl005_mix_b8 @250 是 samp 高、det 低的采样效应
判断：支持。该 run 出现清楚的 samp-det gap，采样撞甜区解释力强。

H3: 0~250 若渐变则支持早期快变
判断：支持早期快变，否定“eval 稀疏假象”。但快变内容更像甜区搜索/坍缩。

H4: samp 稳定段约等于 B2 且高于 B0
判断：弱支持。部分 run 高于 B0，但多数 final 没有稳定超过 cloud sample。
```

## 8. 当前认为的问题在哪里

问题不只是 eval 噪声，而是训练目标本身容易把 actor 推到“高 reward 但低多样性”的区域：

```text
Adv/AWR 早期有信号
  -> actor 快速靠近高 reward latent 甜区
  -> combo 重复上升、unique 下降、log_std 收缩
  -> Q/V/Adv 信号后期变弱或排序失真
  -> 后期 det/samp 回落或不稳定
```

所以后续算法改进应该优先围绕：

- 保持 Advantage 信号有效，尤其避免 Q 对 OOD/重复动作过估。
- 限制 AWR 过强推力，如降低 beta/clip 或 warmup。
- 防止 `log_std` 过早坍缩。
- 把 diversity-aware metric 纳入 checkpoint 选择，而不是只按 reward peak。
