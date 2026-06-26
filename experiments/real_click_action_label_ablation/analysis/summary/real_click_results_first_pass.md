# Real Click Action Label Ablation 第一轮结果分析

## 一句话结论

`label_click_mode=real` 有帮助，但它主要不是把峰值整体拉高，而是让后期退化变慢、变轻。当前 3 条 run 已完整到 step 10000，`topdown_b0_rc` 只到 step 5100，所以 topdown_b0 这里只能当阶段性结果看。

## 这次对比的是谁

新实验只改一件事：训练时把数据动作 `data_action` 的 GeMS 编码从 `fake_zero clicks` 改成真实 clicks。

对照组是原来的 `early10k_validation`：

- `kl001_mix_b8_ideal_init` vs `kl001_mix_b8_ideal_init_rc`
- `kl001_mix_b0_ideal_init` vs `kl001_mix_b0_ideal_init_rc`
- `kl001_topdown_b8_ideal_init` vs `kl001_topdown_b8_ideal_init_rc`
- `kl001_topdown_b0_ideal_init` vs `kl001_topdown_b0_ideal_init_rc`

所有新 run 都写到 SwanLab project `Early10K_Validation_202606`，run 名统一追加 `_rc`。

当前完成状态：

- `kl001_mix_b8_ideal_init_rc`：已到 step 10000。
- `kl001_mix_b0_ideal_init_rc`：已到 step 10000。
- `kl001_topdown_b8_ideal_init_rc`：已到 step 10000。
- `kl001_topdown_b0_ideal_init_rc`：日志和 timeline 当前到 step 5100，仍需等它跑完或处理卡住问题。

## 先看结果

### mix_b8

`real clicks` 明显改善了后期保持能力。

- det 峰值：`265.12 -> 272.44`，略升。
- det 末值：`115.44 -> 194.36`，大幅提升。
- det 回撤：`149.68 -> 78.08`，少掉约一半。
- samp 末值：`180.88 -> 197.40`，也更高。

这说明 mix_b8 本来就比较能保持优势，`real clicks` 进一步减少了后期掉下去的幅度。

### mix_b0

`real clicks` 对 mix_b0 更像是“让 sampled policy 后期别塌太狠”，但 det 仍然会明显回落。

- det 峰值：`244.08 -> 280.64`，提升很明显。
- det 末值：`99.48 -> 130.44`，有提升但仍低。
- det 回撤：`144.60 -> 150.20`，没有改善。
- samp 末值：`151.32 -> 193.76`，明显提升。
- samp 回撤：`82.44 -> 43.20`，退化减轻。

这说明 `real clicks` 对 mix_b0 的收益主要体现在 sampled 策略和中后期稳定性，deterministic policy 仍然容易走偏。

### topdown_b8

`real clicks` 没有救回 topdown_b8 的主问题。

- det 峰值：`334.20 -> 331.92`，基本不变。
- det 末值：`105.72 -> 112.00`，只小幅提升。
- det 回撤：`228.48 -> 219.92`，只小幅减轻。
- samp 峰值：`298.52 -> 230.60`，反而降低。
- samp 末值：`58.24 -> 69.52`，略升。

这说明 topdown_b8 的问题不是简单的 action label clicks 信息不匹配。它早期还是能冲到很高，但后面仍然掉下去。

### topdown_b0

这是最有意思的一条，但注意：`rc` 目前只到 step 5100，不是完整 step 10000。阶段性看，`real clicks` 明显改善 det 的当前表现，但 sampled policy 的峰值变差。

- det 峰值：`330.12 -> 315.80`，略降。
- det 当前/末值：`53.04 @ old step10000 -> 143.40 @ rc step5100`，大幅提升，但口径不是完整终点对比。
- det 当前回撤：`277.08 -> 172.40`，阶段性明显减轻。
- samp 峰值：`228.52 -> 106.52`，明显变差。
- samp 当前/末值：`80.64 @ old step10000 -> 79.00 @ rc step5100`，基本没变。

这说明 real clicks 至少能让 topdown_b0 的 deterministic actor 在中期不至于完全崩掉，但要等它完整跑到 10000 才能下最终判断。

## 关键指标怎么解释

### 1. Reward：看最终有没有真的更好

这里主要看 `det_iqm_reward` 和 `samp_iqm_reward`。

- `det_iqm_reward`：actor 直接出均值动作，比较像“最终学到的固定策略能不能打”。
- `samp_iqm_reward`：actor 按分布采样动作，比较像“策略周围一圈动作整体质量如何”。
- `drop = best - final`：峰值到最后掉了多少，是判断“先升后降”严重程度的核心指标。

从 reward 看，`real clicks` 对 mix 的后期保持有明显帮助；对 topdown 只能缓解，不能根治。

### 2. Advantage：看 actor 还有没有学习信号

关键看两个指标：

- `train_adv_q90`：Advantage 的高分位，越高说明“好动作比普通动作高多少”还有区分度。
- `train_adv_near_zero_rate`：Advantage 接近 0 的比例，越高说明很多动作在 critic/value 眼里差不多，actor 学不到方向。

最终或当前最后 step 的变化：

- mix_b8：`adv_q90 0.1415 -> 0.1829`，`near_zero 0.6276 -> 0.5213`
- mix_b0：`adv_q90 0.1415 -> 0.1829`，`near_zero 0.6276 -> 0.5213`
- topdown_b8：`adv_q90 0.1188 -> 0.1233`，`near_zero 0.7685 -> 0.7633`
- topdown_b0：`adv_q90 0.1188 @ old step10000 -> 0.1579 @ rc step5100`，`near_zero 0.7685 -> 0.4953`

解释成白话：real clicks 在 mix 上确实让“好动作和普通动作的差距”更明显；topdown_b0 的阶段性 near-zero 也从 `76.85%` 降到 `49.53%`，但这条还不能当最终结果。topdown_b8 几乎没动，说明它的坍缩主要不是 clicks label 造成的。

### 3. log_std：看 actor 是不是过早变窄

关键看：

- `log_std_mean`：越低说明策略分布越窄。
- `log_std_floor_hit_rate`：有多少维度被压到下限，越高越像“探索方差死了”。

最终或当前最后 step 的 `floor_hit_rate`：

- mix_b8：`0.5530 -> 0.1990`
- mix_b0：`0.5642 -> 0.2105`
- topdown_b8：`0.4797 -> 0.1912`
- topdown_b0：`0.4873 @ old step10000 -> 0.1209 @ rc step5100`

这是最稳定的正向信号：real clicks 让 actor 不那么快把方差压死。换句话说，action label 更合理后，actor 拟合数据动作时没那么别扭，不需要把分布压得那么窄。

### 4. combo_hit：看是不是又开始重复推荐

`combo_hit` 可以粗略理解成策略是不是老在打数据集中常见的 slate 组合。太低可能是跑飞，太高可能是过度复读。

最终或当前最后 step 的 sampled combo_hit：

- mix_b8：`0.3676 -> 0.5378`
- mix_b0：`0.1672 -> 0.4978`
- topdown_b8：`0.0782 -> 0.1968`
- topdown_b0：`0.1346 @ old step10000 -> 0.1948 @ rc step5100`

这个方向和 reward 改善基本一致：real clicks 让策略更贴近数据高密度区域，少一些完全跑飞。但 mix_b8 后期 combo_hit 也变高了，所以后续要警惕它是不是用“更像数据/更重复”换来了稳定。

## 最重要的机制判断

### 判断 1：`fake_zero` 确实是问题之一

如果 `fake_zero` 完全不是问题，那改成 `real` 后 Advantage、log_std、后期 reward 不应该系统性改善。

但现在看到：

- `train_adv_q90` 在 3 条完整 run 上变高，但 topdown_b8 只小幅变高；`topdown_b0_rc` 的阶段性结果也变高。
- `train_adv_near_zero_rate` 在 3 条完整 run 上变低，但 topdown_b8 只小幅变低；`topdown_b0_rc` 的阶段性结果降得尤其明显。
- `log_std_floor_hit_rate` 在 3 条完整 run 上都大幅下降，`topdown_b0_rc` 阶段性也下降。
- mix 的 final reward 明显更稳。

所以可以说：真实 clicks 让 action latent label 更接近数据里真实发生的动作条件，训练信号更顺了。

### 判断 2：它不是根因的全部

如果 clicks label 是唯一根因，topdown_b8 应该明显改善。但 topdown_b8 仍然是典型“早期很高，后面掉光”：

- det 峰值仍在 step 50 左右。
- final det 只从 `105.72` 到 `112.00`。
- final near-zero 仍然高达 `0.7633`。

所以 topdown_b8 的主问题更像是环境/动作几何 + IQL 策略提取共同导致的：早期 actor 能撞进高奖励区域，但 critic/value 后期仍然不能稳定提供局部排序，actor 最后还是掉出去。

### 判断 3：real clicks 更像“延缓/减轻坍缩”，不是“让策略持续变强”

很多 run 的峰值没有稳定提升，甚至 topdown 的 sampled 峰值下降了。更稳定的改善是：

- mix 的 final reward 更高；
- mix_b8 和 topdown_b8 的 best-final drop 更小，mix_b0 的 sampled drop 更小；
- log_std floor hit 更低；
- Adv near-zero 更低。

这说明 real clicks 在改善训练健康度，但还没解决“Q 对 actor 生成动作的排序是否可靠”这个更核心的问题。

## 当前结论

这次实验支持原假设的一半：

> action label 使用 fake-zero clicks 会破坏训练信号；改成 real clicks 后，Advantage 信号和 actor 方差都更健康。

但它也否定了一个更强版本的假设：

> 只要换成 real clicks，就能解决 topdown 先升后降。

实际结果是：mix 明显受益，topdown 只部分受益，topdown_b8 几乎没被救回来。

## 下一步建议

下一步应该补一个 real-click 版本的 Q/V/action-ranking probe，直接回答：

1. `elite_data_action` 的 Q 是否比普通 `data_action` 更高；
2. `policy_mu` 是否仍然被 Q 高估；
3. `random_latent` 是否还会被 Q 错误打高；
4. real-click 是否只是让 actor 更稳，还是也真的改善了 Q 的动作排序。

如果 probe 显示 Q 排序没有明显改善，那下一步就不应该继续纠结 label，而应该转向 Q 约束/排序正则，例如 RankQ、CQL/Cal-QL 或 policy action vs data action 的 pairwise ranking loss。
