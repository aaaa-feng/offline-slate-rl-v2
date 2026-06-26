# Actor 温度缩放实验

> 不改训练、不改 GeMS，只对已训好的 IQL checkpoint 做推理时的 Actor 输出缩放。
> 温度 < 1.0 让 Actor 输出更靠近动作云中心，测试 decode 敏感度是否降低。

---

## 原理

IQLAgent.act() 中：
```python
temperature = getattr(self, '_eval_temperature', 1.0)
latent_action = raw_action * self.action_scale * temperature + self.action_center
```

T=1.0 = 原样输出。T=0.5 = Actor 输出向 dataset_center 缩一半，离云心更近。

## 实验

| 变量 | 值 |
|------|-----|
| 温度 | 0.3, 0.5, 0.7, 1.0 |
| ckpt | kl005_iql 中的代表性 run 的 final ckpt |
| 评估 | 100 episodes, RecSim |

## 选择 ckpt

| ckpt | 终值 | 为什么选它 |
|------|------|----------|
| kl005_mf_init/td_b0 | 154 (跌 7%) | 不崩塌，看温度是否能进一步提分 |
| kl005_ideal_init/td_b0 | 91 (跌 19%) | 轻微崩塌 |
| kl005_ideal_init/mix_b2 | 145 (跌 18%) | mix 中唯一不崩 |
| kl005_ideal_init/mix_b8 | 65 (跌 68%) | mix 典型崩塌 run |
| beta_ablation_repreduce/mix_b8 | 73 (跌 54%) | KL=1.0 baseline 崩塌 |

共 5 个 ckpt × 4 温度 = 20 个评估。

## 启动

```bash
bash experiments/action_cloud/temperature_ablation/run.sh
```

## 产出

`logs/temperature_ablation/temperature_results.txt` — 每行: ckpt, T, mean_reward, iqm, unique_items

## 预期

如果 T=0.5 让崩塌 run（mix_b8）的 reward 从 65→90+，则说明问题在 decode 敏感度——Actor 输出离云心太远，温度一降就能纠正。如果温度对 reward 没有明显改善，则说明问题在 Actor 输出的其他方面（比如它学到了错误的方向）。
