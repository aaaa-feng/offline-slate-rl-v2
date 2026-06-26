# 在线 SAC 几何监测 — 事后画图

训练产物在 `offline-slate-rl/experiments/online_geometry/`；本目录只保留 milestone panel 图及脚本。

## 输出图

| 文件 | 内容 |
|------|------|
| `outputs/trajectory_panels/trajectory_panels_belief_actor_reward.png` | GRU belief actor，6 panel，按 reward 着色 |
| `outputs/trajectory_panels/trajectory_panels_belief_critic_reward.png` | GRU belief critic，6 panel，按 reward 着色 |
| `outputs/trajectory_panels/trajectory_panels_action_reward.png` | policy latent (action z)，6 panel，按 reward 着色 |
| `outputs/trajectory_panels/trajectory_panels_action_global_unique.png` | policy latent，6 panel，按 eval 累积 unique item 着色 |
| `outputs/trajectory_panels/trajectory_panels_belief_actor_global_unique.png` | GRU belief actor，同上 |
| `outputs/trajectory_panels/trajectory_panels_belief_critic_global_unique.png` | GRU belief critic，同上 |

灰底 = 在线 step-999 eval 云；彩色 = 各 milestone eval。global_unique 图：颜色 = 该 eval 内截至当前步已出现过的不同 item 数（panel 标题为整段 eval 终值，/1000）。

## 重画

默认自动扫描 run 目录下所有 milestone npz（当前至 step 40k 共 13 个 panel）。

```bash
PY=/data/liyuefeng/miniconda3/envs/gems/bin/python
cd offline-slate-rl-v2/experiments/online_sac_geometry

$PY plot_gru_panels_reward.py      # belief actor + critic (reward)
$PY plot_action_panels_reward.py   # action z (reward)
$PY plot_panels_global_unique.py   # action + GRU (global_unique)
```

计划文档：[PLAN.md](./PLAN.md)
