# 在线 SAC+GeMS 几何监测复现实验计划

> **目标**：在 `offline-slate-rl` 原架构上复现 `diffuse_mix` Expert SAC 训练，同步记录与离线 IQL 几何诊断**可对齐的 scalar 指标**，并在上升阶段**密集存档 checkpoint + eval 向量**，保证事后随时能抽数据画 PCA/t-SNE。
>
> **对照关系**：本实验的行为策略 = `mix_divpen_b5` 离线数据中 60% 轨迹的来源（`diffuse_mix` Expert, bt=5, penalty=1.0）。
>
> **实现仓库**：`offline-slate-rl`（训练 + probe hook）  
> **计划与事后画图脚本**：`offline-slate-rl-v2/experiments/online_sac_geometry/`（本目录）

---

## 一、核心命题

```text
在线 SAC 训练期间（尤其 2k~40k 上升段），validation 阶段的：
  - latent action 空间（层 A）
  - GRU belief actor vs critic 空间（层 B）
  - 相对早期 baseline 的漂移（层 C）
如何演化？是否与离线 IQL 的「先升后塌」有结构相似性？
```

**不试图证明**：在线 100k 后仍单调变好（原 log 显示 74k best、后期波动）。

---

## 二、设计原则（已确认）

| 原则 | 说明 |
|------|------|
| **存够再画** | 密集步存 PL checkpoint + `eval_trajectory.npz`；事后 reload 即可 probe/画图 |
| **训练内轻、存档重** | 每 1k val 只写 scalar → SwanLab + `timeline.csv`；大 npz 仅 milestone |
| **在线参考系** | 不用离线 `mix_divpen_b5` 做灰云；用 **step=999** 与 **step=2000** 两个 baseline eval 云作层 C 锚点 |
| **双流** | `belief_actor` + `belief_critic`（在线原生，不与离线 `critic_v` 硬比） |
| **环境** | `diffuse_mix`, bt=5, penalty=1.0, mixPBM（严格复现原 Expert 训练） |
| **范围** | 仅 Expert 100k，seed=58407201 |

---

## 三、原始收敛节奏（来自 2025-11-29 log）

日志：`offline-slate-rl/experiments/logs/online/log_58407201/SAC_GeMS/replication_experiment_20251129/diffuse_mix_KL1.0_click0.5_20251129.log`

| Step | Val Reward | 阶段 |
|------|------------|------|
| 0 | 39.1 | 起步 |
| 10k | 88.2 | 缓升 |
| 15k | 232.7 | **陡升** |
| 30k | 270.5 | 高原 |
| 40k | 270.8 | 平台 |
| 74k | 297.0 | **全局 best val** |
| 100k | 270.0 | final |

→ checkpoint / 全量 npz **前密后疏**。

---

## 四、存档策略

### 4.1 三类产物

```
offline-slate-rl/experiments/online_geometry/diffuse_mix/{run_name}/
├── timeline.csv                    # 每 1k val 一行 scalar（始终）
├── baselines/
│   ├── step_00999/eval_trajectory.npz + metrics.json + *.ckpt(可选)
│   └── step_02000/eval_trajectory.npz + metrics.json + *.ckpt(可选)
├── probes/                         # 上升段 + final 全量向量
│   └── step_{XXXXX}/eval_trajectory.npz, metrics.json
├── checkpoints/                    # PL checkpoint 镜像或软链
│   └── SAC+GeMS_..._step{step}.ckpt
└── references/
    ├── baseline_00999_cloud.npz    # 层 C：从 baseline eval 汇总的质心/p90
    └── baseline_02000_cloud.npz
```

### 4.2 Milestone 表

| 类型 | Steps | PL checkpoint | eval_trajectory.npz | 说明 |
|------|-------|:-------------:|:-------------------:|------|
| **Baseline 锚点** | 999, 2000 | ✅ | ✅ | 层 C 双 baseline |
| **上升段密集** | 2k, 5k, 8k, 10k, 12k, 15k, 18k, 20k, 25k, 30k, 35k, 40k | ✅ | ✅ | 与 baseline 2k 可合并存一次 |
| **收敛后稀疏** | 50k, 74k, 100k | ✅ | ❌（仅 final 100k 可选存 npz） | 只保证能 reload 测试 |
| **每 1k val** | 1k…100k | ❌ | ❌ | 仅 `timeline.csv` + SwanLab |

> 注：2k 同时是 baseline B 与上升段点，只存一份即可。

**合计**：约 **2 baseline + 12 上升 + 3 ckpt-only + 1 best** ≈ 15 个全量 npz，15~18 个 checkpoint。

### 4.3 磁盘估算

| 项目 | 单份大小 | 数量 | 小计 |
|------|----------|------|------|
| PL checkpoint | ~3.5 MB | ~18 | ~63 MB |
| eval_trajectory.npz（5ep×100step） | ~1–3 MB | ~15 | ~30 MB |
| timeline + json | <1 MB | 1 | ~1 MB |
| **单次 run 总计** | | | **~100 MB** |

当前磁盘可用 **~326 GB**，单次实验余量充足。

### 4.4 `eval_trajectory.npz` 字段（每个 milestone）

```python
# 形状均为 (N,) 或 (N, D)，N = 5 episodes × 100 steps = 500（上限）
policy_latent_raw      # (N, 32)  SAC 输出 z
belief_actor           # (N, 20)
belief_critic          # (N, 20)
slates                 # (N, 10)  int
rewards                # (N,)
clicks                 # (N, 10)  float
episode_ids            # (N,)
timesteps              # (N,)
item_freq_pct          # (N,)     per-step decode 后 item 频率百分位
combo_hit              # (N,)     0/1
metadata               # json: step, seed, env, n_episodes, deterministic
```

事后可从 PL ckpt **重新跑 eval 生成**；训练时存的 npz 是「快照」，避免 PL 版本差异。

---

## 五、指标与 SwanLab 命名

### 5.1 `Train/SAC/*`（每 training log step，原有 + 补充）

- `train_reward`, `critic_loss`, `actor_loss`, `alpha`, `entropy`
- `policy_log_std_mean`, `policy_log_std_min`（若可取自 SAC）

### 5.2 `Evidence/Geometry/*`（每 1k val，写 timeline）

与离线 probe **subset 对齐**：

| SwanLab key | 含义 |
|-------------|------|
| `Evidence/val_reward_mean` | validation reward |
| `Evidence/probe_item_freq_pct_mean` | eval slate item 频率百分位均值 |
| `Evidence/probe_combo_hit_rate` | combo 命中率 |
| `Evidence/probe_global_unique_items` | eval 内唯一 item 数 |
| `Evidence/policy_to_centroid_mean` | 相对 **baseline_00999** 的 z 质心距离 |
| `Evidence/policy_to_centroid_mean_b2k` | 相对 **baseline_02000** |
| `Evidence/policy_in_cloud_p90` | 相对 baseline_00999 p90 半径 |
| `Evidence/policy_in_cloud_p90_b2k` | 相对 baseline_02000 |
| `Evidence/policy_dim_std_ratio` | policy z per-dim std / baseline cloud std |
| `Evidence/belief_actor_critic_l2_mean` | 同一步 actor vs critic belief L2 距离（层 B） |
| `Evidence/belief_actor_critic_cosine_mean` | 同一步 cosine 相似度 |

层 C 的 `to_centroid` / `in_cloud_p90` 在 **两个 baseline 定义下各算一套**，便于和离线叙事对照。

### 5.3 `timeline.csv` 列

与上表一致，外加 `step`, `val_reward_mean`, `val_reward_std`, `checkpoint_path`（若该步有存 ckpt）。

---

## 六、三层监测落地

| 层 | 训练时 | 事后画图 |
|----|--------|----------|
| **A 自演化** | milestone 存 `policy_latent_raw` | 跨 step PCA；step N vs M |
| **B actor vs critic** | 每步存双流 belief + `belief_actor_critic_l2` scalar | 同色 eval 轨迹上 actor/critic 分色 PCA |
| **C 锚点漂移** | baseline 999/2000 存 cloud + 每 val 算相对漂移 scalar | 与离线 `in_cloud_p90` 曲线对比 |

---

## 七、训练配置（对齐原 Expert log）

```bash
# 工作目录：offline-slate-rl
python scripts/train_online_rl.py \
  --agent=SAC --belief=GRU --ranker=GeMS --item_embedds=scratch \
  --env_name=topics --device=cuda \
  --seed=58407201 --ranker_seed=58407201 \
  --max_steps=100000 \
  --check_val_every_n_epoch=1000 \
  --val_step_length=200 \
  --random_steps=2000 \
  --name=SAC+GeMS \
  --latent_dim=32 --lambda_KL=1.0 --lambda_click=0.5 --lambda_prior=0.0 \
  --ranker_embedds=scratch --ranker_sample=False \
  --ranker_dataset=diffuse_mix \
  --click_model=mixPBM \
  --env_embedds=item_embeddings_diffuse.pt \
  --diversity_penalty=1.0 \
  --boredom_threshold=5 \
  --gamma=0.8 --beliefs actor critic \
  --swan_project=Online_SAC_Geometry_202606 \
  --swan_workspace=Cliff \
  --run_name=SAC_GeMS_diffuse_mix_geometry_seed58407201 \
  ... # + 新增 geometry 相关参数（见实现清单）
```

**Probe eval 规模**（仅 milestone / baseline）：**5 episodes × 100 steps = 500 transitions**（比原 val 的 2ep 更大，专用于几何）。

---

## 八、PL checkpoint 能否事后画图？

**可以。** 现有 `model_loader.py` 已从同格式 PL ckpt 加载 SAC+GeMS 做数据采集，包含：

- `agent` + `belief` + `ranker` 权重
- `action_center` / `action_scale`
- GRU `hidden` 状态可 `reset_hidden()` 后逐步 forward

事后流程：

```text
PL ckpt → model_loader 或 SAC.load_from_checkpoint
       → deterministic eval（5×100）
       → 写 eval_trajectory.npz（若训练时未存）
       → 本目录 plot_*.py 画 PCA/t-SNE
```

训练时**额外存 npz** 是为了省时间、避免 PL 版本变动；**checkpoint  alone 已足够复现**。

---

## 九、`offline-slate-rl` 实现清单

### 9.1 新增文件

| 路径 | 作用 |
|------|------|
| `src/diagnostics/online_geometry_probe.py` | milestone eval、npz 写入、层 C 指标、baseline cloud 构建 |
| `src/diagnostics/online_geometry_callbacks.py` | PL Callback：按 milestone 列表存 ckpt；val 后调 probe |
| `scripts/batch_runs/run_mix_expert_geometry.sh` | 一键启动 |
| `experiments/online_geometry/README.md` | 输出目录说明 |

### 9.2 修改文件

| 路径 | 改动 |
|------|------|
| `scripts/train_online_rl.py` | 接入 geometry callback；`--geometry_output_dir`；`--geometry_milestones`；`--swan_project` 默认新 project |
| `src/training/online_loops.py` | `ValEpisodeLoop`：可选返回 belief_actor/critic、latent（供 probe 复用 val 环境） |
| `src/common/online/argument_parser.py` | 新增 geometry 相关 CLI 参数 |

### 9.3 本目录（v2）事后脚本（Phase 2）

| 路径 | 作用 |
|------|------|
| `offline-slate-rl-v2/experiments/online_sac_geometry/plot_timeline.py` | 读 timeline.csv 画漂移曲线 |
| `offline-slate-rl-v2/experiments/online_sac_geometry/plot_pca_belief.py` | actor/critic belief PCA |
| `offline-slate-rl-v2/experiments/online_sac_geometry/plot_pca_action.py` | latent PCA + baseline 云 |
| `offline-slate-rl-v2/experiments/online_sac_geometry/extract_from_ckpt.py` | 从 PL ckpt 补抽 eval_trajectory.npz |

---

## 十、与离线 IQL 对比时的注意事项

| 项目 | 在线 SAC | 离线 IQL |
|------|----------|----------|
| GRU 流 | actor + **critic** | actor + **critic_v** |
| 参考云 | baseline 999 / 2000 eval | 固定 offline dataset |
| 算法指标 | entropy, Q | adv_q*, weight_entropy |
| 可比 | actor belief、latent drift 形状、item_freq 趋势 | 同左 |

画图时：**只对比 actor 流**；critic vs critic_v 不做数值对齐。

---

## 十一、执行顺序

1. ✅ 本计划文档（当前）
2. `offline-slate-rl`：实现 `online_geometry_probe` + callback + `train_online_rl` 接入
3. 跑通 smoke test：`max_steps=3000`，确认 baseline 999/2000 存盘
4. 正式跑 Expert 100k（`run_mix_expert_geometry.sh`）
5. `offline-slate-rl-v2/experiments/online_sac_geometry/`：事后画图脚本
6. 组会：在线上升段 vs 离线 IQL best→final 拼图

---

## 十二、参考路径速查

```text
原 Expert 训练 log:
  offline-slate-rl/experiments/logs/online/log_58407201/SAC_GeMS/
    replication_experiment_20251129/diffuse_mix_KL1.0_click0.5_20251129.log

数据采集用 Expert ckpt:
  offline-slate-rl/src/data_collection/.../expert/sac_gems_models/diffuse_mix/
    SAC_GeMS_diffuse_mix_expert_beta1.0_click0.5_div1.0_gamma0.8_dim32_seed58407201.ckpt

离线几何参考实现:
  offline-slate-rl-v2/src/diagnostics/training_geometry_probe.py
  offline-slate-rl-v2/experiments/action_cloud/policy_geometry_tsne/

新实验输出（计划）:
  offline-slate-rl/experiments/online_geometry/diffuse_mix/{run_name}/
```
