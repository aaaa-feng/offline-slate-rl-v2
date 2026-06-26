# Diffusion Policy 当前实现说明

本文档记录 `offline-slate-rl-v2` 中 Diffusion Policy / Flow Matching 方案的当前代码实现状态。它不是早期设计草案，而是面向现在项目代码的使用说明、模块说明和实验检查清单。

当前已完成的是 Phase 1：`flow_bc`，即用 Flow Matching 做 Behavior Cloning。Phase 2 的 `flow_iql` / Q-guidance 入口已预留，但当前仍复用 `FlowBCAgent`，还没有接入 Critic、Value 和 Q 梯度引导。

---

## 1. 当前目标

原 IQL 管线的动作生成链路是：

```text
obs -> GRU belief -> TanhGaussian actor -> GeMS latent z -> GeMS decoder -> discrete slate
```

现在 Phase 1 改成：

```text
obs -> GRU belief -> Flow Matching sampler -> slate item embeddings -> kNN -> discrete slate
```

核心变化是：不再让策略输出 GeMS latent，也不再经过 GeMS decoder 的 argmax；策略直接生成 10 个位置的 item embedding，然后用 kNN 映射回 item id。

Phase 1 只做模仿学习，不用 reward、Q、V、advantage。

---

## 2. 代码结构

当前新增/修改的主要文件：

```text
src/agents/diffusion_slate/
  flow_bc_agent.py       # Phase 1 agent: GRU + VelocityNet + FlowPolicy
  flow_policy.py         # Euler sampling + kNN / dedup kNN
  velocity_net.py        # Flow Matching velocity field network
  embedding_loader.py    # 从 .pt 或 .ckpt 加载 item embedding table

scripts/
  train_agent.py              # 接入 flow_bc / flow_iql
  export_gems_embeddings.py   # 从 GeMS checkpoint 导出 item embeddings

config.py
  ExperimentConfig 中新增 flow 参数和 gems_embedding_path
```

`experiments/diffusion_policy/DESIGN.md` 和 `IMPLEMENTATION.md` 仍保留作为设计背景和实施计划；本文档以当前代码为准。

---

## 3. Embedding 管理

建议把 GeMS 训练出来的 item embedding 统一导出到：

```text
data/embeddings/gems/
```

推荐命名：

```text
data/embeddings/gems/gems_mix_divpen_b5_ideal_init_latent32_kl0.05_click1.0_seed58407201.pt
data/embeddings/gems/gems_topdown_divpen_b5_mf_fixed_latent32_kl0.05_click1.0_seed58407201.pt
```

导出脚本会同时生成同名 `.json` 元数据文件，记录 `source_ckpt`、`state_dict_key`、`shape`、`mean/std/min/max`。这样之后跑 `flow_bc`、`flow_iql`、action cloud probe 或其它分析脚本时，都可以直接复用同一个 `.pt` embedding，不需要每次重新解析 GeMS checkpoint。

### 3.1 导出命令

单个 checkpoint：

```bash
python scripts/export_gems_embeddings.py \
  --ckpt checkpoints/gems/GeMS_mix_divpen_b5_ideal_init_latent32_beta0.05_click1.0_seed58407201.ckpt \
  --out data/embeddings/gems/gems_mix_divpen_b5_ideal_init_latent32_kl0.05_click1.0_seed58407201.pt
```

批量导出：

```bash
python scripts/export_gems_embeddings.py \
  --ckpt-dir checkpoints/gems \
  --out-dir data/embeddings/gems
```

如果目标文件已存在，默认会报错，防止误覆盖。需要覆盖时加 `--overwrite`。

### 3.2 Flow agent 的 embedding 加载优先级

`scripts/train_agent.py` 中，`flow_bc` / `flow_iql` 的 embedding 加载优先级是：

```text
--gems_embedding_path > --gems_checkpoint > cfg.gems_checkpoint_path
```

推荐优先用导出的 `.pt`：

```bash
--gems_embedding_path data/embeddings/gems/xxx.pt
```

也可以直接从 `.ckpt` 加载：

```bash
--gems_checkpoint checkpoints/gems/xxx.ckpt
```

如果两者都不传，则用 `config.py` 的 `cfg.gems_checkpoint_path` 根据 `env_name`、`dataset_quality`、`gems_embedding_mode`、`lambda_KL`、`lambda_click`、`seed` 自动推导 checkpoint 路径。

### 3.3 loader 支持的格式

`src/agents/diffusion_slate/embedding_loader.py` 的 `load_embedding_table(path, device)` 现在支持：

```text
.pt: 直接保存的 2D tensor [num_items, item_dim]
.pt: dict，包含 weight
.ckpt: Lightning checkpoint，包含 state_dict
.ckpt/.pt: 直接是 state_dict
```

支持识别的 embedding key 包括：

```text
item_embeddings.weight
ranker.item_embeddings.weight
item_embeddings.embedd.weight
ranker.item_embeddings.embedd.weight
```

最终返回 frozen `nn.Embedding`，不会训练 item embedding 本身。

---

## 4. FlowBCAgent

文件：

```text
src/agents/diffusion_slate/flow_bc_agent.py
```

`FlowBCAgent` 接收：

```python
FlowBCAgent(config, embedding_table, device)
```

内部组件：

```text
embedding_table        frozen item embedding table
GRUBelief              复用现有 GRU belief encoder
VelocityNet            预测 Flow Matching velocity
FlowPolicy             采样 + kNN 离散化
Adam optimizer         优化 VelocityNet + actor GRU
```

当前 GRU belief 使用 `beliefs=["actor", "critic_v"]`。Phase 1 训练只用 `states["actor"]`，`critic_v` 暂时不参与 loss，也不进入 optimizer。保留它主要是为了和后续 Phase 2 的 critic/value 路径靠近。

GRU 输入由每个位置的 item embedding 拼接 click 得到：

```text
[item_embedding, click] * rec_size
```

默认配置下：

```text
rec_size = 10
item_embedd_dim = 20
input_dim = 10 * (20 + 1) = 210
belief_hidden_dim = 20
```

### 4.1 训练 batch 语义

`TrajectoryReplayBuffer.sample(batch_size)` 返回的是若干 episode，而不是若干 transition。

例如：

```text
batch_size = 256 episodes
episode length ~= 100
flatten 后 transition 数 ~= 25,600
```

`FlowBCAgent.train()` 会先调用：

```python
states, _ = self.belief.forward_batch(batch)
s = states["actor"]
```

然后把 batch 中的 slate ids 查 embedding：

```python
slate_ids = torch.cat(batch.obs["slate"], dim=0)
x0 = self.embedding_table(slate_ids).flatten(1)
```

为了避免 OOM，当前每次从 flatten 后 transition 中随机采样最多 4096 个：

```python
n = min(4096, x0.shape[0])
idx = torch.randperm(x0.shape[0], device=self.device)[:n]
x0, s = x0[idx], s[idx]
```

### 4.2 Flow Matching loss

当前 Phase 1 在 raw embedding space 训练，不做 embedding 标准化：

```text
emb_mean = 0
emb_std = 1
```

训练方向是 `t=0` 噪声，`t=1` 数据：

```python
t = torch.rand(n, 1)
noise = torch.randn_like(x0)
xt = (1 - t) * noise + t * x0
target_vel = x0 - noise
pred_vel = velocity_net(s, xt, t)
loss = mse(pred_vel, target_vel)
```

optimizer 只更新：

```text
VelocityNet parameters
belief.gru["actor"] parameters
```

训练返回 metrics：

```python
{"flow_loss": loss.item()}
```

因此 `train_agent.py` 对 `flow_bc` / `flow_iql` 使用单独的简化日志分支，不调用 IQL 的 25 类指标日志。

---

## 5. VelocityNet

文件：

```text
src/agents/diffusion_slate/velocity_net.py
```

输入：

```text
state: [B, belief_hidden_dim]
x_t:   [B, action_dim]
t:     [B, 1]
```

默认：

```text
belief_hidden_dim = 20
rec_size = 10
item_dim = 20
action_dim = 200
hidden_dim = 512
n_blocks = 3
```

网络结构：

```text
state -> state_proj -> [B, hidden_dim]
x_t   -> x_proj     -> [B, hidden_dim]
t     -> SinusoidalPosEmb(128) -> Linear -> [B, hidden_dim]

h = x_proj(x_t) + t_proj(t)
for block in residual_blocks:
    h = ResidualBlock(h, state_proj(state))
h = h + x_proj(x_t)
out(h) -> velocity [B, action_dim]
```

FiLM 的条件维度现在是 `hidden_dim`，和 `state_proj(state)` 对齐。

---

## 6. FlowPolicy

文件：

```text
src/agents/diffusion_slate/flow_policy.py
```

`FlowPolicy` 负责从 belief state 采样离散 slate：

```text
state -> noise -> Euler integration -> generated embedding -> kNN -> item ids
```

当前已去掉运行逻辑里的硬编码维度：

```text
num_items  = embedding_table.weight.shape[0]
item_dim   = embedding_table.weight.shape[1]
action_dim = velocity_net.out.out_features
rec_size   = action_dim // item_dim
```

如果 `action_dim` 不能被 `item_dim` 整除，会直接报错。

确定性评估时使用固定 `_eval_noise`：

```python
x = self._eval_noise.to(device).expand(B, -1)
```

非确定性采样时：

```python
x = torch.randn(B, action_dim, device=device)
```

Euler integration：

```python
dt = 1.0 / flow_steps
for i in range(flow_steps):
    t = torch.full((B, 1), i * dt, device=device)
    v = velocity_net(state, x, t)
    x = x + v * dt
```

生成结果 reshape 为 `[B, rec_size, item_dim]` 后做 kNN。默认使用 dedup kNN：逐位置贪心选最近 item，并把已经选过的 item 距离置为 `inf`，避免同一个 slate 内重复 item。

开关：

```bash
--flow_dedup_knn 1   # 默认，去重
--flow_dedup_knn 0   # 不去重
```

---

## 7. train_agent.py 集成

文件：

```text
scripts/train_agent.py
```

`--algo` 支持：

```text
iql
bc
flow_bc
flow_iql
```

当前：

```text
flow_bc  -> FlowBCAgent
flow_iql -> FlowBCAgent + TODO
```

也就是说 `flow_iql` 目前只是预留入口，还不是完整 IQL + Q-guidance。

对 `iql` / `bc`：加载完整 GeMS ranker，并用 `ranker.run_inference()` 计算 latent action normalization。

对 `flow_bc` / `flow_iql`：不加载 GeMS ranker，不计算 latent action normalization，只加载 item embedding table，然后创建 `FlowBCAgent`。

新增 CLI 参数：

```bash
--flow_steps          Euler integration 步数，默认 10
--guidance_scale      Phase 2 预留，当前 Phase 1 不使用
--flow_dedup_knn      是否启用 slate 内去重 kNN，默认 1
--gems_embedding_path 导出的 embedding .pt 路径
```

`--gems_checkpoint` 仍可用，用于直接从 GeMS checkpoint 抽 embedding。

`flow_bc` / `flow_iql` 的训练日志目前只记录 `flow_loss`。SwanLab 也只上传 `metrics` 中数值型字段。不会调用 IQL 专用的 `actor_loss`、`critic_loss`、`advantage` 等指标。

eval 仍复用 `evaluate_policy()`，会记录 reward、slate diversity、combo hit、global unique items 等策略层指标。

---

## 8. config.py 相关字段

`ExperimentConfig` 中当前相关字段：

```python
algo: str = "iql"

flow_steps: int = 10
guidance_scale: float = 0.0
flow_dedup_knn: int = 1

gems_embedding_path: str = ""
gems_embedding_mode: str = "scratch"
lambda_KL: float = 1.0
lambda_click: float = 1.0
latent_dim: int = 32
```

`gems_embedding_path` 现在既可用于 GeMS pretrained embedding，也可用于 flow agent 直接加载导出的 item embedding `.pt`。

---

## 9. 常用命令

### 9.1 导出 GeMS embedding

```bash
python scripts/export_gems_embeddings.py \
  --ckpt checkpoints/gems/GeMS_mix_divpen_b5_ideal_init_latent32_beta0.05_click1.0_seed58407201.ckpt \
  --out data/embeddings/gems/gems_mix_divpen_b5_ideal_init_latent32_kl0.05_click1.0_seed58407201.pt
```

### 9.2 批量导出

```bash
python scripts/export_gems_embeddings.py \
  --ckpt-dir checkpoints/gems \
  --out-dir data/embeddings/gems
```

### 9.3 Flow BC 短跑 smoke experiment

```bash
python scripts/train_agent.py \
  --algo flow_bc \
  --env_name mix_divpen \
  --dataset_quality b5 \
  --gems_embedding_path data/embeddings/gems/gems_mix_divpen_b5_ideal_init_latent32_kl0.05_click1.0_seed58407201.pt \
  --experiment_name diffusion_policy/flow_bc_smoke \
  --max_timesteps 1000 \
  --log_freq 100 \
  --eval_freq 500 \
  --eval_episodes 10 \
  --final_eval_episodes 20 \
  --flow_steps 10 \
  --flow_dedup_knn 1
```

### 9.4 不提前导出，直接从 checkpoint 加载

```bash
python scripts/train_agent.py \
  --algo flow_bc \
  --env_name mix_divpen \
  --dataset_quality b5 \
  --gems_checkpoint checkpoints/gems/GeMS_mix_divpen_b5_ideal_init_latent32_beta0.05_click1.0_seed58407201.ckpt \
  --experiment_name diffusion_policy/flow_bc_from_ckpt \
  --max_timesteps 1000 \
  --log_freq 100 \
  --eval_freq 500 \
  --flow_steps 10
```

### 9.5 使用 config 自动推导 checkpoint

```bash
python scripts/train_agent.py \
  --algo flow_bc \
  --env_name mix_divpen \
  --dataset_quality b5 \
  --gems_embedding_mode ideal_init \
  --lambda_KL 0.05 \
  --seed 58407201 \
  --experiment_name diffusion_policy/flow_bc_auto_ckpt
```

这种方式依赖 `cfg.gems_checkpoint_path` 的命名规则完全匹配实际 checkpoint 文件名。为了减少误加载，推荐实验主路径使用 `--gems_embedding_path`。

---

## 10. 当前已验证项

已做过的轻量验证：

```text
python3 -m py_compile
ReadLints 无 linter error
FlowPolicy.sample() smoke test
FlowBCAgent.train() smoke test
export_gems_embeddings.py 伪 checkpoint 导出 smoke test
load_embedding_table() 加载导出 .pt 并接入 FlowBCAgent smoke test
```

验证覆盖了：

```text
VelocityNet forward shape
FlowPolicy 动态维度推导
dedup kNN 输出 shape
GRUBelief forward_batch 与 FlowBCAgent.train 接口
导出 .pt / 读取 .pt 的基本路径
```

尚未覆盖：

```text
真实 GeMS checkpoint 批量导出
真实数据集上的完整 1000+ step 训练
SwanLab 云端日志写入
长时间 eval 的 reward/diversity 曲线
```

---

## 11. 当前风险和注意事项

### 11.1 Phase 1 没有 reward 优化

`flow_bc` 是纯 BC。它只能回答：

```text
直接生成 item embedding + kNN 是否能稳定复现数据策略？
是否避免原 GeMS latent/decoder 路径中的动作空间脆性？
```

它不能回答：

```text
Q-guidance 是否能提升 reward？
IQL advantage 是否会继续死亡？
Flow policy 是否能超越行为数据？
```

这些属于 Phase 2。

### 11.2 raw embedding space 训练

当前 Phase 1 不做 embedding 标准化。好处是训练/推理尺度一致，逻辑简单。风险是如果某些 GeMS embedding 的 std 很小或很大，Flow Matching 的噪声尺度可能不匹配。

如果观察到 `flow_loss` 难以下降，或生成 embedding 的 kNN 结果非常集中，可以考虑 Phase 1.1 加回标准化：

```text
训练: x0_norm = (x0 - emb_mean) / emb_std
推理: x = x * emb_std + emb_mean
```

### 11.3 eval noise 固定

当前 `FlowBCAgent` 初始化时设置固定 eval noise：

```python
self.policy.set_eval_noise(torch.randn(1, action_dim))
```

这保证同一个 agent 在 deterministic eval 中输出稳定，但 checkpoint 目前没有保存 `_eval_noise`。如果后续要严格复现实验，可把 `_eval_noise` 加入 `state_dict()`。

### 11.4 `flow_iql` 只是入口占位

`flow_iql` 当前还没有真正接入：

```text
Critic
Value
target critic
expectile value loss
AWR / Q-guidance
act_with_q_guidance
```

不要把当前 `flow_iql` 结果解释为 IQL + diffusion policy。

### 11.5 GeMS embedding 来源要严格记录

不同 GeMS checkpoint 的 item embedding 可能来自不同训练方式：

```text
scratch
ideal_init
mf_fixed
不同 lambda_KL
不同 seed
不同 env_name / dataset_quality
```

因此建议每次 `flow_bc` 运行都显式传：

```bash
--gems_embedding_path data/embeddings/gems/xxx.pt
```

并保留导出的 `.json` 元数据，避免后续实验对不上 embedding 来源。

---

## 12. 下一步建议

建议 Phase 1 接下来按以下顺序推进：

```text
1. 先批量导出 KL=0.05 的 GeMS embeddings
2. 跑 flow_bc smoke experiments：mix/topdown 各 1000 step
3. 检查 flow_loss 是否下降
4. 检查 eval reward、global_unique_items、slate_unique_mean、combo_top1_repeat_share
5. 如果输出过于集中，先对比 flow_dedup_knn=0/1，再考虑 embedding 标准化
6. Phase 1 稳定后再进入 flow_iql / Q-guidance
```

最小判断标准：

```text
程序能完整训练和 eval
flow_loss 有下降趋势
slate 内重复 item 少
global unique 不塌缩到极小
eval reward 至少接近行为数据/BC baseline 的合理区间
```
