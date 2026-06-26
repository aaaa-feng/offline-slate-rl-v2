# 实施计划：在现有代码框架下实现 Flow Matching

> **修正记录**:
> - Flow Matching 方向: t=0 噪声, t=1 数据, 速度场从噪声→数据
> - batch_size 实际 ≠ 256: GRU forward_batch 会把 256 episode flatten 成 ~25600 transitions
> - ranker 依赖: 需要拆加载路径, flow_bc/flow_iql 不走 GeMS ranker

---

## 一、三个关键修正

### 1.1 Flow Matching 方向（已修正）

```python
# 训练: t=0 噪声, t=1 数据. 速度场从噪声指向数据.
t ~ Uniform(0, 1)
noise = randn_like(x0)
x_t = (1 - t) * noise + t * x0              # 直线插值: noise→data
target_vel = x0 - noise                      # 真实速度场: noise→data
pred = velocity_net(state, x_t, t)           # 预测速度场
loss = MSE(pred, target_vel)

# 推理: 从 t=0 (噪声) 出发, 沿速度场走到 t=1 (数据)
x = randn(200)                                # 噪声 (t=0)
for i in range(flow_steps):
    t = i / flow_steps                        # [0, 1)
    v = velocity_net(state, x, t)             # 预测速度场
    x = x + v / flow_steps                    # Euler步
output = x  # clean slate_embedding
```

### 1.2 batch_size ≠ 256

现有 `buffer.sample(256)` 返回 256 个 episode。`GRUBelief.forward_batch()` 把他们 flatten 成 ~256×100 ≈ 25600 个 transition。

**Phase 1 应对**: 每次训练从 flatten 后的 transitions 中随机 subsample 4096 个。

```python
def train(self, batch):
    states, _ = self.belief.forward_batch(batch)
    s = states["actor"]                              # [~25k, 20]

    slate_ids = torch.cat(batch.obs["slate"], dim=0) # [~25k, 10]
    x0 = self.embedding_table(slate_ids).flatten(1)  # [~25k, 200]

    # Sub-sample 避免 OOM
    n = min(4096, x0.shape[0])
    idx = torch.randperm(x0.shape[0], device=s.device)[:n]
    x0, s = x0[idx], s[idx]

    # Flow Matching loss
    t = torch.rand(n, 1, device=s.device)
    noise = torch.randn_like(x0)
    xt = (1 - t) * noise + t * x0
    target_vel = x0 - noise
    pred_vel = self.velocity_net(s, xt, t)
    loss = F.mse_loss(pred_vel, target_vel)
    ...
```

### 1.3 train_agent.py 拆 GeMS 路径

当前入口强依赖 ranker 加载和 `ranker.run_inference()` 计算 action norm。Flow BC 不需要这些。

```python
# 改后的 train_agent.py
if cfg.algo in ("iql", "bc"):
    # 现有路径: 加载 GeMS ranker + action norm
    ranker, action_dim, item_embeddings = load_gems_ranker(...)
    ...compute action_center/action_scale...

elif cfg.algo in ("flow_bc", "flow_iql"):
    # 新路径: 只加载 item embedding 表, 跳过 GeMS ranker
    # 优先使用 --gems_checkpoint 直接指定 (见 Section 3.3)
    from src.agents.diffusion_slate.embedding_loader import load_embedding_table
    ckpt_path = args.gems_checkpoint or cfg.gems_checkpoint_path
    embedding_table = load_embedding_table(str(ckpt_path), device)
    action_dim = 200  # 10 positions × 20 dim embeddings
    ranker_params = {'embedding_table': embedding_table}
```

### 1.4 kNN 不是完全平滑

top-1 kNN 仍是离散操作（Voronoi 分区边界）。Phase 1 训练不需要穿过 kNN（没问题）。Phase 2 如果加 Q guidance：Q 梯度把 embedding 推到两个 item 边界附近时，离散结果会不稳定。

**应对**: 监测以下指标，如果变差说明 Q guidance scale 太大：

```python
knn_dist_1st, knn_dist_2nd = top2_smallest(dists)        # 最近和次近距离
knn_margin = knn_dist_2nd - knn_dist_1st                 # 边界余量, 越小越危险
duplicate_rate = items中重复item的比例
slate_unique = 每个slate中的unique item数
```

---

## 二、Phase 1: Flow Matching BC（最小可行）

### 2.1 目标

验证 "不经过 GeMS latent + decoder argmax 时，strategy 是否不再坍塌"。

只用模仿学习——给定 belief state，用 Flow Matching 生成 item embedding，kNN 映射为 discrete slate。

### 2.2 CLI / Config 修改

**`scripts/train_agent.py`** — 修改已有 `--algo` 的 choices，不新增 parser：

```python
# 找到现有 --algo 定义 (line ~63), 修改 choices:
parser.add_argument("--algo", type=str, default=None,
                    choices=["iql", "bc", "flow_bc", "flow_iql"])

# 在 create_parser() 中新增:
parser.add_argument("--flow_steps", type=int, default=10)
parser.add_argument("--guidance_scale", type=float, default=0.0)
parser.add_argument("--flow_dedup_knn", type=int, default=1)
```

**`config.py`** 的 `ExperimentConfig` 里加（`build_config()` 需要拷贝这些字段）：

```python
flow_steps: int = 10
guidance_scale: float = 0.0
flow_dedup_knn: int = 1
```

### 2.3 train_agent.py 拆 GeMS 路径

当前入口强依赖 ranker 加载和 `ranker.run_inference()` 计算 action norm。Flow BC 不需要这些。

```python
# 改后的 train_agent.py main()
if cfg.algo in ("iql", "bc"):
    # === 现有路径: 加载 GeMS ranker ===
    ranker, action_dim, item_embeddings = load_gems_ranker(
        env_name=cfg.env_name, dataset_quality=cfg.dataset_quality,
        gems_embedding_mode=cfg.gems_embedding_mode, device=device, ...
    )
    # ... 用 ranker.run_inference() 算 action center/scale ...
    ranker_params = {'action_center': ..., 'action_scale': ..., 'item_embeddings': ...}

elif cfg.algo in ("flow_bc", "flow_iql"):
    # === 新路径: 只加载 item embedding, 跳过 GeMS ranker ===
    # 方法 A: 通过 ckpt 路径加载 (推荐, 避免误加载 KL=1.0)
    #   --gems_checkpoint checkpoints/gems/GeMS_mix_divpen_b5_pretrained_latent32_beta0.05_click1.0_seed58407201_ideal_init.ckpt
    # 方法 B: 从 GeMS ckpt 提取 (需要完整命名参数)
    #   --env_name mix_divpen --dataset_quality b5 --gems_embedding_mode ideal_init --gems_kl 0.05 --seed 58407201

    from src.agents.diffusion_slate.embedding_loader import load_embedding_table
    embedding_table = load_embedding_table(
        ckpt_path=args.gems_checkpoint,         # 直接指定 ckpt 路径
        # 或:
        env_name=cfg.env_name,                   # 通过参数推断
        dataset_quality=cfg.dataset_quality,
        gems_mode=cfg.gems_embedding_mode,        # ideal_init / mf_init
        kl=cfg.lambda_KL,                         # 0.05 ← 关键, 别误用 1.0
        seed=cfg.seed,
    )
    action_dim = 200  # 10 positions × 20 dim embeddings
    ranker_params = {'embedding_table': embedding_table}
    # 不需要 ranker, 不需要 action_center/action_scale
```

### 2.4 预期结果

| 指标 | 基线 (Gaussian BC) | 本 Phase 预期 |
|------|:---:|:---:|
| eval reward | ~100 (mix_b0) | ≥ 基线 |
| global_unique @100k | 23 | > 50 (不崩) |
| combo_hit @100k | ~0% | > 20% (不归零) |
| KNN 重构精度 | — | > 50% |
| duplicate rate | — | < 10% |

---

## 三、Phase 2: IQL + Flow Matching

### 3.1 推荐顺序

不要一次做完全部 joint training。按这个顺序来：

**Step 2.1**: 训 Flow BC (Phase 1)，保存模型。

**Step 2.2**: 冻结 Flow BC，单独训 200D Critic/Value（IQL 方式）。

```python
# 训练时: Critic 评估 Q(s, true_action_200d)
# true_action 直接从数据生成: embedding_table(slate_ids).flatten(1)
# 不需要从 policy 采样动作给 Critic
```

**Step 2.3**: 只在 eval 时扫 guidance_scale ∈ {0, 0.01, 0.03, 0.05, 0.1}。

```python
def act_with_q_guidance(self, obs, guidance_scale=0.05):
    # GRU 编码 + 加 batch 维度 ([20] → [1, 20])
    s = self.belief.forward(obs, done=False)["actor"]    # [20]
    s = s.unsqueeze(0)                                    # [1, 20]
    device = s.device

    # 噪声起点 (t=0)
    x = torch.randn(1, 200, device=device)

    # 如果 deterministic, 用固定噪声 [1, 200]
    if hasattr(self, '_eval_noise') and self._eval_noise is not None:
        x = self._eval_noise.to(device)      # 已经是 [1, 200]

    dt = 1.0 / self.flow_steps
    for i in range(self.flow_steps):
        t = torch.full((1, 1), i * dt, device=device)

        # 无条件速度场
        v_flow = self.velocity_net(s, x, t)

        # Q 引导 (仅当 guidance_scale > 0)
        if guidance_scale > 0:
            x_grad = x.detach().requires_grad_(True)
            q1, q2 = self.critic_1(s, x_grad)           # Critic 返回 tuple (q1, q2)
            q = q1                                        # 只用 q1 做引导
            q_grad = torch.autograd.grad(q.sum(), x_grad)[0]
            q_grad = torch.clamp(q_grad, -1.0, 1.0)
            x = (x + (v_flow + guidance_scale * q_grad) * dt).detach()  # detach 防止 eval 累积计算图
        else:
            x = x + v_flow * dt

    # kNN 离散化
    emb = x.reshape(10, 20)
    items = self.policy._dedup_knn(emb.unsqueeze(0), device)

    return items.squeeze(0).detach().cpu().numpy()
```

**Step 2.4**: 如果 Q guidance 在 eval 上有提升（reward > BC 且 unique 不崩），再考虑 Q-weighted flow loss 或 joint fine-tune。

### 3.2 Phase 2 风险控制

| 风险 | 控制 |
|------|------|
| Q guidance 噪声大 | guidance_scale ≤ 0.05，clamp v_q |
| Critic 在 200D 上外推 | 先只用 BC 推理的 action 范围评估 Critic |
| Advantage 仍死亡 | 记录 adv_q90, near_zero_rate 趋势 |
| Q 梯度和离散 kNN 不对齐 | 监测 knn_margin 和 duplicate_rate |

### 3.3 embedding_loader.py

**不要手写 ckpt 命名规则。** 优先使用 `--gems_checkpoint` 直接传路径。`config.py` 已有 `cfg.gems_checkpoint_path` 属性，调用 `--gems_embedding_mode ideal_init --lambda_KL 0.05` 即可自动解析。`embedding_loader` 只接受 ckpt_path：

```python
def load_embedding_table(ckpt_path: str, device: torch.device) -> nn.Embedding:
    """从 GeMS ckpt 提取 item_embeddings.weight, 返回冻结 Embedding 表。

    Args:
        ckpt_path: GeMS checkpoint 完整路径, 如:
          checkpoints/gems/GeMS_mix_divpen_b5_pretrained_latent32_beta0.05_click1.0_seed58407201_ideal_init.ckpt
        device: torch device

    Returns:
        nn.Embedding [1000, 20], requires_grad=False
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    weight = ckpt["state_dict"]["item_embeddings.weight"]

    emb_mean = weight.mean().item()
    emb_std = weight.std().item()
    logging.info(f"Loaded embedding: mean={emb_mean:.4f}, std={emb_std:.4f}")

    table = nn.Embedding(weight.shape[0], weight.shape[1], _weight=weight.clone())
    table.requires_grad_(False)
    return table.to(device)
```

**train_agent.py 中的调用**:

```python
elif cfg.algo in ("flow_bc", "flow_iql"):
    ckpt_path = args.gems_checkpoint or cfg.gems_checkpoint_path
    embedding_table = load_embedding_table(str(ckpt_path), device)
    action_dim = 200

    # 显式指定 KL=0.05 (不要依赖默认值!)
    # 命令行: --gems_checkpoint checkpoints/gems/GeMS_mix_..._beta0.05_..._ideal_init.ckpt
    #    或: --gems_embedding_mode ideal_init --lambda_KL 0.05 (让 cfg.gems_checkpoint_path 自动解析)
```

### 3.4 噪声尺度标准化

`noise ~ N(0,1)` 与 item embedding 的 L2 norm (~1.7-4.0) 可能不匹配。建议在 Flow BC 训练时把 embedding 标准化到单位方差：

```python
# 在 FlowBCAgent.__init__ 中
self.emb_mean = embedding_table.weight.mean()
self.emb_std = embedding_table.weight.std()

# 训练时标准化 target
x0 = (x0_raw - self.emb_mean) / self.emb_std    # 标准化
noise = torch.randn_like(x0)                      # N(0,1) 匹配标准化后的尺度

# 推理后反标准化
x_raw = x * self.emb_std + self.emb_mean
items = knn(x_raw.reshape(10, 20), embedding_table.weight)
```

### 3.5 推荐监控指标

```python
# 训练日志中加 (每次 eval 后)
knn_margin = self.policy.knn_margin(emb)          # 越小越危险
duplicate_rate = (items_unique < 10).float().mean()
slate_unique_mean = count_unique(items).mean()

# 记录到 SwanLab
metrics["eval/knn_margin"] = knn_margin
metrics["eval/duplicate_rate"] = duplicate_rate
metrics["eval/slate_unique"] = slate_unique_mean
```

---

## 四、网络架构（Phase 1）

```python
class VelocityNet(nn.Module):
    """输入 [state, x_t, t] → 输出 速度场 v"""
    def __init__(self, state_dim=20, action_dim=200, hidden_dim=512, n_blocks=3):
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.x_proj = nn.Linear(action_dim, hidden_dim)
        self.t_proj = SinusoidalPosEmb(128) → nn.Linear(128, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, state_dim) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, x_t, t):
        s = self.state_proj(state)
        x = self.x_proj(x_t)
        t_emb = self.t_proj(t)
        h = x + t_emb
        for block in self.blocks:
            h = block(h, s)
        return self.out(h + x)                          # 残差 + 输出


class FlowPolicy:
    def __init__(self, velocity_net, embedding_table, flow_steps=10,
                 dedup_knn=True, emb_mean=0.0, emb_std=1.0):
        self.velocity_net = velocity_net
        self.embedding_table = embedding_table           # [1000, 20], frozen
        self.flow_steps = flow_steps
        self.dedup_knn = dedup_knn                       # 默认去重 kNN
        self.emb_mean = emb_mean                          # 标准化参数
        self.emb_std = emb_std
        self._eval_noise = None                           # 评估时固定噪声 [1, 200]

    def set_eval_noise(self, noise):
        """评估时固定噪声 tensor [1, 200], 保证可重复性"""
        self._eval_noise = noise

    @torch.no_grad()
    def sample(self, state, deterministic=True):
        """
        Args:
            state: [B, 20] belief state
            deterministic: True 时用固定噪声 (eval); False 时随机 (训练时采样)

        Returns:
            items: [B, 10] discrete item IDs
        """
        B = state.shape[0]
        device = state.device

        # 噪声 [B, 200]: deterministic 时用固定 seed 并 expand
        if deterministic and self._eval_noise is not None:
            x = self._eval_noise.to(device).expand(B, -1)   # [1,200] -> [B,200]
        else:
            x = torch.randn(B, 200, device=device)

        # Euler 积分: t 从 0→1
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((B, 1), i * dt, device=device)
            v = self.velocity_net(state, x, t)
            x = x + v * dt

        # 反标准化 (如果训练时做过标准化)
        x = x * self.emb_std + self.emb_mean

        # kNN 离散化
        emb = x.reshape(B, 10, 20)
        if self.dedup_knn:
            items = self._dedup_knn(emb, device)
        else:
            dists = torch.cdist(emb, self.embedding_table.weight)
            items = dists.argmin(dim=-1)
        return items

    def _dedup_knn(self, emb, device):
        """逐位置选最近 item, 已选的从候选中排除"""
        B = emb.shape[0]
        used = torch.zeros(B, 1000, dtype=torch.bool, device=device)
        items = torch.zeros(B, 10, dtype=torch.long, device=device)

        for pos in range(10):
            dists = torch.cdist(emb[:, pos:pos+1, :],
                                self.embedding_table.weight)
            dists = dists.squeeze(1)
            dists[used] = float('inf')
            chosen = dists.argmin(dim=-1)
            items[:, pos] = chosen
            used.scatter_(1, chosen.unsqueeze(1), True)

        return items

    def knn_margin(self, emb):
        """边界余量: 越小越危险"""
        dists = torch.cdist(emb, self.embedding_table.weight)
        top2 = dists.topk(2, dim=-1, largest=False).values
        return (top2[:, :, 1] - top2[:, :, 0]).mean().item()
```
```

---

## 五、文件总览

```
新建:
  src/agents/diffusion_slate/
  ├── __init__.py
  ├── velocity_net.py                # VelocityNet + FiLM + ResBlock
  ├── flow_policy.py                 # FlowPolicy (Euler + kNN)
  ├── flow_bc_agent.py               # Phase 1: FlowBCAgent
  ├── embedding_loader.py            # 从 GeMS ckpt 加载 item_embeddings.weight

修改:
  scripts/train_agent.py             # 拆 GeMS 路径 + --algo flow_bc

不修改:
  src/belief/gru.py                  # GRU Belief 完全复用
  src/data/trajectory_buffer.py      # buffer 完全复用
  src/env/simulators.py              # 评估环境完全复用
  src/utils/ (common, logger, checkpoint)  # 工具完全复用
```

---

## 六、训练入口

```bash
# Phase 1: Flow BC (显式指定 KL=0.05 checkpoint, 避免误用 KL=1.0)
python scripts/train_agent.py \
    --algo flow_bc \
    --env_name mix_divpen \
    --dataset_quality b5 \
    --gems_checkpoint checkpoints/gems/GeMS_mix_divpen_b5_pretrained_latent32_beta0.05_click1.0_seed58407201_ideal_init.ckpt \
    --flow_steps 10 \
    --max_timesteps 100000 \
    --seed 58407201
```

---

## 七、推荐实验顺序

| 顺序 | 实验 | 预计时间 | 产出 |
|------|------|---------|------|
| **0** | 完成 KL=0.05 IQL 的指标提取 | 已有数据，只需分析 | 确定 KL=0.05 改善了多少 |
| **1** | Flow BC, mix_divpen, 100k 步 | 半天训练 | eval reward vs Gaussian BC |
| **2** | Flow BC, topdown_divpen, 100k 步 | 半天训练 | 跨环境验证 |
| **3** | kNN ablation: top1 vs top3 + dedup | 半天评估 | 离散化方式对比 |
| **4** | 冻结 Flow BC, 训 200D Critic | 半天训练 | Q(s,a) 评估质量 |
| **5** | Q guidance sweep (0.01-0.1) | 半天评估 | 是否有 Q 引导收益 |
| **6** | (可选) joint fine-tune | 1天 | 完整 Flow-IQL |

**第 0 步最便宜也最必要**——先看清楚 KL=0.05 改善了哪些东西、哪些没改善，再决定 Flow Matching 的发力方向。