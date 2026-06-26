# 扩散策略直接生成 Slate：扔掉 GeMS，端到端训练

> **目标**：用扩散模型或 Flow Matching 直接从 belief state 生成 slate，完全放弃 GeMS VAE 预训练。

---

## 一、为什么扔掉 GeMS

当前管线有三层瓶颈，且它们互相放大：

```
slate → GeMS encoder → 32维 latent → GeMS decoder → argmax → discrete items
                          ↑                            ↑
                    KL把云压扁 (已缓解)         极其敏感 (未解决)
```

KL=0.05 解决了第一层。但第二层是 argmax 固有的——1000 选 1，连续空间里偏移 0.1 就可以让 argmax 选不同的 item。这种"脆性"不是调参数能消除的，是架构决定的。

**扔掉 GeMS 后**：不存在"连续动作→离散 slates 的脆性映射"。扩散直接生成 10 个 item 的嵌入向量，用 kNN / soft nearest neighbor 找到最近的真实 item。kNN 比 argmax 平滑得多——连续空间里的小移动对应 kNN 结果的小变化，不会导致整个 slate 翻脸。

---

## 二、新管线：Flow Matching → Slate

### 2.1 推理链路

```
obs → GRU → belief_state [20]
  → flow matching sampler (T=10-20 步)
    → 从纯噪声逐步生成 clean slate_embedding [10, 20]
  → 每个位置的 20 维向量 → kNN 搜索 item_embedding 表 [1000, 20]
  → 10 个 item ID → 推荐列表
```

### 2.2 为什么 Flow Matching 而不是 DDPM

| | DDPM | Flow Matching |
|--|------|--------------|
| 采样步数 | 100-1000 | **5-20** |
| 训练速度 | 慢（需要很多步去噪） | 快（直线路径，一步到位） |
| 实现复杂度 | 中等 | **更简单** |
| 推荐场景适用性 | 推理延迟太高 | 延迟可控 |

Flow Matching 的核心思想：定义一条从噪声 x₁ 到目标 x₀ 的直线路径，训练网络预测方向向量。推理时沿预测方向走即可——几步就到。

```python
# 训练: t=0 是噪声, t=1 是数据. 速度场从噪声指向数据.
t ~ Uniform(0, 1)
noise = randn_like(x0)
x_t = (1 - t) * noise + t * x0               # 直线插值: noise→data
target_vel = x0 - noise                       # 真实速度场: 噪声→数据
pred = velocity_net(x_t, t, state)            # 预测速度场
loss = MSE(pred, target_vel)

# 推理: 从 t=0 (噪声) 出发, 沿速度场走到 t=1 (数据)
x = randn(200)                                # 噪声起点 (t=0)
dt = 1.0 / flow_steps
for i in range(flow_steps):
    t = torch.full((1,1), i * dt)             # 当前时间步
    v = velocity_net(state, x, t)             # 预测速度场
    x = x + v * dt                            # Euler步: 从噪声→数据
output = x  # clean slate_embedding (t≈1)
```

### 2.3 slate embedding 的训练标签

不需要 GeMS 编码。直接用训练数据中的 item ID 查 item embedding 表：

```python
# 从 D4RL 数据提取
slate_ids = batch["slate"]                           # [256, 10] int
target = item_embedding(slate_ids)                   # [256, 10, 20] float

# Flow matching 训练
t = rand(256, 1, 1)
noise = randn_like(target)
x_t = (1 - t) * target + t * noise                  # 直线插值
v_true = noise - target                              # 真实方向
v_pred = velocity_net(x_t, t.squeeze(), belief_state) # 预测方向
loss = MSE(v_pred, v_true)
```

**item embedding 表从哪来**：直接用 GeMS 训练好的 embedding（KL=0.05 那版就行），或者从头训一个简单的 embedding。embedding 本身不参与梯度更新（冻结），只用来把离散 item ID 变成连续向量。

---

## 三、网络架构

### 3.1 生成目标

输入: belief_state [batch, 20]
输出: slate_embedding [batch, 10, 20]

### 3.2 Velocity Network

```
输入:
  x_t:    [batch, 10, 20]   带噪声的 slate embedding
  t:      [batch]            时间步 (0→1)
  state:  [batch, 20]        GRU belief state

处理:
  # 将 (10, 20) 展平成 200 维，加上时间和状态条件
  x_flat = x_t.reshape(batch, 200)                    # [batch, 200]
  t_embed = SinPosEmb(t) → Linear(128→256)            # [batch, 256]
  s_proj  = Linear(20→256)(state)                      # [batch, 256]

  h = Linear(200→512)(x_flat)
  h = h + t_embed + s_proj                             # 融合条件

  # 3 个残差块
  for i in range(3):
      h = h + ResidualFFN(h, film(s_proj, t_embed))

  h = Linear(512→200)(h)
  output = h.reshape(batch, 10, 20)                    # 预测的速度场
```

**参数量**: ~2M（很少，比 GeMS decoder + TanhGaussian 加起来还小）

### 3.3 Item Embedding 表

```
embedding_table = nn.Embedding(1000, 20)   # 1000 个 item, 每个 20 维
embedding_table.weight.requires_grad = False  # 冻结
```

- 初始化来源：KL=0.05 GeMS 训练后的 `item_embeddings.weight`
- 可以后续做 ablation：用随机初始化 vs GeMS 初始化 vs MF 初始化

---

## 四、离散化：从连续嵌入到 item ID

### 4.1 当前问题（argmax）

```python
slate_logits = slate_embedding @ embedding_table.T  # [10, 1000]
items = argmax(slate_logits, dim=-1)                  # 每个位置独立取 max
```

argmax 是脆的——logits 里第二高的 item 跟第一高的只差 0.01，但 argmax 只选第一。

### 4.2 kNN 方案（更平滑）

```python
# 对每个位置: 计算生成的 embedding 与所有 item embedding 的 L2 距离
dists = cdist(generated_embedding, embedding_table)   # [10, 1000]
_, items = topk(dists, k=1, largest=False)            # kNN: 找最近的 item
```

比 argmax 平滑：连续变化 → 距离连续变化 → kNN 结果渐变而非跳变。

### 4.3 可微 kNN 方案（Straight-Through Gumbel Softmax）

训练时如果需要梯度穿过 discretization：

```python
# 用 softmax 近似 argmax
logits = -cdist(generated_embedding, embedding_table)  # [10, 1000]
soft_items = softmax(logits / temperature)               # [10, 1000] soft
hard_items = one_hot(argmax(logits))                     # [10, 1000] hard
items = hard_items - soft_items.detach() + soft_items    # STE
```

STE 允许训练时梯度回传到 embedding 生成网络。

---

## 五、训练流程

### 5.1 纯 BC 模式（Phase 1）

```python
for step in range(max_steps):
    batch = buffer.sample(256)
    
    # Belief encoding (不变)
    states, _ = belief.forward_batch(batch)
    s = states["actor"]                      # [256, 20]
    
    # Target: 训练数据中的真实 slate 的 item embedding
    slate_ids = batch["slate"]               # [~25k, 10] (flattened from 256 episodes)
    x0 = embedding_table(slate_ids)           # [~25k, 10, 20]
    x0_flat = x0.flatten(1)                   # [~25k, 200]
    
    # Sub-sample to avoid OOM (256 episodes ≈ 25600 transitions)
    n_sample = min(4096, x0_flat.shape[0])
    idx = torch.randperm(x0_flat.shape[0])[:n_sample]
    x0_flat = x0_flat[idx]
    s_sub = s[idx]
    
    # Flow Matching 训练: t=0 noise, t=1 data, 速度场从 noise→data
    t = torch.rand(n_sample, 1, device=s.device)
    noise = torch.randn_like(x0_flat)
    xt = (1 - t) * noise + t * x0_flat                  # 直线插值
    target_vel = x0_flat - noise                         # 真实速度场
    
    v_pred = velocity_net(s_sub, xt, t)
    loss = MSE(v_pred, target_vel)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这个阶段只做模仿学习——学到"行为策略推荐了什么 slate"。

### 5.2 RL 模式（Phase 2，后续）

在 BC 基础上用 Critic Q 值引导推理时的采样方向：

```python
# 推理时 classifier guidance
for t in linspace(1, 0, steps):
    v = velocity_net(xt, t, s)               # 无条件方向
    with grad():
        q_val = critic(s, xt.reshape(b, 200)) # Q(s, slate_emb)
    v_q = grad(q_val, xt)                     # Q 梯度: 往高 Q 方向推
    xt = xt + (v + λ * v_q) * dt              # 条件方向
```

但这个需要 Critic 接受 slate embedding 作为输入（200 维），跟当前 Critic 不同（当前 Critic 接受 32 维 latent）。需要重新设计 Critic——或者先跳过 RL 模式，纯 BC 的扩散已经比 Gaussian BC 强了。

---

## 六、与当前代码的集成

### 6.1 新建文件

```
src/agents/diffusion_slate/
├── __init__.py
├── velocity_net.py          # Flow Matching 速度场网络
├── slate_policy.py          # 采样逻辑 (flow matching + kNN)
├── slate_agent.py           # DiffusionSlateAgent (BC模式)
└── schedule.py              # 时间步 schedule 工具
```

### 6.2 现有代码复用

| 模块 | 是否复用 | 备注 |
|------|:---:|------|
| GRU Belief | ✅ 完全复用 | 输入仍是 slate+clicks 观测 |
| GeMS ranker | ❌ 不再使用 | 扔掉 |
| GeMS encoder | ❌ 不再使用 | 扔掉 |
| IQL Critic/Value | ⚠️ 暂不复用 | Phase 1 纯 BC，不需要 Q/V |
| TrajectoryReplayBuffer | ✅ 完全复用 | 数据加载不变 |
| RecSim Eval Env | ✅ 完全复用 | 评估不变 |
| train_agent.py | 加新 agent 类型 | `--algo diffusion_bc` |
| item embedding | ✅ 复用 | 从 GeMS ckpt 提取，冻结 |

### 6.3 训练入口

```bash
python scripts/train_agent.py \
    --algo diffusion_bc \
    --env_name mix_divpen \
    --dataset_quality b5 \
    --gems_embedding_mode ideal_init \   # 只用来提取 item embedding
    --max_timesteps 100000 \
    --seed 58407201
```

---

## 七、实验计划

### 7.1 Phase 1: 纯 BC 验证

| 实验 | 内容 | 对比 |
|------|------|------|
| D-BC-1 | Flow Matching BC, mix_divpen, KL=0.05 embedding | Gaussian BC (β=0, 已有) |
| D-BC-2 | Flow Matching BC, topdown_divpen, KL=0.05 embedding | Gaussian BC (β=0, 已有) |
| D-BC-3 | kNN vs argmax 离散化对比 | — |

**验证目标**: 扩散 BC 的 reward 终值 > Gaussian BC 终值, combo_hit 不归零。

### 7.2 Phase 2: RL 扩展

| 实验 | 内容 |
|------|------|
| D-RL-1 | Classifier guidance with pretrained Critic |
| D-RL-2 | End-to-end diffusion + critic joint training |

### 7.3 Ablation

| 实验 | 内容 |
|------|------|
| D-A-1 | Flow Matching vs DDPM (推理速度 vs 质量) |
| D-A-2 | 不同采样步数 (5/10/20) 对最终 reward 的影响 |
| D-A-3 | Item embedding 初始化对比 (GeMS / MF / Random) |

---

## 八、风险与可行性

| 风险 | 等级 | 应对 |
|------|:---:|------|
| 纯 diffusion BC 学不会 | 低 | Flow Matching 在图像/视频生成已成熟，200 维小问题不会难 |
| kNN 离散化仍有脆性 | 中 | 可尝试 soft kNN（top-3 采样）或温度 softmax |
| 推理速度不够快 | 低 | Flow Matching 只需 10 步，每步 2M 参数，< 5ms |
| BC 效果不如 IQL（缺 RL） | 中 | Phase 2 加 classifier guidance |

---

## 九、总结：为什么这个方案值得尝试

| 维度 | 当前 (GeMS + IQL) | 扩散 Slate (新方案) |
|------|------------------|-------------------|
| 预训练 | 需要训 VAE（GeMS） | **不需要** |
| 动作空间 | 32 维 latent → 脆性 argmax | **10×20 维 embedding → 平滑 kNN** |
| 策略表达 | 单峰高斯 | **任意分布（Flow Matching）** |
| 推理速度 | 1 步（Gaussian 采样） + 1 步（argmax） | **10 步（Flow Matching）+ 1 步（kNN）** |
| 模块数 | GeMS encoder + decoder + Actor + Critic + Value | **VelocityNet + kNN** |

最核心的优势：**不存在"解码器脆性"这个根本问题了**。扩散直接生成的是 item embedding（语义空间），kNN 把这个 embedding 映射到最近的 item——连续变化 → kNN 结果渐变。不再有"连续向量挪 0.1 → slate 全换"的灾难。

---

## 十、方案 B：IQL + Flow Matching（保留值学习，替换策略）

### 10.1 核心思想

上面的方案把 IQL 也扔了——只有 Flow Matching BC + Q 引导。但如果想**保留 IQL 的值学习框架**，让它和 Flow Matching 策略共存呢？

```
IQL 的 Critic/Value  →  负责评价（Q/V 学习）— 告诉策略"什么动作好"
Flow Matching       →  负责生成（策略）— 从"好动作"的分布里采样

训练时：IQL 训 Q/V，Flow Matching 训 BC（或从 Q 引导里受益）
推理时：Flow Matching 采样 + Q 引导 = 生成高质量动作
```

这就是 Diffusion-QL / IDQL 的思路，但针对 slate 推荐场景。

### 10.2 训练流程

```python
for step in range(max_steps):
    batch = buffer.sample(256)
    states, next_states = belief.forward_batch(batch)
    s = states["actor"]              # [256, 20]
    s_critic = states["critic_v"]    # [256, 20]
    ns = next_states["critic_v"]     # [256, 20]

    # === 1. 构造 action label（从数据，不需要 GeMS） ===
    slate_ids = batch["slate"]                            # [256, 10]
    true_action = embedding_table(slate_ids).flatten(1)   # [256, 200]

    # === 2. IQL Value 训练（不改） ===
    v = self.value(s_critic)
    q1, q2 = self.critic_1(s_critic, true_action), self.critic_2(s_critic, true_action)
    q = torch.min(q1, q2)
    value_loss = expectile_loss(q - v, tau=0.8).mean()
    ...

    # === 3. IQL Critic 训练（不改） ===
    target_q = rewards + gamma * self.value(ns)
    critic_loss = MSE(q1, target_q) + MSE(q2, target_q)
    ...

    # === 4. Flow Matching 策略训练（新增） ===
    # t=0 noise, t=1 data, 速度场从 noise→data
    t = rand(n_sample, 1)
    noise = randn_like(x0_flat)
    x_t = (1-t) * noise + t * x0_flat                     # 直线插值
    target_vf = x0_flat - noise                            # 真实速度场
    pred_vf = velocity_net(s_sub, x_t, t)                  # 预测速度场
    flow_loss = MSE(pred_vf, target_vf)

    # 可选: 用 Q 值加权——高 Q 动作的样本权重大
    with torch.no_grad():
        q_vals = self.critic_1(s, true_action)
        weights = torch.softmax(q_vals / temperature, dim=0)
    flow_loss = (weights * (pred_vf - target_vf)**2).mean()

    flow_optimizer.zero_grad()
    flow_loss.backward()
    flow_optimizer.step()
```

**关键**：IQL 的 Critic 同时用于：(1) 计算 Advantage 更新 V，(2) 给 Flow Matching 的样本加权。Q 值高的 slate 在 BC 里被更重视——这就是 AWR 在 Flow Matching 上的等价形式。

### 10.3 推理流程

```python
def act(self, obs):
    # 1. GRU 编码
    belief_state = self.belief.forward(obs)["actor"]  # [20]

    # 2. Flow Matching + Q 引导采样
    x = torch.randn(1, 200)                            # 噪声起点
    for t_step in torch.linspace(1.0, 0.0, 10):       # 10步
        dt = 1/10

        # 无条件方向
        v_flow = self.velocity_net(x, t_step, belief_state)

        # Q 引导方向（critic 的梯度指向更高 Q 值）
        x.requires_grad_(True)
        q = self.critic_1(belief_state, x)
        v_q = torch.autograd.grad(q.sum(), x)[0]

        # 组合方向
        v = v_flow + self.guidance_scale * v_q
        x = x + v * dt

    # 3. KNN 离散化
    slate_emb = x.detach().reshape(10, 20)              # [10, 20]
    dists = torch.cdist(slate_emb, self.embedding_table.weight)  # [10, 1000]
    slate = dists.argmin(dim=-1)                        # KNN: 每位置选最近item

    return slate.cpu().numpy()
```

### 10.4 与方案 A（纯 Flow Matching BC）的对比

| | 纯 BC（方案 A） | IQL + Flow Matching（方案 B） |
|---|:---:|:---:|
| 值函数 | 无 | IQL Q/V（复用的） |
| 策略学习信号 | 只有 BC（模仿数据） | BC + Q 加权（AWR 等价） |
| 推理引导 | 无 | Q 梯度引导流场 |
| 能否超越行为策略 | 不能（纯模仿） | **能**（Q 指向更好的动作） |
| 改动量 | 小 | 中 |

### 10.5 为什么方案 B 更好

纯 BC 的 Flow Matching 只能模仿行为策略——看到什么学什么。如果行为策略在某个状态下推了某种 slate，Flow Matching 就学这个。但数据里有些 slate 比另一些好（reward 高的），BC 同等对待。

IQL + Flow Matching 用 Q 值来区分——Q 值高的 slate 在训练时权重更大，推理时 Q 梯度把流场往高 Q 方向推。**这在数学上等价于 AWR**（Advantage Weighted Regression），只不过策略从 Gaussian 换成了 Flow Matching。

**一句话：保留 IQL 的"脑子"（Q/V），把"手脚"从 Gaussian 换成 Flow Matching。**

### 10.6 改动量

| 文件 | 改动 |
|------|------|
| `agents/iql/networks.py` | Critic action_dim: 32→200, hidden: 256→512 |
| `agents/iql/agent.py` | 加 `FlowIQLAgent` 类，替换 actor 为 velocity_net |
| **新建** `agents/diffusion_slate/` | velocity_net.py, flow_policy.py, schedules.py |
| `scripts/train_agent.py` | `--algo flow_iql` 选项 |

不需要改：GRU Belief、buffer、eval env、SwanLab 日志、checkpoint 保存。

---

## 十一、来自 DEAS-FQL 的启发

### 11.1 DEAS 是什么

DEAS (Kim et al. 2025, https://arxiv.org/abs/2510.07730) 是离线 RL 的 flow matching 方案，用于机器人操作（OGBench cube-puzzle 等）。核心组件和我们需要的完全对应：

| DEAS 概念 | 我们的场景 | 对应关系 |
|-----------|----------|---------|
| **action sequence** (4-8 个连续动作) | **slate** (10 个 item) | 都是"一批动作"，可 flatten |
| **BC flow matching** | Flow Matching BC | 完全相同——直线插值+速度场预测 |
| **one-step distillation** | 推理加速 | 训练一个单步学生模型，避免多步 Euler |
| **FQL critic** | IQL critic | 都是离线 Q-learning，结构类似 |
| **distill + Q loss** | BC + Q-weighted loss | α·distill + Q ——数学形式一致 |

### 11.2 两个关键设计（可直接借鉴）

**设计 1: BC flow → one-step 蒸馏**

DEAS 的做法：

```python
# Step A: 训 BC flow model (T步Euler推理)
vel = ActorVectorField(obs, x_t, t)     # 预测速度场
bc_flow_loss = MSE(vel, target - noise)

# Step B: 蒸馏出一个 one-step model (直接从噪声到动作)
noises = randn(B, action_dim)
target_actions = compute_flow_actions(obs, noises)  # 用 BC flow model 推理 T 步
student_actions = ActorOneStep(obs, noises)          # 学生模型一步出结果
distill_loss = MSE(student_actions, target_actions)
```

**好处**：推理时只用 `ActorOneStep`，一步出结果——不需要 10 步流场。训练时多步推理只在蒸馏阶段发生一次，之后就用不着了。

**设计 2: Q-guided distillation with normalization**

```python
# Actor loss = α * distill_loss + q_loss
q_loss = -Q(s, student_action).mean()

# 可选: normalize q_loss 让 α 在不同任务间一致
lam = 1 / |q|.mean()
q_loss = lam * q_loss
```

DEAS 论文建议 α 从 [0.03, 0.1, 0.3, 1, 3, 10] 中搜。

### 11.3 对我们需要调整的地方

DEAS 的 action space 是机器人关节角（连续值空间，[-1, 1]^D）。slate 推荐的空间本质是离散的（1000 选 10）。两者的差异需要在两个地方处理：

| DEAS | 我们需要改成 |
|------|------------|
| action ∈ [-1, 1]^D (连续) | action = slate_embedding ∈ R^{200} (连续，但通过 kNN 映射回离散) |
| bc_flow 输出 clamp 到 [-1, 1] | bc_flow 输出不需要 clamp——kNN 离最近 item 取就行 |
| actor_one_step 直接用于环境 | actor_one_step → kNN → item IDs → 环境 |

### 11.4 可以直接复用的 DEAS 代码模式

DEAS 的 `compute_flow_actions` 函数（Euler 法推理）可以直接改写用在 slate embedding 上：

```python
# DEAS 的 compute_flow_actions, 改写成 PyTorch + slate embedding 形式
def compute_flow_actions(velocity_net, obs, noises, flow_steps=10):
    """从噪声 + 状态出发，用 flow matching 生成 slate_embedding"""
    actions = noises  # [B, 200] 噪声起点
    for i in range(flow_steps):
        t = torch.full((B, 1), i / flow_steps)
        vel = velocity_net(obs, actions, t)           # 预测当前步的速度场
        actions = actions + vel / flow_steps            # Euler步
    return actions  # [B, 200] slate_embedding
```

`ActorVectorField` 的网络结构也非常简单——就是一个 MLP（`(*hidden_dims, action_dim)`），输入 `[obs, action, time]`，输出速度场。我们在 PyTorch 里实现的话，一个标准的 3-4 层 MLP 就够。

### 11.5 和 DEAS 的不同之处（我们的创新点）

| 维度 | DEAS | 我们（IQL + FM + KNN） |
|------|------|----------------------|
| 动作空间 | 连续关节角 | 连续 embedding → KNN 离散化 |
| 值函数 | FQL（HL-Gauss + Q-learning） | **IQL** (expectile regression) |
| 场景 | 机器人操作 | **slate 推荐** |
| BC 来源 | expert data 采样 | **logging policy 记录** |
| 独特挑战 | — | decoder 脆性（argmax → kNN 解决）；动作云结构（KL 消融验证） |

组合的独特贡献：**在 slate 推荐这个离散动作空间场景下，用 flow matching + kNN 同时解决了策略表达力（高斯不够）和解码脆性（argmax 太敏感）两个问题，并保留 IQL 的稳定值学习框架。**
