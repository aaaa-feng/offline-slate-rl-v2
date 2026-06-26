# GeMS VAE: Loss 分解 与 离线 RL 场景下的重训策略

## 1. GeMS VAE 是干什么的

GeMS (Generative Slate Recommendation) 是一个 Slate-VAE：

- **编码**：把推荐列表（10 个 item ID + 对应的 0/1 click）压成一个 32 维潜向量 z
- **解码**：从 z 重构出这 10 个 item（slate reconstruction）+ 预测每个 item 是否被点击（click prediction）

在 IQL 离线 RL 中，GeMS 的角色是**动作空间转换器**——把 RL 难以处理的离散 slate（C(1000,10) 种组合）转换为 32 维连续向量，Actor 只需要学会输出 32 维向量即可。

## 2. GeMS VAE 的 Loss

`rankers.py:176`:

```python
loss = slate_loss + λ_click * click_loss + λ_KL * KLLoss + λ_prior * prior_reg
```

### 2.1 slate_loss — 列表重构损失

```python
slate_loss = CrossEntropyLoss(item_logits, slates.flatten())
```

- **输入**：encoder(inference) 把 slab → z，decoder 把 z → 每个位置的 item logits [batch*10, 1000]
- **标签**：真实的 item ID（离散）
- **含义**：decoder 输出的每个位置，softmax 后预测正确 item 的概率
- **作用**：保证 z 保留了"列表里有什么 item"的信息

### 2.2 click_loss — 点击预测损失

```python
click_loss = BCEWithLogitsLoss(click_logits, clicks.flatten())
```

- **输入**：decoder 从 z 预测每个位置是否被点击 [batch*10]
- **标签**：真实的 click（0/1）
- **含义**：模型能否从 z 中推测出用户会不会点击这个 item
- **作用**：让 z 包含"用户喜欢哪种 item"的信息

### 2.3 KLLoss — KL 散度 / 潜空间正则化

```python
mean_term = ((μ - μ_prior)²) / σ²_prior
KLLoss = 0.5 * (log σ²_prior - log σ² + σ² / σ²_prior + mean_term - 1).mean()
```

- **μ, σ²**：encoder 对每个输入输出的后验参数
- **μ_prior, σ²_prior**：先验分布参数。`run_prior()` 返回 `μ_prior=0, σ²_prior=I`（标准高斯）

**这项是标准 VAE 的 KL 散度 D_KL( q(z|x) || p(z) )**。

物理含义：
- 约束 encoder 输出的 z 分布不要偏离标准高斯太远
- 每个维度的 μ→0, σ→1 → 潜空间"紧凑"、"连续"、"可采样"
- **λ_KL 越大 → encoder 被压得越厉害 → σ 越小 → 数据云越窄**
- **λ_KL 越越小 → encoder 自由度越大 → σ 越大 → 数据云越宽**

**这就是本次 KL 消融实验的逻辑**：降低 λ_KL 让潜空间更宽。

### 2.4 prior_reg — 先验正则化

```python
prior_reg = sum(μ_prior² + log(σ²_prior)²)
```

- 防止 `run_prior()` 学出奇怪的值

### 2.5 各项权重

| 参数 | 默认值 | 控制什么 |
|------|--------|---------|
| `λ_click` | 1.0 | 点击预测在 loss 中的占比 |
| `λ_KL` | 1.0 | KL 正则化强度（**本次要调的核心参数**） |
| `λ_prior` | 1.0 | 先验正则化强度（一般不调） |

### 2.6 当前 KL=1.0 时发生什么

```
slate_loss:  ~3.5   (主项，模型必须学会重构列表)
click_loss:  ~0.6   (次要但有用，让 z 包含点击信息)
KLLoss:      ~0.3   (KL 把 z 的方差约束在 ≈1 附近)
prior_reg:   ~0.0   (先验几乎是固定的，这项很小)
```

结合之前的静态分析结果：80% 方差只在 2 个主成分，条件数 761。这意味着 KL loss 把没用上的维度压向 0（方差极小），只留 2-4 个维度携带 slate 的信息。

## 3. 离线 RL 场景下该如何重训 GeMS

### 3.1 Embedding 初始化：必须用 scratch

`ideal_init` 是把模拟器内部的 ground-truth embedding 喂给 GeMS 作为 encoder 输入。这是作弊——等于告诉 GeMS "item 之间的真实关系是什么"。

离线 RL 中 Agent 只能访问数据集中的 item ID，不应该接触环境内部状态。所以必须：

```
--item_embedds scratch
```

从随机初始化开始，让 embedding 在 VAE 的 slate reconstruction + click prediction 目标下学习。学到的是"item 在数据集中的共现模式"，而不是"item 的 ground-truth topic 结构"。

### 3.2 KL 权重：应该降低

λ_KL=1.0 产生的数据云只有 2-4 个有效维，Actor 永远打不中靶子。降低 λ_KL：

- λ_KL=0.5：温和降低，云稍宽
- λ_KL=0.1：显著放宽，预期 PC80 从 2→6+
- λ_KL=0.05：激进放宽，PC80 可能 8-12
- λ_KL=0.01：几乎无正则化，encoder 自由放大的风险

太低的风险：z 分布偏离高斯 → decoder 在采样时没法 interpolate → 可能 GM sleneration 时采出无效的 z。

### 3.3 其他可调参数

| 参数 | 建议 | 理由 |
|------|------|------|
| `λ_click` | 保持 1.0 | 点击信息对 IQL 很重要（Advantage 需要通过 click 区分好坏动作） |
| `latent_dim` | 保持 32 | 改 KL 比改维度更直接；后续可以尝试 64 |
| `hidden_layers_infer/decoder` | 保持 [512,256]/[256,512] | 不改架构，只调 loss |
| `fixed_embedds` | `scratch` | embedding 必须可训练 |

### 3.4 训练数据

GeMS VAE 的训练数据是**离线 D4RL npz 数据集**（而非在线交互 .pt 文件）：

```
data/datasets/offline/mix_divpen/mix_divpen_b5_data_d4rl.npz
data/datasets/offline/topdown_divpen/topdown_divpen_b5_data_d4rl.npz
```

这些数据由在线 SAC+GeMS agent（scratch embedding）与 RecSim 交互采集而来，格式为 `(slate, clicks, rewards, ...)`。GeMS VAE 只用其中的 `slate` 和 `clicks`。

## 4. 训练命令（确认版）

```bash
cd /data/liyuefeng/offline-slate-rl-v2

# mix_divpen, lambda_KL=0.05
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_gems.py \
    --dataset mix_divpen_b5 \
    --item_embedds scratch \
    --lambda_KL 0.05 \
    --lambda_click 1.0 \
    --seed 58407201 \
    --max_epochs 50 \
    > logs/gems/mix_kl005.log 2>&1 &
```

所有 10 个模型训完后，对每个 ckpt 跑静态云分析（实验 1），对比 PC 数、条件数、per_dim_std。

如果 λ_KL=0.05 的云明显更宽（PC80 ≥ 8，条件数 < 50，per_dim_std > 0.3），就用这个 ckpt 重跑 IQL β ablation 看 eval reward 是否不再崩塌。

---

## 5. clicks 的角色：在线 vs 离线场景

### 5.1 GeMS VAE 训练阶段：必须用真实 click

GeMS VAE 的 loss 有两项和 click 直接相关：

- **click_loss**：decoder 从 z 预测每个位置是否被点击。没有真实 click 这个 loss 就是垃圾——模型学不到"用户喜欢哪种 item"
- **KLLoss 中间接影响**：同一 slate + 不同 click pattern → encoder 输出不同的 z。如果没有 click（全 0），同一 slate 永远映射到同一个 z。这叫**确定性映射**，有效 cloud complexity 更低

所以 GeMS VAE **训练时必须用真实 click**。D4RL .npz 数据集里有 clicks 字段（由 behavior policy 与 RecSim 交互时的真实点击产生），用这个就行。

### 5.2 IQL 训练阶段：fake_zero vs real click

IQL 在生成动作标签时调用 `GeMS.run_inference(slate, clicks)`。两个选项：

**fake_zero（当前做法）**：
```
slate → GeMS.run_inference(slate, clicks=zeros) → latent_mu（固定）
```
- 每个 slate 一个确定性的 z → Actor 学的是"这个 slate 对应的那个 z"
- 简单、稳定、可复现
- **问题**：丢弃了 click 信息。一个"好 slate 被点了 5 个 item"和一个"同样的 item 列表但只被点了 1 个"在 fake_zero 下得到完全一样的 z。这意味着 Actor 无法区分"好 slate"和"一般 slate"的连续表示

**真实 click（旧报告建议的改进）**：
```
slate → GeMS.run_inference(slate, clicks=real_clicks) → latent_mu（变化）
```
- 同一 slate + 不同 click → 不同 z → 动作云更丰富
- 旧报告估计多样性 +28%
- **潜在问题**：真实 click 来自 behavior policy，可能引入 behavior policy 的偏差——但这对 IQL 来说是**正常的数据分布**，因为 reward 也是 behavior policy 产生的

### 5.3 在线 vs 离线在这个位置的本质区别

| | 在线 RL (SAC+GeMS) | 离线 RL (IQL+GeMS) |
|---|---|---|
| GeMS 用途 | decoder: z → slate（动作生成） | encoder: slate → z（动作标签生成） |
| click 来源 | 环境实时返回（真实） | 数据集中的 logged clicks |
| click 用途 | 驱动 reward，影响下一轮推荐 | GeMS 编码时作为额外上下文 |
| 核心差异 | **在线交互中产生新 click** | **只能使用历史 logged click** |

在在线 RL 中，GeMS 的 decoder 把 Actor 输出的 z 解码成 slate，环境返回真实的 click。闭环。在离线 RL 中，GeMS 的 encoder 从历史数据中的 slate+click 编码成 z（标签），Actor 学这个 z。开环。

**在线和离线对同一 slate 的 click 会不同吗？**
会。不同用户状态（用户 embedding）下，同一 slate 会产生不同的 relevance → 不同的 click 概率。而离线数据只记录了 behavior policy 那一次交互的 click——这个 click 是**条件于当时的用户状态和 behavior policy 的推荐历史**的。Actor 使用 fake_zero 相当于忽略了所有这些上下文差异。

**这对 GeMS 训练有什么潜在影响？**
没有。GeMS 训练不关心 click 是怎么产生的——它只学习 slate+click → z 的编码映射。无论 click 来自在线交互还是 logged data，slate reconstruction + click prediction 都是有效的自监督信号。

### 5.4 结论：离线场景下应该怎么用 click

- **GeMS VAE 训练**：用离线数据中的真实 click → 保持丰富的潜空间结构
- **IQL 标签生成**：当前 fake_zero 可以工作，但用真实 click 理论上能增加动作云多样性。**建议在 KL 消融之后，作为独立的对照实验来验证**

---

## 6. GeMS 训练的数据对比

### 6.1 在线 GeMS（原论文）

| 项目 | 值 |
|------|-----|
| 数据格式 | .pt 文件（用户交互序列） |
| 数据内容 | `{user_id: [(slate_1, clicks_1), (slate_2, clicks_2), ...]}` |
| 采集方式 | epsilon-greedy logging policy (ε=0.5) 与 RecSim 交互 |
| embedding 初始 | `from_scratch()`（随机） |
| **参数** | **KL=1.0, click=0.5, prior=0.0, lr=0.001, epochs=15** |

参数来源：`GeMS/config/pretrain_GeMS.yml` (原论文官方配置)，非 README 中的示例命令。

### 6.2 离线 GeMS（旧项目 ideal_init）

| 项目 | 值 |
|------|-----|
| 数据格式 | D4RL .npz |
| 数据内容 | slates [1M,10], clicks [1M,10], rewards, etc. |
| 采集方式 | SAC+GeMS agent (scratch embed) + epsilon-greedy 混合策略 |
| embedding 初始 | `from_pretrained(item_embeddings_diffuse.pt)` ← 环境真值 |
| 参数 | KL=1.0, click=1.0, latent=32, seed=58407201 |

### 6.3 离线 GeMS（旧项目 "scratch" — 其实是 MF 初始化）

| 项目 | 值 |
|------|-----|
| embedding 初始 | `from_pretrained(data/embeddings/mf/mf_mix_b5.pt)` ← MF embedding |
| 命令 | `--embedding_path data/embeddings/mf/mf_mix_b5.pt --fixed_embedds scratch` |

**注意**：旧项目 `train_gems_offline.py` 没有 `--item_embedds scratch` 参数。所谓的 "scratch" ckpt 实际上是用 MF embedding 初始化的（然后 GeMS 训练中 embedding 可更新）。这和真正的随机初始化不是一回事。

### 6.4 离线 GeMS（新项目，本次计划）

| 项目 | 值 |
|------|-----|
| 数据格式 | 同一份 D4RL .npz（从旧项目复制） |
| embedding 初始 | `from_scratch()`（真正随机初始化） |
| 参数 | KL={0.01,0.05,0.1,0.5,1.0}, click=1.0, latent=32, seed=58407201 |

### 6.5 数据差异对 GeMS 训练的潜在影响

三个版本的 GeMS 训练用的是**同一份 D4RL .npz 数据**（slates + clicks 来自相同的 behavior policy 采集），区别只在 embedding 初始化和 KL 权重。

- ideal_init → embedding 从环境真值出发 → encoder 输入一开始就有"哪些 item 客观上相似"的信息 → 潜空间编码可能偏离纯数据驱动的结果
- MF_init → embedding 从协同过滤出发 → encoder 输入有"用户历史上爱点哪些 item"的信息 → 更接近纯数据驱动，但 MF embedding 的 item 间 cos 高达 0.86，区分度差
- scratch → embedding 随机初始化 → encoder 必须从零学习 item 之间的关系 → 最纯粹的"离线数据驱动"方式

理想情况下，scratch 初始化的 GeMS 学出的 embedding cos 应该 < 0.4，比 MF_init（0.86）和 ideal_init（0.44）更分散。这就是我们换到 scratch 的核心原因。

---

## 7. 旧 GeMS 训练参数总结

### 旧项目 mix_b5 ideal_init (当前 IQL 用的)

```
python scripts/train_gems_offline.py \
    --dataset_path data/datasets/offline/mix_divpen/mix_divpen_v2_b5_data_d4rl.npz \
    --embedding_path data/embeddings/item_embeddings_diffuse.pt \
    --latent_dim 32 \
    --lambda_KL 1.0 \
    --lambda_click 1.0 \
    --lambda_prior 1.0 \
    --max_epochs 50 \
    --batch_size 256 \
    --device cuda \
    --seed 58407201
```

### 旧项目 mix_b5 "scratch" (其实 MF-init, 未用于 IQL)

```
python scripts/train_gems_offline.py \
    --dataset_path data/datasets/offline/mix_divpen/mix_divpen_v2_b5_data_d4rl.npz \
    --embedding_path data/embeddings/mf/mf_mix_b5.pt \
    --fixed_embedds scratch \
    --latent_dim 32 \
    --lambda_KL 1.0 \
    --lambda_click 1.0 \
    --lambda_prior 1.0 \
    --max_epochs 50 \
    --batch_size 256 \
    --device cuda \
    --seed 58407201
```

### 新项目 mix_divpen_b5 scratch (本次计划，KL 消融)

```bash
python scripts/train_gems.py \
    --dataset mix_divpen_b5 \
    --item_embedds scratch \
    --lambda_KL 0.05 \
    --lambda_click 1.0 \
    --seed 58407201 \
    --max_epochs 50
```

**差异总结**：
- 旧 ideal_init：env embedding 初始化 + KL=1.0 → 作弊，不用于正式实验
- 旧 "scratch"：MF embedding 初始化 + KL=1.0 + embedding 可训练 → 不是真正的随机初始化
- 新 scratch：真正随机初始化 + KL sweep → 本次实验
