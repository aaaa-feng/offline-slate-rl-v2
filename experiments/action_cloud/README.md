# 动作云（Action Cloud）调研与实验设计

> 整合自旧项目 `motivation_test/logs/action_space_20260603/` 的调研结论 与 新项目 `beta_ablation_repreduce` 的实验发现

---

## 1. 什么是"动作云"

动作云 = 训练数据中所有 slate 经 GeMS 编码后，在 IQL 使用的归一化坐标 [-1,+1]^32 中形成的点集。

```
slate (10个item ID) + fake_zero clicks
  → GeMS.run_inference()  →  μ ∈ R^32   (原始潜空间 latent_mu)
  → normalize: (μ - action_center) / action_scale  →  clamp[-0.99,0.99]
  → x ∈ R^32   (归一化坐标，IQL 训练标签 = true_actions)
                                                        ↑
                                              这就是"动作云"里的一个点
```

Actor 要学的就是输出接近这些点的 32 维连续向量。云的形状决定 Actor 的"目标"有多大、多窄。

---

## 2. 已知的云特征（静态度量，训练前就存在）

### 2.1 来自旧项目离线统计（1M 条数据）

| 指标 | 数值 | 含义 |
|------|------|------|
| 有效 PC 数（80% 方差） | 2 | 32 维里只有 2 个方向携带了 80% 的信息 |
| 条件数 | ~761 | 最宽维 ÷ 最窄维 = 761×，极度扁平 |
| 每维 std 均值 | ~0.13 | 归一化后只用了 [-1,1] 的 13% 幅度 |
| L2 质心散布 | ~1.16 | 任意两个标签点的典型 L2 距离 |
| 标签到 dataset_center 的 L2 | ~2.76 | 云偏在坐标系一侧，但内部很紧 |

### 2.2 来自旧项目和新实验的训练侧指标

| 来源 | z_center | svd_rank | env |
|------|---------|---------|-----|
| 旧 mix_b8 | ~2.65-2.77 | — | mix |
| 新 mix_all | ~2.65-2.75 | 9-11 | mix |
| 新 td_all | ~3.37-3.42 | 11-12 | topdown |

关键结论：

云的内部 L2 散布 ~1.16，但 Actor 输出在 ~2.68（mix）或 ~3.39（topdown）远处。这意味着 Actor 在云外，且距离是云内部尺度的 2.3-2.9 倍。

topdown 的 z_center 更大（3.39 vs 2.68），但 svd_rank 也更高（11-12 vs 9-11）。topdown 的数据云更宽、有效维更多——这可能解释了为什么 topdown 的 slate 多样性崩塌较慢（topdown_b8 在 100k 仍有 unique=464）。

### 2.3 云的宽度由什么决定（旧报告 §3-§8 的因果链，按上游到下游排序）

根因 1: Item embedding 余弦相似度过高（0.40-0.51）
10 个 item 拼接后，210 维输入的有效自由度极低。SVD 分析：50% 方差仅需 6 个 PC。

根因 2: GeMS encoder 瓶颈（210→512→256→32）
已在低维输入上再压缩一层。

根因 3: KL 正则化（lambda_KL=1.0, prior=N(0,I)）
标准高斯先验把所有维往 0 压。KL loss 把"没用"的维压得极扁（条件数 761）。

根因 4: fake_zero clicks
每个 slate 只有一个对应的 latent_mu（确定性映射），无多样性。用真实 click 会让同一 slate 产生多个不同的 latent。

根因 5: clamp to [-0.99, 0.99]
IQL 的 action normalization 再走一层 clamp，把本来已经在边缘的点截断。

---

## 3. 实验设计：如何研究并拓宽动作云

### 实验 1：静态云表征（不训练，只分析） ✅ 已完成

**目的**：量化当前 ideal_init GeMS 产生的动作云到底多窄，为后续实验提供 baseline。

**方法**：对 mix_divpen 和 topdown_divpen 各随机抽 100k 条 slate，fake_zero 后过 GeMS 编码器得到 latent_mu（32 维原始空间），再经 min-max 归一化 + clamp 得到 true_actions（IQL 训练坐标）。

**运行**：
```bash
bash experiments/action_cloud/static_analysis/run.sh
```

#### 结果

**A. 原始潜空间 latent_mu（GeMS 编码器直接输出，未经归一化）**

| 指标 | mix | topdown | 对比 |
|------|-----|---------|------|
| L2 norm 均值 | 1.46 | 1.68 | topdown +15% |
| per_dim_std 均值 | 0.096 | 0.135 | topdown +41% |
| **有效 PC 数 (50%方差)** | 1 | 1 | — |
| **有效 PC 数 (80%方差)** | **2** | **4** | topdown 多 2 维 |
| 有效 PC 数 (95%方差) | 4 | 8 | topdown 多一倍 |
| **条件数** | **677** | **24205** | topdown 严重得多 |
| Top5 奇异值比 | [0.57, 0.14, 0.07, 0.05, 0.03] | [0.62, 0.10, 0.06, 0.03, 0.02] | — |

这两个数看似矛盾：topdown 的 PC80 更高（4 维 vs 2 维，云更"宽"），但条件数也更差（24205 vs 677）。原因是 topdown 的 latent_mu 在最宽的方向上特别宽，但在最窄的方向上也特别窄——宽的方向更宽、窄的方向更窄，导致整体更"扁平"而非"紧凑"。per_dim_std 均值更高（0.135 vs 0.096）说明单维尺度上 topdown 更分散。

**B. 归一化坐标 true_actions（IQL 实际使用的标签）**

| 指标 | mix | topdown | 对比 |
|------|-----|---------|------|
| L2 norm 均值 | 1.80 | 2.51 | topdown +39% |
| **z_center（到 dataset_center 的 L2）** | **1.80** | **2.51** | topdown 远 40% |
| z_center 中位数 | 1.79 | 2.52 | — |
| z_center p90 | 2.08 | 2.73 | — |
| **L2 质心散布（标签间典型距离）** | **1.03** | **0.96** | topdown 更紧 7% |
| per_dim_std 均值 | 0.184 | 0.162 | topdown 更窄 12% |
| **有效 PC 数 (80%方差)** | **5** | **6** | topdown 多 1 维 |
| 条件数 | 68 | 2509 | topdown 严重得多 |
| clamp 前越界率 | 0% | 0% | 都不越界 |

这里有一个关键的对比反转：raw latent_mu 里 topdown 的 per_dim_std 更高（0.135 vs 0.096），但 norm 后 true_actions 里 topdown 的 per_dim_std 反而更低（0.162 vs 0.184）。原因是 topdown 的 action_scale 比 mix 大（0.56 vs 0.36），归一化把 topdown 的云"压"得更扁。

z_center 1.80（mix）vs 2.51（topdown）——这和训练中观察到的结果一致：mix Actor 在 ~2.68，topdown 在 ~3.39。数据云本身在 topdown 就更偏，所以 Actor 天然站得更远。

**C. Item Embedding（GeMS 编码器的输入层）**

| 指标 | mix | topdown |
|------|-----|---------|
| L2 norm 均值 | 1.73 | 3.96 |
| **item 间余弦相似度均值** | **0.404** | **0.292** |
| cos_sim p10 | 0.248 | 0.161 |
| cos_sim p90 | 0.544 | 0.415 |
| 有效 PC 数 (80%方差) | 13 | 11 |

topdown 的 item embedding 区分度更好（cos_sim 0.29 vs 0.40）——item 之间更不相似。这应该有利于 encoder 输入的信息多样性。但 p80 维度反而 topdown 更低（11 vs 13）——可能是因为 topdown 的 item embedding L2 norm 差异太大（3.96±大方差 vs 1.73±小方差），导致少数"强"item 主导了方差。

#### 初步结论

1. topdown 的 raw latent_mu 比 mix 更多样（PC80 4 vs 2）但更"扁平"（条件数 24205 vs 677）
2. 归一化后 topdown 的云更紧（spread 0.96 vs 1.03）、更远（z_center 2.51 vs 1.80）
3. 这和训练观察一致：topdown SVD rank 更高但 eval reward 更差——更紧的云需要更精确的 Actor 输出才能命中好 slate
4. 归一化改变了两个环境云的相对"宽度"——topdown 的 action_scale 更大，把云压得更扁
5. clamp 前越界率 0%，说明归一化坐标下的"挤"不是因为截断造成的——是 GeMS 编码本身就把输出集中在中心区域

**详细数据**：`experiments/action_cloud/static_analysis/{mix,topdown}_divpen_b5_ideal_init_cloud_stats.json`

---

### 人话版本：动作云到底是什么，跟训练表现有什么关系

#### 动作云就是 Actor 的"靶子"

训练 IQL 的时候，每条数据是一个 slate（10 个 item ID）。这个 slate 喂给 GeMS，GeMS 吐出一个 32 维向量。一百万条 slate → 一百万个 32 维向量 → 这些点组成一团"云"。

Actor 的任务就是输出一个 32 维向量，这个向量要尽量落在云里——落在云里意味着 decode 出来的 slate 长得像训练数据里的推荐列表，用户会点，reward 高。所以**云就是靶子，Actor 要打中靶子才能得分**。

#### 这团云长什么样

实验 1 抽了 10 万个点，问这团云三个问题：

**云占了几维？**

32 维听起来很多，但实际上这些点只沿着两三个方向散开，其余方向几乎不动：

- mix 的云：80% 的方差只用 2 个方向就解释了。剩下 30 个方向是"扁的"——所有点在这些方向上几乎在同一个位置
- topdown 的云：用了 4 个方向。比 mix 好一点，但仍然是 4/32

类比：你有一个 32 层楼的大厦，但所有数据点只分布在 2-4 层。其他 28-30 层是空的。GeMS 把 slate 的信息"压扁"到了极少数维度上。

**云有多大？（点跟点之间离多远）**

- mix：云内部两个点的典型 L2 距离 ~1.03
- topdown：~0.96，比 mix 更紧一点

**云离原点有多远？**

- mix：云的中心离原点 ~1.80
- topdown：云的中心离原点 ~2.51，远了不少

#### 这跟训练里看到的现象怎么连起来

训练第一天（Step 500），Actor 随机初始化后输出的位置：
- mix 环境：Actor 站在 ~2.68（云中心在 1.80）→ 差了 0.88
- topdown 环境：Actor 站在 ~3.39（云中心在 2.51）→ 差了 0.88

两个环境里 Actor 离云的距离差不多（都差 ~0.88）。但 topdown 的云本身更紧（内部宽度 0.96 vs 1.03），所以 Actor 相对更"在外面"。

关键是——**整个训练过程中 Actor 几乎没动**。mix 的 z_center 从 2.68 变到 2.75，topdown 从 3.37 变到 3.42。Advantage 信号试图把 Actor 往云的方向拉，但力不够——拉了几千步就耗尽了，之后 Actor 就停在原地。

**结论：Actor 从一开始就在靶子外面，训练了 10 万步也从来没进去过。**

#### mix vs topdown 的差异怎么解释训练表现

- topdown 云的有效维度多（4 维 vs 2 维）：Actor 有更多方向可以走，所以 slate 多样性维持得更久（topdown_b8 的 unique_items=464，mix 最高才 87）
- topdown 云更紧更远（内部宽度 0.96 vs 1.03，离原点 2.51 vs 1.80）：靶子更小、更远，打中高 reward slate 更难，所以 topdown 的终值整体比 mix 差
- topdown 的 item embedding 区分度更好（cos_sim 0.29 vs 0.40）：item 之间更不像，encoder 能捕捉更多信息，所以有效维度更多

#### 一句话

GeMS 把一百万个 slate 压进了 32 维空间里一个很小的角落（有效 2-4 维，内部宽度 ~1）。Actor 一出生就站在这团云外面（距离 ~2.7），Advantage 信号拉了几下没拉动，随后就再也进不去了。topdown 的靶子比 mix 更小更远，所以训练表现更差。要解决这个问题，要么把靶子变大（改 GeMS 架构或 KL 权重），要么帮 Actor 走近靶子（改 Actor 输出归一化或加 BC 兜底）。

---

### 实验 2：lambda_KL 消融（重训 GeMS）

**目的**：验证"KL 正则化是云窄的主因"这个假设。

**方法**：用 `from_scratch` embedding 重训 GeMS VAE，扫 lambda_KL ∈ {0.01, 0.05, 0.1, 0.5, 1.0}。每个 λ_KL 训完后跑实验 1 的静态云分析，对比有效 PC 数、条件数、每维 std。

**命令模板**：
```bash
python scripts/train_gems.py \
    --item_embedds scratch \
    --dataset mix_divpen_b5 \
    --lambda_KL 0.05 \
    --lambda_click 1.0 \
    --seed 58407201 \
    --max_epochs 50
```

**预期**：λ_KL 越小，有效 PC 越多，条件数越小，云越宽。

**产出**：`experiments/action_cloud/kl_ablation.md` + 静态云指标对比表

---

### 实验 3：embedding 正交化（改 GeMS 输入）

**目的**：验证"item embedding 余弦相似度高导致 encoder 输入有效维低"这个假设。

**方法**：在 GeMS 训练时对 item embedding 加一个正交正则化损失（或直接用正交初始化），对比 encoder 输入 SVD 的变化和最终云的宽度变化。

**方案 A**：训练时加 Cosine Similarity Penalty
```python
cos_sim = (e_norm @ e_norm.T)  # [N, N]
loss_orth = (cos_sim - torch.eye(N)) ** 2  # 鼓励对角线=1, 非对角线=0
loss = vae_loss + lambda_orth * loss_orth.mean()
```

**方案 B**：用 SVD 正交化后的 embedding 初始化
对现有的 env embedding 做 SVD 后取 U（正交基）作为初始化。

**产出**：`experiments/action_cloud/embedding_ortho.md`

---

### 实验 4：用真实 click 替代 fake_zero（改 GeMS 编码输入）

**目的**：验证"fake_zero 导致确定性映射、挤压多样性"这个假设。

**方法**：不改 GeMS 模型，只改 IQL 标签生成时用真实 click 而非全 0。比较同一 slate 在不同 click 下产生的 latent 多样性。

**预期**：同一 slate 的真实 click 会因用户状态不同而变化→同一 slate 对应多个不同 latent→有效云变宽。旧报告估计多样性 +28%。

**产出**：`experiments/action_cloud/real_click.md`

---

### 实验 5：Actor 输出温度缩放（不改 GeMS，只在 IQL 端）

**目的**：最小的改动——把 Actor 输出乘以一个 < 1 的温度系数，让 z_center 从 2.7 降到 1.0。

**方法**：在 IQLAgent.act() 的反归一化时加温度：
```python
# 当前
latent_action = raw_action * self.action_scale + self.action_center
# 改为
latent_action = raw_action * self.action_scale * temperature + self.action_center
```

扫 temperature ∈ {0.3, 0.4, 0.5, 0.7}，看 eval reward 是否不再崩塌。

**产出**：`experiments/action_cloud/temperature_ablation.md`

---

## 4. 优先级

| 优先级 | 实验 | 为什么 |
|--------|------|--------|
| P0 | 实验 1（静态云表征） | 先知道云到底长什么样，建立 baseline。几小时跑完 |
| P0 | 实验 2（λ_KL 消融） | 最可能改变云宽度的 lever。需要重训 GeMS（一天） |
| P1 | 实验 5（Actor 温度） | 最小改动，不改 GeMS。如果有效，直接验证"云外→问题" |
| P1 | 实验 4（真实 click） | 不改模型结构，只改数据。验证多样性假设 |
| P2 | 实验 3（embedding 正交化） | 需要改 GeMS 训练代码，工作量大但治本 |

---

## 5. 目录结构

```
experiments/action_cloud/
├── README.md                    # 本文件
├── static_analysis.md           # 实验 1 结果
├── kl_ablation.md               # 实验 2 结果
├── embedding_ortho.md           # 实验 3 结果
├── real_click.md                # 实验 4 结果
├── temperature_ablation.md      # 实验 5 结果
├── kl_ablation/
│   ├── config.yaml
│   └── run.sh
└── temperature_ablation/
    ├── config.yaml
    └── run.sh
```
