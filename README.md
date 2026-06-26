# offline-slate-rl-v2

离线 Slate 推荐强化学习实验项目（v2 重构版）。

从旧项目 `offline-slate-rl` 提取核心代码，去除历史包袱，专注于 mix_divpen / topdown_divpen 两个 benchmark。

## 快速开始

### 环境

```bash
conda activate gems
pip install -r requirements.txt
```

### 1. 训练 GeMS VAE（随机初始化，推荐 scratch 模式）

```bash
python scripts/train_gems.py \
    --item_embedds scratch \
    --dataset mix_divpen_v2_b5 \
    --latent_dim 32 \
    --lambda_KL 1.0 \
    --lambda_click 1.0 \
    --seed 58407201 \
    --max_epochs 50
```

Checkpoint 保存到 `checkpoints/gems/`。

### 2. 训练离线 RL Agent

```bash
# IQL
python scripts/train_agent.py \
    --algo iql \
    --env_name mix_divpen \
    --dataset_quality v2_b5 \
    --gems_embedding_mode scratch \
    --beta 3.0 \
    --lambda_bc 0.3 \
    --seed 58407201

# BC baseline
python scripts/train_agent.py \
    --algo bc \
    --env_name mix_divpen \
    --dataset_quality v2_b5 \
    --gems_embedding_mode scratch \
    --seed 58407201
```

### 3. 用 YAML 配置跑实验

```bash
python scripts/train_agent.py --config experiments/beta_ablation/config.yaml
```

## 项目结构

```
offline-slate-rl-v2/
├── README.md
├── config.py                  # 统一配置 dataclass
├── requirements.txt
├── src/
│   ├── env/                   # RecSim 模拟器（评估用）
│   ├── data/                  # 数据加载 + 数据采集
│   ├── rankers/               # GeMS VAE + TopK + Wolpertinger + Greedy
│   ├── belief/                # GRU Belief Encoder
│   ├── agents/                # IQL + BC（可扩展）
│   └── utils/                 # 日志、通用工具
├── scripts/                   # 入口脚本
├── data/                      # 数据集 + embeddings
├── checkpoints/               # 模型 ckpt
├── logs/                      # 训练日志
└── experiments/               # 实验配置（每个实验一个子目录）
```

## 支持的 Ranker

| Ranker | `--ranker_type` | 说明 |
|--------|----------------|------|
| GeMS (VAE) | `gems` | VAE-based, latent action space |
| TopK | `topk` | kNN retrieval from action embedding |
| kHeadArgmax | `kheadargmax` | Position-independent TopK |
| Wolpertinger | `wolpertinger` | Single-item kNN with Actor |
| WolpertingerSlate | `wolpertinger_slate` | Multi-position kNN |
| GreedySlate | `greedy` | Iterative greedy selection |

## 支持的算法

| 算法 | `--algo` | 说明 |
|------|---------|------|
| IQL | `iql` | Implicit Q-Learning |
| BC | `bc` | Behavior Cloning (baseline) |

## GeMS Embedding 模式

| 模式 | `--gems_embedding_mode` | 说明 |
|------|------------------------|------|
| scratch | `scratch` | 随机初始化，从数据中学习（推荐） |
| default | `default` | 从 env embedding 初始化（不推荐） |
| mf_fixed | `mf_fixed` | MF embedding 冻结 |

## 与旧项目的关系

旧项目 `/data/liyuefeng/offline-slate-rl` 保持不变，所有历史代码和实验日志均可查阅。

本项目的源码从旧项目提取并清理，不依赖旧项目。
