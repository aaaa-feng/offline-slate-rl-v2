# offline-slate-rl-v2 施工计划

**创建日期**: 2026-06-06
**原则**: 老项目 `/data/liyuefeng/offline-slate-rl` 只读，不做任何删除或修改

---

## 1. 项目定位

从老项目提取核心代码和数据，构建一个干净的离线 slate 推荐 RL 实验项目。

- **主力环境**: mix_divpen (mixPBM) + topdown_divpen (tdPBM)
- **主力数据集**: v2_b5 (boredom=5) + v2_b3 (boredom=3)
- **主力算法**: IQL + BC（后续可加 CQL / TD3+BC / Decision Transformer）
- **全部 Ranker**: GeMS / TopK / kHeadArgmax / Wolpertinger / WolpertingerSlate / GreedySlate

---

## 2. 目标目录结构

```
offline-slate-rl-v2/
├── README.md                         # 快速开始指南
├── CONSTRUCTION_PLAN.md              # 本文件
├── requirements.txt
│
├── config.py                         # 唯一配置 dataclass（统一所有参数）
│
├── src/
│   ├── __init__.py
│   │
│   ├── env/                          # RecSim 模拟器（仅用于评估）
│   │   ├── __init__.py
│   │   ├── simulators.py             # TopicRec 模拟器
│   │   ├── sim_config.py             # 环境参数配置
│   │   └── modules/                  # PBM、diversity penalty 等子模块
│   │       ├── __init__.py
│   │       └── ...
│   │
│   ├── data/                         # 数据加载
│   │   ├── __init__.py
│   │   ├── slate_dataset.py          # OfflineSlateDataModule（GeMS 训练用）
│   │   ├── trajectory_buffer.py      # TrajectoryReplayBuffer（离线 RL 用）
│   │   └── collection/              # 数据采集代码（完整保留）
│   │       ├── __init__.py
│   │       ├── collector.py          # OfflineDataCollector
│   │       ├── model_loader.py       # ModelLoader（加载 SAC+GeMS ckpt）
│   │       ├── env_factory.py        # EnvironmentFactory
│   │       ├── formats.py            # SlateDataset, SlateTransition 等
│   │       ├── metrics.py            # SlateMetrics
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── merge_datasets.py
│   │           └── analyze_quality.py
│   │
│   ├── rankers/
│   │   ├── __init__.py
│   │   ├── gems/                     # GeMS VAE（核心）
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py         # ItemEmbeddings, MFEmbeddings
│   │   │   ├── ranker.py             # Ranker → AbstractGeMS → GeMS
│   │   │   ├── argument_parser.py    # MyParser
│   │   │   ├── data_utils.py         # Trajectory, SlateDataModule 等
│   │   │   └── matrix_factorization/ # BPR MF
│   │   │       ├── __init__.py
│   │   │       ├── models.py
│   │   │       └── ...
│   │   ├── topk.py                   # TopKRanker (+ kHeadArgmaxRanker)
│   │   ├── wolpertinger.py           # WolpertingerRanker + WolpertingerSlateRanker
│   │   └── greedy.py                 # GreedySlateRanker
│   │
│   ├── belief/
│   │   ├── __init__.py
│   │   └── gru.py                    # BeliefEncoder → GRUBelief
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseOfflineAgent（抽象基类）
│   │   ├── bc.py                     # Behavior Cloning
│   │   └── iql/
│   │       ├── __init__.py
│   │       ├── agent.py              # IQLAgent（从 2470 行裁到 ~600 行）
│   │       └── networks.py           # Actor / Critic / Value 网络
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # SwanlabLogger
│       ├── common.py                 # set_seed, soft_update
│       └── checkpoint.py             # GeMS ckpt 路径解析 + embedding 提取
│
├── scripts/
│   ├── train_gems.py                 # GeMS VAE 预训练入口
│   ├── train_agent.py                # 离线 RL 训练入口（--algo iql/bc）
│   ├── train_mf.py                   # BPR MF embedding 训练入口
│   ├── collect_data.py               # 数据采集入口
│   └── eval.py                       # 独立评估入口
│
├── checkpoints/
│   ├── gems/                         # GeMS 预训练 ckpt
│   └── agents/                       # IQL/BC/... 训练 ckpt
│
├── logs/
│   ├── gems/                         # GeMS 训练日志
│   └── agents/                       # IQL 训练日志
│
├── data/
│   ├── datasets/offline/
│   │   ├── mix_divpen/
│   │   │   ├── mix_divpen_v2_b5_data_d4rl.npz
│   │   │   ├── mix_divpen_v2_b5_oracle.npz
│   │   │   ├── mix_divpen_v2_b5.pt
│   │   │   ├── mix_divpen_v2_b5_meta.json
│   │   │   ├── mix_divpen_v2_b3_data_d4rl.npz
│   │   │   ├── mix_divpen_v2_b3_oracle.npz
│   │   │   ├── mix_divpen_v2_b3.pt
│   │   │   └── mix_divpen_v2_b3_meta.json
│   │   └── topdown_divpen/
│   │       ├── topdown_divpen_v2_b5_data_d4rl.npz
│   │       ├── topdown_divpen_v2_b5_oracle.npz
│   │       ├── topdown_divpen_v2_b5.pt
│   │       ├── topdown_divpen_v2_b5_meta.json
│   │       ├── topdown_divpen_v2_b3_data_d4rl.npz
│   │       ├── topdown_divpen_v2_b3_oracle.npz
│   │       ├── topdown_divpen_v2_b3.pt
│   │       └── topdown_divpen_v2_b3_meta.json
│   └── embeddings/
│       ├── item_embeddings_diffuse.pt    # 评估环境用（RecSim 模拟器内部）
│       └── mf/
│           ├── mf_mix_b5.pt
│           ├── mf_mix_b3.pt
│           ├── mf_topdown_b5.pt
│           └── mf_topdown_b3.pt
│
└── experiments/                     # 每个实验自包含
    └── <exp_name>/
        ├── config.yaml
        └── run.sh
```

---

## 3. 源码提取清单：逐文件操作

### 3.1 `src/env/` — RecSim 模拟器

| 新文件 | 来源 | 操作 |
|--------|------|------|
| `src/env/__init__.py` | — | 新建 |
| `src/env/simulators.py` | `src/envs/RecSim/simulators.py` | 复制；去掉 `from common.online.argument_parser` 依赖，改为本地 config |
| `src/env/sim_config.py` | `config/offline/env_params.py` | 提取 mix_divpen / topdown_divpen 的配置，写入新文件 |
| `src/env/modules/` | `src/envs/RecSim/modules/` | 复制整个目录 |

### 3.2 `src/data/` — 数据加载

| 新文件 | 来源 | 操作 |
|--------|------|------|
| `src/data/__init__.py` | — | 新建 |
| `src/data/slate_dataset.py` | `src/rankers/gems/offline_data_utils.py` | 复制；去掉 PL dependency 中不必要的部分 |
| `src/data/trajectory_buffer.py` | `src/common/offline/buffer.py` | 复制；清理 |
| `src/data/collection/__init__.py` | — | 新建 |
| `src/data/collection/collector.py` | `src/data_collection/offline_data_collection/collect_data.py` | 复制；去掉 model_tests 目录依赖 |
| `src/data/collection/model_loader.py` | `src/data_collection/offline_data_collection/core/model_loader.py` | 复制；仅保留 mix_divpen / topdown_divpen 环境配置 |
| `src/data/collection/env_factory.py` | `src/data_collection/offline_data_collection/core/environment_factory.py` | 复制 |
| `src/data/collection/formats.py` | `src/data_collection/offline_data_collection/core/data_formats.py` | 复制 |
| `src/data/collection/metrics.py` | `src/data_collection/offline_data_collection/core/metrics.py` | 复制 |
| `src/data/collection/utils/__init__.py` | — | 新建 |
| `src/data/collection/utils/merge_datasets.py` | `src/data_collection/offline_data_collection/utils/merge_datasets.py` | 复制 |
| `src/data/collection/utils/analyze_quality.py` | `src/data_collection/offline_data_collection/utils/analyze_quality.py` | 复制 |

### 3.3 `src/rankers/` — 全部 Ranker

| 新文件 | 来源 | 操作 |
|--------|------|------|
| `src/rankers/__init__.py` | — | 新建 |
| `src/rankers/gems/__init__.py` | — | 新建 |
| `src/rankers/gems/embeddings.py` | `src/rankers/gems/item_embeddings.py` | 复制；去掉旧 benchmark 路径硬编码 |
| `src/rankers/gems/ranker.py` | `src/rankers/gems/rankers.py` | 复制；**只保留 Ranker / AbstractGeMS / GeMS 三个类**；其余 TopK/Wolpertinger/Greedy 拆到独立文件 |
| `src/rankers/gems/argument_parser.py` | `src/rankers/gems/argument_parser.py` | 复制 |
| `src/rankers/gems/data_utils.py` | `src/rankers/gems/data_utils.py` | 复制 |
| `src/rankers/gems/matrix_factorization/` | `src/rankers/gems/matrix_factorization/` | 复制整个目录 |
| `src/rankers/topk.py` | `src/rankers/gems/rankers.py` (TopKRanker + kHeadArgmaxRanker) | **拆分**：从 rankers.py 提取 TopKRanker 和 kHeadArgmaxRanker 类 |
| `src/rankers/wolpertinger.py` | `src/rankers/gems/rankers.py` (WolpertingerActor + WolpertingerRanker + WolpertingerActorSlate + WolpertingerSlateRanker) | **拆分**：从 rankers.py 提取 4 个类 |
| `src/rankers/greedy.py` | `src/rankers/gems/rankers.py` (GreedySlateRanker) | **拆分**：从 rankers.py 提取 GreedySlateRanker 类 |

**拆分 rankers.py 的原因**：原文件 877 行，包含了 7 个类。拆开后每个文件 100-200 行，维护清晰。

### 3.4 `src/belief/` — GRU Belief

| 新文件 | 来源 | 操作 |
|--------|------|------|
| `src/belief/__init__.py` | — | 新建 |
| `src/belief/gru.py` | `src/belief_encoders/gru_belief.py` | 复制；去掉 `forward_batch_shared`（TD3+BC 残留）、去掉 `common.online.argument_parser` 依赖 |

### 3.5 `src/agents/` — 离线 RL 算法

| 新文件 | 来源 | 操作 |
|--------|------|------|
| `src/agents/__init__.py` | — | 新建 |
| `src/agents/base.py` | — | **新建**：定义 `BaseOfflineAgent` 抽象基类（train/select_action/save/load 接口） |
| `src/agents/bc.py` | — | **新建**：BC agent（简单实现，约 100 行） |
| `src/agents/iql/__init__.py` | — | 新建 |
| `src/agents/iql/agent.py` | `src/agents/offline/iql.py` | **大幅裁剪**：从 2470 行到约 600 行；去掉 main()/argparse（移到脚本）、去掉训练入口函数（移到脚本）、保留 IQLAgent 核心训练逻辑 + 关键诊断指标 |
| `src/agents/iql/networks.py` | `src/common/offline/networks.py` | 复制 |

### 3.6 `src/utils/` — 工具

| 新文件 | 来源 | 操作 |
|--------|------|------|
| `src/utils/__init__.py` | — | 新建 |
| `src/utils/logger.py` | `src/common/offline/logger.py` | 复制 |
| `src/utils/common.py` | `src/common/offline/utils.py` | 复制（set_seed, soft_update） |
| `src/utils/checkpoint.py` | `src/common/offline/checkpoint_utils.py` + `src/common/offline/ranker_factory.py` | **合并简化**：只保留 `resolve_gems_checkpoint()` + embedding 提取逻辑；去掉旧 benchmark（diffuse_* / focused_*）的配置 |

### 3.7 `config.py` — 统一配置

| 新文件 | 来源 | 操作 |
|--------|------|------|
| `config.py` | `config/offline/config.py` + `config/offline/paths.py` | **重写**：单一 dataclass，无旧 benchmark 残留，路径基于项目根目录 |

### 3.8 `scripts/` — 入口脚本

| 新文件 | 来源 | 操作 |
|--------|------|------|
| `scripts/train_gems.py` | `scripts/train_gems_offline.py` | 重写；~100 行，支持 `--item_embedds scratch/pretrained`；配置文件驱动 |
| `scripts/train_agent.py` | `src/agents/offline/iql.py` (main 部分) | 重写；~150 行，CLI → config → 选择 algo → 训练；配置文件驱动 |
| `scripts/train_mf.py` | `scripts/train_mf.py` | 复制；清理路径 |
| `scripts/collect_data.py` | — | 重写；~80 行，调用 `src/data/collection/` |
| `scripts/eval.py` | — | 新建；~80 行，加载 ckpt 独立评估 |

---

## 4. 数据复制清单

### 4.1 离线数据集 (D4RL .npz / Oracle .npz)

来源：`/data/liyuefeng/offline-slate-rl/data/datasets/offline/`

```
mix_divpen/
  mix_divpen_v2_b5_data_d4rl.npz        (~150MB)
  mix_divpen_v2_b5_oracle.npz           (~50MB)
  mix_divpen_v2_b5.pt                   (~10MB)
  mix_divpen_v2_b5_meta.json            (~2KB)
  mix_divpen_v2_b3_data_d4rl.npz        (~150MB)
  mix_divpen_v2_b3_oracle.npz           (~50MB)
  mix_divpen_v2_b3.pt                   (~10MB)
  mix_divpen_v2_b3_meta.json            (~2KB)

topdown_divpen/
  topdown_divpen_v2_b5_data_d4rl.npz    (~150MB)
  topdown_divpen_v2_b5_oracle.npz       (~50MB)
  topdown_divpen_v2_b5.pt              (~10MB)
  topdown_divpen_v2_b5_meta.json        (~2KB)
  topdown_divpen_v2_b3_data_d4rl.npz    (~150MB)
  topdown_divpen_v2_b3_oracle.npz       (~50MB)
  topdown_divpen_v2_b3.pt              (~10MB)
  topdown_divpen_v2_b3_meta.json        (~2KB)
```

总计约 16 个文件，~800MB。

### 4.2 Embedding 文件

来源：`/data/liyuefeng/offline-slate-rl/data/embeddings/`

```
item_embeddings_diffuse.pt              (~80KB)
mf/mf_mix_b5.pt                         (~80KB)
mf/mf_mix_b3.pt                         (~80KB)
mf/mf_topdown_b5.pt                     (~80KB)
mf/mf_topdown_b3.pt                     (~80KB)
```

### 4.3 不复制的内容

| 类别 | 说明 |
|------|------|
| `diffuse_mix/` `diffuse_topdown/` `diffuse_divpen/` | 旧 benchmark，diffuse 环境 |
| `focused_*` | 旧 benchmark，focused 环境 |
| `*epsilon-greedy*` | 临时实验用 |
| `gems_data_*/` | 旧 GeMS 数据目录 |
| `temp_verification/` | 临时测试 |
| `item_embeddings_focused.pt` | focused 环境 embedding |
| `diffuse_*.pt` `focused_*.pt` 等旧 MF embedding | 旧 benchmark MF |

---

## 5. 关键修复（在提取过程中一并完成）

### 5.1 GeMS 预训练支持 scratch 初始化

`scripts/train_gems.py` 增加：
```python
parser.add_argument("--item_embedds", type=str, default="scratch",
    choices=["scratch", "pretrained"])
```

当 `scratch` 时调用 `ItemEmbeddings.from_scratch()`，当 `pretrained` 时调用 `ItemEmbeddings.from_pretrained(path)`。

### 5.2 IQL CLI 暴露 GeMS embedding 模式

`scripts/train_agent.py` 增加：
```python
parser.add_argument("--gems_embedding_mode", type=str, default="scratch",
    choices=["scratch", "default", "mf_fixed"])
```

直接传给 `resolve_gems_checkpoint()`，控制加载哪个 GeMS ckpt。

### 5.3 IQL 默认 lambda_bc = 0.3

`config.py` 中 `lambda_bc: float = 0.3`（而非旧默认 0.5 或实验中的 0.0）。

### 5.4 config.py 无旧 benchmark 残留

`config.py` 只包含 `mix_divpen` / `topdown_divpen` 的环境配置，不包含 `diffuse_*` / `focused_*`。

---

## 6. config.py 设计（核心简化）

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent

@dataclass
class ExperimentConfig:
    """离线 Slate RL 实验统一配置"""

    # === 实验标识 ===
    experiment_name: str = "default"
    seed: int = 58407201
    device: str = "cuda"

    # === 环境 ===
    env_name: str = "mix_divpen"          # mix_divpen | topdown_divpen
    dataset_quality: str = "v2_b5"         # v2_b5 | v2_b3

    # === GeMS Ranker ===
    ranker_type: str = "gems"              # gems | topk | kheadargmax | wolpertinger | wolpertinger_slate | greedy
    gems_embedding_mode: str = "scratch"   # scratch | default | mf_fixed
    latent_dim: int = 32
    lambda_KL: float = 1.0
    lambda_click: float = 1.0

    # === 算法 ===
    algo: str = "iql"                      # iql | bc
    # IQL params
    beta: float = 3.0
    expectile: float = 0.8
    lambda_bc: float = 0.3
    gamma: float = 0.99
    iql_tau: float = 0.005
    # BC params (no extra)

    # === 训练 ===
    max_timesteps: int = 1_000_000
    batch_size: int = 256
    eval_freq: int = 500
    eval_episodes: int = 50
    final_eval_episodes: int = 100
    save_freq: int = 50_000
    log_freq: int = 500

    # === 网络 ===
    hidden_dim: int = 256
    n_hidden: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4

    # === GRU ===
    belief_hidden_dim: int = 20
    item_embedd_dim: int = 20
    num_items: int = 1000
    rec_size: int = 10
    gru_mode: str = "qv_shared_detach"

    # === Actor ===
    actor_type: str = "gaussian"

    # === SwanLab ===
    use_swanlab: bool = True
    swan_project: str = "Offline_Slate_RL_v2"
    swan_workspace: str = "Cliff"
    swan_mode: str = "cloud"

    # === 路径（自动推导） ===
    @property
    def dataset_path(self) -> Path:
        return PROJECT_ROOT / "data/datasets/offline" / self.env_name / \
               f"{self.env_name}_{self.dataset_quality}_data_d4rl.npz"

    @property
    def oracle_path(self) -> Path:
        return PROJECT_ROOT / "data/datasets/offline" / self.env_name / \
               f"{self.env_name}_{self.dataset_quality}_oracle.npz"

    @property
    def gems_checkpoint_path(self) -> Path:
        # 根据 gems_embedding_mode 解析 GeMS ckpt 路径
        mode_suffix = "" if self.gems_embedding_mode == "default" else f"_{self.gems_embedding_mode}"
        ckpt_name = f"GeMS_{self.env_name}_{self.dataset_quality}{mode_suffix}_latent32_beta{self.lambda_KL}_click{self.lambda_click}_seed{self.seed}.ckpt"
        return PROJECT_ROOT / "checkpoints/gems" / ckpt_name

    @property
    def item_embedds_path(self) -> Path:
        return PROJECT_ROOT / "data/embeddings/item_embeddings_diffuse.pt"
```

---

## 7. 实验模板

### 7.1 YAML 配置文件 (`experiments/beta_ablation/config.yaml`)

```yaml
experiment_name: beta_ablation_scratch
seed: 58407201

env_name: mix_divpen
dataset_quality: v2_b5

ranker_type: gems
gems_embedding_mode: scratch

algo: iql
beta: 3.0
expectile: 0.8
lambda_bc: 0.3

max_timesteps: 1000000
batch_size: 256
eval_freq: 500
```

### 7.2 运行脚本 (`experiments/beta_ablation/run.sh`)

```bash
#!/bin/bash
cd /data/liyuefeng/offline-slate-rl-v2
python scripts/train_agent.py --config experiments/beta_ablation/config.yaml
```

---

## 8. 施工阶段

### Phase 1: 骨架搭建（不涉及任何源码提取）

| # | 任务 | 产出 |
|---|------|------|
| 1.1 | 创建全部目录结构 | 空目录树 |
| 1.2 | 写 `README.md` | 快速开始指南 |
| 1.3 | 写 `requirements.txt` | 依赖列表 |
| 1.4 | 写 `config.py` | 统一配置 dataclass |

### Phase 2: 数据复制

| # | 任务 | 产出 |
|---|------|------|
| 2.1 | 复制离线数据集（16 个文件） | `data/datasets/offline/` |
| 2.2 | 复制 embedding 文件（5 个文件） | `data/embeddings/` |

### Phase 3: 基础工具层（无算法依赖）

| # | 任务 | 来源文件 |
|---|------|---------|
| 3.1 | `src/utils/common.py` | `common/offline/utils.py` |
| 3.2 | `src/utils/logger.py` | `common/offline/logger.py` |
| 3.3 | `src/utils/checkpoint.py` | `checkpoint_utils.py` + `ranker_factory.py` |

### Phase 4: 环境与数据层

| # | 任务 | 来源文件 |
|---|------|---------|
| 4.1 | `src/env/` (simulators + modules) | `envs/RecSim/` |
| 4.2 | `src/data/slate_dataset.py` | `rankers/gems/offline_data_utils.py` |
| 4.3 | `src/data/trajectory_buffer.py` | `common/offline/buffer.py` |
| 4.4 | `src/data/collection/` (6 个文件) | `data_collection/offline_data_collection/` |

### Phase 5: Ranker 层

| # | 任务 | 来源文件 |
|---|------|---------|
| 5.1 | `src/rankers/gems/` (embeddings, ranker, data_utils, mf) | `rankers/gems/` |
| 5.2 | 拆分 TopK → `src/rankers/topk.py` | `rankers.py` |
| 5.3 | 拆分 Wolpertinger → `src/rankers/wolpertinger.py` | `rankers.py` |
| 5.4 | 拆分 Greedy → `src/rankers/greedy.py` | `rankers.py` |

### Phase 6: Belief + Agent 层

| # | 任务 | 来源文件 |
|---|------|---------|
| 6.1 | `src/belief/gru.py` | `belief_encoders/gru_belief.py` |
| 6.2 | `src/agents/base.py` | 新建 |
| 6.3 | `src/agents/iql/networks.py` | `common/offline/networks.py` |
| 6.4 | `src/agents/iql/agent.py` | `agents/offline/iql.py`（大幅裁剪） |
| 6.5 | `src/agents/bc.py` | 新建 |

### Phase 7: 入口脚本

| # | 任务 | 产出 |
|---|------|------|
| 7.1 | `scripts/train_gems.py` | GeMS 预训练入口 |
| 7.2 | `scripts/train_agent.py` | 离线 RL 训练入口 |
| 7.3 | `scripts/train_mf.py` | MF 训练入口 |
| 7.4 | `scripts/collect_data.py` | 数据采集入口 |
| 7.5 | `scripts/eval.py` | 独立评估入口 |

### Phase 8: 关键修复集成

| # | 任务 |
|---|------|
| 8.1 | GeMS: `--item_embedds scratch` 支持 |
| 8.2 | IQL: `--gems_embedding_mode` CLI 暴露 |
| 8.3 | IQL: 默认 `lambda_bc=0.3` |

### Phase 9: 验证

| # | 任务 |
|---|------|
| 9.1 | GeMS scratch 预训练跑通（mix_divpen v2_b5） |
| 9.2 | IQL 训练跑通（用新的 scratch GeMS ckpt） |
| 9.3 | BC 训练跑通 |
| 9.4 | 评估跑通 |

### Phase 10: 实验模板

| # | 任务 |
|---|------|
| 10.1 | 写实验配置模板 + run.sh |
| 10.2 | 写 README（快速开始 + 实验流程） |

---

## 9. 老项目对照表：哪些对应关系

| 老项目路径 | 新项目路径 |
|-----------|-----------|
| `src/envs/RecSim/simulators.py` | `src/env/simulators.py` |
| `src/envs/RecSim/modules/` | `src/env/modules/` |
| `config/offline/env_params.py` | `src/env/sim_config.py` |
| `src/rankers/gems/item_embeddings.py` | `src/rankers/gems/embeddings.py` |
| `src/rankers/gems/rankers.py` (877行) | `src/rankers/gems/ranker.py` + `topk.py` + `wolpertinger.py` + `greedy.py` |
| `src/rankers/gems/offline_data_utils.py` | `src/data/slate_dataset.py` |
| `src/common/offline/buffer.py` | `src/data/trajectory_buffer.py` |
| `src/data_collection/offline_data_collection/` | `src/data/collection/` |
| `src/belief_encoders/gru_belief.py` | `src/belief/gru.py` |
| `src/agents/offline/iql.py` (2470行) | `src/agents/iql/agent.py` (约600行) + `scripts/train_agent.py` |
| `src/common/offline/networks.py` | `src/agents/iql/networks.py` |
| `src/common/offline/utils.py` | `src/utils/common.py` |
| `src/common/offline/logger.py` | `src/utils/logger.py` |
| `src/common/offline/checkpoint_utils.py` + `ranker_factory.py` | `src/utils/checkpoint.py` |
| `config/offline/config.py` + `paths.py` | `config.py` |
| `scripts/train_gems_offline.py` | `scripts/train_gems.py` |
| `scripts/train_mf.py` | `scripts/train_mf.py` |

---

## 10. 不被复制的内容（完整列表）

老项目中以下内容**不会**出现在新项目中：

| 类别 | 具体路径 |
|------|---------|
| 旧 benchmark 数据 | `data/datasets/offline/diffuse_*/`, `focused_*/` |
| 旧 embedding | `data/embeddings/item_embeddings_focused.pt`, `mf/diffuse_*.pt`, `mf/focused_*.pt`, `mf/mix_divpen_epsilon-greedy.pt` 等 |
| 在线 RL agent | `src/agents/online/` (SAC, SlateQ, REINFORCE, WolpertingerSAC) |
| 在线 RL 日志 | `experiments/logs/online/` |
| TD3+BC agent | `src/agents/offline/td3bc.py` (可在 Phase 后续加) |
| CQL agent | `src/agents/offline/cql.py` (可在 Phase 后续加) |
| 全部 motivation_test | `motivation_test/` |
| 旧实验脚本 | `motivation_test/logs/*/run_*.sh`（20+ 个散落的） |
| 旧实验日志 | `motivation_test/logs/*/run_logs/` |
| 旧实验分析 | `motivation_test/logs/*/extracted_metrics/`, `figures/`, `analysis_*/` |
| 旧 GeMS ckpt | `checkpoints/gems/offline/` （需重训，旧的不带 scratch 标记的都有 env 初始化污染） |
| 旧 IQL ckpt | `checkpoints/offline_rl/` |
| 在线 RL ckpt | `checkpoints/online_rl/` |
| 旧 benchmark 配置 | `config/offline/config.py` 中 diffuse_* / focused_* 部分 |
| 旧 benchmark 路径 | `config/offline/paths.py` 中旧 benchmark 路径 |
| model_tests | `src/data_collection/offline_data_collection/model_tests/` |
| 旧文档 | `document/` 除 PROJECT_MASTER_DOCUMENT.md 外的所有文件 |

---

## 11. 注意事项

1. **老项目只读**：所有操作都是 `cp`（复制），不做 `mv`（移动）或 `rm`（删除）
2. **import 路径统一**：新项目内部用 `from src.xxx import yyy`，不再需要 `sys.path.insert`
3. **每次 Phase 完成后验证**：每个 Phase 结束后检查文件是否完整、import 是否能解析
4. **Git**：新项目初始化为独立 git repo，老项目不加任何新 commit（除非用户主动要求）
