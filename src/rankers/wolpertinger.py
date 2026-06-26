"""
Wolpertinger-style Rankers: kNN-based recommendation.

从 GeMS rankers.py 提取，包含:
- WolpertingerActor: 单 item proto-action 网络
- WolpertingerRanker: 单 item kNN ranker
- WolpertingerActorSlate: 多位置 proto-action 网络
- WolpertingerSlateRanker: 多位置 kNN ranker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .gems.ranker import Ranker
from .gems.argument_parser import MyParser


class WolpertingerActor(nn.Module):
    """
    Wolpertinger Actor 网络：生成 proto-action
    
    输入：state (GRU belief state)
    输出：proto-action (item embedding 空间中的连续向量)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(WolpertingerActor, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(Linear(prev_dim, dim))
            layers.append(ReLU())
            prev_dim = dim
        layers.append(Linear(prev_dim, action_dim))
        
        self.network = Sequential(*layers)
    
    def forward(self, state):
        """输出 proto-action"""
        return self.network(state)


class WolpertingerRanker(Ranker):
    """
    Wolpertinger-style ranker: Actor → proto-action → kNN → Top-K
    
    核心思想：
    1. Actor 网络生成 proto-action（原型动作）
    2. 在 item embeddings 中进行 kNN 搜索
    3. 从 k 个候选中选择 Top-rec_size
    
    与 TopKRanker 的区别：
    - TopKRanker: 直接相似度排序
    - WolpertingerRanker: 先 kNN 筛选，再选择
    
    Action space: [item_embedd_dim] (20-dim)
    """
    def __init__(
        self,
        item_embeddings: ItemEmbeddings,
        item_embedd_dim: int,
        rec_size: int,
        device: torch.device,
        k: int = 50,  # kNN 候选数量
        actor_hidden_dims: List[int] = None,
        state_dim: int = 20,  # GRU belief state 维度
        **kwargs
    ):
        super().__init__(item_embeddings, item_embedd_dim, device, rec_size, **kwargs)
        self.k = min(k, self.num_items)  # 确保 k 不超过 item 总数
        
        # 创建 Actor 网络（注意：在联合训练中，这个 Actor 不会被使用）
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        
        self.actor = WolpertingerActor(
            state_dim=state_dim,
            action_dim=item_embedd_dim,
            hidden_dims=actor_hidden_dims
        )
    
    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim, 1  # 20
    
    def rank(self, action, clicked=None) -> torch.LongTensor:
        """
        将 proto-action 解码为 slate
        
        Args:
            action: [batch_size, item_embedd_dim] - proto-action
            clicked: 可选，已点击的 items
        
        Returns:
            [batch_size, rec_size] - slate of item IDs
        """
        # 处理批次维度
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        batch_size = action.shape[0]
        slates = []
        
        for i in range(batch_size):
            proto_action = action[i]  # [item_embedd_dim]
            
            # kNN 搜索：计算与所有 items 的欧氏距离
            distances = torch.linalg.norm(
                self.item_embeddings.get_weights() - proto_action.unsqueeze(0),
                dim=1
            )
            
            # 选择距离最小的 k 个 items
            topk_indices = torch.argsort(distances)[:self.k]
            
            # 从 k 个候选中选择 Top-rec_size
            # 简化版：直接选择最近的 rec_size 个
            slate = topk_indices[:self.rec_size]
            slates.append(slate)
        
        return torch.stack(slates)
    
    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Inverse mapping: slate → proto-action
        
        使用 slate 中 items 的平均 embedding 作为 proto-action
        （与 TopKRanker 相同的策略）
        """
        slate_embeddings = self.item_embeddings(slates)  # [batch, rec_size, embedd_dim]
        proto_action = slate_embeddings.mean(dim=1)  # [batch, embedd_dim]
        log_var = torch.full_like(proto_action, -10.0)  # 确定性（低方差）
        return proto_action, log_var


class WolpertingerActorSlate(nn.Module):
    """
    Wolpertinger Slate Actor 网络：生成 proto-slate
    
    输入：state (GRU belief state)
    输出：proto-slate (rec_size 个 item embeddings)
    """
    def __init__(self, state_dim: int, action_dim: int, rec_size: int, hidden_dims: List[int]):
        super(WolpertingerActorSlate, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(Linear(prev_dim, dim))
            layers.append(ReLU())
            prev_dim = dim
        layers.append(Linear(prev_dim, action_dim * rec_size))
        
        self.network = Sequential(*layers)
    
    def forward(self, state):
        """输出 proto-slate"""
        return self.network(state)


class WolpertingerSlateRanker(Ranker):
    """
    Wolpertinger Slate ranker: Actor → proto-slate → 多位置 kNN → slate
    
    核心思想：
    1. Actor 网络生成 proto-slate（每个位置一个 proto-item）
    2. 对每个位置独立进行 kNN 搜索
    3. 选择每个位置最近的 item
    
    与 kHeadArgmaxRanker 的区别：
    - kHeadArgmaxRanker: 每个位置独立 argmax
    - WolpertingerSlateRanker: 每个位置独立 kNN
    
    Action space: [rec_size * item_embedd_dim] (200-dim)
    """
    def __init__(
        self,
        item_embeddings: ItemEmbeddings,
        item_embedd_dim: int,
        rec_size: int,
        device: torch.device,
        k: int = 50,
        actor_hidden_dims: List[int] = None,
        state_dim: int = 20,
        **kwargs
    ):
        super().__init__(item_embeddings, item_embedd_dim, device, rec_size, **kwargs)
        self.k = min(k, self.num_items)
        
        # 扩展 action_center/scale 以支持每个位置
        self.action_center = self.action_center.repeat(rec_size)
        self.action_scale = self.action_scale.repeat(rec_size)
        
        # 创建 Actor 网络
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        
        self.actor = WolpertingerActorSlate(
            state_dim=state_dim,
            action_dim=item_embedd_dim,
            rec_size=rec_size,
            hidden_dims=actor_hidden_dims
        )
    
    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim * self.rec_size, 1  # 200
    
    def rank(self, action, clicked=None) -> torch.LongTensor:
        """
        将 proto-slate 解码为 slate
        
        Args:
            action: [batch_size, rec_size * item_embedd_dim] - proto-slate
            clicked: 可选，已点击的 items
        
        Returns:
            [batch_size, rec_size] - slate of item IDs
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        batch_size = action.shape[0]
        slates = []
        
        for i in range(batch_size):
            # Reshape 为 [rec_size, item_embedd_dim]
            proto_slate = action[i].reshape(self.rec_size, self.item_embedd_dim)
            
            slate = []
            for pos in range(self.rec_size):
                proto_item = proto_slate[pos]
                
                # 对每个位置做 kNN
                distances = torch.linalg.norm(
                    self.item_embeddings.get_weights() - proto_item.unsqueeze(0),
                    dim=1
                )
                topk_indices = torch.argsort(distances)[:self.k]
                
                # 选择最近的 item
                slate.append(topk_indices[0])
            
            slates.append(torch.tensor(slate, device=self.device))
        
        return torch.stack(slates)
    
    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Inverse mapping: slate → proto-slate
        
        精确重构：每个位置的 embedding 按顺序排列
        （与 kHeadArgmaxRanker 相同的策略）
        """
        slate_embeddings = self.item_embeddings(slates)  # [batch, rec_size, embedd_dim]
        # Transpose 后 flatten: [batch, embedd_dim, rec_size] → [batch, embedd_dim * rec_size]
        proto_slate = slate_embeddings.transpose(1, 2).flatten(start_dim=1)
        log_var = torch.full_like(proto_slate, -10.0)
        return proto_slate, log_var


