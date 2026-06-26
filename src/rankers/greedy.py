"""
Greedy Slate Ranker: 迭代贪心选择。

从 GeMS rankers.py 提取。
每个位置依次选择与 proto-action 最相似的未选 item。
"""

import torch
from typing import Tuple

from .gems.ranker import Ranker
from .gems.argument_parser import MyParser


class GreedySlateRanker(Ranker):
    """
    Greedy slate generator: 迭代贪心选择，考虑累积效应
    
    核心思想：
    1. 迭代选择 items，每次选择使边际收益最大的 item
    2. 考虑已选 items 对后续选择的影响（累积分子/分母）
    3. 使用 mask 机制防止重复选择
    
    与 TopKRanker 的区别：
    - TopKRanker: 一次性选择 Top-K
    - GreedySlateRanker: 迭代选择，考虑累积效应
    
    Action space: [item_embedd_dim] (20-dim)
    
    参考：rl_slate_wolpertinger 项目的 GreedySlateGenerator
    """
    def __init__(
        self,
        item_embeddings: ItemEmbeddings,
        item_embedd_dim: int,
        rec_size: int,
        device: torch.device,
        s_no_click: float = -1.0,  # 无点击的基准分数
        **kwargs
    ):
        super().__init__(item_embeddings, item_embedd_dim, device, rec_size, **kwargs)
        self.s_no_click = s_no_click
    
    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim, 1  # 20
    
    def rank(self, action, clicked=None) -> torch.LongTensor:
        """
        使用贪心算法生成 slate
        
        Args:
            action: [batch_size, item_embedd_dim] - 用于计算 item scores
            clicked: 可选，已点击的 items
        
        Returns:
            [batch_size, rec_size] - slate of item IDs
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        batch_size = action.shape[0]
        slates = []
        
        for i in range(batch_size):
            # 计算所有 items 的评分（相似度）
            action_vec = action[i].unsqueeze(1)  # [item_embedd_dim, 1]
            scores = torch.matmul(
                self.item_embeddings.get_weights(),
                action_vec
            ).squeeze(1)  # [num_items]
            
            # 假设 Q-values 为 1（简化版）
            # 在实际应用中，可以从 Critic 网络获取 Q-values
            qvals = torch.ones_like(scores)
            
            # 贪心选择
            numerator = torch.tensor(0.0, device=self.device)
            denominator = torch.tensor(self.s_no_click, device=self.device)
            mask = torch.ones_like(qvals, dtype=torch.bool)
            
            slate = []
            for _ in range(self.rec_size):
                # 计算每个候选的边际收益
                # marginal_value = (累积奖励 + 新item奖励) / (累积评分 + 新item评分)
                marginal_value = (numerator + scores * qvals) / (denominator + scores)
                
                # 排除已选的 items
                marginal_value[~mask] = float('-inf')
                
                # 选择最大边际收益的 item
                k = torch.argmax(marginal_value)
                
                slate.append(k.item())
                mask[k] = False
                
                # 更新累积值
                numerator = numerator + scores[k] * qvals[k]
                denominator = denominator + scores[k]
            
            slates.append(torch.tensor(slate, device=self.device))
        
        return torch.stack(slates)
    
    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Inverse mapping: slate → action
        
        使用 slate 中 items 的平均 embedding
        """
        slate_embeddings = self.item_embeddings(slates)
        action = slate_embeddings.mean(dim=1)
        log_var = torch.full_like(action, -10.0)
        return action, log_var
