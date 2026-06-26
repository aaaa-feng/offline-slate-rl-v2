"""
TopK Ranker and kHeadArgmax Ranker.

从 GeMS rankers.py 提取，依赖 Ranker 基类和 ItemEmbeddings。
"""

import torch
from typing import Tuple

from .gems.ranker import Ranker
from .gems.argument_parser import MyParser


class TopKRanker(Ranker):
    '''
        Retrieves the k items closest to the latent action.
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.modules = []

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[Ranker.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim, 1

    def get_random_action(self) -> torch.FloatTensor:
        return self.action_center + self.action_scale * (torch.rand(self.item_embedd_dim, device = self.device) - 0.5)

    def rank(self, action, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be in the space of item embeddings.
        '''
        with torch.inference_mode():
            # Handle batch dimension: action shape [batch_size, item_embedd_dim] or [item_embedd_dim]
            if action.dim() == 1:
                action_vec = action.unsqueeze(1)  # [item_embedd_dim, 1]
            else:
                action_vec = action.squeeze(0).unsqueeze(1)  # [batch_size, item_embedd_dim] -> [item_embedd_dim, 1]

            similarity = torch.matmul(self.item_embeddings.get_weights(), action_vec).squeeze(1)  # [num_items]
            #similarity /= torch.linalg.vector_norm(similarity, dim = 1)
        if clicked is None:
            return torch.topk(similarity, k = self.rec_size, sorted = True)[1]
        else:
            unique, counts = torch.cat([torch.arange(self.num_items, device = self.device), clicked]).unique(return_counts = True)
            return unique[counts == 1][torch.topk(similarity[unique[counts == 1]], k = self.rec_size, sorted = True)[1]]

    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
            Inverse mapping: slate → action (approximate for TopKRanker).

            For TopKRanker, the inverse mapping is not unique (many actions can produce
            the same Top-K slate). We use simple average of item embeddings as a
            representative action, avoiding position weighting to prevent training
            target drift.

            Args:
                slates: [batch_size, rec_size] or [batch_size, traj_len, rec_size]
                clicks: Not used, kept for interface compatibility

            Returns:
                actions: [batch_size, item_embedd_dim]
                log_var: [batch_size, item_embedd_dim] (set to -10.0 for deterministic)
        '''
        # Handle batch of trajectories
        if len(slates.shape) == 3:
            slates = slates.flatten(end_dim=1)

        batch_size = slates.shape[0]

        # Get embeddings for all items in slates: [batch_size, rec_size, item_embedd_dim]
        slate_embeddings = self.item_embeddings.embedd(slates)

        # Simple average (no position weighting to avoid training target drift)
        actions = slate_embeddings.mean(dim=1)  # [batch_size, item_embedd_dim]

        # Log variance set to -10.0 (deterministic, exp(-10) ≈ 0)
        log_var = torch.full_like(actions, -10.0)

        return actions, log_var

class kHeadArgmaxRanker(TopKRanker):
    '''
        Retrieves the closest item for each slot of the slate
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.action_center = self.action_center.repeat(self.rec_size)
        self.action_scale = self.action_scale.repeat(self.rec_size)

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[TopKRanker.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim * self.rec_size, 1

    def get_random_action(self) -> torch.FloatTensor:
        return self.action_center + self.action_scale * (torch.rand(self.item_embedd_dim * self.rec_size, device = self.device) - 0.5)

    def rank(self, action, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be of size item_embedd_dim * rec_size.
        '''
        with torch.inference_mode():
            similarity = torch.matmul(self.item_embeddings.get_weights(), action.reshape(self.item_embedd_dim, self.rec_size))
            #similarity /= torch.linalg.vector_norm(similarity, dim = 1)
        if clicked is None:
            return torch.argmax(similarity, dim = 0)
        else:
            unique, counts = torch.cat([torch.arange(self.num_items, device = self.device), clicked]).unique(return_counts = True)
            return unique[counts == 1][torch.argmax(similarity[unique[counts == 1], :], dim = 0)]

    def run_inference(self, slates, clicks=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
            Inverse mapping: slate → action (exact reconstruction for kHeadArgmaxRanker).

            For kHeadArgmaxRanker, the inverse mapping is exact because each position
            independently selects an item. The action format must match rank()'s expectation:
            action.reshape(item_embedd_dim, rec_size) where each column is a position's embedding.

            Args:
                slates: [batch_size, rec_size] or [batch_size, traj_len, rec_size]
                clicks: Not used, kept for interface compatibility

            Returns:
                actions: [batch_size, item_embedd_dim * rec_size]
                log_var: [batch_size, item_embedd_dim * rec_size] (set to -10.0 for deterministic)
        '''
        # Handle batch of trajectories
        if len(slates.shape) == 3:
            slates = slates.flatten(end_dim=1)

        batch_size = slates.shape[0]

        # Get embeddings for all items: [batch_size, rec_size, item_embedd_dim]
        slate_embeddings = self.item_embeddings.embedd(slates)

        # Transpose to [batch_size, item_embedd_dim, rec_size] then flatten
        # This ensures that after reshape(item_embedd_dim, rec_size), each column is a position's embedding
        actions = slate_embeddings.transpose(1, 2).flatten(start_dim=1)  # [batch_size, item_embedd_dim * rec_size]

        # Log variance set to -10.0 (deterministic, exp(-10) ≈ 0)
        log_var = torch.full_like(actions, -10.0)

        return actions, log_var

