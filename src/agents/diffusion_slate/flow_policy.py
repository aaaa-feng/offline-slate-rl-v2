"""FlowPolicy: Flow Matching sampling + dedup kNN decoding."""

import torch
import torch.nn as nn


class FlowPolicy:
    """Flow Matching strategy: noise → slate_embedding → dedup kNN → discrete items.

    Args:
        velocity_net: VelocityNet instance
        embedding_table: nn.Embedding [1000, 20], frozen
        flow_steps: number of Euler integration steps (default 10)
        dedup_knn: if True, greedily select items without repeats
        emb_mean: embedding mean (for un-normalization, default 0.0)
        emb_std: embedding std (for un-normalization, default 1.0)
    """
    def __init__(self, velocity_net, embedding_table, flow_steps=10,
                 dedup_knn=True, emb_mean=0.0, emb_std=1.0):
        self.velocity_net = velocity_net
        self.embedding_table = embedding_table       # [1000, 20], frozen
        self.num_items, self.item_dim = embedding_table.weight.shape
        self.action_dim = velocity_net.out.out_features
        if self.action_dim % self.item_dim != 0:
            raise ValueError(
                f"action_dim={self.action_dim} must be divisible by item_dim={self.item_dim}"
            )
        self.rec_size = self.action_dim // self.item_dim
        self.flow_steps = flow_steps
        self.dedup_knn = dedup_knn
        self.emb_mean = emb_mean
        self.emb_std = emb_std
        self._eval_noise = None                       # [1, action_dim], fixed for eval

    def set_eval_noise(self, noise):
        """Fix noise tensor [1, action_dim] for deterministic evaluation."""
        if noise.shape[-1] != self.action_dim:
            raise ValueError(
                f"eval noise dim={noise.shape[-1]} does not match action_dim={self.action_dim}"
            )
        self._eval_noise = noise

    @torch.no_grad()
    def sample(self, state, deterministic=True):
        """Generate discrete slates from belief state.

        Args:
            state: [B, 20] belief state
            deterministic: if True and eval_noise set, use fixed noise
        Returns:
            items: [B, 10] discrete item IDs (long tensor on state.device)
        """
        B = state.shape[0]
        device = state.device

        # Noise [B, action_dim]
        if deterministic and self._eval_noise is not None:
            x = self._eval_noise.to(device).expand(B, -1)
        else:
            x = torch.randn(B, self.action_dim, device=device)

        # Euler integration: t from 0 → 1
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((B, 1), i * dt, device=device)
            v = self.velocity_net(state, x, t)
            x = x + v * dt

        # Un-normalize (if training normalized)
        x = x * self.emb_std + self.emb_mean

        # kNN decoding
        emb = x.reshape(B, self.rec_size, self.item_dim)
        if self.dedup_knn:
            items = self._dedup_knn(emb, device)
        else:
            dists = torch.cdist(emb, self.embedding_table.weight)
            items = dists.argmin(dim=-1)
        return items

    def _dedup_knn(self, emb, device):
        """Greedy nearest-neighbor search, skipping already-selected items."""
        B = emb.shape[0]
        used = torch.zeros(B, self.num_items, dtype=torch.bool, device=device)
        items = torch.zeros(B, self.rec_size, dtype=torch.long, device=device)

        for pos in range(self.rec_size):
            dists = torch.cdist(emb[:, pos:pos+1, :],
                                self.embedding_table.weight)
            dists = dists.squeeze(1)
            dists[used] = float('inf')
            chosen = dists.argmin(dim=-1)
            items[:, pos] = chosen
            used.scatter_(1, chosen.unsqueeze(1), True)

        return items

    def knn_margin(self, emb):
        """Mean (2nd-NN dist − 1st-NN dist). Small margin = near decision boundary."""
        dists = torch.cdist(emb, self.embedding_table.weight)
        top2 = dists.topk(2, dim=-1, largest=False).values
        return (top2[:, :, 1] - top2[:, :, 0]).mean().item()
