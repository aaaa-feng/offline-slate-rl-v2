"""Flow Matching BC Agent.

Phase 1: Pure imitation learning on D4RL data.
No GeMS ranker, no IQL critic – just belief → flow matching → kNN → discrete slate.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F

from src.belief.gru import GRUBelief
from src.agents.diffusion_slate.velocity_net import VelocityNet
from src.agents.diffusion_slate.flow_policy import FlowPolicy


class FlowBCAgent:
    """Flow Matching Behavior Cloning Agent.

    Replaces TanhGaussian Actor + GeMS decoder with:
      belief_state → Flow Matching (Euler) → slate_embedding → dedup kNN → items
    """

    def __init__(self, config, embedding_table, device):
        self.config = config
        self.device = device
        self.total_it = 0

        # ---- Item embedding table (frozen) ----
        self.embedding_table = embedding_table                         # [1000, 20]
        # Phase 1: no normalization (train & infer in raw embedding space)
        # Phase 2+: train with (x0 - mean) / std, infer with *std + mean
        self.emb_mean = 0.0   # embedding_table.weight.mean().item()
        self.emb_std = 1.0    # embedding_table.weight.std().item()
        logging.info(f"Embedding raw: mean={embedding_table.weight.mean().item():.4f}, "
                     f"std={embedding_table.weight.std().item():.4f} | "
                     f"Phase 1: no normalization applied (emb_mean=0, emb_std=1)")

        # ---- GRU Belief (reuse existing, embed with item embedding for slate lookup) ----
        input_dim = config.rec_size * (config.item_embedd_dim + 1)    # 210
        # Wrap item embedding as ItemEmbeddings for GRUBelief (it needs nn.Module)
        from src.rankers.gems.embeddings import ItemEmbeddings
        item_emb_wrapper = ItemEmbeddings(
            num_items=config.num_items, item_embedd_dim=config.item_embedd_dim,
            device=device, weights=embedding_table.weight.data.clone())
        item_emb_wrapper.freeze()  # ItemEmbeddings has .freeze()

        logging.info("Initializing GRU belief encoder for Flow BC...")
        self.belief = GRUBelief(
            item_embeddings=item_emb_wrapper,
            belief_state_dim=config.belief_hidden_dim,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            ranker=None,
            device=device,
            belief_lr=0.0,
            hidden_layers_reduction=[],
            beliefs=["actor", "critic_v"],
            hidden_dim=config.belief_hidden_dim,
            input_dim=input_dim,
        )
        # Ensure all GRU embedding copies are frozen
        for module in self.belief.item_embeddings:
            self.belief.item_embeddings[module].embedd.requires_grad_(False)

        # ---- Velocity Network ----
        action_dim = config.rec_size * config.item_embedd_dim        # 200
        self.velocity_net = VelocityNet(
            state_dim=config.belief_hidden_dim,                       # 20
            action_dim=action_dim,
            hidden_dim=512,
            n_blocks=3,
        ).to(device)

        # ---- Flow Policy ----
        self.policy = FlowPolicy(
            velocity_net=self.velocity_net,
            embedding_table=self.embedding_table,
            flow_steps=getattr(config, 'flow_steps', 10),
            dedup_knn=getattr(config, 'flow_dedup_knn', 1) == 1,
            emb_mean=self.emb_mean,           # pass through for un-normalization
            emb_std=self.emb_std,
        )
        # Fix eval noise for reproducibility
        self.policy.set_eval_noise(torch.randn(1, action_dim))

        # ---- Optimizer ----
        self.optimizer = torch.optim.Adam(
            list(self.velocity_net.parameters())
            + list(self.belief.gru["actor"].parameters()),
            lr=config.actor_lr,
        )

        logging.info(f"FlowBCAgent initialized: action_dim={action_dim}, "
                     f"flow_steps={self.policy.flow_steps}, "
                     f"emb_mean={self.emb_mean:.4f}, emb_std={self.emb_std:.4f}")
        logging.info(f"  VelocityNet params: {sum(p.numel() for p in self.velocity_net.parameters()):,}")

    # ========================================================================
    # Training
    # ========================================================================

    def train(self, batch) -> dict:
        """One training step: flow matching BC loss."""
        self.total_it += 1

        # 1. Belief encoding (flattens episodes → [sum_seq, 20])
        states, _ = self.belief.forward_batch(batch)
        s = states["actor"]                                           # [~25k, 20]

        # 2. Target: slate IDs → item embeddings
        slate_ids = [item.cpu() for item in batch.obs["slate"]]
        slate_ids = torch.cat(slate_ids, dim=0).to(self.device)       # [~25k, 10]
        x0 = self.embedding_table(slate_ids).flatten(1)               # [~25k, 200]

        # 3. Sub-sample to avoid OOM (256 episodes ≈ 25k transitions)
        n = min(4096, x0.shape[0])
        idx = torch.randperm(x0.shape[0], device=self.device)[:n]
        x0, s = x0[idx], s[idx]

        # 4. Flow Matching loss: t=0 noise, t=1 data, velocity from noise→data
        t = torch.rand(n, 1, device=self.device)
        noise = torch.randn_like(x0)
        xt = (1 - t) * noise + t * x0                                # linear interpolation
        target_vel = x0 - noise                                       # true velocity field

        pred_vel = self.velocity_net(s, xt, t)
        loss = F.mse_loss(pred_vel, target_vel)

        # 5. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.velocity_net.parameters())
            + list(self.belief.gru["actor"].parameters()),
            1.0,
        )
        self.optimizer.step()

        return {"flow_loss": loss.item()}

    # ========================================================================
    # Inference
    # ========================================================================

    def act(self, obs, deterministic=True):
        """Generate slate from current observation.

        Args:
            obs: dict with 'slate' [10], 'clicks' [10]
            deterministic: if True, use fixed eval noise
        Returns:
            numpy array [10] of item IDs
        """
        slate_t = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
        clicks_t = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)
        obs_tensor = {"slate": slate_t, "clicks": clicks_t}

        belief_states = self.belief.forward(obs_tensor, done=False)
        s = belief_states["actor"].unsqueeze(0)                       # [1, 20]

        items = self.policy.sample(s, deterministic=deterministic)    # [1, 10]
        return items.squeeze(0).detach().cpu().numpy()                # [10]

    def reset_hidden(self):
        """Reset GRU hidden state for new episode."""
        dummy_obs = {
            "slate": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device),
            "clicks": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device),
        }
        self.belief.forward(dummy_obs, done=True)

    # ========================================================================
    # Save / Load
    # ========================================================================

    def state_dict(self):
        return {
            'velocity_net': self.velocity_net.state_dict(),
            'belief': {k: v.state_dict() for k, v in self.belief.gru.items()},
            'total_it': self.total_it,
        }

    def load_state_dict(self, sd):
        self.velocity_net.load_state_dict(sd['velocity_net'])
        for k, v in sd['belief'].items():
            self.belief.gru[k].load_state_dict(v)
        self.total_it = sd.get('total_it', 0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.load_state_dict(sd)
        logging.info(f"FlowBCAgent loaded from {path}")
