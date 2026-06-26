"""
Behavior Cloning (BC) Agent for offline slate recommendation.

从 belief state → GeMS latent action 的确定性映射。
"""

import copy
import logging
from typing import Dict

import torch
import torch.nn.functional as F

from src.belief.gru import GRUBelief
from src.agents.iql.networks import DeterministicActor


class BCAgent:
    """Behavior Cloning Agent"""

    def __init__(self, action_dim: int, config, ranker_params: dict, ranker=None):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        self.total_it = 0
        self.ranker = ranker

        self.action_center = ranker_params['action_center'].to(self.device)
        self.action_scale = ranker_params['action_scale'].to(self.device)
        self.item_embeddings = ranker_params['item_embeddings']

        input_dim = config.rec_size * (config.item_embedd_dim + 1)

        logging.info("Initializing BC GRU belief encoder...")
        self.belief = GRUBelief(
            item_embeddings=self.item_embeddings,
            belief_state_dim=config.belief_hidden_dim,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            ranker=None,
            device=self.device,
            belief_lr=0.0,
            hidden_layers_reduction=[],
            beliefs=["actor", "critic_v"],
            hidden_dim=config.belief_hidden_dim,
            input_dim=input_dim,
        )

        for module in self.belief.item_embeddings:
            self.belief.item_embeddings[module].freeze()
        logging.info("BC: Item embeddings frozen")

        self.actor = DeterministicActor(
            state_dim=config.belief_hidden_dim,
            action_dim=action_dim,
            max_action=1.0,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["actor"].parameters()},
            {'params': self.actor.parameters()},
        ], lr=config.actor_lr)

        logging.info("BCAgent initialized")

    def train(self, batch) -> Dict[str, float]:
        self.total_it += 1

        states, next_states = self.belief.forward_batch(batch)
        s_actor = states["actor"]

        true_actions = torch.cat(batch.obs["action"], dim=0).to(self.device)
        true_actions = torch.clamp(true_actions, min=-0.99, max=0.99)

        pred_action = self.actor(s_actor, deterministic=True)[0] \
            if hasattr(self.actor, 'forward') else self.actor(s_actor)

        loss = F.mse_loss(pred_action, true_actions)

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.belief.gru["actor"].parameters()) + list(self.actor.parameters()),
            1.0,
        )
        self.actor_optimizer.step()

        return {"actor_loss": loss.item()}

    def select_action(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(state, deterministic=True)[0]
