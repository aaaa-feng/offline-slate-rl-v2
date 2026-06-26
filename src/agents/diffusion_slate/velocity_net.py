"""Velocity Network for Flow Matching.

Input: state [B,20], x_t [B,200], t [B,1]
Output: predicted velocity field [B,200]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time steps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: [B, 1] → [B, dim]"""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float() * emb.unsqueeze(0)        # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: condition on state."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, dim)
        self.beta = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        return x * (1.0 + self.gamma(cond)) + self.beta(cond)


class ResidualBlock(nn.Module):
    """Residual block with FiLM conditioning."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film1 = FiLM(hidden_dim, cond_dim)
        self.film2 = FiLM(hidden_dim, cond_dim)

    def forward(self, h, cond):
        out = self.fc1(h)
        out = self.film1(out, cond)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.film2(out, cond)
        return h + out


class VelocityNet(nn.Module):
    """Flow Matching velocity field network.

    Predicts v(x_t, t, s) — the instantaneous velocity from noise toward data at time t.

    Args:
        state_dim: belief state dimension (20)
        action_dim: flattened slate_embedding dimension (10 × 20 = 200)
        hidden_dim: hidden layer dimension
        n_blocks: number of residual blocks with FiLM conditioning
    """
    def __init__(self, state_dim=20, action_dim=200, hidden_dim=512, n_blocks=3):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.x_proj = nn.Linear(action_dim, hidden_dim)
        self.t_proj = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, x_t, t):
        """
        Args:
            state: [B, 20] belief state
            x_t: [B, 200] noisy slate_embedding at time t
            t: [B, 1] flow time step ∈ [0, 1)
        Returns:
            v: [B, 200] predicted velocity field
        """
        s = self.state_proj(state)           # [B, hidden]
        x = self.x_proj(x_t)                 # [B, hidden]
        t_emb = self.t_proj(t)               # [B, hidden]

        h = x + t_emb                        # time conditioning

        for block in self.blocks:
            h = block(h, s)                  # FiLM with state

        h = h + x                            # residual
        return self.out(h)                   # [B, 200]
