from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    """Maps low-dimensional state vectors to latent space."""

    def __init__(self, state_dim: int = 2, latent_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class DynamicsModel(nn.Module):
    """Predicts next latent state from current latent and action."""

    def __init__(self, latent_dim: int = 16, action_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        if action.dtype != torch.long:
            action = action.long()
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=self.action_dim).float()
        inputs = torch.cat([z, action_one_hot], dim=-1)
        return self.net(inputs)


class RewardModel(nn.Module):
    """Predicts scalar reward from latent state."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(-1)


class WorldModel(nn.Module):
    """Encoder + latent dynamics + reward predictor."""

    def __init__(
        self,
        state_dim: int = 2,
        latent_dim: int = 16,
        action_dim: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(state_dim=state_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.dynamics = DynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        self.reward = RewardModel(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def encode(self, state: Tensor) -> Tensor:
        return self.encoder(state)

    def predict_next(self, z: Tensor, action: Tensor) -> Tensor:
        return self.dynamics(z, action)

    def predict_reward(self, z: Tensor) -> Tensor:
        return self.reward(z)

    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(state)
        z_next = self.predict_next(z, action)
        reward_pred = self.predict_reward(z_next)
        return z_next, reward_pred

