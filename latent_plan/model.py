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


class Decoder(nn.Module):
    """Maps latent vectors back to normalized state space."""

    def __init__(self, latent_dim: int = 16, state_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class WorldModel(nn.Module):
    """Encoder + latent dynamics + reward predictor + decoder."""

    def __init__(
        self,
        state_dim: int = 2,
        latent_dim: int = 16,
        action_dim: int = 4,
        hidden_dim: int = 64,
        num_dynamics_models: int = 1,
    ) -> None:
        super().__init__()
        if num_dynamics_models <= 0:
            raise ValueError("num_dynamics_models must be > 0")

        self.encoder = Encoder(state_dim=state_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.dynamics = DynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        self.extra_dynamics = nn.ModuleList(
            [
                DynamicsModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim)
                for _ in range(num_dynamics_models - 1)
            ]
        )
        self.reward = RewardModel(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(latent_dim=latent_dim, state_dim=state_dim, hidden_dim=hidden_dim)

    @property
    def num_dynamics_models(self) -> int:
        return 1 + len(self.extra_dynamics)

    def predict_next_ensemble(self, z: Tensor, action: Tensor) -> Tensor:
        outputs = [self.dynamics(z, action)]
        outputs.extend(model(z, action) for model in self.extra_dynamics)
        return torch.stack(outputs, dim=0)

    def encode(self, state: Tensor) -> Tensor:
        return self.encoder(state)

    def predict_next(self, z: Tensor, action: Tensor) -> Tensor:
        return self.predict_next_ensemble(z, action).mean(dim=0)

    def predict_reward(self, z: Tensor) -> Tensor:
        return self.reward(z)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self.encode(state)
        z_next = self.predict_next(z, action)
        reward_pred = self.predict_reward(z_next)
        state_next_pred = self.decode(z_next)
        return z_next, reward_pred, state_next_pred
