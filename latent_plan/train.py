from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from latent_plan.env import GridWorldEnv
from latent_plan.model import WorldModel


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray


class TransitionDataset(Dataset[Dict[str, Tensor]]):
    def __init__(self, transitions: TransitionBatch) -> None:
        self.states = torch.from_numpy(transitions.states).float()
        self.actions = torch.from_numpy(transitions.actions).long()
        self.next_states = torch.from_numpy(transitions.next_states).float()
        self.rewards = torch.from_numpy(transitions.rewards).float()

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "next_state": self.next_states[idx],
            "reward": self.rewards[idx],
        }


def collect_random_transitions(
    env: GridWorldEnv,
    num_episodes: int = 400,
    horizon: int = 32,
    seed: int = 0,
) -> TransitionBatch:
    rng = np.random.default_rng(seed)

    states: List[np.ndarray] = []
    actions: List[int] = []
    next_states: List[np.ndarray] = []
    rewards: List[float] = []

    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(horizon):
            action = int(rng.integers(env.n_actions))
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)

            state = next_state
            if done:
                break

    return TransitionBatch(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        next_states=np.asarray(next_states, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
    )


def train_world_model(
    model: WorldModel,
    transitions: TransitionBatch,
    epochs: int = 150,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    model.to(device)
    dataset = TransitionDataset(transitions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss()

    epoch_losses: List[float] = []
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            next_state = batch["next_state"].to(device)
            reward_target = batch["reward"].to(device)

            z = model.encode(state)
            z_next_pred = model.predict_next(z, action)
            reward_pred = model.predict_reward(z_next_pred)

            with torch.no_grad():
                z_next_target = model.encode(next_state)

            transition_loss = mse(z_next_pred, z_next_target)
            reward_loss = mse(reward_pred, reward_target)
            loss = transition_loss + reward_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * state.size(0)

        avg_loss = running_loss / len(dataset)
        epoch_losses.append(avg_loss)
        if (epoch + 1) % max(epochs // 10, 1) == 0:
            print(f"Epoch {epoch + 1:4d}/{epochs}: loss={avg_loss:.6f}")

    return epoch_losses

