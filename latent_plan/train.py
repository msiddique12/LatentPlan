from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LatentPlan world model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="outputs/world_model.pt")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    env = GridWorldEnv()
    model = WorldModel(state_dim=env.state_dim, action_dim=env.n_actions, latent_dim=args.latent_dim)

    transitions = collect_random_transitions(
        env=env,
        num_episodes=args.episodes,
        horizon=args.horizon,
        seed=args.seed,
    )
    losses = train_world_model(
        model=model,
        transitions=transitions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": args.latent_dim,
            "action_dim": env.n_actions,
            "state_dim": env.state_dim,
            "losses": losses,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
