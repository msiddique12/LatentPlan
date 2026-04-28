from __future__ import annotations

import argparse
import csv
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
    action_2: np.ndarray
    next_states_2: np.ndarray
    has_next_2: np.ndarray


@dataclass
class TrainHistory:
    total: List[float]
    transition: List[float]
    reward: List[float]
    reconstruction: List[float]
    multistep: List[float]


class TransitionDataset(Dataset[Dict[str, Tensor]]):
    def __init__(self, transitions: TransitionBatch) -> None:
        self.states = torch.from_numpy(transitions.states).float()
        self.actions = torch.from_numpy(transitions.actions).long()
        self.next_states = torch.from_numpy(transitions.next_states).float()
        self.rewards = torch.from_numpy(transitions.rewards).float()
        self.action_2 = torch.from_numpy(transitions.action_2).long()
        self.next_states_2 = torch.from_numpy(transitions.next_states_2).float()
        self.has_next_2 = torch.from_numpy(transitions.has_next_2).float()

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "next_state": self.next_states[idx],
            "reward": self.rewards[idx],
            "action_2": self.action_2[idx],
            "next_state_2": self.next_states_2[idx],
            "has_next_2": self.has_next_2[idx],
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
    action_2: List[int] = []
    next_states_2: List[np.ndarray] = []
    has_next_2: List[float] = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_states: List[np.ndarray] = [state]
        episode_actions: List[int] = []
        episode_rewards: List[float] = []
        for _ in range(horizon):
            action = int(rng.integers(env.n_actions))
            next_state, reward, done = env.step(action)

            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_states.append(next_state)
            state = next_state
            if done:
                break

        steps = len(episode_actions)
        for t in range(steps):
            states.append(episode_states[t])
            actions.append(episode_actions[t])
            next_states.append(episode_states[t + 1])
            rewards.append(episode_rewards[t])

            if t + 1 < steps:
                has_next_2.append(1.0)
                action_2.append(episode_actions[t + 1])
                next_states_2.append(episode_states[t + 2])
            else:
                has_next_2.append(0.0)
                action_2.append(0)
                next_states_2.append(episode_states[t + 1])

    return TransitionBatch(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        next_states=np.asarray(next_states, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        action_2=np.asarray(action_2, dtype=np.int64),
        next_states_2=np.asarray(next_states_2, dtype=np.float32),
        has_next_2=np.asarray(has_next_2, dtype=np.float32),
    )


def train_world_model(
    model: WorldModel,
    transitions: TransitionBatch,
    epochs: int = 150,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    transition_weight: float = 1.0,
    reward_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    multistep_weight: float = 0.5,
    device: str = "cpu",
) -> TrainHistory:
    model.to(device)
    dataset = TransitionDataset(transitions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss()

    total_losses: List[float] = []
    transition_losses: List[float] = []
    reward_losses: List[float] = []
    reconstruction_losses: List[float] = []
    multistep_losses: List[float] = []
    for epoch in range(epochs):
        running_total = 0.0
        running_transition = 0.0
        running_reward = 0.0
        running_recon = 0.0
        running_multistep = 0.0
        for batch in loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            next_state = batch["next_state"].to(device)
            reward_target = batch["reward"].to(device)
            action_2 = batch["action_2"].to(device)
            next_state_2 = batch["next_state_2"].to(device)
            has_next_2 = batch["has_next_2"].to(device)

            z = model.encode(state)
            z_next_ensemble = model.predict_next_ensemble(z, action)
            z_next_pred = z_next_ensemble.mean(dim=0)
            reward_pred = model.predict_reward(z_next_pred)
            next_state_pred = model.decode(z_next_pred)

            with torch.no_grad():
                z_next_target = model.encode(next_state)
                z_next_2_target = model.encode(next_state_2)

            transition_loss = mse(z_next_ensemble, z_next_target.unsqueeze(0).expand_as(z_next_ensemble))
            transition_loss = transition_loss * transition_weight
            reward_loss = mse(reward_pred, reward_target) * reward_weight
            reconstruction_loss = mse(next_state_pred, next_state) * reconstruction_weight
            two_step_loss = torch.tensor(0.0, device=device)

            if multistep_weight > 0.0:
                mask = has_next_2 > 0.5
                if mask.any():
                    z_next_2_pred = model.predict_next(z_next_pred[mask], action_2[mask])
                    two_step_loss = mse(z_next_2_pred, z_next_2_target[mask]) * multistep_weight

            loss = transition_loss + reward_loss + reconstruction_loss + two_step_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = state.size(0)
            running_total += loss.item() * batch_n
            running_transition += transition_loss.item() * batch_n
            running_reward += reward_loss.item() * batch_n
            running_recon += reconstruction_loss.item() * batch_n
            running_multistep += two_step_loss.item() * batch_n

        avg_loss = running_total / len(dataset)
        avg_transition = running_transition / len(dataset)
        avg_reward = running_reward / len(dataset)
        avg_reconstruction = running_recon / len(dataset)
        avg_multistep = running_multistep / len(dataset)

        total_losses.append(avg_loss)
        transition_losses.append(avg_transition)
        reward_losses.append(avg_reward)
        reconstruction_losses.append(avg_reconstruction)
        multistep_losses.append(avg_multistep)

        if (epoch + 1) % max(epochs // 10, 1) == 0:
            print(
                f"Epoch {epoch + 1:4d}/{epochs}: total={avg_loss:.6f} "
                f"transition={avg_transition:.6f} reward={avg_reward:.6f} "
                f"recon={avg_reconstruction:.6f} multistep={avg_multistep:.6f}"
            )

    return TrainHistory(
        total=total_losses,
        transition=transition_losses,
        reward=reward_losses,
        reconstruction=reconstruction_losses,
        multistep=multistep_losses,
    )


def save_train_history_csv(history: TrainHistory, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total", "transition", "reward", "reconstruction", "multistep"])
        for idx, total in enumerate(history.total):
            writer.writerow(
                [
                    idx + 1,
                    total,
                    history.transition[idx],
                    history.reward[idx],
                    history.reconstruction[idx],
                    history.multistep[idx],
                ]
            )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LatentPlan world model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--multistep-weight", type=float, default=0.5)
    parser.add_argument("--num-dynamics-models", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="outputs/world_model.pt")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    env = GridWorldEnv()
    model = WorldModel(
        state_dim=env.state_dim,
        action_dim=env.n_actions,
        latent_dim=args.latent_dim,
        num_dynamics_models=args.num_dynamics_models,
    )

    transitions = collect_random_transitions(
        env=env,
        num_episodes=args.episodes,
        horizon=args.horizon,
        seed=args.seed,
    )
    history = train_world_model(
        model=model,
        transitions=transitions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        transition_weight=args.transition_weight,
        reward_weight=args.reward_weight,
        reconstruction_weight=args.reconstruction_weight,
        multistep_weight=args.multistep_weight,
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
            "num_dynamics_models": model.num_dynamics_models,
            "losses": history.total,
            "loss_breakdown": {
                "transition": history.transition,
                "reward": history.reward,
                "reconstruction": history.reconstruction,
                "multistep": history.multistep,
            },
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to: {ckpt_path}")
    history_path = save_train_history_csv(history, ckpt_path.with_name("train_history.csv"))
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()
