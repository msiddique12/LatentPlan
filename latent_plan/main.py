from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_plan.env import GridPos, GridWorldEnv
from latent_plan.model import WorldModel
from latent_plan.plan import plan_action
from latent_plan.train import collect_random_transitions, train_world_model
from latent_plan.visualize import decode_latent_trajectory_to_positions, save_trajectory_plot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_mpc_episode(
    env: GridWorldEnv,
    model: WorldModel,
    horizon: int,
    num_sequences: int,
    max_steps: int,
    device: str,
    seed: int,
) -> List[GridPos]:
    state = env.reset()
    path = [env.get_position()]
    step_seed = seed

    for _ in range(max_steps):
        action = plan_action(
            model=model,
            state=state,
            horizon=horizon,
            num_sequences=num_sequences,
            action_dim=env.n_actions,
            seed=step_seed,
            device=device,
            return_info=False,
        )
        step_seed += 1
        state, _, done = env.step(action)
        path.append(env.get_position())
        if done:
            break
    return path


def run_demo(args: argparse.Namespace) -> Tuple[Path, Path]:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    env = GridWorldEnv()
    model = WorldModel(state_dim=env.state_dim, action_dim=env.n_actions, latent_dim=args.latent_dim)

    transitions = collect_random_transitions(
        env=env,
        num_episodes=args.collect_episodes,
        horizon=args.collect_horizon,
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

    state = env.reset()
    _, info = plan_action(
        model=model,
        state=state,
        horizon=args.plan_horizon,
        num_sequences=args.num_sequences,
        action_dim=env.n_actions,
        seed=args.seed,
        device=device,
        return_info=True,
    )
    imagined_latents = info["imagined_latents"]
    imagined_path = decode_latent_trajectory_to_positions(
        model=model,
        env=env,
        latent_trajectory=imagined_latents,  # type: ignore[arg-type]
        device=device,
    )

    real_path = run_mpc_episode(
        env=env,
        model=model,
        horizon=args.plan_horizon,
        num_sequences=args.num_sequences,
        max_steps=args.eval_steps,
        device=device,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_plot_path = save_trajectory_plot(
        env=env,
        real_trajectory=real_path,
        imagined_trajectory=imagined_path,
        output_path=output_dir / "trajectory_comparison.png",
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses, color="#1971c2", linewidth=2)
    ax.set_title("World Model Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    loss_plot_path = output_dir / "loss_curve.png"
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved trajectory plot to: {trajectory_plot_path}")
    print(f"Saved loss curve to: {loss_plot_path}")
    return trajectory_plot_path, loss_plot_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatentPlan demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--collect-episodes", type=int, default=400)
    parser.add_argument("--collect-horizon", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plan-horizon", type=int, default=12)
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


if __name__ == "__main__":
    run_demo(build_parser().parse_args())

