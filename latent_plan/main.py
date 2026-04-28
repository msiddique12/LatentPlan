from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_plan.env import GridPos, GridWorldEnv
from latent_plan.metrics import compute_episode_metrics
from latent_plan.model import WorldModel
from latent_plan.plan import plan_action
from latent_plan.train import TrainHistory, collect_random_transitions, save_train_history_csv, train_world_model
from latent_plan.visualize import (
    decode_latent_trajectory_to_positions,
    save_rollout_animation,
    save_trajectory_plot,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_action_sequence_episode(
    env: GridWorldEnv,
    actions: List[int],
    max_steps: int,
) -> List[GridPos]:
    env.reset()
    path = [env.get_position()]

    for action in actions[:max_steps]:
        _, _, done = env.step(int(action))
        path.append(env.get_position())
        if done:
            break
    return path


def _build_checkpoint_payload(
    model: WorldModel,
    env: GridWorldEnv,
    latent_dim: int,
    history: TrainHistory,
) -> Dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "latent_dim": latent_dim,
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
    }


def load_world_model_from_checkpoint(
    checkpoint_path: Path,
    expected_state_dim: int,
    expected_action_dim: int,
    device: str,
) -> Tuple[WorldModel, TrainHistory]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    num_dynamics_models = int(checkpoint.get("num_dynamics_models", 1))
    if state_dim != expected_state_dim or action_dim != expected_action_dim:
        raise ValueError(
            "Checkpoint environment dimensions do not match current env: "
            f"state_dim={state_dim} action_dim={action_dim}"
        )

    model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        num_dynamics_models=num_dynamics_models,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    total_losses = [float(v) for v in checkpoint.get("losses", [])]
    loss_breakdown = checkpoint.get("loss_breakdown", {})
    history = TrainHistory(
        total=total_losses,
        transition=[float(v) for v in loss_breakdown.get("transition", total_losses)],
        reward=[float(v) for v in loss_breakdown.get("reward", [0.0] * len(total_losses))],
        reconstruction=[float(v) for v in loss_breakdown.get("reconstruction", [0.0] * len(total_losses))],
        multistep=[float(v) for v in loss_breakdown.get("multistep", [0.0] * len(total_losses))],
    )
    return model, history


def run_demo(args: argparse.Namespace) -> Tuple[Path, Path]:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    env = GridWorldEnv()
    checkpoint_path = Path(args.checkpoint)
    goal_state = np.array([env.goal[0] / max(env.width - 1, 1), env.goal[1] / max(env.height - 1, 1)], dtype=np.float32)

    if checkpoint_path.exists() and not args.force_train:
        model, history = load_world_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            expected_state_dim=env.state_dim,
            expected_action_dim=env.n_actions,
            device=device,
        )
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        model = WorldModel(
            state_dim=env.state_dim,
            action_dim=env.n_actions,
            latent_dim=args.latent_dim,
            num_dynamics_models=args.num_dynamics_models,
        )
        transitions = collect_random_transitions(
            env=env,
            num_episodes=args.collect_episodes,
            horizon=args.collect_horizon,
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
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            _build_checkpoint_payload(model=model, env=env, latent_dim=args.latent_dim, history=history),
            checkpoint_path,
        )
        print(f"Saved checkpoint to: {checkpoint_path}")

    state = env.reset()
    _, info = plan_action(
        model=model,
        state=state,
        horizon=args.plan_horizon,
        num_sequences=args.num_sequences,
        action_dim=env.n_actions,
        seed=args.seed,
        device=device,
        method=args.planner,
        cem_iters=args.cem_iters,
        cem_elite_frac=args.cem_elite_frac,
        cem_alpha=args.cem_alpha,
        discount=args.discount,
        risk_penalty=args.risk_penalty,
        goal_state=goal_state,
        goal_bonus=args.goal_bonus,
        goal_tolerance=args.goal_tolerance,
        return_info=True,
    )
    imagined_latents = cast(np.ndarray, info["imagined_latents"])
    imagined_rewards = cast(np.ndarray, info["imagined_rewards"])
    imagined_uncertainty = cast(np.ndarray, info["imagined_uncertainty"])
    best_sequence = cast(List[int], info["best_sequence"])
    imagined_path = decode_latent_trajectory_to_positions(
        model=model,
        env=env,
        latent_trajectory=imagined_latents,
        device=device,
        use_decoder=not args.no_decoder_decode,
    )

    real_path = run_action_sequence_episode(
        env=env,
        actions=best_sequence,
        max_steps=args.eval_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = save_train_history_csv(history, output_dir / "train_history.csv")

    manifest_path = output_dir / "run_manifest.json"
    manifest_payload: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "main_demo",
        "args": vars(args),
        "env": {
            "width": env.width,
            "height": env.height,
            "start": env.start,
            "goal": env.goal,
            "obstacles": sorted(list(env.obstacles)),
        },
        "model": {
            "latent_dim": int(getattr(model, "encoder").net[-1].out_features),
            "num_dynamics_models": model.num_dynamics_models,
        },
        "device": device,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    trajectory_plot_path = save_trajectory_plot(
        env=env,
        real_trajectory=real_path,
        imagined_trajectory=imagined_path,
        output_path=output_dir / "trajectory_comparison.png",
        title=f"Real vs Imagined Trajectory ({args.planner})",
    )

    losses = history.total
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

    fig, ax = plt.subplots(figsize=(6, 3))
    steps = np.arange(1, len(imagined_rewards) + 1)
    ax.plot(steps, imagined_rewards, label="Predicted Reward", color="#1c7ed6", linewidth=2)
    if imagined_uncertainty.size > 0:
        ax.plot(steps, imagined_uncertainty, label="Uncertainty", color="#d9480f", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_title("Planning Diagnostics")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    diagnostics_plot_path = output_dir / "planning_diagnostics.png"
    fig.savefig(diagnostics_plot_path, dpi=150)
    plt.close(fig)

    metrics = compute_episode_metrics(
        imagined_path=imagined_path,
        real_path=real_path,
        goal=env.goal,
        predicted_return=float(cast(float, info["predicted_return"])),
    )
    metrics_payload: Dict[str, Any] = {"planner_method": args.planner, **metrics.to_dict()}
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if args.save_animation:
        animation_path = save_rollout_animation(
            env=env,
            real_trajectory=real_path,
            imagined_trajectory=imagined_path,
            output_path=output_dir / "rollout.gif",
            fps=args.animation_fps,
        )
        print(f"Saved animation to: {animation_path}")

    print(f"Saved trajectory plot to: {trajectory_plot_path}")
    print(f"Saved loss curve to: {loss_plot_path}")
    print(f"Saved planning diagnostics to: {diagnostics_plot_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved training history to: {history_csv_path}")
    print(f"Saved run manifest to: {manifest_path}")
    print(f"Alignment match ratio: {metrics.match_ratio:.3f}")
    print(f"Goal reached: {bool(metrics.goal_reached)}")
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
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--multistep-weight", type=float, default=0.5)
    parser.add_argument("--num-dynamics-models", type=int, default=1)
    parser.add_argument("--plan-horizon", type=int, default=12)
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--planner", type=str, default="random", choices=["random", "cem"])
    parser.add_argument("--cem-iters", type=int, default=4)
    parser.add_argument("--cem-elite-frac", type=float, default=0.1)
    parser.add_argument("--cem-alpha", type=float, default=0.7)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--risk-penalty", type=float, default=0.0)
    parser.add_argument("--goal-bonus", type=float, default=0.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--no-decoder-decode", action="store_true")
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--animation-fps", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default="outputs/world_model.pt")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Ignore checkpoint and retrain from scratch.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


if __name__ == "__main__":
    run_demo(build_parser().parse_args())
