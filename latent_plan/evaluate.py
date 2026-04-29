from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from latent_plan.calibration import (
    build_calibration_bins,
    collect_uncertainty_error_samples,
    summarize_calibration,
)
from latent_plan.env import GridWorldEnv
from latent_plan.main import run_action_sequence_episode, set_seed
from latent_plan.metrics import compute_episode_metrics, summarize_metric_dicts
from latent_plan.model import WorldModel
from latent_plan.plan import plan_action
from latent_plan.train import TransitionBatch, collect_random_transitions, train_world_model
from latent_plan.visualize import decode_latent_trajectory_to_positions


def train_model_for_seed(
    seed: int,
    epochs: int,
    collect_episodes: int,
    collect_horizon: int,
    latent_dim: int,
    num_dynamics_models: int,
) -> tuple[GridWorldEnv, WorldModel, TransitionBatch]:
    set_seed(seed)
    env = GridWorldEnv()
    model = WorldModel(
        state_dim=env.state_dim,
        action_dim=env.n_actions,
        latent_dim=latent_dim,
        num_dynamics_models=num_dynamics_models,
    )

    transitions = collect_random_transitions(
        env=env,
        num_episodes=collect_episodes,
        horizon=collect_horizon,
        seed=seed,
    )
    train_world_model(
        model=model,
        transitions=transitions,
        epochs=epochs,
        batch_size=128,
        learning_rate=1e-3,
        transition_weight=1.0,
        reward_weight=1.0,
        reconstruction_weight=1.0,
        multistep_weight=0.5,
        device="cpu",
    )
    return env, model, transitions


def evaluate_planner_on_model(
    env: GridWorldEnv,
    model: WorldModel,
    planner: str,
    seed: int,
    plan_horizon: int,
    num_sequences: int,
    discount: float,
    risk_penalty: float,
    goal_bonus: float,
    goal_tolerance: float,
) -> Dict[str, float]:

    state = env.reset()
    goal_state = np.array(
        [env.goal[0] / max(env.width - 1, 1), env.goal[1] / max(env.height - 1, 1)],
        dtype=np.float32,
    )
    _, info = plan_action(
        model=model,
        state=state,
        horizon=plan_horizon,
        num_sequences=num_sequences,
        action_dim=env.n_actions,
        seed=seed,
        method=planner,
        cem_iters=4,
        cem_elite_frac=0.1,
        cem_alpha=0.7,
        discount=discount,
        risk_penalty=risk_penalty,
        goal_state=goal_state,
        goal_bonus=goal_bonus,
        goal_tolerance=goal_tolerance,
        return_info=True,
    )

    imagined_path = decode_latent_trajectory_to_positions(
        model=model,
        env=env,
        latent_trajectory=info["imagined_latents"],  # type: ignore[arg-type]
        device="cpu",
        use_decoder=True,
    )
    real_path = run_action_sequence_episode(env=env, actions=info["best_sequence"], max_steps=plan_horizon)
    metrics = compute_episode_metrics(
        imagined_path=imagined_path,
        real_path=real_path,
        goal=env.goal,
        predicted_return=float(info["predicted_return"]),
    )
    return metrics.to_dict()


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "planner_comparison",
                "args": vars(args),
                "planners": ["random", "cem"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    planners = ["random", "cem"]
    results: Dict[str, List[Dict[str, float]]] = {planner: [] for planner in planners}
    uncertainty_chunks: List[np.ndarray] = []
    error_chunks: List[np.ndarray] = []

    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        env, model, transitions = train_model_for_seed(
            seed=seed,
            epochs=args.epochs,
            collect_episodes=args.collect_episodes,
            collect_horizon=args.collect_horizon,
            latent_dim=args.latent_dim,
            num_dynamics_models=args.num_dynamics_models,
        )

        calibration_samples = collect_uncertainty_error_samples(model=model, transitions=transitions, device="cpu")
        uncertainty_chunks.append(calibration_samples["uncertainty"])
        error_chunks.append(calibration_samples["error"])

        for planner in planners:
            metrics = evaluate_planner_on_model(
                env=env,
                model=model,
                planner=planner,
                seed=seed,
                plan_horizon=args.plan_horizon,
                num_sequences=args.num_sequences,
                discount=args.discount,
                risk_penalty=args.risk_penalty,
                goal_bonus=args.goal_bonus,
                goal_tolerance=args.goal_tolerance,
            )
            results[planner].append(metrics)
            print(
                f"seed={seed:3d} planner={planner:6s} "
                f"match={metrics['match_ratio']:.3f} goal={int(metrics['goal_reached'])}"
            )

    summary = {planner: summarize_metric_dicts(records) for planner, records in results.items()}
    all_uncertainty = np.concatenate(uncertainty_chunks, axis=0) if uncertainty_chunks else np.empty((0,))
    all_error = np.concatenate(error_chunks, axis=0) if error_chunks else np.empty((0,))
    bins = build_calibration_bins(all_uncertainty, all_error, num_bins=args.calibration_bins)
    calibration_stats = summarize_calibration(all_uncertainty, all_error, bins)

    summary_path = output_dir / "planner_comparison.json"
    summary_path.write_text(json.dumps({"summary": summary, "raw": results}, indent=2), encoding="utf-8")
    calibration_path = output_dir / "uncertainty_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "stats": calibration_stats,
                "bins": [item.to_dict() for item in bins],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = planners
    scores = [summary[p].get("match_ratio", {}).get("mean", 0.0) for p in planners]
    ax.bar(labels, scores, color=["#1c7ed6", "#2b8a3e"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean Match Ratio")
    ax.set_title("Planner Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "planner_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = [item.mean_uncertainty for item in bins if item.count > 0]
    y = [item.mean_error for item in bins if item.count > 0]
    ax.plot(x, y, "o-", color="#d9480f", label="Binned error vs uncertainty")
    max_val = max(x + y + [1e-8])
    ax.plot([0.0, max_val], [0.0, max_val], "--", color="#868e96", label="Ideal y=x")
    ax.set_xlabel("Predicted Uncertainty")
    ax.set_ylabel("Observed Error")
    ax.set_title("Uncertainty Calibration")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "uncertainty_calibration.png", dpi=150)
    plt.close(fig)

    print(f"Saved comparison summary to: {summary_path}")
    print(f"Saved uncertainty calibration to: {calibration_path}")
    print(f"Saved run manifest to: {manifest_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare planners across random seeds")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--collect-episodes", type=int, default=120)
    parser.add_argument("--collect-horizon", type=int, default=20)
    parser.add_argument("--plan-horizon", type=int, default=12)
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-dynamics-models", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--risk-penalty", type=float, default=0.05)
    parser.add_argument("--goal-bonus", type=float, default=0.1)
    parser.add_argument("--goal-tolerance", type=float, default=0.1)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
