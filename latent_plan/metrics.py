from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from latent_plan.env import GridPos


@dataclass
class EpisodeMetrics:
    match_ratio: float
    goal_reached: float
    imagined_len: float
    real_len: float
    predicted_return: float
    final_goal_distance: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_episode_metrics(
    imagined_path: Sequence[GridPos],
    real_path: Sequence[GridPos],
    goal: GridPos,
    predicted_return: float,
) -> EpisodeMetrics:
    shared_len = min(len(imagined_path), len(real_path))
    if shared_len == 0:
        return EpisodeMetrics(
            match_ratio=0.0,
            goal_reached=0.0,
            imagined_len=0.0,
            real_len=0.0,
            predicted_return=predicted_return,
            final_goal_distance=float("inf"),
        )

    match_count = sum(1 for i in range(shared_len) if imagined_path[i] == real_path[i])
    goal_reached = 1.0 if real_path[-1] == goal else 0.0
    final_x, final_y = real_path[-1]
    goal_x, goal_y = goal
    final_goal_distance = float(abs(final_x - goal_x) + abs(final_y - goal_y))

    return EpisodeMetrics(
        match_ratio=match_count / shared_len,
        goal_reached=goal_reached,
        imagined_len=float(len(imagined_path)),
        real_len=float(len(real_path)),
        predicted_return=predicted_return,
        final_goal_distance=final_goal_distance,
    )


def summarize_metric_dicts(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}
    keys = sorted(records[0].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary

