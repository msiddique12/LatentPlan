from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

from latent_plan.env import GridPos, GridWorldEnv


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    width: int
    height: int
    start: GridPos
    goal: GridPos
    obstacles: Sequence[GridPos]
    max_steps: int = 64


def create_env(spec: BenchmarkSpec) -> GridWorldEnv:
    return GridWorldEnv(
        width=spec.width,
        height=spec.height,
        start=spec.start,
        goal=spec.goal,
        obstacles=spec.obstacles,
        max_steps=spec.max_steps,
    )


def get_benchmark_specs(kind: Literal["easy", "hard", "all"] = "all") -> List[BenchmarkSpec]:
    easy = [
        BenchmarkSpec(
            name="open_field",
            width=7,
            height=7,
            start=(0, 0),
            goal=(6, 6),
            obstacles=[],
        ),
        BenchmarkSpec(
            name="default_blocks",
            width=7,
            height=7,
            start=(0, 0),
            goal=(6, 6),
            obstacles=[(2, 2), (2, 3), (3, 2), (4, 4)],
        ),
    ]
    hard = [
        BenchmarkSpec(
            name="zigzag_corridor",
            width=9,
            height=9,
            start=(0, 0),
            goal=(8, 8),
            obstacles=[
                (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
                (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
            ],
            max_steps=96,
        ),
        BenchmarkSpec(
            name="wall_gaps",
            width=9,
            height=9,
            start=(0, 8),
            goal=(8, 0),
            obstacles=[
                (2, y) for y in range(9) if y != 2
            ] + [
                (5, y) for y in range(9) if y != 6
            ],
            max_steps=96,
        ),
    ]

    if kind == "easy":
        return easy
    if kind == "hard":
        return hard
    if kind == "all":
        return easy + hard
    raise ValueError(f"Unknown benchmark kind: {kind}")

