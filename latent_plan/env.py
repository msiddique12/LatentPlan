from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    state: np.ndarray
    reward: float
    done: bool


class GridWorldEnv:
    """Deterministic 2D gridworld with obstacles and a goal."""

    ACTIONS: Dict[int, GridPos] = {
        0: (0, -1),  # up
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
    }

    def __init__(
        self,
        width: int = 7,
        height: int = 7,
        start: GridPos = (0, 0),
        goal: GridPos = (6, 6),
        obstacles: Iterable[GridPos] | None = None,
        max_steps: int = 64,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.obstacles: Set[GridPos] = set(obstacles or {(2, 2), (2, 3), (3, 2), (4, 4)})

        self.n_actions = len(self.ACTIONS)
        self.state_dim = 2

        self._position: GridPos = self.start
        self._steps = 0

        self._validate_layout()

    def _validate_layout(self) -> None:
        if not self._in_bounds(self.start):
            raise ValueError("Start must be inside the grid.")
        if not self._in_bounds(self.goal):
            raise ValueError("Goal must be inside the grid.")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Start/goal cannot be on an obstacle.")

    def _in_bounds(self, pos: GridPos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _normalize_state(self, pos: GridPos) -> np.ndarray:
        x, y = pos
        x_norm = x / max(self.width - 1, 1)
        y_norm = y / max(self.height - 1, 1)
        return np.array([x_norm, y_norm], dtype=np.float32)

    def _transition(self, pos: GridPos, action: int) -> GridPos:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Expected one of {list(self.ACTIONS)}")

        dx, dy = self.ACTIONS[action]
        candidate = (pos[0] + dx, pos[1] + dy)
        if not self._in_bounds(candidate) or candidate in self.obstacles:
            return pos
        return candidate

    def reset(self) -> np.ndarray:
        self._position = self.start
        self._steps = 0
        return self._normalize_state(self._position)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self._steps += 1
        self._position = self._transition(self._position, action)

        reached_goal = self._position == self.goal
        timeout = self._steps >= self.max_steps
        done = reached_goal or timeout

        reward = 1.0 if reached_goal else -0.01
        return self._normalize_state(self._position), reward, done

    def get_position(self) -> GridPos:
        return self._position

    def denormalize_state(self, state: np.ndarray) -> GridPos:
        x = int(np.clip(round(float(state[0]) * max(self.width - 1, 1)), 0, self.width - 1))
        y = int(np.clip(round(float(state[1]) * max(self.height - 1, 1)), 0, self.height - 1))
        return (x, y)

    def valid_positions(self) -> List[GridPos]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.obstacles
        ]

