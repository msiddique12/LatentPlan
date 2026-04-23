from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.colors import ListedColormap

from latent_plan.env import GridPos, GridWorldEnv
from latent_plan.model import WorldModel


def plot_gridworld(env: GridWorldEnv, ax: plt.Axes | None = None) -> plt.Axes:
    """Render grid, obstacles, start, and goal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    grid = np.zeros((env.height, env.width), dtype=np.int32)
    for x, y in env.obstacles:
        grid[y, x] = 1
    sx, sy = env.start
    gx, gy = env.goal
    grid[sy, sx] = 2
    grid[gy, gx] = 3

    cmap = ListedColormap(["#f8f9fa", "#495057", "#4dabf7", "#51cf66"])
    ax.imshow(grid, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="#adb5bd", linewidth=0.6)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def decode_latent_trajectory_to_positions(
    model: WorldModel,
    env: GridWorldEnv,
    latent_trajectory: np.ndarray,
    device: str = "cpu",
) -> List[GridPos]:
    """
    Approximate latent -> grid position by nearest encoded valid grid cell.
    This avoids adding a decoder while still making trajectory overlays interpretable.
    """
    valid_positions = env.valid_positions()
    valid_states = np.asarray(
        [
            [x / max(env.width - 1, 1), y / max(env.height - 1, 1)]
            for x, y in valid_positions
        ],
        dtype=np.float32,
    )

    model.eval()
    with torch.no_grad():
        state_tensor = torch.tensor(valid_states, dtype=torch.float32, device=device)
        ref_latents = model.encode(state_tensor).cpu().numpy()

    decoded_positions: List[GridPos] = []
    for latent in latent_trajectory:
        distance = np.sum((ref_latents - latent[None, :]) ** 2, axis=1)
        nearest_idx = int(np.argmin(distance))
        decoded_positions.append(valid_positions[nearest_idx])

    return decoded_positions


def _to_xy(path: Sequence[GridPos]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray([p[0] for p in path], dtype=np.float32) + 0.5
    ys = np.asarray([p[1] for p in path], dtype=np.float32) + 0.5
    return xs, ys


def save_trajectory_plot(
    env: GridWorldEnv,
    real_trajectory: Sequence[GridPos],
    imagined_trajectory: Sequence[GridPos],
    output_path: str | Path,
    title: str = "Real vs Imagined Trajectory",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_gridworld(env, ax=ax)

    if imagined_trajectory:
        x_i, y_i = _to_xy(imagined_trajectory)
        ax.plot(x_i, y_i, "o--", color="#1c7ed6", linewidth=2, markersize=4, label="Imagined")
    if real_trajectory:
        x_r, y_r = _to_xy(real_trajectory)
        ax.plot(x_r, y_r, "o-", color="#e03131", linewidth=2, markersize=4, label="Real")

    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_rollout_animation(
    env: GridWorldEnv,
    real_trajectory: Sequence[GridPos],
    imagined_trajectory: Sequence[GridPos],
    output_path: str | Path,
    fps: int = 3,
) -> Path:
    """Optional lightweight animation of real and imagined rollout."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_gridworld(env, ax=ax)
    real_line, = ax.plot([], [], "o-", color="#e03131", linewidth=2, markersize=4, label="Real")
    imag_line, = ax.plot([], [], "o--", color="#1c7ed6", linewidth=2, markersize=4, label="Imagined")
    ax.legend(loc="upper left")

    max_frames = max(len(real_trajectory), len(imagined_trajectory))

    def init() -> tuple:
        real_line.set_data([], [])
        imag_line.set_data([], [])
        return real_line, imag_line

    def animate(frame: int) -> tuple:
        if frame < len(real_trajectory):
            x_r, y_r = _to_xy(real_trajectory[: frame + 1])
            real_line.set_data(x_r, y_r)
        if frame < len(imagined_trajectory):
            x_i, y_i = _to_xy(imagined_trajectory[: frame + 1])
            imag_line.set_data(x_i, y_i)
        return real_line, imag_line

    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=max_frames,
        init_func=init,
        interval=int(1000 / fps),
        blit=True,
    )
    ani.save(output_path, fps=fps)
    plt.close(fig)
    return output_path

