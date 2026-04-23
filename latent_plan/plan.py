from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import Tensor

from latent_plan.model import WorldModel


def rollout_latent_trajectory(
    model: WorldModel,
    state: np.ndarray,
    actions: Iterable[int],
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Roll out latent states and predicted rewards for a fixed action sequence."""
    action_list = list(actions)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor(action_list, dtype=torch.long, device=device)

    latents: List[Tensor] = []
    rewards: List[Tensor] = []

    model.eval()
    with torch.no_grad():
        z = model.encode(state_tensor)
        latents.append(z.squeeze(0))
        for a in action_tensor:
            z = model.predict_next(z, a.view(1))
            r = model.predict_reward(z)
            latents.append(z.squeeze(0))
            rewards.append(r.squeeze(0))

    latent_array = torch.stack(latents, dim=0).cpu().numpy()
    reward_array = torch.stack(rewards, dim=0).cpu().numpy() if rewards else np.empty((0,))
    return latent_array, reward_array


def plan_action(
    model: WorldModel,
    state: np.ndarray,
    horizon: int = 12,
    num_sequences: int = 256,
    action_dim: int = 4,
    seed: int | None = None,
    device: str = "cpu",
    return_info: bool = False,
) -> int | Tuple[int, Dict[str, np.ndarray | List[int] | float]]:
    """
    Random-shooting planner in latent space.

    Samples random action sequences, rolls them out with the learned dynamics,
    sums predicted rewards, and selects the best first action.
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if num_sequences <= 0:
        raise ValueError("num_sequences must be > 0")

    rng = np.random.default_rng(seed)
    action_sequences = rng.integers(low=0, high=action_dim, size=(num_sequences, horizon), dtype=np.int64)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    sequence_tensor = torch.tensor(action_sequences, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        z0 = model.encode(state_tensor)  # [1, latent_dim]
        z = z0.repeat(num_sequences, 1)
        value = torch.zeros(num_sequences, device=device)

        for t in range(horizon):
            a_t = sequence_tensor[:, t]
            z = model.predict_next(z, a_t)
            value = value + model.predict_reward(z)

        best_idx = int(torch.argmax(value).item())
        best_sequence = action_sequences[best_idx].tolist()
        best_first_action = int(best_sequence[0])
        best_value = float(value[best_idx].item())

    if not return_info:
        return best_first_action

    imagined_latents, imagined_rewards = rollout_latent_trajectory(
        model=model,
        state=state,
        actions=best_sequence,
        device=device,
    )

    info: Dict[str, np.ndarray | List[int] | float] = {
        "best_sequence": best_sequence,
        "imagined_latents": imagined_latents,
        "imagined_rewards": imagined_rewards,
        "predicted_return": best_value,
    }
    return best_first_action, info

