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


def _validate_planner_args(
    model: WorldModel,
    horizon: int,
    num_sequences: int,
    action_dim: int,
) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if num_sequences <= 0:
        raise ValueError("num_sequences must be > 0")
    if action_dim <= 0:
        raise ValueError("action_dim must be > 0")

    model_action_dim = getattr(getattr(model, "dynamics", None), "action_dim", None)
    if model_action_dim is not None and action_dim != model_action_dim:
        raise ValueError(
            f"action_dim mismatch: planner uses {action_dim}, model expects {model_action_dim}"
        )


def _score_action_sequences(
    model: WorldModel,
    state: np.ndarray,
    action_sequences: np.ndarray,
    device: str,
    discount: float,
    risk_penalty: float,
    goal_state: np.ndarray | None,
    goal_bonus: float,
    goal_tolerance: float,
) -> torch.Tensor:
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1].")
    if risk_penalty < 0.0:
        raise ValueError("risk_penalty must be >= 0.")
    if goal_bonus < 0.0:
        raise ValueError("goal_bonus must be >= 0.")
    if goal_tolerance <= 0.0:
        raise ValueError("goal_tolerance must be > 0.")

    num_sequences, horizon = action_sequences.shape

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    sequence_tensor = torch.tensor(action_sequences, dtype=torch.long, device=device)
    goal_state_tensor = None
    if goal_state is not None:
        goal_state_tensor = torch.tensor(goal_state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        z0 = model.encode(state_tensor)
        z = z0.repeat(num_sequences, 1)
        value = torch.zeros(num_sequences, device=device)
        discount_scale = 1.0

        for t in range(horizon):
            a_t = sequence_tensor[:, t]
            if risk_penalty > 0.0 and model.num_dynamics_models > 1:
                z_ensemble = model.predict_next_ensemble(z, a_t)
                z = z_ensemble.mean(dim=0)
                uncertainty = z_ensemble.var(dim=0).mean(dim=1)
            else:
                z = model.predict_next(z, a_t)
                uncertainty = torch.zeros(num_sequences, device=device)

            reward_term = model.predict_reward(z)
            goal_term = torch.zeros(num_sequences, device=device)
            if goal_bonus > 0.0 and goal_state_tensor is not None and hasattr(model, "decode"):
                state_pred = model.decode(z)
                distance = torch.norm(state_pred - goal_state_tensor, dim=1)
                goal_term = goal_bonus * torch.exp(-distance / goal_tolerance)

            step_value = reward_term + goal_term - risk_penalty * uncertainty
            value = value + discount_scale * step_value
            discount_scale *= discount
    return value


def _plan_action_random(
    model: WorldModel,
    state: np.ndarray,
    horizon: int,
    num_sequences: int,
    action_dim: int,
    rng: np.random.Generator,
    device: str,
    discount: float,
    risk_penalty: float,
    goal_state: np.ndarray | None,
    goal_bonus: float,
    goal_tolerance: float,
) -> Tuple[List[int], float]:
    action_sequences = rng.integers(low=0, high=action_dim, size=(num_sequences, horizon), dtype=np.int64)
    model.eval()
    values = _score_action_sequences(
        model=model,
        state=state,
        action_sequences=action_sequences,
        device=device,
        discount=discount,
        risk_penalty=risk_penalty,
        goal_state=goal_state,
        goal_bonus=goal_bonus,
        goal_tolerance=goal_tolerance,
    )

    best_idx = int(torch.argmax(values).item())
    best_sequence = action_sequences[best_idx].tolist()
    best_value = float(values[best_idx].item())
    return best_sequence, best_value


def _plan_action_cem(
    model: WorldModel,
    state: np.ndarray,
    horizon: int,
    num_sequences: int,
    action_dim: int,
    rng: np.random.Generator,
    device: str,
    cem_iters: int,
    cem_elite_frac: float,
    cem_alpha: float,
    discount: float,
    risk_penalty: float,
    goal_state: np.ndarray | None,
    goal_bonus: float,
    goal_tolerance: float,
) -> Tuple[List[int], float]:
    if cem_iters <= 0:
        raise ValueError("cem_iters must be > 0")
    if not 0.0 < cem_elite_frac <= 1.0:
        raise ValueError("cem_elite_frac must be in (0, 1].")
    if not 0.0 < cem_alpha <= 1.0:
        raise ValueError("cem_alpha must be in (0, 1].")

    probs = np.full((horizon, action_dim), 1.0 / action_dim, dtype=np.float64)
    n_elite = max(1, int(num_sequences * cem_elite_frac))

    best_sequence: List[int] = [0] * horizon
    best_value = float("-inf")

    model.eval()
    for _ in range(cem_iters):
        sampled = np.empty((num_sequences, horizon), dtype=np.int64)
        for t in range(horizon):
            sampled[:, t] = rng.choice(action_dim, size=num_sequences, p=probs[t])

        values = _score_action_sequences(
            model=model,
            state=state,
            action_sequences=sampled,
            device=device,
            discount=discount,
            risk_penalty=risk_penalty,
            goal_state=goal_state,
            goal_bonus=goal_bonus,
            goal_tolerance=goal_tolerance,
        )
        values_np = values.cpu().numpy()

        elite_idx = np.argsort(values_np)[-n_elite:]
        elite = sampled[elite_idx]

        for t in range(horizon):
            counts = np.bincount(elite[:, t], minlength=action_dim).astype(np.float64)
            target_probs = counts / counts.sum()
            probs[t] = (1.0 - cem_alpha) * probs[t] + cem_alpha * target_probs
            probs[t] = np.clip(probs[t], 1e-3, None)
            probs[t] /= probs[t].sum()

        iter_best_idx = int(np.argmax(values_np))
        iter_best_value = float(values_np[iter_best_idx])
        if iter_best_value > best_value:
            best_value = iter_best_value
            best_sequence = sampled[iter_best_idx].tolist()

    return best_sequence, best_value


def plan_action(
    model: WorldModel,
    state: np.ndarray,
    horizon: int = 12,
    num_sequences: int = 256,
    action_dim: int = 4,
    seed: int | None = None,
    device: str = "cpu",
    method: str = "random",
    cem_iters: int = 4,
    cem_elite_frac: float = 0.1,
    cem_alpha: float = 0.7,
    discount: float = 0.98,
    risk_penalty: float = 0.0,
    goal_state: np.ndarray | None = None,
    goal_bonus: float = 0.0,
    goal_tolerance: float = 0.1,
    return_info: bool = False,
) -> int | Tuple[int, Dict[str, np.ndarray | List[int] | float | str]]:
    """Plan in latent space via random-shooting or CEM."""
    _validate_planner_args(
        model=model,
        horizon=horizon,
        num_sequences=num_sequences,
        action_dim=action_dim,
    )
    if method not in {"random", "cem"}:
        raise ValueError("method must be one of {'random', 'cem'}")

    rng = np.random.default_rng(seed)
    if method == "random":
        best_sequence, best_value = _plan_action_random(
            model=model,
            state=state,
            horizon=horizon,
            num_sequences=num_sequences,
            action_dim=action_dim,
            rng=rng,
            device=device,
            discount=discount,
            risk_penalty=risk_penalty,
            goal_state=goal_state,
            goal_bonus=goal_bonus,
            goal_tolerance=goal_tolerance,
        )
    else:
        best_sequence, best_value = _plan_action_cem(
            model=model,
            state=state,
            horizon=horizon,
            num_sequences=num_sequences,
            action_dim=action_dim,
            rng=rng,
            device=device,
            cem_iters=cem_iters,
            cem_elite_frac=cem_elite_frac,
            cem_alpha=cem_alpha,
            discount=discount,
            risk_penalty=risk_penalty,
            goal_state=goal_state,
            goal_bonus=goal_bonus,
            goal_tolerance=goal_tolerance,
        )

    best_first_action = int(best_sequence[0])

    if not return_info:
        return best_first_action

    imagined_latents, imagined_rewards = rollout_latent_trajectory(
        model=model,
        state=state,
        actions=best_sequence,
        device=device,
    )

    info: Dict[str, np.ndarray | List[int] | float | str] = {"best_sequence": best_sequence}
    info["imagined_latents"] = imagined_latents
    info["imagined_rewards"] = imagined_rewards
    info["predicted_return"] = best_value; info["planner_method"] = method
    return best_first_action, info
