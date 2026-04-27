import numpy as np
import pytest

from latent_plan.model import WorldModel
from latent_plan.plan import plan_action


def test_planner_returns_valid_action() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    state = np.array([0.0, 0.0], dtype=np.float32)

    action = plan_action(
        model=model,
        state=state,
        horizon=6,
        num_sequences=32,
        action_dim=4,
        seed=0,
    )
    assert action in {0, 1, 2, 3}


def test_planner_trajectory_length() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    state = np.array([0.2, 0.4], dtype=np.float32)

    _, info = plan_action(
        model=model,
        state=state,
        horizon=7,
        num_sequences=32,
        action_dim=4,
        seed=1,
        return_info=True,
    )

    best_sequence = info["best_sequence"]
    imagined_latents = info["imagined_latents"]

    assert len(best_sequence) == 7
    assert imagined_latents.shape[0] == 8


def test_planner_cem_returns_valid_action() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    state = np.array([0.3, 0.3], dtype=np.float32)

    action = plan_action(
        model=model,
        state=state,
        horizon=6,
        num_sequences=64,
        action_dim=4,
        seed=2,
        method="cem",
        cem_iters=3,
        cem_elite_frac=0.2,
        cem_alpha=0.7,
    )
    assert action in {0, 1, 2, 3}


def test_planner_raises_on_action_dim_mismatch() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    state = np.array([0.1, 0.1], dtype=np.float32)

    with pytest.raises(ValueError, match="action_dim mismatch"):
        plan_action(
            model=model,
            state=state,
            horizon=4,
            num_sequences=16,
            action_dim=3,
        )


def test_planner_raises_on_invalid_method() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    state = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="method must be one of"):
        plan_action(
            model=model,
            state=state,
            horizon=4,
            num_sequences=16,
            action_dim=4,
            method="not_a_method",
        )
