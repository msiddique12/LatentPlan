import numpy as np

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

