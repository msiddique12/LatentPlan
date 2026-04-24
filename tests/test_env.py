import numpy as np

from latent_plan.env import GridWorldEnv


def test_movement_correctness() -> None:
    env = GridWorldEnv(
        width=5,
        height=5,
        start=(1, 1),
        goal=(4, 4),
        obstacles={(2, 1)},
        max_steps=20,
    )
    env.reset()

    _, _, _ = env.step(3)  # right, blocked by obstacle
    assert env.get_position() == (1, 1)

    _, _, _ = env.step(0)  # up
    assert env.get_position() == (1, 0)

    _, _, _ = env.step(1)  # down
    assert env.get_position() == (1, 1)


def test_boundary_conditions() -> None:
    env = GridWorldEnv(
        width=4,
        height=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=set(),
        max_steps=20,
    )
    state = env.reset()
    assert np.allclose(state, np.array([0.0, 0.0], dtype=np.float32))

    _, _, _ = env.step(0)  # up at top boundary
    assert env.get_position() == (0, 0)

    _, _, _ = env.step(2)  # left at left boundary
    assert env.get_position() == (0, 0)

