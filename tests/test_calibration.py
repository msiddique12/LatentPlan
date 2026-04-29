import numpy as np

from latent_plan.calibration import build_calibration_bins, collect_uncertainty_error_samples, summarize_calibration
from latent_plan.env import GridWorldEnv
from latent_plan.model import WorldModel
from latent_plan.train import collect_random_transitions


def test_build_calibration_bins_counts() -> None:
    uncertainty = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    error = np.array([0.2, 0.1, 0.35, 0.45], dtype=np.float32)
    bins = build_calibration_bins(uncertainty, error, num_bins=2)
    assert len(bins) == 2
    assert bins[0].count + bins[1].count == 4


def test_summarize_calibration_fields() -> None:
    uncertainty = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    error = np.array([0.08, 0.22, 0.28, 0.41], dtype=np.float32)
    bins = build_calibration_bins(uncertainty, error, num_bins=3)
    summary = summarize_calibration(uncertainty, error, bins)
    for key in ("corr", "slope", "intercept", "ece"):
        assert key in summary


def test_collect_uncertainty_error_samples_shape() -> None:
    env = GridWorldEnv()
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16, num_dynamics_models=3)
    transitions = collect_random_transitions(env=env, num_episodes=2, horizon=4, seed=0)
    out = collect_uncertainty_error_samples(model=model, transitions=transitions)
    assert out["uncertainty"].shape == out["error"].shape
    assert out["uncertainty"].ndim == 1
