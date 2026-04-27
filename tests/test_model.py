import torch

from latent_plan.model import WorldModel


def test_encoder_output_shape() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    state = torch.randn(6, 2)
    z = model.encode(state)
    assert z.shape == (6, 8)


def test_dynamics_output_shape() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    z = torch.randn(6, 8)
    action = torch.randint(0, 4, (6,))
    z_next = model.predict_next(z, action)
    assert z_next.shape == (6, 8)


def test_forward_pass_works() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    state = torch.randn(6, 2)
    action = torch.randint(0, 4, (6,))

    z_next, reward_pred, state_next_pred = model(state, action)
    assert z_next.shape == (6, 8)
    assert reward_pred.shape == (6,)
    assert state_next_pred.shape == (6, 2)
    assert torch.isfinite(z_next).all()
    assert torch.isfinite(reward_pred).all()
    assert torch.isfinite(state_next_pred).all()


def test_decoder_output_shape() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16)
    z = torch.randn(6, 8)
    state_pred = model.decode(z)
    assert state_pred.shape == (6, 2)


def test_dynamics_ensemble_output_shape() -> None:
    model = WorldModel(state_dim=2, latent_dim=8, action_dim=4, hidden_dim=16, num_dynamics_models=3)
    z = torch.randn(5, 8)
    action = torch.randint(0, 4, (5,))
    z_ensemble = model.predict_next_ensemble(z, action)
    assert z_ensemble.shape == (3, 5, 8)
