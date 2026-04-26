# LatentPlan

## Description
A minimal world model that learns environment dynamics in latent space and plans via imagined rollouts.

## Key Idea
- Encode environment state into a compact latent vector `z`: `z_t = encoder(s_t)`.
- Learn latent dynamics for one-step prediction: `z_{t+1} = dynamics(z_t, a_t)`.
- Learn a reward predictor on latent states and choose actions by evaluating imagined latent trajectories.

## Architecture
- `Encoder`: MLP from normalized 2D state to latent embedding.
- `Dynamics model`: MLP from `(z_t, action)` to `z_{t+1}`.
- `Planner`: random-shooting in latent space; sample action sequences, roll out latent transitions, sum predicted rewards, execute best first action.

## Demo
- GIF placeholder: `outputs/planning_rollout.gif`
- Image placeholder: `outputs/trajectory_comparison.png`

## Results
- Works for short horizons where latent predictions stay accurate.
- Degrades for longer horizons due to compounding model error in imagined rollouts.

## Project Structure
- `latent_plan/env.py`: deterministic 2D gridworld with obstacles and goal.
- `latent_plan/model.py`: encoder, dynamics, reward model, and world model wrapper.
- `latent_plan/train.py`: random trajectory collection and supervised world-model training.
- `latent_plan/plan.py`: reusable latent planner and imagined rollout utilities.
- `latent_plan/visualize.py`: grid rendering, trajectory overlays, and optional animation.
- `latent_plan/main.py`: end-to-end demo (train, plan, evaluate, visualize).
- `tests/test_env.py`: movement and boundary tests for gridworld.
- `tests/test_model.py`: shape and forward-pass checks for model components.
- `tests/test_planner.py`: planner validity and imagined trajectory-length checks.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the world model:
   ```bash
   python -m latent_plan.train --epochs 150
   ```
3. Run demo with planning + visualization (loads checkpoint from `outputs/world_model.pt` if present):
   ```bash
   python -m latent_plan.main --num-sequences 256 --plan-horizon 12
   ```
4. Optional: force retraining inside `main.py`:
   ```bash
   python -m latent_plan.main --force-train --epochs 150 --num-sequences 256 --plan-horizon 12
   ```
5. Run tests:
   ```bash
   pytest -q
   ```
