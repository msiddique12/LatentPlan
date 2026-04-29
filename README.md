# LatentPlan

## Description
A minimal world model that learns environment dynamics in latent space and plans via imagined rollouts.

## Key Idea
- Encode environment state into a compact latent vector `z`: `z_t = encoder(s_t)`.
- Learn latent dynamics for one-step prediction: `z_{t+1} = dynamics(z_t, a_t)`.
- Learn a reward predictor on latent states and choose actions by evaluating imagined latent trajectories.

## Architecture
- `Encoder`: MLP from normalized 2D state to latent embedding.
- `Dynamics model`: supports an ensemble of MLPs from `(z_t, action)` to `z_{t+1}` for uncertainty-aware planning.
- `Reward model`: predicts reward from latent state.
- `Decoder`: reconstructs normalized state from latent (`z -> s`) for improved interpretability.
- `Planner`: supports `random` shooting and `cem` (cross-entropy method), plus discounting, risk penalty, and goal shaping.

## Demo
- GIF placeholder: `outputs/planning_rollout.gif`
- Image placeholder: `outputs/trajectory_comparison.png`
- Metrics placeholder: `outputs/metrics.json`
- Repro manifest placeholder: `outputs/run_manifest.json`
- Train history CSV placeholder: `outputs/train_history.csv`
- Planning diagnostics placeholder: `outputs/planning_diagnostics.png`

## Results
- Works for short horizons where latent predictions stay accurate.
- Degrades for longer horizons due to compounding model error in imagined rollouts.

## Project Structure
- `latent_plan/env.py`: deterministic 2D gridworld with obstacles and goal.
- `latent_plan/model.py`: encoder, dynamics, reward model, and world model wrapper.
- `latent_plan/train.py`: random trajectory collection and supervised training (transition + reward + reconstruction losses).
- `latent_plan/plan.py`: reusable latent planner with random-shooting and CEM.
- `latent_plan/visualize.py`: grid rendering, trajectory overlays, and optional animation.
- `latent_plan/main.py`: end-to-end demo (train, plan, evaluate, visualize).
- `latent_plan/evaluate.py`: batch comparison (`random` vs `cem`) across seeds.
- `app.py`: streamlit app for interactive demo runs.
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
4. Use CEM planning and save animation:
   ```bash
   python -m latent_plan.main --planner cem --num-sequences 256 --cem-iters 5 --save-animation
   ```
5. Compare random vs CEM across seeds:
   ```bash
   python -m latent_plan.evaluate --num-seeds 5 --epochs 30
   ```
   Saves:
   - `outputs/eval/planner_comparison.json` (mean/std/min/max per metric and raw per-seed results)
   - `outputs/eval/planner_comparison.png`
   - `outputs/eval/uncertainty_calibration.json` (calibration bins + ECE/correlation/slope)
   - `outputs/eval/uncertainty_calibration.png`
   - `outputs/eval/run_manifest.json` (exact run configuration)
6. Optional: force retraining inside `main.py`:
   ```bash
   python -m latent_plan.main --force-train --epochs 150 --num-sequences 256 --plan-horizon 12 --num-dynamics-models 3 --risk-penalty 0.05 --goal-bonus 0.1
   ```
7. Optional interactive UI:
   ```bash
   streamlit run app.py
   ```
8. Run tests:
   ```bash
   pytest -q
   ```
