from __future__ import annotations

import subprocess
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="LatentPlan", layout="wide")
st.title("LatentPlan Interactive Demo")
st.caption("Train a world model, run latent planning, and inspect outputs.")

planner = st.selectbox("Planner", options=["random", "cem"], index=1)
epochs = st.slider("Training epochs", min_value=5, max_value=120, value=30, step=5)
num_sequences = st.slider("Num action sequences", min_value=32, max_value=512, value=128, step=32)
plan_horizon = st.slider("Plan horizon", min_value=4, max_value=24, value=12, step=1)
collect_episodes = st.slider("Collect episodes", min_value=40, max_value=600, value=200, step=20)
num_dynamics_models = st.slider("Dynamics ensemble size", min_value=1, max_value=5, value=3, step=1)
risk_penalty = st.slider("Risk penalty", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
goal_bonus = st.slider("Goal bonus", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

output_dir = Path("outputs/streamlit_demo")
ckpt = output_dir / "world_model.pt"

if st.button("Run Demo", use_container_width=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "latent_plan.main",
        "--force-train",
        "--epochs",
        str(epochs),
        "--collect-episodes",
        str(collect_episodes),
        "--num-sequences",
        str(num_sequences),
        "--plan-horizon",
        str(plan_horizon),
        "--planner",
        planner,
        "--num-dynamics-models",
        str(num_dynamics_models),
        "--risk-penalty",
        str(risk_penalty),
        "--goal-bonus",
        str(goal_bonus),
        "--save-animation",
        "--output-dir",
        str(output_dir),
        "--checkpoint",
        str(ckpt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        st.error("Run failed.")
        st.code(proc.stderr or proc.stdout)
    else:
        st.success("Run completed.")
        st.code(proc.stdout)

if (output_dir / "trajectory_comparison.png").exists():
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(output_dir / "trajectory_comparison.png"), caption="Real vs Imagined Trajectory")
    with col2:
        st.image(str(output_dir / "loss_curve.png"), caption="Loss Curve")

if (output_dir / "rollout.gif").exists():
    st.image(str(output_dir / "rollout.gif"), caption="Rollout Animation")

if (output_dir / "metrics.json").exists():
    st.subheader("Metrics")
    st.json((output_dir / "metrics.json").read_text(encoding="utf-8"))
