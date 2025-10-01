# dashboard.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from reverse_aging_sim.model import (
    Simulation, default_config, plot_tissue_health, plot_bio_age, plot_km
)
from reverse_aging_sim.scenarios import simulate_population


st.set_page_config(page_title="Reverse Aging Simulator", layout="wide")

st.title("🧬 Reverse Aging Simulator (Toy Model)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Therapies")
    apply_stem = st.checkbox("Stem Cell Therapy", True)
    apply_repair = st.checkbox("Repair Therapy", True)
    apply_reprogram = st.checkbox("Reprogramming", True)
    apply_rejuvenation = st.checkbox("Tissue Rejuvenation", True)

    st.header("Global Settings")
    time_steps = st.slider("Time steps", 60, 300, 150, step=10)
    death_threshold = st.slider("Death threshold (%)", 5, 50, 20, step=1)
    skin_tone = st.slider("Skin tone index (0–1, toy effect)", 0.0, 1.0, 0.5, step=0.05)
    pop_n = st.slider("Population size for survival curve", 10, 200, 80, step=10)

    st.header("Rejuvenation Schedules")
    brain_time = st.slider("Brain reset age", 30, 100, 60, step=5)
    brain_reset = st.slider("Brain reset level (%)", 50, 100, 90, step=5)
    brain_decay = st.slider("Brain decay slowdown (×)", 0.4, 1.0, 0.7, step=0.05)

    muscle_time = st.slider("Muscle reset age", 30, 100, 65, step=5)
    muscle_reset = st.slider("Muscle reset level (%)", 50, 100, 80, step=5)
    muscle_decay = st.slider("Muscle decay slowdown (×)", 0.4, 1.0, 0.6, step=0.05)

    skin_time = st.slider("Skin reset age", 30, 100, 50, step=5)
    skin_reset = st.slider("Skin reset level (%)", 50, 100, 95, step=5)
    skin_decay = st.slider("Skin decay slowdown (×)", 0.4, 1.0, 0.5, step=0.05)

# ---------------- Config ----------------
cfg = default_config()
cfg.time_steps = int(time_steps)
cfg.death_threshold = float(death_threshold)
cfg.skin_tone = float(skin_tone)
cfg.rejuvenation_schedule = {
    "Brain":  [(int(brain_time),  float(brain_reset),  float(brain_decay))],
    "Muscle": [(int(muscle_time), float(muscle_reset), float(muscle_decay))],
    "Skin":   [(int(skin_time),   float(skin_reset),   float(skin_decay))],
}

# ---------------- Simulations ----------------
sim_nat = Simulation(config=cfg, apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
res_nat, bio_nat, life_nat = sim_nat.run()

sim_tx = Simulation(config=cfg, apply_stem=apply_stem, apply_repair=apply_repair, apply_reprogram=apply_reprogram, apply_rejuvenation=apply_rejuvenation)
res_tx, bio_tx, life_tx = sim_tx.run()

# ---------------- Layout ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Tissue Functional Health")
    plot_tissue_health(
        res_nat, res_tx,
        death_threshold=cfg.death_threshold,
        title=f"Tissue Health — Natural={life_nat}, Therapy={life_tx}"
    )
    st.caption("Dashed = Natural; Solid = With selected therapies")

with col2:
    st.subheader("Biological vs Chronological Age")
    plot_bio_age(bio_nat, bio_tx, title="Biological vs Chronological Age (Toy Mapping)")
    st.caption("Blue line lower than dashed means 'slower' biological aging in the toy model.")

# ---------------- Population Survival ----------------
st.subheader("Kaplan–Meier Survival (Population)")
lifespans_nat = simulate_population(pop_n, apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
lifespans_tx  = simulate_population(pop_n, apply_stem=apply_stem, apply_repair=apply_repair, apply_reprogram=apply_reprogram, apply_rejuvenation=apply_rejuvenation)
plot_km(lifespans_nat, lifespans_tx, title="Kaplan–Meier Survival — Natural vs Selected Therapies")

st.success(
    f"Avg lifespan (Natural): {np.mean(lifespans_nat):.1f}  |  "
    f"Avg lifespan (Therapies): {np.mean(lifespans_tx):.1f}  |  "
    f"Extension: {np.mean(lifespans_tx)-np.mean(lifespans_nat):.1f}"
)

# ---------------- Dynamic Analytics ----------------
st.subheader("📊 Key Analytics")

colA, colB, colC = st.columns(3)
colA.metric("Natural Lifespan", f"{life_nat} steps")
colB.metric("Therapy Lifespan", f"{life_tx} steps")
colC.metric("Extension", f"{life_tx - life_nat:+} steps")

# Tissue Health Stats
st.subheader("Tissue Health Stats (Final State)")
for tissue, vals in res_tx.items():
    raw_val = vals["func"][-1]
    clamped_val = min(max(raw_val, 0), 100)  # clamp 0–100 for progress bar
    st.progress(int(clamped_val), text=f"{tissue}: {raw_val:.1f}%")

# Comparative Table
st.subheader("Natural vs Therapy Comparison")
tissue_data = {
    "Tissue": list(res_nat.keys()),
    "Natural Func (%)": [res_nat[t]["func"][-1] for t in res_nat],
    "Therapy Func (%)": [res_tx[t]["func"][-1] for t in res_tx],
}
df = pd.DataFrame(tissue_data)
st.dataframe(df, use_container_width=True)

# ---------------- Interactive Plot ----------------
st.subheader("Interactive Tissue Health (Plotly)")
fig = go.Figure()
for t in res_nat:
    fig.add_trace(go.Scatter(y=res_nat[t]["func"], mode="lines", name=f"{t} (Natural)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(y=res_tx[t]["func"], mode="lines", name=f"{t} (Therapy)"))

fig.update_layout(title="Tissue Health Over Time", xaxis_title="Steps", yaxis_title="Health %")
st.plotly_chart(fig, use_container_width=True)
