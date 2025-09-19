# Reverse Aging Simulator (Toy Model)

This project implements a **toy** (conceptual) simulator for reverse aging using stem cells, reprogramming, and repair concepts.
It includes:
- A Python library (`reverse_aging_sim`) with a simulation engine.
- A CLI to run scenarios and save plots.
- A Streamlit dashboard for interactive exploration.

> ⚠️ Scientific disclaimer: this is *not* a biological simulator. It is a conceptual system-dynamics toy model for education and prototyping.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run CLI (save plots under `outputs/`)
```bash
python -m reverse_aging_sim.cli --scenario all --time-steps 150
```

### Launch Streamlit dashboard
```bash
streamlit run reverse_aging_sim/dashboard.py
```

## What it models

- **Tissues**: Brain, Muscle, Skin (configurable rates).
- **Aging**: Functional cells decline; stem cell pools also decline.
- **Regeneration**: Stem cells replenish functional cells (regen efficiency).
- **Interventions**:
  - Stem cell boosts (schedule + amount)
  - Repair therapy (periodic health boost, lower decay temporarily)
  - Reprogramming (partial reset to higher health on chosen steps)
  - Tissue-specific rejuvenation (per-tissue reset time/level + slower decay)
- **Biological vs chronological age** (inverse health mapping)
- **Lifespan**: organism "dies" when average functional health < threshold
- **Population runs** + Kaplan–Meier-style survival curves

## Files

- `reverse_aging_sim/model.py` — core engine
- `reverse_aging_sim/scenarios.py` — ready-made scenarios
- `reverse_aging_sim/cli.py` — command-line entry point
- `reverse_aging_sim/dashboard.py` — Streamlit interactive app

---

© 2025 Toy model for educational use.
