\
from typing import Dict, List, Tuple
import numpy as np
from .model import Simulation, default_tissues, default_config

def run_pair(time_steps: int = 150):
    # Natural vs all therapies
    sim_nat = Simulation(config=default_config(), apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
    sim_nat.cfg.time_steps = time_steps
    res_nat, bio_nat, life_nat = sim_nat.run()

    sim_all = Simulation(config=default_config(), apply_stem=True, apply_repair=True, apply_reprogram=True, apply_rejuvenation=True)
    sim_all.cfg.time_steps = time_steps
    res_all, bio_all, life_all = sim_all.run()

    return (res_nat, bio_nat, life_nat), (res_all, bio_all, life_all)

def simulate_population(n: int = 100, **kwargs):
    lifespans = []
    for _ in range(n):
        sim = Simulation(config=default_config(), rand_variation=True, **kwargs)
        _, _, life = sim.run()
        lifespans.append(life)
    return lifespans
