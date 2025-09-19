\
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Data classes ------------------

@dataclass
class Tissue:
    name: str
    func: float = 100.0          # functional cell "health" (%)
    stem: float = 50.0           # stem cell pool (%)
    decay_func: float = 0.5      # loss per step (absolute % points)
    decay_stem: float = 0.2
    regen_eff: float = 0.05      # functional gain per stem unit per step

@dataclass
class SimConfig:
    time_steps: int = 150
    death_threshold: float = 20.0

    # Therapy: stem
    stem_boost_times: List[int] = field(default_factory=lambda: [30,60,90,120])
    stem_boost_amount: float = 30.0

    # Therapy: repair
    repair_interval: int = 10
    repair_boost: float = 5.0
    repair_decay_factor: float = 0.8  # multiply decay_func during repair step

    # Therapy: reprogramming
    reprogram_times: List[int] = field(default_factory=lambda: [100])
    reprogram_multiplier: float = 1.5  # func *= multiplier (capped at 100)

    # Tissue-specific rejuvenation: dict[tissue] = [(time, reset_level, decay_factor_after)]
    rejuvenation_schedule: Dict[str, List[Tuple[int, float, float]]] = field(default_factory=lambda: {
        "Brain":  [(60, 90, 0.7)],
        "Muscle": [(65, 80, 0.6)],
        "Skin":   [(50, 95, 0.5)],
    })

    # Optional: skin tone index (0..1). Higher may slow skin aging slightly in this toy model.
    skin_tone: float = 0.5

def default_tissues() -> Dict[str, Tissue]:
    return {
        "Brain":  Tissue("Brain",  func=100, stem=40, decay_func=0.3, decay_stem=0.1, regen_eff=0.02),
        "Muscle": Tissue("Muscle", func=100, stem=50, decay_func=0.5, decay_stem=0.2, regen_eff=0.05),
        "Skin":   Tissue("Skin",   func=100, stem=60, decay_func=0.8, decay_stem=0.3, regen_eff=0.08),
    }

def default_config() -> SimConfig:
    return SimConfig()

# ------------------ Engine ------------------

class Simulation:
    def __init__(
        self,
        tissues: Optional[Dict[str, Tissue]] = None,
        config: Optional[SimConfig] = None,
        apply_stem: bool = True,
        apply_repair: bool = True,
        apply_reprogram: bool = True,
        apply_rejuvenation: bool = True,
        rand_variation: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.tissues: Dict[str, Tissue] = tissues or default_tissues()
        self.cfg: SimConfig = config or default_config()
        self.apply_stem = apply_stem
        self.apply_repair = apply_repair
        self.apply_reprogram = apply_reprogram
        self.apply_rejuvenation = apply_rejuvenation
        self.rand_variation = rand_variation
        self.rng = rng or np.random.default_rng()

        # optional random variation for population runs
        if self.rand_variation:
            for t in self.tissues.values():
                t.decay_func *= self.rng.normal(1.0, 0.1)
                t.decay_stem *= self.rng.normal(1.0, 0.1)
                t.regen_eff  *= self.rng.normal(1.0, 0.1)

        # simple skin tone effect (illustrative only)
        tone = float(self.cfg.skin_tone)
        if "Skin" in self.tissues:
            self.tissues["Skin"].decay_func *= (1.0 - 0.1 * (tone - 0.5))  # +/-5%

    def run(self):
        results = {name: {"func": [], "stem": []} for name in self.tissues}
        bio_age: List[float] = []
        lifespan: Optional[int] = None

        # clone tissues (so repeated runs do not accumulate state)
        tissues = {n: Tissue(**vars(t)) for n, t in self.tissues.items()}

        for t in range(self.cfg.time_steps):
            avg_func_health: List[float] = []
            for name, vals in tissues.items():
                decay_func = vals.decay_func
                decay_stem = vals.decay_stem

                # Repair therapy
                if self.apply_repair and t % self.cfg.repair_interval == 0 and t > 0:
                    vals.func += self.cfg.repair_boost
                    decay_func *= self.cfg.repair_decay_factor

                # Natural decline
                vals.func -= decay_func
                vals.stem -= decay_stem

                # Regeneration
                vals.func += vals.regen_eff * vals.stem

                # Stem therapy
                if self.apply_stem and t in self.cfg.stem_boost_times:
                    vals.stem += self.cfg.stem_boost_amount

                # Reprogramming
                if self.apply_reprogram and t in self.cfg.reprogram_times:
                    vals.func = min(vals.func * self.cfg.reprogram_multiplier, 100.0)

                # Tissue-specific rejuvenation
                if self.apply_rejuvenation and name in self.cfg.rejuvenation_schedule:
                    for (ev_t, reset_level, decay_factor) in self.cfg.rejuvenation_schedule[name]:
                        if t == ev_t:
                            vals.func = float(reset_level)
                            vals.decay_func *= float(decay_factor)

                # Clamp
                vals.func = max(vals.func, 0.0)
                vals.stem = max(vals.stem, 0.0)

                # Record
                results[name]["func"].append(vals.func)
                results[name]["stem"].append(vals.stem)
                avg_func_health.append(vals.func)

            # Biological age = chronological / avg_health_fraction (toy mapping)
            avg_health = np.mean(avg_func_health) / 100.0
            bio_age.append(t / avg_health if avg_health > 0 else self.cfg.time_steps * 2)

            # Lifespan condition
            if lifespan is None and np.mean(avg_func_health) < self.cfg.death_threshold:
                lifespan = t

        if lifespan is None:
            lifespan = self.cfg.time_steps

        return results, bio_age, lifespan

# ------------------ Helpers ------------------

def plot_tissue_health(results_a: Dict[str, Dict[str, List[float]]],
                       results_b: Optional[Dict[str, Dict[str, List[float]]]] = None,
                       death_threshold: float = 20.0,
                       title: str = "Tissue Health") -> None:
    plt.figure(figsize=(12,6))
    for name in results_a:
        plt.plot(results_a[name]["func"], "--", label=f"{name} (A)")
        if results_b:
            plt.plot(results_b[name]["func"], "-", label=f"{name} (B)")
    plt.axhline(y=death_threshold, linestyle="--")
    plt.xlabel("Time (steps)")
    plt.ylabel("Functional Health (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bio_age(bio_a: List[float], bio_b: Optional[List[float]] = None, title: str = "Biological vs Chronological Age") -> None:
    t = range(len(bio_a))
    plt.figure(figsize=(10,6))
    plt.plot(t, t, "k--", label="Chronological Age")
    plt.plot(t, bio_a, label="Biological Age (A)")
    if bio_b is not None:
        plt.plot(t, bio_b, label="Biological Age (B)")
    plt.xlabel("Chronological Age (steps)")
    plt.ylabel("Biological Age (relative units)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def km_curve(lifespans: List[int], max_time: Optional[int] = None):
    ls = np.array(lifespans, dtype=float)
    mt = int(max_time if max_time is not None else (np.max(ls) if len(ls) else 0))
    times = np.arange(0, mt + 1)
    surv = [(ls >= t).mean() for t in times]
    return times, surv

def plot_km(lifespans_a: List[int], lifespans_b: List[int], title: str = "Kaplan–Meier Survival"):
    mt = max(max(lifespans_a) if lifespans_a else 0, max(lifespans_b) if lifespans_b else 0)
    ta, sa = km_curve(lifespans_a, mt)
    tb, sb = km_curve(lifespans_b, mt)
    plt.figure(figsize=(10,6))
    plt.step(ta, sa, where="post", label="Scenario A")
    plt.step(tb, sb, where="post", label="Scenario B")
    plt.xlabel("Time (steps)")
    plt.ylabel("Survival Probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
