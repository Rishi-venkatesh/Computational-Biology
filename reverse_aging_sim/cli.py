\
import argparse, os, json
# from .model import Simulation, default_config, default_tissues, plot_tissue_health, plot_bio_age, plot_km
# from .scenarios import simulate_population
from reverse_aging_sim.model import Simulation, default_tissues, default_config, plot_tissue_health, plot_bio_age, plot_km
from reverse_aging_sim.scenarios import simulate_population


def main():
    p = argparse.ArgumentParser(description="Reverse Aging Simulator (Toy Model)")
    p.add_argument("--time-steps", type=int, default=150)
    p.add_argument("--scenario", choices=["natural", "all", "stem", "repair", "reprogram", "rejuvenation", "custom", "all-pop"], default="all")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--population", type=int, default=100, help="For *-pop scenarios: number of individuals")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    def run_sim(sim):
        sim.cfg.time_steps = args.time_steps
        res, bio, life = sim.run()
        return res, bio, life

    # Build scenarios
    if args.scenario == "natural":
        sim_a = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        res_a, bio_a, life_a = run_sim(sim_a)
        sim_b = None; res_b=bio_b=life_b=None

    elif args.scenario == "all":
        sim_a = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        res_a, bio_a, life_a = run_sim(sim_a)
        sim_b = Simulation(apply_stem=True, apply_repair=True, apply_reprogram=True, apply_rejuvenation=True)
        res_b, bio_b, life_b = run_sim(sim_b)

    elif args.scenario == "stem":
        sim_a = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        res_a, bio_a, life_a = run_sim(sim_a)
        sim_b = Simulation(apply_stem=True, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        res_b, bio_b, life_b = run_sim(sim_b)

    elif args.scenario == "repair":
        sim_a = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        res_a, bio_a, life_a = run_sim(sim_a)
        sim_b = Simulation(apply_stem=False, apply_repair=True, apply_reprogram=False, apply_rejuvenation=False)
        res_b, bio_b, life_b = run_sim(sim_b)

    elif args.scenario == "reprogram":
        sim_a = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        res_a, bio_a, life_a = run_sim(sim_a)
        sim_b = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=True, apply_rejuvenation=False)
        res_b, bio_b, life_b = run_sim(sim_b)

    elif args.scenario == "rejuvenation":
        sim_a = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        res_a, bio_a, life_a = run_sim(sim_a)
        sim_b = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=True)
        res_b, bio_b, life_b = run_sim(sim_b)

    elif args.scenario == "all-pop":
        # Compare survival across population for natural vs all
        from .model import km_curve
        lifespans_nat = simulate_population(args.population, apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        lifespans_all = simulate_population(args.population, apply_stem=True, apply_repair=True, apply_reprogram=True, apply_rejuvenation=True)
        plot_km(lifespans_nat, lifespans_all, title="Kaplan–Meier Survival: Natural vs All Therapies")
        # Save numeric stats
        with open(os.path.join(args.outdir, "population_summary.json"), "w") as f:
            json.dump({
                "avg_lifespan_natural": sum(lifespans_nat)/len(lifespans_nat),
                "avg_lifespan_all": sum(lifespans_all)/len(lifespans_all),
                "extension": (sum(lifespans_all)/len(lifespans_all)) - (sum(lifespans_nat)/len(lifespans_nat))
            }, f, indent=2)
        return

    else:  # custom (toggle all on by default)
        sim_a = Simulation(apply_stem=False, apply_repair=False, apply_reprogram=False, apply_rejuvenation=False)
        res_a, bio_a, life_a = run_sim(sim_a)
        sim_b = Simulation(apply_stem=True, apply_repair=True, apply_reprogram=True, apply_rejuvenation=True)
        res_b, bio_b, life_b = run_sim(sim_b)

    # Plots & summary
    plot_tissue_health(res_a, res_b, death_threshold=20.0, title=f"Tissue Health — {args.scenario}")
    plot_bio_age(bio_a, bio_b, title=f"Biological vs Chronological — {args.scenario}")

    # Save summary json
    with open(os.path.join(args.outdir, f"summary_{args.scenario}.json"), "w") as f:
        json.dump({
            "scenario": args.scenario,
            "lifespan_A": life_a,
            "lifespan_B": (life_b if 'life_b' in locals() else None)
        }, f, indent=2)

if __name__ == "__main__":
    main()
