from experiments.scripts.flhc import run_flhc_experiment
from experiments.scripts.hcfl import run_hcfl_experiment
from experiments.scripts.lcfed import run_lcfed_experiment

if __name__ == "__main__":
    base_seed = 2026
    # run_flhc_experiment(nb_runs=10, base_seed=base_seed)
    # run_hcfl_experiment(nb_runs=10, base_seed=base_seed)
    run_lcfed_experiment(nb_runs=6, base_seed=base_seed)
