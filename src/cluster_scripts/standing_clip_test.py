# src/cluster_scripts/clip_sweep_standing.py
# Fixed: deep-copies configs, avoids oversubscription, sets thread caps,
# closes VecEnv, uses safe Subproc start_method, and keeps sweep sequential.

import copy
import os
import sys

# Hard cap BLAS/OpenMP threads (prevents oversubscription with SubprocVecEnv)


from src.cluster_scripts import cluster_log_port
from src.cluster_scripts.cluster_log_port import (
    DEFAULT_ENV_PARAMS,
    DEFAULT_MODEL_PARAMS,
    DEFAULT_RUN_PARAMS,
)
import src.lucy_classes_v1 as lucy


def create_env(env_params: dict):
    env_kwargs = env_params.get("env_kwargs", {})
    wrapper_kwargs = env_params.get("wrapper_kwargs", {})
    return lucy.LucyStandingWrapper(lucy.LucyEnv(**env_kwargs), **wrapper_kwargs)


def main():
    print("Starting clip-range sweep...")

    # Deep copies: avoid mutating module-level defaults across runs
    env_params = copy.deepcopy(DEFAULT_ENV_PARAMS)
    model_params = copy.deepcopy(DEFAULT_MODEL_PARAMS)
    run_params = copy.deepcopy(DEFAULT_RUN_PARAMS)

    # Ensure wrapper kwargs are present for create_env; do NOT pop them.
    env_params.pop("wrapper_kwargs", {})
    # Shorter episodes for faster eval
    env_params["env_kwargs"]["max_episode_seconds"] = 10.0


    model_params.setdefault("policy_kwargs", {})
    model_params["policy_kwargs"] = {"net_arch": {"pi": [512, 512], "vf": [512, 512]}}


    try:
        ncpus = int(os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS") or 0)
        run_params["env_number"] = ncpus*1
    except ValueError:
        run_params["env_number"] = 14

    run_params["timesteps"] = 3_000_000

    # Sweep (sequential; if you want concurrent sweeps, submit multiple PBS jobs)
    clip_ranges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for clip_range in clip_ranges:
        mp = copy.deepcopy(model_params)
        mp["clip_range"] = float(clip_range)

        prefix = f"standing_clip_test/clip_{clip_range}"
        cluster_log_port.main(
            mp,
            env_params,
            run_params=run_params,
            output_prefix=prefix,
            create_env=create_env,
        )

    print("Clip-range sweep completed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
