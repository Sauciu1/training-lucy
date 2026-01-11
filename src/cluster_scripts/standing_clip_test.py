from pdb import run
from pyexpat import model
import sys

from tensorboard import default
from src.cluster_scripts import cluster_log_port
from src.cluster_scripts.cluster_log_port import DEFAULT_ENV_PARAMS, DEFAULT_MODEL_PARAMS, DEFAULT_RUN_PARAMS
import src.lucy_classes_v1 as lucy


def create_env(env_params: dict):
    env_kwargs = env_params.get("env_kwargs", {})
    wrapper_kwargs = env_params.get("wrapper_kwargs", {})
    return lucy.LucyStandingWrapper(lucy.LucyEnv(**env_kwargs), **wrapper_kwargs)

if __name__ == "__main__":
    model_params = DEFAULT_MODEL_PARAMS

    print("Starting network architecture tests...")

    env_params = DEFAULT_ENV_PARAMS.copy()
    env_params.pop("wrapper_kwargs", None)
    env_params["env_kwargs"]["max_episode_seconds"] = 10.0


    model_params.update({"policy_kwargs": {"net_arch": {"pi": [512, 512], "vf": [512, 512]}}})


    run_params = DEFAULT_RUN_PARAMS
    run_params.update({"timesteps": 5_000_000, "env_number": 14})

    for clip_range in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        model_params.update({"clip_range": clip_range})
        cluster_log_port.main(model_params, env_params, run_params=run_params, output_prefix=f"clip_test/standing_clip_{clip_range}", create_env=create_env)



    print("Network architecture tests completed.")
    sys.exit(0)
