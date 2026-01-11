import os
import json
from tracemalloc import start
from typing import Any, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src import helpers
from src.definitions import enforce_absolute_path
import src.lucy_classes_v1 as lucy
import sys


output_prefix = "cluster_walking_v0"


# -----------------------------
# Paths / run directory
# -----------------------------
def run_dir_for(run_name: str, dt_str: str) -> str:
    return os.path.join("cluster_copy", f"{run_name}_{dt_str}")


# -----------------------------
# Defaults tuned for HPC CPU
# -----------------------------
DEFAULT_MODEL_PARAMS = {
    "policy": "MlpPolicy",
    "verbose": 0,
    "device": "cpu",
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,
    "learning_rate": 2e-4,
    "target_kl": 0.03,
    "policy_kwargs": {"net_arch": {"pi": [512, 512], "vf": [512, 512]}},
}

DEFAULT_ENV_PARAMS = {
    "env_kwargs": {
        "xml_file": enforce_absolute_path("animals/lucy_v3.xml"),
        "render_mode": "None",
        "max_episode_seconds": 30,
    },
    "wrapper_kwargs": {
        "stillness_weight": -2.0,
        "forward_weight": 3.0,
    },
}

DEFAULT_RUN_PARAMS = {
    "env_number": 40,
    "timesteps": 3_000_000,
    # regular logging ignores these; kept for compatibility with callers
    "print_every_s": 20,
    "flush_every_episodes": 50,
}


def create_env(env_params: dict):
    env_kwargs = env_params.get("env_kwargs", {})
    wrapper_kwargs = env_params.get("wrapper_kwargs", {})
    return lucy.LucyWalkingWrapper(lucy.LucyEnv(**env_kwargs), **wrapper_kwargs)


# -----------------------------
# Run logging (single JSON)
# -----------------------------
class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_info_path = os.path.join(self.run_dir, "run_info.json")
        self.info: Dict[str, Any] = {}

    def log_params(self, model_params, env_params, run_params):
        self.info.update(
            {"model_params": model_params, "env_params": env_params, "run_params": run_params}
        )
        self._save()

    def log_artifacts(self, artifacts: Dict[str, str]):
        self.info.update({"artifacts": artifacts})
        self._save()

    def _save(self):
        with open(self.run_info_path, "w", encoding="utf-8") as f:
            json.dump(self.info, f, indent=2)


# -----------------------------
# Main (REGULAR SB3 LOGGING)
# -----------------------------
def main(
    model_params: dict = DEFAULT_MODEL_PARAMS,
    env_params: dict = DEFAULT_ENV_PARAMS,
    run_params: dict = DEFAULT_RUN_PARAMS,
    output_prefix: str = "cluster_walking_v0",
    create_env: callable = create_env,
):
    import datetime

    dt_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = run_dir_for(output_prefix, dt_str)
    os.makedirs(run_dir, exist_ok=True)

    # SB3 paths
    monitor_path, model_path = helpers.generate_paths_monitor_model(output_prefix)
    monitor_path = os.path.join(run_dir, os.path.basename(monitor_path))
    os.makedirs(monitor_path, exist_ok=True)
    model_path = os.path.join(run_dir, os.path.basename(model_path))

    # TensorBoard logs
    tb_log_dir = os.path.join(run_dir, "tb_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    # Run metadata
    run_logger = RunLogger(run_dir)
    run_logger.log_params(model_params, env_params, run_params)

    vec_env = None
    try:
        vec_env = make_vec_env(
            lambda: create_env(env_params),
            n_envs=int(run_params["env_number"]),
            vec_env_cls=SubprocVecEnv,
            monitor_dir=monitor_path,   
            vec_env_kwargs={"start_method": "forkserver" if sys.platform != "win32" else "spawn"},
            
        )

        # Optional: aggregates episode stats into infos (cheap)
        vec_env = VecMonitor(vec_env)

        model = PPO(
            env=vec_env,
            **model_params,
            tensorboard_log=tb_log_dir,
        )

        model.learn(
            total_timesteps=int(run_params["timesteps"]),
            tb_log_name=output_prefix,
            progress_bar=False,
            log_interval=50,
        )

        model.save(model_path)

        run_logger.log_artifacts(
            {
                "run_dir": run_dir,
                "model_path": model_path,
                "monitor_dir": monitor_path,
                "tensorboard_log_dir": tb_log_dir,
            }
        )
    finally:
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    main(DEFAULT_MODEL_PARAMS, DEFAULT_ENV_PARAMS, DEFAULT_RUN_PARAMS, output_prefix)
