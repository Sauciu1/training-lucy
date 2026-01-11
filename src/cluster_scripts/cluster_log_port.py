import os
import json
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor, load_results
from src import helpers
from src.definitions import enforce_absolute_path

import src.lucy_classes_v1 as lucy


ouput_prefix = "cluster_walking_v0"
monitor_path, model_path = helpers.generate_paths_monitor_model(ouput_prefix)


DEFAULT_MODEL_PARAMS = {
    "policy": "MlpPolicy",
    "verbose": 1,
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
    "env_number": 46,
    "timesteps": 5_000_000,
}





def create_env(env_params: dict):
    # env_params should contain all LucyEnv and LucyWalkingWrapper params
    env_kwargs = env_params.get("env_kwargs", {})
    wrapper_kwargs = env_params.get("wrapper_kwargs", {})
    return lucy.LucyWalkingWrapper(lucy.LucyEnv(**env_kwargs), **wrapper_kwargs)


class RunLogger:
    def __init__(self, run_history_dir=None):
        import datetime

        self.run_history_dir = run_history_dir or enforce_absolute_path("run_history")
        os.makedirs(self.run_history_dir, exist_ok=True)
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(self.run_history_dir, f"run_{run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_info_path = os.path.join(self.run_dir, "run_info.json")
        self.info = {}

    def log_params(self, model_params, env_params, run_params):
        self.info.update(
            {
                "model_params": model_params,
                "env_params": env_params,
                "run_params": run_params,
            }
        )
        self._save()

    def log_results(self, file_links, training_summary):
        self.info.update(
            {"file_links": file_links, "training_summary": training_summary}
        )
        self._save()

    def _save(self):
        with open(self.run_info_path, "w") as f:
            json.dump(self.info, f, indent=2)

    @property
    def dir(self):
        return self.run_dir


def train_and_log(
    model_params, env_params, run_params, ouput_prefix, logger: RunLogger
):
    monitor_path, model_path = helpers.generate_paths_monitor_model(ouput_prefix)
    vec_env = make_vec_env(
        lambda: create_env(env_params),
        n_envs=run_params["env_number"],
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_path,
    )
    vec_env = VecMonitor(vec_env, monitor_path)
    model = PPO(env=vec_env, **model_params)
    print(f"Training standing policy for {run_params['timesteps']:,} timesteps...")
    model.learn(total_timesteps=run_params["timesteps"])
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training complete.")
    walking_df = load_results(monitor_path)
    helpers.print_training_summary(walking_df)
    helpers.plot_training_progress(walking_df)
    file_links = {
        "lucy_env": env_params["env_kwargs"]["xml_file"],
        "script": os.path.abspath(__file__),
        "model_path": model_path,
        "monitor_path": monitor_path,
    }
    training_summary = {
        "episodes": (
            int(getattr(walking_df, "shape", [0])[0]) if walking_df is not None else 0
        )
    }
    logger.log_results(file_links, training_summary)


def make_relative(path, base=None):
    base = base or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return os.path.relpath(path, base)
    except Exception:
        return path


def rel_file_links(env_params, model_path, monitor_path):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "lucy_env": make_relative(env_params["env_kwargs"]["xml_file"], base),
        "script": make_relative(__file__, base),
        "model_path": make_relative(model_path, base),
        "monitor_path": make_relative(monitor_path, base),
    }


def main(
    model_params:dict=DEFAULT_MODEL_PARAMS,
    env_params:dict=DEFAULT_ENV_PARAMS,
    run_params:dict=DEFAULT_RUN_PARAMS,
    output_prefix:str="cluster_walking_v0",
):
    logger = RunLogger()
    logger.log_params(model_params, env_params, run_params)
    # Patch file_links to use relative paths

    # Train and log
    monitor_path, model_path = helpers.generate_paths_monitor_model(output_prefix)
    vec_env = make_vec_env(
        lambda: create_env(env_params),
        n_envs=run_params["env_number"],
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_path,
    )
    vec_env = VecMonitor(vec_env, monitor_path)
    model = PPO(env=vec_env, **model_params)
    print(f"Training standing policy for {run_params['timesteps']:,} timesteps...")
    model.learn(total_timesteps=run_params["timesteps"])
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training complete.")
    walking_df = load_results(monitor_path)
    helpers.print_training_summary(walking_df)
    #helpers.plot_training_progress(walking_df)
    file_links = rel_file_links(env_params, model_path, monitor_path)
    training_summary = {
        "episodes": (
            int(getattr(walking_df, "shape", [0])[0]) if walking_df is not None else 0
        )
    }
    logger.log_results(file_links, training_summary)




if __name__ == "__main__":
    model_params = DEFAULT_MODEL_PARAMS
    env_params = DEFAULT_ENV_PARAMS
    run_params = DEFAULT_RUN_PARAMS

    main(model_params, env_params, run_params)
