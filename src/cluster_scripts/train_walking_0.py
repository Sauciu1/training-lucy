import os
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor, load_results
from datetime import datetime
from src import helpers
from src.definitions import PROJECT_ROOT, enforce_absolute_path

import src.lucy_classes_v1 as lucy


env_number = 30
xml_path = enforce_absolute_path("animals/lucy_v2.xml")

output_path = "cluster_standing_v0"
model = mujoco.MjModel.from_xml_path(xml_path)


time_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M")

monitor_dir = enforce_absolute_path(
    os.path.join(PROJECT_ROOT, "logs", "logs", f"{output_path}_{time_suffix}")
)

model_path = enforce_absolute_path(
    os.path.join(
        PROJECT_ROOT, "outputs", "trained_models", f"{output_path}_{time_suffix}"
    )
)


def create_env(*args, **kwargs):
    return lucy.LucyStandingWrapper(
        lucy.LucyEnv(xml_file=xml_path, render_mode="None", max_episode_seconds=10),
        *args,
        **kwargs,
    )


if __name__ == "__main__":
    STANDING_TIMESTEPS = 10_000_000
    vec_env = make_vec_env(
        create_env,
        n_envs=env_number,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_dir,
    )

    vec_env = VecMonitor(vec_env, monitor_dir)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cpu",
        n_steps=2048,
        batch_size=256,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        learning_rate=1e-4,
        target_kl=0.02,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512, 512, 256], vf=[512, 512, 512, 256])
        ),
    )

    print(f"Training standing policy for {STANDING_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=STANDING_TIMESTEPS)

    model.save(model_path)

    print(f"Model saved to {model_path}")

    print("Training complete.")

    walking_df = load_results(monitor_dir)

    helpers.print_training_summary(walking_df)
