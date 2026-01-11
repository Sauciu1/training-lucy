import os
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

def create_env(*args, **kwargs):
    return lucy.LucyWalkingWrapper(
        lucy.LucyEnv(xml_file=xml_path, render_mode="None", max_episode_seconds=30),
        *args,
        **kwargs,
        stillness_weight=-2.0,
        forward_weight=3.0
    )


if __name__ == "__main__":

    env_number = 7
    xml_path = enforce_absolute_path("animals/lucy_v3.xml")
    TIMESTEPS = 2_000_000
    
    vec_env = make_vec_env(
        create_env,
        n_envs=env_number,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_path,
    )

    vec_env = VecMonitor(vec_env, monitor_path)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device="cpu",
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        learning_rate=2e-4,
        target_kl=0.03,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512], vf=[512, 512]
            )
        ),
    )

    print(f"Training standing policy for {TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TIMESTEPS)

    model.save(model_path)

    print(f"Model saved to {model_path}")

    print("Training complete.")

    walking_df = load_results(monitor_path)

    helpers.print_training_summary(walking_df)
    helpers.plot_training_progress(walking_df)
