from turtle import pd
import gymnasium
from stable_baselines3 import PPO
import time
import sys
import warnings
import pandas as pd
# Suppress the GPU warning from stable-baselines3
warnings.filterwarnings("ignore", message=".*PPO on the GPU.*")

from src import enforce_absolute_path
from src.definitions import PROJECT_ROOT
from src.lucy_classes_v0 import LucyEnv
from src.ant import BipedalAntWrapper


# Available environment types
ENV_TYPES = {
    "lucy": lambda xml_path: LucyEnv(xml_file=xml_path, render_mode="human"),
    "ant": lambda xml_path: gymnasium.make("Ant-v5", xml_file=xml_path, render_mode="human", width=1600, height=800),
    "bipedal": lambda xml_path: BipedalAntWrapper(gymnasium.make("Ant-v5", xml_file=xml_path, render_mode="human", width=1600, height=800)),
}


from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
import matplotlib.pyplot as plt
import mujoco

def collect_frames(env:MujocoEnv, max_frames: int = 20, attr_keys=None):
    """Collect up to ``max_frames`` frames and attribute dicts from ``env``.

    Returns (frames, attrs) where ``frames`` is a list of RGB arrays and
    ``attrs`` is a list of dictionaries. Each attr dict contains the keys
    in ``attr_keys`` (defaults provided) plus ``time`` and ``reward``.

    Note: ``time`` is the canonical timestamp key (seconds). We do not
    include ``elapsed_sim_time`` to avoid duplication.
    """
    if attr_keys is None:
        # Do not include 'elapsed_sim_time' here; 'time' will be provided
        attr_keys = ("chest_height", "forward_velocity", "ctrl_cost")

    # Helper to build a row and coerce numeric types
    def _make_row(info, time_val, reward):
        base = {k: (info.get(k) if isinstance(info, dict) else None) for k in attr_keys}
        base["time"] = float(time_val) if time_val is not None else None
        base["reward"] = float(reward) if reward is not None else None
        return base

    frames = []
    attrs = []

    obs, info = env.reset()
    prev_time = float(info.get("elapsed_sim_time", 0.0)) if isinstance(info, dict) else 0.0



    frame = env.render()
    if frame is not None:
        frames.append(frame)
        attrs.append(_make_row(info, prev_time, None))

    for _ in range(max_frames - 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)


        prev_time = float(info["elapsed_sim_time"])


        frame = env.render()



        if frame is not None:
            frames.append(frame)
            attrs.append(_make_row(info, prev_time, reward))

        if terminated:
            print("Episode terminated during frame collection.")
            break

    return frames, attrs



def show_grid(frames, times, width_per_img=2, height_per_img=2.5):
    """Display frames in a grid; wide=True biases toward more columns than rows."""
    n = len(frames)
    if n == 0:
        print("No frames captured (render returned None).")
        return


    base = np.sqrt(n)
    rows = max(1, int(np.ceil(base / 2.0)))
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * width_per_img, rows * height_per_img))
    axes_flat = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for ax in axes_flat:
        ax.axis('off')

    for i, frame in enumerate(frames):
        axes_flat[i].imshow(frame)
        axes_flat[i].set_title(f"{times[i]:.2f}s", fontsize=8)
        axes_flat[i].axis('off')

    plt.suptitle(f'First {n} frames ({rows}×{cols})', y=1.02)
    plt.tight_layout()


def display_test_env(env, max_frames: int = 21, frame_skip:int = 1, attr_keys=None):
    """Collect frames/attributes and display a grid. Returns a DataFrame of attrs."""
    if attr_keys is None:
        attr_keys = []

    frames, attrs = collect_frames(env, max_frames*frame_skip, attr_keys=attr_keys)
    if frame_skip > 1:
        frames = frames[::frame_skip]
        attrs = attrs[::frame_skip]

    # Use canonical 'time' field
    times = [attr.get("time") for attr in attrs]
    if times and times[0] is None:
        times[0] = 0.0

    df = pd.DataFrame(attrs)

    df = df[["time"]+[str(col) for col in df.columns if col is not "time"]]

    show_grid(frames, times)
    plt.suptitle(f'First {len(frames)} frames (frame skip={frame_skip})', y=1.02)
    return df





def render_model_gym(model_path: str, xml_path: str, env_type: str = "lucy", speed: float = 1.0):
    if env_type not in ENV_TYPES:
        print(f"Unknown env type: {env_type}. Available: {list(ENV_TYPES.keys())}")
        sys.exit(1)
    
    eval_env = ENV_TYPES[env_type](xml_path)
    model = PPO.load(model_path.replace(".zip", ""), env=eval_env)

    # Configure camera
    mujoco_env = eval_env.unwrapped
    # Try chest (Lucy) or torso (Ant)

    body_id = mujoco_env.model.body("chest").id




    obs, info = eval_env.reset()
    
    # Auto-calculate real-time delay from environment's dt (timestep * frame_skip)
    # dt is automatically set by MuJoCo env based on XML timestep and frame_skip
    sim_dt = mujoco_env.dt  # Time per step in simulation seconds
    print(f"Simulation dt: {sim_dt:.4f}s per step")
    
    print(f"Running simulation at {speed}x speed... Close the window to exit.")
    try:
        while True:
            step_start = time.perf_counter()
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Render the frame - this updates the display
            eval_env.render()
            
            # Calculate remaining time to sleep for real-time playback
            elapsed = time.perf_counter() - step_start
            target_time = sim_dt / speed  # Adjust for playback speed
            sleep_time = max(0, target_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            if terminated or truncated:
                obs, info = eval_env.reset()
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        eval_env.close()

if __name__ == "__main__":

    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python render_model_gym.py <model_path> <xml_path> [env_type] [speed]")
        print(f"  env_type: {list(ENV_TYPES.keys())} (default: lucy)")
        print("  speed: playback speed multiplier (default: 1.0, use 0.5 for half speed)")
        sys.exit(1)

    model_path = enforce_absolute_path(sys.argv[1])
    xml_path = enforce_absolute_path(sys.argv[2])
    env_type = sys.argv[3] if len(sys.argv) >= 4 else "lucy"
    speed = float(sys.argv[4]) if len(sys.argv) == 5 else 1.0

    print(f"Loading model from: {model_path}")
    print(f"Using environment: {env_type}")
    render_model_gym(model_path, xml_path, env_type, speed)