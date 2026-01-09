import gymnasium
from stable_baselines3 import PPO
import time
import sys
import warnings

# Suppress the GPU warning from stable-baselines3
warnings.filterwarnings("ignore", message=".*PPO on the GPU.*")

from src import enforce_absolute_path
from src.definitions import PROJECT_ROOT
from src.lucy import LucyEnv
from src.ant import BipedalAntWrapper


# Available environment types
ENV_TYPES = {
    "lucy": lambda xml_path: LucyEnv(xml_file=xml_path, render_mode="human"),
    "ant": lambda xml_path: gymnasium.make("Ant-v5", xml_file=xml_path, render_mode="human", width=1600, height=800),
    "bipedal": lambda xml_path: BipedalAntWrapper(gymnasium.make("Ant-v5", xml_file=xml_path, render_mode="human", width=1600, height=800)),
}


def render_model_gym(model_path: str, xml_path: str, env_type: str = "lucy", speed: float = 1.0):
    if env_type not in ENV_TYPES:
        print(f"Unknown env type: {env_type}. Available: {list(ENV_TYPES.keys())}")
        sys.exit(1)
    
    eval_env = ENV_TYPES[env_type](xml_path)
    model = PPO.load(model_path.replace(".zip", ""), env=eval_env)

    # Configure camera
    mujoco_env = eval_env.unwrapped
    # Try chest (Lucy) or torso (Ant)
    try:
        body_id = mujoco_env.model.body("chest").id
        print("Chest body ID:", body_id)
    except:
        body_id = mujoco_env.model.body("torso").id
        print("Torso body ID:", body_id)

    obs, info = eval_env.reset()
    
    # Calculate delay based on speed (1.0 = real-time, 0.5 = half speed, 2.0 = double speed)
    frame_time = 0.025 / speed  # Base: 0.005 timestep * 5 frame_skip = 0.025s per step
    
    print(f"Running simulation at {speed}x speed... Close the window to exit.")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Render the frame - this updates the display
            eval_env.render()
            
            # Control playback speed
            time.sleep(frame_time)

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