import gymnasium
from stable_baselines3 import PPO
import time
import sys
import warnings

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