import gymnasium
from stable_baselines3 import PPO
import os
import time
from .definitions import PROJECT_ROOT
import sys


def render_model_gym(model_path:str, xml_path:str):

    eval_env = gymnasium.make("Ant-v5", xml_file=xml_path, render_mode="human", width=1600, height=800)
    model = PPO.load(model_path.replace(".zip", ""), env=eval_env)

    # Configure camera to follow the ant
    mujoco_env = eval_env.unwrapped
    torso_id = mujoco_env.model.body("torso").id
    print("Torso body ID:", torso_id)


    obs, info = eval_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        

        viewer = mujoco_env.mujoco_renderer.viewer
        viewer.cam.fixedcamid = 0  # Use camera ID 0 from your XML
        viewer.cam.type = 2  # 2 = fixed camera mode (mujoco.mjtCamera.mjCAMERA_FIXED)
        viewer.cam.distance = 8  # Increase distance from the object (adjust as needed)
        time.sleep(0.065)  # Control the speed of the simulation
        
        if terminated or truncated:
            obs, info = eval_env.reset()

if __name__ == "__main__":
    xml_path = os.path.join(PROJECT_ROOT, "animals", "ant.xml")
    

    model_path = sys.argv[1]
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    
    print(f"Loading model from: {model_path}")
    render_model_gym(model_path, xml_path)