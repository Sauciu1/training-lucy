from src.old_lucy.lucy_classes_v0 import LucyEnv
import numpy as np

env = LucyEnv(render_mode="human")
obs, info = env.reset()

# Step with zero actions to let physics settle
for _ in range(5000):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        break

env.close()