"""
Wrapper to train Lucy to stand in place.
Rewards upright posture and penalizes falling/moving.
"""
import gymnasium as gym
import numpy as np


class LucyStandingWrapper(gym.Wrapper):
    """
    Wrapper that rewards Lucy for standing still and upright.
    
    Reward components:
        + Height bonus: reward for keeping chest at target height
        + Upright bonus: reward for keeping chest level (not tilted)
        + Stillness bonus: reward for low velocity
        - Fall penalty: terminate and penalize if chest falls below threshold
        - Body contact penalty: penalize non-foot parts touching ground
    """
    
    def __init__(
        self,
        env,
        target_height: float = 0.15,
        height_weight: float = 2.0,
        upright_weight: float = 1.0,
        stillness_weight: float = 0.5,
        body_contact_penalty: float = -1.0,
        fall_threshold: float = 0.05,
    ):
        super().__init__(env)
        
        self.target_height = target_height
        self.height_weight = height_weight
        self.upright_weight = upright_weight
        self.stillness_weight = stillness_weight
        self.body_contact_penalty = body_contact_penalty
        self.fall_threshold = fall_threshold
        
        # Get chest body ID
        import mujoco
        self._chest_id = mujoco.mj_name2id(
            self.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "chest"
        )
    
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # === Height reward ===
        chest_height = self.unwrapped.data.xpos[self._chest_id, 2]
        height_error = abs(chest_height - self.target_height)
        height_reward = self.height_weight * np.exp(-5 * height_error)
        
        # === Upright reward (based on chest z-axis alignment) ===
        # Get rotation matrix for chest
        chest_xmat = self.unwrapped.data.xmat[self._chest_id].reshape(3, 3)
        # z-axis should point up (0, 0, 1)
        z_axis = chest_xmat[:, 2]
        upright_score = z_axis[2]  # Should be close to 1 when upright
        upright_reward = self.upright_weight * max(0, upright_score)
        
        # === Stillness reward (penalize movement) ===
        # Penalize both linear and angular velocity
        qvel = self.unwrapped.data.qvel
        linear_vel = np.linalg.norm(qvel[:3])  # Root linear velocity
        angular_vel = np.linalg.norm(qvel[3:6])  # Root angular velocity
        stillness_reward = self.stillness_weight * np.exp(-2 * (linear_vel + 0.5 * angular_vel))
        
        # === Body contact penalty ===
        body_contacts = self.unwrapped.get_body_contacts()
        contact_penalty = self.body_contact_penalty * len(body_contacts)
        
        # === Fall termination ===
        if chest_height < self.fall_threshold:
            terminated = True
            fall_penalty = -10.0
        else:
            fall_penalty = 0.0
        
        # === Total reward ===
        reward = (
            height_reward +
            upright_reward +
            stillness_reward +
            contact_penalty +
            fall_penalty +
            0.5  # Small alive bonus
        )
        
        # === Info for debugging ===
        info.update({
            "height_reward": height_reward,
            "upright_reward": upright_reward,
            "upright_score": upright_score,
            "stillness_reward": stillness_reward,
            "body_contacts": len(body_contacts),
            "contact_penalty": contact_penalty,
            "chest_height": chest_height,
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
