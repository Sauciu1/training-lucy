"""
Wrapper to train Lucy to walk forward.
Rewards forward progress while maintaining stability.
"""
import gymnasium as gym
import numpy as np


class LucyWalkingWrapper(gym.Wrapper):
    """
    Wrapper that rewards Lucy for walking forward while staying upright.
    
    Reward components:
        + Forward velocity: primary reward for moving in +x direction
        + Height bonus: reward for maintaining proper height
        + Upright bonus: reward for staying level
        + Gait bonus: reward for proper foot contact patterns
        - Control cost: penalize excessive torque usage
        - Body contact penalty: penalize non-foot parts touching ground
        - Fall penalty: terminate if fallen
    """
    
    def __init__(
        self,
        env,
        forward_weight: float = 2.0,
        target_height: float = 0.12,
        height_weight: float = 0.5,
        upright_weight: float = 0.3,
        gait_weight: float = 0.2,
        ctrl_cost_weight: float = 0.001,
        body_contact_penalty: float = -2.0,
        fall_threshold: float = 0.04,
        target_velocity: float = 0.5,
    ):
        super().__init__(env)
        
        self.forward_weight = forward_weight
        self.target_height = target_height
        self.height_weight = height_weight
        self.upright_weight = upright_weight
        self.gait_weight = gait_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.body_contact_penalty = body_contact_penalty
        self.fall_threshold = fall_threshold
        self.target_velocity = target_velocity
        
        # Get chest body ID
        import mujoco
        self._chest_id = mujoco.mj_name2id(
            self.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "chest"
        )
        
        self._prev_x = None
    
    def step(self, action):
        # Track x position before step
        if self._prev_x is None:
            self._prev_x = self.unwrapped.data.qpos[0]
        
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # === Forward velocity reward ===
        x_pos = self.unwrapped.data.qpos[0]
        forward_vel = (x_pos - self._prev_x) / self.unwrapped.dt
        self._prev_x = x_pos
        
        # Reward moving toward target velocity, penalize backwards
        if forward_vel > 0:
            # Bonus for reaching target velocity, diminishing returns beyond
            vel_ratio = forward_vel / self.target_velocity
            forward_reward = self.forward_weight * min(vel_ratio, 1.5)
        else:
            # Penalize backwards movement
            forward_reward = self.forward_weight * forward_vel * 2
        
        # === Height reward ===
        chest_height = self.unwrapped.data.xpos[self._chest_id, 2]
        height_error = abs(chest_height - self.target_height)
        height_reward = self.height_weight * np.exp(-3 * height_error)
        
        # === Upright reward ===
        chest_xmat = self.unwrapped.data.xmat[self._chest_id].reshape(3, 3)
        z_axis = chest_xmat[:, 2]
        upright_score = z_axis[2]
        upright_reward = self.upright_weight * max(0, upright_score)
        
        # === Gait reward (encourage alternating foot contacts) ===
        foot_contacts = self.unwrapped.get_foot_contacts()
        n_feet_down = sum(foot_contacts.values())
        # Ideal: 2-3 feet on ground while walking
        gait_reward = self.gait_weight if 1 <= n_feet_down <= 3 else 0
        
        # === Control cost ===
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        
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
            forward_reward +
            height_reward +
            upright_reward +
            gait_reward -
            ctrl_cost +
            contact_penalty +
            fall_penalty +
            0.2  # Small alive bonus
        )
        
        # === Info ===
        info.update({
            "forward_vel": forward_vel,
            "forward_reward": forward_reward,
            "height_reward": height_reward,
            "upright_reward": upright_reward,
            "upright_score": upright_score,
            "gait_reward": gait_reward,
            "feet_down": n_feet_down,
            "ctrl_cost": ctrl_cost,
            "body_contacts": len(body_contacts),
            "contact_penalty": contact_penalty,
            "x_position": x_pos,
            "chest_height": chest_height,
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self._prev_x = None
        return self.env.reset(**kwargs)
