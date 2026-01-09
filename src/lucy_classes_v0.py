"""
Custom Gymnasium environment for Lucy quadruped/bipedal locomotion.
"""
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import mujoco


import os
from src.definitions import PROJECT_ROOT


class LucyEnv(MujocoEnv):
    """
    Lucy locomotion environment.
    
    Observation space (106 dims):
        - qpos[2:]: positions excluding x,y (57 dims: z + orientations)  
        - qvel: velocities (47 dims)
        - sensor data from XML (optional, adds ~110 dims if used)
    
    Action space (19 dims):
        - One action per actuator (motor torques normalized to [-1, 1])
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }
    
    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 5,
        reset_noise_scale: float = 0.02,  # Reduced from 0.1 to keep feet on ground
        use_sensors: bool = False,
        render_mode: str = None,
        max_episode_seconds: float = None,
        **kwargs
    ):
        if xml_file is None:
            xml_file = os.path.join(PROJECT_ROOT, "animals", "lucy_v0.xml")
        
        self.reset_noise_scale = reset_noise_scale
        self.use_sensors = use_sensors
        self._init_qpos = None
        self._init_qvel = None
        # Optional episode length limit (simulated seconds)
        self._max_episode_seconds = max_episode_seconds
        # Simulated-time counter (seconds)
        self._sim_time = 0.0
        
        # Calculate observation space size
        # Will be set properly after model loads
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(104,), dtype=np.float64
        )
        
        super().__init__(
            xml_file,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,

            **kwargs
        )
        
        # Update observation space based on actual model
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        
        # Load keyframe "quad_stance" for proper initial pose
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "quad_stance")
        if keyframe_id >= 0:
            # Use keyframe qpos
            self._init_qpos = self.model.key_qpos[keyframe_id].copy()
            self._init_qvel = np.zeros(self.model.nv)
        else:
            # Fallback to default
            self._init_qpos = self.data.qpos.copy()
            self._init_qvel = self.data.qvel.copy()
        
        # Cache body and geom IDs for reward computation
        self._chest_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chest")
        self._head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        # Foot geom IDs for contact detection
        self._foot_geom_ids = {}
        for name in ["front_left_foot_geom", "front_right_foot_geom", 
                     "hind_left_foot_geom", "hind_right_foot_geom"]:
            try:
                self._foot_geom_ids[name] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
            except:
                pass

    
    def _get_obs(self) -> np.ndarray:
        """
        Build observation vector.
        Excludes x,y position for translation invariance.
        """
        # Position (excluding global x, y for translation invariance)
        position = self.data.qpos[2:].copy()  # 57 dims
        
        # Velocity
        velocity = self.data.qvel.copy()  # 47 dims
        
        obs = np.concatenate([position, velocity])
        
        if self.use_sensors:
            # Add sensor readings
            sensor_data = self.data.sensordata.copy()
            obs = np.concatenate([obs, sensor_data])
        
        return obs
    
    def step(self, action: np.ndarray):
        """Execute one timestep."""
        # Get position before step for velocity reward
        x_before = self.data.qpos[0]
        
        # Apply action
        self.do_simulation(action, self.frame_skip)
        
        # Get new observation
        obs = self._get_obs()
        
        # Compute reward components
        x_after = self.data.qpos[0]
        forward_velocity = (x_after - x_before) / self.dt
        
        # Basic reward: forward progress + alive bonus - control cost
        forward_reward = forward_velocity
        alive_bonus = 1.0
        ctrl_cost = 0.001 * np.sum(np.square(action))
        
        reward = forward_reward + alive_bonus - ctrl_cost
        
        # Termination conditions
        z_pos = self.data.qpos[2]
        chest_height = self.data.xpos[self._chest_id, 2]
        
        # Terminate if fallen (chest too low or tilted too much)
        terminated = chest_height < 0.05

        # Update simulated-time and apply truncation if configured
        # self.dt is seconds per step (timestep * frame_skip)
        self._sim_time += self.dt
        truncated = False
        if self._max_episode_seconds is not None and self._sim_time >= self._max_episode_seconds:
            truncated = True

        info = {
            "forward_velocity": forward_velocity,
            "chest_height": chest_height,
            "ctrl_cost": ctrl_cost,
            "x_position": x_after,
            "elapsed_sim_time": self._sim_time,
            "max_episode_seconds": self._max_episode_seconds,
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset_model(self) -> np.ndarray:
        """Reset to initial state with noise."""
        noise_scale = self.reset_noise_scale
        
        qpos = self._init_qpos + noise_scale * self.np_random.uniform(
            low=-1, high=1, size=self.model.nq
        )
        qvel = self._init_qvel + noise_scale * self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nv
        )
        
        # Keep root quaternion normalized
        qpos[3:7] = qpos[3:7] / np.linalg.norm(qpos[3:7])
        
        self.set_state(qpos, qvel)
        # Reset simulated-time counter when episode/reset starts
        self._sim_time = 0.0

        return self._get_obs()
    
    def get_foot_contacts(self) -> dict:
        """Return dict of which feet are in contact with floor."""
        contacts = {name: False for name in self._foot_geom_ids}
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            
            for name, gid in self._foot_geom_ids.items():
                if (g1 == gid and g2 == self._floor_id) or \
                   (g2 == gid and g1 == self._floor_id):
                    contacts[name] = True
        
        return contacts
    
    def get_body_contacts(self) -> list:
        """Return list of body parts touching floor (excluding feet)."""
        body_contacts = []
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            
            # Check if floor is involved
            if g1 != self._floor_id and g2 != self._floor_id:
                continue
            
            other_geom = g2 if g1 == self._floor_id else g1
            
            # Skip if it's a foot
            if other_geom in self._foot_geom_ids.values():
                continue
            
            # Get geom name
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
            if geom_name:
                body_contacts.append(geom_name)
        
        return body_contacts
    


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
        target_height: list = [0.30,0.35],
        height_weight: float = 2.0,
        upright_weight: float = 1.0,
        stillness_weight: float = 0.5,
        body_contact_penalty: float = -1.0,
        fall_threshold: list = [0.01, 1.0],
        fall_penalty: float = -50.0,
        head_direction_cone_deg: float = 20,
        head_direction_weight: float = 1.0,
        
    ):
        super().__init__(env)
        
        self.target_height = target_height
        self.height_weight = height_weight
        self.upright_weight = upright_weight
        self.stillness_weight = stillness_weight
        self.body_contact_penalty = body_contact_penalty
        self.fall_threshold = fall_threshold
        self.fall_penalty = fall_penalty
        # Head direction params
        self.head_direction_cone_deg = head_direction_cone_deg
        self.head_direction_weight = head_direction_weight
        
        # Get chest body ID
        import mujoco
        self._chest_id = mujoco.mj_name2id(
            self.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "chest"
        )
        # Cache head id too (may be used by head_direction_reward)
        self._head_id = mujoco.mj_name2id(self.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "head")

    @property
    def xpos(self):
        return self.unwrapped.data.xpos
    @property
    def xmat(self):
        return self.unwrapped.data.xmat

    @property
    def height_reward(self):
        chest_height = self.xpos[self._chest_id, 2]

        if  self.target_height[0] < chest_height < self.target_height[1]:
            height_error = 0.0
        else:
            height_error = abs(min([abs(chest_height - h) for h in self.target_height]))

        

        height_reward = self.height_weight * np.exp(-5 * height_error**0.5)
        return height_reward, chest_height

    @property
    def upright_reward(self):
        chest_xmat = self.xmat[self._chest_id].reshape(3, 3)
        # z-axis should point up (0, 0, 1)
        z_axis = chest_xmat[:, 2]
        upright_score = z_axis[2]  # Should be close to 1 when upright
        upright_reward = self.upright_weight * max(0, upright_score)

        return upright_reward, upright_score
    
    @property
    def stillness_reward(self):
        qvel = self.unwrapped.data.qvel
        linear_vel = np.linalg.norm(qvel[:3])  # Root linear velocity
        angular_vel = np.linalg.norm(qvel[3:6])  # Root angular velocity
        stillness_reward = self.stillness_weight * np.exp(-2 * (linear_vel + 0.5 * angular_vel))
        return stillness_reward
    
    @property
    def head_direction_reward(self):
        """Return (reward, angle_deg) — simplified.

        Projects head x-axis onto the horizontal plane and measures angle to
        world +x; gives linear reward inside cone, else zero.
        """
        hid = getattr(self, "_head_id", None)
        if hid is None or hid < 0:
            return 0.0, None

        vx, vy = self.xmat[hid].reshape(3, 3)[:, 0][:2]
        norm = np.hypot(vx, vy)
        if norm < 1e-6:
            return 0.0, None

        angle_deg = abs(float(np.degrees(np.arctan2(vy, vx))))
        cone = float(self.head_direction_cone_deg)
        if angle_deg > cone:
            return 0.0, angle_deg

        score = 1.0 - (angle_deg / cone)
        return float(self.head_direction_weight * score), angle_deg


    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        


        height_reward, chest_height = self.height_reward
        
        upright_reward, upright_score = self.upright_reward
        stillness_reward = self.stillness_reward
        head_dir_reward, head_dir_angle = self.head_direction_reward


        body_contacts = self.unwrapped.get_body_contacts()
        contact_penalty = self.body_contact_penalty * len(body_contacts)
        
        # === Fall termination ===
        if (chest_height < self.fall_threshold[0]) or (chest_height > self.fall_threshold[1]):
            terminated = True
            fall_penalty = self.fall_penalty
        else:
            fall_penalty = 0.0
        
        # === Total reward ===
        reward = (
            height_reward +
            upright_reward +
            stillness_reward +
            head_dir_reward +
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
            "head_direction_reward": head_dir_reward,
            "head_direction_angle": head_dir_angle,
            "body_contacts": len(body_contacts),
            "contact_penalty": contact_penalty,
            "chest_height": chest_height,
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



"""
Wrapper to train Lucy to walk forward.
Inherits from LucyStandingWrapper and adds forward movement rewards.
"""


class LucyWalkingWrapper(LucyStandingWrapper):
    """
    Wrapper that rewards Lucy for walking forward while staying upright.
    Inherits standing rewards from LucyStandingWrapper.
    
    Additional reward components:
        + Forward velocity: primary reward for moving in +x direction
        + Gait bonus: reward for proper foot contact patterns
        - Control cost: penalize excessive torque usage
    """
    
    def __init__(
        self,
        env,
        # Walking-specific params
        forward_weight: float = 2.0,
        target_velocity: float = 0.5,
        gait_weight: float = 0.2,
        ctrl_cost_weight: float = 0.001,
        # Standing params (passed to parent)
        target_height: float = 0.12,
        height_weight: float = 0.5,
        upright_weight: float = 0.3,
        stillness_weight: float = 0.0,  # Disable stillness reward for walking
        body_contact_penalty: float = -2.0,
        fall_threshold: float = 0.04,
    ):
        # Initialize parent (standing wrapper)
        super().__init__(
            env,
            target_height=target_height,
            height_weight=height_weight,
            upright_weight=upright_weight,
            stillness_weight=stillness_weight,
            body_contact_penalty=body_contact_penalty,
            fall_threshold=fall_threshold,
        )
        
        # Walking-specific params
        self.forward_weight = forward_weight
        self.target_velocity = target_velocity
        self.gait_weight = gait_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        
        self._prev_x = None
    
    def step(self, action):
        # Track x position before step
        if self._prev_x is None:
            self._prev_x = self.unwrapped.data.qpos[0]
        
        # Get standing rewards from parent
        obs, standing_reward, terminated, truncated, info = super().step(action)
        
        # === Forward velocity reward ===
        x_pos = self.unwrapped.data.qpos[0]
        forward_vel = (x_pos - self._prev_x) / self.unwrapped.dt
        self._prev_x = x_pos
        
        # Reward moving toward target velocity, penalize backwards
        if forward_vel > 0:
            vel_ratio = forward_vel / self.target_velocity
            forward_reward = self.forward_weight * min(vel_ratio, 1.5)
        else:
            forward_reward = self.forward_weight * forward_vel * 2
        
        # === Gait reward (encourage alternating foot contacts) ===
        foot_contacts = self.unwrapped.get_foot_contacts()
        n_feet_down = sum(foot_contacts.values())
        gait_reward = self.gait_weight if 1 <= n_feet_down <= 3 else 0
        
        # === Control cost ===
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        
        # === Total reward (standing + walking) ===
        reward = (
            standing_reward +
            forward_reward +
            gait_reward -
            ctrl_cost
        )
        
        # === Info ===
        info.update({
            "forward_vel": forward_vel,
            "forward_reward": forward_reward,
            "gait_reward": gait_reward,
            "feet_down": n_feet_down,
            "ctrl_cost": ctrl_cost,
            "x_position": x_pos,
            "standing_reward": standing_reward,
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self._prev_x = None
        return super().reset(**kwargs)
