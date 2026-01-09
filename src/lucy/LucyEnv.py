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
        reset_noise_scale: float = 0.1,
        use_sensors: bool = False,
        render_mode: str = None,
        **kwargs
    ):
        if xml_file is None:
            xml_file = os.path.join(PROJECT_ROOT, "animals", "lucy_v0.xml")
        
        self.reset_noise_scale = reset_noise_scale
        self.use_sensors = use_sensors
        self._init_qpos = None
        self._init_qvel = None
        
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
        
        # Cache initial state
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
        
        truncated = False
        
        info = {
            "forward_velocity": forward_velocity,
            "chest_height": chest_height,
            "ctrl_cost": ctrl_cost,
            "x_position": x_after,
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
