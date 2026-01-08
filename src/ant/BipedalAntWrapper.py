import gymnasium as gym
import numpy as np


class BipedalAntWrapper(gym.Wrapper):
    """
    Wrapper that penalizes the ant when forbidden legs touch the ground.
    This encourages the agent to walk using only the allowed legs.
    """
    
    def __init__(self, env, allowed_legs=[0, 1], contact_penalty=-2.0):
        """
        Args:
            env: The base Ant environment
            allowed_legs: List of leg indices that CAN touch the ground (0-3)
                         Default [0, 1] = front two legs allowed
            contact_penalty: Penalty applied per forbidden leg touching ground
        """
        super().__init__(env)
        self.allowed_legs = allowed_legs
        self.forbidden_legs = [i for i in range(4) if i not in allowed_legs]
        self.contact_penalty = contact_penalty
        
        # Ant leg geom names (adjust if your XML uses different names)
        # Standard Ant-v5 has: aux_1, aux_2, aux_3, aux_4 (hip geoms)
        #                      ankle_1, ankle_2, ankle_3, ankle_4 (foot geoms)
        self.leg_geom_names = {
            0: ["aux_1_geom", "ankle_geom_1"],   # Front left
            1: ["aux_2_geom", "ankle_geom_2"],   # Front right
            2: ["aux_3_geom", "ankle_geom_3"],   # Back left
            3: ["aux_4_geom", "ankle_geom_4"],   # Back right
        }
        
        # Cache geom IDs for forbidden legs
        self._forbidden_geom_ids = None
        self._floor_geom_id = None
    
    def _get_geom_ids(self):
        """Cache geom IDs on first call."""
        if self._forbidden_geom_ids is None:
            model = self.unwrapped.model
            self._forbidden_geom_ids = set()
            
            for leg_idx in self.forbidden_legs:
                for geom_name in self.leg_geom_names[leg_idx]:
                    try:
                        geom_id = model.geom(geom_name).id
                        self._forbidden_geom_ids.add(geom_id)
                    except KeyError:
                        print(f"Warning: Geom '{geom_name}' not found in model")
            
            # Get floor geom ID
            try:
                self._floor_geom_id = model.geom("floor").id
            except KeyError:
                self._floor_geom_id = 0  # Usually floor is geom 0
                
    def _count_forbidden_contacts(self):
        """Count how many forbidden legs are touching the ground."""
        self._get_geom_ids()
        
        data = self.unwrapped.data
        forbidden_contacts = 0
        
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if contact is between floor and a forbidden leg geom
            is_floor_contact = (geom1 == self._floor_geom_id or geom2 == self._floor_geom_id)
            involves_forbidden = (geom1 in self._forbidden_geom_ids or 
                                  geom2 in self._forbidden_geom_ids)
            
            if is_floor_contact and involves_forbidden:
                forbidden_contacts += 1
        
        return forbidden_contacts
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Count forbidden leg contacts and apply penalty
        forbidden_contacts = self._count_forbidden_contacts()
        contact_penalty = self.contact_penalty * forbidden_contacts
        
        # Add penalty to reward
        modified_reward = reward + contact_penalty
        
        # Track in info for debugging
        info["forbidden_leg_contacts"] = forbidden_contacts
        info["contact_penalty"] = contact_penalty
        
        return obs, modified_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self._forbidden_geom_ids = None  # Reset cache
        self._floor_geom_id = None
        return self.env.reset(**kwargs)