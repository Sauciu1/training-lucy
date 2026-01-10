"""
Custom Gymnasium environment for Lucy quadruped/bipedal locomotion.
"""

from math import floor
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
        reset_noise_scale: float = 0.2,
        render_mode: str = None,
        max_episode_seconds: float = None,
    
        stance: str = "quad_stance",

        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(PROJECT_ROOT, "animals", "lucy_v1.xml")

        self.reset_noise_scale = reset_noise_scale
        self._init_qpos = None
        self._init_qvel = None

        self._max_episode_seconds = max_episode_seconds


        self._sim_time = 0.0

        # Calculate observation space size (use model counts to avoid heavy ops)
        pos_len = None
        vel_len = None
        sens_len = None

        super().__init__(
            xml_file,
            frame_skip=frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(104,), dtype=np.float32),
            render_mode=render_mode,
            **kwargs,
        )

        self.__init_model_stance(stance)

        # Determine lengths directly from model to avoid allocations
        pos_len = int(self.model.nq) - 2
        vel_len = int(self.model.nv)
        sens_len = int(self.model.nsensordata)
        obs_dim = pos_len + vel_len + sens_len

        # Preallocate float32 observation buffer to avoid concatenations each step
        self._obs_buffer = np.empty(obs_dim, dtype=np.float32)
        self._pos_len = pos_len
        self._vel_len = vel_len
        self._sens_len = sens_len

        # Update observation space to correct shape/dtype
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # ID cache and debug flags
        self._id_cache: dict[tuple[str, int], int] = {}
        self.debug_sensors = False

        self._preset_geometry_ids()

        # Convenience helpers for working with sensors in notebooks

    @property
    def sensor_names(self) -> list:
        """Return list of sensor names in model order."""
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(int(self.model.nsensor))
        ]

    def sensor_info(self) -> dict:
        """Return dict mapping sensor_name -> {'id', 'adr', 'dim'}."""
        info = {}
        ns = int(self.model.nsensor)
        for i in range(ns):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = int(self.model.sensor_adr[i])
            if i + 1 < ns:
                dim = int(self.model.sensor_adr[i + 1]) - adr
            else:
                dim = int(self.model.nsensordata) - adr
            info[name] = {"id": i, "adr": adr, "dim": dim}
        return info

    def sensors_dict(self) -> dict:
        """Return dict mapping sensor_name -> value (float or numpy array).

        To avoid per-step allocations, this is gated by `self.debug_sensors`.
        When disabled this returns an empty dict quickly.
        """
        if not getattr(self, "debug_sensors", False):
            return {}

        out = {}
        for name, (adr, dim) in self._sensor_meta.items():
            if dim == 1:
                out[name] = float(self.data.sensordata[adr])
            else:
                out[name] = self.data.sensordata[adr : adr + dim].copy()
        return out

    @property
    def sensors_array(self) -> np.ndarray:
        """Return a copy of the full sensordata array."""
        return self.data.sensordata.copy()

    def _preset_geometry_ids(self):
        """Cache common IDs and maps to avoid repeated name lookups at runtime."""
        # Foot geom IDs for contact detection (name -> gid)
        self._foot_geom_ids = {}
        self._geomid_to_footname = {}
        for name in [
            "front_left_foot_geom",
            "front_right_foot_geom",
            "hind_left_foot_geom",
            "hind_right_foot_geom",
        ]:
            gid = self.name_to_id(name, "geom")
            if gid is not None:
                self._foot_geom_ids[name] = gid
                self._geomid_to_footname[gid] = name

        # Quick lookup sets
        self._foot_geom_id_set = set(self._foot_geom_ids.values())

        # Cache chest and floor IDs
        self._chest_id = self.name_to_id("chest", "body")
        self._floor_id = self.name_to_id("floor", "geom")

        # Cache tail actuator ids (names starting with 'tail')
        self._tail_act_ids = []
        na = int(self.model.nu)
        for aid in range(na):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            if aname and aname.startswith("tail"):
                self._tail_act_ids.append(aid)

        # Cache sensor metadata for fast slicing
        self._sensor_meta = {}
        ns = int(self.model.nsensor)
        for sid in range(ns):
            sname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            adr = int(self.model.sensor_adr[sid])
            if sid + 1 < ns:
                dim = int(self.model.sensor_adr[sid + 1]) - adr
            else:
                dim = int(self.model.nsensordata) - adr
            self._sensor_meta[sname] = (adr, dim)

    def __init_model_stance(self, stance="quad_stance"):
        """Load initial pose from keyframe in XML."""
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, stance)

        self._init_qpos = self.model.key_qpos[keyframe_id].copy()
        self._init_qvel = np.zeros(self.model.nv)

    def get_sensor_by_name(self, sensor_name: str):
        """Return sensor value for given sensor name.

        Uses cached `_sensor_meta` for fast adr/dim lookup. Returns None if sensor
        is not present.
        """
        meta = self._sensor_meta.get(sensor_name)
        if meta is None:
            return None
        adr, dim = meta
        if dim == 1:
            return float(self.data.sensordata[adr])
        return self.data.sensordata[adr : adr + dim].copy()

    def get_part_velocity(self, part_name: str) -> dict | None:
        """Return {'linear': np.array([vx,vy,vz]), 'angular': np.array([wx,wy,wz])} or None.

        Accepts either a body name or a geom name. Returns None if the named
        part is not found.
        """
        # Try body first
        bid = self.name_to_id(part_name, "body")
        if bid is None:
            # Fall back to geom name -> body id
            gid = self.name_to_id(part_name, "geom")
            if gid is None:
                return None
            body_id = int(self.model.geom_bodyid[gid])
        else:
            body_id = int(bid)

        # Safely access xvel (may not be present in some backends)
        xvel = getattr(self.data, "xvel", None)

        # Validate body index
        if xvel is None or body_id < 0 or body_id >= int(self.model.nbody):
            ang = np.zeros(3, dtype=float)
            lin = np.zeros(3, dtype=float)
        else:
            # xvel layout: [wx, wy, wz, vx, vy, vz] (angular then linear)
            row = np.array(xvel[body_id, :6], dtype=float)
            ang = row[:3]
            lin = row[3:6]

        return {"linear": lin, "angular": ang, "linear_mag": float(np.linalg.norm(lin))}

    def name_to_id(self, name: str, obj_type: object) -> int | None:
        """Convert a name to its Mujoco ID with caching.

        Usage: self.name_to_id('chest', 'body') -> int id or None if not found.
        Results are cached in self._id_cache to avoid repeated lookups.
        """
        if isinstance(obj_type, str):
            type_map = {
                "body": mujoco.mjtObj.mjOBJ_BODY,
                "geom": mujoco.mjtObj.mjOBJ_GEOM,
                "joint": mujoco.mjtObj.mjOBJ_JOINT,
                "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
                "sensor": mujoco.mjtObj.mjOBJ_SENSOR,
                "key": mujoco.mjtObj.mjOBJ_KEY,
            }
            if obj_type not in type_map:
                raise ValueError(f"Unknown object type: {obj_type}")
            tm = type_map[obj_type]
        elif isinstance(obj_type, int):
            tm = obj_type
        else:
            raise ValueError(f"Unsupported obj_type: {obj_type}")

        key = (name, int(tm))
        if key in getattr(self, "_id_cache", {}):
            cached = self._id_cache[key]
            return None if cached < 0 else cached

        obj_id = mujoco.mj_name2id(self.model, tm, name)
        # Store -1 or id; return None for not found
        self._id_cache[key] = obj_id
        return None if obj_id < 0 else obj_id

    def _get_obs(self) -> np.ndarray:
        """Fill and return the preallocated `_obs_buffer`.

        Copies qpos[2:], qvel, and sensordata into the preallocated
        float32 buffer to avoid per-step concatenations. Returns a *copy*
        of the buffer to preserve the previous contract (caller-safe).
        """
        # Use local refs for speed
        buf = self._obs_buffer
        pos_len = self._pos_len
        vel_len = self._vel_len
        sens_len = self._sens_len

        # Copy qpos (exclude global x, y)
        # Assumes self.data.qpos has at least pos_len + 2 elements
        np.copyto(buf[:pos_len], self.data.qpos[2 : 2 + pos_len], casting="unsafe")

        # Copy qvel
        start = pos_len
        end = start + vel_len
        np.copyto(buf[start:end], self.data.qvel[:vel_len], casting="unsafe")

        # Copy sensordata (may be zero-length)
        if sens_len:
            start = end
            end = start + sens_len
            np.copyto(buf[start:end], self.data.sensordata[:sens_len], casting="unsafe")

        # Return a copy so callers can't mutate internal buffer
        return buf.copy()

    def _compute_forward(self, x_before: float) -> tuple[float, float]:
        """Return (x_after, forward_velocity)."""
        x_after = float(self.data.qpos[0])
        forward_velocity = (x_after - x_before) / self.dt
        return x_after, forward_velocity

    def _compute_reward_components(
        self, forward_velocity: float, action: np.ndarray
    ) -> tuple[float, float, float]:
        """Return (reward, forward_reward, ctrl_cost)."""
        forward_reward = forward_velocity
        ctrl_cost = 0.001 * np.sum(np.square(action))
        alive_bonus = 1.0
        reward = forward_reward + alive_bonus - ctrl_cost
        return float(reward), float(forward_reward), float(ctrl_cost)

    def _check_termination(self) -> tuple[bool, float]:
        """Return (terminated, chest_height)."""
        chest_id = self.name_to_id("chest", "body")
        chest_height = float(self.data.xpos[chest_id, 2])
        terminated = chest_height < 0.05
        return terminated, chest_height

    def _update_sim_time(self) -> bool:
        """Advance sim time and return whether the episode is truncated."""
        self._sim_time += self.dt
        if (
            self._max_episode_seconds is not None
            and self._sim_time >= self._max_episode_seconds
        ):
            return True
        return False

    def _build_step_info(
        self,
        forward_velocity: float,
        chest_height: float,
        ctrl_cost: float,
        x_position: float,
    ) -> dict:
        info = {
            "forward_velocity": float(forward_velocity),
            "chest_height": float(chest_height),
            "ctrl_cost": float(ctrl_cost),
            "x_position": float(x_position),
            "elapsed_sim_time": self._sim_time,
            "max_episode_seconds": self._max_episode_seconds,
        }

        info["sensors"] = self.sensors_dict()
        return info

    def step(self, action: np.ndarray):
        """Execute one timestep."""
        # Snapshot state before stepping
        x_before = float(self.data.qpos[0])

        # Advance simulation
        self.do_simulation(action, self.frame_skip)

        # Observation after step
        obs = self._get_obs()

        # Compute kinematics & rewards
        x_after, forward_velocity = self._compute_forward(x_before)
        reward, forward_reward, ctrl_cost = self._compute_reward_components(
            forward_velocity, action
        )

        # Termination / truncation
        terminated, chest_height = self._check_termination()
        truncated = self._update_sim_time()

        # Diagnostics/info
        info = self._build_step_info(forward_velocity, chest_height, ctrl_cost, x_after)

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

        # Prevent root from being lowered below initial pose to avoid leg-foot clipping
        root_z_idx = 2
        qpos[root_z_idx] = 0.2

        self.set_state(qpos, qvel)
        # Reset simulated-time counter when episode/reset starts
        self._sim_time = 0.0

        return self._get_obs()

    def get_foot_contacts(self) -> dict:
        """Return dict of which feet are in contact with floor using cached ids.

        Optimized to minimize attribute lookups and use set membership checks.
        """
        contacts = {name: False for name in self._foot_geom_ids}
        floor_id = getattr(self, "_floor_id", None) or self.name_to_id("floor", "geom")

        if floor_id is None:
            return contacts

        data = self.data
        contact_arr = data.contact
        ncon = int(data.ncon)
        foot_set = self._foot_geom_id_set
        geom_to_name = self._geomid_to_footname

        for i in range(ncon):
            c = contact_arr[i]
            g1 = c.geom1
            g2 = c.geom2

            if g1 == floor_id and g2 in foot_set:
                fname = geom_to_name.get(g2)
                if fname:
                    contacts[fname] = True
            elif g2 == floor_id and g1 in foot_set:
                fname = geom_to_name.get(g1)
                if fname:
                    contacts[fname] = True

        return contacts

    def get_body_contacts(self) -> list:
        """Return list of body parts touching floor (excluding feet), using cached ids.

        Optimized to minimize attribute access and delay name lookups until needed.
        """
        body_contacts = []

        floor_id = getattr(self, "_floor_id", None) or self.name_to_id("floor", "geom")
        if floor_id is None:
            return body_contacts

        data = self.data
        contact_arr = data.contact
        ncon = int(data.ncon)
        foot_set = self._foot_geom_id_set
        mj_id2name = mujoco.mj_id2name

        for i in range(ncon):
            c = contact_arr[i]
            g1 = c.geom1
            g2 = c.geom2

            # Check if floor is involved
            if g1 != floor_id and g2 != floor_id:
                continue
            other_geom = g2 if g1 == floor_id else g1

            # Skip if it's a foot
            if other_geom in foot_set:
                continue

            geom_name = mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
            if geom_name:
                body_contacts.append(geom_name)

        return body_contacts


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
        env: LucyEnv,
        height_target_dicts=[
            {"part": "chest", "target_height": [0.22, 0.4], "reward_weight": 1.0},
            {"part": "hips", "target_height": [0.22, 0.4], "reward_weight": 0.5},
            {"part": "head", "target_height": [0.4, 0.5], "reward_weight": 0.3},
        ],
        upright_parts: list[str] = ["chest", "head", "hips"],
        upright_weight=1.0,
        stillness_weight=0.2,
        body_contact_penalty=-2.0,
        fall_threshold=[0.12, 0.7],
        fall_penalty=-80.0,
        head_direction_cone_deg=20.0,
        head_direction_weight=0.3,
        leg_position_weight: float = 0.8,
        straight_tail_weight = 0.15,
        tail_action_penalty = 0.15,

        which_legs: str = "all",
        **kwargs,
    ):
        super().__init__(env, **kwargs)

        self.height_target_dicts = height_target_dicts
        self.upright_weight = upright_weight
        self.stillness_weight = stillness_weight
        self.body_contact_penalty = body_contact_penalty
        self.fall_threshold = fall_threshold
        self.fall_penalty = fall_penalty
        self.upright_parts = upright_parts
        self.straight_tail_weight = straight_tail_weight
        self.tail_action_penalty = tail_action_penalty  # penalty weight for using tail actuators

        # Leg position reward weight (thigh/shin down + foot orientation)
        self.leg_position_weight = leg_position_weight

        self.get_part_velocity = env.get_part_velocity
        self.get_sensor_by_name = env.get_sensor_by_name

        self.name_to_id = env.name_to_id

        # Head direction params
        self.head_direction_cone_deg = head_direction_cone_deg
        self.head_direction_weight = head_direction_weight



        self.which_legs = which_legs

    @property
    def xpos(self):
        return self.unwrapped.data.xpos

    @property
    def xmat(self):
        return self.unwrapped.data.xmat

    @property
    def height_reward(self) -> dict[str, list[float]]:
        """Reward for keeping specified body parts at target heights.
        """


        output_dict = {}

        for ht_dict in self.height_target_dicts:
            assert [
                k in ht_dict for k in ("part", "target_height", "reward_weight")
            ], "Missing keys in height_target_dict"

            part_id = self.name_to_id(ht_dict["part"], "body")
            part_height = float(self.xpos[part_id, 2])

            if ht_dict["target_height"][0] < part_height < ht_dict["target_height"][1]:
                height_error = 0.0
            else:
                height_error = abs(
                    min([abs(part_height - h) for h in ht_dict["target_height"]])
                )

            r = ht_dict["reward_weight"] * np.exp(-1000 * height_error)


            output_dict[ht_dict["part"]] = {
                "height": part_height,
                "reward": r,
            }

        return output_dict

    @property
    def upright_reward(self):
        """Reward for upright posture of specified body parts."""
        part_name = "chest"
        part_id = self.name_to_id(part_name, "body")
        part_xmat = self.xmat[part_id].reshape(3, 3)
        # z-axis should point up (0, 0, 1)
        z_axis = part_xmat[:, 2]
        upright_score = z_axis[2]  # Should be close to 1 when upright
        upright_reward = self.upright_weight * max(0, upright_score)

        return upright_reward, upright_score

    @property
    def stillness_reward(self):
        """Reward for not moving much"""
        qvel = self.unwrapped.data.qvel
        linear_vel = np.linalg.norm(qvel[:3])  # Root linear velocity
        angular_vel = np.linalg.norm(qvel[3:6])  # Root angular velocity
        stillness_reward = self.stillness_weight * np.exp(
            -2 * (linear_vel + 0.5 * angular_vel)
        )
        return stillness_reward

    @property
    def head_direction_reward(self):
        """Return (reward, angle_deg) — simplified.

        Projects head x-axis onto the horizontal plane and measures angle to
        world +x; gives linear reward inside cone, else zero.
        """
        hid = self.name_to_id("head", "body")
        
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

    @property
    def _leg_position_reward(self):
        """Reward for leg positions: thigh/shin pointing downwards and foot pointing forward.

        For each leg we compute:
          * thigh_vec = shin_pos - thigh_pos
          * shin_vec = foot_pos - shin_pos
        Thigh/shin are rewarded for having a negative z component (pointing down);
        foot is rewarded for being roughly perpendicular to the shin (≈90°) and
        for pointing forward (world +x).

        Returns (reward, details_dict).
        """

        legs = [
            ("front_left_thigh", "front_left_shin", "front_left_foot"),
            ("front_right_thigh", "front_right_shin", "front_right_foot")]
        
        if self.which_legs =="all":
            legs.extend([
                ("hind_left_thigh", "hind_left_shin", "hind_left_foot"),
                ("hind_right_thigh", "hind_right_shin", "hind_right_foot"),
            ])

        scores = []
        details = {}
        for thigh_name, shin_name, foot_name in legs:
            tid = self.name_to_id(thigh_name, "body")
            sid = self.name_to_id(shin_name, "body")
            fid = self.name_to_id(foot_name, "body")
            if tid is None or sid is None or fid is None:
                # missing parts -> no contribution
                scores.append(0.0)
                details[thigh_name.replace("_thigh", "")] = {
                    "thigh": 0.0,
                    "shin": 0.0,
                    "foot": 0.0,
                    "score": 0.0,
                }
                continue

            thigh_pos = self.unwrapped.data.xpos[tid].copy()
            shin_pos = self.unwrapped.data.xpos[sid].copy()
            foot_pos = self.unwrapped.data.xpos[fid].copy()

            thigh_vec = shin_pos - thigh_pos
            shin_vec = foot_pos - shin_pos

            t_score = max(0.0, -float(thigh_vec[2]) / (np.linalg.norm(thigh_vec) + 1e-8))
            s_score = max(0.0, -float(shin_vec[2]) / (np.linalg.norm(shin_vec) + 1e-8))

            # Foot orientation: reward when the foot's forward axis is roughly
            # perpendicular to the shin vector (≈90°) AND the foot points forward.
            foot_x = self.unwrapped.data.xmat[fid].reshape(3, 3)[:, 0]
            fx_norm = foot_x / (np.linalg.norm(foot_x) + 1e-8)
            s_norm = shin_vec / (np.linalg.norm(shin_vec) + 1e-8)

            cosang = float(np.dot(s_norm, fx_norm))
            perp_score = float(np.sqrt(max(0.0, 1.0 - cosang * cosang)))
            # Require forward-facing: project foot axis onto world +x
            forward_factor = float(max(0.0, fx_norm[0]))
            f_score = float(perp_score * forward_factor)

            leg_score = float((t_score + s_score + f_score) / 3.0)

            details[thigh_name.replace("_thigh", "")] = {
                "thigh": float(t_score),
                "shin": float(s_score),
                "foot": float(f_score),
                "score": float(leg_score),
            }
            scores.append(leg_score)

        mean_score = float(np.mean(scores)) if scores else 0.0
        reward = float(self.leg_position_weight * mean_score)
        return reward, details
    

    def straight_tail_reward(self, action=None):
        """Reward for keeping the tail straight and not actuating it.

        Computes straightness by checking alignment between consecutive tail
        segment vectors (tail_1 -> tail_2 and tail_2 -> tail_3).  Score is
        |cos(angle)| between the two segment directions (1.0 = perfectly
        colinear).  Penalizes actuation magnitude on tail motors (sum of abs
        controls) when `action` is provided or falls back to `data.ctrl`.

        Returns (reward, details_dict).
        """
        # Tail body names
        tail_bodies = ["tail_1", "tail_2", "tail_3"]
        poses = []
        for b in tail_bodies:
            bid = self.name_to_id(b, "body")
            if bid is None or bid < 0:
                return 0.0, {"straight_score": 0.0, "actuation": 0.0}
            poses.append(self.unwrapped.data.xpos[bid].copy())

        v1 = poses[1] - poses[0]
        v2 = poses[2] - poses[1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            straight_score = 0.0
        else:
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            straight_score = float(abs(max(-1.0, min(1.0, cosang))))

        # Actuation magnitude for tail motors
        act_vals = []

        # Require cached tail actuator ids to be present — fail fast if missing
        tail_act_ids = self.unwrapped._tail_act_ids
        assert isinstance(tail_act_ids, list) and tail_act_ids, (
            "Tail actuator IDs not found — ensure the model defines tail actuators and `_preset_geometry_ids` was called"
        )

        # Gather actuation values from provided `action` or `data.ctrl` without try/except
        ctrl = self.unwrapped.data.ctrl
        if action is not None and hasattr(action, "__len__"):
            action_len = len(action)
        else:
            action_len = 0

        for aid in tail_act_ids:
            # assume aid is valid index — will raise naturally if not
            val = 0.0
            if action_len and 0 <= aid < action_len:
                val = float(action[aid])
            elif len(ctrl) > aid:
                val = float(ctrl[aid])
            act_vals.append(val)

        act_mag = float(sum(abs(v) for v in act_vals)) if act_vals else 0.0

        reward = float(self.straight_tail_weight * straight_score - self.tail_action_penalty * act_mag)
        details = {"straight_score": straight_score, "actuation": act_mag}
        return reward, details

    def _standing_components(self, action=None) -> dict:
        # compute numeric components in one place
        height_reward_dict = self.height_reward
        total_height_reward = sum(
            v["reward"] for v in height_reward_dict.values()
        )

        upright_reward, upright_score = self.upright_reward
        stillness_reward = self.stillness_reward
        head_dir_reward, head_dir_angle = self.head_direction_reward

        body_contacts = self.unwrapped.get_body_contacts()
        contact_penalty = self.body_contact_penalty * len(body_contacts)

        # fall penalty and termination flag come from chest_height
        fall_penalty = (
            self.fall_penalty
            if (
                height_reward_dict["chest"]["height"] < self.fall_threshold[0]
                or height_reward_dict["chest"]["height"] > self.fall_threshold[1]
            )
            else 0.0
        )

        # Leg position reward is part of standing diagnostics
        leg_pos_reward, leg_pos_details = self._leg_position_reward

        # Straight tail reward and actuation penalty
        straight_tail_reward, straight_tail_details = self.straight_tail_reward(action)

        return {
            "height_dict": height_reward_dict,
            "total_height_reward": float(total_height_reward),
            "upright_reward": float(upright_reward),
            "upright_score": float(upright_score),
            "stillness_reward": float(stillness_reward),
            "head_direction_reward": float(head_dir_reward),
            "head_direction_angle": head_dir_angle,
            "leg_pos_reward": float(leg_pos_reward),
            "leg_pos_details": leg_pos_details,
            "straight_tail_reward": float(straight_tail_reward),
            "straight_tail_details": straight_tail_details,
            "body_contacts": len(body_contacts),
            "contact_penalty": float(contact_penalty),
            "fall_penalty": float(fall_penalty),
        }

    def _stability_components(self) -> dict:
        """height, upright, and head directon components only."""

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        comps = self._standing_components(action)

        # if fall occurred, terminate
        if comps["fall_penalty"] != 0.0:
            terminated = True

        # sum main reward terms and add small alive bonus
        reward = (
            comps["total_height_reward"]
            + comps["upright_reward"]
            + comps["stillness_reward"]
            + comps["head_direction_reward"]
            + comps.get("leg_pos_reward", 0.0)
            + comps.get("straight_tail_reward", 0.0)
            + comps["contact_penalty"]
            + comps["fall_penalty"]
            + 0.5
        )

        # attach all computed diagnostics
        info.update(comps)
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
        env:LucyEnv,
        # Walking-specific params
        forward_weight: float = 2.0,

        gait_weight: float = 0.2,
        ctrl_cost_weight: float = 0.000,
        stillness_weight: float = -0.5,  # Negative reward for stillness to encourage movement
        body_contact_penalty: float = -2.0,
        leg_position_weight: float = 0.1,
        standing_reward_discount_factor: float = 0.2,
        **kwargs,
    ):
        # Initialize parent (standing wrapper)
        super().__init__(
            env,
            stillness_weight=stillness_weight,
            body_contact_penalty=body_contact_penalty,
            leg_position_weight=leg_position_weight,
            **kwargs,
        )

        if isinstance(env, LucyStandingWrapper):
            # Adaptively copy configuration from the existing standing wrapper
            for name, val in env.__dict__.items():
                if name.startswith("__") or name in ("env", "unwrapped"):
                    continue
                if isinstance(val, (int, float, str, bool, list, tuple, dict)):
                    setattr(self, name, val)

        self.forward_weight = forward_weight

        self.gait_weight = gait_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.standing_reward_discount_factor = standing_reward_discount_factor

        self._prev_x = None
        self._prev_foot_vels = None


    @property
    def _forward_reward(self):
        """Reward for forward velocity toward target (uses `acc_trunk`)."""

        accel = self.get_sensor_by_name("vel_trunk")
        forward_vel = accel[0]
        forward_reward = self.forward_weight * forward_vel

        return float(forward_reward), float(forward_vel)

    @property
    def _gait_reward(self):
        """Reward for gait based on limb accelerations.

        Computes mean linear acceleration magnitude across foot bodies by
        differencing successive velocity samples (a = dv/dt). Rewards are
        proportional to `self.gait_weight * mean_accel`. Only applies when
        the agent has positive forward velocity (encourages accelerations while
        moving forward).

        Returns (reward, mean_accel).
        """
        foot_names = [
            "front_left_foot_geom",
            "front_right_foot_geom",
            "hind_left_foot_geom",
            "hind_right_foot_geom",
        ]

        # Gather current linear velocities for each foot
        velocities = [self.env.get_part_velocity(name) for name in foot_names]
        lin_vels = [v["linear"] for v in velocities if v is not None and "linear" in v]
        if not lin_vels:
            return 0.0, 0.0

        # Initialize storage for previous velocities if not present
        if not hasattr(self, "_prev_foot_vels") or self._prev_foot_vels is None:
            self._prev_foot_vels = {}

        acc_mags = []
        for name, v in zip(foot_names, velocities):
            if v is None or "linear" not in v:
                continue
            cur = np.array(v["linear"], dtype=float)
            prev = self._prev_foot_vels.get(name)
            if prev is None:
                # First sample -> assume zero accel
                acc = np.zeros_like(cur)
            else:
                dt = float(getattr(self.unwrapped, "dt", 0.0)) or 1e-8
                acc = (cur - prev) / dt

            self._prev_foot_vels[name] = cur.copy() 
            acc_mags.append(float(np.linalg.norm(acc)))

        if not acc_mags:
            return 0.0, 0.0

        mean_acc = np.mean(acc_mags)

        # Reward proportional to mean limb acceleration magnitude
        gait_reward = float(self.gait_weight * mean_acc)
        return gait_reward, mean_acc

    def _walking_components(self, action) -> dict:
        """Compute walking-specific reward components for a single step.

        Returns a dict with numeric values that `step()` can sum and attach to
        `info`. This keeps the walking logic separated and testable.
        """
        # Ensure we have a baseline x position
        forward_reward, forward_vel = self._forward_reward

        # Gait reward based on foot linear accelerations (magnitude)
        gait_reward, feet_acc_mean = self._gait_reward
        # Also include feet-down count for diagnostics
        foot_contacts = self.unwrapped.get_foot_contacts()
        n_feet_down = int(sum(foot_contacts.values()))

        ctrl_cost = float(self.ctrl_cost_weight * np.sum(np.square(action)))

        # Update internal prev_x for next step
        self._prev_x = float(self.unwrapped.data.qpos[0])

        return {
            "forward_vel": float(forward_vel),
            "forward_reward": float(forward_reward),
            "gait_reward": float(gait_reward),
            "feet_down": n_feet_down,
            "feet_acc_mean": float(feet_acc_mean),
            "ctrl_cost": float(ctrl_cost),
            "x_position": float(self._prev_x),
        }

    def step(self, action):
        """Step the environment and add walking components on top of standing.

        Keeps `step()` concise by delegating walking computations to
        `_walking_components(action)` and standing computations to the parent.
        """
        # Advance simulation and get standing reward from parent
        obs, standing_reward, terminated, truncated, info = super().step(action)
        standing_reward = standing_reward * self.standing_reward_discount_factor

        # Compute walking components
        comps = self._walking_components(action)

        # Compose final reward (standing + walking - control cost)
        reward = float(
            standing_reward
            + comps["forward_reward"]
            + comps["gait_reward"]
            - comps["ctrl_cost"]
            + 0.5
        )

        # Attach diagnostics
        comps["standing_reward"] = float(standing_reward)

        # If fall occurred during standing, honor termination
        # (parent already sets termination flag; keep it as-is)
        info.update(comps)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._prev_x = None
        self._prev_foot_vels = None
        return super().reset(**kwargs)
