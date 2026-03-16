__credits__ = ["Kallinteris-Andreas", "Rushiv Arora"]

from pathlib import Path

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class AntGoalEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str | None = None,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward_weight: float = 0.1,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = False,
        healthy_z_range=(0.3, 2.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        if xml_file is None:
            # Use bundled asset relative to this file for portability.
            xml_file = Path(__file__).resolve().parent / "assets" / "antgoal.xml"

        xml_file = str(xml_file)
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._healthy_reward_weight = healthy_reward_weight

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        
        # Gymnasium MujocoEnv init
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.safety_geom_id = None
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if name == "safety_marker":
                self.safety_geom_id = i
                break

        if self.safety_geom_id is None:
            raise RuntimeError("safety_marker geom not found in XML")

        self.metadata = {
            "render_modes": self.metadata["render_modes"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    # -------------------------
    # Properties / helpers
    # -------------------------
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        return np.isfinite(state).all() and (min_z <= state[2] <= max_z)

    @property
    def terminated(self):
        return (not self.is_healthy) if self._terminate_when_unhealthy else False

    # -------------------------
    # Step / Reset
    # -------------------------
    def step(self, action):

        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = float(xy_velocity[0]), float(xy_velocity[1])

        # ===== reward: speed tracking (v = 0.25) =====
        v_target = 0.25
        if x_velocity < v_target:
            speed_reward = x_velocity / v_target
        else:
            speed_reward = 1.0

        rewards = speed_reward
        
        observation = self._get_obs()
        reward = rewards

        # gymnasium API
        terminated = self.terminated
        truncated = False 

        # ===== safety: body height =====
        z_height = float(self._get_ball_obs()[2])
        z_threshold = 1.0  # tune this

        safety_signal = 1.0 if z_height >= z_threshold else 0.0

        info = {
            "safety": safety_signal,
            "z_height": z_height,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }

        if self.render_mode == "human":
            self.render()


        ## Marker
        if safety_signal == 1.0:
            self.model.geom_rgba[self.safety_geom_id] = np.array([0.0, 1.0, 0.0, 0.3])
        else:
            self.model.geom_rgba[self.safety_geom_id] = np.array([1.0, 0.0, 0.0, 0.3])

        torso_pos = self.get_body_com("torso")
        marker_pos = torso_pos + np.array([0.0, 0.0, 2.5])
        self.model.geom_pos[self.safety_geom_id] = marker_pos

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # gymnasium: self.data
        position = self.data.qpos.flatten().copy()
        velocity = self.data.qvel.flatten().copy()

        if self._exclude_current_positions_from_observation:
            ant_state = position[7+2:]

        torso_pos = self.get_body_com("torso")
        ball_pos  = self.get_body_com("falling_ball")
        rel_pos = ball_pos - torso_pos

        observations = np.concatenate((rel_pos, ant_state, velocity[6:])).ravel()
        return observations

    def _get_ball_obs(self):
        ball_pos  = self.get_body_com("falling_ball")

        return np.array(ball_pos, dtype=np.float64)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        # -------------------------
        # Randomize ball (x, y)
        # -------------------------
        ball_xy_range = 0.4  # 
        qpos[0] = self.np_random.uniform(-ball_xy_range, ball_xy_range)  # x
        qpos[1] = self.np_random.uniform(-ball_xy_range, ball_xy_range)  # y

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {}
