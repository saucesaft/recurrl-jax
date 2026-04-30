import jax
import jax.numpy as jnp

from examples.leap_hand.env import MJXLeapHandEnv
from examples.leap_hand.observation_buffer import build_asymmetric_observation
from recurrl_jax.utils.wrappers import MJXGymWrapper
from recurrl_jax.utils.quat_utils import rotate_vec_by_quat


class LeapHandGymWrapper(MJXGymWrapper):
    def __init__(
        self,
        num_envs: int = 1,
        use_domain_randomization: bool = False,
        action_scale: float = 0.6,
        action_ema_alpha: float = 0.0,
        grasp_cache_path: str = None,
        **kwargs,
    ):
        self._use_dr = use_domain_randomization
        self._action_scale = action_scale
        self._action_ema_alpha = action_ema_alpha
        self._grasp_cache_path = grasp_cache_path
        super().__init__(obs_dim=105, action_dim=16, num_envs=num_envs, policy_obs_dim=32, **kwargs)

    def _make_env(self, key):
        return MJXLeapHandEnv(
            xml_path='xmls/scene_mjx_cube.xml',
            num_envs=self.num_envs,
            key=key,
            action_scale=self._action_scale,
            action_ema_alpha=self._action_ema_alpha,
            use_domain_randomization=self._use_dr,
            grasp_cache_path=self._grasp_cache_path,
        )

    def _get_obs(self) -> jnp.ndarray:
        self.key, key = jax.random.split(self.key)
        mjx_data = self.env.mjx_data_batch
        qpos = mjx_data.qpos
        qvel = mjx_data.qvel
        cube_quat = qpos[:, 19:23]

        return build_asymmetric_observation(
            joint_angles=qpos[:, :16],
            joint_velocities=qvel[:, :16],
            joint_torques=mjx_data.qfrc_actuator[:, :16],
            last_action=self.last_actions,
            fingertip_positions=self.env.get_fingertip_positions(),
            cube_pos=qpos[:, 16:19],
            palm_pos=self.env.get_palm_position(),
            cube_quat=cube_quat,
            cube_angvel=rotate_vec_by_quat(qvel[:, 19:22], cube_quat),
            cube_linvel=qvel[:, 16:19],
            key=key,
        )
