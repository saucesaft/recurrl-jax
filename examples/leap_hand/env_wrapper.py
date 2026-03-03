import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from examples.leap_hand.env import MJXLeapHandEnv
from examples.leap_hand.observation_buffer import build_asymmetric_observation
from recurrl_jax.utils.running_mean_std import RunningMeanStd, normalize_jit
from recurrl_jax.utils.quat_utils import rotate_vec_by_quat

class LeapHandGymWrapper(gym.Env):
    def __init__(self, num_envs=1, use_domain_randomization=False, normalize_obs=True, reward_scale=0.05, action_scale=0.6, grasp_cache_path=None, shared_running_mean_std=None, update_norm_stats=True, history_len=1):
        self.num_envs = num_envs
        self.reward_scale = reward_scale
        self.key = jax.random.PRNGKey(0)
        self.key, env_key = jax.random.split(self.key)

        # Initialize underlying MJX environment
        self.env = MJXLeapHandEnv(
            xml_path='xmls/scene_mjx_cube.xml',
            num_envs=num_envs,
            key=env_key,
            action_scale=action_scale,
            use_domain_randomization=use_domain_randomization,
            grasp_cache_path=grasp_cache_path
        )

        self.normalize_obs = normalize_obs
        self.update_norm_stats = update_norm_stats

        # asymmetric actor-critic observations:
        # policy (actor) gets 32D: noisy joints (16) + last action (16)
        # critic gets full 105D: policy obs (32) + privileged state (73)
        self.policy_obs_dim = 32
        self.priv_obs_dim = 105
        self.obs_dim = self.priv_obs_dim
        self.action_dim = 16

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.priv_obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        if self.normalize_obs:
            if shared_running_mean_std is not None:
                self.running_mean_std = shared_running_mean_std
            else:
                self.running_mean_std = RunningMeanStd(shape=(self.policy_obs_dim,))

        # State persistence
        self.last_actions = jnp.zeros((num_envs, self.action_dim))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)

        self.env.reset()
        self.last_actions = jnp.zeros((self.num_envs, self.action_dim))

        obs = self._get_obs()
        return obs, {}

    def step(self, actions):
        if not isinstance(actions, jax.Array):
             actions_jax = jnp.array(actions)
        else:
             actions_jax = actions

        raw_state, reward, reset_mask, termination, info, _, _, _, _ = self.env.step(actions_jax)

        self.last_actions = actions_jax

        self.last_actions = jnp.where(
            reset_mask[:, None],
            jnp.zeros((self.num_envs, self.action_dim)),
            self.last_actions
        )

        obs = self._get_obs()

        reward = reward * self.reward_scale

        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        truncation = jnp.logical_and(reset_mask, jnp.logical_not(termination))

        return obs, reward, termination, truncation, info

    def _get_obs(self):
        self.key, key = jax.random.split(self.key)

        mjx_data = self.env.mjx_data_batch
        qpos = mjx_data.qpos
        qvel = mjx_data.qvel

        joint_angles = qpos[:, :16]
        joint_velocities = qvel[:, :16]
        joint_torques = mjx_data.qfrc_actuator[:, :16]

        fingertip_positions = self.env.get_fingertip_positions()
        cube_pos = qpos[:, 16:19]
        palm_pos = self.env.get_palm_position()
        cube_quat = qpos[:, 19:23]

        cube_angvel_local = qvel[:, 19:22]
        cube_angvel = rotate_vec_by_quat(cube_angvel_local, cube_quat)

        cube_linvel = qvel[:, 16:19]

        full_obs = build_asymmetric_observation(
            joint_angles=joint_angles,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            last_action=self.last_actions,
            fingertip_positions=fingertip_positions,
            cube_pos=cube_pos,
            palm_pos=palm_pos,
            cube_quat=cube_quat,
            cube_angvel=cube_angvel,
            cube_linvel=cube_linvel,
            key=key,
        )

        policy_obs = full_obs[:, :self.policy_obs_dim]

        if self.normalize_obs:
            if self.update_norm_stats:
                self.running_mean_std.update(policy_obs)
            policy_obs_normalized = self.running_mean_std.normalize(policy_obs)

            full_obs_normalized = jnp.concatenate([
                policy_obs_normalized,
                full_obs[:, self.policy_obs_dim:]
            ], axis=-1)
            return full_obs_normalized
        else:
            return full_obs

    def close(self):
        pass
