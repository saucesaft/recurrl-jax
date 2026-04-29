import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp

from recurrl_jax.utils.running_mean_std import RunningMeanStd


class MJXGymWrapper(gym.Env):
    """base gym wrapper for MJX vectorized environments.

    subclasses implement two methods:
        _make_env(key) -> underlying MJX env
        _get_obs()     -> jnp.ndarray of shape (num_envs, obs_dim)

    the underlying env must return
        (state, reward, reset_mask, termination, info, ...)
    from its step() call.

    args:
        obs_dim:                  total observation dimension
        action_dim:               action dimension
        num_envs:                 number of parallel environments
        policy_obs_dim:           first N dims are normalized; rest pass through raw.
                                  defaults to obs_dim (normalize everything).
        normalize_obs:            whether to apply RunningMeanStd normalization
        reward_scale:             scalar multiplied into reward
        shared_running_mean_std:  share stats from a training wrapper (for eval)
        update_norm_stats:        whether to update RunningMeanStd on each step
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_envs: int = 1,
        policy_obs_dim: int = None,
        normalize_obs: bool = True,
        reward_scale: float = 1.0,
        shared_running_mean_std=None,
        update_norm_stats: bool = True,
    ):
        self.num_envs = num_envs
        self.reward_scale = reward_scale
        self.normalize_obs = normalize_obs
        self.update_norm_stats = update_norm_stats
        self.policy_obs_dim = policy_obs_dim if policy_obs_dim is not None else obs_dim
        self.action_dim = action_dim

        self.key = jax.random.PRNGKey(0)
        self.key, env_key = jax.random.split(self.key)
        self.env = self._make_env(env_key)

        self.last_actions = jnp.zeros((num_envs, action_dim))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        if normalize_obs:
            self.running_mean_std = (
                shared_running_mean_std
                if shared_running_mean_std is not None
                else RunningMeanStd(shape=(self.policy_obs_dim,))
            )

    def _make_env(self, key):
        """create and return the underlying MJX environment."""
        raise NotImplementedError

    def _get_obs(self) -> jnp.ndarray:
        """build and return observation array of shape (num_envs, obs_dim)."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        self.env.reset()
        self.last_actions = jnp.zeros((self.num_envs, self.action_dim))
        return self._normalize(self._get_obs()), {}

    def step(self, actions):
        actions_jax = jnp.asarray(actions)
        _, reward, reset_mask, termination, info, *_ = self.env.step(actions_jax)

        self.last_actions = jnp.where(
            reset_mask[:, None], jnp.zeros((self.num_envs, self.action_dim)), actions_jax
        )

        obs = self._normalize(self._get_obs())
        reward = jnp.nan_to_num(reward * self.reward_scale, nan=0.0, posinf=0.0, neginf=0.0)
        truncation = jnp.logical_and(reset_mask, jnp.logical_not(termination))

        return obs, reward, termination, truncation, info

    def _normalize(self, obs: jnp.ndarray) -> jnp.ndarray:
        if not self.normalize_obs:
            return obs
        policy_obs = obs[:, :self.policy_obs_dim]
        if self.update_norm_stats:
            self.running_mean_std.update(policy_obs)
        normalized = self.running_mean_std.normalize(policy_obs)
        if self.policy_obs_dim < obs.shape[-1]:
            return jnp.concatenate([normalized, obs[:, self.policy_obs_dim:]], axis=-1)
        return normalized

    def close(self):
        pass

class VectorEpisodeStatisticsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = jnp.zeros(self.num_envs, dtype=jnp.float32)
        self.episode_lengths = jnp.zeros(self.num_envs, dtype=jnp.int32)

    def step(self, action):
        obs, rewards, terminated, truncated, infos = self.env.step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1

        dones = jnp.logical_or(terminated, truncated)

        dones_np = np.array(dones)

        if np.any(dones_np):
             for i in range(self.num_envs):
                if dones_np[i]:
                    info_item = {
                        "r": float(self.episode_returns[i]),
                        "l": int(self.episode_lengths[i])
                    }
                    infos.setdefault("episode", []).append(info_item)

        # reset stats for done envs
        self.episode_returns = jnp.where(dones, 0.0, self.episode_returns)
        self.episode_lengths = jnp.where(dones, 0, self.episode_lengths)

        return obs, rewards, terminated, truncated, infos

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

class SqueezeWrapper(gym.Wrapper):
    """squeezes batch dim of a VectorEnv with num_envs=1"""
    def __init__(self, env):
        super().__init__(env)
        assert getattr(env, "num_envs", 1) == 1, "SqueezeWrapper only supports num_envs=1"
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[0], info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs[0], float(r[0]), bool(term[0]), bool(trunc[0]), info
