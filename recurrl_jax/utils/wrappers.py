import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp

from recurrl_jax.utils.running_mean_std import RunningMeanStd, rms_update, rms_normalize


# functional env protocol
# -----------------------
# an env (or wrapper) can opt into fully-jitted rollouts by exposing
#     functional_state                      (property, get/set) -> pytree
#     functional_step(fstate, actions)      pure, jittable
#         -> (fstate, obs, reward, terminated, truncated, info)
#     supports_functional                   (property) -> bool
# the agent then pulls functional_state off the env, runs the whole rollout
# inside one lax.scan and writes the final state back, so the env object
# stays the source of truth between updates (e.g. sync_curriculum_step keeps
# working). envs without it fall back to the python step loop.
#
# the inner env wrapped by MJXGymWrapper uses the same idea, minus obs:
#     functional_state                      (property, get/set) -> pytree
#     functional_step(fstate, actions)
#         -> (fstate, reward, reset_mask, termination, info)

def is_functional_env(env):
    # look the method up on the class: gym.Wrapper forwards unknown instance
    # attributes to the wrapped env, which would wrongly report support.
    return hasattr(type(env), 'functional_step') and bool(getattr(env, 'supports_functional', False))


class MJXGymWrapper(gym.Env):
    """base gym wrapper for MJX vectorized environments.

    subclasses implement two methods:
        _make_env(key) -> underlying MJX env
        _get_obs()     -> jnp.ndarray of shape (num_envs, obs_dim)

    the underlying env must return
        (state, reward, reset_mask, termination, info, ...)
    from its step() call.

    to enable fully-jitted rollouts, the underlying env also implements the
    inner functional protocol (see top of file) and the subclass implements
        _get_obs_fn(env_fstate, last_actions) -> jnp.ndarray (num_envs, obs_dim)
    as a pure function of the inner env's functional state.

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

        self._functional_step_jit = jax.jit(self.functional_step)

    def _make_env(self, key):
        """create and return the underlying MJX environment."""
        raise NotImplementedError

    def _get_obs(self) -> jnp.ndarray:
        """build and return observation array of shape (num_envs, obs_dim)."""
        raise NotImplementedError

    def _get_obs_fn(self, env_fstate, last_actions) -> jnp.ndarray:
        """pure variant of _get_obs, from the inner env's functional state."""
        raise NotImplementedError

    @property
    def supports_functional(self):
        return (hasattr(type(self.env), 'functional_step')
                and type(self)._get_obs_fn is not MJXGymWrapper._get_obs_fn)

    @property
    def functional_state(self):
        fstate = {'env': self.env.functional_state, 'last_actions': self.last_actions}
        if self.normalize_obs:
            fstate['rms'] = self.running_mean_std.state
        return fstate

    @functional_state.setter
    def functional_state(self, fstate):
        self.env.functional_state = fstate['env']
        self.last_actions = fstate['last_actions']
        # a shared (eval) rms is read-only here, never write it back
        if self.normalize_obs and self.update_norm_stats:
            self.running_mean_std.state = fstate['rms']

    def functional_step(self, fstate, actions):
        env_fstate, reward, reset_mask, termination, info = self.env.functional_step(fstate['env'], actions)

        last_actions = jnp.where(reset_mask[:, None], jnp.zeros_like(actions), actions)

        obs = self._get_obs_fn(env_fstate, last_actions)
        new_fstate = {'env': env_fstate, 'last_actions': last_actions}
        if self.normalize_obs:
            obs, new_fstate['rms'] = self._normalize_fn(fstate['rms'], obs)

        reward = jnp.nan_to_num(reward * self.reward_scale, nan=0.0, posinf=0.0, neginf=0.0)
        truncation = jnp.logical_and(reset_mask, jnp.logical_not(termination))

        return new_fstate, obs, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        self.env.reset()
        self.last_actions = jnp.zeros((self.num_envs, self.action_dim))
        return self._normalize(self._get_obs()), {}

    def step(self, actions):
        actions_jax = jnp.asarray(actions)

        if self.supports_functional:
            # one fused dispatch instead of a dozen eager ops
            self.functional_state, obs, reward, termination, truncation, info = \
                self._functional_step_jit(self.functional_state, actions_jax)
            return obs, reward, termination, truncation, info

        _, reward, reset_mask, termination, info, *_ = self.env.step(actions_jax)

        self.last_actions = jnp.where(
            reset_mask[:, None], jnp.zeros((self.num_envs, self.action_dim)), actions_jax
        )

        obs = self._normalize(self._get_obs())
        reward = jnp.nan_to_num(reward * self.reward_scale, nan=0.0, posinf=0.0, neginf=0.0)
        truncation = jnp.logical_and(reset_mask, jnp.logical_not(termination))

        return obs, reward, termination, truncation, info

    def _normalize_fn(self, stats, obs):
        """pure normalize; returns (obs, updated stats)."""
        policy_obs = obs[:, :self.policy_obs_dim]
        if self.update_norm_stats:
            stats = rms_update(stats, policy_obs)
        normalized = rms_normalize(stats, policy_obs, self.running_mean_std.epsilon)
        if self.policy_obs_dim < obs.shape[-1]:
            normalized = jnp.concatenate([normalized, obs[:, self.policy_obs_dim:]], axis=-1)
        return normalized, stats

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


def _track_episodes(ep_ret, ep_len, rewards, terminated, truncated):
    ep_ret = ep_ret + rewards
    ep_len = ep_len + 1
    dones = jnp.logical_or(terminated, truncated)
    episode = {'done': dones, 'r': ep_ret, 'l': ep_len}
    # reset stats for done envs
    ep_ret = jnp.where(dones, 0.0, ep_ret)
    ep_len = jnp.where(dones, 0, ep_len)
    return ep_ret, ep_len, episode

_track_episodes_jit = jax.jit(_track_episodes)

class VectorEpisodeStatisticsWrapper(gym.Wrapper):
    """tracks per-env episode return/length on device.

    every step adds infos['episode'] = {'done', 'r', 'l'}, each of shape
    (num_envs,): 'r'/'l' are the running return/length *including* this step,
    only meaningful where 'done' is set. kept as masked arrays (no host sync)
    so it can live inside a jitted rollout; the agent reduces them on device.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = jnp.zeros(self.num_envs, dtype=jnp.float32)
        self.episode_lengths = jnp.zeros(self.num_envs, dtype=jnp.int32)

    @property
    def supports_functional(self):
        return is_functional_env(self.env)

    @property
    def functional_state(self):
        return {'env': self.env.functional_state,
                'ep_ret': self.episode_returns, 'ep_len': self.episode_lengths}

    @functional_state.setter
    def functional_state(self, fstate):
        self.env.functional_state = fstate['env']
        self.episode_returns = fstate['ep_ret']
        self.episode_lengths = fstate['ep_len']

    def functional_step(self, fstate, action):
        env_fstate, obs, rewards, terminated, truncated, infos = self.env.functional_step(fstate['env'], action)
        ep_ret, ep_len, episode = _track_episodes(fstate['ep_ret'], fstate['ep_len'], rewards, terminated, truncated)
        new_fstate = {'env': env_fstate, 'ep_ret': ep_ret, 'ep_len': ep_len}
        return new_fstate, obs, rewards, terminated, truncated, {**infos, 'episode': episode}

    def step(self, action):
        obs, rewards, terminated, truncated, infos = self.env.step(action)
        self.episode_returns, self.episode_lengths, episode = _track_episodes_jit(
            self.episode_returns, self.episode_lengths,
            jnp.asarray(rewards, dtype=jnp.float32), jnp.asarray(terminated), jnp.asarray(truncated))
        return obs, rewards, terminated, truncated, {**infos, 'episode': episode}

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
