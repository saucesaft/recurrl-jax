import gymnasium as gym
import numpy as np
import jax.numpy as jnp

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
