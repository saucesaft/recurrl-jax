import gymnasium as gym
import numpy as np
import jax.numpy as jnp


class InvertedPendulumVecEnv(gym.Env):
    """vectorized InvertedPendulum-v4 (MuJoCo) wrapper for recurrl_jax.

    policy outputs actions in [-1, 1]; scaled to env's [-3, 3] force range.
    """

    def __init__(self, num_envs: int = 32):
        self.num_envs = num_envs

        self._venv = gym.vector.make("InvertedPendulum-v4", num_envs=num_envs)

        obs_dim = self._venv.single_observation_space.shape[0]   # 4
        action_dim = self._venv.single_action_space.shape[0]     # 1
        self._action_scale = float(self._venv.single_action_space.high[0])  # 3.0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = self._venv.reset(seed=seed)
        return jnp.asarray(obs, dtype=jnp.float32), {}

    def step(self, actions):
        actions_np = np.asarray(actions) * self._action_scale
        obs, rewards, terminated, truncated, _ = self._venv.step(actions_np)
        return (
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.asarray(rewards, dtype=jnp.float32),
            jnp.asarray(terminated, dtype=bool),
            jnp.asarray(truncated, dtype=bool),
            {},
        )

    def close(self):
        self._venv.close()
