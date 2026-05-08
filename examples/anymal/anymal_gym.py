import jax
import jax.numpy as jnp
import jax.random as jr

from examples.anymal.anymal_env import DiffAnymal
from recurrl_jax.utils.wrappers import MJXGymWrapper


class BraxAnymalEnv:
    """vectorized wrapper around anymal env with auto-reset."""

    def __init__(self, num_envs: int, key: jax.Array, **env_kwargs):
        self.env = DiffAnymal(**env_kwargs)
        self.num_envs = num_envs
        self.key = key
        self.state = None

        self._vmap_reset = jax.jit(jax.vmap(self.env.reset))
        self._vmap_step = jax.jit(jax.vmap(self.env.step))

        self.reset()

    def reset(self):
        self.key, *keys = jr.split(self.key, self.num_envs + 1)
        keys = jnp.stack(keys)
        self.state = self._vmap_reset(keys)
        return self.state

    def step(self, actions: jnp.ndarray):
        self.key, reset_key = jr.split(self.key)

        next_state = self._vmap_step(self.state, actions)
        done = next_state.done.astype(bool)

        # auto-reset terminated environments
        reset_keys = jr.split(reset_key, self.num_envs)
        reset_states = self._vmap_reset(reset_keys)

        # keep next_state for ongoing envs; use reset_states for done envs
        self.state = jax.tree_util.tree_map(
            lambda r, n: jnp.where(
                done.reshape((-1,) + (1,) * (r.ndim - 1)) if r.ndim > 1 else done,
                r, n,
            ),
            reset_states, next_state,
        )

        # curriculum_step is global training progress — never reset on episode end
        self.state = self.state.replace(
            info={**self.state.info,
                  'curriculum_step': next_state.info['curriculum_step']}
        )

        return self.state, next_state.reward, done, done, next_state.metrics

    def sync_curriculum_step(self, env_steps: int):
        """Sync curriculum counter into all envs' state_info."""
        cs = jnp.full((self.num_envs,), env_steps, dtype=jnp.int32)
        self.state = self.state.replace(
            info={**self.state.info, 'curriculum_step': cs}
        )


class AnymalGymWrapper(MJXGymWrapper):
    """clean Gym wrapper for ANYmal locomotion."""

    def __init__(self, num_envs: int = 512, **kwargs):
        self._num_envs = num_envs
        
        # separate MJXGymWrapper args from env_kwargs
        wrapper_arg_names = [
            'obs_dim', 'policy_obs_dim', 'normalize_obs', 
            'reward_scale', 'shared_running_mean_std', 'update_norm_stats'
        ]
        wrapper_kwargs = {k: v for k, v in kwargs.items() if k in wrapper_arg_names}
        self._env_kwargs = {k: v for k, v in kwargs.items() if k not in wrapper_arg_names}

        super().__init__(
            obs_dim=kwargs.get('obs_dim', 49),
            action_dim=12,
            num_envs=num_envs,
            **wrapper_kwargs,
        )

    def _make_env(self, key):
        return BraxAnymalEnv(num_envs=self._num_envs, key=key, **self._env_kwargs)

    def _get_obs(self) -> jnp.ndarray:
        return self.env.state.obs
