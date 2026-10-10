import jax
import jax.numpy as jnp
import jax.random as jr

from examples.anymal.anymal_env import DiffAnymal
from recurrl_jax.utils.wrappers import MJXGymWrapper
from recurrl_jax.utils.auto_reset import sparse_auto_reset


class BraxAnymalEnv:
    """vectorized wrapper around anymal env with auto-reset."""

    def __init__(self, num_envs: int, key: jax.Array, **env_kwargs):
        self.env = DiffAnymal(**env_kwargs)
        self.num_envs = num_envs
        self.key = key
        self.state = None
        # done envs are auto-reset in chunks of this size (see sparse_auto_reset)
        self.max_resets_per_step = max(1, num_envs // 16)

        self._vmap_reset = jax.jit(jax.vmap(self.env.reset))
        self._vmap_step = jax.jit(jax.vmap(self.env.step))
        self._functional_step_jit = jax.jit(self.functional_step)

        self.reset()

    def reset(self):
        self.key, *keys = jr.split(self.key, self.num_envs + 1)
        keys = jnp.stack(keys)
        self.state = self._vmap_reset(keys)
        return self.state

    @property
    def functional_state(self):
        return (self.state, self.key)

    @functional_state.setter
    def functional_state(self, fstate):
        self.state, self.key = fstate

    def functional_step(self, fstate, actions: jnp.ndarray):
        """pure step + auto-reset; (state, key) in, (state, key) out."""
        state, key = fstate
        key, reset_key = jr.split(key)

        next_state = jax.vmap(self.env.step)(state, actions)
        done = next_state.done.astype(bool)

        # auto-reset terminated environments: keep next_state for ongoing
        # envs, fresh reset states for done envs (only done envs are reset)
        reset_keys = jr.split(reset_key, self.num_envs)
        new_state = sparse_auto_reset(
            self.env.reset, (reset_keys,), done, next_state, self.max_resets_per_step)

        # curriculum_step is global training progress — never reset on episode end
        new_state = new_state.replace(
            info={**new_state.info,
                  'curriculum_step': next_state.info['curriculum_step']}
        )

        return (new_state, key), next_state.reward, done, done, next_state.metrics

    def step(self, actions: jnp.ndarray):
        fstate, reward, reset_mask, termination, metrics = self._functional_step_jit(
            self.functional_state, actions)
        self.functional_state = fstate
        return self.state, reward, reset_mask, termination, metrics

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

    def _get_obs_fn(self, env_fstate, last_actions) -> jnp.ndarray:
        state, _ = env_fstate
        return state.obs
