from typing import Tuple

import jax
import jax.numpy as jnp


# pure (mean, var, count) helpers, usable inside a jitted rollout
def rms_update(stats, x: jax.Array):
    mean, var, count = stats
    batch_mean  = jnp.mean(x, axis=0)
    batch_var   = jnp.var(x, axis=0)
    batch_count = x.shape[0]

    delta = batch_mean - mean
    total_count = count + batch_count

    new_mean = mean + delta * batch_count / total_count

    m_a = var * count
    m_b = batch_var * batch_count
    M2  = m_a + m_b + delta**2 * count * batch_count / total_count
    new_var = M2 / total_count

    return new_mean, new_var, total_count

def rms_normalize(stats, x: jax.Array, epsilon: float = 1e-5) -> jax.Array:
    mean, var, _ = stats
    return (x - mean) / jnp.sqrt(var + epsilon)

_rms_update_jit = jax.jit(rms_update)

class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-5):
        self.mean    = jnp.zeros(shape)
        self.var     = jnp.ones(shape)
        self.count   = jnp.asarray(epsilon, dtype=jnp.float32)
        self.epsilon = epsilon

    @property
    def state(self):
        return (self.mean, self.var, jnp.asarray(self.count, dtype=jnp.float32))

    @state.setter
    def state(self, stats):
        self.mean, self.var, self.count = stats

    def update(self, x: jax.Array):
        self.state = _rms_update_jit(self.state, x)

    def normalize(self, x: jax.Array) -> jax.Array:
        return normalize_jit(x, self.mean, self.var, self.epsilon)

    def denormalize(self, x: jax.Array) -> jax.Array:
        return denormalize_jit(x, self.mean, self.var, self.epsilon)

@jax.jit
def normalize_jit(x: jax.Array, mean: jax.Array, var: jax.Array, epsilon: float = 1e-5) -> jax.Array:
    return (x - mean) / jnp.sqrt(var + epsilon)

@jax.jit
def denormalize_jit(x: jax.Array, mean: jax.Array, var: jax.Array, epsilon: float = 1e-5) -> jax.Array:
    return x * jnp.sqrt(var + epsilon) + mean
