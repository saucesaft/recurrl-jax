from typing import Tuple

import jax
import jax.numpy as jnp

class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-5):
        self.mean    = jnp.zeros(shape)
        self.var     = jnp.ones(shape)
        self.count   = epsilon
        self.epsilon = epsilon

    def update(self, x: jax.Array):
        batch_mean  = jnp.mean(x, axis=0)
        batch_var   = jnp.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count

        self.count = total_count

    def normalize(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / jnp.sqrt(self.var + self.epsilon)

    def denormalize(self, x: jax.Array) -> jax.Array:
        return x * jnp.sqrt(self.var + self.epsilon) + self.mean

@jax.jit
def normalize_jit(x: jax.Array, mean: jax.Array, var: jax.Array, epsilon: float = 1e-5) -> jax.Array:
    return (x - mean) / jnp.sqrt(var + epsilon)

@jax.jit
def denormalize_jit(x: jax.Array, mean: jax.Array, var: jax.Array, epsilon: float = 1e-5) -> jax.Array:
    return x * jnp.sqrt(var + epsilon) + mean
