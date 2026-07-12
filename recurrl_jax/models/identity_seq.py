import jax.numpy as jnp
import flax.linen as nn


class IdentitySeq(nn.Module):
    """pass-through seq slot. no recurrence; memory is a dummy scalar."""

    @nn.compact
    def __call__(self, inputs, terminations, last_memory):
        return inputs, last_memory

    @staticmethod
    def initialize_state():
        return jnp.zeros(())
