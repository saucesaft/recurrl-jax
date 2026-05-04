import jax.numpy as jnp
import flax.linen as nn

from flax.linen.initializers import constant, orthogonal


class MLPSeq(nn.Module):
    """MLP drop-in for the seq_model slot. no recurrent state; memory is a dummy scalar."""
    d_model: int
    n_layers: int

    @nn.compact
    def __call__(self, inputs, terminations, last_memory):
        x = inputs
        for _ in range(self.n_layers):
            x = nn.Dense(
                self.d_model,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.tanh(x)
        return x, last_memory

    @staticmethod
    def initialize_state():
        return jnp.zeros(())
