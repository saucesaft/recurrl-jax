import flax.linen as nn
import jax.numpy as jnp

from flax.linen.initializers import constant, orthogonal

class Flatten(nn.Module):
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

def flatten_repr_model():
    def thurn():
        return Flatten()
    return thurn
