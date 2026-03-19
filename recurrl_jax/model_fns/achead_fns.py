import flax.linen as nn
import jax.numpy as jnp

from flax.linen.initializers import constant, orthogonal

# actor fn factory
def actor_model_discete(dense_dim,action_space):
    def thurn():
        return nn.Sequential(
                [
                    nn.Dense(
                        dense_dim,
                        kernel_init = orthogonal(jnp.sqrt(2)),
                        bias_init = constant(0.0)
                    ),
                    nn.tanh,
                    nn.Dense(
                        action_space,
                        kernel_init = orthogonal(jnp.sqrt(2)),
                        bias_init = constant(0.0)
                    )
                ]
            )

    return thurn

class ActorContinuous(nn.Module):
    dense_dim: int
    action_dim: int
    hidden_dims: tuple = (256, 128)
    initial_log_std: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def setup(self):
        layers = []
        # first hidden layer
        layers.extend([
            nn.Dense( self.dense_dim, kernel_init = orthogonal(jnp.sqrt(2)), bias_init = constant(0.0) ),
            nn.elu,
        ])
        # additional hidden layers
        for dim in self.hidden_dims:
            layers.extend([
                nn.Dense( dim, kernel_init = orthogonal(jnp.sqrt(2)), bias_init = constant(0.0) ),
                nn.elu, #TODO been able to change activation function in config
            ])
        # final layer with small init for small initial actions
        layers.append(
            nn.Dense( self.action_dim, kernel_init = orthogonal(0.01), bias_init = constant(0.0) )
        )
        self.net = nn.Sequential(layers)

        # learnable log_std
        self.log_std = self.param('log_std', constant(self.initial_log_std), (self.action_dim,))

    def __call__(self, x):
        # tanh bounds outputs to [-1, 1]
        # TODO activate/deactivate bounds in config
        mean = jnp.tanh(self.net(x))

        # clamp log_std
        log_std = jnp.clip( self.log_std, self.log_std_min, self.log_std_max )
        log_std = jnp.broadcast_to( log_std, mean.shape )

        return mean, log_std

def actor_model_continuous(dense_dim, action_dim, hidden_dims=None, initial_log_std=-0.5,
                           log_std_min=-5.0, log_std_max=2.0):
    if hidden_dims is None:
        hidden_dims = (256, 128)

    # convert to tuple for Flax hashability
    hidden_dims = tuple(hidden_dims)

    def thurn():
        return ActorContinuous(
                dense_dim, action_dim, hidden_dims = hidden_dims,
                initial_log_std = initial_log_std,
                log_std_min=log_std_min, log_std_max = log_std_max
            )

    return thurn

# critic fn factory
def critic_model(dense_dim, hidden_dims=None):
    
    # default init
    if hidden_dims is None:
        hidden_dims = (256, 128)

    def thurn():
        layers = []
        # first hidden layer
        layers.extend([
            nn.Dense( dense_dim, kernel_init = orthogonal(jnp.sqrt(2)), bias_init = constant(0.0) ),
            nn.elu,
        ])
        # additional hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Dense(dim, kernel_init = orthogonal(jnp.sqrt(2)), bias_init = constant(0.0)),
                nn.elu,
            ])
        # output layer
        layers.extend([
            nn.Dense(1, kernel_init = orthogonal(1.0), bias_init = constant(0.0)),
            lambda x: jnp.squeeze(x, axis=-1),
        ])
        return nn.Sequential(layers)

    return thurn
