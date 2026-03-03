import flax.linen as nn
import jax.numpy as jnp

from typing import Callable
from recurrl_jax.utils.recurrent_utils import tree_index
from flax.linen.initializers import constant, orthogonal

class ActorCriticModel(nn.Module):
    repr_model_fn:Callable
    seq_model_fn:Callable
    actor_fn:Callable
    critic_fn:Callable
    use_asymmetric_obs:bool=False  # asymmetric actor-critic observations
    policy_obs_dim:int=102  # split index for asymmetric obs


    def setup(self):
        self.repr_model=self.repr_model_fn()
        self.seq_model=self.seq_model_fn()
        self.actor=self.actor_fn()
        self.critic=self.critic_fn()

    def __call__(self,inputs,terminations,last_memory):
        if self.use_asymmetric_obs:
            # split observations at policy_obs_dim
            policy_obs = inputs[..., :self.policy_obs_dim]
            privileged_obs = inputs[..., self.policy_obs_dim:]

            rep = self.repr_model(policy_obs)
            rep = rep.reshape(rep.shape[0], -1)
            seq_rep, memory = self.seq_model(rep, terminations, last_memory)

            # actor uses recurrent features only
            actor_out = self.actor(seq_rep)

            # critic gets recurrent features + privileged observations
            critic_input = jnp.concatenate([seq_rep, privileged_obs], axis=-1)
            critic_out = self.critic(critic_input)
        else:
            # symmetric: same obs for both actor and critic
            rep = self.repr_model(inputs)
            rep = rep.reshape(rep.shape[0], -1)
            seq_rep, memory = self.seq_model(rep, terminations, last_memory)
            actor_out = self.actor(seq_rep)
            critic_out = self.critic(seq_rep)

        return actor_out, critic_out, memory
