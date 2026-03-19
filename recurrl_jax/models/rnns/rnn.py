import jax.numpy as jnp
import jax.nn as nn
import numpy as np
import jax

from recurrl_jax.utils.recurrent_utils import tree_index
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

class LSTM(nn.Module):
    d_model: int
    reset_on_terminate: bool = True

    @nn.compact
    def __call__(self,inputs,terminations,last_state):
        #inputs; TXinput_dim
        #carry:
        #last_hidden: d_model
        reset_on_terminate=self.reset_on_terminate
        d_model=self.d_model

        class LSTMout(nn.Module):

            @nn.compact
            def __call__(self,carry,inputs):

                # separate inputs to their respective variables
                inputs,terminate=inputs

                if reset_on_terminate:

                    # reset (zero-out) hidden state on termination
                    carry = jax.lax.cond(
                            terminate,
                            lambda: jax.tree.map(lambda x: jnp.zeros_like(x), carry), lambda: carry
                        )

                # return the new carry and hidden state
                (new_c, new_h), new_h = nn.OptimizedLSTMCell(
                        features = d_model,
                        kernel_init = orthogonal(jnp.sqrt(2)),
                        recurrent_kernel_init = orthogonal(jnp.sqrt(2)),
                        bias_init = constant(0.0)
                    )(carry, inputs)

                return (new_c, new_h), ((new_c, new_h),new_h)
        
        # flax's vmap - a single cell 
        model = nn.scan( LSTMout, variable_broadcast="params", split_rngs={"params": False} )
        
        # call the layer with respective params
        carry, (new_states,y_t) = model()(last_state, (inputs, terminations))

        return y_t,new_states

    def initialize_state(self):
        return (jnp.zeros((self.d_model,)),jnp.zeros((self.d_model,)))



class LSTMMultiLayer(nn.Module):
    d_model: int
    n_layers: int
    reset_on_terminate: bool = True

    @nn.compact
    def __call__(self, inputs, terminations, last_states):

        new_memory=[None]*self.n_layers
        
        # stack multiple layers
        for i in range(self.n_layers):
            if i == 0:
                y_t, new_memory[i] = LSTM(self.d_model,self.reset_on_terminate)(inputs, terminations,last_states[i])
            else:
                y_t, new_memory[i] = LSTM(self.d_model,self.reset_on_terminate)(y_t, terminations,last_states[i])

        # extract the last state of the sequence
        new_memory = tree_index(new_memory,-1)

        return y_t, new_memory

    @staticmethod
    def initialize_state(d_model,n_layers):
        return [(jnp.zeros((d_model,)),jnp.zeros((d_model,))) for _ in range(n_layers)]

class GRU(nn.Module):
    d_model: int
    reset_on_terminate: bool = True

    @nn.compact
    def __call__(self,inputs,terminations,last_state):
        #inputs; TXinput_dim
        #carry:
        #last_hidden: d_model

        reset_on_terminate=self.reset_on_terminate
        d_model=self.d_model
        
        class GRUout(nn.Module):
            @nn.compact
            def __call__(self,carry,inputs):

                # separate inputs to their respective variables
                inputs,terminate=inputs

                if reset_on_terminate:


                    # reset (zero-out) hidden state on termination
                    carry = jax.lax.cond(
                            terminate,
                            lambda: jax.tree.map(lambda x: jnp.zeros_like(x), carry), lambda: carry
                        )

                # return the new carry and hidden state
                new_c, new_h = nn.GRUCell(
                        features = d_model,
                        kernel_init = orthogonal(jnp.sqrt(2)),
                        recurrent_kernel_init = orthogonal(jnp.sqrt(2)),
                        bias_init = constant(0.0)
                    )(carry, inputs)

                return new_c, (new_c,new_h)

        model = nn.scan( GRUout,variable_broadcast="params", split_rngs={"params": False} )

        # call the layer with respective params
        carry, (new_states,y_t) = model()(last_state, (inputs,terminations))

        return y_t,new_states

    def initialize_state(self):
        return jnp.zeros((self.d_model,))


class GRUMultiLayer(nn.Module):
    d_model: int
    n_layers: int
    reset_on_terminate: bool = True

    @nn.compact
    def __call__(self, inputs, terminations, last_states):

        new_memory=[None]*self.n_layers
        
        # stack multiple layers
        for i in range(self.n_layers):
            if i == 0:
                y_t, new_memory[i] = GRU(self.d_model,self.reset_on_terminate)(inputs, terminations,last_states[i])
            else:
                y_t, new_memory[i] = GRU(self.d_model,self.reset_on_terminate)(y_t, terminations,last_states[i])

        # extract the last state of the sequence
        new_memory=tree_index(new_memory,-1)

        return y_t, new_memory

    @staticmethod
    def initialize_state(d_model,n_layers):
        return [jnp.zeros((d_model,)) for _ in range(n_layers)]
