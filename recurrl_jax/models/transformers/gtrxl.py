import flax.linen as nn
import jax.numpy as jnp
import jax

from flax import struct
from typing import Optional, Dict, List,Callable
from recurrl_jax.utils.recurrent_utils import jax_pad, masked_fill


class PositionalEmbedding(nn.Module):
    embedding_dim: int

    def setup(self):
        inv_freq = 1 / (10000 ** (jnp.arange(0.0, self.embedding_dim, 2.0) / self.embedding_dim))
        self.inv_freq = inv_freq

    def __call__(self, pos_seq):
        sinusoid_inp = jnp.outer(pos_seq, self.inv_freq)
        pos_embedding = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
        return jnp.expand_dims(pos_embedding, axis=1)


class GRUGatingUnit(nn.Module):
    """GRU gating unit used in GTrXL."""
    input_dim: int
    bg: float = 2.0

    def setup(self):
        self.Wr = nn.Dense(self.input_dim, use_bias=False)
        self.Ur = nn.Dense(self.input_dim, use_bias=False)
        self.Wz = nn.Dense(self.input_dim, use_bias=False)
        self.Uz = nn.Dense(self.input_dim, use_bias=False)
        self.Wg = nn.Dense(self.input_dim, use_bias=False)
        self.Ug = nn.Dense(self.input_dim, use_bias=False)
        self.bgp = self.param('bgp', jax.nn.initializers.constant(self.bg), (self.input_dim,))
        self.sigmoid = nn.sigmoid
        self.tanh = nn.tanh

    def __call__(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bgp)
        h = self.tanh(self.Wg(y) + self.Ug(jnp.multiply(r , x)))
        g=jnp.multiply(1-z,x)+jnp.multiply(z,h)
        return g



class AttentionXL(nn.Module):
    """AttentionXL module"""
    input_dim:int
    head_num: int
    head_dim: int
    dropout: float = 0.0
    train: bool = True


    def setup(self):
        self.attention_kv = nn.Dense(self.head_num * self.head_dim * 2)
        self.attention_q = nn.Dense(self.head_num * self.head_dim)
        self.project = nn.Dense(self.input_dim)
        self.project_pos = nn.Dense(self.head_dim * self.head_num)
        self.scale = 1 / (self.head_dim ** 0.5)
        self.dropout_fn = nn.Dropout(self.dropout,deterministic=not self.train)

    def _rel_shift(self, x: jnp.ndarray,zero_upper:bool=False):
        """relatively shift attention score matrix
        see https://github.com/kimiyoung/transformer-xl/issues/8
        """
        x_padded=jax_pad(x,[1,0]) #step 1
        x_padded=x_padded.reshape(x.shape[0],x.shape[1],x.shape[3]+1,x.shape[2]) #step 2
        x=x_padded[:,:,1:].reshape(*x.shape) #step 3
        if zero_upper:
            ones = jnp.expand_dims(jnp.expand_dims(jnp.ones((x.shape[2], x.shape[3]), dtype=x.dtype), 0), 0)
            x = x * jnp.tril(ones, x.shape[3] - x.shape[2])
        return x


    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        pos_embedding: jnp.ndarray,
        full_input: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        key, value = jnp.split(kv,2,axis=-1)
        query = self.attention_q(inputs)
        r = self.project_pos(pos_embedding)

        key = key.reshape(full_seq, bs, self.head_num, self.head_dim)
        query = query.reshape(cur_seq, bs, self.head_num, self.head_dim)
        value = value.reshape(cur_seq + prev_seq, bs, self.head_num, self.head_dim)
        r = r.reshape(full_seq, self.head_num, self.head_dim)

        # (query + u) * key^T
        q_u = query + u
        content_attn=jnp.transpose(q_u,(1,2,0,3))@jnp.transpose(key,(1,2,3,0))

        # (query + v) * R^T
        q_v = query + v
        position_attn=jnp.transpose(q_v,(1,2,0,3))@jnp.transpose(r,(1,2,0))
        position_attn = self._rel_shift(position_attn)

        attn = content_attn + position_attn
        attn=jnp.multiply(attn,self.scale)

        # fill -inf where mask is True
        mask = jnp.expand_dims(jnp.transpose(mask,(2,0,1)),1)
        assert mask.shape[2:] == attn.shape[2:]
        attn=jnp.where(mask,-float("1e20"),attn)
        attn = nn.softmax(attn, axis=-1)
        attn = self.dropout_fn(attn)

        attn_vec = attn @ jnp.transpose(value,(1,2,0,3))
        attn_vec = jnp.transpose(attn_vec,(2,0,1,3))

        attn_vec = attn_vec.ravel().reshape(cur_seq, bs, self.head_num * self.head_dim)
        output = self.dropout_fn(self.project(attn_vec))
        return output


class GatedTransformerXLLayer(nn.Module):
    input_dim: int
    head_dim: int
    hidden_dim: int
    head_num: int
    mlp_num: int
    dropout: float
    activation: Callable
    gru_gating: bool = True
    gru_bias: float = 2.
    train: bool = True

    def setup(self):
        self.gating = self.gru_gating
        if self.gating is True:
            self.gate1 = GRUGatingUnit(self.input_dim, self.gru_bias)
            self.gate2 = GRUGatingUnit(self.input_dim, self.gru_bias)
        self.attention = AttentionXL(
            self.input_dim,
            self.head_num,
            self.head_dim,
            self.dropout,
        )
        layers = []
        dims = [self.input_dim] + [self.hidden_dim] * (self.mlp_num - 1) + [self.input_dim]
        for i in range(self.mlp_num):
            layers.append(nn.Sequential([nn.Dense(features=dims[i + 1],),self.activation]))
            if i != self.mlp_num - 1:
                layers.append(nn.Dropout(self.dropout,deterministic=not self.train))
        layers.append(nn.Dropout(self.dropout,deterministic=not self.train))
        self.mlp = nn.Sequential(layers)
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.dropout_fn=nn.Dropout(self.dropout,deterministic=not self.train)

    def __call__(self,inputs,pos_embedding,u,v,memory,mask=None):
        # concat memory with input across sequence dimension
        full_input = jnp.concatenate([memory, inputs], axis=0)
        x1 = self.layernorm1(full_input)
        a1 = self.dropout_fn(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
        x2 = self.layernorm2(o1)
        m2 = self.dropout_fn(self.mlp(x2))
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2


class GTrXL(nn.Module):
    head_dim: int = 128
    embedding_dim: int = 256
    head_num: int = 2
    mlp_num: int = 2
    layer_num: int = 3
    memory_len: int = 64
    dropout_ratio: float = 0.
    activation: nn.Module = nn.relu
    gru_gating: bool = True
    gru_bias: float = 2.
    use_embedding_layer: bool = True
    train: bool = True
    reset_on_terminate: bool = True

    def setup(self):
        self.pos_embedding = PositionalEmbedding(self.embedding_dim)
        layers = []
        dims = [self.embedding_dim] + [self.embedding_dim] * self.layer_num
        self.dropout = nn.Dropout(self.dropout_ratio,deterministic=not self.train)
        if self.use_embedding_layer:
            self.embedding = nn.Sequential([nn.Dense(features=self.embedding_dim), self.activation])
        for i in range(self.layer_num):
            layers.append(
                GatedTransformerXLLayer(
                    dims[i], self.head_dim, self.embedding_dim, self.head_num, self.mlp_num, self.dropout_ratio, self.activation, self.gru_gating,
                    self.gru_bias,train=self.train
                )
            )
        self.layers=layers
        self.u = self.param('u',jax.nn.initializers.zeros,(self.head_num, self.head_dim))
        self.v = self.param('v',jax.nn.initializers.zeros,(self.head_num, self.head_dim))

    @staticmethod
    def init_memory(memory_len: int = 20,batch_size: int = 1,embedding_dim: int = 256,
                    layer_num: int = 3):
        memory = jnp.zeros(
            (layer_num + 1, memory_len, batch_size, embedding_dim)
        )

        return memory

    @staticmethod
    def initialize_state(memory_len,embedding_dim,layer_num):
        last_mask=jnp.ones((memory_len,),dtype=bool)
        return (GTrXL.init_memory(memory_len,1,embedding_dim,layer_num),last_mask)

    @staticmethod
    def update_memory(memory, hidden_state: List[jnp.ndarray]):
        """update memory given a sequence of hidden states"""
        if memory is None or hidden_state is None:
            raise ValueError('Failed to update memory! Memory would be None')
        layer_num_plus1, memory_len, batch_size, embedding_dim = memory.shape
        layer_num = layer_num_plus1 - 1
        sequence_len = hidden_state[0].shape[0]
        new_memory = []
        end = memory_len + sequence_len
        beg = max(0, end - memory_len)
        for i in range(layer_num + 1):
            m = memory[i]
            h = hidden_state[i]
            cat = jnp.concatenate([m, h], axis=0)
            new_memory.append(jax.lax.stop_gradient(cat[beg:end])) #stop gradient to avoid backprop through memory
        new_memory = jnp.stack(new_memory, axis=0)
        return new_memory

    def __call__(self,inputs,terminations,last_memory):
        #reshape inputs to (T,1,input_dim)
        last_state,last_mask=last_memory #so that memory has a tree structure
        inputs=jnp.expand_dims(inputs,1)
        cur_seq, bs = inputs.shape[:2]
        if self.use_embedding_layer:
            inputs = self.dropout(self.embedding(inputs))
        prev_seq = self.memory_len
        full_seq = cur_seq + prev_seq

        def term_scan(carry,x):
            term,idx=x
            if self.reset_on_terminate:
                new_mask=jax.lax.cond(term,lambda: jnp.ones_like(carry),lambda:carry)
            else:
                new_mask=carry
            attn_mask=new_mask
            attn_mask=attn_mask.at[self.memory_len+idx].set(False)
            new_carry=attn_mask
            return new_carry,attn_mask

        carry=jnp.concatenate([last_mask,jnp.ones((cur_seq,),dtype=bool)])
        new_mask,attn_mask=jax.lax.scan(term_scan,carry,(terminations,jnp.arange(cur_seq)),)

        attn_mask=jnp.expand_dims(attn_mask,-1)
        new_mask=new_mask[-prev_seq:]


        pos_ips = jnp.arange(full_seq - 1, -1, -1.0,dtype=float)
        pos_embedding = self.pos_embedding(pos_ips)
        pos_embedding = self.dropout(pos_embedding)

        hidden_state = [inputs]
        out = inputs
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(
                out,
                pos_embedding,
                self.u,
                self.v,
                mask=attn_mask,
                memory=last_state[i],
            )
            hidden_state.append(jnp.copy(out))

        out = self.dropout(out)
        new_state=self.update_memory(last_state,hidden_state)

        #reshape out to (T,embedding_dim)
        out=jnp.squeeze(out,1)
        return out,(new_state,new_mask.copy())
