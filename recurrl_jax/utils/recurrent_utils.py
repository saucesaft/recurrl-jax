import jax.numpy as jnp
import jax
import jax.lax as lax

def masked_fill(mask, a, fill):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))

def jax_pad(input, pad, mode='constant', value=0):
  """JAX implementation of constant padding (last N dims)"""
  if mode != 'constant':
    raise NotImplementedError("Only mode='constant' is implemented")
  assert len(pad) % 2 == 0
  assert len(pad) // 2 <= input.ndim
  pad = list(zip(*[iter(pad)]*2))
  pad += [(0, 0)] * (input.ndim - len(pad))
  return lax.pad(
      input,
      padding_config=[(i, j, 0) for i, j in pad[::-1]],
      padding_value=jnp.array(value, input.dtype))

def tree_dot(a,b):
    prod=jax.tree.map(lambda x, y: jnp.tensordot(x,y,axes=y.ndim), a, b)
    sum_tree=jnp.stack(jax.tree_util.tree_flatten(prod)[0],axis=0).sum(axis=0)
    return sum_tree


def tree_sum(a,b):
    return jax.tree.map(lambda x,y: x+y,a,b)

def tree_scalar_multiply(a,scalar):
    return jax.tree.map(lambda x:x*scalar,a)

def tree_subtract(a,b):
    return jax.tree.map(lambda x,y: x-y,a,b)


def tree_index(tree,i):
    return jax.tree.map(lambda x:x[i] ,tree)

def stack_trees(tree_list,axis=0):
    return jax.tree.map(lambda *x:jnp.stack(x,axis=axis) ,tree_list[0],*tree_list[1:])
