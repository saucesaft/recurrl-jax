import jax
import jax.numpy as jnp


def sparse_auto_reset(reset_fn, reset_args, done, next_state, max_resets):
    """auto-reset only the envs flagged in ``done``.

    the naive auto-reset runs ``vmap(reset_fn)`` over *every* env each step and
    then keeps a handful of rows; with an expensive reset (mjx.forward, DR
    sampling) that is a large share of the step. here the done envs are
    gathered, reset and scattered back in chunks of ``max_resets`` (a
    while_loop: zero iterations when nothing is done, one in the common case,
    more only when many envs finish together). per env the result is
    identical to ``where(done, vmap(reset_fn)(*reset_args), next_state)``.

    reset_fn:   per-env reset, called as reset_fn(*args_for_one_env)
    reset_args: tuple of per-env batched args (leading axis num_envs), e.g. keys
    done:       bool (num_envs,)
    next_state: batched state pytree (leading axis num_envs)
    max_resets: static int, envs reset per chunk
    """
    num_envs = done.shape[0]
    max_resets = max(1, min(int(max_resets), num_envs))

    def reset_chunk(carry):
        state, pending = carry
        # indices of pending envs, padded with an out-of-bounds index whose
        # writes are dropped by the scatters below
        idx = jnp.nonzero(pending, size=max_resets, fill_value=num_envs)[0]
        args = jax.tree_util.tree_map(lambda a: a[jnp.minimum(idx, num_envs - 1)], reset_args)
        fresh = jax.vmap(reset_fn)(*args)
        state = jax.tree_util.tree_map(lambda n, f: n.at[idx].set(f.astype(n.dtype), mode='drop'), state, fresh)
        return state, pending.at[idx].set(False, mode='drop')

    state, _ = jax.lax.while_loop(lambda carry: jnp.any(carry[1]), reset_chunk, (next_state, done))
    return state
