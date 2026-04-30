"""
generic MJX training step and reset utilities.

usage:
    1. define EnvSpec describing the qpos/qvel layout and contact geom IDs.
    2. implement reward_fn, termination_fn, reset_state_fn callbacks.
    3. call mjx_training_step each env step — physics, reward, auto-reset included.

callback signatures:
    reward_fn(state, reset_height_threshold)
        -> (reward, info, new_aux_state)

    termination_fn(state, progress_buf, max_episode_length, reset_height_threshold)
        -> (reset_mask, termination)

    reset_state_fn(key, data_batch, reset_mask, spec, use_dr, base_mjx, extra_cache, extra_cache_size)
        -> data_batch

    extra_state_fn(mjx_data)
        -> dict  (merged into state dict, optional)

state dict keys populated by the library:
    dof_pos, dof_vel, torques, obj_pos, obj_quat, obj_linvel, obj_angvel,
    actions, initial_dof_pos, prev_aux
    contact           (if spec.contact_geom_ids and spec.obj_geom_id >= 0)
    sensors           (if spec.sensor_ids)
    + anything from extra_state_fn
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from mujoco import mjx

from recurrl_jax.utils.domain_randomization import get_batched_fields, re_randomize_on_reset
from recurrl_jax.utils.quat_utils import rotate_vec_by_quat


@dataclass(frozen=True)
class EnvSpec:
    """static description of an MJX environment's state layout.

    all fields are tuples/ints so EnvSpec is hashable and usable as a JIT static arg.

    sensor_ids: indices into mjx_data.sensordata — add any custom sensors here and
    they will appear as state['sensors'] in reward_fn / termination_fn.
    """
    n_dofs: int
    dof_slice: Tuple[int, int]          # qpos/qvel indices for actuated DOFs
    obj_pos_slice: Tuple[int, int]      # qpos indices for object position
    obj_quat_slice: Tuple[int, int]     # qpos indices for object quaternion
    obj_linvel_slice: Tuple[int, int]   # qvel indices for object linear velocity
    obj_angvel_slice: Tuple[int, int]   # qvel indices for object angular velocity (local frame)
    default_pose: tuple                 # default DOF positions, length n_dofs
    contact_geom_ids: tuple = ()        # geom IDs to test contact against obj_geom_id
    obj_geom_id: int = -1               # geom ID of the manipulated object
    sensor_ids: tuple = ()              # indices into mjx_data.sensordata


@partial(jax.jit, static_argnames=[
    'control_freq_inv', 'use_domain_randomization', 'extra_cache_size',
    'spec', 'dr_params', 'reward_fn', 'termination_fn', 'reset_state_fn', 'extra_state_fn',
])
def mjx_training_step(
    actions: jax.Array,
    mjx_model,
    data_batch,
    progress_buf: jax.Array,
    initial_dof_pos: jax.Array,
    reset_height_threshold: float,
    max_episode_length: int,
    key: jax.Array,
    prev_aux_state,
    spec: EnvSpec,
    reward_fn: Callable,
    termination_fn: Callable,
    reset_state_fn: Callable,
    extra_state_fn: Optional[Callable] = None,
    control_freq_inv: int = 5,
    use_domain_randomization: bool = False,
    dr_params: tuple = (),
    base_mjx=None,
    action_scale: float = 1.0,
    extra_cache=None,
    extra_cache_size: int = 0,
):
    # 1. apply actions as offsets from default pose
    default_pose = jnp.array(spec.default_pose)
    data_batch = data_batch.replace(ctrl=default_pose + actions * action_scale)

    # 2. physics step — per-env model slice when DR is active
    _batched_fields = get_batched_fields(dr_params) if use_domain_randomization else ()

    def physics_step(_, data):
        if use_domain_randomization:
            def step_one(idx, d):
                env_model = mjx_model.tree_replace(
                    {f: getattr(mjx_model, f)[idx] for f in _batched_fields}
                )
                return mjx.step(env_model, d)
            return jax.vmap(step_one)(jnp.arange(data.qpos.shape[0]), data)
        else:
            return jax.vmap(mjx.step, in_axes=(None, 0))(mjx_model, data)

    new_data = jax.lax.fori_loop(0, control_freq_inv, physics_step, data_batch)

    # 3. extract state using spec
    qpos = new_data.qpos
    qvel = new_data.qvel
    ds, de = spec.dof_slice
    ops, ope = spec.obj_pos_slice
    oqs, oqe = spec.obj_quat_slice
    ols, ole = spec.obj_linvel_slice
    oas, oae = spec.obj_angvel_slice

    obj_quat = qpos[:, oqs:oqe]
    obj_angvel_local = qvel[:, oas:oae]

    state = {
        'dof_pos':         qpos[:, ds:de],
        'dof_vel':         qvel[:, ds:de],
        'torques':         new_data.qfrc_actuator[:, ds:de],
        'obj_pos':         qpos[:, ops:ope],
        'obj_quat':        obj_quat,
        'obj_linvel':      qvel[:, ols:ole],
        'obj_angvel':      rotate_vec_by_quat(obj_angvel_local, obj_quat),
        'actions':         actions,
        'initial_dof_pos': initial_dof_pos,
        'prev_aux':        prev_aux_state,
    }

    # 4. contact detection (generic, enabled when spec has contact IDs)
    if spec.contact_geom_ids and spec.obj_geom_id >= 0:
        _cids = jnp.array(spec.contact_geom_ids)
        _oid = spec.obj_geom_id

        def check_contact(g1, g2):
            a1 = jnp.any(jnp.isin(g1, _cids))
            a2 = jnp.any(jnp.isin(g2, _cids))
            b1 = jnp.any(g1 == _oid)
            b2 = jnp.any(g2 == _oid)
            return jnp.logical_or(
                jnp.logical_and(a1, b2),
                jnp.logical_and(a2, b1),
            )

        state['contact'] = jax.vmap(check_contact)(
            new_data.contact.geom1, new_data.contact.geom2
        )

    # 5. sensor data
    if spec.sensor_ids:
        state['sensors'] = new_data.sensordata[:, jnp.array(spec.sensor_ids)]

    # 6. custom extra state from escape-hatch callback
    if extra_state_fn is not None:
        state.update(extra_state_fn(new_data))

    # 7. reward
    reward, info, new_aux_state = reward_fn(state, reset_height_threshold)

    # 8. termination
    new_progress_buf = progress_buf + 1
    reset_mask, termination = termination_fn(
        state, new_progress_buf, max_episode_length, reset_height_threshold
    )

    # 9. auto-reset via lax.cond (only traces reset branch when any env resets)
    done = reset_mask

    reset_partial = partial(
        _mjx_reset_jit,
        use_domain_randomization=use_domain_randomization,
        dr_params=dr_params,
        base_mjx=base_mjx,
        extra_cache_size=extra_cache_size,
        spec=spec,
        reset_state_fn=reset_state_fn,
    )

    def no_reset(k, m, d, mask, ec):
        return m, d

    new_mjx_model, new_data_reset = jax.lax.cond(
        jnp.any(reset_mask),
        reset_partial,
        no_reset,
        key, mjx_model, new_data, reset_mask, extra_cache,
    )

    # 10. zero per-env aux state and progress buffer for reset envs
    new_progress_buf = jnp.where(reset_mask, 0, new_progress_buf)
    new_aux_state = jax.tree_util.tree_map(
        lambda x: jnp.where(
            reset_mask.reshape((-1,) + (1,) * (x.ndim - 1)) if x.ndim > 1 else reset_mask,
            jnp.zeros_like(x),
            x,
        ),
        new_aux_state,
    )

    return (
        state, reward, done, termination, info,
        new_data_reset, new_progress_buf, new_mjx_model, new_aux_state,
    )


def _mjx_reset_jit(
    key, mjx_model, data_batch, reset_mask, extra_cache,
    use_domain_randomization: bool = False,
    dr_params: tuple = (),
    base_mjx=None,
    extra_cache_size: int = 0,
    spec: EnvSpec = None,
    reset_state_fn: Callable = None,
):
    """generic reset: fresh data, env-specific state init, DR re-randomize."""
    num_envs = reset_mask.shape[0]

    # create fresh mjx data for each env (using its own model slice if DR)
    if use_domain_randomization:
        _batched_fields = get_batched_fields(dr_params)

        def make_data_slice(idx, model):
            env_model = model.tree_replace(
                {f: getattr(model, f)[idx] for f in _batched_fields}
            )
            return mjx.make_data(env_model)

        fresh_data = jax.vmap(make_data_slice, in_axes=(0, None))(
            jnp.arange(num_envs), mjx_model
        )
    else:
        fresh_data = jax.vmap(lambda _: mjx.make_data(mjx_model))(jnp.arange(num_envs))

    # mask-merge: only overwrite resetting envs
    def select(fresh, old):
        mask = reset_mask.reshape((num_envs,) + (1,) * (fresh.ndim - 1))
        return jnp.where(mask, fresh, old)

    data_batch = jax.tree_util.tree_map(select, fresh_data, data_batch)

    # env-specific: set qpos / qvel / ctrl for resetting envs
    key, reset_key = jr.split(key)
    data_batch = reset_state_fn(
        reset_key, data_batch, reset_mask, spec,
        use_domain_randomization, base_mjx, extra_cache, extra_cache_size,
    )

    # DR re-randomize physics parameters for resetting envs
    if use_domain_randomization:
        mjx_model = re_randomize_on_reset(base_mjx, mjx_model, reset_mask, key, dr_params)

    return mjx_model, data_batch
