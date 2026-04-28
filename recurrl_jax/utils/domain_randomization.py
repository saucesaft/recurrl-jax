"""
generic domain randomization for MJX environments.

usage:
    1. Define a list of DRParam describing what to randomize.
    2. Call create_batched_randomized_models once at env init.
    3. Call re_randomize_on_reset at each episode reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx


@dataclass(frozen=True)
class DRParam:
    """specifies one randomizable model parameter.

    indices and linked_indices must be tuples (not numpy arrays) for hashability
    when used as a static JIT argument.
    """
    field: str              # MJX model field name, e.g. "body_mass"
    indices: tuple          # row indices to randomize
    mode: str               # "multiply" | "add" | "set"
    low: float              # uniform lower bound
    high: float             # uniform upper bound
    col: Optional[int] = None               # column index (None = full row)
    linked_field: Optional[str] = None      # second field updated from same random values
    linked_col: Optional[int] = None        # column in linked field
    linked_indices: Optional[tuple] = None  # rows in linked field (defaults to indices)
    linked_transform: Optional[Callable] = None  # transform(new_vals) -> linked_vals
    init_only: bool = False  # if True, only randomize at init, not on episode reset


def get_batched_fields(params: Tuple[DRParam, ...]) -> Tuple[str, ...]:
    """return the unique set of model fields that will be batched."""
    fields = []
    seen = set()
    for p in params:
        if p.field not in seen:
            fields.append(p.field)
            seen.add(p.field)
        if p.linked_field and p.linked_field not in seen:
            fields.append(p.linked_field)
            seen.add(p.linked_field)
    return tuple(fields)


def _apply_params(
    base_mjx: mjx.Model,
    existing_updates: dict,
    params: Tuple[DRParam, ...],
    keys: jax.Array,
    num_envs: int,
    batched: bool,
) -> dict:
    """
    generate randomized field values and accumulate into existing_updates dict.

    if batched=True, each field gets shape [num_envs, *field_shape].
    if batched=False, each field gets shape [*field_shape] (single env).
    """
    updates = dict(existing_updates)

    for i, param in enumerate(params):
        base_field = getattr(base_mjx, param.field)
        indices = jnp.array(param.indices)

        if param.col is None:
            base_vals = base_field[indices]
        else:
            base_vals = base_field[indices, param.col]

        if batched:
            noise = jax.random.uniform(
                keys[i], (num_envs,) + base_vals.shape, minval=param.low, maxval=param.high
            )
        else:
            noise = jax.random.uniform(keys[i], base_vals.shape, minval=param.low, maxval=param.high)

        if param.mode == "multiply":
            new_vals = base_vals * noise
        elif param.mode == "add":
            new_vals = base_vals + noise
        else:
            new_vals = noise

        if param.field not in updates:
            if batched:
                updates[param.field] = jnp.repeat(base_field[None], num_envs, axis=0)
            else:
                updates[param.field] = base_field

        if param.col is None:
            if batched:
                updates[param.field] = updates[param.field].at[:, indices].set(new_vals)
            else:
                updates[param.field] = updates[param.field].at[indices].set(new_vals)
        else:
            if batched:
                updates[param.field] = updates[param.field].at[:, indices, param.col].set(new_vals)
            else:
                updates[param.field] = updates[param.field].at[indices, param.col].set(new_vals)

        if param.linked_field is not None:
            linked_vals = param.linked_transform(new_vals)
            linked_base = getattr(base_mjx, param.linked_field)
            linked_indices = jnp.array(
                param.linked_indices if param.linked_indices is not None else param.indices
            )
            if param.linked_field not in updates:
                if batched:
                    updates[param.linked_field] = jnp.repeat(linked_base[None], num_envs, axis=0)
                else:
                    updates[param.linked_field] = linked_base
            if param.linked_col is None:
                if batched:
                    updates[param.linked_field] = updates[param.linked_field].at[:, linked_indices].set(linked_vals)
                else:
                    updates[param.linked_field] = updates[param.linked_field].at[linked_indices].set(linked_vals)
            else:
                if batched:
                    updates[param.linked_field] = updates[param.linked_field].at[:, linked_indices, param.linked_col].set(linked_vals)
                else:
                    updates[param.linked_field] = updates[param.linked_field].at[linked_indices, param.linked_col].set(linked_vals)

    return updates


def create_batched_randomized_models(
    mjx_model: mjx.Model,
    num_envs: int,
    rng: jax.Array,
    params: Tuple[DRParam, ...],
) -> Tuple[mjx.Model, mjx.Model]:
    """
    create batched MJX models with per-environment randomized parameters.

    only the randomized fields have shape [num_envs, ...]; all other fields
    remain shared scalars (memory-efficient for large num_envs).

    returns:
        batched_model: randomized fields shaped [num_envs, ...]
        in_axes: axis=0 for batched fields, None for shared fields
    """
    keys = jax.random.split(rng, len(params))
    updates = _apply_params(mjx_model, {}, params, keys, num_envs, batched=True)

    batched_model = mjx_model.tree_replace(updates)

    in_axes = jax.tree.map(lambda _: None, mjx_model)
    in_axes = in_axes.tree_replace({k: 0 for k in updates})

    return batched_model, in_axes


def re_randomize_on_reset(
    base_mjx: mjx.Model,
    batched_model: mjx.Model,
    reset_mask: jax.Array,
    rng: jax.Array,
    params: Tuple[DRParam, ...],
) -> mjx.Model:
    """
    re-randomize only environments flagged by reset_mask.

    base_mjx: original un-batched model (base values; always randomize relative to these)
    batched_model: current batched model with randomized fields shaped [num_envs, ...]
    reset_mask: bool array of shape [num_envs]; True = this env is resetting
    """
    num_envs = reset_mask.shape[0]
    reset_params = tuple(p for p in params if not p.init_only)
    keys = jax.random.split(rng, len(reset_params))

    # Generate candidate values for ALL envs (as if all are resetting)
    candidate_updates = _apply_params(base_mjx, {}, reset_params, keys, num_envs, batched=True)

    # For each updated field, fill in base (current) values where reset_mask is False
    final_updates = {}
    for field, candidate in candidate_updates.items():
        current = getattr(batched_model, field)
        mask = reset_mask.reshape((num_envs,) + (1,) * (candidate.ndim - 1))
        final_updates[field] = jnp.where(mask, candidate, current)

    return batched_model.tree_replace(final_updates)
