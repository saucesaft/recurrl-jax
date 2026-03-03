import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
from functools import partial

@jax.jit
def check_for_physics_explosion(obs: jax.Array) -> jax.Array:
    has_nan = jnp.any(jnp.isnan(obs), axis=-1)
    has_inf = jnp.any(jnp.isinf(obs), axis=-1)
    return jnp.logical_or(has_nan, has_inf)

@jax.jit
def add_observation_noise(joint_angles: jax.Array, key: jax.Array, noise_level: float = 0.05) -> jax.Array:
    noise = jr.uniform(key, shape=joint_angles.shape, minval=-noise_level, maxval=noise_level)
    return joint_angles + noise

@jax.jit
def build_asymmetric_observation(
    joint_angles: jax.Array,           # (num_envs, 16)
    joint_velocities: jax.Array,       # (num_envs, 16)
    joint_torques: jax.Array,          # (num_envs, 16)
    last_action: jax.Array,            # (num_envs, 16)
    fingertip_positions: jax.Array,    # (num_envs, 12) - 4 tips × 3 coords
    cube_pos: jax.Array,               # (num_envs, 3)
    palm_pos: jax.Array,               # (num_envs, 3)
    cube_quat: jax.Array,              # (num_envs, 4)
    cube_angvel: jax.Array,            # (num_envs, 3)
    cube_linvel: jax.Array,            # (num_envs, 3)
    key: jax.Array,                    # Random key for noise
    noise_level: float = 0.3          # Observation noise magnitude
) -> jax.Array:
    # add noise to joint angles for policy
    noisy_joint_angles = add_observation_noise(joint_angles, key, noise_level)

    # policy observation (32D)
    policy_obs = jnp.concatenate([
        noisy_joint_angles,  # 16
        last_action,         # 16
    ], axis=-1)

    # cube position error (palm_pos - cube_pos)
    cube_pos_error = palm_pos - cube_pos

    # privileged state (105D)
    privileged_state = jnp.concatenate([
        policy_obs,              # 32 (noisy state + last action)
        joint_angles,            # 16 (true angles)
        joint_velocities,        # 16
        joint_torques,           # 16
        fingertip_positions,     # 12
        cube_pos_error,          # 3
        cube_quat,               # 4
        cube_angvel,             # 3
        cube_linvel,             # 3
    ], axis=-1)

    return privileged_state  # (num_envs, 105)
