"""JAX quaternion utilities for angular velocity computation."""

import jax
import jax.numpy as jnp


@jax.jit
def quat_conjugate(q: jax.Array) -> jax.Array:
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


@jax.jit
def quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return jnp.stack([w, x, y, z], axis=-1)


@jax.jit
def quat_to_euler(q: jax.Array) -> jax.Array:
    """convert quaternion [w, x, y, z] to euler angles (roll, pitch, yaw)"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.stack([roll, pitch, yaw], axis=-1)


@jax.jit
def compute_angvel_finite_diff(
    current_quat: jax.Array,
    previous_quat: jax.Array,
    dt: float
) -> jax.Array:
    """compute angular velocity via finite differences on quaternions"""
    q_prev_conj = quat_conjugate(previous_quat)
    q_delta = quat_mul(current_quat, q_prev_conj)
    euler_delta = quat_to_euler(q_delta)
    angvel = euler_delta / dt
    return angvel


@jax.jit
def rotate_vec_by_quat(v: jax.Array, q: jax.Array) -> jax.Array:
    """rotate vector v by quaternion q [w, x, y, z]"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # uv = cross(q_xyz, v)
    uv_x = y * vz - z * vy
    uv_y = z * vx - x * vz
    uv_z = x * vy - y * vx

    # uuv = cross(q_xyz, uv)
    uuv_x = y * uv_z - z * uv_y
    uuv_y = z * uv_x - x * uv_z
    uuv_z = x * uv_y - y * uv_x

    # result = v + 2 * (w * uv + uuv)
    res_x = vx + 2 * (w * uv_x + uuv_x)
    res_y = vy + 2 * (w * uv_y + uuv_y)
    res_z = vz + 2 * (w * uv_z + uuv_z)

    return jnp.stack([res_x, res_y, res_z], axis=-1)
