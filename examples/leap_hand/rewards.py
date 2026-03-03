import jax
import jax.numpy as jnp

def compute_total_reward(
    object_angvel: jax.Array,
    object_pos: jax.Array,
    # State for smoothed reward
    prev_angvel_z_smooth: jax.Array = None,  # (num_envs,) - previous smoothed angvel
    smooth_alpha: float = 0.9,  # EMA factor: higher = more smoothing
    # Keep these params for compatibility but don't use them in simple mode
    object_linvel: jax.Array = None,
    actions: jax.Array = None,
    dof_vel: jax.Array = None,
    torques: jax.Array = None,
    dof_pos: jax.Array = None,
    init_dof_pos: jax.Array = None,
    reset_height_threshold: float = -0.05,
    fingertip_cube_contact: jax.Array = None,
    # Reward scales
    rotation_scale: float = 1.0,
    work_penalty_scale: float = 0.0,
    pose_diff_penalty_scale: float = 0.0,
    torque_penalty_scale: float = 0.0,
    linvel_penalty_scale: float = 0.0,
) -> tuple:

    # current angular velocity
    angvel_z = object_angvel[:, 2]
    angvel_z_safe = jnp.nan_to_num(angvel_z, nan=0.0, posinf=0.0, neginf=0.0)

    # EMA-smoothed angular velocity (oscillation cancels, sustained rotation accumulates)
    if prev_angvel_z_smooth is None:
        angvel_z_smooth = angvel_z_safe
    else:
        angvel_z_smooth = smooth_alpha * prev_angvel_z_smooth + (1.0 - smooth_alpha) * angvel_z_safe

    rotation_reward = angvel_z_smooth * rotation_scale

    termination_penalty = (object_pos[:, 2] < reset_height_threshold).astype(jnp.float32) * -100.0

    total_reward = rotation_reward + termination_penalty

    # Clip total reward to prevent extreme values
    total_reward = jnp.clip(total_reward, -10.0, 10.0)

    # diagnostics
    info = {
        'rotation_reward': rotation_reward,
        'termination_penalty': termination_penalty,
        'mean_angvel_z': angvel_z_safe,
        'abs_angvel_z': jnp.abs(angvel_z_safe),
        'smooth_angvel_z': angvel_z_smooth,
        'abs_smooth_angvel_z': jnp.abs(angvel_z_smooth),

        'raw_work_penalty': jnp.sum((torques * dof_vel) ** 2, axis=-1) if torques is not None and dof_vel is not None else jnp.zeros_like(rotation_reward),
        'raw_pose_diff_penalty': jnp.sum((dof_pos - init_dof_pos) ** 2, axis=-1) if dof_pos is not None and init_dof_pos is not None else jnp.zeros_like(rotation_reward),
        'raw_torque_penalty': jnp.sum(torques ** 2, axis=-1) if torques is not None else jnp.zeros_like(rotation_reward),
        'raw_linvel_penalty': jnp.linalg.norm(object_linvel, ord=1, axis=-1) if object_linvel is not None else jnp.zeros_like(rotation_reward),
        'torque_magnitude': jnp.sqrt(jnp.sum(torques ** 2, axis=-1)) if torques is not None else jnp.zeros_like(rotation_reward),
        'dof_vel_magnitude': jnp.sqrt(jnp.sum(dof_vel ** 2, axis=-1)) if dof_vel is not None else jnp.zeros_like(rotation_reward),
        'pose_diff_magnitude': jnp.sqrt(jnp.sum((dof_pos - init_dof_pos) ** 2, axis=-1)) if dof_pos is not None and init_dof_pos is not None else jnp.zeros_like(rotation_reward),
        'linvel_magnitude': jnp.linalg.norm(object_linvel, axis=-1) if object_linvel is not None else jnp.zeros_like(rotation_reward),
        'max_abs_torque': jnp.max(jnp.abs(torques), axis=-1) if torques is not None else jnp.zeros_like(rotation_reward),
        'max_abs_dof_vel': jnp.max(jnp.abs(dof_vel), axis=-1) if dof_vel is not None else jnp.zeros_like(rotation_reward),
        'max_abs_pose_diff': jnp.max(jnp.abs(dof_pos - init_dof_pos), axis=-1) if dof_pos is not None and init_dof_pos is not None else jnp.zeros_like(rotation_reward),
        'rotation_reward_mean': rotation_reward,
        'fingertip_contact': fingertip_cube_contact if fingertip_cube_contact is not None else jnp.ones(object_angvel.shape[0], dtype=bool),
    }

    return total_reward, info, angvel_z_smooth


def check_termination(
    object_pos: jax.Array,
    progress_buf: jax.Array,
    dof_vel: jax.Array = None,
    cube_linvel: jax.Array = None,
    cube_angvel: jax.Array = None,
    max_episode_length: int = 500,
    reset_height_threshold: float = -0.05,
    max_dof_vel: float = 50.0,
    max_cube_linvel: float = 5.0,
    max_cube_angvel: float = 50.0
) -> tuple:
    """
    Check if episode should terminate.
    """
    # Cube fell below hand
    object_fallen = object_pos[:, 2] < reset_height_threshold

    # Episode timeout
    timeout = progress_buf >= max_episode_length

    # Physics explosion (NaN/Inf)
    has_nan = jnp.any(jnp.isnan(object_pos), axis=-1)
    has_inf = jnp.any(jnp.isinf(object_pos), axis=-1)
    extreme_position = jnp.any(jnp.abs(object_pos) > 10.0, axis=-1)

    physics_explosion = jnp.logical_or(jnp.logical_or(has_nan, has_inf), extreme_position)

    # True termination: cube fell or physics exploded
    termination = jnp.logical_or(object_fallen, physics_explosion)

    # Reset on termination OR timeout
    reset_mask = jnp.logical_or(termination, timeout)

    return reset_mask, termination
