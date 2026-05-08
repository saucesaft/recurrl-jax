"""ANYmal reward and termination functions — pure JAX, no env class dependency."""
import jax
import jax.numpy as jp

from brax import math


# ---------------------------------------------------------------------------
# termination
# ---------------------------------------------------------------------------

def check_termination(
    x,
    obs: jax.Array,
    data,
    termination_height: float = 0.25,
    early_termination: bool = True,
):
    """returns (done, term_reasons) where term_reasons is a dict of float flags."""
    t_height = (x.pos[0, 2] < termination_height).astype(jp.float32)

    nonfinite = (jp.any(~jp.isfinite(data.qpos))
                 | jp.any(~jp.isfinite(data.qvel))
                 | jp.any(~jp.isfinite(obs)))
    t_invalid = nonfinite.astype(jp.float32)

    up = jp.array([0.0, 0.0, 1.0])
    t_flip = (jp.dot(math.rotate(up, x.rot[0]), up) < 0).astype(jp.float32)

    done = jp.maximum(jp.maximum(t_height, t_invalid), t_flip)
    done = jp.where(early_termination, done, 0.0)
    term_reasons = {
        'term_height':  t_height,
        'term_invalid': t_invalid,
        'term_flip':    t_flip,
    }
    return done, term_reasons


# ---------------------------------------------------------------------------
# reward terms
# ---------------------------------------------------------------------------

def reward_lin_vel_tracking(v_lin_body: jax.Array, vel_cmd: jax.Array) -> jax.Array:
    """exp(-||v_xy - v_xy*||² / 0.25)"""
    return jp.exp(-jp.sum(jp.square(v_lin_body[:2] - vel_cmd[:2])) / 0.25)


def reward_ang_vel_tracking(v_ang_body: jax.Array, vel_cmd: jax.Array) -> jax.Array:
    """exp(-(ωz - ωz*)² / 0.25)"""
    return jp.exp(-jp.square(v_ang_body[2] - vel_cmd[2]) / 0.25)


def reward_foot_height(
    data,
    phase_rad: jax.Array,
    foot_geom_ids: jax.Array,
    foot_phase_offsets: jax.Array,
    swing_height: float = 0.1,
) -> jax.Array:
    """Σ_j (z*_j / 0.1) · exp(-(z_j - z*_j)² / 0.01)"""
    foot_z = data.geom_xpos[foot_geom_ids, 2]
    z_star = jp.maximum(0.0, swing_height * jp.sin(phase_rad + foot_phase_offsets))
    weight = z_star / 0.1
    return jp.sum(weight * jp.exp(-jp.square(foot_z - z_star) / 0.01))


def reward_swing_contact(
    data,
    phase_rad: jax.Array,
    foot_geom_ids: jax.Array,
    foot_phase_offsets: jax.Array,
    swing_height: float = 0.1,
    contact_threshold: float = 0.07,
) -> jax.Array:
    """dense penalty for swing foot remaining on ground. fires every step during shuffle."""
    foot_z = data.geom_xpos[foot_geom_ids, 2]
    z_star = jp.maximum(0.0, swing_height * jp.sin(phase_rad + foot_phase_offsets))
    swing_weight = z_star / swing_height
    in_contact = jax.nn.sigmoid((contact_threshold - foot_z) / 0.01)
    return -jp.sum(swing_weight * in_contact)


def reward_lin_vel_error(v_lin_body: jax.Array) -> jax.Array:
    """-vz²"""
    return jp.maximum(-jp.square(v_lin_body[2]), -25.0)


def reward_ang_vel_error(v_ang_body: jax.Array) -> jax.Array:
    """-||ωxy||²"""
    return jp.maximum(-jp.sum(jp.square(v_ang_body[:2])), -200.0)


def reward_base_height(x, target: float = 0.50) -> jax.Array:
    """exp(-(z - target)² / 0.1)"""
    return jp.exp(-jp.square(x.pos[0, 2] - target) / 0.1)


def reward_base_orientation(g_body: jax.Array) -> jax.Array:
    """-||g_xy||²"""
    return -jp.sum(jp.square(g_body[:2]))


def reward_action_magnitude(raw_action: jax.Array) -> jax.Array:
    """-Σ|a_i|"""
    return -jp.sum(jp.abs(raw_action))


def reward_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    """-||a - a_prev||²"""
    return -jp.sum(jp.square(act - last_act))


def reward_joint_acceleration(joint_acc: jax.Array) -> jax.Array:
    """-||q̈||²"""
    return jp.maximum(-jp.sum(jp.square(joint_acc)), -1e7)


def reward_joint_torque(data) -> jax.Array:
    """-||τ||²"""
    return -jp.sum(jp.square(data.actuator_force))


def reward_joint_velocity(joint_vel: jax.Array) -> jax.Array:
    """-||q̇||²  (capped at 12 joints × 100 rad/s termination threshold)"""
    return jp.maximum(-jp.sum(jp.square(joint_vel)), -1.2e5)


def reward_feet_air_time(feet_air_time: jax.Array, first_contact: jax.Array, v_lin_body: jax.Array, threshold: float = 0.15) -> jax.Array:
    """Σ_f first_contact_f · max(t_air_f - threshold, 0) — adapted from Rudin et al. 2022."""
    return jp.sum(first_contact * jp.maximum(feet_air_time - threshold, 0.0))


def reward_pose(joint_pos: jax.Array, default_pos: jax.Array, std: jax.Array) -> jax.Array:
    """exp(-sum((q - q_default)² / σ²))"""
    return jp.exp(-jp.sum(jp.square((joint_pos - default_pos) / std)))
