"""
model-level domain randomization for MJX environments.

1. creates per-environment randomized MJX models
2. randomizes physics parameters (mass, friction, PD gains, etc.)
3. applies randomization ONCE at reset, affecting all physics calculations
"""

import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco
from typing import Tuple
import numpy as np


def randomize_mjx_model(
    mj_model: mujoco.MjModel,
    mjx_model: mjx.Model,
    rng: jax.Array
) -> mjx.Model:
    """
    randomize a single MJX model

    - fingertip friction: U(0.5, 1.0)
    - cube mass: U(0.8, 1.2)
    - cube inertia: U(0.8, 1.2)
    - cube position offset: U(-0.005, 0.005) per axis
    - initial joint positions: U(-0.05, 0.05)
    - joint friction: U(0.5, 2.0)
    - joint armature: U(1.0, 1.05)
    - hand link masses: U(0.9, 1.1)
    - PD controller gains (kp): U(0.8, 1.2)
    - PD controller bias: U(0.8, 1.2)
    - joint damping: U(0.8, 1.2)
    """
    # get IDs from mj_model
    cube_geom_id = mj_model.geom("cube").id
    cube_body_id = mj_model.body("cube").id

    # hand joint IDs (16 actuated joints)
    joint_names = [
        "if_abd", "if_mcp", "if_pip", "if_dip",
        "mf_mcp", "mf_pip", "mf_dip",
        "rf_mcp", "rf_pip", "rf_dip",
        "th_cmc_abd", "th_cmc_flex", "th_mcp", "th_ip"
    ]
    # Note: LEAP hand has additional joints, total 16 actuated
    hand_qids = slice(0, 16)

    # fingertip geom IDs
    fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
    fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

    # hand body IDs
    hand_body_names = [
        "palm",
        "if_bs", "if_px", "if_md", "if_ds",
        "mf_bs", "mf_px", "mf_md", "mf_ds",
        "rf_bs", "rf_px", "rf_md", "rf_ds",
        "th_mp", "th_bs", "th_px", "th_ds",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])

    # split random keys
    keys = jax.random.split(rng, 11)  # added one more for cube friction
    key_idx = 0

    # 1. fingertip friction: U(0.5, 1.0)
    fingertip_friction = jax.random.uniform(keys[key_idx], (1,), minval=0.5, maxval=1.0)[0]
    geom_friction = mjx_model.geom_friction.at[fingertip_geom_ids, 0].set(fingertip_friction)
    key_idx += 1

    # 1b. cube friction: U(0.7, 1.5) multiplier on base friction
    cube_friction_mult = jax.random.uniform(keys[10], (1,), minval=0.7, maxval=1.5)[0]
    geom_friction = geom_friction.at[cube_geom_id, 0].multiply(cube_friction_mult)

    # 2. cube mass scale: U(0.8, 1.2)
    dmass = jax.random.uniform(keys[key_idx], minval=0.8, maxval=1.2)
    key_idx += 1

    # 3. cube inertia scale + position offset
    body_inertia = mjx_model.body_inertia.at[cube_body_id].set(
        mjx_model.body_inertia[cube_body_id] * dmass
    )
    dpos = jax.random.uniform(keys[key_idx], (3,), minval=-5e-3, maxval=5e-3)
    body_ipos = mjx_model.body_ipos.at[cube_body_id].set(
        mjx_model.body_ipos[cube_body_id] + dpos
    )
    key_idx += 1

    # 4. initial joint positions: U(-0.05, 0.05)
    qpos0_jitter = jax.random.uniform(keys[key_idx], shape=(16,), minval=-0.05, maxval=0.05)
    qpos0 = mjx_model.qpos0.at[hand_qids].set(
        mjx_model.qpos0[hand_qids] + qpos0_jitter
    )
    key_idx += 1

    # 5. joint friction: U(0.5, 2.0)
    frictionloss_scale = jax.random.uniform(keys[key_idx], shape=(16,), minval=0.5, maxval=2.0)
    dof_frictionloss = mjx_model.dof_frictionloss.at[hand_qids].set(
        mjx_model.dof_frictionloss[hand_qids] * frictionloss_scale
    )
    key_idx += 1

    # 6. joint armature: U(1.0, 1.05)
    armature_scale = jax.random.uniform(keys[key_idx], shape=(16,), minval=1.0, maxval=1.05)
    dof_armature = mjx_model.dof_armature.at[hand_qids].set(
        mjx_model.dof_armature[hand_qids] * armature_scale
    )
    key_idx += 1

    # 7. hand link masses: U(0.9, 1.1)
    hand_mass_scale = jax.random.uniform(
        keys[key_idx], shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
    )
    body_mass = mjx_model.body_mass.at[hand_body_ids].set(
        mjx_model.body_mass[hand_body_ids] * hand_mass_scale
    )
    key_idx += 1

    # 8. PD controller gains (kp): U(0.8, 1.2)
    kp_scale = jax.random.uniform(keys[key_idx], (mjx_model.nu,), minval=0.8, maxval=1.2)
    kp = mjx_model.actuator_gainprm[:, 0] * kp_scale
    actuator_gainprm = mjx_model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = mjx_model.actuator_biasprm.at[:, 1].set(-kp)
    key_idx += 1

    # 9. joint damping: U(0.8, 1.2)
    kd_scale = jax.random.uniform(keys[key_idx], (16,), minval=0.8, maxval=1.2)
    dof_damping = mjx_model.dof_damping.at[hand_qids].set(
        mjx_model.dof_damping[hand_qids] * kd_scale
    )
    key_idx += 1

    # apply all randomizations to model
    randomized_model = mjx_model.tree_replace({
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "body_inertia": body_inertia,
        "body_ipos": body_ipos,
        "qpos0": qpos0,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
        "dof_damping": dof_damping,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
    })

    return randomized_model


def create_batched_randomized_models(
    mj_model: mujoco.MjModel,
    mjx_model: mjx.Model,
    num_envs: int,
    rng: jax.Array
) -> Tuple[mjx.Model, jax.Array]:
    """
    create batched randomized MJX models for parallel environments.
    """
    # get IDs from mj_model
    cube_geom_id = mj_model.geom("cube").id
    cube_body_id = mj_model.body("cube").id
    hand_qids = slice(0, 16)  # First 16 qpos indices

    fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
    fingertip_geom_ids = jnp.array([mj_model.geom(g).id for g in fingertip_geoms])

    hand_body_names = [
        "palm",
        "if_bs", "if_px", "if_md", "if_ds",
        "mf_bs", "mf_px", "mf_md", "mf_ds",
        "rf_bs", "rf_px", "rf_md", "rf_ds",
        "th_mp", "th_bs", "th_px", "th_ds",
    ]
    hand_body_ids = jnp.array([mj_model.body(n).id for n in hand_body_names])

    # generate random keys for each environment
    env_rngs = jax.random.split(rng, num_envs)

    # split keys for different randomizations
    keys = jax.vmap(lambda r: jax.random.split(r, 11))(env_rngs)  # (num_envs, 11) - added cube friction

    # this creates batched arrays [num_envs, ...] for each field

    # 1. fingertip friction: U(0.5, 1.0) - [num_envs, num_geoms, 3]
    # 1b. cube friction: U(0.7, 1.5) multiplier
    @jax.vmap
    def gen_geom_friction(key_fingertip, key_cube):
        friction_val = jax.random.uniform(key_fingertip, (1,), minval=0.5, maxval=1.0)[0]
        cube_friction_mult = jax.random.uniform(key_cube, (1,), minval=0.7, maxval=1.5)[0]
        geom_friction = mjx_model.geom_friction.at[fingertip_geom_ids, 0].set(friction_val)
        geom_friction = geom_friction.at[cube_geom_id, 0].multiply(cube_friction_mult)
        return geom_friction
    geom_friction_batch = gen_geom_friction(keys[:, 0], keys[:, 10])

    # 2. cube mass scale: U(0.8, 1.2) - [num_envs]
    @jax.vmap
    def gen_dmass(key):
        return jax.random.uniform(key, minval=0.8, maxval=1.2)
    dmass_batch = gen_dmass(keys[:, 1])

    # 3. cube inertia scale - [num_envs, num_bodies, 3]
    @jax.vmap
    def gen_body_inertia(dmass):
        return mjx_model.body_inertia.at[cube_body_id].set(
            mjx_model.body_inertia[cube_body_id] * dmass
        )
    body_inertia_batch = gen_body_inertia(dmass_batch)

    # 4. cube position offset - [num_envs, num_bodies, 3]
    @jax.vmap
    def gen_body_ipos(key):
        dpos = jax.random.uniform(key, (3,), minval=-5e-3, maxval=5e-3)
        return mjx_model.body_ipos.at[cube_body_id].set(
            mjx_model.body_ipos[cube_body_id] + dpos
        )
    body_ipos_batch = gen_body_ipos(keys[:, 2])

    # 5. initial joint positions - [num_envs, nq]
    @jax.vmap
    def gen_qpos0(key):
        qpos0_jitter = jax.random.uniform(key, shape=(16,), minval=-0.05, maxval=0.05)
        return mjx_model.qpos0.at[hand_qids].set(mjx_model.qpos0[hand_qids] + qpos0_jitter)
    qpos0_batch = gen_qpos0(keys[:, 3])

    # 6. joint friction - [num_envs, nv]
    @jax.vmap
    def gen_dof_frictionloss(key):
        frictionloss_scale = jax.random.uniform(key, shape=(16,), minval=0.5, maxval=2.0)
        return mjx_model.dof_frictionloss.at[hand_qids].set(
            mjx_model.dof_frictionloss[hand_qids] * frictionloss_scale
        )
    dof_frictionloss_batch = gen_dof_frictionloss(keys[:, 4])

    # 7. joint armature - [num_envs, nv]
    @jax.vmap
    def gen_dof_armature(key):
        armature_scale = jax.random.uniform(key, shape=(16,), minval=1.0, maxval=1.05)
        return mjx_model.dof_armature.at[hand_qids].set(
            mjx_model.dof_armature[hand_qids] * armature_scale
        )
    dof_armature_batch = gen_dof_armature(keys[:, 5])

    # 8. hand link masses - [num_envs, num_bodies]
    @jax.vmap
    def gen_body_mass(key):
        hand_mass_scale = jax.random.uniform(
            key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
        )
        return mjx_model.body_mass.at[hand_body_ids].set(
            mjx_model.body_mass[hand_body_ids] * hand_mass_scale
        )
    body_mass_batch = gen_body_mass(keys[:, 6])

    # 9. PD controller gains (kp) - [num_envs, nu, ...]
    @jax.vmap
    def gen_actuator_gains(key):
        kp_scale = jax.random.uniform(key, (mjx_model.nu,), minval=0.8, maxval=1.2)
        kp = mjx_model.actuator_gainprm[:, 0] * kp_scale
        gainprm = mjx_model.actuator_gainprm.at[:, 0].set(kp)
        biasprm = mjx_model.actuator_biasprm.at[:, 1].set(-kp)
        return gainprm, biasprm
    actuator_gainprm_batch, actuator_biasprm_batch = gen_actuator_gains(keys[:, 7])

    # 10. joint damping - [num_envs, nv]
    @jax.vmap
    def gen_dof_damping(key):
        kd_scale = jax.random.uniform(key, (16,), minval=0.8, maxval=1.2)
        return mjx_model.dof_damping.at[hand_qids].set(
            mjx_model.dof_damping[hand_qids] * kd_scale
        )
    dof_damping_batch = gen_dof_damping(keys[:, 8])

    # only these fields are batched, rest are shared!
    batched_model = mjx_model.tree_replace({
        "geom_friction": geom_friction_batch,
        "body_mass": body_mass_batch,
        "body_inertia": body_inertia_batch,
        "body_ipos": body_ipos_batch,
        "qpos0": qpos0_batch,
        "dof_frictionloss": dof_frictionloss_batch,
        "dof_armature": dof_armature_batch,
        "dof_damping": dof_damping_batch,
        "actuator_gainprm": actuator_gainprm_batch,
        "actuator_biasprm": actuator_biasprm_batch,
    })

    # step 1: create a Model with all None values
    in_axes = jax.tree.map(lambda x: None, mjx_model)

    # step 2: use tree_replace to set batched fields to 0
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "body_mass": 0,
        "body_inertia": 0,
        "body_ipos": 0,
        "qpos0": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "dof_damping": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
    })

    return batched_model, in_axes


def re_randomize_models_on_reset(
    mj_model: mujoco.MjModel,
    batched_model: mjx.Model,
    reset_mask: jax.Array,
    rng: jax.Array
) -> mjx.Model:
    """
    re-randomize models for environments that are resetting
    """
    num_envs = reset_mask.shape[0]

    # get IDs from mj_model
    cube_body_id = mj_model.body("cube").id
    hand_qids = slice(0, 16)

    fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
    fingertip_geom_ids = jnp.array([mj_model.geom(g).id for g in fingertip_geoms])

    hand_body_names = [
        "palm",
        "if_bs", "if_px", "if_md", "if_ds",
        "mf_bs", "mf_px", "mf_md", "mf_ds",
        "rf_bs", "rf_px", "rf_md", "rf_ds",
        "th_mp", "th_bs", "th_px", "th_ds",
    ]
    hand_body_ids = jnp.array([mj_model.body(n).id for n in hand_body_names])

    # generate random keys for each environment
    env_rngs = jax.random.split(rng, num_envs)
    keys = jax.vmap(lambda r: jax.random.split(r, 11))(env_rngs)  # cube friction

    # vmap only field generation!
    from mujoco import mjx
    base_mjx = mjx.put_model(mj_model)

    # 1. fingertip friction + cube friction
    @jax.vmap
    def gen_geom_friction(key_fingertip, key_cube):
        friction_val = jax.random.uniform(key_fingertip, (1,), minval=0.5, maxval=1.0)[0]
        cube_friction_mult = jax.random.uniform(key_cube, (1,), minval=0.7, maxval=1.5)[0]
        geom_friction = base_mjx.geom_friction.at[fingertip_geom_ids, 0].set(friction_val)
        geom_friction = geom_friction.at[cube_geom_id, 0].multiply(cube_friction_mult)
        return geom_friction
    new_geom_friction = gen_geom_friction(keys[:, 0], keys[:, 10])

    # 2. cube mass scale
    @jax.vmap
    def gen_dmass(key):
        return jax.random.uniform(key, minval=0.8, maxval=1.2)
    dmass_batch = gen_dmass(keys[:, 1])

    # 3. cube inertia
    @jax.vmap
    def gen_body_inertia(dmass):
        return base_mjx.body_inertia.at[cube_body_id].set(
            base_mjx.body_inertia[cube_body_id] * dmass
        )
    new_body_inertia = gen_body_inertia(dmass_batch)

    # 4. cube position offset
    @jax.vmap
    def gen_body_ipos(key):
        dpos = jax.random.uniform(key, (3,), minval=-5e-3, maxval=5e-3)
        return base_mjx.body_ipos.at[cube_body_id].set(
            base_mjx.body_ipos[cube_body_id] + dpos
        )
    new_body_ipos = gen_body_ipos(keys[:, 2])

    # 5. initial joint positions
    @jax.vmap
    def gen_qpos0(key):
        qpos0_jitter = jax.random.uniform(key, shape=(16,), minval=-0.05, maxval=0.05)
        return base_mjx.qpos0.at[hand_qids].set(base_mjx.qpos0[hand_qids] + qpos0_jitter)
    new_qpos0 = gen_qpos0(keys[:, 3])

    # 6. joint friction
    @jax.vmap
    def gen_dof_frictionloss(key):
        frictionloss_scale = jax.random.uniform(key, shape=(16,), minval=0.5, maxval=2.0)
        return base_mjx.dof_frictionloss.at[hand_qids].set(
            base_mjx.dof_frictionloss[hand_qids] * frictionloss_scale
        )
    new_dof_frictionloss = gen_dof_frictionloss(keys[:, 4])

    # 7. joint armature
    @jax.vmap
    def gen_dof_armature(key):
        armature_scale = jax.random.uniform(key, shape=(16,), minval=1.0, maxval=1.05)
        return base_mjx.dof_armature.at[hand_qids].set(
            base_mjx.dof_armature[hand_qids] * armature_scale
        )
    new_dof_armature = gen_dof_armature(keys[:, 5])

    # 8. hand link masses
    @jax.vmap
    def gen_body_mass(key):
        hand_mass_scale = jax.random.uniform(
            key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
        )
        return base_mjx.body_mass.at[hand_body_ids].set(
            base_mjx.body_mass[hand_body_ids] * hand_mass_scale
        )
    new_body_mass = gen_body_mass(keys[:, 6])

    # 9. PD controller gains
    @jax.vmap
    def gen_actuator_gains(key):
        kp_scale = jax.random.uniform(key, (base_mjx.nu,), minval=0.8, maxval=1.2)
        kp = base_mjx.actuator_gainprm[:, 0] * kp_scale
        gainprm = base_mjx.actuator_gainprm.at[:, 0].set(kp)
        biasprm = base_mjx.actuator_biasprm.at[:, 1].set(-kp)
        return gainprm, biasprm
    new_actuator_gainprm, new_actuator_biasprm = gen_actuator_gains(keys[:, 7])

    # 10. joint damping
    @jax.vmap
    def gen_dof_damping(key):
        kd_scale = jax.random.uniform(key, (16,), minval=0.8, maxval=1.2)
        return base_mjx.dof_damping.at[hand_qids].set(
            base_mjx.dof_damping[hand_qids] * kd_scale
        )
    new_dof_damping = gen_dof_damping(keys[:, 8])

    # select new or old values based on reset_mask
    def select_field(old, new):
        # broadcast reset_mask to match field shape
        mask_shape = [num_envs] + [1] * (old.ndim - 1)
        mask = reset_mask.reshape(mask_shape)
        return jnp.where(mask, new, old)

    # apply conditional updates
    updates = {
        "geom_friction": select_field(batched_model.geom_friction, new_geom_friction),
        "body_mass": select_field(batched_model.body_mass, new_body_mass),
        "body_inertia": select_field(batched_model.body_inertia, new_body_inertia),
        "body_ipos": select_field(batched_model.body_ipos, new_body_ipos),
        "qpos0": select_field(batched_model.qpos0, new_qpos0),
        "dof_frictionloss": select_field(batched_model.dof_frictionloss, new_dof_frictionloss),
        "dof_armature": select_field(batched_model.dof_armature, new_dof_armature),
        "dof_damping": select_field(batched_model.dof_damping, new_dof_damping),
        "actuator_gainprm": select_field(batched_model.actuator_gainprm, new_actuator_gainprm),
        "actuator_biasprm": select_field(batched_model.actuator_biasprm, new_actuator_biasprm),
    }

    return batched_model.tree_replace(updates)


# helper function to check memory usage
def estimate_memory_usage(num_envs: int) -> str:
    """estimate memory usage for batched models."""
    # approximate size per environment (rough estimate)
    bytes_per_env = 60_000  # ~60 KB per model
    total_bytes = bytes_per_env * num_envs
    total_mb = total_bytes / (1024 * 1024)
    return f"~{total_mb:.1f} MB for {num_envs} environments"
