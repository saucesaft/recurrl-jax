"""
leap Hand domain randomization parameter factory.

defines which physics parameters to randomize and their ranges.
the generic DR logic lives in recurrl_jax.utils.domain_randomization.
"""

import numpy as np
import mujoco

from recurrl_jax.utils.domain_randomization import DRParam


def build_leap_hand_dr_params(mj_model: mujoco.MjModel) -> tuple:
    """return a tuple of DRParam describing all randomizable parameters for LEAP Hand."""
    fingertip_geom_ids = tuple(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("th_tip", "if_tip", "mf_tip", "rf_tip")
    )
    cube_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    hand_body_ids = tuple(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (
            "palm",
            "if_bs", "if_px", "if_md", "if_ds",
            "mf_bs", "mf_px", "mf_md", "mf_ds",
            "rf_bs", "rf_px", "rf_md", "rf_ds",
            "th_mp", "th_bs", "th_px", "th_ds",
        )
    )
    hand_joint_ids = tuple(range(16))
    actuator_ids = tuple(range(mj_model.nu))

    return (
        DRParam("geom_friction", fingertip_geom_ids, "multiply", 0.5, 1.0, col=0),
        DRParam("geom_friction", (cube_geom_id,), "multiply", 0.7, 1.5, col=0),
        DRParam("body_mass", (cube_body_id,), "multiply", 0.8, 1.2),
        DRParam("body_inertia", (cube_body_id,), "multiply", 0.5, 1.5),
        DRParam(
            "body_ipos", (cube_body_id,), "add", -5e-3, 5e-3,
            init_only=True,
        ),
        DRParam("qpos0", hand_joint_ids, "add", -0.05, 0.05),
        DRParam("dof_frictionloss", hand_joint_ids, "multiply", 0.5, 2.0),
        DRParam("dof_armature", hand_joint_ids, "multiply", 1.0, 1.05),
        DRParam("body_mass", hand_body_ids, "multiply", 0.9, 1.1),
        DRParam(
            "actuator_gainprm", actuator_ids, "multiply", 0.8, 1.2, col=0,
            linked_field="actuator_biasprm", linked_col=1,
            linked_transform=lambda kp: -kp,
        ),
        DRParam("dof_damping", hand_joint_ids, "multiply", 0.8, 1.2),
    )
