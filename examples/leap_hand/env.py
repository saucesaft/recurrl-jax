from etils import epath
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import mujoco
from mujoco import mjx

from examples.leap_hand.rewards import compute_total_reward, check_termination
from examples.leap_hand.domain_randomization_model import create_batched_randomized_models, re_randomize_models_on_reset
from recurrl_jax.utils.quat_utils import rotate_vec_by_quat




class MJXLeapHandEnv:
    def __init__(self, xml_path: str, num_envs: int, key: jax.Array,
                 action_scale: float = 0.6, use_domain_randomization: bool = False,
                 grasp_cache_path: str = None):
        self.mjx_path = epath.Path(xml_path).as_posix()
        self.num_envs = num_envs
        self.key = key
        self.action_scale = action_scale
        self.use_domain_randomization = use_domain_randomization

        self.progress_buf = jnp.zeros(num_envs, dtype=jnp.int32)
        self.initial_dof_pos = None
        self.reset_height_threshold = -0.05  # Cube fell below hand

        # default hand pose (stable grasp configuration)
        # actions are offsets from this pose, not deltas from current position
        self.default_pose = jnp.array([
            # Index finger: 0.8, 0, 0.8, 0.8
            0.8, 0.0, 0.8, 0.8,
            # Middle finger: 0.8, 0, 0.8, 0.8
            0.8, 0.0, 0.8, 0.8,
            # Ring finger: 0.8, 0, 0.8, 0.8
            0.8, 0.0, 0.8, 0.8,
            # Thumb: 0.8, 0.8, 0.8, 0
            0.8, 0.8, 0.8, 0.0
        ])
        self.max_episode_length = 500

        # EMA smoothed angular velocity state (for smoothed reward)
        self.angvel_z_smooth = jnp.zeros(num_envs)

        # Load grasp cache if provided (for curriculum learning)
        self.grasp_cache = None
        self.grasp_cache_size = 0
        if grasp_cache_path is not None:
            cache_data = np.load(grasp_cache_path)
            self.grasp_cache = jnp.array(cache_data)
            self.grasp_cache_size = self.grasp_cache.shape[0]
            print(f"Loaded grasp cache from {grasp_cache_path}: {self.grasp_cache_size} grasps")

        # control frequency
        self.control_freq_inv = 5  # ctrl_dt / sim_dt = 0.05 / 0.01 = 5

        # single model
        self.mj_model   = mujoco.MjModel.from_xml_path(self.mjx_path)
        self.mj_data    = mujoco.MjData( self.mj_model )
        self.mjx_model  = mjx.put_model( self.mj_model )
        self.mjx_data   = mjx.put_data( self.mj_model, self.mj_data )

        # Get site and body IDs for fingertip positions and palm
        self.if_tip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "if_tip")
        self.mf_tip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "mf_tip")
        self.rf_tip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rf_tip")
        self.th_tip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "th_tip")
        self.palm_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "palm")

        # Get IDs for domain randomization (collision geoms, not visual)
        self.fingertip_geom_ids = jnp.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "th_tip"),
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "if_tip"),
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "mf_tip"),
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "rf_tip"),
        ])
        self.cube_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.cube_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        # Hand body IDs for mass randomization
        hand_body_names = [
            "palm", "if_bs", "if_px", "if_md", "if_ds",
            "mf_bs", "mf_px", "mf_md", "mf_ds",
            "rf_bs", "rf_px", "rf_md", "rf_ds",
            "th_mp", "th_bs", "th_px", "th_ds",
        ]
        self.hand_body_ids = jnp.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in hand_body_names
        ])

        # Store BASE model values for domain randomization (to avoid compounding)
        # These are the original values before any randomization
        self.base_fingertip_friction = self.mjx_model.geom_friction[self.fingertip_geom_ids, 0]
        self.base_cube_friction = self.mjx_model.geom_friction[self.cube_geom_id, 0]
        self.base_cube_mass = self.mjx_model.body_mass[self.cube_body_id]
        self.base_cube_inertia = self.mjx_model.body_inertia[self.cube_body_id]
        self.base_hand_body_mass = self.mjx_model.body_mass[self.hand_body_ids]
        self.base_dof_frictionloss = self.mjx_model.dof_frictionloss[:16]
        self.base_dof_armature = self.mjx_model.dof_armature[:16]
        self.base_dof_damping = self.mjx_model.dof_damping[:16]
        self.base_actuator_gainprm = self.mjx_model.actuator_gainprm[:, 0]
        self.base_qpos0 = self.mjx_model.qpos0[:16]
        
        # batched model and data (with optional domain randomization)
        self.mjx_model_batch,  self.mjx_data_batch = self._create_batch()

        # useful info
        self.num_dofs = self.mj_model.nu # numbers of actuators
        self.ctrl_range = self.mjx_model.actuator_ctrlrange # control limits

        # jit magic
        self.step_fn  = jax.jit( self._step )
        self.reset_fn = jax.jit( self._reset )

        # use correct indexes
        self.joint_lower_limits = self.mjx_model.jnt_range[:, 0][:16]
        self.joint_upper_limits = self.mjx_model.jnt_range[:, 1][:16]


    def _create_batch(self):
        if self.use_domain_randomization:
            # create per-environment randomized models
            self.key, dr_key = jr.split(self.key)
            batched_model, in_axes = create_batched_randomized_models(
                self.mj_model,
                self.mjx_model,
                self.num_envs,
                dr_key
            )
            self._model_in_axes = in_axes
        else:
            # Single shared model (no randomization)
            batched_model = self.mjx_model
            self._model_in_axes = None

        # Create batched data (one per environment)
        mjx_data_batch = jax.vmap(lambda _: mjx.make_data(self.mjx_model))(
            jnp.arange(self.num_envs)
        )

        return batched_model, mjx_data_batch

    def _step(self, mjx_model, mjx_data_batch, action):
        # scale actions from policy [-1, 1] to control range
        scaled_action = action * self.action_scale
        mjx_data_batch = mjx_data_batch.replace(ctrl=scaled_action)  # set control actions

        # repeat physics step while holding action constant
        in_axes_model = self._model_in_axes if self.use_domain_randomization else None

        def physics_step(i, data):
            return jax.vmap(mjx.step, in_axes=(in_axes_model, 0))(mjx_model, data)

        mjx_data_batch = jax.lax.fori_loop(
            0, self.control_freq_inv, physics_step, mjx_data_batch
        )

        return mjx_data_batch

    def _reset(self, key, mjx_model, mjx_data_batch, reset_mask: jnp.ndarray):
        # Determine model_in_axes based on DR flag (needed for _reset_jit)
        model_in_axes = self._model_in_axes if self.use_domain_randomization else None

        # Delegate to the unified JIT-compatible reset function
        return _reset_jit(
            key,
            mjx_model,
            mjx_data_batch,
            reset_mask,
            use_domain_randomization=self.use_domain_randomization,
            model_in_axes=model_in_axes,
            grasp_cache=self.grasp_cache,
            grasp_cache_size=self.grasp_cache_size,
            fingertip_geom_ids=self.fingertip_geom_ids,
            cube_geom_id=self.cube_geom_id,
            cube_body_id=self.cube_body_id,
            hand_body_ids=self.hand_body_ids,
            # Base values for DR (to avoid compounding)
            base_fingertip_friction=self.base_fingertip_friction,
            base_cube_friction=self.base_cube_friction,
            base_cube_mass=self.base_cube_mass,
            base_cube_inertia=self.base_cube_inertia,
            base_hand_body_mass=self.base_hand_body_mass,
            base_dof_frictionloss=self.base_dof_frictionloss,
            base_dof_armature=self.base_dof_armature,
            base_dof_damping=self.base_dof_damping,
            base_actuator_gainprm=self.base_actuator_gainprm,
            base_qpos0=self.base_qpos0
        )

    def step(self, actions: jnp.ndarray, mjx_data_batch=None, progress_buf=None):
        self.key, reset_key = jr.split(self.key)

        # use provided state or fall back to self
        use_self = (mjx_data_batch is None and progress_buf is None)
        if mjx_data_batch is None:
            mjx_data_batch = self.mjx_data_batch
        if progress_buf is None:
            progress_buf = self.progress_buf

        # Determine model_in_axes based on DR flag
        model_in_axes = self._model_in_axes if self.use_domain_randomization else None

        # Call the JIT-compiled step function with absolute control
        (
            raw_state,
            reward,
            reset_mask,
            termination,
            info,
            new_mjx_data_batch_reset,
            new_progress_buf,
            new_mjx_model,
            new_angvel_z_smooth
        ) = env_step_jit(
            actions,
            self.mjx_model_batch,
            mjx_data_batch,
            progress_buf,
            self.initial_dof_pos,  # For pose diff penalty
            self.reset_height_threshold,
            self.max_episode_length,
            self.key,
            prev_angvel_z_smooth=self.angvel_z_smooth,
            control_freq_inv=self.control_freq_inv,
            use_domain_randomization=self.use_domain_randomization,
            model_in_axes=model_in_axes,
            action_scale=self.action_scale,
            grasp_cache=self.grasp_cache,
            grasp_cache_size=self.grasp_cache_size,
            fingertip_geom_ids=self.fingertip_geom_ids,
            cube_geom_id=self.cube_geom_id,
            cube_body_id=self.cube_body_id,
            hand_body_ids=self.hand_body_ids,
            # Base values for DR (to avoid compounding)
            base_fingertip_friction=self.base_fingertip_friction,
            base_cube_friction=self.base_cube_friction,
            base_cube_mass=self.base_cube_mass,
            base_cube_inertia=self.base_cube_inertia,
            base_hand_body_mass=self.base_hand_body_mass,
            base_dof_frictionloss=self.base_dof_frictionloss,
            base_dof_armature=self.base_dof_armature,
            base_dof_damping=self.base_dof_damping,
            base_actuator_gainprm=self.base_actuator_gainprm,
            base_qpos0=self.base_qpos0
        )

        # update self if using default parameters (backward compatibility)
        if use_self:
            self.mjx_data_batch = new_mjx_data_batch_reset
            self.progress_buf = new_progress_buf
            self.mjx_model_batch = new_mjx_model
            self.angvel_z_smooth = new_angvel_z_smooth
            # Update initial_dof_pos for reset environments (for pose diff penalty in next episode)
            new_pose = new_mjx_data_batch_reset.qpos[:, :16]
            self.initial_dof_pos = jnp.where(
                reset_mask[:, None],
                new_pose,
                self.initial_dof_pos
            )

        return raw_state, reward, reset_mask, termination, info, new_mjx_data_batch_reset, new_progress_buf, new_mjx_model, new_angvel_z_smooth


    def reset(self, env_ids: jnp.ndarray = None):
        if env_ids is None:
            reset_mask = jnp.ones( self.num_envs, dtype=bool )
        else:
            reset_mask = jnp.zeros( self.num_envs, dtype=bool )
            reset_mask = reset_mask.at[env_ids].set(True)

        self.key, subkey = jr.split( self.key )
        self.mjx_model_batch, self.mjx_data_batch = self._reset(
            subkey,
            self.mjx_model_batch,
            self.mjx_data_batch,
            reset_mask
        )

        # reset progress buffer
        self.progress_buf = jnp.where( reset_mask, 0, self.progress_buf )

        # reset smoothed angular velocity for reset environments
        self.angvel_z_smooth = jnp.where( reset_mask, 0.0, self.angvel_z_smooth )

        # Store initial pose (for pose_diff penalty)
        new_pose = self.mjx_data_batch.qpos[:, :16]
        if self.initial_dof_pos is None or env_ids is None:
            self.initial_dof_pos = new_pose  # full reset
        else:
            self.initial_dof_pos = jnp.where(
                reset_mask[:, None],
                new_pose,
                self.initial_dof_pos
            )

    def get_joint_state(self):
        # extract LEAP hand joint states (16 hinge joints)
        hand_qpos = self.mjx_data_batch.qpos[:, :16]
        hand_qvel = self.mjx_data_batch.qvel[:, :16]

        return hand_qpos, hand_qvel        

    def _get_object_pos(self):
        return self.mjx_data_batch.qpos[:, 16:19]  # x, y, z

    def _get_object_linvel(self):
        # Read from qvel (16-18), as sensors are unreliable
        return self.mjx_data_batch.qvel[:, 16:19]

    def _get_object_angvel(self):
        # Read from qvel (19-21) which is in LOCAL frame for free joints
        local_angvel = self.mjx_data_batch.qvel[:, 19:22]
        # rotate to world frame
        return local_angvel

    def get_fingertip_positions(self, mjx_data_batch=None):
        if mjx_data_batch is None:
            mjx_data_batch = self.mjx_data_batch

        # site_xpos is (num_envs, num_sites, 3)
        if_tip_pos = mjx_data_batch.site_xpos[:, self.if_tip_id, :]  # (num_envs, 3)
        mf_tip_pos = mjx_data_batch.site_xpos[:, self.mf_tip_id, :]  # (num_envs, 3)
        rf_tip_pos = mjx_data_batch.site_xpos[:, self.rf_tip_id, :]  # (num_envs, 3)
        th_tip_pos = mjx_data_batch.site_xpos[:, self.th_tip_id, :]  # (num_envs, 3)

        # Concatenate to (num_envs, 12)
        fingertip_positions = jnp.concatenate([
            if_tip_pos, mf_tip_pos, rf_tip_pos, th_tip_pos
        ], axis=-1)

        return fingertip_positions

    def get_palm_position(self, mjx_data_batch=None):
        if mjx_data_batch is None:
            mjx_data_batch = self.mjx_data_batch

        # xpos is (num_envs, num_bodies, 3)
        palm_pos = mjx_data_batch.xpos[:, self.palm_id, :]  # (num_envs, 3)

        return palm_pos

@partial(jax.jit, static_argnames=['control_freq_inv', 'use_domain_randomization', 'grasp_cache_size', 'cube_geom_id', 'cube_body_id'])
def env_step_jit(
    actions: jnp.ndarray,
    mjx_model,  # Batched model (with DR) or single model (without DR)
    mjx_data_batch,
    progress_buf: jnp.ndarray,
    initial_dof_pos: jnp.ndarray,  # Initial DOF positions from grasp (for pose diff penalty)
    reset_height_threshold: float,
    max_episode_length: int,
    key: jax.Array,
    control_freq_inv: int = 6,
    use_domain_randomization: bool = False,
    model_in_axes = None,  # MUST BE STATIC
    action_scale: float = 0.6,
    grasp_cache: jnp.ndarray = None,  # Grasp cache for stable initialization
    grasp_cache_size: int = 0,  # Size of grasp cache (static for JIT)
    fingertip_geom_ids: jnp.ndarray = None,  # Fingertip collision geom IDs for DR
    cube_geom_id: int = None,  # Cube collision geom ID for DR
    cube_body_id: int = None,  # Cube body ID for DR
    hand_body_ids: jnp.ndarray = None,  # Hand body IDs for DR
    # Base values for DR (to avoid compounding)
    base_fingertip_friction: jnp.ndarray = None,
    base_cube_friction: float = None,
    base_cube_mass: float = None,
    base_cube_inertia: jnp.ndarray = None,
    base_hand_body_mass: jnp.ndarray = None,
    base_dof_frictionloss: jnp.ndarray = None,
    base_dof_armature: jnp.ndarray = None,
    base_dof_damping: jnp.ndarray = None,
    base_actuator_gainprm: jnp.ndarray = None,
    base_qpos0: jnp.ndarray = None,
    prev_angvel_z_smooth: jnp.ndarray = None,  # EMA state for smoothed reward
):

    # actions are offsets from the default pose
    default_pose = jnp.array([0.8, 0, 0.8, 0.8, 0.8, 0, 0.8, 0.8, 0.8, 0, 0.8, 0.8, 0.8, 0.8, 0.8, 0])
    scaled_actions = default_pose + actions * action_scale
    mjx_data_batch = mjx_data_batch.replace(ctrl=scaled_actions)

    # Domain randomization: manually slice batched model fields (in_axes doesn't work in our JAX version)
    def physics_step(i, data):
        if use_domain_randomization:
            # Manually vmap: slice batched model fields for each environment
            def step_with_model_slice(model_idx, d):
                # Extract this environment's model by slicing batched fields
                env_model = mjx_model.tree_replace({
                    'geom_friction': mjx_model.geom_friction[model_idx],
                    'body_mass': mjx_model.body_mass[model_idx],
                    'body_inertia': mjx_model.body_inertia[model_idx],
                    'body_ipos': mjx_model.body_ipos[model_idx],
                    'qpos0': mjx_model.qpos0[model_idx],
                    'dof_frictionloss': mjx_model.dof_frictionloss[model_idx],
                    'dof_armature': mjx_model.dof_armature[model_idx],
                    'dof_damping': mjx_model.dof_damping[model_idx],
                    'actuator_gainprm': mjx_model.actuator_gainprm[model_idx],
                    'actuator_biasprm': mjx_model.actuator_biasprm[model_idx],
                })
                return mjx.step(env_model, d)

            num_envs = data.qpos.shape[0]
            return jax.vmap(step_with_model_slice)(jnp.arange(num_envs), data)
        else:
            # No DR: broadcast single model
            return jax.vmap(mjx.step, in_axes=(None, 0))(mjx_model, data)

    new_mjx_data_batch = jax.lax.fori_loop(
        0, control_freq_inv, physics_step, mjx_data_batch
    )

    # extract state
    qpos = new_mjx_data_batch.qpos
    qvel = new_mjx_data_batch.qvel

    object_pos = qpos[:, 16:19]
    object_quat = qpos[:, 19:23]  # [w, x, y, z]
    # Read velocity from qvel (more reliable than sensors)
    object_linvel = qvel[:, 16:19]   # cube linear velocity (global frame)
    object_angvel_local = qvel[:, 19:22]   # cube angular velocity (local frame)
    # rotate angular velocity to world frame so "rotate around Z" means world Z-axis
    object_angvel = rotate_vec_by_quat(object_angvel_local, object_quat)
    
    dof_pos = qpos[:, :16]
    dof_vel = qvel[:, :16]
    torques = new_mjx_data_batch.qfrc_actuator[:, :16]

    # contact detection for gated reward
    # contact.geom1 and contact.geom2 are (num_envs, ncon) arrays of geom IDs in contact
    contact_geom1 = new_mjx_data_batch.contact.geom1  # (num_envs, ncon)
    contact_geom2 = new_mjx_data_batch.contact.geom2  # (num_envs, ncon)

    # For each environment, check if any contact pair involves a fingertip AND the cube
    def check_fingertip_cube_contact(geom1, geom2):
        # Check if fingertip is in geom1 and cube is in geom2, or vice versa
        fingertip_in_1 = jnp.any(jnp.isin(geom1, fingertip_geom_ids))
        fingertip_in_2 = jnp.any(jnp.isin(geom2, fingertip_geom_ids))
        cube_in_1 = jnp.any(geom1 == cube_geom_id)
        cube_in_2 = jnp.any(geom2 == cube_geom_id)
        # Contact exists if (fingertip in one and cube in other)
        return jnp.logical_or(
            jnp.logical_and(fingertip_in_1, cube_in_2),
            jnp.logical_and(fingertip_in_2, cube_in_1)
        )

    fingertip_cube_contact = jax.vmap(check_fingertip_cube_contact)(contact_geom1, contact_geom2)

    # compute rewards (with EMA smoothed angular velocity)
    reward, reward_info, new_angvel_z_smooth = compute_total_reward(
        object_angvel=object_angvel,
        object_pos=object_pos,
        prev_angvel_z_smooth=prev_angvel_z_smooth,
        object_linvel=object_linvel,
        actions=actions,
        dof_vel=dof_vel,
        torques=torques,
        dof_pos=dof_pos,
        init_dof_pos=initial_dof_pos,
        reset_height_threshold=reset_height_threshold,
        fingertip_cube_contact=fingertip_cube_contact,
    )

    # check termination conditions
    new_progress_buf = progress_buf + 1
    reset_mask, termination = check_termination(
        object_pos=object_pos,
        progress_buf=new_progress_buf,
        dof_vel=dof_vel,
        cube_linvel=object_linvel,
        cube_angvel=object_angvel,
        max_episode_length=max_episode_length,
        reset_height_threshold=reset_height_threshold,
    )

    # auto-reset terminated environments
    done = reset_mask  # For backward compatibility, 'done' means any reset (timeout OR termination)
    
    def no_reset(k, m, d, mask):
        return m, d

    # Partially apply static arguments to _reset_jit so they are not passed as tracers
    reset_fn_partial = partial(_reset_jit,
                               use_domain_randomization=use_domain_randomization,
                               model_in_axes=model_in_axes,
                               grasp_cache=grasp_cache,
                               grasp_cache_size=grasp_cache_size,
                               fingertip_geom_ids=fingertip_geom_ids,
                               cube_geom_id=cube_geom_id,
                               cube_body_id=cube_body_id,
                               hand_body_ids=hand_body_ids,
                               # Base values for DR (to avoid compounding)
                               base_fingertip_friction=base_fingertip_friction,
                               base_cube_friction=base_cube_friction,
                               base_cube_mass=base_cube_mass,
                               base_cube_inertia=base_cube_inertia,
                               base_hand_body_mass=base_hand_body_mass,
                               base_dof_frictionloss=base_dof_frictionloss,
                               base_dof_armature=base_dof_armature,
                               base_dof_damping=base_dof_damping,
                               base_actuator_gainprm=base_actuator_gainprm,
                               base_qpos0=base_qpos0)

    new_mjx_model, new_mjx_data_batch_reset = jax.lax.cond(
        jnp.any(reset_mask),
        reset_fn_partial,
        no_reset,
        key, mjx_model, new_mjx_data_batch, reset_mask
    )

    # reset progress buffer and smoothed angvel for done environments
    new_progress_buf = jnp.where(reset_mask, 0, new_progress_buf)
    new_angvel_z_smooth = jnp.where(reset_mask, 0.0, new_angvel_z_smooth)

    raw_state = {
        'qpos': qpos.copy(),
        'qvel': qvel.copy(),
        'dof_pos': dof_pos.copy(),
        'object_pos': object_pos.copy(),
        'object_angvel': object_angvel.copy()  # Use physics angvel
    }

    info = {**reward_info, 'done': done, 'termination': termination}

    # done = all resets (timeout OR termination), termination = true terminations only (for GAE)
    return raw_state, reward, done, termination, info, new_mjx_data_batch_reset, new_progress_buf, new_mjx_model, new_angvel_z_smooth


def _reset_jit(key, mjx_model, mjx_data_batch, reset_mask: jnp.ndarray,
               use_domain_randomization: bool = False, model_in_axes = None,
               grasp_cache: jnp.ndarray = None, grasp_cache_size: int = 0,
               fingertip_geom_ids: jnp.ndarray = None, cube_geom_id: int = None,
               cube_body_id: int = None, hand_body_ids: jnp.ndarray = None,
               # Base model values for DR (to avoid compounding)
               base_fingertip_friction: jnp.ndarray = None,
               base_cube_friction: float = None,
               base_cube_mass: float = None,
               base_cube_inertia: jnp.ndarray = None,
               base_hand_body_mass: jnp.ndarray = None,
               base_dof_frictionloss: jnp.ndarray = None,
               base_dof_armature: jnp.ndarray = None,
               base_dof_damping: jnp.ndarray = None,
               base_actuator_gainprm: jnp.ndarray = None,
               base_qpos0: jnp.ndarray = None):
    # Create fresh data for environments that need reset
    num_envs = reset_mask.shape[0]
    use_grasp_cache = grasp_cache is not None and grasp_cache_size > 0

    if use_domain_randomization:
        # When DR is enabled, manually slice batched model fields for each environment
        def make_data_with_model_slice(model_idx, model):
            env_model = model.tree_replace({
                'geom_friction': model.geom_friction[model_idx],
                'body_mass': model.body_mass[model_idx],
                'body_inertia': model.body_inertia[model_idx],
                'body_ipos': model.body_ipos[model_idx],
                'qpos0': model.qpos0[model_idx],
                'dof_frictionloss': model.dof_frictionloss[model_idx],
                'dof_armature': model.dof_armature[model_idx],
                'dof_damping': model.dof_damping[model_idx],
                'actuator_gainprm': model.actuator_gainprm[model_idx],
                'actuator_biasprm': model.actuator_biasprm[model_idx],
            })
            return mjx.make_data(env_model)

        fresh_data_batch = jax.vmap(make_data_with_model_slice, in_axes=(0, None))(jnp.arange(num_envs), mjx_model)
    else:
        # When DR is disabled, mjx_model is a single model - create num_envs copies
        fresh_data_batch = jax.vmap(
            lambda _: mjx.make_data(mjx_model)
        )(jnp.arange(num_envs))

    # Select fresh data or keep old data based on reset_mask
    def select_data_field(fresh, old):
        mask_shape = [num_envs] + [1] * (fresh.ndim - 1)
        mask = reset_mask.reshape(mask_shape)
        return jnp.where(mask, fresh, old)

    mjx_data_batch = jax.tree_util.tree_map(
        select_data_field,
        fresh_data_batch,
        mjx_data_batch
    )

    # Fully vectorized reset randomization (avoid vmap/cond shape issues)
    num_envs = reset_mask.shape[0]
    keys = jr.split(key, num_envs + 1)
    env_keys = keys[:-1]
    dr_key = keys[-1]

    # Split keys for randomization
    keys_split = jax.vmap(lambda k: jr.split(k, 4))(env_keys)
    hand_keys = keys_split[:, 0]
    cube_pos_keys = keys_split[:, 1]
    cube_quat_keys = keys_split[:, 2]
    cube_vel_keys = keys_split[:, 3]

    # Randomize hand joint positions (vectorized)
    joint_lower = mjx_model.jnt_range[:16, 0]
    joint_upper = mjx_model.jnt_range[:16, 1]
    default_pose = jnp.array([0.8, 0, 0.8, 0.8, 0.8, 0, 0.8, 0.8, 0.8, 0, 0.8, 0.8, 0.8, 0.8, 0.8, 0])

    if use_grasp_cache:
        # Sample from grasp cache for stable initial states
        cache_key = hand_keys[0]  # Use first key for cache sampling
        cache_indices = jr.randint(cache_key, shape=(num_envs,), minval=0, maxval=grasp_cache_size)
        cached_states = grasp_cache[cache_indices]  # (num_envs, 23)

        # Extract DOF positions and object states from cache
        cached_dofs = cached_states[:, :16]
        cached_cube_pos = cached_states[:, 16:19]
        cached_cube_quat = cached_states[:, 19:23]

        # Add noise for diversity
        if use_domain_randomization:
            # Full DR noise
            dof_noise = jax.vmap(lambda k: jr.uniform(k, shape=(16,), minval=-0.05, maxval=0.05))(hand_keys)
            randomized_dofs = jnp.clip(cached_dofs + dof_noise, joint_lower, joint_upper)
            pos_noise = jax.vmap(lambda k: jr.uniform(k, shape=(3,), minval=-0.01, maxval=0.01))(cube_pos_keys)
            cube_positions = cached_cube_pos + pos_noise
        else:
            # Small eval noise for robustness (not full DR, but prevents overfitting to exact cached states)
            dof_noise = jax.vmap(lambda k: jr.uniform(k, shape=(16,), minval=-0.02, maxval=0.02))(hand_keys)
            randomized_dofs = jnp.clip(cached_dofs + dof_noise, joint_lower, joint_upper)
            pos_noise = jax.vmap(lambda k: jr.uniform(k, shape=(3,), minval=-0.005, maxval=0.005))(cube_pos_keys)
            cube_positions = cached_cube_pos + pos_noise

        cube_quats = cached_cube_quat
    else:
        # Original random initialization (fallback when no cache)
        # Aggressive randomization for training to force exploration
        hand_noise_scale = 0.3 if use_domain_randomization else 0.0
        hand_noise = jax.vmap(lambda k: jr.uniform(k, shape=(16,), minval=-hand_noise_scale, maxval=hand_noise_scale))(hand_keys)
        randomized_dofs = jnp.clip(default_pose + hand_noise, joint_lower, joint_upper)

        # Randomize cube positions (vectorized)
        cube_positions = jax.vmap(lambda k: jr.uniform(
            k, shape=(3,),
            minval=jnp.array([0.08, -0.02, 0.04]),
            maxval=jnp.array([0.12, 0.02, 0.06])
        ))(cube_pos_keys)

        # Randomize cube orientations (vectorized)
        def random_quat(key):
            u = jr.uniform(key, shape=(3,))
            return jnp.array([
                jnp.sqrt(1-u[0]) * jnp.sin(2*jnp.pi*u[1]),
                jnp.sqrt(1-u[0]) * jnp.cos(2*jnp.pi*u[1]),
                jnp.sqrt(u[0]) * jnp.sin(2*jnp.pi*u[2]),
                jnp.sqrt(u[0]) * jnp.cos(2*jnp.pi*u[2])
            ])
        cube_quats = jax.vmap(random_quat)(cube_quat_keys)

    # Build randomized qpos
    randomized_qpos = jnp.concatenate([randomized_dofs, cube_positions, cube_quats], axis=1)

    # Select based on reset_mask
    new_qpos = jnp.where(reset_mask[:, None], randomized_qpos, mjx_data_batch.qpos)

    # set ctrl to match hand joints (prevents PD controller from opening hand)
    # set qvel to 0 (start stationary)
    new_ctrl = new_qpos[:, :16]
    new_qvel = jnp.zeros_like(mjx_data_batch.qvel)

    mjx_data_batch = mjx_data_batch.replace(
        qpos=new_qpos,
        qvel=new_qvel,  # Clear velocities (except cube spin if training)
        ctrl=new_ctrl   # Initialize PD targets to match hand pose!
    )

    if use_domain_randomization:
        # Perform batched domain randomization directly
        def re_randomize_fn():
            # We need to generate random parameters for ALL environments
            # IDs are passed in from env.__init__ (looked up by name)
            # BASE VALUES are used to compute absolute randomized values (avoid compounding!)
            hand_joint_ids = jnp.arange(16)

            keys = jr.split(dr_key, 10)

            # 1. Fingertip friction: base * U(0.5, 1.0)
            fingertip_friction_mult = jr.uniform(keys[0], (num_envs, 4), minval=0.5, maxval=1.0)
            # Compute absolute values from base
            new_fingertip_friction = base_fingertip_friction[None, :] * fingertip_friction_mult  # (num_envs, 4)
            geom_friction = mjx_model.geom_friction
            for i in range(4):
                gid = fingertip_geom_ids[i]
                geom_friction = geom_friction.at[:, gid, 0].set(new_fingertip_friction[:, i])

            # 2. Cube friction: base * U(0.7, 1.5)
            cube_friction_mult = jr.uniform(keys[9], (num_envs,), minval=0.7, maxval=1.5)
            new_cube_friction = base_cube_friction * cube_friction_mult
            geom_friction = geom_friction.at[:, cube_geom_id, 0].set(new_cube_friction)

            # 3. Cube mass: base * U(0.8, 1.2)
            cube_mass_mult = jr.uniform(keys[1], (num_envs,), minval=0.8, maxval=1.2)
            new_cube_mass = base_cube_mass * cube_mass_mult
            body_mass = mjx_model.body_mass.at[:, cube_body_id].set(new_cube_mass)

            # 4. Cube inertia: base * U(0.5, 1.5)
            cube_inertia_mult = jr.uniform(keys[2], (num_envs, 3), minval=0.5, maxval=1.5)
            new_cube_inertia = base_cube_inertia[None, :] * cube_inertia_mult
            body_inertia = mjx_model.body_inertia.at[:, cube_body_id].set(new_cube_inertia)

            # 5. Joint frictionloss: base * U(0.5, 2.0)
            frictionloss_mult = jr.uniform(keys[3], (num_envs, 16), minval=0.5, maxval=2.0)
            new_frictionloss = base_dof_frictionloss[None, :] * frictionloss_mult
            dof_frictionloss = mjx_model.dof_frictionloss.at[:, hand_joint_ids].set(new_frictionloss)

            # 6. Joint armature: base * U(1.0, 1.05)
            armature_mult = jr.uniform(keys[4], (num_envs, 16), minval=1.0, maxval=1.05)
            new_armature = base_dof_armature[None, :] * armature_mult
            dof_armature = mjx_model.dof_armature.at[:, hand_joint_ids].set(new_armature)

            # 7. Link masses: base * U(0.9, 1.1)
            link_mass_mult = jr.uniform(keys[5], (num_envs, 17), minval=0.9, maxval=1.1)
            new_hand_mass = base_hand_body_mass[None, :] * link_mass_mult
            for i in range(17):
                bid = hand_body_ids[i]
                body_mass = body_mass.at[:, bid].set(new_hand_mass[:, i])

            # 8. Joint stiffness (kp): base * U(0.8, 1.2)
            kp_mult = jr.uniform(keys[6], (num_envs, mjx_model.nu), minval=0.8, maxval=1.2)
            new_kp = base_actuator_gainprm[None, :] * kp_mult
            actuator_gainprm = mjx_model.actuator_gainprm.at[:, :, 0].set(new_kp)
            # Update biasprm to match (biasprm[1] = -kp)
            actuator_biasprm = mjx_model.actuator_biasprm.at[:, :, 1].set(-new_kp)

            # 9. Joint damping (kd): base * U(0.8, 1.2)
            kd_mult = jr.uniform(keys[7], (num_envs, 16), minval=0.8, maxval=1.2)
            new_damping = base_dof_damping[None, :] * kd_mult
            dof_damping = mjx_model.dof_damping.at[:, hand_joint_ids].set(new_damping)

            # 10. Initial qpos: base + U(-0.05, 0.05)
            qpos0_offset = jr.uniform(keys[8], (num_envs, 16), minval=-0.05, maxval=0.05)
            new_qpos0 = base_qpos0[None, :] + qpos0_offset
            qpos0 = mjx_model.qpos0.at[:, hand_joint_ids].set(new_qpos0)

            return mjx_model.tree_replace({
                "geom_friction": geom_friction,
                "body_mass": body_mass,
                "body_inertia": body_inertia,
                "dof_frictionloss": dof_frictionloss,
                "dof_armature": dof_armature,
                "dof_damping": dof_damping,
                "actuator_gainprm": actuator_gainprm,
                "actuator_biasprm": actuator_biasprm,
                "qpos0": qpos0,
            })

        # Generate candidate new model
        candidate_model = re_randomize_fn()
        
        # Selectively update fields based on reset_mask
        # We need to iterate over all fields of mjx_model that are batched and apply mask
        # mjx_model is a pytree.
        
        # Mask shape: (num_envs,)
        def merge_field(new_val, old_val):
             # Broadcast mask to value shape
             # Assuming dim 0 is batch
             if hasattr(new_val, 'shape') and new_val.shape and new_val.shape[0] == num_envs:
                  mask_reshaped = reset_mask.reshape( (num_envs,) + (1,)*(new_val.ndim-1) )
                  return jnp.where(mask_reshaped, new_val, old_val)
             return old_val # Keep old if not batched (shouldn't happen for randomized fields)

        mjx_model = jax.tree_util.tree_map(merge_field, candidate_model, mjx_model)

    return mjx_model, mjx_data_batch
