from etils import epath

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import mujoco
from mujoco import mjx

from examples.leap_hand.rewards import compute_total_reward, check_termination
from examples.leap_hand.domain_randomization_model import build_leap_hand_dr_params
from recurrl_jax.utils.domain_randomization import create_batched_randomized_models
from recurrl_jax.utils.mjx_env import EnvSpec, mjx_training_step, _mjx_reset_jit


_LEAP_DEFAULT_POSE = (0.8, 0.0, 0.8, 0.8, 0.8, 0.0, 0.8, 0.8, 0.8, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0)


def build_leap_spec(mj_model) -> EnvSpec:
    fingertip_geom_ids = tuple(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("th_tip", "if_tip", "mf_tip", "rf_tip")
    )
    cube_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    return EnvSpec(
        n_dofs=16,
        dof_slice=(0, 16),
        obj_pos_slice=(16, 19),
        obj_quat_slice=(19, 23),
        obj_linvel_slice=(16, 19),
        obj_angvel_slice=(19, 22),
        default_pose=_LEAP_DEFAULT_POSE,
        contact_geom_ids=fingertip_geom_ids,
        obj_geom_id=cube_geom_id,
    )


def _leap_reward_fn(state, reset_height_threshold):
    reward, info, new_angvel_z_smooth = compute_total_reward(
        object_angvel=state['obj_angvel'],
        object_pos=state['obj_pos'],
        prev_angvel_z_smooth=state['prev_aux'],
        object_linvel=state['obj_linvel'],
        actions=state['actions'],
        dof_vel=state['dof_vel'],
        torques=state['torques'],
        dof_pos=state['dof_pos'],
        init_dof_pos=state['initial_dof_pos'],
        reset_height_threshold=reset_height_threshold,
        fingertip_cube_contact=state.get('contact'),
    )
    return reward, info, new_angvel_z_smooth


def _leap_termination_fn(state, progress_buf, max_episode_length, reset_height_threshold):
    return check_termination(
        object_pos=state['obj_pos'],
        progress_buf=progress_buf,
        dof_vel=state['dof_vel'],
        cube_linvel=state['obj_linvel'],
        cube_angvel=state['obj_angvel'],
        max_episode_length=max_episode_length,
        reset_height_threshold=reset_height_threshold,
    )


def _leap_reset_state_fn(key, data_batch, reset_mask, spec, use_dr, base_mjx, grasp_cache, grasp_cache_size):
    num_envs = reset_mask.shape[0]
    use_grasp_cache = grasp_cache is not None and grasp_cache_size > 0

    keys = jr.split(key, num_envs + 1)
    env_keys = keys[:-1]
    keys_split = jax.vmap(lambda k: jr.split(k, 4))(env_keys)
    hand_keys = keys_split[:, 0]
    cube_pos_keys = keys_split[:, 1]
    cube_quat_keys = keys_split[:, 2]

    joint_lower = base_mjx.jnt_range[:16, 0]
    joint_upper = base_mjx.jnt_range[:16, 1]
    default_pose = jnp.array(spec.default_pose)

    if use_grasp_cache:
        cache_indices = jr.randint(hand_keys[0], shape=(num_envs,), minval=0, maxval=grasp_cache_size)
        cached_states = grasp_cache[cache_indices]  # (num_envs, 23)
        cached_dofs = cached_states[:, :16]
        cached_cube_pos = cached_states[:, 16:19]
        cached_cube_quat = cached_states[:, 19:23]

        dof_noise_scale = 0.05 if use_dr else 0.02
        dof_noise = jax.vmap(lambda k: jr.uniform(k, shape=(16,), minval=-dof_noise_scale, maxval=dof_noise_scale))(hand_keys)
        randomized_dofs = jnp.clip(cached_dofs + dof_noise, joint_lower, joint_upper)
        pos_noise_scale = 0.01 if use_dr else 0.005
        pos_noise = jax.vmap(lambda k: jr.uniform(k, shape=(3,), minval=-pos_noise_scale, maxval=pos_noise_scale))(cube_pos_keys)
        cube_positions = cached_cube_pos + pos_noise
        cube_quats = cached_cube_quat
    else:
        hand_noise_scale = 0.3 if use_dr else 0.0
        hand_noise = jax.vmap(lambda k: jr.uniform(k, shape=(16,), minval=-hand_noise_scale, maxval=hand_noise_scale))(hand_keys)
        randomized_dofs = jnp.clip(default_pose + hand_noise, joint_lower, joint_upper)

        cube_positions = jax.vmap(lambda k: jr.uniform(
            k, shape=(3,),
            minval=jnp.array([0.08, -0.02, 0.04]),
            maxval=jnp.array([0.12, 0.02, 0.06])
        ))(cube_pos_keys)

        def random_quat(k):
            u = jr.uniform(k, shape=(3,))
            return jnp.array([
                jnp.sqrt(1 - u[0]) * jnp.sin(2 * jnp.pi * u[1]),
                jnp.sqrt(1 - u[0]) * jnp.cos(2 * jnp.pi * u[1]),
                jnp.sqrt(u[0]) * jnp.sin(2 * jnp.pi * u[2]),
                jnp.sqrt(u[0]) * jnp.cos(2 * jnp.pi * u[2]),
            ])
        cube_quats = jax.vmap(random_quat)(cube_quat_keys)

    randomized_qpos = jnp.concatenate([randomized_dofs, cube_positions, cube_quats], axis=1)
    new_qpos = jnp.where(reset_mask[:, None], randomized_qpos, data_batch.qpos)
    new_ctrl = new_qpos[:, :16]
    new_qvel = jnp.zeros_like(data_batch.qvel)

    return data_batch.replace(qpos=new_qpos, qvel=new_qvel, ctrl=new_ctrl)


class MJXLeapHandEnv:
    def __init__(self, xml_path: str, num_envs: int, key: jax.Array,
                 action_scale: float = 0.6, use_domain_randomization: bool = False,
                 grasp_cache_path: str = None, action_ema_alpha: float = 0.0):
        self.mjx_path = epath.Path(xml_path).as_posix()
        self.num_envs = num_envs
        self.key = key
        self.action_scale = action_scale
        self.action_ema_alpha = action_ema_alpha
        self.use_domain_randomization = use_domain_randomization

        self.progress_buf = jnp.zeros(num_envs, dtype=jnp.int32)
        self.initial_dof_pos = None
        self.prev_smoothed_actions = None
        self.reset_height_threshold = -0.05
        self.max_episode_length = 500
        self.control_freq_inv = 5
        self.angvel_z_smooth = jnp.zeros(num_envs)

        self.grasp_cache = None
        self.grasp_cache_size = 0
        if grasp_cache_path is not None:
            cache_data = np.load(grasp_cache_path)
            self.grasp_cache = jnp.array(cache_data)
            self.grasp_cache_size = self.grasp_cache.shape[0]
            print(f"Loaded grasp cache from {grasp_cache_path}: {self.grasp_cache_size} grasps")

        self.mj_model = mujoco.MjModel.from_xml_path(self.mjx_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.if_tip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "if_tip")
        self.mf_tip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "mf_tip")
        self.rf_tip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rf_tip")
        self.th_tip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "th_tip")
        self.palm_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "palm")

        self.base_mjx = self.mjx_model
        self.dr_params = build_leap_hand_dr_params(self.mj_model)
        self.spec = build_leap_spec(self.mj_model)

        self.mjx_model_batch, self.mjx_data_batch = self._create_batch()

        self.num_dofs = self.mj_model.nu
        self.ctrl_range = self.mjx_model.actuator_ctrlrange
        self.joint_lower_limits = self.mjx_model.jnt_range[:, 0][:16]
        self.joint_upper_limits = self.mjx_model.jnt_range[:, 1][:16]

    def _create_batch(self):
        if self.use_domain_randomization:
            self.key, dr_key = jr.split(self.key)
            batched_model, _ = create_batched_randomized_models(
                self.mjx_model, self.num_envs, dr_key, self.dr_params
            )
        else:
            batched_model = self.mjx_model

        mjx_data_batch = jax.vmap(lambda _: mjx.make_data(self.mjx_model))(
            jnp.arange(self.num_envs)
        )
        return batched_model, mjx_data_batch

    def step(self, actions: jnp.ndarray):
        self.key, step_key = jr.split(self.key)

        if self.action_ema_alpha > 0.0:
            if self.prev_smoothed_actions is None:
                self.prev_smoothed_actions = actions
            smoothed_actions = self.action_ema_alpha * self.prev_smoothed_actions + (1.0 - self.action_ema_alpha) * actions
        else:
            smoothed_actions = actions

        (
            state, reward, reset_mask, termination, info,
            new_data, new_progress_buf, new_mjx_model, new_angvel_z_smooth
        ) = mjx_training_step(
            smoothed_actions,
            self.mjx_model_batch,
            self.mjx_data_batch,
            self.progress_buf,
            self.initial_dof_pos,
            self.reset_height_threshold,
            self.max_episode_length,
            step_key,
            self.angvel_z_smooth,
            spec=self.spec,
            reward_fn=_leap_reward_fn,
            termination_fn=_leap_termination_fn,
            reset_state_fn=_leap_reset_state_fn,
            control_freq_inv=self.control_freq_inv,
            use_domain_randomization=self.use_domain_randomization,
            dr_params=self.dr_params,
            base_mjx=self.base_mjx,
            action_scale=self.action_scale,
            extra_cache=self.grasp_cache,
            extra_cache_size=self.grasp_cache_size,
        )

        self.mjx_data_batch = new_data
        self.progress_buf = new_progress_buf
        self.mjx_model_batch = new_mjx_model
        self.angvel_z_smooth = new_angvel_z_smooth
        new_pose = new_data.qpos[:, :16]
        self.initial_dof_pos = jnp.where(reset_mask[:, None], new_pose, self.initial_dof_pos)

        if self.action_ema_alpha > 0.0:
            self.prev_smoothed_actions = jnp.where(reset_mask[:, None], jnp.zeros_like(smoothed_actions), smoothed_actions)

        return state, reward, reset_mask, termination, info, new_data, new_progress_buf, new_mjx_model, new_angvel_z_smooth

    def reset(self, env_ids: jnp.ndarray = None):
        if env_ids is None:
            reset_mask = jnp.ones(self.num_envs, dtype=bool)
        else:
            reset_mask = jnp.zeros(self.num_envs, dtype=bool).at[env_ids].set(True)

        self.key, subkey = jr.split(self.key)
        self.mjx_model_batch, self.mjx_data_batch = _mjx_reset_jit(
            subkey,
            self.mjx_model_batch,
            self.mjx_data_batch,
            reset_mask,
            self.grasp_cache,
            use_domain_randomization=self.use_domain_randomization,
            dr_params=self.dr_params,
            base_mjx=self.base_mjx,
            extra_cache_size=self.grasp_cache_size,
            spec=self.spec,
            reset_state_fn=_leap_reset_state_fn,
        )

        self.progress_buf = jnp.where(reset_mask, 0, self.progress_buf)
        self.angvel_z_smooth = jnp.where(reset_mask, 0.0, self.angvel_z_smooth)
        if self.action_ema_alpha > 0.0 and self.prev_smoothed_actions is not None:
            self.prev_smoothed_actions = jnp.where(reset_mask[:, None], jnp.zeros_like(self.prev_smoothed_actions), self.prev_smoothed_actions)

        new_pose = self.mjx_data_batch.qpos[:, :16]
        if self.initial_dof_pos is None or env_ids is None:
            self.initial_dof_pos = new_pose
        else:
            self.initial_dof_pos = jnp.where(reset_mask[:, None], new_pose, self.initial_dof_pos)

    def get_fingertip_positions(self, mjx_data_batch=None):
        if mjx_data_batch is None:
            mjx_data_batch = self.mjx_data_batch
        return jnp.concatenate([
            mjx_data_batch.site_xpos[:, self.if_tip_id, :],
            mjx_data_batch.site_xpos[:, self.mf_tip_id, :],
            mjx_data_batch.site_xpos[:, self.rf_tip_id, :],
            mjx_data_batch.site_xpos[:, self.th_tip_id, :],
        ], axis=-1)

    def get_palm_position(self, mjx_data_batch=None):
        if mjx_data_batch is None:
            mjx_data_batch = self.mjx_data_batch
        return mjx_data_batch.xpos[:, self.palm_id, :]
