"""ANYmal velocity-tracking environment (standalone, no external package deps)."""
import warnings
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from brax import math
from ml_collections import config_dict

from examples.anymal.mjx_env_base import MjxEnv, State
from examples.anymal.domain_randomization_model import AnymalDRConfig, DEFAULT_DR
import examples.anymal.rewards as R

_ANYMAL_XML = str(Path(__file__).parent / "xmls/scene_mjx.xml")

# actuator indices grouped by leg: LF RF LH RH (3 joints each)
_LEG_IDS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@jax.custom_jvp
def _safe_nan_to_num(x):
    return jp.nan_to_num(x)

@_safe_nan_to_num.defjvp
def _safe_nan_jvp(primals, tangents):
    x, = primals
    t, = tangents
    return jp.nan_to_num(x), jp.where(jp.isnan(x), 0.0, t)


def _axis_angle_to_quaternion(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """axis (3,), angle (1,) → quaternion (w x y z) shape (4,)."""
    half = angle / 2.0
    return jp.concatenate([jp.cos(half), jp.sin(half) * axis])


# ---------------------------------------------------------------------------
# eeward scale configs
# ---------------------------------------------------------------------------

def get_config():
    # all scales multiplied by dt=0.02 for frame-rate independence (matches paper magnitude)
    return config_dict.ConfigDict(dict(
        rewards=config_dict.ConfigDict(dict(
            scales=config_dict.ConfigDict(dict(
                lin_vel_tracking=0.2,
                ang_vel_tracking=0.1,
                pose=0.5,
                foot_height=0.5,
                lin_vel_error=0.02,
                ang_vel_error=0.001,
                base_height=0.1,
                base_orientation=5.0,
                action_magnitude=0.01,
                action_rate=0.003,
                joint_acceleration=1e-8,
                joint_torque=1e-6,
                joint_velocity=1e-5,
                feet_air_time=2.0,
                swing_contact=0.5,
            ))
        ))
    ))

class DiffAnymal(MjxEnv):
    """velocity-tracking environment

    obs (49-dim):
        lin base vel (body frame)       3
        ang base vel (body frame)       3
        projected gravity               3
        velocity command [vx* vy* ωz*]  3
        joint positions                 12
        joint velocities                12
        previous action                 12
        phase sin(4πt)                  1
    """

    def __init__(
        self,
        action_scale: float = 0.5,
        termination_height: float = 0.25,
        s_afilt_buf: int = 1,
        smooth_sigma_q: float = 0.0,
        smooth_sigma_v: float = 0.0,
        swing_height: float = 0.1,
        reward_scales: dict = None,
        use_domain_randomization: bool = True,
        dr_config: AnymalDRConfig = DEFAULT_DR,
        cmd_scale: float = 1.0,
        cmd_min_speed: float = 0.0,
        cmd_scale_curriculum_steps: int = 20_000_000,
        **kwargs,
    ):
        self.early_termination = kwargs.pop('early_termination', True)
        kwargs.pop('model_variant', None)
        self.cmd_min_speed = cmd_min_speed
        self._cmd_curriculum_steps = float(cmd_scale_curriculum_steps)

        mj_model = mujoco.MjModel.from_xml_path(_ANYMAL_XML)

        self.s_afilt_buf = s_afilt_buf
        if s_afilt_buf > 1:
            warnings.warn("s_afilt_buf > 1 gives undefined observations")

        kwargs.setdefault('physics_steps_per_control_step', 4)
        super().__init__(mj_model=mj_model, **kwargs)

        self.action_scale = action_scale
        self.termination_height = termination_height
        self.smooth_sigma_q = smooth_sigma_q
        self.smooth_sigma_v = smooth_sigma_v
        self.swing_height = swing_height
        self._use_dr = use_domain_randomization
        self._dr = dr_config
        self._cmd_scale = jp.array([cmd_scale, cmd_scale, cmd_scale])

        # trot gait: LF+RH in phase (offset=0), RF+LH offset by π. Order: LF RF LH RH
        self.foot_phase_offsets = jp.array([0.0, jp.pi, jp.pi, 0.0])
        # foot contact geom IDs — LF=21, RF=28, LH=35, RH=42
        self.foot_geom_ids = jp.array([21, 28, 35, 42])
        # per-joint pose std: order is [HAA, HFE, KFE] * 4 legs
        self._pose_std_standing = jp.array([0.15, 0.15, 0.25] * 4)
        self._pose_std_walking  = jp.array([0.30, 0.30, 0.60] * 4)

        self._init_q = mj_model.keyframe('standing').qpos
        self._default_ap_pose = mj_model.keyframe('standing').qpos[7:]
        self.reward_config = get_config()
        if reward_scales:
            for k, v in reward_scales.items():
                self.reward_config.rewards.scales[k] = v

        self._resample_min = int(round(10.0 / self.dt))
        self._resample_max = int(round(15.0 / self.dt))
        self._base_body_id = 1
        self._kick_min = int(round(self._dr.kick_interval_min_s / self.dt))
        self._kick_max = int(round(self._dr.kick_interval_max_s / self.dt))
        self._curriculum_stage2 = self._dr.curriculum_stage2

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_body_frame(self, v_world: jax.Array, q_body: jax.Array) -> jax.Array:
        q_inv = jp.array([q_body[0], -q_body[1], -q_body[2], -q_body[3]])
        return math.rotate(v_world, q_inv)

    def _build_dr_sys(self, foot_friction, added_mass, leg_scales=None):
        new_geom_friction = self.sys.geom_friction.at[self.foot_geom_ids, 0].set(foot_friction)
        new_body_mass = self.sys.body_mass.at[self._base_body_id].add(added_mass)
        sys = self.sys.replace(geom_friction=new_geom_friction, body_mass=new_body_mass)
        if leg_scales is not None:
            new_gainprm = sys.actuator_gainprm
            new_biasprm = sys.actuator_biasprm
            for i, leg_ids in enumerate(_LEG_IDS):
                new_gainprm = new_gainprm.at[leg_ids, 0].set(
                    self.sys.actuator_gainprm[leg_ids, 0] * leg_scales[i])
                new_biasprm = new_biasprm.at[leg_ids, 1].set(
                    self.sys.actuator_biasprm[leg_ids, 1] * leg_scales[i])
            sys = sys.replace(actuator_gainprm=new_gainprm, actuator_biasprm=new_biasprm)
        return sys

    def _apply_extra_dr(self, sys, info):
        return sys

    def _pipeline_step_dr(self, data: mjx.Data, ctrl: jax.Array, sys) -> mjx.Data:
        def f(data, _):
            data = data.replace(ctrl=ctrl)
            return mjx.step(sys, data), None
        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    # ------------------------------------------------------------------
    # reset
    def _enforce_min_speed(self, cmd, rng):
        xy = cmd[:2]
        speed = jp.linalg.norm(xy)
        rand_dir = jax.random.normal(rng, (2,))
        rand_dir = rand_dir / jp.linalg.norm(rand_dir)
        safe_dir = jp.where(speed > 1e-6, xy / speed, rand_dir)
        return cmd.at[:2].set(safe_dir * jp.maximum(speed, self.cmd_min_speed))

    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, key_xyz, key_ang, key_ax, key_q, key_qd, key_cmd, key_t, \
            key_fr, key_mass, key_kick, key_ls = jax.random.split(rng, 12)

        qpos = jp.array(self._init_q)
        qvel = jp.zeros(18)

        r_xy   = 0.2 * (jax.random.uniform(key_xyz, (2,)) - 0.5)
        r_xyz  = jp.concatenate([r_xy, jp.zeros(1)])
        r_angle = jp.zeros(1)
        r_axis  = jp.array([0.0, 0.0, 1.0])
        r_quat  = _axis_angle_to_quaternion(r_axis, r_angle)
        r_joint_q  = 0.1 * (jax.random.uniform(key_q,  (12,)) - 0.5)   # ±0.05 rad
        r_joint_qd = 1.0 * (jax.random.uniform(key_qd, (12,)) - 0.5)   # ±0.5 rad/s

        qpos = qpos.at[0:3].set(qpos[0:3] + r_xyz)
        qpos = qpos.at[3:7].set(r_quat)
        qpos = qpos.at[7:19].set(qpos[7:19] + r_joint_q)
        qvel = qvel.at[6:18].set(r_joint_qd)

        data = self.pipeline_init(qpos, qvel)

        def _sample_cmd(_):
            raw = (jax.random.uniform(key_cmd, (3,)) * 2.0 - 1.0) * jp.array([0.5, 0.5, 0.3]) * self._cmd_scale
            xy_speed = jp.linalg.norm(raw[:2])
            return jp.where(xy_speed < 0.02, raw.at[:2].set(jp.zeros(2)), raw)

        # eval env (curriculum bypassed) gets a real command immediately; training starts at zero
        init_cmd = jax.lax.cond(
            self._cmd_curriculum_steps <= 0,
            _sample_cmd,
            lambda _: jp.zeros(3),
            operand=None,
        )
        init_countdown = jax.random.randint(
            key_t, shape=(), minval=self._resample_min, maxval=self._resample_max)

        dr = self._dr
        dr_foot_friction = jax.lax.cond(
            self._use_dr,
            lambda: jax.random.uniform(key_fr, (), minval=dr.foot_friction_min, maxval=dr.foot_friction_max),
            lambda: self.sys.geom_friction[self.foot_geom_ids[0], 0],
        )
        dr_added_mass = jax.lax.cond(
            self._use_dr,
            lambda: jax.random.uniform(key_mass, (), minval=dr.added_mass_min, maxval=dr.added_mass_max),
            lambda: jp.zeros(()),
        )
        dr_leg_scales = jax.lax.cond(
            self._use_dr,
            lambda: jax.random.uniform(key_ls, (4,), minval=dr.leg_scale_init_min, maxval=dr.leg_scale_max),
            lambda: jp.ones(4),
        )
        vel_kick_countdown = jax.random.randint(
            key_kick, shape=(), minval=self._kick_min, maxval=self._kick_max)

        state_info = {
            'rng': rng,
            'reward_tuple': {
                'lin_vel_tracking': 0.0,
                'ang_vel_tracking': 0.0,
                'pose':             0.0,
                'foot_height':      0.0,
                'lin_vel_error':    0.0,
                'ang_vel_error':    0.0,
                'base_height':      0.0,
                'base_orientation': 0.0,
                'action_magnitude': 0.0,
                'action_rate':      0.0,
                'joint_acceleration': 0.0,
                'joint_torque':     0.0,
                'joint_velocity':   0.0,
                'feet_air_time':    0.0,
                'swing_contact':    0.0,
            },
            'last_action':        jp.array(self._default_ap_pose),
            'afilt_buf':          jp.tile(jp.array(self._default_ap_pose)[None], (self.s_afilt_buf, 1)),
            'step_count':         jp.array(0, dtype=jp.int32),
            'vel_cmd':            init_cmd,
            'resample_countdown': init_countdown,
            'last_joint_vel':     jp.zeros(12),
            'feet_air_time':      jp.zeros(4),
            'last_contact':       jp.zeros(4, dtype=jp.bool_),
            'dr_foot_friction':   dr_foot_friction,
            'dr_added_mass':      dr_added_mass,
            'dr_leg_scales':      dr_leg_scales,
            'curriculum_step':    jp.zeros((), dtype=jp.int32),
            'vel_kick_countdown': vel_kick_countdown,
        }

        x, xd = self._pos_vel(data)
        obs = self._get_obs(data.qpos, data.qvel, x, xd, state_info)
        reward, done = jp.zeros(2)
        metrics = {k: state_info['reward_tuple'][k] for k in state_info['reward_tuple']}
        metrics.update({'term_height': jp.zeros(()), 'term_invalid': jp.zeros(()), 'term_flip': jp.zeros(())})
        return State(data, obs, reward, done, metrics, state_info)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1, 1)
        raw_action = action
        action_target = jp.array(self._default_ap_pose) + action * self.action_scale
        afilt_buf = jp.roll(state.info['afilt_buf'], shift=1, axis=0).at[0, :].set(action_target)
        f_action  = jp.mean(afilt_buf, axis=0)
        state.info['afilt_buf'] = afilt_buf

        env_sys = self._build_dr_sys(
            jax.lax.stop_gradient(state.info['dr_foot_friction']),
            jax.lax.stop_gradient(state.info['dr_added_mass']),
            jax.lax.stop_gradient(state.info['dr_leg_scales']),
        )
        env_sys = self._apply_extra_dr(env_sys, state.info)

        # velocity kick
        rng, kick_key, kick_t_key = jax.random.split(state.info['rng'], 3)
        state.info['rng'] = rng
        dr = self._dr
        vel_kick = jax.random.uniform(kick_key, (3,), minval=dr.kick_vel_min, maxval=dr.kick_vel_max)
        should_kick = state.info['vel_kick_countdown'] <= 0
        ps = state.pipeline_state
        new_kick_countdown = jax.random.randint(
            kick_t_key, shape=(), minval=self._kick_min, maxval=self._kick_max)
        state.info['vel_kick_countdown'] = jp.where(
            should_kick, new_kick_countdown, state.info['vel_kick_countdown'] - 1)

        if self.smooth_sigma_q > 0.0:
            rng, key = jax.random.split(state.info['rng'])
            state.info['rng'] = rng
            eps_q = jax.random.normal(key, (12,)) * self.smooth_sigma_q
            eps_v = jax.random.normal(key, (12,)) * self.smooth_sigma_v
            data_p = self._pipeline_step_dr(
                ps.replace(qpos=ps.qpos.at[7:].add(+eps_q), qvel=ps.qvel.at[6:].add(+eps_v)),
                f_action, env_sys)
            data_m = self._pipeline_step_dr(
                ps.replace(qpos=ps.qpos.at[7:].add(-eps_q), qvel=ps.qvel.at[6:].add(-eps_v)),
                f_action, env_sys)
            data = jax.tree_util.tree_map(
                lambda a, b: 0.5 * (a + b) if jp.issubdtype(a.dtype, jp.floating) else a,
                data_p, data_m)
        else:
            data = self._pipeline_step_dr(ps, f_action, env_sys)

        step_count = state.info['step_count'] + 1
        state.info['step_count'] = step_count
        phase_rad = 4.0 * jp.pi * jp.asarray(step_count, jp.float32) * self.dt

        # cmd curriculum: 0→full as curriculum_step advances
        # _cmd_curriculum_steps<=0 bypasses curriculum (full scale always, used for eval env)
        cs = state.info['curriculum_step']
        cmd_progress = jp.where(
            self._cmd_curriculum_steps <= 0,
            1.0,
            jp.clip(cs.astype(jp.float32) / jp.maximum(self._cmd_curriculum_steps, 1.0), 0.0, 1.0),
        )
        effective_cmd_scale = self._cmd_scale * cmd_progress
        # feet_air_time threshold: 0.2s at start → 0.4s at full curriculum
        air_threshold = 0.2 + 0.2 * cmd_progress

        # velocity command resampling with dead-zone zeroing
        rng, key_cmd, key_t = jax.random.split(state.info['rng'], 3)
        state.info['rng'] = rng
        new_cmd = (jax.random.uniform(key_cmd, (3,)) * 2.0 - 1.0) * jp.array([0.5, 0.5, 0.3]) * effective_cmd_scale
        xy_speed = jp.linalg.norm(new_cmd[:2])
        new_cmd = jp.where(xy_speed < 0.02, new_cmd.at[:2].set(jp.zeros(2)), new_cmd)
        should_resample = state.info['resample_countdown'] <= 0
        vel_cmd = jp.where(should_resample, new_cmd, state.info['vel_cmd'])
        new_countdown = jax.random.randint(
            key_t, shape=(), minval=self._resample_min, maxval=self._resample_max)
        state.info['vel_cmd']            = vel_cmd
        state.info['resample_countdown'] = jp.where(should_resample, new_countdown,
                                                     state.info['resample_countdown'] - 1)

        x, xd = self._pos_vel(data)
        obs   = self._get_obs(data.qpos, data.qvel, x, xd, state.info, phase_rad)
        done, term_reasons = R.check_termination(x, obs, data, self.termination_height, self.early_termination)

        data = jax.tree_util.tree_map(
            lambda v: _safe_nan_to_num(v) if jp.issubdtype(v.dtype, jp.floating) else v, data)
        x, xd = self._pos_vel(data)
        obs   = self._get_obs(data.qpos, data.qvel, x, xd, state.info, phase_rad)

        q          = x.rot[0]
        v_lin_body = self._to_body_frame(xd.vel[0], q)
        v_ang_body = self._to_body_frame(xd.ang[0], q)
        g_body     = self._to_body_frame(jp.array([0., 0., -1.]), q)
        joint_vel  = data.qvel[6:]
        joint_acc  = (joint_vel - state.info['last_joint_vel']) / self.dt

        foot_z      = data.geom_xpos[self.foot_geom_ids, 2]
        contact     = foot_z < 0.07
        first_contact = contact & ~state.info['last_contact']

        s = self.reward_config.rewards.scales
        cmd_speed = jp.linalg.norm(vel_cmd[:2])
        pose_alpha = jp.tanh(cmd_speed / 0.5)
        pose_std = self._pose_std_standing * (1.0 - pose_alpha) + self._pose_std_walking * pose_alpha
        reward_tuple = {
            'lin_vel_tracking':   R.reward_lin_vel_tracking(v_lin_body, vel_cmd)    * s.lin_vel_tracking,
            'ang_vel_tracking':   R.reward_ang_vel_tracking(v_ang_body, vel_cmd)    * s.ang_vel_tracking,
            'pose':               R.reward_pose(data.qpos[7:], jp.array(self._default_ap_pose), pose_std) * s.pose,
            'foot_height':        R.reward_foot_height(data, phase_rad,
                                      self.foot_geom_ids, self.foot_phase_offsets,
                                      self.swing_height)                             * s.foot_height,
            'lin_vel_error':      R.reward_lin_vel_error(v_lin_body)                * s.lin_vel_error,
            'ang_vel_error':      R.reward_ang_vel_error(v_ang_body)                * s.ang_vel_error,
            'base_height':        R.reward_base_height(x, target=float(self._init_q[2])) * s.base_height,
            'base_orientation':   R.reward_base_orientation(g_body)                 * s.base_orientation,
            'action_magnitude':   R.reward_action_magnitude(raw_action)             * s.action_magnitude,
            'action_rate':        R.reward_action_rate(f_action, state.info['last_action']) * s.action_rate,
            'joint_acceleration': R.reward_joint_acceleration(joint_acc)            * s.joint_acceleration,
            'joint_torque':       R.reward_joint_torque(data)                       * s.joint_torque,
            'joint_velocity':     R.reward_joint_velocity(joint_vel)                * s.joint_velocity,
            'feet_air_time':      R.reward_feet_air_time(
                                      state.info['feet_air_time'], first_contact, v_lin_body, air_threshold) * s.feet_air_time,
            'swing_contact':      R.reward_swing_contact(data, phase_rad,
                                      self.foot_geom_ids, self.foot_phase_offsets,
                                      self.swing_height)                           * s.swing_contact,
        }

        reward = sum(reward_tuple.values())
        state.info['reward_tuple']   = reward_tuple
        state.info['last_action']    = f_action
        state.info['last_joint_vel'] = joint_vel
        state.info['feet_air_time']  = jp.where(contact, jp.zeros(4),
                                                state.info['feet_air_time'] + self.dt)
        state.info['last_contact']   = contact
        for k in reward_tuple:
            state.metrics[k] = reward_tuple[k]
        for k, v in term_reasons.items():
            state.metrics[k] = v

        # curriculum: anneal leg-scale lower bound over training
        rng, key_ls = jax.random.split(state.info['rng'])
        state.info['rng'] = rng
        cs = state.info['curriculum_step']
        progress = jp.clip(cs.astype(jp.float32) / float(self._curriculum_stage2), 0.0, 1.0)
        lower_bound = (dr.leg_scale_init_min
                       - (dr.leg_scale_init_min - dr.leg_scale_final_min) * progress)
        new_leg_scales = jax.random.uniform(key_ls, (4,), minval=lower_bound, maxval=dr.leg_scale_max)
        max_floor = jp.minimum(lower_bound + 0.2, 1.0)
        best_idx = jp.argmax(new_leg_scales)
        new_leg_scales = new_leg_scales.at[best_idx].set(
            jp.maximum(new_leg_scales[best_idx], max_floor))
        new_leg_scales = jp.where(self._use_dr, new_leg_scales, jp.ones(4))
        state.info['dr_leg_scales'] = jp.where(done, new_leg_scales, state.info['dr_leg_scales'])

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def _get_obs(self, qpos, qvel, x, xd, state_info, phase_rad=None):
        q          = x.rot[0]
        v_lin_body = self._to_body_frame(xd.vel[0], q)
        v_ang_body = self._to_body_frame(xd.ang[0], q)
        g_body     = self._to_body_frame(jp.array([0., 0., -1.]), q)
        if phase_rad is None:
            phase_rad = jp.array(0.0)
        return jp.concatenate([
            v_lin_body,
            v_ang_body,
            g_body,
            state_info['vel_cmd'],
            qpos[7:],
            qvel[6:],
            (state_info['last_action'] - jp.array(self._default_ap_pose)) / self.action_scale,
            jp.sin(phase_rad).reshape(1),
        ])
