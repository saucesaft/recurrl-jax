"""brax-style MJX environment base class """
from typing import Any, Dict, Tuple

import jax
from jax import numpy as jp
from flax import struct
import mujoco
from mujoco import mjx
import numpy as np

from brax.base import Base, Motion, Transform


@struct.dataclass
class State(Base):
    pipeline_state: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class MjxEnv:
    """base class for MJX environments with brax-style step/reset API."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        physics_steps_per_control_step: int = 1,
        **kwargs,
    ):
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.put_model(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        data = mjx.put_data(self.model, self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)
        return data

    def pipeline_step(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        def f(data, _):
            data = data.replace(ctrl=ctrl)
            return mjx.step(self.sys, data), None
        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(self) -> float:
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def action_size(self) -> int:
        return self.sys.nu

    def _pos_vel(self, data: mjx.Data) -> Tuple[Transform, Motion]:
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[
            self.model.body_rootid[np.arange(1, self.model.nbody)]
        ]
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd
