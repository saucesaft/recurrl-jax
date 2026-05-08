"""domain randomization configuration"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AnymalDRConfig:
    # foot-ground friction (scalar per env)
    foot_friction_min: float = 0.5
    foot_friction_max: float = 1.25

    # added trunk mass (kg) — Rudin et al. 2022
    added_mass_min: float = -1.0
    added_mass_max: float = 3.0

    # per-leg actuator torque scale — Rudin et al. 2022
    leg_scale_init_min: float = 0.9
    leg_scale_final_min: float = 0.8
    leg_scale_max: float = 1.1

    # random velocity kick applied to trunk (m/s per axis) — Rudin et al. 2022
    kick_vel_min: float = -0.5
    kick_vel_max: float = 0.5
    kick_interval_min_s: float = 5.0   # seconds between kicks
    kick_interval_max_s: float = 10.0

    # curriculum step boundaries (in env_steps = num_envs × unroll × updates)
    curriculum_stage1: int = 40_000 * 512 * 32
    curriculum_stage2: int = 80_000 * 512 * 32


DEFAULT_DR = AnymalDRConfig()
