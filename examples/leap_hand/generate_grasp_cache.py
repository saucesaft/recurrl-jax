"""
Generate a grasp cache for the LEAP hand reorientation task.

Usage:
    python generate_grasp_cache.py --num_grasps 10000 --output grasp_cache/grasp_cache.npy
"""

import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from examples.leap_hand.env import MJXLeapHandEnv


# canonical grasp pose (cupped hand shape)
# joint ordering: [MCP, ABD, PIP, DIP] x 4 fingers
CANONICAL_POSE_MJX = jnp.array([
    # index finger: flexed MCP, slight abd, flexed PIP/DIP
    1.2,   0.1,  0.3,  0.3,
    # middle finger: most flexed (contacts cube)
    1.3,   0.0,  0.9,  0.15,
    # ring finger: flexed
    1.1,   0.0,  0.1,  0.15,
    # thumb: opposing, flexed to contact cube
    1.3,   0.3,  0.3,  0.3,
])

# search radius for diversity
DOF_SEARCH_RADIUS = 0.25

# distance threshold for fingertip proximity
FINGER_DIST_THRESHOLD = 0.11

# survival steps: how long grasp must hold
SURVIVAL_STEPS = 200


def check_grasp_quality_strict(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    fingertip_positions: jnp.ndarray,
    cube_pos: jnp.ndarray,
    finger_dist_threshold: float = FINGER_DIST_THRESHOLD
) -> jnp.ndarray:
    """check grasp quality: requires all 4 fingertips within threshold"""
    # reshape fingertips to (num_envs, 4, 3)
    fingertips = fingertip_positions.reshape(-1, 4, 3)

    # cube height (not fallen, not exploded)
    cube_z = cube_pos[:, 2]
    z_valid = (cube_z > 0.02) & (cube_z < 0.15)

    # low velocity (stable grasp)
    cube_linvel = qvel[:, 16:19]
    cube_angvel = qvel[:, 19:22]
    linvel_mag = jnp.linalg.norm(cube_linvel, axis=-1)
    angvel_mag = jnp.linalg.norm(cube_angvel, axis=-1)
    velocity_stable = (linvel_mag < 0.2) & (angvel_mag < 1.0)

    # all 4 fingertips within distance threshold
    distances = jnp.sqrt(jnp.sum((fingertips - cube_pos[:, None, :]) ** 2, axis=-1))  # (num_envs, 4)
    all_fingers_close = jnp.all(distances < finger_dist_threshold, axis=-1)

    # no NaN/Inf
    no_nan = ~jnp.any(jnp.isnan(qpos), axis=-1) & ~jnp.any(jnp.isnan(qvel), axis=-1)
    no_inf = ~jnp.any(jnp.isinf(qpos), axis=-1) & ~jnp.any(jnp.isinf(qvel), axis=-1)

    valid = z_valid & velocity_stable & all_fingers_close & no_nan & no_inf

    return valid, distances


def generate_grasp_cache(
    num_grasps: int = 10000,
    num_envs: int = 8192,
    survival_steps: int = SURVIVAL_STEPS,
    seed: int = 42,
    action_scale: float = 0.3,
    verbose: bool = True
) -> np.ndarray:
    """
    Generate grasp cache by simulating random poses near the canonical grasp.

    Initializes with canonical pose + random noise, lets physics settle for
    survival_steps, then saves grasps where all fingertips remain close to cube.
    """
    if verbose:
        print(f"Target grasps: {num_grasps}")
        print(f"DOF search radius: {DOF_SEARCH_RADIUS}")
        print(f"Finger distance threshold: {FINGER_DIST_THRESHOLD}")
        print(f"Survival steps: {survival_steps}")
        print()

    key = jr.PRNGKey(seed)
    collected_grasps = []

    # statistics for debugging
    total_attempts = 0
    total_envs_tested = 0

    pbar = tqdm(total=num_grasps, desc="Collecting grasps") if verbose else None

    # initialize environment
    key, env_key = jr.split(key)
    env = MJXLeapHandEnv(
        xml_path='leap_hand/xmls/scene_mjx_cube.xml',
        num_envs=num_envs,
        key=env_key,
        action_scale=action_scale,
        use_domain_randomization=False
    )

    joint_lower = env.joint_lower_limits
    joint_upper = env.joint_upper_limits

    max_attempts = 1000

    while len(collected_grasps) < num_grasps and total_attempts < max_attempts:
        total_attempts += 1
        total_envs_tested += num_envs

        # generate random DOF positions around canonical pose
        key, dof_key, cube_key, quat_key = jr.split(key, 4)

        dof_noise = jr.uniform(dof_key, shape=(num_envs, 16),
                               minval=-DOF_SEARCH_RADIUS,
                               maxval=DOF_SEARCH_RADIUS)
        random_dofs = CANONICAL_POSE_MJX + dof_noise
        random_dofs = jnp.clip(random_dofs, joint_lower, joint_upper)

        # cube position: center of palm workspace
        cube_pos_base = jnp.array([0.09, 0.0, 0.045])
        cube_pos_noise = jr.uniform(cube_key, shape=(num_envs, 3),
                                    minval=jnp.array([-0.01, -0.01, -0.005]),
                                    maxval=jnp.array([0.01, 0.01, 0.005]))
        cube_positions = cube_pos_base + cube_pos_noise

        # small rotations around Z for stable initial grasps
        angle_noise = jr.uniform(quat_key, shape=(num_envs,), minval=-0.3, maxval=0.3)
        cube_quats = jnp.stack([
            jnp.zeros(num_envs),
            jnp.zeros(num_envs),
            jnp.sin(angle_noise / 2),
            jnp.cos(angle_noise / 2)
        ], axis=1)

        # build initial qpos
        init_qpos = jnp.concatenate([random_dofs, cube_positions, cube_quats], axis=1)

        # set environment state
        new_qvel = jnp.zeros_like(env.mjx_data_batch.qvel)
        new_ctrl = random_dofs

        env.mjx_data_batch = env.mjx_data_batch.replace(
            qpos=init_qpos,
            qvel=new_qvel,
            ctrl=new_ctrl
        )
        env.prev_targets = random_dofs
        env.initial_dof_pos = random_dofs
        env.progress_buf = jnp.zeros(num_envs, dtype=jnp.int32)

        # let physics settle with zero action (maintain pose)
        zero_action = jnp.zeros((num_envs, 16))
        survived = jnp.ones(num_envs, dtype=bool)

        for step in range(survival_steps):
            _, _, done, _, _, _, _, _ = env.step(zero_action)
            survived = survived & ~done

        # check quality of surviving grasps
        qpos = env.mjx_data_batch.qpos
        qvel = env.mjx_data_batch.qvel
        cube_pos = qpos[:, 16:19]
        fingertip_pos = env.get_fingertip_positions()

        quality_ok, distances = check_grasp_quality_strict(qpos, qvel, fingertip_pos, cube_pos)
        valid_mask = survived & quality_ok

        if verbose and total_attempts % 20 == 0:
            cube_z = cube_pos[:, 2]
            z_ok = (cube_z > 0.02) & (cube_z < 0.15)
            linvel_mag = jnp.linalg.norm(qvel[:, 16:19], axis=-1)
            angvel_mag = jnp.linalg.norm(qvel[:, 19:22], axis=-1)
            vel_ok = (linvel_mag < 0.2) & (angvel_mag < 1.0)
            all_fingers = jnp.all(distances < FINGER_DIST_THRESHOLD, axis=-1)

            print(f"\n  Attempt {total_attempts}:")
            print(f"    Survived physics: {jnp.sum(survived)}/{num_envs}")
            print(f"    Z height OK: {jnp.sum(z_ok & survived)}")
            print(f"    Velocity OK: {jnp.sum(vel_ok & survived)}")
            print(f"    All fingers close: {jnp.sum(all_fingers & survived)}")
            print(f"    Final valid: {jnp.sum(valid_mask)}")
            print(f"    Mean finger distances: {distances.mean(axis=0)}")

        if jnp.any(valid_mask):
            valid_indices = jnp.where(valid_mask)[0]
            grasp_states = qpos[valid_indices, :23]

            for i in range(len(valid_indices)):
                if len(collected_grasps) < num_grasps:
                    collected_grasps.append(np.array(grasp_states[i]))
                    if pbar:
                        pbar.update(1)

    if pbar:
        pbar.close()

    if len(collected_grasps) < num_grasps:
        print(f"\nWARNING: Only collected {len(collected_grasps)}/{num_grasps} grasps")
        print(f"Try increasing max_attempts or relaxing criteria")

    cache = np.stack(collected_grasps[:num_grasps], axis=0)

    if verbose:
        print(f"\nTotal grasps: {len(cache)}")
        print(f"Total attempts: {total_attempts}")
        print(f"Total envs tested: {total_envs_tested}")
        print(f"Success rate: {len(cache) / total_envs_tested * 100:.2f}%")
        dof = cache[:, :16]
        print(f"DOF std (mean across joints): {dof.std(axis=0).mean():.3f}")
        print(f"Cube Z range: [{cache[:, 18].min():.3f}, {cache[:, 18].max():.3f}]")

    return cache


def main():
    parser = argparse.ArgumentParser(description="Generate grasp cache for LEAP hand")
    parser.add_argument("--num_grasps", type=int, default=10000)
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--output", type=str, default="grasp_cache/grasp_cache.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--survival_steps", type=int, default=SURVIVAL_STEPS)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache = generate_grasp_cache(
        num_grasps=args.num_grasps,
        num_envs=args.num_envs,
        survival_steps=args.survival_steps,
        seed=args.seed,
        verbose=True
    )

    np.save(output_path, cache)
    print(f"\nSaved grasp cache to {output_path}")


if __name__ == "__main__":
    main()
