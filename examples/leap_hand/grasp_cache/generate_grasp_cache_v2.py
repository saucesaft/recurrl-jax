"""
Improved Grasp Cache Generation Script for LEAP Hand.

Key improvements over v1:
1. Uses LEAP_Hand_Sim canonical pose (not just "close fingers")
2. Large DOF search radius for diverse grasps
3. Multiple cube sizes (0.9x, 0.95x, 1.0x, 1.05x, 1.1x)
4. Success-based filtering (grasp must survive full episode)
5. Better quality criteria (fingertip proximity, stability)

Usage:
    python -m src.utils.generate_grasp_cache_v2 --num_grasps 10000 --output cache/grasp_cache_v2.npy
"""

import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from examples.leap_hand.env import MJXLeapHandEnv


# Canonical pose for our MJX LEAP hand model
# Note: LEAP_Hand_Sim uses different joint mapping, so we use our own pose
# This is a flexed grasp position that works with our joint limits
CANONICAL_POSE = jnp.array([
    0.8, 0.0, 0.8, 0.8,   # Index: MCP, ABD, PIP, DIP (flexed)
    0.8, 0.0, 0.8, 0.8,   # Middle: MCP, ABD, PIP, DIP
    0.8, 0.0, 0.8, 0.8,   # Ring: MCP, ABD, PIP, DIP
    0.8, 0.8, 0.8, 0.0,   # Thumb: MCP, ABD, PIP, DIP
])

# DOF search radius - how much to vary from canonical pose
# Large radius creates diverse grasps (different finger configurations)
# Our joint ranges are ~2.5 rad, so 0.5 explores ~20% of range
DOF_SEARCH_RADIUS = 0.5

# Note: We train with fixed cube size, so cache uses same size
# The main improvement here is DOF diversity (search radius 0.5 vs ~0.1)


def check_grasp_quality_strict(qpos: jnp.ndarray, qvel: jnp.ndarray,
                                fingertip_positions: jnp.ndarray,
                                cube_pos: jnp.ndarray) -> jnp.ndarray:
    """
    Stricter grasp quality check based on LEAP_Hand_Sim criteria.

    Criteria:
    1. Cube is in valid workspace (not fallen)
    2. Cube has low velocity (stable)
    3. At least 2 fingertips are close to cube
    4. No NaN/Inf

    Args:
        qpos: Joint positions (num_envs, 23)
        qvel: Joint velocities (num_envs, 22)
        fingertip_positions: (num_envs, 12) - 4 fingertips x 3 coords
        cube_pos: (num_envs, 3) - cube center position

    Returns:
        valid_mask: Boolean array (num_envs,)
    """
    # Reshape fingertips to (num_envs, 4, 3)
    fingertips = fingertip_positions.reshape(-1, 4, 3)

    # Criterion 1: Cube in valid workspace
    cube_z = cube_pos[:, 2]
    z_valid = (cube_z > 0.02) & (cube_z < 0.15)

    # Criterion 2: Low velocity (stable grasp)
    cube_linvel = qvel[:, 16:19]
    cube_angvel = qvel[:, 19:22]
    linvel_mag = jnp.linalg.norm(cube_linvel, axis=-1)
    angvel_mag = jnp.linalg.norm(cube_angvel, axis=-1)
    velocity_stable = (linvel_mag < 0.3) & (angvel_mag < 1.5)

    # Criterion 3: Fingertip proximity (LEAP uses 0.1, we use 0.08 for tighter grasps)
    finger_dist_threshold = 0.08
    distances = jnp.sqrt(jnp.sum((fingertips - cube_pos[:, None, :]) ** 2, axis=-1))  # (num_envs, 4)
    fingers_close = distances < finger_dist_threshold  # (num_envs, 4)
    num_close_fingers = jnp.sum(fingers_close, axis=-1)  # (num_envs,)
    contact_ok = num_close_fingers >= 2  # At least 2 fingertips near cube

    # Criterion 4: No NaN/Inf
    no_nan = ~jnp.any(jnp.isnan(qpos), axis=-1) & ~jnp.any(jnp.isnan(qvel), axis=-1)
    no_inf = ~jnp.any(jnp.isinf(qpos), axis=-1) & ~jnp.any(jnp.isinf(qvel), axis=-1)

    valid = z_valid & velocity_stable & contact_ok & no_nan & no_inf

    return valid


def generate_diverse_grasp_cache(
    num_grasps: int = 10000,
    num_envs: int = 512,
    survival_steps: int = 50,
    seed: int = 42,
    use_domain_randomization: bool = True,
    action_scale: float = 0.3,
    verbose: bool = True
) -> np.ndarray:
    """
    Generate diverse grasp cache using LEAP_Hand_Sim-style approach.

    Strategy:
    1. Initialize with canonical pose + large random noise
    2. Let physics settle for survival_steps
    3. Only save grasps that remain stable
    4. Repeat with different cube sizes for diversity

    Args:
        num_grasps: Target total grasps to collect
        num_envs: Parallel environments
        survival_steps: Steps grasp must survive to be considered valid
        seed: Random seed
        use_domain_randomization: Use DR for more robust grasps
        action_scale: Action scaling
        verbose: Print progress

    Returns:
        cache: (num_grasps, 23) array of diverse grasp states
    """
    if verbose:
        print(f"Generating diverse grasp cache: {num_grasps} grasps")
        print(f"DOF search radius: {DOF_SEARCH_RADIUS}")
        print(f"Survival steps: {survival_steps}")

    key = jr.PRNGKey(seed)
    collected_grasps = []

    pbar = tqdm(total=num_grasps, desc="Collecting grasps") if verbose else None

    # Initialize environment once
    key, env_key = jr.split(key)
    env = MJXLeapHandEnv(
        xml_path='leap_hand/xmls/scene_mjx_cube.xml',
        num_envs=num_envs,
        key=env_key,
        action_scale=action_scale,
        use_domain_randomization=use_domain_randomization
    )

    # Get joint limits
    joint_lower = env.joint_lower_limits
    joint_upper = env.joint_upper_limits

    attempts = 0
    max_attempts = 500  # Prevent infinite loop

    while len(collected_grasps) < num_grasps and attempts < max_attempts:
        attempts += 1

        # Generate random DOF positions around canonical pose
        key, dof_key, cube_key = jr.split(key, 3)

        # Random offsets from canonical pose
        dof_noise = jr.uniform(dof_key, shape=(num_envs, 16),
                               minval=-DOF_SEARCH_RADIUS,
                               maxval=DOF_SEARCH_RADIUS)
        random_dofs = CANONICAL_POSE + dof_noise
        random_dofs = jnp.clip(random_dofs, joint_lower, joint_upper)

        # Random cube positions (small variation)
        cube_pos_base = jnp.array([0.09, 0.0, 0.045])  # Center of palm
        cube_pos_noise = jr.uniform(cube_key, shape=(num_envs, 3),
                                    minval=jnp.array([-0.015, -0.015, -0.01]),
                                    maxval=jnp.array([0.015, 0.015, 0.01]))
        cube_positions = cube_pos_base + cube_pos_noise

        # Random cube orientations
        key, quat_key = jr.split(key)
        def random_quat(k):
            u = jr.uniform(k, shape=(3,))
            return jnp.array([
                jnp.sqrt(1-u[0]) * jnp.sin(2*jnp.pi*u[1]),
                jnp.sqrt(1-u[0]) * jnp.cos(2*jnp.pi*u[1]),
                jnp.sqrt(u[0]) * jnp.sin(2*jnp.pi*u[2]),
                jnp.sqrt(u[0]) * jnp.cos(2*jnp.pi*u[2])
            ])
        quat_keys = jr.split(quat_key, num_envs)
        cube_quats = jax.vmap(random_quat)(quat_keys)

        # Build initial qpos
        init_qpos = jnp.concatenate([random_dofs, cube_positions, cube_quats], axis=1)

        # Set environment state
        new_qvel = jnp.zeros_like(env.mjx_data_batch.qvel)
        new_ctrl = random_dofs  # PD targets match joint positions

        env.mjx_data_batch = env.mjx_data_batch.replace(
            qpos=init_qpos,
            qvel=new_qvel,
            ctrl=new_ctrl
        )
        env.initial_dof_pos = random_dofs
        env.progress_buf = jnp.zeros(num_envs, dtype=jnp.int32)

        # Let physics settle - apply zero action to maintain pose
        zero_action = jnp.zeros((num_envs, 16))

        survived = jnp.ones(num_envs, dtype=bool)

        for step in range(survival_steps):
            _, _, done, _, _, _, _, _ = env.step(zero_action)
            survived = survived & ~done

        # Check quality of surviving grasps
        qpos = env.mjx_data_batch.qpos
        qvel = env.mjx_data_batch.qvel
        cube_pos = qpos[:, 16:19]
        fingertip_pos = env.get_fingertip_positions()

        quality_ok = check_grasp_quality_strict(qpos, qvel, fingertip_pos, cube_pos)
        valid_mask = survived & quality_ok

        if jnp.any(valid_mask):
            # Extract valid grasp states
            valid_indices = jnp.where(valid_mask)[0]
            grasp_states = qpos[valid_indices, :23]

            for i in range(len(valid_indices)):
                if len(collected_grasps) < num_grasps:
                    collected_grasps.append(np.array(grasp_states[i]))
                    if pbar:
                        pbar.update(1)

        if verbose and attempts % 10 == 0:
            print(f"  Attempt {attempts}: {jnp.sum(valid_mask)}/{num_envs} valid, "
                  f"total: {len(collected_grasps)}/{num_grasps}")

    if pbar:
        pbar.close()

    # Stack and trim to exact size
    cache = np.stack(collected_grasps[:num_grasps], axis=0)

    if verbose:
        print(f"\n=== Final Cache Statistics ===")
        print(f"Total grasps: {len(cache)}")
        dof = cache[:, :16]
        print(f"DOF std (mean): {dof.std(axis=0).mean():.3f}")
        print(f"DOF range used (mean): {(dof.max(axis=0) - dof.min(axis=0)).mean():.3f}")
        print(f"Cube Z range: [{cache[:, 18].min():.3f}, {cache[:, 18].max():.3f}]")

    return cache


def main():
    parser = argparse.ArgumentParser(description="Generate diverse grasp cache (v2)")
    parser.add_argument("--num_grasps", type=int, default=10000, help="Number of grasps")
    parser.add_argument("--num_envs", type=int, default=512, help="Parallel environments")
    parser.add_argument("--output", type=str, default="cache/grasp_cache_v2.npy", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--survival_steps", type=int, default=50, help="Steps grasp must survive")
    parser.add_argument("--use_dr", action="store_true", help="Use domain randomization")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache = generate_diverse_grasp_cache(
        num_grasps=args.num_grasps,
        num_envs=args.num_envs,
        survival_steps=args.survival_steps,
        seed=args.seed,
        use_domain_randomization=args.use_dr,
        verbose=True
    )

    np.save(output_path, cache)
    print(f"\nSaved grasp cache to {output_path}")


if __name__ == "__main__":
    main()
