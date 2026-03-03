"""
Grasp Cache Generation Script for LEAP Hand.

Generates a cache of successful grasp states by running a simple grasping policy.
The cache is used to initialize training episodes with known good grasps.

Usage:
    python -m src.utils.generate_grasp_cache --num_grasps 10000 --output cache/grasp_cache.npy
"""

import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Import environment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from examples.leap_hand.env import MJXLeapHandEnv


def generate_closing_action():
    """Generate action that closes all fingers toward palm."""
    # Action values that tend to close fingers (positive = flex for most joints)
    # Based on the default pose: [0.8, 0, 0.8, 0.8] pattern per finger
    # We want to increase flexion (close fingers)
    closing_action = jnp.array([
        0.5, 0.0, 0.5, 0.5,   # Index: flex MCP, neutral abd, flex PIP, flex DIP
        0.5, 0.0, 0.5, 0.5,   # Middle
        0.5, 0.0, 0.5, 0.5,   # Ring
        0.3, 0.3, 0.3, 0.0,   # Thumb: moderate flexion
    ])
    return closing_action


def check_grasp_quality(qpos: jnp.ndarray, qvel: jnp.ndarray) -> jnp.ndarray:
    """
    Check if current state represents a valid grasp.

    Criteria:
    1. Cube is in valid workspace (not fallen, not exploded)
    2. Cube has low velocity (stable grasp)
    3. Hand joints are in reasonable range

    Args:
        qpos: Joint positions (num_envs, 23) - [hand(16), cube_pos(3), cube_quat(4)]
        qvel: Joint velocities (num_envs, 22) - [hand(16), cube_linvel(3), cube_angvel(3)]

    Returns:
        valid_mask: Boolean array (num_envs,) indicating valid grasps
    """
    # Extract cube position
    cube_pos = qpos[:, 16:19]
    cube_z = cube_pos[:, 2]

    # Extract cube velocities
    cube_linvel = qvel[:, 16:19]
    cube_angvel = qvel[:, 19:22]

    # Criterion 1: Cube is in workspace
    # z should be positive (not fallen) and reasonable (not exploded)
    z_valid = (cube_z > 0.02) & (cube_z < 0.15)

    # xy should be near hand center
    xy_valid = (jnp.abs(cube_pos[:, 0] - 0.1) < 0.05) & (jnp.abs(cube_pos[:, 1]) < 0.05)

    position_valid = z_valid & xy_valid

    # Criterion 2: Cube is stable (low velocity)
    linvel_magnitude = jnp.linalg.norm(cube_linvel, axis=-1)
    angvel_magnitude = jnp.linalg.norm(cube_angvel, axis=-1)

    velocity_stable = (linvel_magnitude < 0.5) & (angvel_magnitude < 2.0)

    # Criterion 3: No NaN/Inf
    no_nan = ~jnp.any(jnp.isnan(qpos), axis=-1) & ~jnp.any(jnp.isnan(qvel), axis=-1)
    no_inf = ~jnp.any(jnp.isinf(qpos), axis=-1) & ~jnp.any(jnp.isinf(qvel), axis=-1)

    valid = position_valid & velocity_stable & no_nan & no_inf

    return valid


def extract_grasp_state(qpos: jnp.ndarray) -> jnp.ndarray:
    """
    Extract grasp state from qpos for caching.

    Returns:
        state: (num_envs, 23) - [dof_pos(16), cube_pos(3), cube_quat(4)]
    """
    # qpos is already in the right format: [hand(16), cube_pos(3), cube_quat(4)]
    return qpos[:, :23]


def generate_grasp_cache(
    num_grasps: int = 10000,
    num_envs: int = 1024,
    max_steps_per_episode: int = 100,
    stabilization_steps: int = 20,
    seed: int = 42,
    use_domain_randomization: bool = False,
    action_scale: float = 0.3,
    verbose: bool = True
) -> np.ndarray:
    """
    Generate grasp cache by running grasping policy.

    Strategy:
    1. Reset environments with random cube positions
    2. Apply closing actions to grasp the cube
    3. After stabilization, check if grasp is valid
    4. Collect valid grasp states
    5. Repeat until we have enough grasps

    Args:
        num_grasps: Target number of grasps to collect
        num_envs: Number of parallel environments
        max_steps_per_episode: Maximum steps before reset
        stabilization_steps: Steps to wait before checking grasp quality
        seed: Random seed
        use_domain_randomization: Whether to use DR (for diverse grasps)
        action_scale: Action scaling factor
        verbose: Print progress

    Returns:
        cache: (num_grasps, 23) array of grasp states
    """
    if verbose:
        print(f"Generating grasp cache with {num_grasps} grasps...")
        print(f"Using {num_envs} parallel environments")

    # Initialize environment
    key = jr.PRNGKey(seed)
    key, env_key = jr.split(key)

    env = MJXLeapHandEnv(
        xml_path='leap_hand/xmls/scene_mjx_cube.xml',
        num_envs=num_envs,
        key=env_key,
        action_scale=action_scale,
        use_domain_randomization=use_domain_randomization
    )

    # Reset environment
    env.reset()

    # Storage for collected grasps
    collected_grasps = []

    # Get closing action
    closing_action = generate_closing_action()
    closing_actions = jnp.tile(closing_action, (num_envs, 1))

    # Progress bar
    pbar = tqdm(total=num_grasps, desc="Collecting grasps") if verbose else None

    episode_step = 0
    total_steps = 0
    max_total_steps = num_grasps * max_steps_per_episode  # Safety limit

    while len(collected_grasps) < num_grasps and total_steps < max_total_steps:
        # Add some noise to closing action for diversity
        key, noise_key = jr.split(key)
        action_noise = jr.uniform(noise_key, shape=(num_envs, 16), minval=-0.2, maxval=0.2)
        noisy_actions = jnp.clip(closing_actions + action_noise, -1.0, 1.0)

        # Step environment (returns 8 values: raw_state, reward, done, termination, info, new_data, new_progress, new_model)
        raw_state, reward, done, termination, info, _, _, _ = env.step(noisy_actions)
        episode_step += 1
        total_steps += 1

        # After stabilization period, check for valid grasps
        if episode_step >= stabilization_steps:
            qpos = env.mjx_data_batch.qpos
            qvel = env.mjx_data_batch.qvel

            valid_mask = check_grasp_quality(qpos, qvel)

            if jnp.any(valid_mask):
                # Extract valid grasp states
                grasp_states = extract_grasp_state(qpos)
                valid_grasps = grasp_states[valid_mask]

                # Add to collection (convert to numpy for storage)
                for grasp in valid_grasps:
                    if len(collected_grasps) < num_grasps:
                        collected_grasps.append(np.array(grasp))
                        if pbar:
                            pbar.update(1)

        # Reset environments that are done or at max steps
        if episode_step >= max_steps_per_episode or jnp.any(done):
            env.reset()
            episode_step = 0

    if pbar:
        pbar.close()

    if len(collected_grasps) < num_grasps:
        print(f"Warning: Only collected {len(collected_grasps)} grasps (target: {num_grasps})")

    # Stack into array
    cache = np.stack(collected_grasps[:num_grasps], axis=0)

    if verbose:
        print(f"Collected {len(cache)} grasps")
        print(f"Cache shape: {cache.shape}")
        print(f"DOF position range: [{cache[:, :16].min():.3f}, {cache[:, :16].max():.3f}]")
        print(f"Cube Z range: [{cache[:, 18].min():.3f}, {cache[:, 18].max():.3f}]")

    return cache


def main():
    parser = argparse.ArgumentParser(description="Generate grasp cache for LEAP Hand")
    parser.add_argument("--num_grasps", type=int, default=10000, help="Number of grasps to collect")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--output", type=str, default="cache/grasp_cache.npy", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_dr", action="store_true", help="Use domain randomization")
    parser.add_argument("--action_scale", type=float, default=0.3, help="Action scale")
    parser.add_argument("--stabilization_steps", type=int, default=30, help="Steps before checking grasp")
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate cache
    cache = generate_grasp_cache(
        num_grasps=args.num_grasps,
        num_envs=args.num_envs,
        stabilization_steps=args.stabilization_steps,
        seed=args.seed,
        use_domain_randomization=args.use_dr,
        action_scale=args.action_scale,
        verbose=True
    )

    # Save cache
    np.save(output_path, cache)
    print(f"Saved grasp cache to {output_path}")


if __name__ == "__main__":
    main()
