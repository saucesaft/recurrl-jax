"""
Grasp cache utilities for loading and using pre-generated successful grasps.

The grasp cache contains states with shape (N, 23):
- First 16 values: DOF positions (joint angles)
- Next 7 values: Object root state (position xyz + quaternion xyzw)
"""

import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class GraspCache:
    """
    Manages loading and sampling from a grasp cache.

    The cache stores successful grasp states that can be used to:
    1. Initialize environments with known good grasps
    2. Provide curriculum learning by starting from successful states
    3. Reduce exploration time during training
    """

    def __init__(self, cache_path: str, device: str = 'cpu'):
        """
        Load grasp cache from file.

        Args:
            cache_path: Path to .npy file containing grasp cache
            device: Device to store cache on (not used in JAX, kept for compatibility)
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        # Load cache
        self.cache = jnp.array(np.load(cache_path))
        self.cache_size = self.cache.shape[0]

        print(f"Loaded grasp cache from {cache_path}")
        print(f"Cache size: {self.cache_size}")
        print(f"Cache shape: {self.cache.shape}")

        # Validate shape
        if self.cache.shape[1] != 23:
            raise ValueError(
                f"Invalid cache shape: {self.cache.shape}. "
                f"Expected (N, 23) for [dof_pos(16) + obj_pos(3) + obj_quat(4)]"
            )

    def sample(self, key: jax.Array, num_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample random grasp states from the cache.

        Args:
            key: JAX random key
            num_samples: Number of samples to draw

        Returns:
            dof_positions: (num_samples, 16) - Joint positions
            object_states: (num_samples, 7) - Object position and quaternion
        """
        # Sample random indices with replacement
        indices = jnr.randint(key, shape=(num_samples,), minval=0, maxval=self.cache_size)

        # Extract samples
        samples = self.cache[indices]

        # Split into DOF positions and object states
        dof_positions = samples[:, :16]
        object_states = samples[:, 16:23]  # [pos(3), quat(4)]

        return dof_positions, object_states

    def get_dof_positions(self, indices: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Get DOF positions from cache.

        Args:
            indices: Optional indices to retrieve. If None, returns all.

        Returns:
            dof_positions: (N, 16) - Joint positions
        """
        if indices is None:
            return self.cache[:, :16]
        return self.cache[indices, :16]

    def get_object_states(self, indices: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Get object states from cache.

        Args:
            indices: Optional indices to retrieve. If None, returns all.

        Returns:
            object_states: (N, 7) - Object position and quaternion
        """
        if indices is None:
            return self.cache[:, 16:23]
        return self.cache[indices, 16:23]

    def apply_to_env(
        self,
        key: jax.Array,
        mjx_data_batch,
        env_mask: Optional[jnp.ndarray] = None
    ):
        """
        Apply cached grasp states to MJX environment data.

        Args:
            key: JAX random key
            mjx_data_batch: MJX data batch to modify
            env_mask: Optional boolean mask indicating which envs to update.
                      If None, updates all environments.

        Returns:
            Updated mjx_data_batch with grasp states applied
        """
        num_envs = mjx_data_batch.qpos.shape[0]

        if env_mask is None:
            env_mask = jnp.ones(num_envs, dtype=bool)

        num_to_sample = jnp.sum(env_mask).item()

        # Sample from cache
        dof_positions, object_states = self.sample(key, num_to_sample)

        # Create new qpos
        new_qpos = mjx_data_batch.qpos.copy()

        # Update DOF positions for masked environments
        new_qpos = jnp.where(
            env_mask[:, None],
            new_qpos.at[:, :16].set(dof_positions),
            new_qpos
        )

        # Update object position and quaternion for masked environments
        # Object position (indices 16:19)
        new_qpos = jnp.where(
            env_mask[:, None],
            new_qpos.at[:, 16:19].set(object_states[:, :3]),
            new_qpos
        )

        # Object quaternion (indices 19:23)
        new_qpos = jnp.where(
            env_mask[:, None],
            new_qpos.at[:, 19:23].set(object_states[:, 3:7]),
            new_qpos
        )

        # Update mjx_data_batch
        mjx_data_batch = mjx_data_batch.replace(qpos=new_qpos)

        return mjx_data_batch

    def __len__(self) -> int:
        """Return cache size."""
        return self.cache_size

    def __repr__(self) -> str:
        return f"GraspCache(size={self.cache_size}, shape={self.cache.shape})"


def load_grasp_cache(cache_path: str) -> GraspCache:
    """
    Convenience function to load a grasp cache.

    Args:
        cache_path: Path to .npy cache file

    Returns:
        GraspCache instance
    """
    return GraspCache(cache_path)


# Example usage in training loop:
"""
# Load cache
grasp_cache = load_grasp_cache("cache/leap_hand_cube_grasp_cache.npy")

# During environment reset, optionally use cached grasps
if use_cached_grasp:
    key, subkey = jax.random.split(key)
    mjx_data_batch = grasp_cache.apply_to_env(
        subkey,
        mjx_data_batch,
        env_mask=reset_mask  # Only update environments that are resetting
    )
"""
