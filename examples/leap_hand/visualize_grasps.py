"""
Visualize grasps from a cache file using native MuJoCo renderer.
Press Enter to cycle through random grasps, 'q' to quit.

Usage:
    python visualize_grasps.py --cache grasp_cache/grasp_cache.npy
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


def load_grasp_cache(cache_path: str) -> np.ndarray:
    """Load grasp cache from file."""
    cache = np.load(cache_path)
    print(f"Loaded {len(cache)} grasps from {cache_path}")
    print(f"Grasp shape: {cache.shape}")
    return cache


def visualize_grasps(cache_path: str, xml_path: str):
    """Visualize grasps from cache using MuJoCo viewer."""

    # Load cache
    cache = load_grasp_cache(cache_path)
    num_grasps = len(cache)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"\nModel has {model.nq} qpos dimensions")
    print(f"Grasp cache has {cache.shape[1]} dimensions per grasp")

    # Verify dimensions match
    if cache.shape[1] != model.nq:
        print(f"WARNING: Cache dimension ({cache.shape[1]}) != model nq ({model.nq})")
        print("Will use min of the two.")

    grasp_idx = 0

    def set_grasp(idx):
        """Set model state to grasp at index."""
        grasp = cache[idx]
        # Set qpos (handle dimension mismatch gracefully)
        dim = min(len(grasp), model.nq)
        data.qpos[:dim] = grasp[:dim]
        data.qvel[:] = 0  # Zero velocity
        mujoco.mj_forward(model, data)
        return idx

    print("\n" + "="*50)
    print("GRASP VISUALIZER")
    print("="*50)
    print("Controls:")
    print("  Enter  - Next random grasp")
    print("  n      - Next sequential grasp")
    print("  p      - Previous grasp")
    print("  r      - Random grasp")
    print("  q      - Quit")
    print("="*50)

    # Set initial grasp
    grasp_idx = np.random.randint(num_grasps)
    set_grasp(grasp_idx)
    print(f"\nShowing grasp {grasp_idx}/{num_grasps}")

    # Print grasp info
    grasp = cache[grasp_idx]
    print(f"  DOF positions: {grasp[:16]}")
    print(f"  Cube pos: {grasp[16:19]}")
    print(f"  Cube quat: {grasp[19:23]}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Check for keyboard input (non-blocking)
            try:
                import select
                import sys

                # Check if input is available (Unix-like systems)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    key = sys.stdin.readline().strip().lower()

                    if key == 'q':
                        print("Quitting...")
                        break
                    elif key == 'n':
                        grasp_idx = (grasp_idx + 1) % num_grasps
                    elif key == 'p':
                        grasp_idx = (grasp_idx - 1) % num_grasps
                    elif key == 'r' or key == '':  # Enter key
                        grasp_idx = np.random.randint(num_grasps)

                    set_grasp(grasp_idx)
                    print(f"\nShowing grasp {grasp_idx}/{num_grasps}")
                    grasp = cache[grasp_idx]
                    print(f"  Cube pos: [{grasp[16]:.3f}, {grasp[17]:.3f}, {grasp[18]:.3f}]")

            except (ImportError, OSError):
                # Fallback for Windows or if select doesn't work
                pass

            # Step simulation to keep viewer responsive
            set_grasp(grasp_idx)
            mujoco.mj_step(model, data)
            viewer.sync()


def main():
    parser = argparse.ArgumentParser(description="Visualize grasp cache")
    parser.add_argument("--cache", type=str, default="./grasp_cache/grasp_cache.npy",
                        help="Path to grasp cache .npy file")
    parser.add_argument("--xml", type=str, default="./leap_hand/xmls/scene_mjx_cube.xml",
                        help="Path to MuJoCo XML file")
    args = parser.parse_args()

    cache_path = args.cache
    xml_path = args.xml

    visualize_grasps(str(cache_path), str(xml_path))


if __name__ == "__main__":
    main()
