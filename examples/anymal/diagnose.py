"""
Per-step trajectory diagnostic for DiffAnymal.
Runs N steps without auto-reset, logs body height, joint velocities,
termination flags, and reward components so we can see exactly when/why
the robot dies.

Usage:
    python -m examples.anymal.diagnose
"""
import jax
import jax.numpy as jp
import numpy as np

from examples.anymal.anymal_env import DiffAnymal


N_STEPS = 60
N_SURVEY = 64   # seeds for episode-length survey
ACTION_DIM = 12


def run_trajectory(env, rng, action_fn, label):
    step_jit = jax.jit(env.step)

    rng, rng_reset = jax.random.split(rng)
    state = env.reset(rng_reset)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'step':>4}  {'z':>6}  {'|qd|_mean':>9}  {'|qd|_max':>8}  "
          f"{'|ctrl|':>6}  {'done':>4}  {'t_ht':>5}  {'t_fl':>5}  "
          f"{'t_inv':>5}  {'reward':>7}")
    print("-" * 80)

    done_at = None
    for t in range(N_STEPS):
        rng, rng_act = jax.random.split(rng)
        action = action_fn(rng_act, state, t)

        state = step_jit(state, action)

        data = state.pipeline_state
        z = float(data.qpos[2])
        qd = np.array(data.qvel[6:])
        qd_mean = float(np.mean(np.abs(qd)))
        qd_max  = float(np.max(np.abs(qd)))
        ctrl    = np.array(data.ctrl)
        ctrl_mean = float(np.mean(np.abs(ctrl)))

        done      = float(state.done)
        t_ht      = float(state.metrics.get('term_height',  0.0))
        t_fl      = float(state.metrics.get('term_flip',    0.0))
        t_inv     = float(state.metrics.get('term_invalid', 0.0))
        reward    = float(state.reward)

        # marker on first terminal step
        marker = " <-- DEAD" if done > 0.5 and done_at is None else ""
        if done > 0.5 and done_at is None:
            done_at = t + 1

        print(f"{t+1:>4}  {z:>6.3f}  {qd_mean:>9.3f}  {qd_max:>8.3f}  "
              f"{ctrl_mean:>6.3f}  {done:>4.0f}  {t_ht:>5.0f}  {t_fl:>5.0f}  "
              f"{t_inv:>5.0f}  {reward:>7.4f}{marker}")

    print()
    print("  Reward components at last step:")
    for k, v in state.metrics.items():
        if k.startswith('term_'):
            continue
        print(f"    {k:<24s} {float(v):>10.6f}")
    if done_at:
        print(f"\n  First terminal step: {done_at}")
    else:
        print(f"\n  Survived all {N_STEPS} steps.")


def run_trajectory_quiet_reset(env, action_fn, label):
    """Same as run_trajectory but starts from exact standing keyframe with zero velocity."""
    step_jit = jax.jit(env.step)

    # Get a full state for the info dict
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    # Replace pipeline_state with noiseless initialization
    qpos = jp.array(env._init_q)   # exact standing keyframe
    qvel = jp.zeros(18)
    data = env.pipeline_init(qpos, qvel)

    x, xd = env._pos_vel(data)
    obs = env._get_obs(qpos, qvel, x, xd, state.info)

    state = state.replace(pipeline_state=data, obs=obs)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'step':>4}  {'z':>6}  {'|qd|_mean':>9}  {'|qd|_max':>8}  "
          f"{'|ctrl|':>6}  {'done':>4}  {'t_ht':>5}  {'t_fl':>5}  "
          f"{'t_inv':>5}  {'reward':>7}")
    print("-" * 80)

    done_at = None
    rng = jax.random.PRNGKey(42)
    for t in range(N_STEPS):
        rng, rng_act = jax.random.split(rng)
        action = action_fn(rng_act, state, t)

        state = step_jit(state, action)

        data = state.pipeline_state
        z = float(data.qpos[2])
        qd = np.array(data.qvel[6:])
        qd_mean = float(np.mean(np.abs(qd)))
        qd_max  = float(np.max(np.abs(qd)))
        ctrl    = np.array(data.ctrl)
        ctrl_mean = float(np.mean(np.abs(ctrl)))

        done      = float(state.done)
        t_ht      = float(state.metrics.get('term_height',  0.0))
        t_fl      = float(state.metrics.get('term_flip',    0.0))
        t_inv     = float(state.metrics.get('term_invalid', 0.0))
        reward    = float(state.reward)

        marker = " <-- DEAD" if done > 0.5 and done_at is None else ""
        if done > 0.5 and done_at is None:
            done_at = t + 1

        print(f"{t+1:>4}  {z:>6.3f}  {qd_mean:>9.3f}  {qd_max:>8.3f}  "
              f"{ctrl_mean:>6.3f}  {done:>4.0f}  {t_ht:>5.0f}  {t_fl:>5.0f}  "
              f"{t_inv:>5.0f}  {reward:>7.4f}{marker}")

    if done_at:
        print(f"\n  First terminal step: {done_at}")
    else:
        print(f"\n  Survived all {N_STEPS} steps.")


def survey_episode_lengths(env, label, n=N_SURVEY, action=None):
    """Run n envs in parallel with zero action, report episode-length stats."""
    if action is None:
        action = jp.zeros((n, ACTION_DIM))

    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step  = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(0), n)
    states = vmap_reset(keys)

    first_done = jp.full((n,), N_STEPS, dtype=jp.int32)
    for t in range(N_STEPS):
        states = vmap_step(states, action)
        just_done = (states.done > 0.5) & (first_done == N_STEPS)
        first_done = jp.where(just_done, jp.full((n,), t + 1, dtype=jp.int32), first_done)

    ep = np.array(first_done)
    print(f"\n  {label}")
    print(f"  n={n}  min={ep.min()}  mean={ep.mean():.1f}  "
          f"median={np.median(ep):.0f}  max={ep.max()}  "
          f"survived_all={int((ep == N_STEPS).sum())}")


def main():
    rng = jax.random.PRNGKey(42)
    env = DiffAnymal(
        use_domain_randomization=False,
        cmd_scale=0.0,
        action_scale=0.25,
    )

    print(f"\nDiffAnymal dt={env.dt:.4f}s  action_scale={env.action_scale}")
    print(f"Physics steps/ctrl: {env._physics_steps_per_control_step}")

    print(f"\n{'='*60}")
    print(f"  EPISODE LENGTH SURVEY ({N_SURVEY} seeds, zero action, max {N_STEPS} steps)")
    print(f"{'='*60}")
    survey_episode_lengths(env, "noisy reset (current config)")

    # --- Zero action, noisy reset (current training setup) ---
    run_trajectory(env, rng, lambda rng, s, t: jp.zeros(ACTION_DIM), "ZERO ACTION (noisy reset)")

    # --- Zero action, quiet reset (tests physics stability without init noise) ---
    run_trajectory_quiet_reset(env, lambda rng, s, t: jp.zeros(ACTION_DIM), "ZERO ACTION (quiet reset)")

    # --- Small random (std=0.1) — simulates early training ---
    run_trajectory(
        env, rng,
        lambda rng, s, t: jax.random.normal(rng, (ACTION_DIM,)) * 0.1,
        "RANDOM ACTION (std=0.1)",
    )

    # --- Larger random (std=0.5) — simulates clipped uniform policy ---
    run_trajectory(
        env, rng,
        lambda rng, s, t: jax.random.normal(rng, (ACTION_DIM,)) * 0.5,
        "RANDOM ACTION (std=0.5)",
    )

    # --- Sustained large action (worst case) ---
    run_trajectory(
        env, rng,
        lambda rng, s, t: jp.ones(ACTION_DIM),
        "CONSTANT ACTION +1 (all joints max)",
    )


if __name__ == '__main__':
    main()
