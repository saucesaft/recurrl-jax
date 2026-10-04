"""Load trained ANYmal checkpoint and run interactive MuJoCo viewer."""
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import mujoco
import mujoco.viewer
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions

from recurrl_jax.models.actor_critic import ActorCriticModel
from recurrl_jax.model_fns.repr_fns import flatten_repr_model
from recurrl_jax.model_fns.seq_fns import seq_model_mlp
from recurrl_jax.model_fns.achead_fns import actor_model_continuous, critic_model

from examples.anymal.anymal_env import DiffAnymal


OBS_DIM = 49
ACTION_DIM = 12
MLP_DIMS = [512, 256, 128]
D_ACTOR = 512
D_CRITIC = 512


def build_model():
    repr_fn = flatten_repr_model()
    seq_fn, seq_init = seq_model_mlp(name="mlp", dims=MLP_DIMS)
    actor_fn = actor_model_continuous(D_ACTOR, ACTION_DIM)
    critic_fn = critic_model(D_CRITIC)

    ac = nn.vmap(
        ActorCriticModel,
        variable_axes={"params": None},
        split_rngs={"params": False},
    )(repr_fn, seq_fn, actor_fn, critic_fn, use_asymmetric_obs=True, policy_obs_dim=OBS_DIM)

    return ac, seq_init


def load_checkpoint(checkpoint_dir: str, seq_init):
    h0 = jax.tree.map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), 1, axis=0),
        seq_init(),
    )

    ckpt_path = Path(checkpoint_dir).resolve()
    manager = CheckpointManager(
        ckpt_path,
        PyTreeCheckpointer(),
        CheckpointManagerOptions(create=False),
    )
    step = manager.latest_step()
    print(f"Loading checkpoint at step {step:,} from {ckpt_path}")

    restored = manager.restore(step)
    params = jax.tree.map(jnp.asarray, restored["params"])

    if "obs_mean" in restored:
        obs_mean = jnp.asarray(restored["obs_mean"])
        obs_var  = jnp.asarray(restored["obs_var"])
        print("Loaded obs normalisation stats from checkpoint.")
    else:
        print("WARNING: checkpoint has no obs_mean/obs_var — using identity normalisation.")
        print("         Policy may behave poorly. Re-save checkpoint after training to include RMS.")
        obs_mean = jnp.zeros(OBS_DIM)
        obs_var  = jnp.ones(OBS_DIM)

    return params, h0, obs_mean, obs_var


def run(checkpoint_dir="checkpoints", episodes=10, cmd_vx=0.5, cmd_vy=0.0, cmd_yaw=0.0,
        action_scale=0.25, fps=50, env_variant="anymal"):

    ac, seq_init = build_model()
    params, h0, obs_mean, obs_var = load_checkpoint(checkpoint_dir, seq_init)

    env = DiffAnymal(
        model_variant=env_variant,
        use_domain_randomization=False,
        action_scale=action_scale,
        cmd_scale=1.0,
        cmd_min_speed=0.0,
        cmd_scale_curriculum_steps=0,  # bypass curriculum — full commands always
    )

    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)

    epsilon = 1e-5

    @jax.jit
    def policy_step(params, obs_raw, term, h, obs_mean, obs_var):
        obs_norm = (obs_raw - obs_mean) / jnp.sqrt(obs_var + epsilon)
        obs_b  = obs_norm[None, None, :]
        term_b = term[None, None]
        (mean, _), _, h_new = ac.apply(
            params, obs_b, term_b, h,
            rngs={"random": jax.random.PRNGKey(0)},
        )
        return mean[0, 0], h_new

    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step  = jax.jit(jax.vmap(env.step))

    cmd = jnp.array([cmd_vx, cmd_vy, cmd_yaw], dtype=jnp.float32)
    print(f"Command: vx={cmd_vx:.2f}  vy={cmd_vy:.2f}  yaw={cmd_yaw:.2f}")
    print("Controls: Q/E yaw, WASD velocity, R reset, ESC quit")

    dt = 1.0 / fps

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for ep in range(episodes):
            key = jax.random.PRNGKey(ep)
            keys = jnp.expand_dims(key, 0)
            state = vmap_reset(keys)

            # override command to desired value
            state = state.replace(info={**state.info, 'vel_cmd': jnp.expand_dims(cmd, 0)})

            h = h0
            term = jnp.array([False])
            ep_return = 0.0
            t0 = time.time()

            for step_i in range(2000):
                obs_raw = state.obs[0]  # (49,)
                action, h = policy_step(params, obs_raw, term, h, obs_mean, obs_var)
                actions = jnp.expand_dims(action, 0)  # (1, 12)

                next_state = vmap_step(state, actions)
                done = bool(next_state.done[0])
                ep_return += float(next_state.reward[0])
                term = jnp.array([done])

                # keep command fixed (env may resample — override it)
                next_state = next_state.replace(
                    info={**next_state.info, 'vel_cmd': jnp.expand_dims(cmd, 0)}
                )

                if done:
                    key = jax.random.PRNGKey(ep * 1000 + step_i)
                    keys = jnp.expand_dims(key, 0)
                    state = vmap_reset(keys)
                    state = state.replace(info={**state.info, 'vel_cmd': jnp.expand_dims(cmd, 0)})
                    h = h0
                    term = jnp.array([False])
                else:
                    state = next_state

                # sync physics to viewer
                ps = state.pipeline_state
                mj_data.qpos[:] = np.array(ps.qpos[0])
                mj_data.qvel[:] = np.array(ps.qvel[0])
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                elapsed = time.time() - t0
                target = step_i * dt
                if target > elapsed:
                    time.sleep(target - elapsed)

                if not viewer.is_running():
                    break

            print(f"Episode {ep + 1}: return={ep_return:.1f}  steps={step_i + 1}")
            if not viewer.is_running():
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--cmd_vx", type=float, default=0.5, help="forward velocity command (m/s)")
    parser.add_argument("--cmd_vy", type=float, default=0.0, help="lateral velocity command (m/s)")
    parser.add_argument("--cmd_yaw", type=float, default=0.0, help="yaw rate command (rad/s)")
    parser.add_argument("--action_scale", type=float, default=0.25)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--env_variant", default="anymal")
    args = parser.parse_args()

    run(
        checkpoint_dir=args.checkpoint_dir,
        episodes=args.episodes,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_yaw=args.cmd_yaw,
        action_scale=args.action_scale,
        fps=args.fps,
        env_variant=args.env_variant,
    )
