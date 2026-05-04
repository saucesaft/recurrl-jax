"""load best checkpoint and render trained InvertedPendulum-v4 policy."""
import argparse
import time
from pathlib import Path

import gymnasium as gym
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


def build_model(d_actor=64, d_critic=128, d_model=64, n_layers=2, action_dim=1):
    repr_fn = flatten_repr_model()
    seq_fn, seq_init = seq_model_mlp(name="mlp", d_model=d_model, n_layers=n_layers)
    actor_fn = actor_model_continuous(d_actor, action_dim)
    critic_fn = critic_model(d_critic)

    ac = nn.vmap(
        ActorCriticModel,
        variable_axes={"params": None},
        split_rngs={"params": False},
    )(repr_fn, seq_fn, actor_fn, critic_fn, use_asymmetric_obs=False)

    return ac, seq_init


def load_params(checkpoint_dir: str, seq_init, num_envs=1):
    h0 = jax.tree.map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), num_envs, axis=0),
        seq_init(),
    )

    ckpt_path = Path(checkpoint_dir).resolve()
    manager = CheckpointManager(
        ckpt_path,
        PyTreeCheckpointer(),
        CheckpointManagerOptions(create=False),
    )
    step = manager.latest_step()
    print(f"Restoring checkpoint at step {step}")

    restored = manager.restore(step)
    params = jax.tree.map(jnp.asarray, restored["params"])
    return params, h0


def run(checkpoint_dir="checkpoints", episodes=3, save_gif=None,
        d_actor=64, d_critic=128, d_model=64, n_layers=2, fps=30):

    ac, seq_init = build_model(d_actor=d_actor, d_critic=d_critic,
                               d_model=d_model, n_layers=n_layers)
    params, h0 = load_params(checkpoint_dir, seq_init)

    # no render_mode — we drive the viewer ourselves
    env = gym.make("InvertedPendulum-v4")
    mj_model = env.unwrapped.model
    mj_data  = env.unwrapped.data
    action_scale = float(env.action_space.high[0])  # 3.0

    @jax.jit
    def policy_step(params, obs, term, h):
        obs_b = obs[None, None, :]
        term_b = term[None, None]
        (mean, _), _, h_new = ac.apply(params, obs_b, term_b, h, rngs={"random": jax.random.PRNGKey(0)})
        return mean[0, 0], h_new

    if save_gif:
        renderer = mujoco.Renderer(mj_model, height=480, width=640)

    dt = 1.0 / fps

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        frames = []
        for ep in range(episodes):
            obs, _ = env.reset()
            obs = jnp.asarray(obs, dtype=jnp.float32)
            h = h0
            term = jnp.array([False])
            ep_return = 0.0
            t0 = time.time()

            for i in range(1000):
                action, h = policy_step(params, obs, term, h)
                action_np = np.asarray(action) * action_scale
                obs, reward, terminated, truncated, _info = env.step(action_np)
                obs = jnp.asarray(obs, dtype=jnp.float32)
                ep_return += float(reward)
                done = terminated or truncated
                term = jnp.array([done])

                viewer.sync()

                if save_gif:
                    renderer.update_scene(mj_data)
                    frames.append(renderer.render())

                # pace to real-time
                elapsed = time.time() - t0
                target = i * dt
                if target > elapsed:
                    time.sleep(target - elapsed)

                if done or not viewer.is_running():
                    break

            print(f"Episode {ep + 1}: return = {ep_return:.1f}")

            if not viewer.is_running():
                break

    env.close()

    if save_gif and frames:
        import imageio
        imageio.mimsave(save_gif, frames, fps=fps)
        print(f"Saved to {save_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--save_gif", default=None, help="path to save gif, e.g. out.gif")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    run(checkpoint_dir=args.checkpoint_dir, episodes=args.episodes,
        save_gif=args.save_gif, fps=args.fps)
