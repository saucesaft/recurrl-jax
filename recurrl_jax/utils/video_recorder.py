"""video recording utilities for evaluation episodes"""

import jax
import jax.numpy as jnp
import numpy as np
import imageio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def record_evaluation_episode(
    eval_env,
    agent,
    random_key,
    episode_id,
    video_path,
    video_config,
    render_fn=None,
    record=True,
):
    # reset environment
    o_tick, _ = eval_env.reset()

    # initialize hidden state (single env)
    h_tickminus1 = jax.tree.map(lambda x: jnp.expand_dims(jnp.zeros(x[0].shape), 0), agent.h_tickminus1)
    term_tick = jnp.zeros((1, 1), dtype=bool)

    frames, rewards = [], []
    done, step_count = False, 0
    max_steps = video_config.get('max_length', 500)

    while not done and step_count < max_steps:
        random_key, model_key = jax.random.split(random_key)
        act_logits, v_tick, htick = agent.actor_critic_fn(
            model_key,
            agent.params,
            jnp.expand_dims(o_tick, axis=(0, 1)),
            term_tick,
            h_tickminus1
        )

        if agent.is_continuous:
            mean, log_std = act_logits
            acts_tick = mean.squeeze(axis=1)
            acts_tick = jnp.clip(acts_tick, -1.0, 1.0)
        else:
            acts_tick = jnp.argmax(act_logits, axis=-1)

        o_tick, r_tick, term, trunc, info = eval_env.step(acts_tick)
        done = term or trunc
        term_tick = jnp.array([[done]], dtype=bool)
        rewards.append(r_tick)
        h_tickminus1 = htick

        if record and render_fn is not None:
            frame = render_fn(eval_env)
            if frame is not None:
                frames.append(frame)

        step_count += 1

    # save video
    if record and len(frames) > 0:
        try:
            imageio.mimsave(video_path, frames, fps=video_config['fps'], codec='libx264')
            logger.info(f"Saved video: {video_path} ({len(frames)} frames)")
        except Exception as e:
            logger.warning(f"Failed to save video {video_path}: {e}")

    rewards_array = jnp.array(rewards, dtype=jnp.float32)
    return len(rewards), float(rewards_array.sum()), random_key


def record_evaluation_videos(
    eval_env,
    agent,
    random_key,
    num_episodes,
    video_dir,
    step_count,
    video_config,
    render_fn=None,
):
    step_video_dir = Path(video_dir) / f"step_{step_count:08d}"
    step_video_dir.mkdir(parents=True, exist_ok=True)

    episode_lengths, episode_returns = [], []

    for ep_id in range(num_episodes):
        video_path = step_video_dir / f"episode_{ep_id:03d}.mp4"

        # only record the first episode
        record = (ep_id == 0)

        ep_len, ep_return, random_key = record_evaluation_episode(
            eval_env,
            agent,
            random_key,
            ep_id,
            video_path,
            video_config,
            render_fn=render_fn,
            record=record,
        )

        episode_lengths.append(ep_len)
        episode_returns.append(ep_return)

    avg_len, avg_return = np.mean(episode_lengths), np.mean(episode_returns)
    logger.info(f"Recorded {num_episodes} videos (actual files: 1) at step {step_count}: "
                f"avg_len={avg_len:.1f}, avg_return={avg_return:.2f}")

    return avg_len, avg_return, random_key
