import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pathlib import Path
import imageio
import mujoco

from examples.leap_hand.observation_buffer import build_asymmetric_observation
from examples.leap_hand.env import _leap_reward_fn, _leap_termination_fn, _leap_reset_state_fn
from recurrl_jax.utils.mjx_env import mjx_training_step
from recurrl_jax.utils.running_mean_std import normalize_jit


def render_policy_video_asymmetric(
    env,
    model,
    policy_running_mean_std,
    critic_running_mean_std,
    last_action,
    key,
    config,
    epoch: int,
    output_dir: str = "videos"
):
    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_path = output_path / f"policy_epoch_{epoch:04d}.mp4"

    # initialize renderer
    renderer = None
    if config.renderer == "mujoco":
        mj_model = mujoco.MjModel.from_xml_path(config.xml_path)
        renderer = mujoco.Renderer(mj_model, height=config.video_height, width=config.video_width)
    elif config.renderer == "madrona":
        print("Warning: Madrona renderer not implemented yet, skipping video generation")
        video_path.touch()
        return video_path
    else:
        print(f"Warning: Unknown renderer '{config.renderer}', skipping video generation")
        video_path.touch()
        return video_path

    # run rollout for video_length steps
    frames = []

    # extract first env state
    single_last_action = last_action[0:1]  # (1, 16)
    single_mjx_data = jax.tree.map(lambda x: x[0:1], env.mjx_data_batch)
    single_mjx_model = env.mjx_model
    single_progress_buf = env.progress_buf[0:1]

    # initial DOF positions for pose diff penalty (from grasp cache)
    single_initial_dof_pos = env.initial_dof_pos[0:1]
    single_angvel_z_smooth = env.angvel_z_smooth[0:1]
    action_ema_alpha = getattr(config, 'action_ema_alpha', 0.0)
    single_smoothed_actions = None

    # convert mjx_data to mujoco data for rendering
    mj_data = mujoco.MjData(mj_model)

    for step in range(config.video_length):
        # extract current state from mjx_data
        qpos = single_mjx_data.qpos
        qvel = single_mjx_data.qvel

        joint_angles = qpos[:, :16]
        joint_velocities = qvel[:, :16]
        joint_torques = single_mjx_data.qfrc_actuator[:, :16]

        cube_pos = qpos[:, 16:19]
        cube_quat = qpos[:, 19:23]
        cube_linvel = qvel[:, 16:19]
        cube_angvel = qvel[:, 19:22]

        # get fingertip and palm positions
        fingertip_positions = env.get_fingertip_positions(single_mjx_data)
        palm_pos = env.get_palm_position(single_mjx_data)

        # build asymmetric observation
        key, obs_key = jr.split(key)
        privileged_obs = build_asymmetric_observation(
            joint_angles=joint_angles,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            last_action=single_last_action,
            fingertip_positions=fingertip_positions,
            cube_pos=cube_pos,
            palm_pos=palm_pos,
            cube_quat=cube_quat,
            cube_angvel=cube_angvel,
            cube_linvel=cube_linvel,
            key=obs_key,
            noise_level=0.0  # no noise for visualization
        )

        # normalize policy and critic observations independently
        policy_obs = privileged_obs[:, :32]
        policy_obs_normalized = normalize_jit(
            policy_obs,
            policy_running_mean_std.mean,
            policy_running_mean_std.var,
            policy_running_mean_std.epsilon
        )
        critic_obs_normalized = normalize_jit(
            privileged_obs,
            critic_running_mean_std.mean,
            critic_running_mean_std.var,
            critic_running_mean_std.epsilon
        )

        # concatenate: [normalized 32D policy, normalized 105D critic]
        obs_normalized = jnp.concatenate([policy_obs_normalized, critic_obs_normalized], axis=-1)

        # get action from policy (deterministic)
        # model returns (action_means, action_std, value)
        mu, _, _ = model(obs_normalized)
        action = mu

        # apply action EMA smoothing
        if action_ema_alpha > 0.0:
            if single_smoothed_actions is None:
                single_smoothed_actions = action
            single_smoothed_actions = action_ema_alpha * single_smoothed_actions + (1.0 - action_ema_alpha) * action
            stepped_action = single_smoothed_actions
        else:
            stepped_action = action

        # step environment
        key, step_key = jr.split(key)
        raw_state, reward, done, termination, info, single_mjx_data, single_progress_buf, single_mjx_model, single_angvel_z_smooth = mjx_training_step(
            stepped_action,
            single_mjx_model,
            single_mjx_data,
            single_progress_buf,
            single_initial_dof_pos,
            env.reset_height_threshold,
            config.episode_length,
            step_key,
            single_angvel_z_smooth,
            spec=env.spec,
            reward_fn=_leap_reward_fn,
            termination_fn=_leap_termination_fn,
            reset_state_fn=_leap_reset_state_fn,
            control_freq_inv=config.control_freq_inv,
            use_domain_randomization=env.use_domain_randomization,
            dr_params=env.dr_params,
            base_mjx=env.base_mjx,
            action_scale=config.action_scale,
            grasp_cache=env.grasp_cache,
            grasp_cache_size=env.grasp_cache_size,
        )

        # render frame
        if config.renderer == "mujoco":
            # copy mjx state to mujoco data
            mj_data.qpos[:] = np.array(single_mjx_data.qpos[0])
            mj_data.qvel[:] = np.array(single_mjx_data.qvel[0])

            mujoco.mj_forward(mj_model, mj_data)

            renderer.update_scene(mj_data)
            frame = renderer.render()
            frames.append(frame)

        # update last_action
        single_last_action = action

        # check if episode ended
        if done[0]:
            print(f"Episode ended at step {step+1}/{config.video_length} (reward: {float(reward[0]):.2f})")
            break

    # save video
    if len(frames) > 0:
        try:
            imageio.mimsave(video_path, frames, fps=config.video_fps, codec='libx264')
            print(f"Saved video to {video_path} ({len(frames)} frames)")
        except Exception as e:
            print(f"Warning: Failed to save video: {e}")
            video_path.touch()
    else:
        print("Warning: No frames generated, creating placeholder")
        video_path.touch()

    return video_path
