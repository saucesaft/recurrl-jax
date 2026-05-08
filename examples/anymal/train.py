"""ANYmal velocity-tracking locomotion training."""
import jax
import numpy as np
import random
import wandb
import logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import recurrl_jax as rjx
import recurrl_jax.utils.wrappers as rjxw
import examples.anymal.anymal_gym as rjx_anymal
from recurrl_jax.model_fns import flatten_repr_model
import hydra

logger = logging.getLogger(__name__)

_train_env_ref = [None]

def make_env(env_config, trainer_config, global_config):
    num_envs = env_config.get('num_envs', trainer_config.get('num_envs', 512))
    env = rjx_anymal.AnymalGymWrapper(
        num_envs=num_envs,
        use_domain_randomization=env_config.get('use_domain_randomization', True),
        action_scale=env_config.get('action_scale', 0.5),
        cmd_scale_curriculum_steps=env_config.get('cmd_scale_curriculum_steps', 20_000_000),
        s_afilt_buf=2,
    )
    env = rjxw.VectorEpisodeStatisticsWrapper(env)
    _train_env_ref[0] = env
    return env

def make_eval_env(env_config, trainer_config, global_config, train_envs):
    shared_rms = getattr(train_envs, 'running_mean_std', getattr(train_envs.env, 'running_mean_std', None))
    eval_env = rjx_anymal.AnymalGymWrapper(
        num_envs=1,
        use_domain_randomization=False,
        action_scale=env_config.get('action_scale', 0.5),
        cmd_scale_curriculum_steps=0,
        shared_running_mean_std=shared_rms,
        update_norm_stats=False,
    )
    return rjxw.SqueezeWrapper(eval_env)

def make_video_render_fn(eval_env):
    import mujoco

    base_env = eval_env.env if hasattr(eval_env, 'env') else eval_env
    brax_env = base_env.env
    mj_model = brax_env.env.model

    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    def render_fn(env):
        base = env.env if hasattr(env, 'env') else env
        pipeline_state = base.env.state.pipeline_state

        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.array(pipeline_state.qpos[0])
        mj_data.qvel[:] = np.array(pipeline_state.qvel[0])
        mujoco.mj_forward(mj_model, mj_data)

        renderer.update_scene(mj_data)
        return renderer.render()

    return render_fn

@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    if config.use_wandb:
        wandb.init(project=config.project_name, config=OmegaConf.to_container(config))

    key = jax.random.PRNGKey(config.seed)
    trainer = rjx.Trainer(
        env_factory=make_env,
        eval_env_factory=make_eval_env,
        repr_fn=flatten_repr_model(),
        is_continuous=True,
        video_render_fn=None,
        global_args=config,
        trainer_config=config.trainer,
        env_config=config.task,
        seed=config.seed,
        key=key,
        wandb_run=wandb.run if config.use_wandb else None,
    )

    if config.get('render_videos', False) and trainer.agent.eval_env is not None:
        trainer.video_render_fn = make_video_render_fn(trainer.agent.eval_env)

    pbar = tqdm(total=config.steps)
    step_count = 0
    last_step_count = 0
    curriculum_counter = 0
    last_ep_len = 0
    
    # extract curriculum thresholds from config
    curriculum_ep_len_threshold = config.task.get('curriculum_ep_len_threshold', 100)
    curriculum_total_steps = config.task.get('cmd_scale_curriculum_steps', 20_000_000)

    with logging_redirect_tqdm():
        while step_count < config.steps:
            _, metrics, new_step_count = trainer.step()
            
            delta = new_step_count - step_count
            pbar.update(delta)
            step_count = new_step_count

            if metrics:
                last_ep_len = metrics.get('episode_length', last_ep_len)
                metrics['curriculum_progress'] = curriculum_counter / max(curriculum_total_steps, 1)
                
                logger.info(f"Steps: {step_count} Curriculum: {curriculum_counter}/{curriculum_total_steps} Metrics: {metrics}")
                if config.use_wandb:
                    wandb.log(metrics, step=step_count)
                    
            # advance curriculum every step based on last known episode_length
            if last_ep_len > curriculum_ep_len_threshold:
                curriculum_counter = min(curriculum_counter + delta, curriculum_total_steps)

            # sync curriculum into all envs every step
            if _train_env_ref[0] is not None:
                _train_env_ref[0].env.env.sync_curriculum_step(curriculum_counter)

    pbar.close()

if __name__ == '__main__':
    main()
