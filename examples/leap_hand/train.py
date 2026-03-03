"""
Task-specific entry point that creates environment factories
and passes them to the generic library trainer.
"""
import sys
import os

os.environ['JAX_LOG_COMPILES'] = '0'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import jax

import numpy as np
import random
import wandb

import logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from pathlib import Path

import recurrl_jax as rjx
import recurrl_jax.utils.wrappers as rjxw

import examples.leap_hand.env_wrapper as rjx_leap

from recurrl_jax.model_fns import flatten_repr_model

import hydra
from hydra.core.plugins import Plugins
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra import compose, initialize_config_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_env(env_config, trainer_config, global_config):
    num_envs = env_config.get('num_envs', trainer_config.get('num_envs', 8192))

    env = rjx_leap.LeapHandGymWrapper(
        num_envs=num_envs,
        use_domain_randomization=env_config.get('use_domain_randomization', True),
        normalize_obs=True,
        action_scale=env_config.get('action_scale', 0.6),
        grasp_cache_path=env_config.get('grasp_cache_path', None),
        update_norm_stats=True,
    )

    # wrap with episode statistics tracker
    env = rjxw.VectorEpisodeStatisticsWrapper(env)

    return env


def make_eval_env(env_config, trainer_config, global_config, train_envs):
    # share running mean std from train envs for consistent normalization
    shared_rms = train_envs.env.running_mean_std if hasattr(train_envs, 'env') and hasattr(train_envs.env, 'running_mean_std') else None

    eval_env = rjx_leap.LeapHandGymWrapper(
        num_envs=1,
        use_domain_randomization=False,  # no DR for eval
        normalize_obs=True,
        action_scale=env_config.get('action_scale', 0.6),
        grasp_cache_path=env_config.get('grasp_cache_path', None),
        shared_running_mean_std=shared_rms,
        update_norm_stats=False,  # don't update stats during eval
    )

    # squeeze wrapper to remove batch dimension for evaluation
    eval_env = rjxw.SqueezeWrapper(eval_env)

    return eval_env

def make_video_render_fn(eval_env):
    import mujoco

    # get the MuJoCo model and renderer
    # access through wrapper hierarchy to get to underlying LeapHandGymWrapper
    if hasattr(eval_env, 'env'):
        base_env = eval_env.env  # SqueezeWrapper.env
    else:
        base_env = eval_env

    # get underlying MJX environment's MuJoCo model
    mjx_env = base_env.env  # LeapHandGymWrapper.env = MJXLeapHandEnv
    mj_model = mjx_env.mj_model  # MuJoCo model

    # create renderer (we'll need to create data from mjx_data)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    def render_fn(env): # render current state to RGB frame
        # get MuJoCo data from MJX
        if hasattr(env, 'env'):
            base = env.env
        else:
            base = env

        mjx_env = base.env
        mjx_data = mjx_env.mjx_data_batch

        # create MuJoCo data from MJX data (first environment)
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.array(mjx_data.qpos[0]) # TODO can we do this with jax?
        mj_data.qvel[:] = np.array(mjx_data.qvel[0])
        mujoco.mj_forward(mj_model, mj_data)

        # render
        renderer.update_scene(mj_data)
        return renderer.render()

    return render_fn

# TODO different way to load configs? abstract Hydra
@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    logger.info("[LEAP Hand Example]\n" + str(OmegaConf.to_yaml(config)))

    tags = config.tags.split(',') if config.tags is not None else []

    if config.use_wandb:
        run = wandb.init(
            project=config.project_name,
            tags=tags,
            settings=wandb.Settings(start_method="fork"),
            config=OmegaConf.to_container(config)
        )
    else:
        run = None

    # random seed
    key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    trainer_config = config.trainer
    env_config = config.task

    # video render function factory
    def video_render_fn_factory(eval_env):
        return make_video_render_fn(eval_env)

    # trainer with env factories
    kwargs = {
        'global_args': config,
        'trainer_config': trainer_config,
        'env_config': env_config,
        'seed': config.seed,
        'key': key,
        'wandb_run': run,
    }

    trainer = rjx.Trainer(
        env_factory=make_env,
        eval_env_factory=make_eval_env,
        repr_fn=flatten_repr_model(),
        is_continuous=True,
        video_render_fn=None,  # set below
        **kwargs
    )

    # set up video rendering if enabled
    if config.get('render_videos', False) and trainer.agent.eval_env is not None:
        trainer.video_render_fn = make_video_render_fn(trainer.agent.eval_env)

    # training loop
    pbar = tqdm(total=config.steps)
    step_count = 0
    last_step_count = 0

    with logging_redirect_tqdm():
        while True:
            loss, metrics, step_count = trainer.step()
            pbar.update(n=step_count - last_step_count)
            last_step_count = step_count

            if metrics is not None:
                logger.info(f"Seed: {config.seed} Steps: {step_count} Metrics: {metrics}")
                if config.use_wandb:
                    run.log({'seed': config.seed, **metrics}, step=step_count)

            if step_count >= config.steps:
                break

    pbar.close()

    if config.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
