"""InvertedPendulum-v4 (MuJoCo) with MLP policy"""
import logging

import hydra
import jax
import numpy as np
import random
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import recurrl_jax as rjx
import recurrl_jax.utils.wrappers as rjxw
from recurrl_jax.model_fns import flatten_repr_model
from examples.pendulum.env_wrapper import InvertedPendulumVecEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(env_config, trainer_config, global_config):
    num_envs = env_config.get("num_envs", trainer_config.get("num_envs", 32))
    env = InvertedPendulumVecEnv(num_envs=num_envs)
    env = rjxw.VectorEpisodeStatisticsWrapper(env)
    return env


def make_eval_env(env_config, trainer_config, global_config, train_envs):
    env = InvertedPendulumVecEnv(num_envs=1)
    env = rjxw.SqueezeWrapper(env)
    return env


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    logger.info("[Pendulum MLP]\n" + OmegaConf.to_yaml(config))

    key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

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
        wandb_run=None,
    )

    pbar = tqdm(total=config.steps)
    step_count = 0
    last_step_count = 0

    with logging_redirect_tqdm():
        while True:
            loss, metrics, step_count = trainer.step()
            pbar.update(n=step_count - last_step_count)
            last_step_count = step_count

            if metrics is not None:
                logger.info(f"Steps: {step_count} | {metrics}")

            if step_count >= config.steps:
                break

    pbar.close()


if __name__ == "__main__":
    main()
