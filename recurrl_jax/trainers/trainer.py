import json
import numpy as np
import optax
import pandas as pd
import rlax
import wandb
import gymnasium as gym
import time
import logging
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from argparse import Namespace
from recurrl_jax.trainers.base_trainer import BaseTrainer
from collections import OrderedDict
from recurrl_jax.agents.a2c import A2CAgent
from recurrl_jax.agents.ppo import PPOAgent
from recurrl_jax.model_fns import *
from recurrl_jax.utils.wrappers import VectorEpisodeStatisticsWrapper, SqueezeWrapper
from recurrl_jax.trainers.utils import *
from gymnasium.wrappers import AutoResetWrapper
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManagerOptions, CheckpointManager
from pathlib import Path
from recurrl_jax.utils.video_recorder import record_evaluation_videos
import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)

# suppress verbose orbax/absl logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('orbax').setLevel(logging.ERROR)


class Trainer(BaseTrainer):
    """generalized trainer using env_factory pattern"""

    def __init__(self, *, env_factory, eval_env_factory=None, repr_fn=None,
                 is_continuous=None, video_render_fn=None, **kwargs):

        self.wandb_run=kwargs['wandb_run']
        self.trainer_config=kwargs['trainer_config']
        self.env_config=kwargs.get('env_config', {})
        self.global_config=kwargs['global_args']
        self.video_render_fn = video_render_fn

        # checkpoint setup
        self.checkpoint_dir = self.global_config.get('checkpoint_dir', 'checkpoints')
        self.best_eval_return = -float('inf')
        self.checkpoint_manager = None

        if self.checkpoint_dir:
            checkpoint_path = Path(self.checkpoint_dir).resolve()
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            self.checkpoint_manager = CheckpointManager(
                checkpoint_path,
                PyTreeCheckpointer(),
                options=options
            )
            logger.info(f"Checkpoint manager initialized at {checkpoint_path}")

        # video rendering setup
        self.video_dir = self.global_config.get('video_dir', None)
        self.render_videos = self.global_config.get('render_videos', False)
        if self.render_videos and self.video_dir:
            video_path = Path(self.video_dir)
            video_path.mkdir(parents=True, exist_ok=True)

            self.video_config = {
                'width': self.global_config.get('video_width', 640),
                'height': self.global_config.get('video_height', 480),
                'fps': self.global_config.get('video_fps', 30),
                'max_length': self.global_config.get('video_length', 500),
            }
            logger.info(f"Video rendering enabled, saving to {video_path}")
        else:
            self.video_config = None

        self.rollout_len=self.trainer_config['rollout_len']
        # env config num_envs takes priority
        if 'num_envs' in self.env_config:
             self.num_envs = self.env_config['num_envs']
             self.trainer_config['num_envs'] = self.num_envs
        else:
             self.num_envs=self.trainer_config['num_envs']

        self.gamma=self.trainer_config['gamma']

        params_key,self.random_key=jax.random.split(kwargs['key'])

        train_envs = env_factory(self.env_config, self.trainer_config, self.global_config)

        if eval_env_factory is not None:
            eval_env = eval_env_factory(self.env_config, self.trainer_config, self.global_config, train_envs)
        else:
            eval_env = None

        train_envs.reset()
        if eval_env is not None:
            eval_env.reset()

        # extract policy_obs_dim from environment (for asymmetric actor-critic)
        policy_obs_dim = getattr(train_envs, 'policy_obs_dim', 102)
        if policy_obs_dim == 102 and hasattr(train_envs, 'env'):
            policy_obs_dim = getattr(train_envs.env, 'policy_obs_dim', 102)

        # auto-detect continuous vs discrete action space
        if is_continuous is None:
            is_continuous = hasattr(train_envs, 'action_space') and isinstance(
                train_envs.action_space, gym.spaces.Box
            )

        if is_continuous:
            action_dim = train_envs.action_space.shape[0]
            actor_fn = actor_model_continuous(
                self.trainer_config['d_actor'], action_dim,
                log_std_min=self.trainer_config.get('log_std_min', -5.0),
                log_std_max=self.trainer_config.get('log_std_max', 2.0)
            )
        else:
            action_dim = train_envs.action_space.n
            actor_fn = actor_model_discete(self.trainer_config['d_actor'], action_dim)

        logger.info("Observation space: "+str(train_envs.observation_space))
        logger.info("Action space: "+str(train_envs.action_space))

        # use provided repr_fn or default
        if repr_fn is None:
            repr_fn = flatten_repr_model()

        if self.trainer_config.seq_model.name=='lstm':
            model_fn=seq_model_lstm(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='gru':
            model_fn=seq_model_gru(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='gtrxl':
            model_fn=seq_model_gtrxl(**self.trainer_config['seq_model'])

        critic_fn=critic_model(self.trainer_config['d_critic'])
        #setup optimizer

        if self.trainer_config['agent']=='a2c':
            self.optimizer=optax.chain(optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),
                                    optax.adam(**self.trainer_config.optimizer))
            self.agent=A2CAgent(train_envs=train_envs,eval_env=eval_env,optimizer=self.optimizer, repr_model_fn=repr_fn,
                                seq_model_fn=model_fn,actor_fn=actor_fn,critic_fn=critic_fn,
                                rollout_len=self.rollout_len,
                                gamma=self.trainer_config['gamma'],lamb=self.trainer_config['lamb'],
                                value_loss_coef=self.trainer_config['value_coef'],
                                entropy_coef=self.trainer_config['entropy_coef'],
                                arg_max=self.trainer_config['arg_max'],
                                is_continuous=is_continuous,
                                policy_obs_dim=policy_obs_dim)
        elif self.trainer_config['agent']=='ppo':
            batch_size = self.trainer_config['num_envs']*self.trainer_config['rollout_len']
            num_updates = self.global_config.steps // batch_size

            try:
                optimizer_config = dict(self.trainer_config.optimizer)
                learning_rate = optimizer_config.pop("learning_rate", None)
            except (AttributeError, KeyError, TypeError, ConfigAttributeError):
                optimizer_config = {}
                learning_rate = None

            if learning_rate is None:
                learning_rate = {
                    'initial': 3e-4,
                    'final': 3e-4,
                    'power': 1,
                    'max_decay_steps': self.global_config.steps
                }
            if learning_rate.get('final') is None:
                learning_rate['final'] = learning_rate['initial']

            if self.trainer_config['ent_coef']['final'] is None:
                self.trainer_config['ent_coef']['final']=self.trainer_config['ent_coef']['initial']
            lr_schedule=optax.polynomial_schedule(learning_rate['initial'],learning_rate['final'],learning_rate.get('power', 1),learning_rate.get('max_decay_steps', self.global_config.steps))
            ent_schedule=optax.polynomial_schedule(self.trainer_config['ent_coef']['initial'],self.trainer_config['ent_coef']['final'],
                                                   self.trainer_config['ent_coef']['power'],self.trainer_config['ent_coef']['max_decay_steps'])

            self.optimizer=optax.chain(
                                optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),
                                optax.inject_hyperparams(optax.adamw)(
                                    learning_rate=lr_schedule,
                                    **optimizer_config
                                ),
                            )
            # adaptive LR config
            adaptive_lr_config = self.trainer_config.get('adaptive_lr', {})
            if isinstance(adaptive_lr_config, bool):
                adaptive_lr_enabled = adaptive_lr_config
                kl_threshold = 0.02
                lr_min = 1e-6
                lr_max = 1e-2
            else:
                adaptive_lr_enabled = adaptive_lr_config.get('enabled', False)
                kl_threshold = adaptive_lr_config.get('kl_threshold', 0.02)
                lr_min = adaptive_lr_config.get('lr_min', 1e-6)
                lr_max = adaptive_lr_config.get('lr_max', 1e-2)

            self.agent=PPOAgent(train_envs=train_envs,eval_env=eval_env,optimizer=self.optimizer, repr_model_fn=repr_fn,
                                seq_model_fn=model_fn,actor_fn=actor_fn,critic_fn=critic_fn,
                                num_steps=self.rollout_len,
                                gamma=self.trainer_config.get('gamma', 0.99),
                                gae_lambda=self.trainer_config.get('gae_lambda', 0.95),
                                num_minibatches=self.trainer_config.get('num_minibatches', 4),
                                update_epochs=self.trainer_config.get('update_epochs', 4),
                                norm_adv=self.trainer_config.get('norm_adv', True),
                                clip_coef=self.trainer_config.get('clip_coef', 0.1),
                                lr_schedule=lr_schedule,
                                ent_schedule=ent_schedule,
                                vf_coef=self.trainer_config.get('vf_coef', 0.5),
                                max_grad_norm=self.trainer_config.get('max_grad_norm', 0.5),
                                target_kl=self.trainer_config.get('target_kl', None),
                                sequence_length=self.trainer_config.get('sequence_length', None),
                                is_continuous=is_continuous,
                                log_std_min=self.trainer_config.get('log_std_min', -5.0),
                                log_std_max=self.trainer_config.get('log_std_max', 2.0),
                                adaptive_lr=adaptive_lr_enabled,
                                kl_threshold=kl_threshold,
                                lr_min=lr_min,
                                lr_max=lr_max,
                                policy_obs_dim=policy_obs_dim)


        self.agent.reset(params_key,self.random_key)
        self.step_count=0
        self.episode_lengths=[]
        self.average_reward_per_episode=[]
        self.average_return_per_episode=[]
        self.losses=[]
        self.critic_losses=[]
        self.actor_losses=[]
        self.entropy_losses=[]
        self.sps=[]
        self.result_data=[]
        self.reward_sum=0
        self.statistic_data=dict()
        self.B=self.num_envs*self.rollout_len
        self.log_interval=self.global_config.log_interval
        self.next_log_step=self.log_interval
        self.average_return_per_episode=[]
        if 'eval_interval' in self.global_config:
            self.eval_interval=self.global_config['eval_interval']
            self.next_eval_step=self.eval_interval
        else:
            self.eval_interval=None


    def step(self, **kwargs):
        self.random_key=jax.random.split(self.random_key)[0]

        #measure steps per second
        start_time=time.time()

        (loss,(value_loss,entropy_loss,actor_loss,rewards,diagnostics),infos)=self.agent.step(self.random_key)

        # NaN detection
        if jnp.isnan(loss) or jnp.isnan(actor_loss) or jnp.isnan(value_loss):
            logger.error(f"NaN detected at step {self.step_count}! "
                        f"loss={loss}, actor_loss={actor_loss}, value_loss={value_loss}")
            raise ValueError(f"NaN loss detected at step {self.step_count}")

        #extract info data across all actors and steps

        # increase the step counter
        self.step_count+=(self.B)

        # generic info extraction
        for step_info in infos:
            # handle episode statistics from VectorEpisodeStatisticsWrapper
            if 'episode' in step_info:
                for ep_info in step_info['episode']:
                    self.statistic_data.setdefault('reward_per_episode', []).append(ep_info['r'])
                    self.statistic_data.setdefault('episode_length', []).append(ep_info['l'])
                    self.average_return_per_episode.append(ep_info['r'])

            # generic diagnostic logging
            for key, val in step_info.items():
                if key in ('episode', 'done', 'termination'):
                    continue
                if hasattr(val, 'mean') and hasattr(val, 'shape') and len(val.shape) > 0:
                    mean_val = float(val.mean())
                    max_val = float(val.max())
                    self.statistic_data.setdefault(f'diag_{key}_mean', []).append(mean_val)
                    self.statistic_data.setdefault(f'diag_{key}_max', []).append(max_val)
                elif isinstance(val, (int, float)):
                    self.statistic_data.setdefault(key, []).append(val)

        # log the data
        end_time=time.time()
        self.sps.append(self.B/(end_time-start_time))
        self.losses.append(loss)
        self.critic_losses.append(value_loss)
        self.actor_losses.append(actor_loss)
        self.entropy_losses.append(entropy_loss)
        self.reward_sum+=rewards.sum()

        # track action distribution diagnostics for continuous control
        if diagnostics['action_std'] is not None:
            if not hasattr(self, 'action_stds'):
                self.action_stds = []
                self.log_std_means = []
                self.action_mean_abs_vals = []
            self.action_stds.append(float(diagnostics['action_std']))
            self.log_std_means.append(float(diagnostics['log_std_mean']))
            self.action_mean_abs_vals.append(float(diagnostics['action_mean_abs']))
        if self.step_count>=self.next_log_step:
            #calculate and log mean of statistic_data
            metrics={}
            for key in self.statistic_data.keys():
                agg_value=np.mean(self.statistic_data[key])
                metrics={**metrics,key:agg_value}
                self.statistic_data[key]=[]
            self.next_log_step+=self.log_interval
            critic_loss=np.mean(self.critic_losses)
            actor_loss=np.mean(self.actor_losses)
            entropy_loss=np.mean(self.entropy_losses)
            loss=np.mean(self.losses)
            reward_mean=float(self.reward_sum/self.log_interval)
            return_mean=np.mean(self.average_return_per_episode)
            mean_sps=np.mean(self.sps)
            self.reward_sum=0
            self.critic_losses=[]
            self.actor_losses=[]
            self.entropy_losses=[]
            self.losses=[]
            self.sps=[]
            self.average_return_per_episode=[]

            # add action distribution diagnostics if available
            if hasattr(self, 'action_stds') and len(self.action_stds) > 0:
                metrics['action_std'] = np.mean(self.action_stds)
                metrics['log_std_mean'] = np.mean(self.log_std_means)
                metrics['action_mean_abs'] = np.mean(self.action_mean_abs_vals)
                self.action_stds = []
                self.log_std_means = []
                self.action_mean_abs_vals = []

            metrics={'step':self.step_count,'sps':mean_sps,'loss':loss,'critic_loss':critic_loss,
                                    'actor_loss':actor_loss,'entropy_loss':entropy_loss,'mean_reward':reward_mean,
                                    'return_per_episode':return_mean,
                                    **metrics
                                    }
            self.result_data.append(metrics)
        else:
            metrics=None
        if self.eval_interval is not None and self.step_count>=self.next_eval_step:
            self.next_eval_step+=self.eval_interval

            # video rendering via callback
            if self.render_videos and self.video_config and self.video_render_fn is not None:
                avg_episode_len, avg_episode_return, self.random_key = record_evaluation_videos(
                    eval_env=self.agent.eval_env,
                    agent=self.agent,
                    random_key=self.random_key,
                    num_episodes=self.global_config['eval_episodes'],
                    video_dir=Path(self.video_dir),
                    step_count=self.step_count,
                    video_config=self.video_config,
                    render_fn=self.video_render_fn,
                )
                rollouts = []
            else:
                avg_episode_len, avg_episode_return, rollouts = self.agent.evaluate(
                    self.random_key,
                    self.global_config['eval_episodes']
                )

                if len(rollouts) > 0:
                    rollouts = np.concatenate(rollouts, axis=0)
                    if metrics is None:
                        metrics = {}
                    metrics['rollouts'] = wandb.Video(
                        rollouts,
                        fps=self.global_config.get('record_fps', 5),
                        format="gif"
                    )

            # add eval metrics
            if metrics is None:
                metrics = {}
            metrics['step'] = self.step_count
            metrics['eval_avg_episode_len'] = float(avg_episode_len)
            metrics['eval_avg_episode_return'] = float(avg_episode_return)

            # save checkpoint if best model
            if self.checkpoint_manager and avg_episode_return > self.best_eval_return:
                self.best_eval_return = avg_episode_return

                checkpoint_state = {
                    'params': self.agent.params,
                    'optimizer_state': self.agent.optimizer_state,
                    'update_tick': np.array(self.agent.update_tick),
                    'step_count': self.step_count,
                    'best_eval_return': float(self.best_eval_return),
                }

                self.checkpoint_manager.save(self.step_count, checkpoint_state)
                logger.info(f"Saved best checkpoint at step {self.step_count} "
                            f"with eval return {self.best_eval_return:.2f}")
                metrics['checkpoint_saved'] = True

        return loss,metrics,self.step_count


    def get_summary_table(self):
        return pd.DataFrame(self.result_data).to_json(default_handler=str)
