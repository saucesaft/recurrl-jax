import jax
import jax.numpy as jnp
import optax
import rlax
import flax
import tqdm
import jax.numpy as jnp
import numpy as np
import time
import optax
import distrax

from flax.training.train_state import TrainState
from recurrl_jax.models.actor_critic import *
from typing import Callable,Tuple
from recurrl_jax.agents.base_agent import BaseAgent
from recurrl_jax.utils.distributions import squashed_gaussian



class PPOAgent(BaseAgent):

    def __init__(self, train_envs,eval_env, repr_model_fn:Callable,seq_model_fn:Tuple[Callable,Callable],
                        actor_fn:Callable,critic_fn:Callable,optimizer:optax.GradientTransformation,
                         num_steps=128, gamma=0.99, lr_schedule=optax.linear_schedule,
                        gae_lambda=0.95, num_minibatches=4, update_epochs=4, norm_adv=True,
                        clip_coef=0.1, ent_schedule=optax.Schedule, vf_coef=0.5, max_grad_norm=0.5,
                        target_kl=None,sequence_length=None, is_continuous=False,
                        log_std_min=-5.0, log_std_max=2.0,
                        adaptive_lr=False, kl_threshold=0.02, lr_min=1e-6, lr_max=1e-2,
                        use_asymmetric_obs=True, policy_obs_dim=102) -> None:

        super(PPOAgent,self).__init__(train_envs=train_envs,eval_env=eval_env,rollout_len=num_steps,repr_model_fn=repr_model_fn,seq_model_fn=seq_model_fn,
                        actor_fn=actor_fn,critic_fn=critic_fn,use_gumbel_sampling=True,sequence_length=sequence_length, is_continuous=is_continuous,
                        use_asymmetric_obs=use_asymmetric_obs, policy_obs_dim=policy_obs_dim)

        self.arg_max = True
        self.optimizer=optimizer
        self.num_envs = self.env.num_envs
        self.gamma = gamma
        self.lr_schedule = lr_schedule
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.ent_schedule = ent_schedule
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.update_tick=jnp.array(0)

        # adaptive LR parameters
        self.adaptive_lr = adaptive_lr
        self.kl_threshold = kl_threshold
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.current_lr = None  # will be set in reset()

        @jax.jit
        def update_ppo(
            params,optimizer_state,random_key,
            data_batch,update_tick
        ):

            # GAE's lambda
            Glambda_fn=jax.vmap(rlax.lambda_returns)

            observations,actions,rewards,terminations,true_terminations,critic_preds,actor_preds=data_batch['observations'],data_batch['actions'], \
                                            data_batch['rewards'],data_batch['terminations'],data_batch['true_terminations'],data_batch['critic_preds'],data_batch['actor_preds']
            # use true terminations only for gamma (not combined done flag)
            gammas=self.gamma*(1-true_terminations)
            lambdas=self.gae_lambda*jnp.ones(self.num_envs)

            # calculate lambda returns (target returns) for timesteps G_{tick} - G_{tick+rollout_len}

            # rewards, gammas, lambdas values at timesteps {tick+1} - {tick+rollout_len+1}
            Glambdas = Glambda_fn(rewards[:,1:], gammas[:,1:], critic_preds[:,1:],lambdas)
            
            # calculate the advantages using timesteps {tick} - {tick+rollout_len}
            
            # td error
            #\delta_t = r_t + \gamma V( S_{t + 1} ) - V( S_{t} )

            # advantage function
            # A_t = G_{t}^{\lambda} - V(s_t)
            advantages = Glambdas - critic_preds[:,:-1]

            if self.is_continuous:
                mean, log_std = actor_preds
                mean = mean.squeeze(axis=2)
                log_std = log_std.squeeze(axis=2)
                # clip log_std
                log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
                std = jnp.exp(log_std)
                dist = squashed_gaussian(mean, std)
                logprobs = dist.log_prob(actions)
            else:
                B,T=actions.shape
                logprobs=jax.nn.log_softmax(actor_preds).reshape(B*T,-1)
                logprobs=logprobs[jnp.arange(B*T),actions.reshape(-1)].reshape(B,T)


            def ppo_loss(params, random_key, mb_observations, mb_actions,mb_terminations,
                            mb_logp, mb_advantages, mb_returns,mb_h_tickminus1):
                logits_new,values_new,_=self.actor_critic_fn(random_key,params,mb_observations,mb_terminations,
                                                             mb_h_tickminus1)

                if self.is_continuous:
                    mean, log_std = logits_new
                    # clip log_std
                    log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
                    std = jnp.exp(log_std)
                    dist = squashed_gaussian(mean, std)
                    newlogprobs = dist.log_prob(mb_actions)
                    # distrax.Transformed has no closed-form entropy (tanh has
                    # no analytic entropy correction); use the base Gaussian's
                    # entropy as the exploration-bonus proxy, as before.
                    entropy = dist.distribution.entropy()
                else:
                    B,T=mb_actions.shape
                    newlogprobs=jax.nn.log_softmax(logits_new).reshape(B*T,-1)
                    newlogprobs=newlogprobs[jnp.arange(B*T),mb_actions.reshape(-1)].reshape(B,T)
                    # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
                    logits_new = logits_new - jax.scipy.special.logsumexp(logits_new, axis=-1, keepdims=True)
                    logits_new = logits_new.clip(min=jnp.finfo(logits_new.dtype).min)
                    p_log_p = logits_new * jax.nn.softmax(logits_new)
                    entropy = -p_log_p.sum(-1)

                logratio = newlogprobs - mb_logp
                ratio = jnp.exp(logratio)
                approx_kl = ((ratio - 1) - logratio).mean()

                if self.norm_adv:
                    adv_mean = mb_advantages.mean()
                    adv_std = mb_advantages.std()
                    mb_advantages = (mb_advantages - adv_mean) / jnp.maximum(adv_std, 1e-4)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                # value loss
                v_loss = 0.5 * ((values_new - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_schedule(update_tick) * entropy_loss + v_loss * self.vf_coef
                return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

            ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)


            #use observations tick to tick+rollout_len
            observations=observations[:,:-1]
            terminations=terminations[:,:-1]
            hiddens=data_batch['hiddens']
            hidden_indices=data_batch['hidden_indices']
            num_seqs=hidden_indices.shape[1]

            #minibatch over num_envs and num_seqs

            def update_epoch(carry,x):
                params,optimizer_state,random_key=carry
                shuffle_key,model_key,random_key = jax.random.split(random_key,3)
                shuffled_inds = jax.random.permutation(shuffle_key, self.num_envs*num_seqs)
                batch_inds = shuffled_inds.reshape((self.num_minibatches, -1))
                def minibatch_update(carry,x):
                    params,optimizer_state,model_key=carry
                    batch_ind=x
                    mbenvinds=batch_ind//num_seqs
                    mbseqinds=batch_ind%num_seqs
                    model_key, _ = jax.random.split(model_key)
                    hidden_indices_mb=hidden_indices[mbenvinds,mbseqinds]
                    mb_h_tickminus1=jax.tree.map(lambda x:x[mbenvinds,mbseqinds],hiddens)

                    mb_observations=observations[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,observations.ndim)))
                    mb_actions=actions[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,actions.ndim)))
                    mb_terminations=terminations[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,terminations.ndim)))
                    mb_logp=logprobs[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,logprobs.ndim)))
                    mb_advantages=advantages[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,advantages.ndim)))
                    mb_returns=Glambdas[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,Glambdas.ndim)))
                    (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                         params,
                         model_key,
                         mb_observations,
                         mb_actions,
                         mb_terminations,
                         mb_logp,
                         mb_advantages,
                         mb_returns,
                         mb_h_tickminus1
                     )
                    # pre-clip grad norms, per group, before the optimizer's
                    # internal clip_by_global_norm rescales them.
                    actor_grad_norm = optax.global_norm(grads['params']['actor'])
                    critic_grad_norm = optax.global_norm(grads['params']['critic'])
                    updates,optimizer_state = self.optimizer.update(grads, optimizer_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params,optimizer_state,model_key),(loss, pg_loss, v_loss, entropy_loss, approx_kl, actor_grad_norm, critic_grad_norm)

                (params,optimizer_state,model_key),losses=jax.lax.scan(minibatch_update,(params,optimizer_state,model_key),batch_inds)
                losses=jax.tree.map(lambda x:x.mean(),losses)
                return (params,optimizer_state,random_key),losses


            (params,optimizer_state,random_key),losses=jax.lax.scan(update_epoch,(params,optimizer_state,random_key),jnp.arange(self.update_epochs))
            losses=jax.tree.map(lambda x:x.mean(),losses)
            loss, pg_loss, v_loss, entropy_loss, approx_kl, actor_grad_norm, critic_grad_norm=losses

            # fraction of *executed* actions sitting near the tanh saturation
            # bound, over the whole rollout just collected.
            action_saturation_frac = (jnp.abs(actions) > 0.95).mean()

            if self.is_continuous:
                mean, log_std = actor_preds
                mean = mean.squeeze(axis=2)
                log_std = log_std.squeeze(axis=2)
                log_std_clipped = jnp.clip(log_std, self.log_std_min, self.log_std_max)
                action_std = jnp.exp(log_std_clipped).mean()
                log_std_mean = log_std.mean()
                action_mean_abs = jnp.abs(mean).mean()
            else:
                action_std = None
                log_std_mean = None
                action_mean_abs = None

            return (loss, pg_loss, v_loss, entropy_loss, approx_kl, action_std, log_std_mean, action_mean_abs,
                     actor_grad_norm, critic_grad_norm, action_saturation_frac),params, optimizer_state
        self.update_ppo = update_ppo

        self._learn_jit = jax.jit(self._learn)
        self._train_step_jit = jax.jit(self._train_step)


    def _group_inject_states(self, optimizer_state=None):
        # self.optimizer is optax.multi_transform({'actor':...,'critic':...,
        # 'shared':...}, ...) so optimizer_state is a PartitionState whose
        # inner_states are each MaskedState(inner_state=(clip_state, inject_state)).
        if optimizer_state is None:
            optimizer_state = self.optimizer_state
        return {group: masked_state.inner_state[1]
                for group, masked_state in optimizer_state.inner_states.items()}

    def _current_lr(self, optimizer_state):
        # the rate actually sitting in the optimizer state, or None for an
        # optimizer without the per-group inject_hyperparams layout
        try:
            return next(iter(self._group_inject_states(optimizer_state).values())).hyperparams['learning_rate']
        except (AttributeError, IndexError, KeyError, TypeError, StopIteration):
            return None

    def _adapt_lr(self, optimizer_state, approx_kl):
        # adaptive LR based on KL divergence, applied on device
        lr = self._current_lr(optimizer_state)
        new_lr = jnp.where(approx_kl > self.kl_threshold * 2.0, jnp.maximum(lr / 1.5, self.lr_min),
                 jnp.where(approx_kl < self.kl_threshold * 0.5, jnp.minimum(lr * 1.5, self.lr_max), lr))
        new_lr = new_lr.astype(jnp.result_type(lr))

        new_inner_states = {}
        for group, masked_state in optimizer_state.inner_states.items():
            clip_state, inject_state = masked_state.inner_state
            new_hyperparams = dict(inject_state.hyperparams)
            new_hyperparams['learning_rate'] = new_lr
            new_inject_state = inject_state._replace(hyperparams=new_hyperparams)
            new_inner_states[group] = masked_state._replace(inner_state=(clip_state, new_inject_state))
        return optimizer_state._replace(inner_states=new_inner_states)

    def _learn(self, params, optimizer_state, random_key, databatch, update_tick):
        """pure PPO update on a collected batch + adaptive LR + diagnostics."""
        (loss, pg_loss, v_loss, entropy_loss, approx_kl, action_std, log_std_mean, action_mean_abs,
            actor_grad_norm, critic_grad_norm, action_saturation_frac), params, optimizer_state = self.update_ppo(
                params, optimizer_state, random_key, databatch, update_tick)

        if self.adaptive_lr:
            optimizer_state = self._adapt_lr(optimizer_state, approx_kl)

        # log the rate actually sitting in the optimizer state -- when
        # inject_hyperparams was built with a schedule callable, it silently
        # re-evaluated the schedule from its own step count every update()
        # and discarded any manual write. This reads back whatever the
        # optimizer will actually apply next step.
        diagnostics = {
            'action_std': action_std,
            'log_std_mean': log_std_mean,
            'action_mean_abs': action_mean_abs,
            'approx_kl': approx_kl,
            'current_lr': self._current_lr(optimizer_state),
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm,
            'action_saturation_frac': action_saturation_frac,
        }
        reward_sum = databatch['rewards'].sum()
        return (loss, (v_loss, entropy_loss, pg_loss, reward_sum, diagnostics)), params, optimizer_state, update_tick + 1

    def _train_step(self, params, optimizer_state, env_fstate, rollout_carry, random_key, update_tick):
        """one whole PPO iteration -- rollout, GAE, update epochs -- as a
        single pure function, so it compiles to one XLA program."""
        unroll_key, update_key = jax.random.split(random_key)
        env_fstate, rollout_carry, databatch, info_stats = self._scan_rollout(
            params, env_fstate, rollout_carry, unroll_key)
        out, params, optimizer_state, update_tick = self._learn(
            params, optimizer_state, update_key, databatch, update_tick)
        return out, info_stats, params, optimizer_state, env_fstate, rollout_carry, update_tick

    def reset(self,params_key,random_key):
        super(PPOAgent,self).reset(params_key,random_key)
        self.optimizer_state=self.optimizer.init(self.params)
        self.update_tick=jnp.array(0)

        if self.adaptive_lr:
            lr = self._current_lr(self.optimizer_state)
            self.current_lr = float(lr) if lr is not None else 3e-4

    def step(self,random_key):
        if self.functional_env:
            # fused: one dispatch for the whole iteration, no host sync
            (out, info_stats, self.params, self.optimizer_state, env_fstate,
                self.rollout_carry, self.update_tick) = self._train_step_jit(
                    self.params, self.optimizer_state, self.env.functional_state,
                    self.rollout_carry, random_key, self.update_tick)
            self.env.functional_state = env_fstate
            self.tick += self.rollout_len
        else:
            #unroll actor for rollout_len steps
            unroll_key,update_key=jax.random.split(random_key)
            databatch=vars(self.unroll_actors(unroll_key))
            info_stats=databatch.pop('info_stats')
            out, self.params, self.optimizer_state, self.update_tick = self._learn_jit(
                self.params, self.optimizer_state, update_key, databatch, self.update_tick)

        loss, (v_loss, entropy_loss, pg_loss, reward_sum, diagnostics) = out
        if self.adaptive_lr:
            self.current_lr = diagnostics['current_lr']
        return (loss,(v_loss,entropy_loss,pg_loss,reward_sum,diagnostics),info_stats)
