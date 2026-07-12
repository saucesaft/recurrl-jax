import jax
import jax.numpy as jnp
import optax
import rlax
import tqdm
import numpy as np
import logging
import distrax

from recurrl_jax.models.actor_critic import *
from typing import Callable
from recurrl_jax.utils.recurrent_utils import tree_index
from recurrl_jax.models.actor_critic import ActorCriticModel
from argparse import Namespace

logger = logging.getLogger(__name__)

def jax_to_numpy(*args):
    return jax.tree.map(lambda x: np.array(x),args)

def numpy_to_jax(*args,dtype=jnp.float32):
    return jax.tree.map(lambda x: jnp.array(x,dtype=dtype),args)


class BaseAgent:
    def __init__(self,train_envs,eval_env,rollout_len,repr_model_fn:Callable,seq_model_fn:Callable,
                        actor_fn:Callable,critic_fn:Callable,use_gumbel_sampling=False,sequence_length=None,
                        is_continuous=False, use_asymmetric_obs=True, policy_obs_dim=102) -> None:
        self.env=train_envs
        self.eval_env=eval_env
        self.rollout_len=rollout_len
        self.is_continuous=is_continuous
        if sequence_length is None:
            self.sequence_length=self.rollout_len
        else:
            assert rollout_len%sequence_length==0
            self.sequence_length=sequence_length
        self.seq_fn,self.seq_init=seq_model_fn
        self.use_gumbel_sampling=use_gumbel_sampling
        self.ac_model=nn.vmap(ActorCriticModel,
                              variable_axes={'params': None},
                                split_rngs={'params': False})(repr_model_fn,self.seq_fn,actor_fn,critic_fn,
                                                              use_asymmetric_obs=use_asymmetric_obs,
                                                              policy_obs_dim=policy_obs_dim)

        @jax.jit
        def actor_critic_fn(random_key,params,inputs,terminations,last_memory):
            act_logits,values,memory=self.ac_model.apply(params,inputs,terminations,last_memory,rngs={'random':random_key})
            return act_logits,values,memory

        self.actor_critic_fn=actor_critic_fn



    def reset(self,params_key,random_key):
        #reset the agent and initialize the parameters
        self.tick=0
        self.o_tick,_=self.env.reset()
        self.r_tick=jnp.zeros(self.env.num_envs)
        self.term_tick=jnp.full((self.env.num_envs),False)  # combined done flag (term OR trunc)
        self.true_term_tick=jnp.full((self.env.num_envs),False)  # true terminations only for GAE
        self.h_tickminus1=jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x,axis=0),self.env.num_envs,axis=0),self.seq_init())
        self._params=self.ac_model.init({'params':params_key,'random':random_key},jnp.expand_dims(self.o_tick,1),jnp.expand_dims(self.term_tick,1),
                                       self.h_tickminus1)
        def params_sum(params):
            return sum(jax.tree_util.tree_leaves(jax.tree.map(lambda x: np.prod(x.shape),params)))
        logger.info("Total Number of params: %d"%params_sum(self.params))
        seq_params = self.params['params'].get('seq_model', {})
        logger.info("Number of params in Seq Model: %d"%params_sum(seq_params))
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    def unroll_actors(self,random_key):
        #unrolls the actor for rollout_len steps, takes rollout_len actions
        num_seqs=self.rollout_len//self.sequence_length
        h_tickminus1=self.h_tickminus1
        o_tick=self.o_tick
        r_tick=self.r_tick
        term_tick=self.term_tick  # combined done flag (term OR trunc)
        true_term_tick=self.true_term_tick  # only true terminations
        actions=[]
        observations=[]
        rewards=[]
        critic_preds=[]
        actor_preds=[]
        terminations=[]  # combined done flag (term OR trunc) for RNN resets
        true_terminations=[]  # only true terminations for GAE bootstrap
        hiddens=[] #we still store the hidden states for every start
        hidden_indices=[] #to map hidden states to the correct timestep
        infos=[]
        for t in range(self.rollout_len):
            #add observation and reward and timestep tick
            observations.append(o_tick.copy())
            rewards.append(r_tick.copy())
            terminations.append(term_tick.copy())  # combined done for RNN
            true_terminations.append(true_term_tick.copy())  # true term only for GAE (carries from prev rollout)
            random_key,model_key=jax.random.split(random_key)
            #add hidden state for each sequence
            if t%self.sequence_length==0: #if it is time to update the hidden state
                #store the hidden state
                hiddens.append(jax.tree.map(lambda x:x,h_tickminus1))
                hidden_indices.append(jnp.repeat(jnp.arange(t,t+self.sequence_length).reshape(1,-1),repeats=self.env.num_envs,axis=0))


            act_logits,v_tick,htick=self.actor_critic_fn(model_key,self.params,jnp.expand_dims(o_tick,1),jnp.expand_dims(term_tick,1),
                                                         h_tickminus1)
            if self.is_continuous:
                mean, log_std = act_logits
                std = jnp.exp(log_std)
                dist = distrax.MultivariateNormalDiag(mean, std)
                acts_tick = dist.sample(seed=random_key).squeeze(axis=1)
            elif self.use_gumbel_sampling:
                # sample action: Gumbel-softmax trick
                # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
                u = jax.random.uniform(random_key, shape=act_logits.shape)
                acts_tick=jnp.argmax(act_logits - jnp.log(-jnp.log(u)), axis=-1).squeeze(axis=-1)
            else:
                acts_tick=jax.random.categorical(random_key,act_logits).squeeze(axis=-1)
            #take a step in the environment
            o_tickplus1,r_tickplus1,term_tickplus1,trunc_tickplus1,info=self.env.step(acts_tick)

            true_term_tickplus1 = term_tickplus1
            done_tickplus1 = jnp.logical_or(term_tickplus1, trunc_tickplus1)

            #add action at timestep tick
            critic_preds.append(v_tick.copy())
            actor_preds.append(jax.tree.map(lambda x: x.copy(), act_logits))
            actions.append(acts_tick.copy())
            infos.append(info)
            o_tick=o_tickplus1
            r_tick=r_tickplus1
            h_tickminus1=htick
            term_tick=done_tickplus1
            true_term_tick=true_term_tickplus1
            self.tick+=1
        #add the last observation and reward
        observations.append(o_tick)
        rewards.append(r_tick)
        terminations.append(term_tick)
        true_terminations.append(true_term_tick)
        #get the value for timestep (tick+rollout_len+1), needed for bootstrapping
        random_key,model_key=jax.random.split(random_key)
        _,v_tick,_=self.actor_critic_fn(model_key,self.params,jnp.expand_dims(o_tick,1),jnp.expand_dims(term_tick,1),h_tickminus1)
        critic_preds.append(v_tick)
        #update to timestep
        self.o_tick=o_tick.copy()
        self.r_tick=r_tick.copy()
        self.h_tickminus1=jax.tree.map(lambda x:x,h_tickminus1) #copy the hidden state, it can be arbitrary tree structure
        self.term_tick=term_tick.copy()
        self.true_term_tick=true_term_tick.copy()
        #shape is num_actorsXrollout_lenX*...
        hidden_stacked=jax.tree.map(lambda *args: jnp.stack(args,1), *hiddens)
        return Namespace(**{
            'observations':jnp.stack(observations,1),
            'actions':jnp.stack(actions,1),
            'rewards':jnp.stack(rewards,1),
            'terminations':jnp.stack(terminations,1),
            'true_terminations':jnp.stack(true_terminations,1),
            'infos':infos,
            'critic_preds':jnp.squeeze(jnp.stack(critic_preds,1),axis=-1),
            'actor_preds':jax.tree.map(lambda *xs: jnp.stack(xs, 1), *actor_preds),
            'hiddens':hidden_stacked,
            'hidden_indices':jnp.stack(hidden_indices,1)
        })


    def _eval_horizon(self):
        # discover max_episode_length by walking the eval_env wrapper chain
        env=self.eval_env
        for _ in range(6):
            if hasattr(env,'max_episode_length'):
                return int(env.max_episode_length)
            env=getattr(env,'env',None)
            if env is None:
                break
        return 1000

    def evaluate(self,random_key,eval_episodes):
        # vectorized, deterministic eval over a *batched* eval_env.
        #
        # runs fixed-horizon rollouts across all eval_env.num_envs in parallel
        # with NO per-step host sync. the previous impl looped one env, one step
        # at a time, converting done->python bool every step (device sync per
        # step); once episodes survived full length this was pathologically slow.
        # each env contributes one episode: reward is accumulated until that env
        # first *terminates* (a real drop), so survivors get full credit and
        # post-drop auto-reset steps are masked out. actions are deterministic
        # (mean / argmax) and return is the undiscounted episode sum, matching
        # the training return_per_episode metric.
        n_envs=self.eval_env.num_envs
        horizon=self._eval_horizon()

        def init_hidden():
            return jax.tree.map(lambda x:jnp.repeat(jnp.expand_dims(x,0),n_envs,axis=0),self.seq_init())

        all_lens=[]
        all_rets=[]
        n_rounds=max(1,-(-eval_episodes//n_envs))  # ceil, so we cover >= eval_episodes
        for _ in tqdm.tqdm(range(n_rounds)):
            o_tick,_=self.eval_env.reset()
            h_tickminus1=init_hidden()
            term=jnp.zeros((n_envs,1),dtype=bool)      # RNN reset signal (done)
            alive=jnp.ones((n_envs,),dtype=bool)       # still in first episode
            ep_ret=jnp.zeros((n_envs,),dtype=jnp.float32)
            ep_len=jnp.zeros((n_envs,),dtype=jnp.int32)
            for t in range(horizon):
                random_key,model_key=jax.random.split(random_key)
                act_logits,_,h_tickminus1=self.actor_critic_fn(model_key,self.params,jnp.expand_dims(o_tick,1),term,h_tickminus1)
                if self.is_continuous:
                    mean,_=act_logits
                    acts_tick=jnp.clip(mean.squeeze(axis=1),-1.0,1.0)
                else:
                    acts_tick=jnp.argmax(act_logits,axis=-1).squeeze(axis=1)
                o_tick,r_tick,term_step,trunc_step,info=self.eval_env.step(acts_tick)
                ep_ret=ep_ret+r_tick*alive.astype(jnp.float32)
                ep_len=ep_len+alive.astype(jnp.int32)
                alive=alive&jnp.logical_not(term_step)  # freeze contribution on real drop
                term=jnp.expand_dims(jnp.logical_or(term_step,trunc_step),1)
            all_lens.append(ep_len)
            all_rets.append(ep_ret)
        lens=jnp.concatenate(all_lens)
        avg_episode_len=lens.mean()
        avg_episode_return=jnp.concatenate(all_rets).mean()
        # success = survived the full horizon (never dropped). stashed for the
        # trainer to log; it's the paper metric (drop-saturated return is not).
        self.last_eval_success_rate=float((lens>=horizon).mean())
        return avg_episode_len,avg_episode_return,[]
