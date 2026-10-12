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
from recurrl_jax.utils.distributions import squashed_gaussian
from recurrl_jax.utils.wrappers import is_functional_env
from argparse import Namespace

logger = logging.getLogger(__name__)

def jax_to_numpy(*args):
    return jax.tree.map(lambda x: np.array(x),args)

def numpy_to_jax(*args,dtype=jnp.float32):
    return jax.tree.map(lambda x: jnp.array(x,dtype=dtype),args)


# info keys that are bookkeeping, not diagnostics
_SKIP_INFO_KEYS=('episode','done','termination')

def summarize_infos(infos):
    """reduce time-stacked step infos (leaves shaped (T, num_envs, ...)) to
    scalars, on device. per-env arrays become diag_{key}_mean (mean over
    steps and envs) and diag_{key}_max (per-step max, averaged over steps);
    per-step scalars are averaged under their own key. infos['episode'] (see
    VectorEpisodeStatisticsWrapper) becomes sums over finished episodes."""
    stats={}
    episode=infos.get('episode')
    if isinstance(episode,dict):
        done=episode['done'].astype(jnp.float32)
        stats['episode']={
            'count':done.sum(),
            'r_sum':(episode['r'].astype(jnp.float32)*done).sum(),
            'l_sum':(episode['l'].astype(jnp.float32)*done).sum(),
        }
    for key,val in infos.items():
        if key in _SKIP_INFO_KEYS or not isinstance(val,(jax.Array,np.ndarray)):
            continue
        if not (jnp.issubdtype(val.dtype,jnp.number) or val.dtype==jnp.bool_):
            continue
        val=jnp.asarray(val,dtype=jnp.float32)
        if val.ndim>1:
            stats[f'diag_{key}_mean']=val.mean()
            stats[f'diag_{key}_max']=val.reshape(val.shape[0],-1).max(axis=1).mean()
        else:
            stats[key]=val.mean()
    return stats

_summarize_infos_jit=jax.jit(summarize_infos)

def _stack_infos(infos):
    """stack a python list of per-step info dicts along a leading time axis,
    keeping only keys every step has and that hold numbers/arrays."""
    if not infos:
        return {}
    keys=set(infos[0]).intersection(*infos[1:])
    stacked={}
    for key in keys:
        vals=[info[key] for info in infos]
        try:
            if isinstance(vals[0],dict):
                stacked[key]=jax.tree.map(lambda *xs: jnp.stack(xs),*vals)
            elif all(isinstance(v,(int,float,np.number,np.ndarray,jax.Array)) for v in vals):
                stacked[key]=jnp.stack([jnp.asarray(v) for v in vals])
        except (TypeError,ValueError):
            pass  # ragged / non-numeric across steps: not loggable
    return stacked


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

        # fully-jitted rollouts when the env implements the functional
        # protocol (see recurrl_jax.utils.wrappers), python step loop otherwise
        self.functional_env=is_functional_env(self.env)
        self.functional_eval_env=eval_env is not None and is_functional_env(eval_env)
        self._policy_step_jit=jax.jit(self._policy_step)
        self._scan_rollout_jit=jax.jit(self._scan_rollout)
        self._eval_step_jit=jax.jit(self._eval_step,static_argnames=('stochastic',))
        self._eval_scan_jit=jax.jit(self._eval_scan,static_argnames=('horizon','stochastic'))
        if self.functional_env:
            logger.info("Env supports functional stepping: rollouts run as one jitted scan")
        else:
            logger.info("Env has no functional_step: rollouts use the python step loop")


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

    @property
    def rollout_carry(self):
        # agent-side state threaded from one rollout into the next
        return {'o':self.o_tick,'r':self.r_tick,'term':self.term_tick,
                'true_term':self.true_term_tick,'h':self.h_tickminus1}

    @rollout_carry.setter
    def rollout_carry(self,carry):
        self.o_tick=carry['o']
        self.r_tick=carry['r']
        self.term_tick=carry['term']
        self.true_term_tick=carry['true_term']
        self.h_tickminus1=carry['h']

    def _sample_actions(self,random_key,act_logits):
        if self.is_continuous:
            mean, log_std = act_logits
            std = jnp.exp(log_std)
            dist = squashed_gaussian(mean, std)
            # sample is already tanh-squashed into (-1, 1); this is what
            # gets executed AND what log_prob is later evaluated on, so
            # there's no train/eval action mismatch and no need to clip.
            return dist.sample(seed=random_key).squeeze(axis=1)
        elif self.use_gumbel_sampling:
            # sample action: Gumbel-softmax trick
            # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
            u = jax.random.uniform(random_key, shape=act_logits.shape)
            return jnp.argmax(act_logits - jnp.log(-jnp.log(u)), axis=-1).squeeze(axis=-1)
        else:
            return jax.random.categorical(random_key,act_logits).squeeze(axis=-1)

    def _policy_step(self,params,random_key,o_tick,term_tick,h_tickminus1):
        # one acting step for all envs: forward pass + action sampling
        model_key,sample_key=jax.random.split(random_key)
        act_logits,v_tick,htick=self.actor_critic_fn(model_key,params,jnp.expand_dims(o_tick,1),jnp.expand_dims(term_tick,1),
                                                     h_tickminus1)
        acts_tick=self._sample_actions(sample_key,act_logits)
        return acts_tick,act_logits,v_tick,htick

    def _assemble_batch(self,steps,carry,v_last,hiddens):
        # steps: per-step records shaped num_actorsXrollout_lenX*...; append
        # the final observation/reward/termination and the bootstrap value
        num_envs=carry['r'].shape[0]
        num_seqs=self.rollout_len//self.sequence_length
        append=lambda xs,x: jnp.concatenate([xs,jnp.expand_dims(x,1)],axis=1)
        hidden_indices=jnp.arange(self.rollout_len).reshape(num_seqs,self.sequence_length)
        return {
            'observations':append(steps['observations'],carry['o']),
            'actions':steps['actions'],
            'rewards':append(steps['rewards'],carry['r']),
            'terminations':append(steps['terminations'],carry['term']),
            'true_terminations':append(steps['true_terminations'],carry['true_term']),
            'critic_preds':jnp.squeeze(append(steps['critic_preds'],v_last),axis=-1),
            'actor_preds':steps['actor_preds'],
            'hiddens':hiddens,
            'hidden_indices':jnp.broadcast_to(hidden_indices,(num_envs,)+hidden_indices.shape),
        }

    def _scan_rollout(self,params,env_fstate,carry,random_key):
        """pure rollout of rollout_len steps over a functional env.

        returns (env_fstate, carry, batch, info_stats); the batch layout
        matches the python-loop unroll exactly."""
        num_seqs=self.rollout_len//self.sequence_length

        def env_step(c,step_key):
            env_fstate,carry=c
            acts_tick,act_logits,v_tick,htick=self._policy_step(params,step_key,carry['o'],carry['term'],carry['h'])
            env_fstate,o_tickplus1,r_tickplus1,term_tickplus1,trunc_tickplus1,info=self.env.functional_step(env_fstate,acts_tick)
            record={
                'observations':carry['o'],
                'rewards':carry['r'],
                'terminations':carry['term'],  # combined done for RNN
                'true_terminations':carry['true_term'],  # true term only for GAE
                'critic_preds':v_tick,
                'actor_preds':act_logits,
                'actions':acts_tick,
                'infos':info,
            }
            carry={
                'o':o_tickplus1.astype(carry['o'].dtype),
                'r':r_tickplus1.astype(carry['r'].dtype),
                'term':jnp.logical_or(term_tickplus1,trunc_tickplus1),
                'true_term':jnp.asarray(term_tickplus1,dtype=bool),
                'h':htick,
            }
            return (env_fstate,carry),record

        def seq_step(c,seq_keys):
            # store the hidden state at the start of every sequence
            h_start=c[1]['h']
            c,records=jax.lax.scan(env_step,c,seq_keys)
            return c,(records,h_start)

        step_keys=jax.random.split(random_key,self.rollout_len+1)
        seq_keys=step_keys[:-1].reshape((num_seqs,self.sequence_length)+step_keys.shape[1:])
        (env_fstate,carry),(records,hiddens)=jax.lax.scan(seq_step,(env_fstate,carry),seq_keys)

        # (num_seqs, sequence_length, num_envs, ...) -> (rollout_len, num_envs, ...)
        records=jax.tree.map(lambda x: x.reshape((self.rollout_len,)+x.shape[2:]),records)
        info_stats=summarize_infos(records.pop('infos'))
        # time-major -> num_actorsXrollout_lenX*...
        records=jax.tree.map(lambda x: jnp.swapaxes(x,0,1),records)
        hiddens=jax.tree.map(lambda x: jnp.swapaxes(x,0,1),hiddens)

        #get the value for timestep (tick+rollout_len+1), needed for bootstrapping
        _,v_last,_=self.actor_critic_fn(step_keys[-1],params,jnp.expand_dims(carry['o'],1),jnp.expand_dims(carry['term'],1),carry['h'])
        return env_fstate,carry,self._assemble_batch(records,carry,v_last,hiddens),info_stats

    def unroll_actors(self,random_key):
        #unrolls the actor for rollout_len steps, takes rollout_len actions
        if self.functional_env:
            env_fstate,carry,batch,info_stats=self._scan_rollout_jit(self.params,self.env.functional_state,
                                                                    self.rollout_carry,random_key)
            self.env.functional_state=env_fstate
            self.rollout_carry=carry
            self.tick+=self.rollout_len
            return Namespace(**batch,info_stats=info_stats)

        carry=self.rollout_carry
        steps={k:[] for k in ('observations','rewards','terminations','true_terminations',
                              'critic_preds','actor_preds','actions')}
        hiddens=[] #we still store the hidden states for every start
        infos=[]
        # same key schedule as _scan_rollout
        step_keys=jax.random.split(random_key,self.rollout_len+1)
        for t in range(self.rollout_len):
            step_key=step_keys[t]
            #add hidden state for each sequence
            if t%self.sequence_length==0:
                hiddens.append(carry['h'])
            acts_tick,act_logits,v_tick,htick=self._policy_step_jit(self.params,step_key,carry['o'],carry['term'],carry['h'])
            #take a step in the environment
            o_tickplus1,r_tickplus1,term_tickplus1,trunc_tickplus1,info=self.env.step(acts_tick)

            #add observation, reward and terminations at timestep tick
            steps['observations'].append(carry['o'])
            steps['rewards'].append(carry['r'])
            steps['terminations'].append(carry['term'])  # combined done for RNN
            steps['true_terminations'].append(carry['true_term'])  # true term only for GAE (carries from prev rollout)
            steps['critic_preds'].append(v_tick)
            steps['actor_preds'].append(act_logits)
            steps['actions'].append(acts_tick)
            infos.append(info)
            carry={
                'o':o_tickplus1,
                'r':r_tickplus1,
                'term':jnp.logical_or(term_tickplus1,trunc_tickplus1),
                'true_term':jnp.asarray(term_tickplus1),
                'h':htick,
            }
            self.tick+=1
        #get the value for timestep (tick+rollout_len+1), needed for bootstrapping
        _,v_last,_=self.actor_critic_fn(step_keys[-1],self.params,jnp.expand_dims(carry['o'],1),jnp.expand_dims(carry['term'],1),carry['h'])
        self.rollout_carry=carry
        #shape is num_actorsXrollout_lenX*...
        steps={k:jax.tree.map(lambda *xs: jnp.stack(xs,1),*v) for k,v in steps.items()}
        hidden_stacked=jax.tree.map(lambda *args: jnp.stack(args,1), *hiddens)
        batch=self._assemble_batch(steps,carry,v_last,hidden_stacked)
        return Namespace(**batch,info_stats=_summarize_infos_jit(_stack_infos(infos)))


    def _eval_horizon(self):
        # discover the episode horizon by walking the eval_env wrapper chain.
        # BraxG1Env exposes it as max_episode_steps; keep max_episode_length
        # too for any other env that names it that way.
        env=self.eval_env
        for _ in range(6):
            for attr in ('max_episode_steps','max_episode_length'):
                if hasattr(env,attr):
                    return int(getattr(env,attr))
            env=getattr(env,'env',None)
            if env is None:
                break
        return 1000

    def _eval_actions(self,act_logits,noise_key,stochastic):
        if self.is_continuous:
            mean,log_std=act_logits
            if stochastic:
                # evaluate the policy as trained (sampled actions).
                # contact-rich tasks can learn noise-dependent behavior
                # where the bare mean is a different, worse controller;
                # selecting checkpoints by mean-eval then freezes early.
                std=jnp.exp(log_std)
                dist=squashed_gaussian(mean,std)
                return dist.sample(seed=noise_key).squeeze(axis=1)
            # deterministic action = tanh(mean), no clip needed:
            # tanh already bounds strictly within (-1, 1).
            return jnp.tanh(mean.squeeze(axis=1))
        return jnp.argmax(act_logits,axis=-1).squeeze(axis=1)

    def _eval_step(self,params,step_key,o_tick,term,h_tickminus1,stochastic):
        model_key,noise_key=jax.random.split(step_key)
        act_logits,_,h_tickminus1=self.actor_critic_fn(model_key,params,jnp.expand_dims(o_tick,1),term,h_tickminus1)
        return self._eval_actions(act_logits,noise_key,stochastic),h_tickminus1

    def _eval_scan(self,params,env_fstate,o_tick,random_key,horizon,stochastic):
        """pure fixed-horizon eval rollout over a functional eval env."""
        n_envs=o_tick.shape[0]
        h_tickminus1=jax.tree.map(lambda x:jnp.repeat(jnp.expand_dims(x,0),n_envs,axis=0),self.seq_init())

        def body(c,step_key):
            env_fstate,o_tick,term,h_tickminus1,alive,ep_ret,ep_len=c
            acts_tick,h_tickminus1=self._eval_step(params,step_key,o_tick,term,h_tickminus1,stochastic)
            env_fstate,o_next,r_tick,term_step,trunc_step,_=self.eval_env.functional_step(env_fstate,acts_tick)
            ep_ret=ep_ret+r_tick*alive.astype(jnp.float32)
            ep_len=ep_len+alive.astype(jnp.int32)
            alive=alive&jnp.logical_not(term_step)  # freeze contribution on real drop
            term=jnp.expand_dims(jnp.logical_or(term_step,trunc_step),1)
            return (env_fstate,o_next.astype(o_tick.dtype),term,h_tickminus1,alive,ep_ret,ep_len),None

        c=(env_fstate,o_tick,jnp.zeros((n_envs,1),dtype=bool),h_tickminus1,
           jnp.ones((n_envs,),dtype=bool),jnp.zeros((n_envs,),dtype=jnp.float32),jnp.zeros((n_envs,),dtype=jnp.int32))
        (env_fstate,_,_,_,_,ep_ret,ep_len),_=jax.lax.scan(body,c,jax.random.split(random_key,horizon))
        return env_fstate,ep_len,ep_ret

    def evaluate(self,random_key,eval_episodes):
        # vectorized eval over a *batched* eval_env.
        #
        # runs fixed-horizon rollouts across all eval_env.num_envs in parallel
        # with NO per-step host sync (one jitted scan per round when the eval
        # env is functional). each env contributes one episode: reward is
        # accumulated until that env first *terminates* (a real drop), so
        # survivors get full credit and post-drop auto-reset steps are masked
        # out. actions are deterministic (mean / argmax) unless
        # self.eval_stochastic is set, and return is the undiscounted episode
        # sum, matching the training return_per_episode metric.
        n_envs=self.eval_env.num_envs
        horizon=self._eval_horizon()
        stochastic=bool(getattr(self,'eval_stochastic',False))

        all_lens=[]
        all_rets=[]
        n_rounds=max(1,-(-eval_episodes//n_envs))  # ceil, so we cover >= eval_episodes
        for _ in tqdm.tqdm(range(n_rounds)):
            random_key,round_key=jax.random.split(random_key)
            o_tick,_=self.eval_env.reset()
            if self.functional_eval_env:
                env_fstate,ep_len,ep_ret=self._eval_scan_jit(self.params,self.eval_env.functional_state,o_tick,round_key,
                                                             horizon=horizon,stochastic=stochastic)
                self.eval_env.functional_state=env_fstate
            else:
                h_tickminus1=jax.tree.map(lambda x:jnp.repeat(jnp.expand_dims(x,0),n_envs,axis=0),self.seq_init())
                term=jnp.zeros((n_envs,1),dtype=bool)      # RNN reset signal (done)
                alive=jnp.ones((n_envs,),dtype=bool)       # still in first episode
                ep_ret=jnp.zeros((n_envs,),dtype=jnp.float32)
                ep_len=jnp.zeros((n_envs,),dtype=jnp.int32)
                for step_key in jax.random.split(round_key,horizon):
                    acts_tick,h_tickminus1=self._eval_step_jit(self.params,step_key,o_tick,term,h_tickminus1,stochastic=stochastic)
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
