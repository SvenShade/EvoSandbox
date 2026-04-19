# INFO --------------------------------------------------------------------- #

# Authors: Steven Spratley
# Purpose: Extends InstaDeep's recurrent MAPPO trainer to allow rendering and
#          use custom critic observations from env state.
#          Originally from github.com/instadeepai/Mava
#
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPORTS ------------------------------------------------------------------ #

import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import copy
import time
import chex
import flax
import hydra
import imageio
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import cv2
import subprocess
from typing import Any, Tuple
from colorama import Fore, Style
from flax import linen as nn, struct
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import orthogonal
from jax import tree
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mava.evaluator import get_eval_fn, get_num_eval_envs, make_rec_eval_act_fn
from mava.networks import RecurrentActor as Actor
from mava.networks import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    OptStates,
    Params,
    RNNLearnerState,
    RNNPPOTransition,
)
from mava.types import (
    ExperimentOutput,
    LearnerFn,
    MarlEnv,
    Metrics,
    RecActorApply,
    RecCriticApply,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


@struct.dataclass
class RNNPPOTransitionCriticObs:
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    critic_obs: chex.Array
    hstates: HiddenStates


class ArrayRecurrentCritic(nn.Module):
    """Array-native recurrent critic for custom critic_obs tensors."""

    pre_torso: nn.Module
    post_torso: nn.Module
    hidden_state_dim: int

    @nn.compact
    def __call__(self, hidden_state: chex.Array, obs_done: Tuple[chex.Array, chex.Array]):
        obs, done = obs_done
        x = self.pre_torso(obs)
        hidden_state, x = ScannedRNN()(hidden_state, (x, done))
        x = self.post_torso(x)
        x = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return hidden_state, jnp.squeeze(x, axis=-1)


def unwrap_env_state(state: Any) -> Any:
    seen = set()
    while True:
        if hasattr(state, "critic_obs"):
            return state
        state_id = id(state)
        if state_id in seen:
            raise RuntimeError("Cycle detected while unwrapping env state.")
        seen.add(state_id)
        if hasattr(state, "env_state"):
            state = state.env_state
        elif hasattr(state, "state"):
            state = state.state
        else:
            raise AttributeError(
                f"Could not find base env state with `critic_obs`. Stopped at type {type(state).__name__}."
            )


def get_critic_obs(state: Any) -> chex.Array:
    return unwrap_env_state(state).critic_obs


matplotlib.use("Agg")  # head-less backend


# FUNCTIONS ---------------------------------------------------------------- #

def save_render(filename, frames, frate):
    h, w = frames[0].shape[:-1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, frate, (w, h))
    [out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) for f in frames]
    out.release()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            f"{filename}",
            "-i",
            "assets/ast_logo_white.png",
            "-filter_complex",
            "[1:v]scale=iw*0.12:ih*0.12[scaled_overlay];[0:v][scaled_overlay]overlay=0:H-h-10",
            "-preset",
            "slow",
            "-profile:v",
            "high",
            "-level:v",
            "4.0",
            "-vcodec",
            "libx264",
            "-an",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "30",
            "-r",
            "25",
            f"{filename[:-4]}_compressed.mp4",
        ],
        check=True,
    )


def transition_states(beg_state, end_state, num_transition=15, env_size=50):
    states = [copy.deepcopy(beg_state) for _ in range(num_transition)]
    mid_p = env_size * 0.5
    mid_state = beg_state.replace(
        p_pos=beg_state.p_pos.at[:, 2].set(jnp.ones_like(beg_state.p_pos[:, 2]) * mid_p)
    )
    mid_state = mid_state.replace(
        p_pos=mid_state.p_pos.at[:, 1].set(jnp.ones_like(mid_state.p_pos[:, 1]) * mid_p)
    )
    mid_state = mid_state.replace(
        p_pos=mid_state.p_pos.at[:, 0].set(jnp.ones_like(mid_state.p_pos[:, 0]) * mid_p)
    )
    blend = lambda beg, end, prog: jnp.stack(
        [(beg.p_pos[:, ax] * (1 - prog) + end.p_pos[:, ax] * prog) for ax in range(3)], axis=-1
    )
    for i in range(len(states)):
        prog = (i + 1) / len(states)
        beg = beg_state if prog < 0.5 else mid_state
        end = mid_state if prog < 0.5 else end_state
        states[i] = states[i].replace(p_pos=blend(beg, end, (prog * 2) % 1.001))
        states[i] = states[i].replace(
            pos_hist=(states[i].pos_hist * (1 - prog) + end_state.pos_hist * prog)
        )
        states[i] = states[i].replace(hmap=(states[i].hmap * (1 - prog) + end_state.hmap * prog))
        states[i] = states[i].replace(transition=True)
    return [*states, end_state]


def final_episodes(config: DictConfig, params: FrozenDict, actor_network: Actor) -> None:
    print(
        f"{Fore.CYAN}{Style.BRIGHT}\nRunning final episode with trained policy...{Style.RESET_ALL}"
    )
    render_config = copy.deepcopy(config)
    render_config.env.scenario.task_config = OmegaConf.merge(
        render_config.env.scenario.task_config,
        render_config.env.scenario.eval_config,
    )

    env, _ = environments.make(config=render_config, add_global_state=False)
    apply_fn = actor_network.apply
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(render_config.system.seed)
    states = []
    steps = []
    framerate = 25

    num_episodes = render_config.env.scenario.task_config.num_final_episodes
    pbar = tqdm(desc=f"Generating states for {num_episodes} final episodes")
    (key, *reset_keys) = jax.random.split(key, num_episodes + 1)
    for ep in range(num_episodes):
        state, timestep = reset_fn(reset_keys[ep], tot_steps=0)
        hidden_state = ScannedRNN.initialize_carry((1, env.num_agents), render_config.network.hidden_state_dim)
        done = jnp.zeros((1, env.num_agents), dtype=bool)
        step = 0
        while not timestep.last():
            step += 1
            key, action_key = jax.random.split(key)
            batched_obs = tree.map(lambda x: x[jnp.newaxis, ...], timestep.observation)
            hidden_state, pi = apply_fn(params, hidden_state, (batched_obs, done))
            if render_config.arch.evaluation_greedy:
                action = pi.mode()
            else:
                action = pi.sample(seed=action_key)
            action = action.squeeze(0)
            state, timestep = step_fn(state, action)
            done = timestep.last().repeat(env.num_agents).reshape(1, -1)
            states.append(unwrap_env_state(state))
            info = f"EPISODE {ep + 1}/{num_episodes}, TIMESTEP {step}."
            steps.append((info, False))
            pbar.update()
            if step == 1 and ep > 0:
                transition = transition_states(states[-2], states[-1], framerate, env_size=env.env_size)
                states.pop()
                steps.pop()
                states.extend(transition)
                steps.extend([(info, True)] * len(transition))
        states.pop()
        steps.pop()
    pbar.close()

    azim = 0.0
    deg_per_step = 0.5
    frames = []
    for state, (step, t) in tqdm(zip(states, steps), desc="Rendering"):
        frames.append(np.asarray(env.render_vtk(state, step, azim)).astype(np.uint8))
        azim += deg_per_step * (0 if t else 1)
    save_render("test.mp4", frames, framerate)
    # imageio.mimsave(fname+'.gif', frames, fps=frate, loop=0)
    print("Saved final episode animation.")


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[RecActorApply, RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RNNLearnerState]:
    """Get the learner function."""
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network."""

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, Tuple[RNNPPOTransitionCriticObs, Metrics]]:
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                last_hstates,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
            actor_in = (batched_observation, last_done[jnp.newaxis, :])
            critic_obs = get_critic_obs(env_state)
            critic_in = (critic_obs[jnp.newaxis, ...], last_done[jnp.newaxis, :])

            policy_hidden_state, actor_policy = actor_apply_fn(
                params.actor_params, last_hstates.policy_hidden_state, actor_in
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, last_hstates.critic_hidden_state, critic_in
            )

            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)
            value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            hstates = HiddenStates(policy_hidden_state, critic_hidden_state)
            transition = RNNPPOTransitionCriticObs(
                last_done,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                critic_obs,
                last_hstates,
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates
            )
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return learner_state, (transition, metrics)

        learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        params, opt_states, key, env_state, last_timestep, last_done, hstates = learner_state
        last_critic_in = (get_critic_obs(env_state)[jnp.newaxis, ...], last_done[jnp.newaxis, :])
        _, last_val = critic_apply_fn(params.critic_params, hstates.critic_hidden_state, last_critic_in)
        last_val = last_val.squeeze(0)

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: RNNPPOTransitionCriticObs,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    obs_and_done = (traj_batch.obs, traj_batch.done)
                    _, actor_policy = actor_apply_fn(
                        actor_params, traj_batch.hstates.policy_hidden_state[0], obs_and_done
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    if config.system.use_ppo_clipping:
                        actor_loss1 = ratio * gae
                        actor_loss2 = jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        ) * gae
                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                    else:
                        actor_loss = -(log_prob * gae).mean()
                    entropy = actor_policy.entropy(seed=key).mean()
                    total_loss = actor_loss - config.system.ent_coef * entropy
                    return total_loss, (actor_loss, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: RNNPPOTransitionCriticObs,
                    targets: chex.Array,
                ) -> Tuple:
                    critic_in = (traj_batch.critic_obs, traj_batch.done)
                    _, value = critic_apply_fn(
                        critic_params, traj_batch.hstates.critic_hidden_state[0], critic_in
                    )
                    if config.system.use_ppo_clipping:
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config.system.clip_eps, config.system.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * jnp.square(value - targets).mean()
                    total_loss = config.system.vf_coef * value_loss
                    return total_loss, value_loss

                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params,
                    traj_batch,
                    advantages,
                    entropy_key,
                )

                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="batch"
                )
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="device"
                )

                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                actor_loss, (_, entropy) = actor_loss_info
                value_loss, unscaled_value_loss = value_loss_info
                total_loss = actor_loss + value_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": unscaled_value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            batch = (traj_batch, advantages, targets)
            num_recurrent_chunks = config.system.rollout_length // config.system.recurrent_chunk_size
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            reshaped_batch = tree.map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = tree.map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
        )

        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
        )
        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: RNNLearnerState) -> ExperimentOutput[RNNLearnerState]:
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")
        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RNNLearnerState], Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    n_devices = len(jax.devices())
    num_agents = env.num_agents
    config.system.num_agents = num_agents
    key, actor_net_key, critic_net_key = keys

    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)

    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=config.network.hidden_state_dim,
    )
    critic_network = ArrayRecurrentCritic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
    )

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    init_obs = env.observation_spec.generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
    init_obs_done = (init_obs, init_done)

    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    key, init_env_key = jax.random.split(key)
    init_env_state, _ = env.reset(init_env_key, 0)
    init_critic_obs = get_critic_obs(init_env_state)[jnp.newaxis, jnp.newaxis, ...]
    init_critic_done = jnp.zeros((1, 1, num_agents), dtype=bool)
    init_critic_obs_done = (init_critic_obs, init_critic_done)
    init_critic_hstate_params = ScannedRNN.initialize_carry(
        (1, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_obs_done)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate_params, init_critic_obs_done)
    critic_opt_state = critic_optim.init(critic_params)

    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(env_keys), 0)
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    dones = jnp.zeros((config.arch.num_envs, num_agents), dtype=bool)
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, hstates, step_keys, dones)
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    params, opt_states, hstates, step_keys, dones = replicate_learner
    init_learner_state = RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=hstates,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "rec_mappo"
    config = copy.deepcopy(_config)
    eval_config = copy.deepcopy(_config)
    eval_config.env.scenario.task_config = OmegaConf.merge(
        eval_config.env.scenario.task_config,
        eval_config.env.scenario.eval_config,
    )

    n_devices = len(jax.devices())

    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert (
            config.system.rollout_length % config.system.recurrent_chunk_size == 0
        ), "Rollout length must be divisible by recurrent chunk size."
        assert (
            config.arch.num_envs % config.system.num_minibatches == 0
        ), "Number of envs must be divisibile by number of minibatches."

    env, _ = environments.make(config=config, add_global_state=False)
    _, eval_env = environments.make(config=eval_config, add_global_state=False)

    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_rec_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    params = unreplicate_batch_dim(learner_state.params.actor_params)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"{Fore.CYAN}{Style.BRIGHT}Actor params: {num_params}{Style.RESET_ALL}")

    best_params = None
    eval_performance = 0.0
    trained_params = unreplicate_batch_dim(learner_state.params.actor_params)

    if config.env.scenario.task_config.render_only:
        pass
    else:
        logger = MavaLogger(config)
        logger.log_config(OmegaConf.to_container(config, resolve=True))

        save_checkpoint = config.logger.checkpointing.save_model
        if save_checkpoint:
            checkpointer = Checkpointer(
                metadata=config,
                model_name=config.logger.system_name,
                **config.logger.checkpointing.save_args,
            )

        eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
        eval_num_agents = eval_env.num_agents
        eval_hs = ScannedRNN.initialize_carry(
            (n_devices, eval_batch_size, eval_num_agents),
            config.network.hidden_state_dim,
        )

        max_episode_return = -jnp.inf
        for eval_step in range(config.arch.num_evaluation):
            start_time = time.time()
            learner_output = learn(learner_state)
            jax.block_until_ready(learner_output)

            elapsed_time = time.time() - start_time
            t = int(steps_per_rollout * (eval_step + 1))
            episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
            episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

            logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
            if ep_completed:
                logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
            logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

            trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
            key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
            eval_keys = jnp.stack(eval_keys)
            eval_keys = eval_keys.reshape(n_devices, -1)
            eval_metrics = evaluator(trained_params, eval_keys, {"hidden_state": eval_hs})
            logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
            episode_return = jnp.mean(eval_metrics["episode_return"])

            if save_checkpoint:
                checkpointer.save(
                    timestep=steps_per_rollout * (eval_step + 1),
                    unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                    episode_return=episode_return,
                )

            if config.arch.absolute_metric and max_episode_return <= episode_return:
                best_params = copy.deepcopy(trained_params)
                max_episode_return = episode_return

            learner_state = learner_output.learner_state

        eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

        if config.arch.absolute_metric:
            eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
            eval_num_agents = eval_env.num_agents
            eval_hs = ScannedRNN.initialize_carry(
                (n_devices, eval_batch_size, eval_num_agents),
                config.network.hidden_state_dim,
            )
            abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
            eval_keys = jax.random.split(key, n_devices)
            eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": eval_hs})
            t = int(steps_per_rollout * (eval_step + 1))
            logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

        logger.stop()

    final_params = (
        best_params if config.arch.absolute_metric and best_params is not None else trained_params
    )
    final_params = unreplicate_n_dims(final_params, unreplicate_depth=1)
    final_episodes(config, final_params, actor_network)

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_mappo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent MAPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
