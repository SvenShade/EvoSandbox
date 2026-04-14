# INFO --------------------------------------------------------------------- #

# Authors: Steven Spratley
# Purpose: Extends InstaDeep's MAPPO trainer to allow rendering.
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
import imageio
import numpy as np
import chex
import flax
from flax import struct, linen as nn
from flax.linen.initializers import orthogonal
import hydra
import jax
import jax.numpy as jnp
import optax
import matplotlib
import cv2
import subprocess
from typing import Any, Tuple
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.networks import FeedForwardActor as Actor
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from mava.types import ActorApply, CriticApply, ExperimentOutput, LearnerFn, MarlEnv, Metrics
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import merge_leading_dims, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics

@struct.dataclass
class PPOTransitionCriticObs:
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    critic_obs: chex.Array


class ArrayCritic(nn.Module):
    """Minimal array-native critic for custom critic_obs tensors."""
    torso: nn.Module

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        x = self.torso(observation)
        x = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return jnp.squeeze(x, axis=-1)


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


matplotlib.use("Agg") # head-less backend

# FUNCTIONS ---------------------------------------------------------------- #

def save_render(filename, frames, frate):
    h, w = frames[0].shape[:-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, frate, (w, h))
    [out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) for f in frames]
    out.release()
    # Compress saved file with ffmpeg. Add AST logo.
    subprocess.run(["ffmpeg","-y",
                    "-i",f"{filename}",
                    "-i","assets/ast_logo_white.png",
                    "-filter_complex",
                    "[1:v]scale=iw*0.12:ih*0.12[scaled_overlay];[0:v][scaled_overlay]overlay=0:H-h-10",                    
                    "-preset","slow",
                    "-profile:v","high",
                    "-level:v","4.0",
                    "-vcodec","libx264",
                    "-an",
                    "-pix_fmt","yuv420p",
                    "-crf","30",
                    "-r","25",
                    f"{filename[:-4]}_compressed.mp4"], check=True)

def transition_states(beg_state, end_state, num_transition=15, env_size=50):
    states = [copy.deepcopy(beg_state) for i in range(num_transition)]
    mid_p = env_size * 0.5
    mid_state = beg_state.replace(
        p_pos=beg_state.p_pos.at[:, 2].set(
            jnp.ones_like(beg_state.p_pos[:, 2]) * mid_p))
    mid_state = mid_state.replace(
        p_pos=mid_state.p_pos.at[:, 1].set(
            jnp.ones_like(mid_state.p_pos[:, 1]) * mid_p))
    mid_state = mid_state.replace(
        p_pos=mid_state.p_pos.at[:, 0].set(
            jnp.ones_like(mid_state.p_pos[:, 0]) * mid_p))
    blend = lambda beg, end, prog : jnp.stack(
        [(beg.p_pos[:, ax] * (1 - prog) + end.p_pos[:, ax] * prog)
        for ax in range(3)], axis=-1)
    for i in range(len(states)):
        prog = (i + 1) / len(states)
        beg = beg_state if prog < 0.5 else mid_state
        end = mid_state if prog < 0.5 else end_state
        states[i] = states[i].replace(p_pos=blend(beg, end, (prog * 2) % 1.001))
        states[i] = states[i].replace(
            pos_hist=(states[i].pos_hist * (1 - prog) + end_state.pos_hist * prog))
        states[i] = states[i].replace(
            hmap=(states[i].hmap * (1 - prog) + end_state.hmap * prog))
        states[i] = states[i].replace(transition=True)
    return [*states, end_state]

def final_episodes(config: DictConfig,
                  params: FrozenDict,
                  actor_network: Actor) -> None:
    print(f"{Fore.CYAN}{Style.BRIGHT}\nRunning final episode with trained policy...{Style.RESET_ALL}")
    # Create a single environment for the final episodes. Use render-specific arguments.
    config.env.scenario.task_config = OmegaConf.merge(config.env.scenario.task_config, config.env.scenario.eval_config)
    env, _ = environments.make(config=config, add_global_state=False)
    apply_fn = actor_network.apply
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(config.system.seed)
    states = []
    steps = []
    framerate = 25

    # Step through episode/s.
    num_episodes = config.env.scenario.task_config.num_final_episodes
    pbar = tqdm(desc=f'Generating states for {num_episodes} final episodes')
    (key, *reset_keys) = jax.random.split(key, num_episodes + 1)
    for ep in range(num_episodes):
        state, timestep = reset_fn(reset_keys[ep], tot_steps=0)
        step = 0
        while not timestep.last():
            step += 1
            key, action_key = jax.random.split(key, )
            # Add a batch dimension to the observation before passing it to the network.
            batched_obs = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, ...], timestep.observation
            )
            pi = apply_fn(params, batched_obs)
            if config.arch.evaluation_greedy:
                action = pi.mode()
            else:
                action = pi.sample(seed=action_key)

            # Remove the batch dimension from the action before passing it to the environment.
            action = action.squeeze(0)
            state, timestep = step_fn(state, action)
            states.append(unwrap_env_state(state))
            info = f'EPISODE {ep + 1}/{num_episodes}, TIMESTEP {step}.'
            steps.append((info, False))
            pbar.update()

            # If another episode has just begun (not including the first), create
            # transition states to morph the terrain (for aesthetic purposes).
            if step == 1 and ep > 0:
                transition = transition_states(
                    states[-2], states[-1], framerate,
                    env_size=env.env_size)
                states.pop()
                steps.pop()
                states.extend(transition)
                steps.extend([(info, True)] * len(transition))
        states.pop() # Workaround for bug causing error in state at last timestep.
        steps.pop()
    pbar.close()

    # Render states and write animation to disk.
    azim = 0.0 # Camera azimuth for perspective render.
    deg_per_step = 0.5 # Degrees per step to shift camera azim.
    frames = []
    for state, (step, t) in tqdm(zip(states, steps), desc='Rendering'):
        frames.append(np.asarray(env.render_vtk(state, step, azim)).astype(np.uint8))
        azim += deg_per_step * (0 if t else 1)
    save_render('test.mp4', frames, framerate)
    # imageio.mimsave(fname+'.gif', frames, fps=frate, loop=0)

    print("Saved final episode animation.")


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""
    # Unpack apply and update functions.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: LearnerState, _: Any
        ) -> Tuple[LearnerState, Tuple[PPOTransitionCriticObs, Metrics]]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep, last_done = learner_state

            # Select action
            key, policy_key = jax.random.split(key)
            critic_obs = get_critic_obs(env_state)
            actor_policy = actor_apply_fn(params.actor_params, last_timestep.observation)
            value = critic_apply_fn(params.critic_params, critic_obs)
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # Step environment
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)

            transition = PPOTransitionCriticObs(
                last_done, action, value, timestep.reward, log_prob, last_timestep.observation, critic_obs
            )
            learner_state = LearnerState(params, opt_states, key, env_state, timestep, done)
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return learner_state, (transition, metrics)

        # Step environment for rollout length
        learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done = learner_state
        last_val = critic_apply_fn(params.critic_params, get_critic_obs(env_state))

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOTransitionCriticObs,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Rerun network
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Nomalise advantage at minibatch level
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    if config.system.use_ppo_clipping:
                        # PPO clipped surrogate objective.
                        actor_loss1 = ratio * gae
                        actor_loss2 = jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        ) * gae
                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                    else:
                        # IA2C/A2C: vanilla policy-gradient loss (on-policy).
                        # Note: ratio is kept for logging/diagnostics; it is not used in the objective.
                        actor_loss = -(log_prob * gae).mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_actor_loss = actor_loss - config.system.ent_coef * entropy
                    return total_actor_loss, (actor_loss, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransitionCriticObs,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    value = critic_apply_fn(critic_params, traj_batch.critic_obs)

                    # MSE loss
                    if config.system.use_ppo_clipping:
                        # PPO-style clipped value loss.
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config.system.clip_eps, config.system.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    else:
                        # IA2C/A2C: standard MSE value regression.
                        value_loss = 0.5 * jnp.square(value - targets).mean()

                    total_value_loss = config.system.vf_coef * value_loss
                    return total_value_loss, value_loss

                # Calculate actor loss
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params,
                    traj_batch,
                    advantages,
                    entropy_key,
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="device"
                )

                # Update params and optimiser state
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

            # Shuffle minibatches
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                shuffled_batch,
            )

            # Update minibatches
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = LearnerState(params, opt_states, key, env_state, last_timestep, last_done)
        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.

        """
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
) -> Tuple[LearnerFn[LearnerState], Actor, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = ArrayCritic(torso=critic_torso)

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

    # Initialise observation with obs of all agents.
    obs = env.observation_spec.generate_value()
    init_x = tree.map(lambda x: x[jnp.newaxis, ...], obs)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = Params(actor_params, critic_params)

    # Pack apply and update functions.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(env_keys), 0)

    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, config.system.num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, step_keys, dones = replicate_learner
    init_learner_state = LearnerState(params, opt_states, step_keys, env_states, timesteps, dones)

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "ff_mappo"
    config      = copy.deepcopy(_config)
    eval_config = copy.deepcopy(_config)
    eval_config.env.scenario.task_config = OmegaConf.merge(
        eval_config.env.scenario.task_config,
        eval_config.env.scenario.eval_config,
    )
    n_devices = len(jax.devices())

    # Create the enviroments for train and eval.
    env, _      = environments.make(config=config, add_global_state=False)
    _, eval_env = environments.make(config=eval_config, add_global_state=False)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    assert (
        config.arch.num_envs % config.system.num_minibatches == 0
    ), "Number of envs must be divisibile by number of minibatches."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Print num params in actor network.
    params = unreplicate_batch_dim(learner_state.params.actor_params)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"{Fore.CYAN}{Style.BRIGHT}Actor params: {num_params}{Style.RESET_ALL}")

    # Skip training and eval steps if only tasked with rendering an episode.
    best_params = None
    eval_performance = 0.0
    if config.env.scenario.task_config.render_only:
        # TODO: Load trained params.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)

    else:
        # Logger setup
        logger = MavaLogger(config)
        logger.log_config(OmegaConf.to_container(config, resolve=True))

        # Set up checkpointer
        save_checkpoint = config.logger.checkpointing.save_model
        if save_checkpoint:
            checkpointer = Checkpointer(
                metadata=config,  # Save all config as metadata in the checkpoint
                model_name=config.logger.system_name,
                **config.logger.checkpointing.save_args,  # Checkpoint args
            )

        # Run experiment for a total number of evaluations.
        max_episode_return = -jnp.inf
        for eval_step in range(config.arch.num_evaluation):
            # Train.
            start_time = time.time()

            learner_output = learn(learner_state)
            jax.block_until_ready(learner_output)

            # Log the results of the training.
            elapsed_time = time.time() - start_time
            t = int(steps_per_rollout * (eval_step + 1))
            episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
            episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

            # Separately log timesteps, actoring metrics and training metrics.
            logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
            if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
                logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
            logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

            # Prepare for evaluation.
            trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
            key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
            eval_keys = jnp.stack(eval_keys)
            eval_keys = eval_keys.reshape(n_devices, -1)
            # Evaluate.
            eval_metrics = evaluator(trained_params, eval_keys, {})
            logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
            episode_return = jnp.mean(eval_metrics["episode_return"])

            if save_checkpoint:
                # Save checkpoint of learner state
                checkpointer.save(
                    timestep=steps_per_rollout * (eval_step + 1),
                    unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                    episode_return=episode_return,
                )

            if config.arch.absolute_metric and max_episode_return <= episode_return:
                best_params = copy.deepcopy(trained_params)
                max_episode_return = episode_return

            # Update runner state to continue training.
            learner_state = learner_output.learner_state

        # Record the performance for the final evaluation run.
        eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

        # Measure absolute metric.
        if config.arch.absolute_metric:
            abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
            eval_keys = jax.random.split(key, n_devices)

            eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})

            t = int(steps_per_rollout * (eval_step + 1))
            logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

        # Stop the logger.
        logger.stop()

    # Pass trained params to final_episodes and render GIF.
    # Unreplicate the device dimension before passing to the non-pmapped function.
    final_params = (
        best_params
        if config.arch.absolute_metric and best_params is not None
        else trained_params
    )
    final_params = unreplicate_n_dims(final_params, unreplicate_depth=1)
    final_episodes(config, final_params, actor_network)

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_mappo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}MAPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()


# END FILE ----------------------------------------------------------------- #
