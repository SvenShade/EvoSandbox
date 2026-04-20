# INFO --------------------------------------------------------------------- #

# Authors: Steven Spratley
# Purpose: MAPPO with custom critic observations, Team-DIAYN intrinsic rewards,
#          and a ComSD-style exploration term.
#          Based on a modified Mava feed-forward MAPPO trainer.

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

matplotlib.use("Agg")


@struct.dataclass
class PPOTransitionTeamDIAYN:
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: Any
    critic_obs: chex.Array
    team_summary: chex.Array
    team_skill: chex.Array
    skill_reward: chex.Array
    expl_reward: chex.Array


@struct.dataclass
class TeamDIAYNParams:
    actor_params: FrozenDict
    critic_params: FrozenDict
    disc_params: FrozenDict


@struct.dataclass
class TeamDIAYNOptStates:
    actor_opt_state: Any
    critic_opt_state: Any
    disc_opt_state: Any


@struct.dataclass
class TeamDIAYNLearnerState:
    params: TeamDIAYNParams
    opt_states: TeamDIAYNOptStates
    key: chex.Array
    env_state: Any
    timestep: Any
    dones: chex.Array


class ArrayCritic(nn.Module):
    """Minimal array-native critic for custom critic_obs tensors."""
    torso: nn.Module

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        x = self.torso(observation)
        x = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return jnp.squeeze(x, axis=-1)


class TeamDiscriminator(nn.Module):
    """Predict team skill z from team-level behavior summaries."""
    num_skills: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, summary: chex.Array) -> chex.Array:
        x = summary.astype(jnp.float32)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        return nn.Dense(self.num_skills, kernel_init=orthogonal(0.01))(x)


def _cfg_get(node: DictConfig, key: str, default: Any) -> Any:
    return node[key] if key in node else default


def unwrap_env_state(state: Any) -> Any:
    seen = set()
    while True:
        if hasattr(state, "critic_obs"):
            return state
        sid = id(state)
        if sid in seen:
            raise RuntimeError("Cycle detected while unwrapping env state.")
        seen.add(sid)
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
    print(f"{Fore.CYAN}{Style.BRIGHT}\nRunning final episode with trained policy...{Style.RESET_ALL}")
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
        step = 0
        while not timestep.last():
            step += 1
            key, action_key = jax.random.split(key)
            batched_obs = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], timestep.observation)
            pi = apply_fn(params, batched_obs)
            action = pi.mode() if render_config.arch.evaluation_greedy else pi.sample(seed=action_key)
            action = action.squeeze(0)
            state, timestep = step_fn(state, action)
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
    print("Saved final episode animation.")


def build_team_helpers(env: MarlEnv):
    team_ids = jnp.asarray(env.team_ids, dtype=jnp.int32)
    num_teams = int(team_ids.max()) + 1
    team_masks = jnp.stack([(team_ids == t).astype(jnp.float32) for t in range(num_teams)], axis=0)
    team_sizes = jnp.maximum(team_masks.sum(axis=-1), 1.0)
    env_size = float(getattr(env, "env_size", 1.0))
    max_speed = float(getattr(env, "max_speed", 1.0))
    init_battery = float(getattr(env, "init_battery", 1.0))
    num_skills = int(getattr(env, "dim_a", 3))

    def _ensure_batch(x: chex.Array, ndims_no_batch: int) -> chex.Array:
        return x[jnp.newaxis, ...] if x.ndim == ndims_no_batch else x

    def extract_team_summary(state: Any) -> Tuple[chex.Array, chex.Array, chex.Array]:
        s = unwrap_env_state(state)
        pos = _ensure_batch(s.p_pos.astype(jnp.float32), 2) / max(env_size, 1.0)
        vel = _ensure_batch(s.p_vel.astype(jnp.float32), 2) / max(max_speed, 1.0)
        done = _ensure_batch(s.done.astype(jnp.float32), 1)
        batt = _ensure_batch(s.batt.astype(jnp.float32), 1) / max(init_battery, 1.0)
        g_hits = _ensure_batch(s.g_hits.astype(jnp.float32), 1)
        g_kills = _ensure_batch(s.g_kills.astype(jnp.float32), 1)
        expl = _ensure_batch(s.expl_map.astype(jnp.float32), 3)[:, :num_teams]
        last_expl = _ensure_batch(s.last_expl_map.astype(jnp.float32), 3)[:, :num_teams]
        op_att = _ensure_batch(s.op_att.astype(jnp.float32), 2)[:, :num_teams, :num_skills]

        alive = 1.0 - done
        mask = team_masks[None, :, :] * alive[:, None, :]
        counts = jnp.maximum(mask.sum(axis=-1, keepdims=True), 1.0)

        centroid = jnp.einsum("btn,bnd->btd", mask, pos) / counts
        mean_vel = jnp.einsum("btn,bnd->btd", mask, vel) / counts
        centered = pos[:, None, :, :] - centroid[:, :, None, :]
        dispersion = (jnp.linalg.norm(centered, axis=-1) * mask).sum(axis=-1, keepdims=True) / counts
        alive_frac = counts / team_sizes[None, :, None]
        batt_mean = jnp.einsum("btn,bn->bt", mask, batt)[:, :, None] / counts
        hits_mean = jnp.einsum("btn,bn->bt", mask, g_hits)[:, :, None] / counts
        kills_mean = jnp.einsum("btn,bn->bt", mask, g_kills)[:, :, None] / counts
        coverage = expl.mean(axis=(-1, -2))[:, :, None]
        delta_cov = jnp.maximum(expl - last_expl, 0.0).mean(axis=(-1, -2))[:, :, None]

        summary = jnp.concatenate(
            [
                centroid,
                mean_vel,
                dispersion,
                alive_frac,
                batt_mean,
                hits_mean,
                kills_mean,
                coverage,
                delta_cov,
            ],
            axis=-1,
        )
        skill = jnp.argmax(op_att, axis=-1).astype(jnp.int32)
        return summary, skill, delta_cov[..., 0]

    return team_ids, num_teams, num_skills, extract_team_summary


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[ActorApply, CriticApply, Any],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[TeamDIAYNLearnerState]:
    actor_apply_fn, critic_apply_fn, disc_apply_fn = apply_fns
    actor_update_fn, critic_update_fn, disc_update_fn = update_fns

    team_ids, num_teams, num_skills, extract_team_summary = build_team_helpers(env)
    skill_coef = float(_cfg_get(config.system, "skill_coef", 1.0))
    exploration_coef = float(_cfg_get(config.system, "exploration_coef", 0.5))
    disc_coef = float(_cfg_get(config.system, "disc_coef", 1.0))

    def compute_intrinsic(disc_params: FrozenDict, env_state: Any) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        team_summary, team_skill, delta_cov = extract_team_summary(env_state)
        logits = disc_apply_fn(disc_params, team_summary)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        skill_logp = jnp.take_along_axis(log_probs, team_skill[..., None], axis=-1)[..., 0]
        skill_reward_team = skill_logp + jnp.log(float(num_skills))
        expl_reward_team = delta_cov
        total_reward_team = skill_coef * skill_reward_team + exploration_coef * expl_reward_team
        reward_agent = total_reward_team[:, team_ids]
        skill_reward_agent = skill_reward_team[:, team_ids]
        expl_reward_agent = expl_reward_team[:, team_ids]
        return reward_agent, team_summary, team_skill, skill_reward_agent, expl_reward_agent

    def _update_step(
        learner_state: TeamDIAYNLearnerState, _: Any
    ) -> Tuple[TeamDIAYNLearnerState, Tuple]:
        def _env_step(
            learner_state: TeamDIAYNLearnerState, _: Any
        ) -> Tuple[TeamDIAYNLearnerState, Tuple[PPOTransitionTeamDIAYN, Metrics]]:
            params, opt_states, key, env_state, last_timestep, last_done = learner_state

            key, policy_key = jax.random.split(key)
            critic_obs = get_critic_obs(env_state)
            actor_policy = actor_apply_fn(params.actor_params, last_timestep.observation)
            value = critic_apply_fn(params.critic_params, critic_obs)
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)
            intrinsic_reward, team_summary, team_skill, skill_reward, expl_reward = compute_intrinsic(
                params.disc_params, env_state
            )

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            transition = PPOTransitionTeamDIAYN(
                last_done,
                action,
                value,
                intrinsic_reward,
                log_prob,
                last_timestep.observation,
                critic_obs,
                team_summary,
                team_skill,
                skill_reward,
                expl_reward,
            )
            learner_state = TeamDIAYNLearnerState(
                params, opt_states, key, env_state, timestep, done
            )
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return learner_state, (transition, metrics)

        learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        params, opt_states, key, env_state, last_timestep, last_done = learner_state
        last_val = critic_apply_fn(params.critic_params, get_critic_obs(env_state))

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        intrinsic_stats = {
            "intrinsic_reward": traj_batch.reward.mean(),
            "skill_reward": traj_batch.skill_reward.mean(),
            "exploration_reward": traj_batch.expl_reward.mean(),
        }

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOTransitionTeamDIAYN,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    if bool(_cfg_get(config.system, "use_ppo_clipping", True)):
                        actor_loss1 = ratio * gae
                        actor_loss2 = jnp.clip(
                            ratio, 1.0 - config.system.clip_eps, 1.0 + config.system.clip_eps
                        ) * gae
                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                    else:
                        actor_loss = -(log_prob * gae).mean()
                    entropy = actor_policy.entropy(seed=key).mean()
                    total_actor_loss = actor_loss - config.system.ent_coef * entropy
                    return total_actor_loss, (actor_loss, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransitionTeamDIAYN,
                    targets: chex.Array,
                ) -> Tuple:
                    value = critic_apply_fn(critic_params, traj_batch.critic_obs)
                    if bool(_cfg_get(config.system, "use_ppo_clipping", True)):
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config.system.clip_eps, config.system.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * jnp.square(value - targets).mean()
                    total_value_loss = config.system.vf_coef * value_loss
                    return total_value_loss, value_loss

                def _disc_loss_fn(
                    disc_params: FrozenDict,
                    traj_batch: PPOTransitionTeamDIAYN,
                ) -> Tuple:
                    logits = disc_apply_fn(disc_params, traj_batch.team_summary)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    nll = -jnp.take_along_axis(
                        log_probs, traj_batch.team_skill[..., None], axis=-1
                    )[..., 0]
                    disc_loss = nll.mean()
                    acc = (jnp.argmax(logits, axis=-1) == traj_batch.team_skill).astype(jnp.float32).mean()
                    return disc_coef * disc_loss, (disc_loss, acc)

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

                disc_grad_fn = jax.value_and_grad(_disc_loss_fn, has_aux=True)
                disc_loss_info, disc_grads = disc_grad_fn(params.disc_params, traj_batch)

                actor_grads, actor_loss_info = jax.lax.pmean((actor_grads, actor_loss_info), axis_name="batch")
                actor_grads, actor_loss_info = jax.lax.pmean((actor_grads, actor_loss_info), axis_name="device")
                critic_grads, value_loss_info = jax.lax.pmean((critic_grads, value_loss_info), axis_name="batch")
                critic_grads, value_loss_info = jax.lax.pmean((critic_grads, value_loss_info), axis_name="device")
                disc_grads, disc_loss_info = jax.lax.pmean((disc_grads, disc_loss_info), axis_name="batch")
                disc_grads, disc_loss_info = jax.lax.pmean((disc_grads, disc_loss_info), axis_name="device")

                actor_updates, actor_new_opt_state = actor_update_fn(actor_grads, opt_states.actor_opt_state)
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)
                critic_updates, critic_new_opt_state = critic_update_fn(critic_grads, opt_states.critic_opt_state)
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)
                disc_updates, disc_new_opt_state = disc_update_fn(disc_grads, opt_states.disc_opt_state)
                disc_new_params = optax.apply_updates(params.disc_params, disc_updates)

                new_params = TeamDIAYNParams(actor_new_params, critic_new_params, disc_new_params)
                new_opt_state = TeamDIAYNOptStates(
                    actor_new_opt_state, critic_new_opt_state, disc_new_opt_state
                )

                actor_loss, (_, entropy) = actor_loss_info
                value_loss, unscaled_value_loss = value_loss_info
                disc_loss, disc_acc = disc_loss_info
                total_loss = actor_loss + value_loss + disc_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": unscaled_value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                    "disc_loss": disc_loss,
                    "disc_acc": disc_acc,
                }
                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = tree.map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = TeamDIAYNLearnerState(
            params, opt_states, key, env_state, last_timestep, last_done
        )

        final_loss_info = tree.map(lambda x: x[-1].mean(), loss_info)
        final_loss_info = final_loss_info | intrinsic_stats
        return learner_state, (episode_metrics, final_loss_info)

    def learner_fn(learner_state: TeamDIAYNLearnerState) -> ExperimentOutput[TeamDIAYNLearnerState]:
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
) -> Tuple[LearnerFn[TeamDIAYNLearnerState], Actor, TeamDIAYNLearnerState]:
    n_devices = len(jax.devices())
    num_agents = env.num_agents
    config.system.num_agents = num_agents
    key, actor_net_key, critic_net_key, disc_net_key = keys

    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso if "pre_torso" in config.network.actor_network else config.network.actor_network.torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso if "pre_torso" in config.network.critic_network else config.network.critic_network.torso)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = ArrayCritic(torso=critic_torso)

    _, num_teams, num_skills, extract_team_summary = build_team_helpers(env)
    disc_hidden_dim = int(_cfg_get(config.system, "disc_hidden_dim", 128))
    disc_network = TeamDiscriminator(num_skills=num_skills, hidden_dim=disc_hidden_dim)

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)
    disc_lr = make_learning_rate(_cfg_get(config.system, "disc_lr", config.system.critic_lr), config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )
    disc_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(disc_lr, eps=1e-5),
    )

    obs = env.observation_spec.generate_value()
    init_x = tree.map(lambda x: x[jnp.newaxis, ...], obs)

    key, init_env_key = jax.random.split(key)
    init_env_state, _ = env.reset(init_env_key, 0)
    init_critic_x = get_critic_obs(init_env_state)[jnp.newaxis, ...]
    init_disc_x = extract_team_summary(init_env_state)[0]

    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_net_key, init_critic_x)
    critic_opt_state = critic_optim.init(critic_params)
    disc_params = disc_network.init(disc_net_key, init_disc_x)
    disc_opt_state = disc_optim.init(disc_params)

    params = TeamDIAYNParams(actor_params, critic_params, disc_params)
    apply_fns = (actor_network.apply, critic_network.apply, disc_network.apply)
    update_fns = (actor_optim.update, critic_optim.update, disc_optim.update)

    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(env_keys), 0)
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    dones = jnp.zeros((config.arch.num_envs, config.system.num_agents), dtype=bool)
    key, step_keys = jax.random.split(key)
    opt_states = TeamDIAYNOptStates(actor_opt_state, critic_opt_state, disc_opt_state)
    replicate_learner = (params, opt_states, step_keys, dones)
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    params, opt_states, step_keys, dones = replicate_learner
    init_learner_state = TeamDIAYNLearnerState(params, opt_states, step_keys, env_states, timesteps, dones)
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    _config.logger.system_name = "ff_mappo_team_diayn_comsd"
    config = copy.deepcopy(_config)
    eval_config = copy.deepcopy(_config)
    eval_config.env.scenario.task_config = OmegaConf.merge(
        eval_config.env.scenario.task_config,
        eval_config.env.scenario.eval_config,
    )
    n_devices = len(jax.devices())

    env, _ = environments.make(config=config, add_global_state=False)
    _, eval_env = environments.make(config=eval_config, add_global_state=False)

    key, key_e, actor_net_key, critic_net_key, disc_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=5
    )

    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key, disc_net_key), config
    )

    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    config = check_total_timesteps(config)
    assert config.system.num_updates > config.arch.num_evaluation
    assert config.arch.num_envs % config.system.num_minibatches == 0

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
    if config.env.scenario.task_config.render_only:
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
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
            eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)
            eval_metrics = evaluator(trained_params, eval_keys, {})
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
            abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
            eval_keys = jax.random.split(key, n_devices)
            eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})
            t = int(steps_per_rollout * (eval_step + 1))
            logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

        logger.stop()

    final_params = best_params if config.arch.absolute_metric and best_params is not None else trained_params
    final_params = unreplicate_n_dims(final_params, unreplicate_depth=1)
    final_episodes(config, final_params, actor_network)
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_mappo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}MAPPO Team-DIAYN/ComSD experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
