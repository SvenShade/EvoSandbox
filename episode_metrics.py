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

from typing import TYPE_CHECKING, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import MarlEnv, State

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class RecordEpisodeMetricsState:
    """State of the `RecordEpisodeMetrics` wrapper."""

    env_state: State
    key: chex.PRNGKey

    # Temporary variables to keep track of the episode return and length.
    running_count_episode_return: chex.Numeric
    running_count_episode_length: chex.Numeric
    running_count_episode_return_t0: chex.Numeric
    running_count_episode_return_t1: chex.Numeric

    # Final episode return and length.
    episode_return: chex.Numeric
    episode_length: chex.Numeric
    episode_return_t0: chex.Numeric
    episode_return_t1: chex.Numeric


class RecordEpisodeMetrics(Wrapper):
    """Record episode returns and lengths.

    Adds the following keys in `timestep.extras["episode_metrics"]`:
      - episode_return: mean return over *all* agents for the completed episode
      - episode_return_T0: mean return over team-0 agents for the completed episode
      - episode_return_T1: mean return over team-1 agents for the completed episode
      - episode_length: number of steps in the completed episode
      - is_terminal_step: whether this step ended the episode

    Notes:
      - This wrapper assumes `timestep.reward` is per-agent with shape (num_agents,).
      - If the env does not expose `team_ids`, all agents default to team 0.
    """

    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: MarlEnv):
        super().__init__(env)
        self._env: MarlEnv

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

        # Team masks for team-split episode returns.
        team_ids = getattr(self._env, "team_ids", None)
        if team_ids is None:
            team_ids = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        else:
            team_ids = jnp.asarray(team_ids, dtype=jnp.int32)

        self._team_ids = team_ids
        self._t0_mask = (team_ids == 0).astype(jnp.float32)  # (N,)
        self._t1_mask = (team_ids == 1).astype(jnp.float32)  # (N,)
        self._t0_n = jnp.maximum(self._t0_mask.sum(), 1.0)
        self._t1_n = jnp.maximum(self._t1_mask.sum(), 1.0)

    def reset(self, key: chex.PRNGKey) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Reset the environment."""
        key, reset_key = jax.random.split(key)
        env_state, timestep = self._env.reset(reset_key)

        state = RecordEpisodeMetricsState(
            env_state=env_state,
            key=key,
            running_count_episode_return=jnp.array(0.0, dtype=jnp.float32),
            running_count_episode_length=jnp.array(0, dtype=jnp.int32),
            running_count_episode_return_t0=jnp.array(0.0, dtype=jnp.float32),
            running_count_episode_return_t1=jnp.array(0.0, dtype=jnp.float32),
            episode_return=jnp.array(0.0, dtype=jnp.float32),
            episode_length=jnp.array(0, dtype=jnp.int32),
            episode_return_t0=jnp.array(0.0, dtype=jnp.float32),
            episode_return_t1=jnp.array(0.0, dtype=jnp.float32),
        )

        timestep.extras["episode_metrics"] = {
            "episode_return": jnp.array(0.0, dtype=jnp.float32),
            "episode_return_T0": jnp.array(0.0, dtype=jnp.float32),
            "episode_return_T1": jnp.array(0.0, dtype=jnp.float32),
            "episode_length": jnp.array(0, dtype=jnp.int32),
            "is_terminal_step": jnp.array(False, dtype=jnp.bool_),
        }
        return state, timestep

    def step(
        self,
        state: RecordEpisodeMetricsState,
        action: chex.Array,
    ) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Step the environment."""
        env_state, timestep = self._env.step(state.env_state, action)

        done = timestep.last()
        not_done = 1 - done

        # `timestep.reward` is expected to be per-agent: shape (N,)
        r = timestep.reward.astype(jnp.float32)

        # Per-step returns.
        step_ret_all = jnp.mean(r)
        step_ret_t0 = jnp.sum(r * self._t0_mask) / self._t0_n
        step_ret_t1 = jnp.sum(r * self._t1_mask) / self._t1_n

        # Counting episode return and length.
        new_episode_return = state.running_count_episode_return + step_ret_all
        new_episode_return_t0 = state.running_count_episode_return_t0 + step_ret_t0
        new_episode_return_t1 = state.running_count_episode_return_t1 + step_ret_t1
        new_episode_length = state.running_count_episode_length + 1

        # Previous episode return/length until done and then the next episode return.
        episode_return_info = state.episode_return * not_done + new_episode_return * done
        episode_return_t0_info = state.episode_return_t0 * not_done + new_episode_return_t0 * done
        episode_return_t1_info = state.episode_return_t1 * not_done + new_episode_return_t1 * done
        episode_length_info = state.episode_length * not_done + new_episode_length * done

        timestep.extras["episode_metrics"] = {
            "episode_return": episode_return_info,
            "episode_return_T0": episode_return_t0_info,
            "episode_return_T1": episode_return_t1_info,
            "episode_length": episode_length_info,
            "is_terminal_step": done,
        }

        state = RecordEpisodeMetricsState(
            env_state=env_state,
            key=state.key,
            running_count_episode_return=new_episode_return * not_done,
            running_count_episode_length=new_episode_length * not_done,
            running_count_episode_return_t0=new_episode_return_t0 * not_done,
            running_count_episode_return_t1=new_episode_return_t1 * not_done,
            episode_return=episode_return_info,
            episode_length=episode_length_info,
            episode_return_t0=episode_return_t0_info,
            episode_return_t1=episode_return_t1_info,
        )
        return state, timestep


def get_final_step_metrics(metrics: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], bool]:
    """Get the metrics for the final step of an episode and check if there was a final step
    within the provided metrics.

    Note: this is not a jittable method. We need to return variable length arrays, since
    we don't know how many episodes have been run. This is done since the logger
    expects arrays for computing summary statistics on the episode metrics.
    """
    is_final_ep = metrics.get("is_terminal_step", np.array([False]))
    has_final_ep_step = bool(np.any(is_final_ep))

    final_metrics: Dict[str, chex.Array]
    # If it didn't make it to the final step, return zeros.
    if not has_final_ep_step:
        final_metrics = tree.map(np.zeros_like, metrics)
    else:
        final_metrics = tree.map(lambda x: x[is_final_ep], metrics)

    # Keep is_terminal_step in the metrics for the logger to use
    final_metrics["is_terminal_step"] = is_final_ep

    return final_metrics, has_final_ep_step
