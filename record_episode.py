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

# ==============================================================================
# ADD THE FOLLOWING IMPORTS TO THE TOP OF ff_ippo.py
# ==============================================================================
import jax
import jax.numpy as jnp
import jax.tree_util
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig

from mava.networks import Actor
from mava.utils import make_env as environments
from mava.utils.jax_utils import unreplicate_n_dims

# ==============================================================================
# ADD THIS FUNCTION DEFINITION WITHIN ff_ippo.py
# (e.g., after the get_learner_fn or learner_setup function)
# ==============================================================================


def final_episode(config: DictConfig, params: FrozenDict, actor_network: Actor) -> None:
    """Run a final episode of the trained policy and collect states.

    Args:
        config: System configuration.
        params: Trained actor network parameters. Must be unreplicated.
        actor_network: The actor network instance.
    """
    # Add the following to your hydra config to enable this functionality:
    # arch:
    #   run_final_episode: True

    print(f"{Fore.CYAN}{Style.BRIGHT}Running final episode with trained policy...{Style.RESET_ALL}")

    # Create a single environment for the final episode.
    # We cannot use the env from the learner as it is vmapped.
    final_env, _ = environments.make(config, add_global_state=False)

    apply_fn = actor_network.apply
    reset_fn = jax.jit(final_env.reset)
    step_fn = jax.jit(final_env.step)
    key = jax.random.PRNGKey(config.system.seed)

    key, reset_key = jax.random.split(key)
    state, timestep = reset_fn(reset_key)

    states = [state]
    episode_return = 0.0
    episode_length = 0

    while not timestep.last():
        key, action_key = jax.random.split(key)
        # Add a batch dimension to the observation before passing it to the network.
        observation_with_batch = jax.tree_util.tree_map(
            lambda x: x[jnp.newaxis, ...], timestep.observation
        )
        pi = apply_fn(params, observation_with_batch)

        if config.arch.evaluation_greedy:
            action = pi.mode()
        else:
            action = pi.sample(seed=action_key)

        # Remove the batch dimension from the action before passing it to the environment.
        action = action.squeeze(0)

        state, timestep = step_fn(state, action)
        states.append(state)
        episode_return += jnp.mean(timestep.reward)
        episode_length += 1

    print("Final Episode:")
    print(f"  Return: {episode_return}")
    print(f"  Length: {episode_length}")
    print(f"  Collected {len(states)} states.")


# ==============================================================================
# IN THE `run_experiment` FUNCTION, ADD THE FOLLOWING CODE BLOCK
# JUST BEFORE THE FINAL `return eval_performance` STATEMENT.
# ==============================================================================
"""
    # ... existing code ...
    # Stop the logger.
    logger.stop()

    # Run a final episode.
    if config.arch.get("run_final_episode", False):
        # Use best params if available, otherwise use final params.
        final_params = (
            best_params
            if config.arch.absolute_metric and best_params is not None
            else trained_params
        )
        # Unreplicate the device dimension before passing to the non-pmapped function.
        final_params = unreplicate_n_dims(final_params, unreplicate_depth=1)
        final_episode(config, final_params, actor_network)


    return eval_performance
"""
