import jax
import jax.numpy as jnp
from jumanji.environments.swarms.search_and_rescue.env import SearchAndRescue

import jax
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass
from jumanji.environments.swarms.search_and_rescue.env import SearchAndRescue
from jumanji.environments.swarms.search_and_rescue.generator import RandomGenerator
from jumanji.types import TimeStep as JumanjiTimeStep
from jumanji.environments.swarms.search_and_rescue.types import State as JumanjiState
from evojax.task.base import VectorizedTask, TaskState

# ===========================================
# EvoJAX Integration
# ===========================================

@flax_dataclass
class SearchAndRescueState(TaskState):
    env_state: JumanjiState
    timestep: JumanjiTimeStep

class SearchAndRescueEvoTask(VectorizedTask):
    multi_agent_training: bool = True

    def __init__(
        self,
        num_agents: int = 2,
        num_targets: int = 40,
        max_steps: int = 400,
        test: bool = False,
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.test = test
        # Build Jumanji env with matching num_agents and num_targets
        self.env = SearchAndRescue(
            generator=RandomGenerator(num_targets=num_targets, num_searchers=num_agents),
            time_limit=max_steps,
        )
        # Compute obs and action shapes
        vision_spec = self.env.observation_spec.searcher_views.shape
        channels, vision = vision_spec[1], vision_spec[2]
        obs_dim = int(channels * vision)
        self.obs_shape = (self.num_agents, obs_dim)
        self.act_shape = tuple(self.env.action_spec.shape)
        self.max_steps = max_steps
        self.test = test
        # Vectorized reset/step functions
        self._reset_fn = jax.jit(jax.vmap(lambda key: self.env.reset(key)))
        self._step_fn = jax.jit(jax.vmap(lambda state, action: self.env.step(state, action)))

    def reset(self, key: jnp.ndarray) -> SearchAndRescueState:
        # key: single PRNGKey or array of keys for tasks
        keys = jax.random.split(key, self.obs_shape[0]) if key.ndim == 1 else key
        states, timesteps = self._reset_fn(keys)
        obs = timesteps.observation.searcher_views.reshape(
            keys.shape[0], self.num_agents, -1
        )
        return SearchAndRescueState(env_state=states, timestep=timesteps, obs=obs)

    def step(
        self, state: SearchAndRescueState, action: jnp.ndarray
    ) -> tuple[SearchAndRescueState, jnp.ndarray, jnp.ndarray]:
        next_states, next_timesteps = self._step_fn(
            state.env_state, action
        )
        obs = next_timesteps.observation.searcher_views.reshape(
            action.shape[0], self.num_agents, -1
        )
        rewards = next_timesteps.reward  # shape (num_tasks, num_agents)
        dones = (next_timesteps.step >= self.max_steps).astype(jnp.int32)
        next_state = SearchAndRescueState(
            env_state=next_states,
            timestep=next_timesteps,
            obs=obs,
        )
        return next_state, rewards, dones

# Example EvoJAX usage:
# from evojax.policy.mlp import MLPPolicy
# from evojax.algo import PGPE
# from evojax import Trainer
#
# train_task = SearchAndRescueEvoTask(num_agents=16, num_targets=40, max_steps=500)
# test_task = SearchAndRescueEvoTask(num_agents=16, num_targets=40, max_steps=500, test=True)
# policy = MLPPolicy(...input_dim=train_task.obs_shape[-1], output_dim=train_task.act_shape[-1])
# solver = PGPE(...)
# trainer = Trainer(policy, solver, train_task, test_task, ...)
# trainer.run()

# EvoJAX Integration
# ===========================================

class SearchAndRescueEvoTask:
    """
    Wraps the Jumanji SearchAndRescue environment as an EvoJAX-compatible task.
    """
    def __init__(self, num_envs: int, key: jax.random.KeyArray):
        self.num_envs = num_envs
        self.env = SearchAndRescue()
        self.key = key

        # Vectorize reset/step over batch dimension
        self._reset = jax.vmap(lambda k: self.env.reset(k), in_axes=0)
        self._step = jax.vmap(lambda state, action: self.env.step(state, action), in_axes=(0, 0))

        # Initialize batch
        keys = jax.random.split(self.key, self.num_envs)
        self.states, self.timesteps = self._reset(keys)

    def reset(self, key: jax.random.KeyArray):
        keys = jax.random.split(key, self.num_envs)
        self.states, self.timesteps = self._reset(keys)
        return self.states, self.timesteps

    def step(self, actions: jnp.ndarray):
        self.states, self.timesteps = self._step(self.states, actions)
        return self.states, self.timesteps

# Example EvoJAX usage (pseudo-code):
# from evojax.policy_gradient import PPO
# task = SearchAndRescueEvoTask(num_envs=128, key=jax.random.PRNGKey(0))
# agent = PPO(task)
# params = agent.init_params()
# for _ in range(10000):
#     trajectories = agent.rollout(params)
#     grads = agent.compute_gradients(trajectories)
#     params = agent.update(params, grads)


# ===========================================
# Mava Integration
# ===========================================

import jax.numpy as jnp
from mava.wrappers.jumanji import JumanjiMarlWrapper
from mava.types import Observation
from jax import numpy as jnp

# Custom wrapper for SearchAndRescue
class SearchAndRescueWrapper(JumanjiMarlWrapper):
    def __init__(self, env: SearchAndRescue, add_global_state: bool = False):
        super().__init__(env, add_global_state)

    def modify_timestep(self, timestep):
        # Convert per-agent view to float
        agents_view = timestep.observation.agents_view.astype(float)
        action_mask = timestep.observation.action_mask
        step_count = jnp.repeat(
            timestep.observation.step_count,
            self.num_agents,
        )
        # Build Mava's Observation type
        observation = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )
        # Broadcast team reward to each agent
        rewards = jnp.repeat(timestep.reward, self.num_agents)
        return timestep.replace(
            observation=observation,
            reward=rewards,
        )

# Factory to create wrapped env
def make_search_and_rescue_env(seed: int = None):
    base_env = SearchAndRescue()
    wrapped_env = SearchAndRescueWrapper(base_env)
    if seed is not None:
        wrapped_env.reset(seed)
    return wrapped_env

# Usage with Mava's FFMappo system
from mava.systems.ppo.anakin.ff_mappo import FFMappo
from mava.utils.seed import get_seed

def run_mava_ffmappo(
    total_environment_steps: int = 1_000_000,
    seed: int = 0,
):
    system = FFMappo(
        environment_factory=make_search_and_rescue_env,
        network_factory=None,  # use defaults in FFMappo
        seed=get_seed(seed),
    )
    system.run(
        num_environment_steps=1,
        total_environment_steps=total_environment_steps,
    )

# Example execution:
# if __name__ == '__main__':
#     run_mava_ffmappo(total_environment_steps=500_000, seed=42)
