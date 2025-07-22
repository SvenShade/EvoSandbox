# Initialise observation with obs of all agents.
obs = env.observation_spec.generate_value()

# Preprocess the dummy observation to ensure it has the correct shape.
def _preprocess_init_obs(x: chex.Array) -> chex.Array:
    """Add a batch dim and flatten the remaining dimensions."""
    # Add a batch dimension: (num_agents, features) -> (1, num_agents, features)
    x = x[jnp.newaxis, ...]
    # Flatten agent and feature dimensions: (1, num_agents, features) -> (1, num_agents * features)
    return x.reshape((x.shape[0], -1))

init_x = tree.map(_preprocess_init_obs, obs)

# Initialise actor params and optimiser state.
actor_params = actor_network.init(actor_net_key, init_x)


 print(f"{Fore.BLUE}{Style.BRIGHT}\nRunning final episode to collect states...{Style.RESET_ALL}")

    # Use the best parameters if they exist, otherwise use the last trained parameters.
    final_params = (
        best_params if config.arch.absolute_metric and best_params is not None else trained_params
    )
    key_e, final_run_key = jax.random.split(key_e)

    # Reset the evaluation environment.
    state, timestep = eval_env.reset(final_run_key)
    done = False
    episode_states = [state]

    while not done:
        key_e, act_key = jax.random.split(key_e)

        # Select an action by passing the entire timestep object.
        # The eval_act_fn wrapper handles extracting the observation and batching.
        action = eval_act_fn(final_params, timestep, act_key)

        # Step the environment.
        state, timestep = eval_env.step(state, action)
        episode_states.append(state)

        # Check if the episode has terminated.
        done = timestep.last()

    print(
        f"{Fore.GREEN}Collected {len(episode_states)} states from the final episode.{Style.RESET_ALL}"
    )
