# Initialise observation with obs of all agents.
obs = env.observation_spec.generate_value()

# --- Create init data for the centralized critic ---
# The critic sees the flattened observations of all agents.
def _preprocess_critic_init_obs(x: chex.Array) -> chex.Array:
    """Adds a batch dim and flattens all other dims for the centralized critic."""
    # Input shape e.g.: (3, 18) -> (1, 3, 18)
    x_batched = x[jnp.newaxis, ...]
    # Output shape e.g.: (1, 3, 18) -> (1, 54)
    return x_batched.reshape((x_batched.shape[0], -1))

critic_init_x = tree.map(_preprocess_critic_init_obs, obs)

# --- Create init data for the independent actor ---
# The actor sees the observation of a single agent.
# We take the obs for the first agent and add a batch dimension.
# Input shape e.g.: (3, 18) -> (18,)
single_agent_obs = jax.tree_util.tree_map(lambda x: x[0], obs)
# Output shape e.g.: (18,) -> (1, 18)
actor_init_x = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], single_agent_obs)


# Initialise actor params and optimiser state.
# The actor expects input for a single agent, e.g. shape (1, 18)
actor_params = actor_network.init(actor_net_key, actor_init_x)
actor_opt_state = actor_optim.init(actor_params)

# Initialise critic params and optimiser state.
# The critic expects the full centralized input, e.g. shape (1, 54)
critic_params = critic_network.init(critic_net_key, critic_init_x)
critic_opt_state = critic_optim.init(critic_params)


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
