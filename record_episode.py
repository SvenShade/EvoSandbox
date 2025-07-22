# --- Start of inserted code ---
    print(f"{Fore.BLUE}{Style.BRIGHT}\nRunning final episode to collect states...{Style.RESET_ALL}")

    # Re-instantiate the Actor network to ensure a clean state for JIT compilation.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(eval_env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=eval_env.action_dim)
    clean_actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    # Get the trained actor parameters from a single device.
    final_params = unreplicate_batch_dim(learner_state.params.actor_params)

    # Define the JIT-compiled action function.
    @jax.jit
    def final_act_fn(params: chex.ArrayTree, observation: chex.ArrayTree, key: chex.PRNGKey) -> chex.Array:
        """
        Action function for a single step. It adds a batch dimension to the observation
        for the network and removes the batch dimension from the resulting action.
        """
        # Add a leading batch dimension to the observation PyTree.
        # e.g., (num_agents, features) -> (1, num_agents, features)
        batched_observation = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], observation)

        # Get the policy distribution from the network.
        pi = clean_actor_network.apply(params, batched_observation)

        # Sample an action.
        action = pi.sample(seed=key)

        # Remove the leading batch dimension from the action to match the environment's expectation.
        # e.g., (1, num_agents, action_dim) -> (num_agents, action_dim)
        return jax.tree_util.tree_map(lambda x: x.squeeze(0), action)

    # --- Run the episode ---
    key_e, final_run_key = jax.random.split(key_e)
    state, timestep = eval_env.reset(final_run_key)
    done = False
    episode_states = [state]

    while not done:
        key_e, act_key = jax.random.split(key_e)

        # Get an action from our new function.
        action = final_act_fn(final_params, timestep.observation, act_key)

        # Step the environment.
        state, timestep = eval_env.step(state, action)
        episode_states.append(state)

        done = timestep.last()

    print(
        f"{Fore.GREEN}Collected {len(episode_states)} states from the final episode.{Style.RESET_ALL}"
    )
    # --- End of inserted code ---
