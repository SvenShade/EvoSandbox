# --- Start of inserted code ---
    print(f"{Fore.BLUE}{Style.BRIGHT}\nRunning final episode to collect states...{Style.RESET_ALL}")

    # Get the trained actor network parameters from a single device.
    final_params = unreplicate_batch_dim(learner_state.params.actor_params)

    # Define a new, clean action function for this specific one-off run.
    # This avoids any JIT compilation issues from the main evaluation loop.
    @jax.jit
    def final_act_fn(params: chex.ArrayTree, observation: chex.ArrayTree, key: chex.PRNGKey) -> chex.Array:
        """A clean action function that correctly handles batch dimensions."""
        # The underlying network expects a batch dimension, so we add one.
        batched_observation = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], observation)
        
        # Get the policy distribution from the network.
        pi = actor_network.apply(params, batched_observation)
        
        # Sample an action from the policy.
        action = pi.sample(seed=key)
        
        # The environment expects a non-batched action, so we remove the batch dimension.
        return jax.tree_util.tree_map(lambda x: x.squeeze(0), action)

    # --- Run the episode using the new action function ---
    key_e, final_run_key = jax.random.split(key_e)
    state, timestep = eval_env.reset(final_run_key)
    done = False
    episode_states = [state]

    while not done:
        key_e, act_key = jax.random.split(key_e)
        
        # Get an action from our new, clean function using the timestep's observation.
        action = final_act_fn(final_params, timestep.observation, act_key)
        
        # Step the environment.
        state, timestep = eval_env.step(state, action)
        episode_states.append(state)
        
        done = timestep.last()

    print(
        f"{Fore.GREEN}Collected {len(episode_states)} states from the final episode.{Style.RESET_ALL}"
    )
    # --- End of inserted code ---
