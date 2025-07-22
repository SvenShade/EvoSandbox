# --- Start of inserted code ---
    print(f"{Fore.BLUE}{Style.BRIGHT}\nRunning final episode to collect states...{Style.RESET_ALL}")

    # Get the trained actor network parameters from a single device.
    final_params = unreplicate_batch_dim(learner_state.params.actor_params)

    # --- Define an action function that works on a batch of environments ---
    # Re-instantiating the network to be safe and avoid any JIT cache issues.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(eval_env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=eval_env.action_dim)
    clean_actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    @jax.jit
    def batch_act_fn(params: chex.ArrayTree, observation: chex.ArrayTree, key: chex.PRNGKey) -> chex.Array:
        """Action function for a batch of observations. The batch dim is already present."""
        pi = clean_actor_network.apply(params, observation)
        action = pi.sample(seed=key)
        return action

    # --- Set up and run two parallel environments to avoid batch size 1 issue ---
    num_parallel_runs = 2
    vmapped_env_reset = jax.vmap(eval_env.reset)
    vmapped_env_step = jax.vmap(eval_env.step)

    key_e, final_run_key = jax.random.split(key_e)
    # Create separate keys for each parallel environment.
    keys = jax.random.split(final_run_key, num_parallel_runs)

    # Reset the batch of environments.
    states, timesteps = vmapped_env_reset(keys)

    # CORRECTED LINE: Unstack the initial states into a list, one for each parallel run.
    initial_states_list = [jax.tree_util.tree_map(lambda x: x[i], states) for i in range(num_parallel_runs)]
    # Initialize a list of lists to store the episode states for each run.
    episode_states_batch = [[s] for s in initial_states_list]
    
    dones = jnp.zeros(num_parallel_runs, dtype=bool)

    while not jnp.all(dones):
        key_e, act_key = jax.random.split(key_e)
        act_keys = jax.random.split(act_key, num_parallel_runs)

        # Get actions for the batch of environments.
        actions = batch_act_fn(final_params, timesteps.observation, act_keys)

        # Step the batch of environments.
        states, timesteps = vmapped_env_step(states, actions)
        
        # Update done flags and store states for any episodes that are still running.
        new_dones = timesteps.last()
        for i in range(num_parallel_runs):
            if not dones[i]:
                # Extract the state for the i-th environment from the batched PyTree.
                s_i = jax.tree_util.tree_map(lambda x: x[i], states)
                episode_states_batch[i].append(s_i)
        dones = new_dones

    # We only need the results from the first of our parallel runs.
    episode_states = episode_states_batch[0]

    print(
        f"{Fore.GREEN}Collected {len(episode_states)} states from the final episode.{Style.RESET_ALL}"
    )
    # --- End of inserted code ---
