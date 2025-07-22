# --- Start of inserted code ---
    print(f"{Fore.BLUE}{Style.BRIGHT}\nRunning final episode to collect states...{Style.RESET_ALL}")

    # 1. Re-instantiate the Actor network to get a clean object for JIT compilation.
    # This is the approach used in Mava's official Quickstart notebook.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(eval_env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=eval_env.action_dim)
    clean_actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    # Get the trained actor parameters from a single device.
    final_params = unreplicate_batch_dim(learner_state.params.actor_params)

    # 2. Define a jit-compiled action function using the clean network.
    @jax.jit
    def final_act_fn(params: chex.ArrayTree, observation: chex.ArrayTree, key: chex.PRNGKey) -> chex.Array:
        """A clean action function for a single, unbatched observation."""
        # Pass the raw, unbatched observation directly to the network's apply function.
        # The network is designed to handle this 2D (num_agents, features) input.
        pi = clean_actor_network.apply(params, observation)
        action = pi.sample(seed=key)
        return action

    # --- Run the episode using the new action function ---
    key_e, final_run_key = jax.random.split(key_e)
    state, timestep = eval_env.reset(final_run_key)
    done = False
    episode_states = [state]

    while not done:
        key_e, act_key = jax.random.split(key_e)
        
        # Get an action from our new function using the timestep's raw observation.
        action = final_act_fn(final_params, timestep.observation, act_key)
        
        # Step the environment.
        state, timestep = eval_env.step(state, action)
        episode_states.append(state)
        
        done = timestep.last()

    print(
        f"{Fore.GREEN}Collected {len(episode_states)} states from the final episode.{Style.RESET_ALL}"
    )
    # --- End of inserted code ---
