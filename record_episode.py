# --- Start of inserted code ---
    print(f"{Fore.BLUE}{Style.BRIGHT}\nRunning final episode to collect states...{Style.RESET_ALL}")

    # 1. Re-instantiate the network's sub-components.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(eval_env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=eval_env.action_dim)

    # Get the trained actor parameters.
    final_params = unreplicate_batch_dim(learner_state.params.actor_params)

    # 2. Define a JIT-compiled action function that calls the sub-components directly.
    @jax.jit
    def final_act_fn(params: chex.ArrayTree, observation: chex.ArrayTree, key: chex.PRNGKey) -> chex.Array:
        """
        Manually calls the torso and action_head to ensure correct data shapes.
        """
        # Extract the parameters for the torso and action_head.
        torso_params = params['params']['torso']
        action_head_params = params['params']['action_head']

        # Call the torso with the raw 2D agents_view from the observation.
        # The torso expects an input shape of (num_agents, features).
        obs_embedding = actor_torso.apply({'params': torso_params}, observation.agents_view)

        # Call the action_head with the torso's output and the action mask.
        pi = actor_action_head.apply({'params': action_head_params}, obs_embedding, observation.action_mask)
        
        # Sample an action from the resulting policy.
        action = pi.sample(seed=key)
        return action

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
