    # --- Start of Final Diagnostic Code ---
    print(f"{Fore.MAGENTA}{Style.BRIGHT}\n--- Starting Final Diagnostic Step ---{Style.RESET_ALL}")

    try:
        # 1. Get the trained parameters and a key.
        final_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, reset_key, act_key = jax.random.split(key_e, 3)

        # 2. Get a single, unbatched observation from the environment.
        _, timestep = eval_env.reset(reset_key)
        observation_to_test = timestep.observation
        print("Shape of observation from single environment:")
        print(jax.tree_map(lambda x: x.shape, observation_to_test))

        # 3. Manually add a batch dimension of 1.
        # This replicates the condition that causes the error.
        batched_observation = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], observation_to_test)
        print("\nShape of observation after manually adding batch dimension:")
        print(jax.tree_map(lambda x: x.shape, batched_observation))

        # 4. Attempt the single failing operation.
        # We use the original actor_network object from the trainer.
        print("\nAttempting to call actor_network.apply()...")
        pi = actor_network.apply(final_params, batched_observation)
        action = pi.sample(seed=act_key)
        
        print(f"{Fore.GREEN}Call to actor_network.apply() was SUCCESSFUL.{Style.RESET_ALL}")
        print("Sampled action shapes:")
        print(jax.tree_map(lambda x: x.shape, action))

    except Exception as e:
        print(f"\n{Fore.RED}Call to actor_network.apply() FAILED with error:{Style.RESET_ALL}")
        # Print the full traceback for the error from this specific call
        import traceback
        traceback.print_exc()

    print(f"{Fore.MAGENTA}--- End of Diagnostic Step ---{Style.RESET_ALL}")
    # --- End of Final Diagnostic Code ---
