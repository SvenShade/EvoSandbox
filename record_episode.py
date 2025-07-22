@jax.jit
    def final_act_fn(params: chex.ArrayTree, observation: chex.ArrayTree, key: chex.PRNGKey) -> chex.Array:
        """
        A clean action function that reshapes the observation to avoid the batch-size-1 error.
        """
        # Create a new observation object with the agents_view reshaped.
        # This transforms the (1, 3, 18) batched view into a (3, 18) view.
        reshaped_obs = observation.replace(
            agents_view=observation.agents_view.reshape(-1, observation.agents_view.shape[-1])
        )

        # Get the policy distribution from the network using the reshaped observation.
        pi = clean_actor_network.apply(params, reshaped_obs)
        
        # Sample an action. The output will be correctly shaped (3, action_dim).
        action = pi.sample(seed=key)
        return action
