@jax.jit
    def final_act_fn(params: chex.ArrayTree, observation: chex.ArrayTree, key: chex.PRNGKey) -> chex.Array:
        """
        Manually vmaps the torso over the agents dimension to correctly handle
        the lifted 3D parameters.
        """
        # Extract the parameters for the torso and action_head.
        torso_params = params['params']['torso']
        action_head_params = params['params']['action_head']

        # Vmap the torso's apply function. This tells JAX to apply the torso
        # to each agent's observation slice in the (3, 18) agents_view,
        # while using the corresponding slice from the (1, 18, 128) kernel.
        vmapped_torso_apply = jax.vmap(
            actor_torso.apply, in_axes=(None, 0), out_axes=0
        )
        obs_embedding = vmapped_torso_apply({'params': torso_params}, observation.agents_view)

        # Call the action_head with the now correctly shaped embeddings.
        pi = actor_action_head.apply({'params': action_head_params}, obs_embedding, observation.action_mask)
        
        # Sample an action.
        action = pi.sample(seed=key)
        return action
