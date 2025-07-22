# After all training and evaluation, run a final episode to collect states.
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

    # Add a batch dimension to the observation for the actor network.
    obs_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, :], timestep.observation)

    # Select an action.
    action = eval_act_fn(final_params, obs_b, act_key)

    # Remove the batch dimension from the action before stepping the environment.
    action = jax.tree_util.tree_map(lambda x: x.squeeze(0), action)

    # Step the environment.
    state, timestep = eval_env.step(state, action)
    episode_states.append(state)

    # Check if the episode has terminated.
    done = timestep.last()

print(
    f"{Fore.GREEN}Collected {len(episode_states)} states from the final episode.{Style.RESET_ALL}"
)
