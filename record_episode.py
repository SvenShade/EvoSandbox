# After all training and evaluation, run a final episode to collect states.
print(f"{Fore.BLUE}{Style.BRIGHT}\nRunning final episode to collect states...{Style.RESET_ALL}")

# Use the actor parameters from the very last training iteration.
final_params = unreplicate_batch_dim(learner_state.params.actor_params)
key_e, final_run_key = jax.random.split(key_e)

# Reset the evaluation environment.
state, timestep = eval_env.reset(final_run_key)
done = False
episode_states = [state]

while not done:
    key_e, act_key = jax.random.split(key_e)

    # Select an action by passing the entire timestep object.
    # The existing eval_act_fn handles the decentralized observations correctly.
    action = eval_act_fn(final_params, timestep, act_key)

    # Step the environment.
    state, timestep = eval_env.step(state, action)
    episode_states.append(state)

    # Check if the episode has terminated.
    done = timestep.last()

print(
    f"{Fore.GREEN}Collected {len(episode_states)} states from the final episode.{Style.RESET_ALL}"
)
