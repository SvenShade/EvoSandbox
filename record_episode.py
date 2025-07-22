print(f"{Fore.BLUE}{Style.BRIGHT}\nRunning final episode to collect states...{Style.RESET_ALL}")

# --- Create a new, clean network instance for the final evaluation ---
# This avoids JIT compilation conflicts with the network used in the main training loop.
actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
# NOTE: Use eval_env here to ensure action spec matches the evaluation environment.
action_head, _ = get_action_head(eval_env.action_spec)
actor_action_head = hydra.utils.instantiate(action_head, action_dim=eval_env.action_dim)
clean_actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

# Get the trained actor parameters from a single device.
final_params = unreplicate_batch_dim(learner_state.params.actor_params)

# Define a clean action function using the new network instance.
@jax.jit
def final_act_fn(params: chex.ArrayTree, observation: chex.ArrayTree, key: chex.PRNGKey) -> chex.Array:
    """A clean action function that correctly handles batch dimensions."""
    # The underlying network's torso expects a batched input.
    batched_observation = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], observation)
    
    # Get the policy distribution from the clean network.
    pi = clean_actor_network.apply(params, batched_observation)
    
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
    
    # Get an action from our new, clean function.
    action = final_act_fn(final_params, timestep.observation, act_key)
    
    # Step the environment.
    state, timestep = eval_env.step(state, action)
    episode_states.append(state)
    
    done = timestep.last()

print(
    f"{Fore.GREEN}Collected {len(episode_states)} states from the final episode.{Style.RESET_ALL}"
)
