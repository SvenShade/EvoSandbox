_add_env_axis = lambda obs: tree.map(lambda x: x[jnp.newaxis, ...], obs)

# learner_setup
obs     = env.observation_spec.generate_value()
init_x  = _add_env_axis(obs)

# env step
obs_b = _add_env_axis(last_timestep.observation)
…apply_fn(…, obs_b)

# advantage bootstrap
last_val = critic_apply_fn(..., _add_env_axis(last_timestep.observation))

# losses
obs_b = _add_env_axis(traj_batch.obs)
…apply_fn(…, obs_b)

# render
actor_policy = actor_apply_fn(actor_params, _add_env_axis(timestep.observation))
