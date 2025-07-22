def select_action(params, obs, key):
    batched_obs = tree.map(lambda x: x[jnp.newaxis, ...], obs)  # (1, n_agents, obs_dim)
    pi = actor_network.apply(params, batched_obs)
    return pi.mode(seed=key)[0]                                # drop env axis
select_action = jax.jit(select_action)

def play_and_render_episode(env, params, seed=0):
    key = jax.random.PRNGKey(seed)
    state, ts = env.reset(key)
    while not ts.last():
        key, act_key = jax.random.split(key)
        action = select_action(params, ts.observation, act_key)
        state, ts = env.step(state, action)
