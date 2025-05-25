def run_agent(agent, env, max_steps=100, min_steps=1, render=False):
    """
    Runs a single episode with the given environment and agent.

    :param env: The environment to run the episode in.
    :param agent: The agent that will interact with the environment.
    :param max_steps: Maximum number of steps in the episode.
    :param render: Whether to render the environment.
    :return: Total reward accumulated during the episode.
    """
    observation, info = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    reward_history = []
    distances = []
    
    for i in range(min_steps):
        if render:
            env.render()

        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        reward_history.append(reward)
        distances.append(info.get('distance_to_goal', 0))
        total_reward += reward
        step_count += 1
    # Ensure we run at least min_steps before checking for done
    if done:
        return total_reward, reward_history, distances  # Return early if the episode is done
    while not done and step_count < max_steps:
        if render:
            env.render()

        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        reward_history.append(reward)
        distances.append(info.get('distance_to_goal', 0))

        total_reward += reward
        step_count += 1

    if render:
        env.render()  # Final render after the episode ends

    return total_reward, reward_history, distances
