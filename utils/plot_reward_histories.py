import matplotlib.pyplot as plt
import numpy as np

def plot_reward_histories(reward_histories):
    """
    Plots the reward histories across episodes, average reward per step, and cumulative reward.
    
    :param reward_histories: A 2D numpy array where each row corresponds to the reward history of an episode.
    """
    if not isinstance(reward_histories, np.ndarray):
        raise ValueError("reward_histories must be a numpy array.")
    
    if reward_histories.ndim != 2:
        raise ValueError("reward_histories must be a 2D array with shape (num_episodes, num_steps).")
    
    if reward_histories.shape[0] == 0 or reward_histories.shape[1] == 0:
        raise ValueError("reward_histories cannot be empty.")
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    axs[0].set_title('Reward History Across Episodes')
    for i in range(reward_histories.shape[0]):
        axs[0].plot(reward_histories[i])
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid()
    axs[1].set_title('Average Reward per Step Across Episodes')
    axs[1].plot(np.mean(reward_histories, axis=0), label='Average Reward per Step', color='orange')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Average Reward')
    axs[1].legend()
    axs[1].grid()
    axs[2].set_title('Cumulative Reward Across Steps')
    axs[2].plot(np.cumsum(np.mean(reward_histories, axis=0)), label='Cumulative Reward', color='green')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Cumulative Reward')
    axs[2].legend()
    axs[2].grid()
    plt.tight_layout()
    plt.show()
    return fig, axs