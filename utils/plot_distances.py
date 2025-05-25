import matplotlib.pyplot as plt
import numpy as np

def plot_distances(distance_histories):
    """
    Plots the distance histories across episodes, average distance per step, and cumulative distance.
    :param distance_histories: A 2D numpy array where each row corresponds to the distance history of an episode."""
    
    if not isinstance(distance_histories, np.ndarray):
        raise ValueError("distance_histories must be a numpy array.")
    if distance_histories.ndim != 2:
        raise ValueError("distance_histories must be a 2D array with shape (num_episodes, num_steps).")
    if distance_histories.shape[0] == 0 or distance_histories.shape[1] == 0:
        raise ValueError("distance_histories cannot be empty.")
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    axs[0].set_title('Distance History Across Episodes')
    for i in range(distance_histories.shape[0]):
        axs[0].plot(distance_histories[i])
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Distance')
    axs[0].legend()
    axs[0].grid()
    axs[1].set_title('Average Distance per Step Across Episodes')
    axs[1].plot(np.mean(distance_histories, axis=0), label='Average Distance per Step', color='orange')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Average Distance')
    axs[1].legend()
    axs[1].grid()
    axs[2].set_title('Cumulative Distance Across Steps')
    axs[2].plot(np.cumsum(np.mean(distance_histories, axis=0)), label='Cumulative Distance', color='green')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Cumulative Distance')
    axs[2].legend()
    axs[2].grid()
    plt.tight_layout()
    plt.show()
    return fig, axs
   