from ..custom.gridworld_env import GridWorldEnv
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SlipGridWorldEnv(GridWorldEnv):
    def __init__(self, size=4, start_pos=None, target_pos=None, obstacles=None, slip_prob=0.1):
        super().__init__(size=size, start_pos=start_pos, target_pos=target_pos, obstacles=obstacles)
        self.slip_prob = slip_prob  # Probability of slipping

    def step(self, action):
        if np.random.rand() < self.slip_prob:
            # Slip to a random action
            action = self.action_space.sample()

        # Call the parent class's step method to execute the action
        return super().step(action)