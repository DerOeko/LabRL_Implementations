import numpy as np
from ..base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        """
        Initializes the RandomAgent with a given action space.
        
        :param action_space: The action space of the environment (e.g., gymnasium.spaces.Discrete).
        """
        self.action_space = action_space

    def act(self, obs):
        """
        Selects a random action from the action space.
        
        :param observation: The current observation from the environment (not used in this agent).
        :return: A randomly selected action.
        """
        return self.action_space.sample()