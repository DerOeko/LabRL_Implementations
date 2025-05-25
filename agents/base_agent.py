from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def act(self, observation):
        """
        Perform an action based on the given observation.
        
        :param observation: The current state or observation from the environment.
        :return: The action to be taken.
        """
        pass
