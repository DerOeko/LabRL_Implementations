import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env): 
    def __init__(self, size=4, start_pos=None, target_pos=None, obstacles=None):
        self.size = size
        assert start_pos is None or (0 <= start_pos[0] < size and 0 <= start_pos[1] < size), "Invalid start position"
        assert target_pos is None or (0 <= target_pos[0] < size and 0 <= target_pos[1] < size), "Invalid target position"
        assert obstacles is None or all(0 <= obs[0] < size and 0 <= obs[1] < size for obs in obstacles), "Invalid obstacle positions"
        assert start_pos is None or not np.array_equal(start_pos, target_pos), "Start and target positions cannot be the same"
        assert obstacles is None or not any(np.array_equal(obs, start_pos) for obs in obstacles), "Obstacles cannot overlap with start position"
        assert obstacles is None or not any(np.array_equal(obs, target_pos) for obs in obstacles), "Obstacles cannot overlap with target position"
        assert obstacles is None or len(obstacles) == len(set(map(tuple, obstacles))), "Obstacles must be unique"
        
        super().__init__()
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.obstacles = obstacles if obstacles is not None else []

        if self.start_pos is None:
            self.agent_pos = np.array([0, 0])
        else:
            self.agent_pos = np.array(self.start_pos)

        if target_pos is None:
            self.target_pos = np.array([size - 1, size - 1])
        else:
            self.target_pos = np.array(target_pos)

        self.obstacles = []
        if obstacles:
            self.obstacles = [np.array(obs) for obs in obstacles]

        # Define action space (essential for ANY agent)
        # 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)

        # Define observation space (essential for ANY agent)
        # For a simple gridworld, agent's (row, col) position can be the state.
        # If using Gymnasium:
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.size - 1, self.size - 1]),
            dtype=np.int32
        )

        # For a "cheating" agent, it needs to know the environment's rules and goal.
        # This is already part of the environment's definition (self.target_pos, self.obstacles, self.size).
        # The cheating agent will be implemented *outside* this class but will use this info.

    def reset(self, seed=None, options=None): 
        super().reset(seed=seed)

        if self.start_pos is None:
            self.agent_pos = np.array([0, 0])
        else:
            self.agent_pos = np.array(self.start_pos)

        observation = self._get_obs()
        info = self._get_info() # Auxiliary info, can be empty for now
        return observation, info

    def step(self, action):
        new_pos = self.agent_pos.copy()
        
        # --- Apply action ---
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Down
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # Right
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        else:
            raise ValueError(f"Invalid action {action}")
        
        # --- Check for termination and calculate reward ---
        terminated = np.array_equal(new_pos, self.target_pos)
        truncated = False # For time limits, not typically used in simple gridworlds unless you add max_steps
        reward = -0.1 # # Small negative reward for each step to encourage efficiency

        if terminated:
            reward += 1.0

        if any(np.array_equal(new_pos, obs) for obs in self.obstacles):
            # Optional: penalty for hitting an obstacle and maybe reset or stay put
            reward += -1.0
            # self.agent_pos = np.array([0,0]) # Example: reset on hitting obstacle
            # Or, if obstacles are walls, undo the move if new pos is obstacle
            new_pos = self.agent_pos  # Stay in place if hitting an obstacle

        self.agent_pos = new_pos  # Update agent's position after checking conditions

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # The observation is the agent's current position
        return np.array(self.agent_pos, dtype=np.int32)

    def _get_info(self):
        # Can contain auxiliary information (e.g., distance to goal)
        # Useful for debugging or for a "cheating" agent if it needs specific structured info.
        return {"distance_to_goal": np.linalg.norm(self.agent_pos - self.target_pos)}

    def render(self, mode='human'): # Optional, but very useful for debugging
        grid = np.full((self.size, self.size), '_', dtype=str)
        for obs_pos in self.obstacles:
            grid[tuple(obs_pos)] = 'X'
        grid[tuple(self.target_pos)] = 'G'
        grid[tuple(self.agent_pos)] = 'A'
        print("\n".join(" ".join(row) for row in grid))
        print("-"*(self.size*2-1))

    def close(self):
        pass