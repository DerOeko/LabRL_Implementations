import numpy as np
from ..base_agent import BaseAgent

class CheatingAgent(BaseAgent):
    def __init__(self, target_pos, grid_size, obstacles=None):
        self.target_pos = np.array(target_pos)
        self.grid_size = grid_size
        self.obstacles = [np.array(obs) for obs in obstacles] if obstacles else []

    def act(self, observation):
        agent_pos = np.array(observation)
        # Simple greedy strategy: move closer to target_pos, avoid obstacles
        # This is a basic example; a true optimal might need pathfinding (A*)

        best_action = -1
        min_dist = float('inf')

        # Try all 4 actions (0:up, 1:down, 2:left, 3:right)
        possible_next_positions = [
            agent_pos + np.array([-1, 0]), # Up
            agent_pos + np.array([1, 0]),  # Down
            agent_pos + np.array([0, -1]), # Left
            agent_pos + np.array([0, 1])   # Right
        ]

        current_dist_to_target = np.linalg.norm(agent_pos - self.target_pos)

        for action, next_pos_candidate in enumerate(possible_next_positions):
            # Check bounds
            if not (0 <= next_pos_candidate[0] < self.grid_size and \
                    0 <= next_pos_candidate[1] < self.grid_size):
                continue
            # Check obstacles
            if any(np.array_equal(next_pos_candidate, obs) for obs in self.obstacles):
                continue

            dist_to_target = np.linalg.norm(next_pos_candidate - self.target_pos)

            # Prefer actions that reduce distance. If multiple, any is fine for simple.
            # A more sophisticated cheat might look further ahead.
            if dist_to_target < current_dist_to_target and dist_to_target < min_dist :
                min_dist = dist_to_target
                best_action = action
            # If no move reduces distance, but some maintain it (e.g. moving parallel to target along an edge)
            # and the current best_action still leads to a greater distance or is uninitialized.
            elif dist_to_target == current_dist_to_target and best_action == -1:
                 min_dist = dist_to_target # though dist hasn't improved, we found a valid move.
                 best_action = action


        if best_action != -1:
            return best_action
        else:
            # If stuck or no 'improving' move, pick a random valid one or a default
            # For simplicity, let's try to pick any valid move if no strictly better move
            valid_actions = []
            for action, next_pos_candidate in enumerate(possible_next_positions):
                if not (0 <= next_pos_candidate[0] < self.grid_size and \
                        0 <= next_pos_candidate[1] < self.grid_size):
                    continue
                if any(np.array_equal(next_pos_candidate, obs) for obs in self.obstacles):
                    continue
                valid_actions.append(action)
            if valid_actions:
                return np.random.choice(valid_actions)
            return np.random.randint(4) # Fallback: truly random if completely boxed in by design flaw