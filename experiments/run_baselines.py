#%%
git_path = '/home/alleinzell/LabRL_Implementations'
import sys
if git_path not in sys.path:
    sys.path.append(git_path)
from environments.custom.gridworld_env import GridWorldEnv
from environments.custom.slip_gridworld_env import SlipGridWorldEnv
from agents.baseline.random_agent import RandomAgent
from agents.baseline.cheating_agent import CheatingAgent
from utils.plot_reward_histories import plot_reward_histories
from utils.run_agent import run_agent
from utils.plot_distances import plot_distances

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import argparse
import logging
from tqdm import tqdm

#%%

grid_size = 5
target = (grid_size - 1, grid_size - 1)
start = (grid_size - 1, 0)
obstacles = [(grid_size -1, j) for j in range(1, grid_size - 1)]

env = GridWorldEnv(size=grid_size, target_pos=target, start_pos=start, obstacles=obstacles)
random_agent = RandomAgent(env.action_space)
cheating_agent = CheatingAgent(target_pos=target, grid_size=grid_size, obstacles=obstacles)

#%% running random agent
reward_histories = []
total_rewards = []
distance_history = []
num_episodes = 100
min_steps = 100
for i in tqdm(range(num_episodes), desc="Running Random Agent Episodes"):
    total_reward, reward_history, distances = run_agent(random_agent, env, max_steps=100, min_steps=min_steps, render=False)
    reward_histories.append(reward_history)
    total_rewards.append(total_reward)
    distance_history.append(distances)

lens = [len(r) for r in reward_histories]
# Normalize the reward histories to the maximum length
max_len = max(lens)
reward_histories = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=0) for r in reward_histories]
reward_histories = np.stack(reward_histories)

lens = [len(r) for r in distance_history]
# Normalize the reward histories to the maximum length
max_len = max(lens)
distance_history = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=0) for r in distance_history]
distance_history = np.stack(distance_history)
fig, axs = plot_reward_histories(reward_histories)
fig, axs = plot_distances(distance_history)

#%% running cheating agent
reward_histories = []
total_rewards = []
distance_history = []

num_episodes = 100
min_steps = 100

for i in tqdm(range(num_episodes), desc="Running Random Agent Episodes"):
    total_reward, reward_history, distances = run_agent(cheating_agent, env, max_steps=100, min_steps=min_steps, render=False)
    reward_histories.append(reward_history)
    total_rewards.append(total_reward)
    distance_history.append(distances)

lens = [len(r) for r in reward_histories]
# Normalize the reward histories to the maximum length
max_len = max(lens)
reward_histories = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=0) for r in reward_histories]
reward_histories = np.stack(reward_histories)
lens = [len(r) for r in distance_history]
# Normalize the reward histories to the maximum length
max_len = max(lens)
distance_history = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=0) for r in distance_history]
distance_history = np.stack(distance_history)
fig, axs = plot_reward_histories(reward_histories)
fig, axs = plot_distances(distance_history)
# %% Slip

env = SlipGridWorldEnv(size=grid_size, target_pos=target, start_pos=start, obstacles=obstacles, slip_prob=0.75)
random_agent = RandomAgent(env.action_space)
cheating_agent = CheatingAgent(target_pos=target, grid_size=grid_size, obstacles=obstacles)

#%% running random agent
reward_histories = []
total_rewards = []
distance_history = []
num_episodes = 100
min_steps = 100
for i in tqdm(range(num_episodes), desc="Running Random Agent Episodes"):
    total_reward, reward_history, distances = run_agent(random_agent, env, max_steps=100, min_steps=min_steps, render=False)
    reward_histories.append(reward_history)
    total_rewards.append(total_reward)
    distance_history.append(distances)

lens = [len(r) for r in reward_histories]
# Normalize the reward histories to the maximum length
max_len = max(lens)
reward_histories = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=0) for r in reward_histories]
reward_histories = np.stack(reward_histories)

lens = [len(r) for r in distance_history]
# Normalize the reward histories to the maximum length
max_len = max(lens)
distance_history = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=0) for r in distance_history]
distance_history = np.stack(distance_history)
fig, axs = plot_reward_histories(reward_histories)
fig, axs = plot_distances(distance_history)

#%% running cheating agent
reward_histories = []
total_rewards = []
distance_history = []

num_episodes = 100
min_steps = 100

for i in tqdm(range(num_episodes), desc="Running Random Agent Episodes"):
    total_reward, reward_history, distances = run_agent(cheating_agent, env, max_steps=100, min_steps=min_steps, render=False)
    reward_histories.append(reward_history)
    total_rewards.append(total_reward)
    distance_history.append(distances)

lens = [len(r) for r in reward_histories]
# Normalize the reward histories to the maximum length
max_len = max(lens)
reward_histories = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=0) for r in reward_histories]
reward_histories = np.stack(reward_histories)
lens = [len(r) for r in distance_history]
# Normalize the reward histories to the maximum length
max_len = max(lens)
distance_history = [np.pad(r, (0, max_len - len(r)), 'constant', constant_values=0) for r in distance_history]
distance_history = np.stack(distance_history)
fig, axs = plot_reward_histories(reward_histories)
fig, axs = plot_distances(distance_history)

# %%
