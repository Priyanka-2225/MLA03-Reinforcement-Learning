import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CleaningRobotEnv(gym.Env):
    def __init__(self):
        super(CleaningRobotEnv, self).__init__()

        self.grid = np.array([
            [0, 1, 0, -1, 0],
            [0, 0, 1, 0, 0],
            [1, 0, -1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, -1, 0, 0, 1]
        ])

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)
        self.state = (0, 0)

    def reset(self, seed=None, options=None):
        self.state = (0, 0)
        return np.array(self.state), {}

    def step(self, action):
        x, y = self.state

        if action == 0 and x > 0: x -= 1
        if action == 1 and x < 4: x += 1
        if action == 2 and y > 0: y -= 1
        if action == 3 and y < 4: y += 1

        self.state = (x, y)
        reward = self.grid[x][y]
        terminated = False
        truncated = False

        return np.array(self.state), reward, terminated, truncated, {}

env = CleaningRobotEnv()
state, _ = env.reset()

total_reward = 0
for _ in range(30):
    action = env.action_space.sample()  # random policy
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

print("Total Reward:", total_reward)
