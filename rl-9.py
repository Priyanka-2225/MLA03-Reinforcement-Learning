# ==============================
# IMPORTS (MUST BE FIRST)
# ==============================

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ==============================
# DEFINE CALL CENTER ENV
# ==============================

class CallCenterEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 0 = Representative A
        # 1 = Representative B
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        return 0, {}

    def step(self, action):
        # Rep A average performance = 5
        # Rep B average performance = 7
        if action == 0:
            reward = np.random.normal(5, 1)
        else:
            reward = np.random.normal(7, 1)

        done = True   # One call per episode
        return 0, reward, done, False, {}

# ==============================
# CREATE ENVIRONMENT
# ==============================

env = CallCenterEnv()

# ==============================
# MONTE CARLO SIMULATION
# ==============================

episodes = 1000
returns = []

for episode in range(episodes):
    state, _ = env.reset()

    # Random assignment policy
    action = env.action_space.sample()

    _, reward, done, _, _ = env.step(action)
    returns.append(reward)

# Monte Carlo Value Estimate
value_estimate = np.mean(returns)

print("Monte Carlo Estimated Value:", value_estimate)
