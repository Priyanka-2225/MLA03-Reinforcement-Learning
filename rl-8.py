# ==============================
# IMPORTS (MUST BE FIRST)
# ==============================

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ==============================
# DEFINE ROAD ENVIRONMENT
# ==============================

class RoadEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.states = ['A', 'B', 'C', 'D']
        self.goal = 'D'

        # 0 = choose safe path, 1 = alternate path
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(len(self.states))

        self.state = 'A'

    def reset(self, seed=None, options=None):
        self.state = 'A'
        return self.states.index(self.state), {}

    def step(self, action):

        # Road transitions
        if self.state == 'A':
            self.state = 'B' if action == 0 else 'C'

        elif self.state in ['B', 'C']:
            self.state = 'D'

        # Rewards
        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return self.states.index(self.state), reward, done, False, {}

# ==============================
# CREATE ENVIRONMENT
# ==============================

env = RoadEnv()

# ==============================
# SAFE POLICY (Always choose action 0)
# ==============================

def safe_policy(state):
    return 0

# ==============================
# SIMULATION
# ==============================

state, _ = env.reset()
total_reward = 0

while True:
    action = safe_policy(state)
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

    if done:
        break

print("Total Reward using Safe Policy:", total_reward)
