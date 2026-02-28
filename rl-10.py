# ==============================
# IMPORTS (MUST BE FIRST)
# ==============================

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ==============================
# DEFINE INVESTMENT ENVIRONMENT
# ==============================

class InvestmentEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 0 = Safe investment
        # 1 = Risky investment
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        return 0, {}

    def step(self, action):

        # Safe investment: low risk, low return
        if action == 0:
            reward = np.random.normal(2, 1)

        # Risky investment: high risk, high return
        else:
            reward = np.random.normal(5, 3)

        done = True  # One investment per episode
        return 0, reward, done, False, {}

# ==============================
# CREATE ENVIRONMENT
# ==============================

env = InvestmentEnv()

# ==============================
# SIMPLE POLICY GRADIENT
# ==============================

theta = np.zeros(2)   # policy parameters
alpha = 0.01          # learning rate
episodes = 5000

for episode in range(episodes):

    # Softmax policy
    probs = np.exp(theta) / np.sum(np.exp(theta))

    action = np.random.choice([0, 1], p=probs)

    _, reward, _, _, _ = env.step(action)

    # Policy gradient update
    grad = -probs
    grad[action] += 1

    theta += alpha * reward * grad

# Final learned policy
final_probs = np.exp(theta) / np.sum(np.exp(theta))

print("Learned Policy Probabilities:")
print("Safe Investment:", final_probs[0])
print("Risky Investment:", final_probs[1])
