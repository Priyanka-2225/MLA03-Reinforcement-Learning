import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ DEFINE THE BANDIT ENV FIRST
# ==============================

class AdBanditEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.probs = [0.05, 0.08, 0.06]  # true click probabilities
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        return 0, {}

    def step(self, action):
        reward = 1 if random.random() < self.probs[action] else 0
        return 0, reward, False, False, {}

# ==============================
# 2️⃣ CREATE ENV AFTER CLASS
# ==============================

env = AdBanditEnv()
rounds = 5000
k = env.action_space.n

# ==============================
# 3️⃣ EPSILON GREEDY
# ==============================

def epsilon_greedy(epsilon=0.1):
    values = np.zeros(k)
    counts = np.zeros(k)
    rewards = []
    env.reset()

    for t in range(rounds):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(values)

        _, reward, _, _, _ = env.step(action)
        counts[action] += 1
        values[action] += (reward - values[action]) / counts[action]
        rewards.append(reward)

    return np.cumsum(rewards) / np.arange(1, rounds+1)

# ==============================
# 4️⃣ UCB
# ==============================

def ucb():
    values = np.zeros(k)
    counts = np.zeros(k)
    rewards = []
    env.reset()

    for t in range(1, rounds+1):
        for i in range(k):
            if counts[i] == 0:
                action = i
                break
        else:
            ucb_values = values + np.sqrt(2*np.log(t)/counts)
            action = np.argmax(ucb_values)

        _, reward, _, _, _ = env.step(action)
        counts[action] += 1
        values[action] += (reward - values[action]) / counts[action]
        rewards.append(reward)

    return np.cumsum(rewards) / np.arange(1, rounds+1)

# ==============================
# 5️⃣ THOMPSON SAMPLING
# ==============================

def thompson():
    successes = np.ones(k)
    failures = np.ones(k)
    rewards = []
    env.reset()

    for t in range(rounds):
        samples = [np.random.beta(successes[i], failures[i]) for i in range(k)]
        action = np.argmax(samples)

        _, reward, _, _, _ = env.step(action)
        if reward == 1:
            successes[action] += 1
        else:
            failures[action] += 1
        rewards.append(reward)

    return np.cumsum(rewards) / np.arange(1, rounds+1)

# ==============================
# 6️⃣ RUN ALL
# ==============================

ctr_eps = epsilon_greedy()
ctr_ucb = ucb()
ctr_ts = thompson()

print("Final CTR Epsilon:", ctr_eps[-1])
print("Final CTR UCB:", ctr_ucb[-1])
print("Final CTR Thompson:", ctr_ts[-1])

plt.plot(ctr_eps, label="Epsilon-Greedy")
plt.plot(ctr_ucb, label="UCB")
plt.plot(ctr_ts, label="Thompson")
plt.legend()
plt.title("CTR Comparison")
plt.xlabel("Rounds")
plt.ylabel("CTR")
plt.show()
