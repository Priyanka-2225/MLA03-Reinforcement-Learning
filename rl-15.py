import numpy as np

def simulate_episode():
    churn_prob = 0.3
    reward = 0
    for t in range(10):
        if np.random.rand() < churn_prob:
            return reward
        reward += 1
    return reward

returns = []

for i in range(1000):
    returns.append(simulate_episode())

print("Estimated Value:", np.mean(returns))
