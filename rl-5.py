import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3")
env = env.unwrapped   # 🔥 IMPORTANT FIX

gamma = 0.9
theta = 0.001

V = np.zeros(env.observation_space.n)

while True:
    delta = 0
    for s in range(env.observation_space.n):
        v = V[s]
        action_values = []

        for a in range(env.action_space.n):
            q = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q += prob * (reward + gamma * V[next_state])
            action_values.append(q)

        V[s] = max(action_values)
        delta = max(delta, abs(v - V[s]))

    if delta < theta:
        break

print("Optimal Value Function Computed Successfully!")
