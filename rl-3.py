import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BanditEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.means = [5,7,6]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        return 0, {}

    def step(self, action):
        reward = np.random.normal(self.means[action],1)
        return 0, reward, False, False, {}

env = BanditEnv()

def epsilon_greedy(env, epsilon=0.1, steps=1000):
    values=np.zeros(3)
    counts=np.zeros(3)
    total=0
    env.reset()
    for t in range(steps):
        if np.random.rand()<epsilon:
            action=env.action_space.sample()
        else:
            action=np.argmax(values)
        _,reward,_,_,_=env.step(action)
        counts[action]+=1
        values[action]+= (reward-values[action])/counts[action]
        total+=reward
    return total

print("Epsilon Greedy Revenue:",epsilon_greedy(env))
