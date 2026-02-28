import numpy as np

class ManufacturingEnv:
    def __init__(self):
        self.settings = 3
    
    def step(self, action):
        quality = np.random.normal(loc=action+1, scale=0.5)
        reward = quality
        return reward

Q = np.zeros(3)
N = np.zeros(3)

env = ManufacturingEnv()

for i in range(1000):
    action = np.argmax(Q)
    reward = env.step(action)
    N[action]+=1
    Q[action]+= (reward-Q[action])/N[action]

print("Best Setting:", np.argmax(Q))
