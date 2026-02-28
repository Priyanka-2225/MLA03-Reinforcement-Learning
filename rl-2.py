import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WarehouseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid = np.array([
            [0, 2, 0],
            [0, -2, 0],
            [0, 0, 5]
        ])
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=(2,), dtype=np.int32)
        self.state = (0,0)

    def reset(self, seed=None, options=None):
        self.state = (0,0)
        return np.array(self.state), {}

    def step(self, action):
        x,y = self.state
        if action==0 and x>0: x-=1
        if action==1 and x<2: x+=1
        if action==2 and y>0: y-=1
        if action==3 and y<2: y+=1

        self.state=(x,y)
        reward=self.grid[x][y]
        return np.array(self.state), reward, False, False, {}

# Policy evaluation (always move RIGHT)
gamma=0.9
V=np.zeros((3,3))
env=WarehouseEnv()

for _ in range(100):
    for i in range(3):
        for j in range(3):
            env.state=(i,j)
            next_state, reward,_,_,_=env.step(3)  # RIGHT
            V[i,j]=reward+gamma*V[next_state[0],next_state[1]]

print("Value Function:\n",V)
