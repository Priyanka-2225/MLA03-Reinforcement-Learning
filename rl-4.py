import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.size=4
        self.goal=(3,3)
        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Box(low=0,high=3,shape=(2,),dtype=np.int32)

    def step_state(self,state,action):
        x,y=state
        if action==0 and x>0: x-=1
        if action==1 and x<3: x+=1
        if action==2 and y>0: y-=1
        if action==3 and y<3: y+=1
        return (x,y)

env=DroneEnv()
gamma=0.9
V=np.zeros((4,4))
policy=np.zeros((4,4),dtype=int)

for _ in range(100):
    for i in range(4):
        for j in range(4):
            values=[]
            for a in range(4):
                ns=env.step_state((i,j),a)
                reward=-1
                values.append(reward+gamma*V[ns])
            V[i,j]=max(values)
            policy[i,j]=np.argmax(values)

print("Optimal Policy:\n",policy)
