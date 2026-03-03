import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(2,2)
    def forward(self,x):
        return torch.tanh(self.fc(x))

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(4,1)
    def forward(self,s,a):
        return self.fc(torch.cat([s,a],-1))

actor=Actor()
critic=Critic()

state=torch.FloatTensor([0.0,0.0])
action=actor(state)
q=critic(state,action)

print("DDPG Drone Model Ready")
