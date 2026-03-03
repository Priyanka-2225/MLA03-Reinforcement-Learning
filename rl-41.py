import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

grid_size = 5
actions = [(0,1),(0,-1),(1,0),(-1,0)]

# Create grid
grid = np.zeros((5,5))
grid[1,2] = 1   # dirt
grid[3,3] = 1
grid[2,1] = -1  # obstacle

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 4)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

gamma = 0.9

def step(state, action):
    x,y = state
    dx,dy = actions[action]
    nx,ny = x+dx,y+dy
    if not (0<=nx<5 and 0<=ny<5):
        return state, -1
    reward = grid[nx,ny]
    grid[nx,ny]=0
    return (nx,ny), reward-0.1

for episode in range(500):
    state=(0,0)
    log_probs=[]
    rewards=[]
    
    for t in range(30):
        s=torch.FloatTensor(state)
        probs=policy(s)
        dist=torch.distributions.Categorical(probs)
        action=dist.sample()
        
        log_probs.append(dist.log_prob(action))
        state,reward=step(state,action.item())
        rewards.append(reward)
    
    G=0
    loss=0
    for log_prob,reward in zip(reversed(log_probs),reversed(rewards)):
        G=reward+gamma*G
        loss-=log_prob*G
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training completed (REINFORCE)")
