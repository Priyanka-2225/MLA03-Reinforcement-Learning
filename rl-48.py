import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2,2)
    def forward(self,x):
        return torch.softmax(self.fc(x),dim=-1)

actor=Actor()
optimizer=optim.Adam(actor.parameters(),lr=0.01)
eps_clip=0.2

for episode in range(500):
    state=torch.FloatTensor([10.0,5.0])  # cars in lanes
    probs=actor(state)
    dist=torch.distributions.Categorical(probs)
    action=dist.sample()

    reward=-state.sum()  # minimize wait

    old_prob=probs[action].detach()
    new_prob=actor(state)[action]
    ratio=new_prob/old_prob
    advantage=reward

    loss=-torch.min(ratio*advantage,
                    torch.clamp(ratio,1-eps_clip,1+eps_clip)*advantage)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Traffic PPO Done")
