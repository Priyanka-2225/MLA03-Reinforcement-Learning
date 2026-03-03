import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(9, 9)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.9

def check_win(board):
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a]==board[b]==board[c]!=0:
            return board[a]
    return 0

for episode in range(1000):
    board = [0]*9
    log_probs = []
    rewards = []

    while True:
        state = torch.FloatTensor(board)
        probs = policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        if board[action] != 0:
            continue

        log_probs.append(dist.log_prob(action))
        board[action] = 1

        if check_win(board)==1:
            rewards.append(1)
            break

        # random opponent
        empty=[i for i in range(9) if board[i]==0]
        if not empty:
            rewards.append(0)
            break
        opp=random.choice(empty)
        board[opp]= -1

        if check_win(board)==-1:
            rewards.append(-1)
            break

        rewards.append(0)

    G=0
    loss=0
    for log_prob,r in zip(reversed(log_probs),reversed(rewards)):
        G=r+gamma*G
        loss-=log_prob*G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Tic-Tac-Toe Training Complete")
