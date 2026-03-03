class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(10,1)
    def forward(self,x):
        return torch.tanh(self.fc(x))

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(11,1)
    def forward(self,state,action):
        return self.fc(torch.cat([state,action],dim=-1))
