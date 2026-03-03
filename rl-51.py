class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(5,3)
    def forward(self,x):
        return torch.softmax(self.fc(x),dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(5,1)
    def forward(self,x):
        return self.fc(x)

print("A2C Treatment Model Ready")
