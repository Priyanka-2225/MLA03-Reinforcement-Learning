import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(1,1)
    def forward(self,x):
        return torch.sigmoid(self.fc(x))

policy=Policy()

def kl(old,new):
    return (old*(old.log()-new.log())).mean()

state=torch.FloatTensor([50.0])
old=policy(state)
new=policy(state)

loss=-(new*10)  # profit objective
print("TRPO structure ready")
