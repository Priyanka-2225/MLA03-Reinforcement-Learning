class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4,2)
    def forward(self,x):
        return torch.softmax(self.fc(x),dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4,1)
    def forward(self,x):
        return self.fc(x)

actor=Actor()
critic=Critic()

optimizerA=optim.Adam(actor.parameters(),lr=0.01)
optimizerC=optim.Adam(critic.parameters(),lr=0.01)

gamma=0.9

for episode in range(300):
    state=torch.FloatTensor([0,0,0,0])
    for t in range(20):
        probs=actor(state)
        dist=torch.distributions.Categorical(probs)
        action=dist.sample()
        
        reward=-1
        next_state=torch.FloatTensor([0,0,0,0])
        
        value=critic(state)
        next_value=critic(next_state)
        td_target=reward+gamma*next_value
        td_error=td_target-value
        
        lossA=-dist.log_prob(action)*td_error.detach()
        lossC=td_error**2
        
        optimizerA.zero_grad()
        optimizerC.zero_grad()
        lossA.backward()
        lossC.backward()
        optimizerA.step()
        optimizerC.step()
        
        state=next_state

print("A2C Training done")
