import numpy as np
import random

grid_size = 5
actions = [(0,1),(0,-1),(1,0),(-1,0)]

Q = {}
returns = {}
gamma = 0.9
epsilon = 0.1

def step(state, action):
    x,y = state
    dx,dy = action
    nx,ny = x+dx, y+dy
    if 0<=nx<grid_size and 0<=ny<grid_size:
        state = (nx,ny)
    reward = -1
    if state == (4,4):
        reward = 20
    return state, reward

for episode in range(5000):
    state = (0,0)
    episode_data = []
    
    while state != (4,4):
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            qs = [Q.get((state,a),0) for a in actions]
            action = actions[np.argmax(qs)]
        
        next_state, reward = step(state, action)
        episode_data.append((state,action,reward))
        state = next_state
    
    G = 0
    for s,a,r in reversed(episode_data):
        G = gamma*G + r
        if (s,a) not in returns:
            returns[(s,a)] = []
        returns[(s,a)].append(G)
        Q[(s,a)] = np.mean(returns[(s,a)])

print("Training Complete")
