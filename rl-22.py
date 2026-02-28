import numpy as np
import random

grid_size = 5
actions = 4
gamma = 0.9
epsilon = 0.1

Q = {}
returns = {}

goal = (4,4)

def get_actions(state):
    return list(range(actions))

def step(state, action):
    x, y = state
    
    if action == 0: x -= 1
    if action == 1: x += 1
    if action == 2: y -= 1
    if action == 3: y += 1
    
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return state, -10
    
    new_state = (x,y)
    if new_state == goal:
        return new_state, 50
    
    return new_state, -1

def epsilon_greedy(state):
    if random.random() < epsilon:
        return random.choice(get_actions(state))
    else:
        if state not in Q:
            Q[state] = np.zeros(actions)
        return np.argmax(Q[state])

episodes = 5000

for episode in range(episodes):
    state = (0,0)
    episode_data = []
    
    while state != goal:
        action = epsilon_greedy(state)
        next_state, reward = step(state, action)
        episode_data.append((state, action, reward))
        state = next_state
    
    G = 0
    visited = set()
    
    for state, action, reward in reversed(episode_data):
        G = gamma * G + reward
        
        if (state, action) not in visited:
            visited.add((state, action))
            
            if state not in Q:
                Q[state] = np.zeros(actions)
            if (state, action) not in returns:
                returns[(state, action)] = []
            
            returns[(state, action)].append(G)
            Q[state][action] = np.mean(returns[(state, action)])

print("Optimal Policy Learned:")
for i in range(grid_size):
    for j in range(grid_size):
        if (i,j) in Q:
            print((i,j), "->", np.argmax(Q[(i,j)]))
