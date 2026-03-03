import numpy as np
import random

maze_size = 4
V = np.zeros((maze_size, maze_size))
alpha = 0.1
gamma = 0.9

goal = (3,3)
trap = (1,1)

actions = [(0,1),(0,-1),(1,0),(-1,0)]

def step(state):
    x,y = state
    dx,dy = random.choice(actions)
    nx,ny = x+dx,y+dy
    if 0<=nx<maze_size and 0<=ny<maze_size:
        state = (nx,ny)
    reward = -1
    if state == goal:
        reward = 10
    if state == trap:
        reward = -10
    return state, reward

for episode in range(1000):
    state = (0,0)
    while state != goal:
        next_state, reward = step(state)
        x,y = state
        nx,ny = next_state
        
        V[x,y] += alpha*(reward + gamma*V[nx,ny] - V[x,y])
        state = next_state

print("State Values:\n", V)
