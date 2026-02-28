import numpy as np
import random

grid = 5
goal = (4,4)
traps = [(1,2),(3,3)]
gamma = 0.9
alpha = 0.1

V = np.zeros((grid,grid))

def step(state, action):
    x,y = state
    if action==0: x-=1
    if action==1: x+=1
    if action==2: y-=1
    if action==3: y+=1
    
    if x<0 or x>=grid or y<0 or y>=grid:
        return state, -5
    
    if (x,y) in traps:
        return (x,y), -50
    if (x,y)==goal:
        return (x,y), 100
    
    return (x,y), -1

for episode in range(2000):
    state=(0,0)
    while state!=goal:
        action=random.randint(0,3)
        next_state,reward=step(state,action)
        V[state]+=alpha*(reward+gamma*V[next_state]-V[state])
        state=next_state

print("State Values:\n",V)
