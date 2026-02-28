import numpy as np
import random

grid=5
goal=(4,4)
ghost=(2,2)
Q=np.zeros((grid,grid,4))
alpha=0.1; gamma=0.9; epsilon=0.1

def step(state,action):
    x,y=state
    if action==0: x-=1
    if action==1: x+=1
    if action==2: y-=1
    if action==3: y+=1
    
    if x<0 or x>=grid or y<0 or y>=grid:
        return state,-5
    if (x,y)==ghost:
        return (x,y),-100
    if (x,y)==goal:
        return (x,y),100
    return (x,y),-1

for ep in range(3000):
    state=(0,0)
    while state!=goal:
        if random.random()<epsilon:
            action=random.randint(0,3)
        else:
            action=np.argmax(Q[state])
        
        next_state,reward=step(state,action)
        Q[state][action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state][action])
        state=next_state

print("Training Done")
